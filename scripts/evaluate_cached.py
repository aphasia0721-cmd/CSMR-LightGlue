"""Evaluate cached LightGlue packets with the paper's same-budget selectors.

The packet directory contains one ``0000.npz`` file per metadata row.  This
script does not run a feature extractor; it evaluates already cached matches.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from csmr.geometry import epipolar_errors, pose_error, pose_metrics  # noqa: E402
from csmr.io import load_metadata, load_packet  # noqa: E402
from csmr.metrics import error_auc, paired_bootstrap_auc_differences, summarize  # noqa: E402
from csmr.selectors import (  # noqa: E402
    anms4d,
    grid_rr,
    select_all,
    select_csmr,
    select_random,
    select_single_view_csmr,
    select_top_confidence,
    select_unconditional_spatial,
)


def evaluate_row(index: int, pair: dict, packet: dict, method: str, selected: np.ndarray, selection_ms: float) -> dict:
    start = time.perf_counter()
    p0 = packet["points0"][selected]
    p1 = packet["points1"][selected]
    rotation, translation, inliers = pose_metrics(
        p0, p1, pair, packet["image_size0"], packet["image_size1"], 1.0
    )
    pose_ms = (time.perf_counter() - start) * 1000.0
    epi = epipolar_errors(p0, p1, pair)
    count = len(selected)
    return {
        "index": int(index),
        "scene_id": pair["scene_id"],
        "method": method,
        "matches": int(count),
        "inliers": int(inliers),
        "inlier_ratio": float(inliers / max(count, 1)),
        "epi_precision_1px": float(np.mean(epi < 1.0)) if len(epi) else 0.0,
        "epi_precision_3px": float(np.mean(epi < 3.0)) if len(epi) else 0.0,
        "rotation_error_deg": float(rotation),
        "translation_error_deg": float(translation),
        "pose_error_deg": float(pose_error(rotation, translation)),
        "selection_ms": float(selection_ms),
        "pose_ms": float(pose_ms),
    }


def selections(index: int, packet: dict, args: argparse.Namespace) -> dict[str, tuple[np.ndarray, float]]:
    p0, p1, scores = packet["points0"], packet["points1"], packet["scores"]
    size0, size1 = packet["image_size0"], packet["image_size1"]
    output: dict[str, tuple[np.ndarray, float]] = {"All": (select_all(len(scores)), 0.0)}
    start = time.perf_counter(); output["Top-95%"] = (select_top_confidence(scores, args.ratio, args.min_matches), 0)
    output["Top-95%"] = (output["Top-95%"][0], (time.perf_counter() - start) * 1000.0)
    for seed in args.random_seeds:
        start = time.perf_counter()
        # Use the paper's deterministic per-pair permutation protocol instead
        # of reusing one subset for every image pair.
        target = len(select_top_confidence(scores, args.ratio, args.min_matches))
        rng = np.random.default_rng(seed * 1_000_003 + index)
        selected = rng.permutation(len(scores))[:target].astype(np.int64)
        output[f"Random-95%-seed{seed}"] = (selected, (time.perf_counter() - start) * 1000.0)
    start = time.perf_counter()
    output["Grid-RR"] = (grid_rr(p0, scores, size0, ratio=args.ratio, grid_size=args.grid_size, min_matches=args.min_matches), (time.perf_counter() - start) * 1000.0)
    start = time.perf_counter()
    output["4D-ANMS"] = (anms4d(p0, p1, scores, size0, size1, ratio=args.ratio, min_matches=args.min_matches), (time.perf_counter() - start) * 1000.0)
    start = time.perf_counter()
    output["Unconditional Spatial"] = (select_unconditional_spatial(p0, p1, scores, size0, size1, ratio=args.ratio, grid_size=args.grid_size, min_matches=args.min_matches), (time.perf_counter() - start) * 1000.0)
    start = time.perf_counter()
    selected, _ = select_single_view_csmr(p0, p1, scores, size0, size1, ratio=args.ratio, grid_size=args.grid_size, min_matches=args.min_matches, max_score_drop=args.max_score_drop)
    output["Single-View Conditional"] = (selected, (time.perf_counter() - start) * 1000.0)
    start = time.perf_counter()
    selected, _ = select_csmr(p0, p1, scores, size0, size1, ratio=args.ratio, grid_size=args.grid_size, min_matches=args.min_matches, max_score_drop=args.max_score_drop)
    output["CSMR"] = (selected, (time.perf_counter() - start) * 1000.0)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True, help="JSON list of image-pair records")
    parser.add_argument("--packet-dir", type=Path, required=True, help="Cached LightGlue .npz packets")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ratio", type=float, default=0.95)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--min-matches", type=int, default=20)
    parser.add_argument("--max-score-drop", type=float, default=0.10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--random-seeds", type=int, nargs="+", default=[17, 29, 41])
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260814)
    parser.add_argument(
        "--code-commit",
        default="",
        help="Override the Git commit recorded in summary.json (normally auto-detected)",
    )
    args = parser.parse_args()
    metadata = load_metadata(args.metadata)
    if args.limit > 0:
        metadata = metadata[: args.limit]
    rows: list[dict] = []
    for index, pair in enumerate(metadata):
        packet = load_packet(args.packet_dir, index)
        for method, (selected, selection_ms) in selections(index, packet, args).items():
            row = evaluate_row(index, pair, packet, method, selected, selection_ms)
            if method == "CSMR":
                _, diag = select_csmr(packet["points0"], packet["points1"], packet["scores"], packet["image_size0"], packet["image_size1"], ratio=args.ratio, grid_size=args.grid_size, min_matches=args.min_matches, max_score_drop=args.max_score_drop)
                row.update({f"selector_{key}": value for key, value in diag.to_dict().items()})
            rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output_dir / "per_pair.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    summaries = {}
    for method in sorted({row["method"] for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        result = summarize(subset)
        result["mean_selection_ms"] = float(np.mean([row["selection_ms"] for row in subset]))
        result["mean_pose_ms"] = float(np.mean([row["pose_ms"] for row in subset]))
        result["mean_total_ms"] = result["mean_selection_ms"] + result["mean_pose_ms"]
        summaries[method] = result
    command_argv = [str(sys.executable), *sys.argv]
    git_executable = shutil.which("git")
    code_commit = args.code_commit.strip()
    if not code_commit and git_executable:
        try:
            code_commit = subprocess.check_output(
                [git_executable, "-C", str(ROOT), "rev-parse", "HEAD"], text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            code_commit = ""
    if not code_commit:
        code_commit = "unknown"
    report: dict = {
        "protocol": {
            "ratio": args.ratio,
            "grid_size": args.grid_size,
            "min_matches": args.min_matches,
            "max_score_drop": args.max_score_drop,
            "random_seeds": args.random_seeds,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_unit": "image-pair row",
            "multiple_comparison_correction": False,
        },
        "reproducibility": {
            "command": shlex.join(command_argv),
            "command_argv": command_argv,
            "code_commit": code_commit,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "working_directory": str(Path.cwd()),
        },
        "results": summaries,
    }
    by_method = {method: [row for row in rows if row["method"] == method] for method in summaries}
    if "All" in by_method and "CSMR" in by_method:
        report["CSMR_vs_All_AUC"] = paired_bootstrap_auc_differences(
            np.asarray([row["pose_error_deg"] for row in by_method["All"]]),
            np.asarray([row["pose_error_deg"] for row in by_method["CSMR"]]),
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
    if "Top-95%" in by_method and "CSMR" in by_method:
        report["CSMR_vs_Top95_AUC"] = paired_bootstrap_auc_differences(
            np.asarray([row["pose_error_deg"] for row in by_method["Top-95%"]]),
            np.asarray([row["pose_error_deg"] for row in by_method["CSMR"]]),
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
