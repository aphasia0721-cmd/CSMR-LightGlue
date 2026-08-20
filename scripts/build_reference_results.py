"""Build the compact paper reference files from controlled evaluator outputs.

The evaluator summaries are the source of truth. This utility deliberately
copies the per-pair outputs without recomputing any metrics, so the JSON and
CSV artifacts remain tied to the same controlled-pose protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def _summary(path: Path) -> dict:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))


def _auc(summary: dict, method: str) -> dict[str, float]:
    return {f"AUC@{threshold}": round(100.0 * summary["results"][method][f"auc@{threshold}"], 2)
            for threshold in (5, 10, 20)}


def _round_direct(summary: dict) -> dict[str, list[float]]:
    methods = [
        ("Random-95%", ["Random-95%-seed17", "Random-95%-seed29", "Random-95%-seed41"]),
        ("Top-95%", ["Top-95%"]),
        ("Unconditional Spatial", ["Unconditional Spatial"]),
        ("Grid-RR", ["Grid-RR"]),
        ("4D-ANMS", ["4D-ANMS"]),
        ("Single-View Conditional", ["Single-View Conditional"]),
        ("CSMR", ["CSMR"]),
    ]
    out: dict[str, list[float]] = {}
    for label, sources in methods:
        if len(sources) == 1:
            out[label] = [round(100.0 * summary["results"][sources[0]][f"auc@{t}"], 2) for t in (5, 10, 20)]
        else:
            out[label] = [round(sum(100.0 * summary["results"][source][f"auc@{t}"] for source in sources) / len(sources), 2) for t in (5, 10, 20)]
    return out


def _bootstrap(summary: dict, key: str) -> dict:
    return summary[key]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--megadepth", type=Path, required=True)
    parser.add_argument("--scannet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", default="unknown")
    args = parser.parse_args()
    mega, scan = _summary(args.megadepth), _summary(args.scannet)
    args.output.mkdir(parents=True, exist_ok=True)
    datasets = {"MegaDepth-1500": mega, "ScanNet-1500": scan}
    reference = {
        "paper": "Conditional Spatial Match Repair for LightGlue-Based Relative Pose Estimation",
        "code_commit": args.commit,
        "protocol": {
            "datasets": list(datasets), "pairs_per_dataset": 1500, "ratio": 0.95,
            "grid_size": 4, "min_matches": 20, "max_score_drop": 0.10,
            "repair_target_policy": "Enumerate newly empty cells once from the initial Top-95% anchor; do not rebuild or prune the target sequence after replacements.",
            "pose_reuse": "Cache one PoseLib result per unique selected-index set within each image pair.",
            "bootstrap": {"samples": 5000, "seed": 20260817, "unit": "image-pair row", "interval": "paired percentile 95% CI", "multiple_comparison_correction": False},
            "random_seeds": [17, 29, 41],
        },
        "main_results_auc_percent": {name: {m: _auc(summary, m) for m in ("All", "Top-95%", "CSMR")} for name, summary in datasets.items()},
        "same_budget_auc_percent": {name: _round_direct(summary) for name, summary in datasets.items()},
        "runtime_ms_per_pair": {name: {m: round(summary["results"][m]["mean_total_ms"], 3) for m in ("All", "Top-95%", "CSMR")} for name, summary in datasets.items()},
        "statistical_summary": {
            "CSMR_vs_Top-95%_main_CIs_include_zero": True,
            "interpretation": "Under shared-pose evaluation, CSMR's additional gains over Top-95% are not statistically supported; all reported intervals are uncorrected descriptive paired-bootstrap intervals.",
        },
    }
    reference["bootstrap"] = {name: {"CSMR_vs_All": _bootstrap(summary, "CSMR_vs_All_AUC"), "CSMR_vs_Top-95%": _bootstrap(summary, "CSMR_vs_Top95_AUC")} for name, summary in datasets.items()}
    (args.output / "paper_reference_results.json").write_text(json.dumps(reference, indent=2), encoding="utf-8")
    bootstrap = {"protocol": reference["protocol"]["bootstrap"], "datasets": reference["bootstrap"]}
    (args.output / "bootstrap_summary.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")
    for source, name in ((args.megadepth, "per_pair_megadepth1500.csv"), (args.scannet, "per_pair_scannet1500.csv")):
        shutil.copy2(source / "per_pair.csv", args.output / name)
    print(json.dumps(reference["main_results_auc_percent"], indent=2))


if __name__ == "__main__":
    main()
