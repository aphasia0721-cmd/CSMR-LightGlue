"""Generate the cached RGB SuperPoint-LightGlue packets used by the evaluator.

This script is intentionally separate from the selector package so packet
generation can be audited without changing the post-processing code.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--resize", type=int, default=1024)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for packet generation.")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    packet_root = args.output_dir / "packets"
    packet_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    extractor = SuperPoint(max_num_keypoints=args.max_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint", depth_confidence=0.95, width_confidence=0.99).eval().to(device)
    rows: list[dict] = []
    for index, pair in tqdm(list(enumerate(metadata)), desc="LightGlue RGB packets"):
        packet_path = packet_root / f"{index:04d}.npz"
        if args.resume and packet_path.exists():
            continue
        image0 = load_image(args.dataset_root / pair["pair_names"][0]).to(device)
        image1 = load_image(args.dataset_root / pair["pair_names"][1]).to(device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            feats0 = extractor.extract(image0, resize=args.resize)
            feats1 = extractor.extract(image1, resize=args.resize)
            result = matcher({"image0": feats0, "image1": feats1})
        torch.cuda.synchronize()
        runtime_ms = (time.perf_counter() - started) * 1000.0
        feats0, feats1, result = [rbd(value) for value in (feats0, feats1, result)]
        matches = result["matches"]
        np.savez_compressed(
            packet_path,
            points0=feats0["keypoints"][matches[:, 0]].cpu().numpy().astype(np.float32),
            points1=feats1["keypoints"][matches[:, 1]].cpu().numpy().astype(np.float32),
            scores=result["scores"].cpu().numpy().astype(np.float32),
            image_size0=np.asarray([image0.shape[-1], image0.shape[-2]], dtype=np.int32),
            image_size1=np.asarray([image1.shape[-1], image1.shape[-2]], dtype=np.int32),
        )
        rows.append({"index": index, "scene_id": pair["scene_id"], "pair_id": pair["pair_id"], "matches": len(matches), "runtime_ms": runtime_ms, "packet": str(packet_path)})
    with (args.output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "scene_id", "pair_id", "matches", "runtime_ms", "packet"])
        writer.writeheader(); writer.writerows(rows)
    summary = {"pairs": len(metadata), "processed": len(rows), "gpu": torch.cuda.get_device_name(0), "mean_runtime_ms": float(np.mean([r["runtime_ms"] for r in rows])) if rows else 0.0, "packet_root": str(packet_root)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
