"""Create a tiny cached-packet fixture for testing evaluate_cached.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "examples" / "demo_packets"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(21)
    size = np.asarray([640.0, 480.0])
    K = [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
    T = [[1.0, 0.0, 0.0, 0.1], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    metadata = []
    for index in range(3):
        points0 = rng.uniform([0, 0], size, size=(64, 2))
        points1 = points0 + rng.normal(0, 1.0, size=(64, 2))
        scores = np.linspace(0.99, 0.35, 64) - index * 0.01
        np.savez_compressed(
            output / f"{index:04d}.npz",
            points0=points0,
            points1=points1,
            scores=scores,
            image_size0=size,
            image_size1=size,
        )
        metadata.append({"scene_id": f"demo_scene_{index}", "K0": K, "K1": K, "T_0to1": T})
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(metadata_path)


if __name__ == "__main__":
    main()
