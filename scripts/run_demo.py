"""Run a dependency-light CSMR demonstration on synthetic matches."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from csmr.selectors import anms4d, grid_rr, select_csmr, select_top_confidence  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(7)
    count = 48
    size0 = np.asarray([640.0, 480.0])
    size1 = np.asarray([800.0, 600.0])
    points0 = rng.uniform([0, 0], size0, size=(count, 2))
    points1 = rng.uniform([0, 0], size1, size=(count, 2))
    scores = np.linspace(0.99, 0.30, count)
    top = select_top_confidence(scores, ratio=0.95, min_matches=20)
    csmr, diagnostics = select_csmr(points0, points1, scores, size0, size1)
    result = {
        "matches": count,
        "K": len(top),
        "Top-95%": top.tolist(),
        "CSMR": csmr.tolist(),
        "CSMR_diagnostics": diagnostics.to_dict(),
        "Grid-RR_count": int(len(grid_rr(points0, scores, size0))),
        "4D-ANMS_count": int(len(anms4d(points0, points1, scores, size0, size1))),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
