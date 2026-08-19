# CSMR-LightGlue

Reference implementation for **Conditional Spatial Match Repair (CSMR)**,
the post-processing method described in:

> Conditional Spatial Match Repair for LightGlue-Based Relative Pose Estimation

CSMR does not retrain LightGlue and does not require depth. It receives cached
LightGlue correspondences and confidence scores, keeps the top 95 percent,
then repairs only image cells that became empty after confidence filtering.
The replacement is accepted only when the candidate is not more than 0.10
confidence points below a spatially redundant selected match. The output
budget is unchanged:

```text
K = min(N, max(20, round(0.95 * N)))
```

## Repository layout

```text
csmr/       Selectors, geometry, metrics, and packet I/O
scripts/    Cached-packet evaluator and a dependency-light demo
configs/    Example run configurations
metadata/   Small schema/example metadata only
tests/      Deterministic unit tests for the selectors
results/    Paper reference tables and reproducibility manifest; no large outputs are committed
```

The repository intentionally does not contain MegaDepth/ScanNet images,
LightGlue weights, depth maps, or large cached packets. Download those from
their official sources and keep them outside this repository.

## Installation

For the selector demo and unit tests:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-demo.txt
```

For the full relative-pose evaluation, install the additional dependencies in
`requirements-full.txt`. PoseLib is the estimator used for the paper results;
the package has an OpenCV fallback for local smoke tests when PoseLib is not
available.

## Quick verification

Run the tests and synthetic demonstration from the repository root:

```powershell
python -m unittest discover -s tests -v
python scripts/run_demo.py
```

The demo prints the selected indices, CSMR diagnostics, and the fact that
Top-95%, Grid-RR, and 4D-ANMS use the same `K` budget.

## Cached packet format

The evaluator expects one NumPy archive per metadata row:

```text
packets/0000.npz
packets/0001.npz
...
```

Each archive must contain:

| key | shape | meaning |
| --- | --- | --- |
| `points0` | `(N, 2)` | LightGlue coordinates in image 0 pixels |
| `points1` | `(N, 2)` | LightGlue coordinates in image 1 pixels |
| `scores` | `(N,)` | LightGlue confidence for each match |
| `image_size0` | `(2,)` | image 0 width, height |
| `image_size1` | `(2,)` | image 1 width, height |

The metadata JSON is a list. Each item must contain `scene_id`, `K0`, `K1`,
and `T_0to1`, as shown in `metadata/example_pairs.json`. `K0` and `K1` are
3x3 intrinsics and `T_0to1` is a 4x4 ground-truth transform.

## Reproduce cached-packet tables

After generating LightGlue packets with the official LightGlue/SuperPoint
implementation, run:

```powershell
python scripts/evaluate_cached.py `
  --metadata D:\data\megadepth1500_metadata.json `
  --packet-dir D:\data\megadepth1500_packets `
  --output-dir results\megadepth1500
```

The command writes `per_pair.csv` and `summary.json`. The JSON stores the exact
command arguments, detected Git commit, Python/NumPy versions, protocol, all
method summaries, and paired-bootstrap intervals for AUC@5, AUC@10, and
AUC@20. If Git is unavailable on `PATH`, pass the exact revision explicitly
with `--code-commit <40-character-hash>`. The evaluator compares:

```text
All
Top-95%
Random-95% (three fixed seeds)
Grid-RR
4D-ANMS
Unconditional Spatial
Single-View Conditional
CSMR
```

The expected paper summaries and the exact environment/checkpoint metadata are
stored in `results/paper_reference_results.json` and
`results/reproducibility_manifest.json`. The reference files are small and
contain no images, weights, or cached packets.

For the paper protocol, keep `--ratio 0.95`, `--grid-size 4`,
`--min-matches 20`, and `--max-score-drop 0.10`. Do not tune these values on
the final test set. The paper selects the engineering working point using a
separate validation split.

## Algorithm definitions

* **Grid-RR:** sort matches within each source-image cell by confidence and
  repeatedly visit all nonempty cells, selecting the next available match
  until `K` matches are collected. Empty cells are skipped.
* **4D-ANMS:** represent each match by normalized `(x0, y0, x1, y1)`;
  suppression radius is the Euclidean distance to the nearest
  higher-confidence match; retain the `K` largest radii.
* **CSMR:** start with Top-95%, identify cells that were occupied before
  filtering but are empty afterward in either view, and replace a selected
  match only if it is redundant in both views and satisfies the confidence
  drop bound. The repair-target sequence is enumerated once from the initial
  Top-95% anchor and is not rebuilt or pruned after individual replacements;
  occupancy counts are still updated for subsequent redundancy checks.

## Dataset and model setup

Use the official MegaDepth-1500 and ScanNet-1500 pair lists and camera
metadata. Generate packets with the same SuperPoint-LightGlue checkpoint,
image preprocessing, maximum keypoint count, and RANSAC settings used in the
paper. Record the checkpoint identifier and package versions in the output
directory. Dataset files and model weights remain outside this repository.

## Reproducibility checklist

Before sharing results, verify:

1. `python -m unittest discover -s tests -v` passes.
2. Every filtered selector returns exactly `K` unique indices.
3. The packet and metadata row counts match.
4. The run configuration, random seeds, checkpoint identifier, and commit hash
   are saved beside `summary.json`.
5. Results are compared against the paper tables only after the full dataset
   protocol has completed.

## License and citation

The CSMR code in this repository is released under the MIT License. LightGlue,
SuperPoint, MegaDepth, and ScanNet retain their own licenses and citations.
