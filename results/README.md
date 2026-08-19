# Paper Results

`paper_reference_results.json` contains the AUC, same-budget, runtime, and
statistical values reported in the ICCIP manuscript. `reproducibility_manifest.json`
records the software environment, fixed seeds, model names, dataset metadata
hashes, and source-file hashes used to produce those values.

Both files identify commit `7c346e712358c86b69425b628f74b165db7bff4a`
as the paper-result reproduction snapshot. Later commits may improve packaging
or reporting without changing the paper tables.

Large datasets, model weights, and cached LightGlue packets are intentionally
not committed. Use the official MegaDepth-1500 and ScanNet-1500 protocols and
the commands in the repository README, then compare the generated
`summary.json` with the reference file.

For auditability, this release includes the per-pair files
`per_pair_megadepth1500.csv` and `per_pair_scannet1500.csv`, plus
`bootstrap_summary.json`. These are derived evaluation outputs rather than
raw images or model packets; they expose the pose-error rows and all three
paired-bootstrap thresholds used in the manuscript.
