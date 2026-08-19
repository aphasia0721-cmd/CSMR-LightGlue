# Paper Results

`paper_reference_results.json` contains the AUC, same-budget, runtime, and
statistical values reported in the ICCIP manuscript. `reproducibility_manifest.json`
records the software environment, fixed seeds, model names, dataset metadata
hashes, and source-file hashes used to produce those values.

Large datasets, model weights, and cached LightGlue packets are intentionally
not committed. Use the official MegaDepth-1500 and ScanNet-1500 protocols and
the commands in the repository README, then compare the generated
`summary.json` with the reference file.
