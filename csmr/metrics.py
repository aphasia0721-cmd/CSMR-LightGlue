"""Metrics and paired uncertainty calculations used by the experiments."""

from __future__ import annotations

import math

import numpy as np


def error_auc(errors: list[float] | np.ndarray, thresholds: tuple[int, ...] = (5, 10, 20)) -> dict[str, float]:
    values = [float(error) if math.isfinite(float(error)) else 1e6 for error in errors]
    sorted_errors = np.asarray([0.0] + sorted(values), dtype=np.float64)
    recall = np.linspace(0.0, 1.0, len(sorted_errors))
    result: dict[str, float] = {}
    trapezoid = getattr(np, "trapezoid", np.trapz)
    for threshold in thresholds:
        index = int(np.searchsorted(sorted_errors, threshold))
        x = np.concatenate([sorted_errors[:index], [threshold]])
        y = np.concatenate([recall[:index], [recall[max(index - 1, 0)]]])
        result[f"auc@{threshold}"] = float(trapezoid(y, x) / threshold)
    return result


def summarize(rows: list[dict]) -> dict[str, float | int]:
    errors = [row["pose_error_deg"] for row in rows]
    return {
        **error_auc(errors),
        "pairs": len(rows),
        "mean_matches": float(np.mean([row["matches"] for row in rows])) if rows else 0.0,
        "mean_inliers": float(np.mean([row["inliers"] for row in rows])) if rows else 0.0,
        "mean_inlier_ratio": float(np.mean([row["inlier_ratio"] for row in rows])) if rows else 0.0,
        "failed_pairs": int(sum(not math.isfinite(float(error)) for error in errors)),
    }


def paired_bootstrap_auc_difference(
    baseline_errors: np.ndarray,
    method_errors: np.ndarray,
    samples: int = 5000,
    seed: int = 20260814,
) -> dict[str, float]:
    """Bootstrap method-minus-baseline AUC differences by image-pair row."""
    baseline = np.asarray(baseline_errors, dtype=np.float64)
    method = np.asarray(method_errors, dtype=np.float64)
    if baseline.shape != method.shape or baseline.ndim != 1 or len(baseline) == 0:
        raise ValueError("baseline_errors and method_errors must be non-empty vectors of equal shape.")
    rng = np.random.default_rng(seed)
    differences = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        sample = rng.integers(0, len(baseline), len(baseline))
        differences[index] = error_auc(method[sample], (10,))["auc@10"] - error_auc(
            baseline[sample], (10,)
        )["auc@10"]
    low, high = np.percentile(differences, (2.5, 97.5))
    return {
        "difference_mean": float(np.mean(differences)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "samples": int(samples),
        "seed": int(seed),
    }
