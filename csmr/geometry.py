"""Relative-pose and correspondence geometry utilities.

The paper experiments use PoseLib.  OpenCV is supported as a fallback for
local smoke tests when PoseLib is unavailable; use PoseLib for exact paper
protocol reproduction.
"""

from __future__ import annotations

import math

import numpy as np


def camera_from_intrinsics(K: np.ndarray, size: np.ndarray) -> dict:
    K = np.asarray(K, dtype=np.float64)
    size = np.asarray(size, dtype=np.int64).reshape(-1)
    return {
        "model": "PINHOLE",
        "width": int(size[0]),
        "height": int(size[1]),
        "params": [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])],
    }


def relative_pose_error(T_0to1: np.ndarray, R: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """Return translation-direction and rotation errors in degrees."""
    T = np.asarray(T_0to1, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    t_gt = T[:3, 3]
    denominator = np.linalg.norm(t) * np.linalg.norm(t_gt)
    if denominator < 1e-12:
        translation_error = 180.0
    else:
        cosine = np.clip(np.dot(t, t_gt) / denominator, -1.0, 1.0)
        translation_error = float(np.rad2deg(np.arccos(cosine)))
        translation_error = min(translation_error, 180.0 - translation_error)
    cosine = np.clip((np.trace(np.asarray(R).T @ T[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    rotation_error = float(np.rad2deg(np.arccos(cosine)))
    return translation_error, rotation_error


def _pose_with_poselib(points0, points1, pair, size0, size1, threshold):
    import poselib  # type: ignore

    model, info = poselib.estimate_relative_pose(
        np.asarray(points0, dtype=np.float64),
        np.asarray(points1, dtype=np.float64),
        camera_from_intrinsics(pair["K0"], size0),
        camera_from_intrinsics(pair["K1"], size1),
        {
            "max_epipolar_error": float(threshold),
            "success_prob": 0.99999,
            "min_iterations": 20,
            "max_iterations": 1000,
        },
    )
    if model is None:
        return None, 0
    return model, int(np.asarray(info["inliers"], dtype=bool).sum())


def _pose_with_opencv(points0, points1, pair, threshold):
    import cv2  # type: ignore

    K0 = np.asarray(pair["K0"], dtype=np.float64)
    K1 = np.asarray(pair["K1"], dtype=np.float64)
    # OpenCV's essential-matrix solver uses one camera matrix; normalize both
    # views explicitly when intrinsics differ.
    p0 = cv2.undistortPoints(np.asarray(points0, dtype=np.float64).reshape(-1, 1, 2), K0, None).reshape(-1, 2)
    p1 = cv2.undistortPoints(np.asarray(points1, dtype=np.float64).reshape(-1, 1, 2), K1, None).reshape(-1, 2)
    E, mask = cv2.findEssentialMat(p0, p1, cameraMatrix=np.eye(3), threshold=float(threshold) / max(K0[0, 0], 1.0), prob=0.99999)
    if E is None:
        return None, 0
    _, R, t, inlier_mask = cv2.recoverPose(E, p0, p1, cameraMatrix=np.eye(3), mask=mask)
    return (R, t.reshape(3)), int(np.asarray(inlier_mask, dtype=bool).sum())


def pose_metrics(points0, points1, pair: dict, size0, size1, threshold: float = 1.0) -> tuple[float, float, int]:
    """Estimate relative pose and return rotation error, translation error, inliers."""
    if len(points0) < 5:
        return float("inf"), float("inf"), 0
    try:
        try:
            model, inliers = _pose_with_poselib(points0, points1, pair, size0, size1, threshold)
            if model is None:
                return float("inf"), float("inf"), 0
            t_error, r_error = relative_pose_error(pair["T_0to1"], model.R, model.t)
        except ImportError:
            model, inliers = _pose_with_opencv(points0, points1, pair, threshold)
            if model is None:
                return float("inf"), float("inf"), 0
            R, t = model
            t_error, r_error = relative_pose_error(pair["T_0to1"], R, t)
    except Exception:
        return float("inf"), float("inf"), 0
    return float(r_error), float(t_error), int(inliers)


def epipolar_errors(points0: np.ndarray, points1: np.ndarray, pair: dict) -> np.ndarray:
    """Compute Sampson epipolar errors in pixels from ground-truth pose."""
    if len(points0) == 0:
        return np.empty(0, dtype=np.float64)
    K0 = np.asarray(pair["K0"], dtype=np.float64)
    K1 = np.asarray(pair["K1"], dtype=np.float64)
    T = np.asarray(pair["T_0to1"], dtype=np.float64)
    R, t = T[:3, :3], T[:3, 3]
    tx = np.array([[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]])
    F = np.linalg.inv(K1).T @ (tx @ R) @ np.linalg.inv(K0)
    x0 = np.column_stack([points0, np.ones(len(points0))])
    x1 = np.column_stack([points1, np.ones(len(points1))])
    Fx0 = (F @ x0.T).T
    Ftx1 = (F.T @ x1.T).T
    numerator = np.sum(x1 * Fx0, axis=1) ** 2
    denominator = Fx0[:, 0] ** 2 + Fx0[:, 1] ** 2 + Ftx1[:, 0] ** 2 + Ftx1[:, 1] ** 2
    return np.sqrt(numerator / np.maximum(denominator, 1e-12))


def pose_error(rotation_error_deg: float, translation_error_deg: float) -> float:
    return max(float(rotation_error_deg), float(translation_error_deg))

