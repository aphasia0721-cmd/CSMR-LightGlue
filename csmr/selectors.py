"""Deterministic same-budget match selectors used in the CSMR paper.

All selectors operate on LightGlue match arrays.  A selector returns integer
indices into the original arrays; it never changes the correspondence values.
The default budget is K=min(N, max(20, round(ratio*N))).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SelectionDiagnostics:
    target_ratio: float
    target_matches: int
    selected_matches: int
    mean_confidence: float
    coverage0: float
    coverage1: float
    selected_coverage0: float
    selected_coverage1: float
    rescued: int = 0

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _as_points(points: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2), got {value.shape}.")
    return value


def _as_scores(scores: np.ndarray, count: int) -> np.ndarray:
    value = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(value) != count:
        raise ValueError("points0, points1, and scores must have equal length.")
    if not np.all(np.isfinite(value)):
        raise ValueError("scores must contain only finite values.")
    return value


def _validate_inputs(
    points0: np.ndarray, points1: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p0 = _as_points(points0, "points0")
    p1 = _as_points(points1, "points1")
    if len(p0) != len(p1):
        raise ValueError("points0, points1, and scores must have equal length.")
    return p0, p1, _as_scores(scores, len(p0))


def _validate_grid(grid_size: int) -> int:
    if int(grid_size) < 1:
        raise ValueError("grid_size must be positive.")
    return int(grid_size)


def cell_ids(points: np.ndarray, image_size: np.ndarray, grid_size: int = 4) -> np.ndarray:
    """Return row-major grid-cell IDs for pixel coordinates."""
    grid_size = _validate_grid(grid_size)
    points = _as_points(points, "points")
    size = np.asarray(image_size, dtype=np.float64).reshape(-1)
    if len(size) != 2 or np.any(size <= 0):
        raise ValueError("image_size must contain positive width and height.")
    x = np.clip((points[:, 0] / size[0] * grid_size).astype(np.int64), 0, grid_size - 1)
    y = np.clip((points[:, 1] / size[1] * grid_size).astype(np.int64), 0, grid_size - 1)
    return y * grid_size + x


def _coverage(cells: np.ndarray, grid_size: int) -> float:
    return float(np.unique(cells).size / (grid_size * grid_size)) if len(cells) else 0.0


def _budget(count: int, ratio: float, min_matches: int) -> int:
    if not 0.0 < float(ratio) <= 1.0:
        raise ValueError("ratio must be in (0, 1].")
    if int(min_matches) < 0:
        raise ValueError("min_matches must be non-negative.")
    return min(count, max(int(min_matches), int(round(count * float(ratio)))))


def _diagnostics(
    scores: np.ndarray,
    selected: np.ndarray,
    cells0: np.ndarray,
    cells1: np.ndarray,
    grid_size: int,
    ratio: float,
    target: int,
    rescued: int = 0,
) -> SelectionDiagnostics:
    return SelectionDiagnostics(
        target_ratio=float(ratio),
        target_matches=int(target),
        selected_matches=int(len(selected)),
        mean_confidence=float(np.mean(scores)) if len(scores) else 0.0,
        coverage0=_coverage(cells0, grid_size),
        coverage1=_coverage(cells1, grid_size),
        selected_coverage0=_coverage(cells0[selected], grid_size) if len(selected) else 0.0,
        selected_coverage1=_coverage(cells1[selected], grid_size) if len(selected) else 0.0,
        rescued=int(rescued),
    )


def select_all(count: int) -> np.ndarray:
    """Return every match index."""
    return np.arange(int(count), dtype=np.int64)


def select_top_confidence(
    scores: np.ndarray, ratio: float = 0.95, min_matches: int = 20
) -> np.ndarray:
    """Keep the K highest-confidence matches with stable tie handling."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    target = _budget(len(values), ratio, min_matches)
    order = np.argsort(-values, kind="mergesort")
    return order[:target].astype(np.int64, copy=False)


def select_random(
    count: int, ratio: float = 0.95, min_matches: int = 20, seed: int = 0
) -> np.ndarray:
    """Keep a deterministic random subset at the same budget."""
    target = _budget(int(count), ratio, min_matches)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(int(count), size=target, replace=False)).astype(np.int64)


def _round_robin_cells(
    scores: np.ndarray, cells: np.ndarray, target: int, grid_size: int
) -> np.ndarray:
    """Select one high-score match per occupied cell per round until K."""
    by_cell: dict[int, list[int]] = {}
    order = np.argsort(-scores, kind="mergesort")
    for index in order:
        by_cell.setdefault(int(cells[index]), []).append(int(index))
    selected: list[int] = []
    cell_order = sorted(by_cell)
    cursor = {cell: 0 for cell in cell_order}
    while len(selected) < target:
        progress = False
        for cell in cell_order:
            position = cursor[cell]
            candidates = by_cell[cell]
            if position >= len(candidates):
                continue
            selected.append(candidates[position])
            cursor[cell] += 1
            progress = True
            if len(selected) >= target:
                break
        if not progress:
            break
    return np.asarray(selected, dtype=np.int64)


def grid_rr(
    points0: np.ndarray,
    scores: np.ndarray,
    image_size0: np.ndarray,
    *,
    ratio: float = 0.95,
    grid_size: int = 4,
    min_matches: int = 20,
) -> np.ndarray:
    """Grid-RR: confidence-sorted round-robin selection in source-image cells."""
    p0 = _as_points(points0, "points0")
    values = _as_scores(scores, len(p0))
    cells = cell_ids(p0, image_size0, grid_size)
    return _round_robin_cells(values, cells, _budget(len(values), ratio, min_matches), grid_size)


def _four_d_radius(points0: np.ndarray, points1: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Compute nearest-higher-confidence distance in normalized four-space."""
    count = len(scores)
    if count == 0:
        return np.empty(0, dtype=np.float64)
    coords = np.concatenate([points0, points1], axis=1)
    radii = np.full(count, np.inf, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    for position in range(1, count):
        index = int(order[position])
        higher = order[:position]
        radii[index] = float(np.min(np.linalg.norm(coords[index] - coords[higher], axis=1)))
    return radii


def anms4d(
    points0: np.ndarray,
    points1: np.ndarray,
    scores: np.ndarray,
    image_size0: np.ndarray,
    image_size1: np.ndarray,
    *,
    ratio: float = 0.95,
    min_matches: int = 20,
) -> np.ndarray:
    """4D-ANMS using normalized (x0, y0, x1, y1) coordinates."""
    p0, p1, values = _validate_inputs(points0, points1, scores)
    size0 = np.asarray(image_size0, dtype=np.float64)
    size1 = np.asarray(image_size1, dtype=np.float64)
    coords0 = p0 / size0.reshape(1, 2)
    coords1 = p1 / size1.reshape(1, 2)
    radii = _four_d_radius(coords0, coords1, values)
    # Stable tie-breaking makes repeated runs independent of hash/random order.
    order = sorted(range(len(values)), key=lambda i: (-radii[i], -values[i], i))
    return np.asarray(order[: _budget(len(values), ratio, min_matches)], dtype=np.int64)


def _conditional_core(
    p0: np.ndarray,
    p1: np.ndarray,
    values: np.ndarray,
    size0: np.ndarray,
    size1: np.ndarray,
    *,
    ratio: float,
    grid_size: int,
    min_matches: int,
    max_score_drop: float,
    check_views: Iterable[int],
) -> tuple[np.ndarray, SelectionDiagnostics]:
    """Run conditional repair with targets frozen from the initial anchor set.

    The paper protocol enumerates newly empty cells once, immediately after
    Top-K selection.  Replacements update occupancy counts for redundancy
    checks, but do not rebuild or prune that initial target sequence.
    """
    if max_score_drop < 0.0:
        raise ValueError("max_score_drop must be non-negative.")
    target = _budget(len(values), ratio, min_matches)
    cells0 = cell_ids(p0, size0, grid_size)
    cells1 = cell_ids(p1, size1, grid_size)
    if target == 0:
        return np.empty(0, dtype=np.int64), _diagnostics(
            values, np.empty(0, dtype=np.int64), cells0, cells1, grid_size, ratio, target
        )
    order = np.argsort(-values, kind="mergesort")
    selected = order[:target].astype(np.int64, copy=True)
    selected_mask = np.zeros(len(values), dtype=bool)
    selected_mask[selected] = True
    counts0 = np.bincount(cells0[selected], minlength=grid_size * grid_size)
    counts1 = np.bincount(cells1[selected], minlength=grid_size * grid_size)
    total0 = np.bincount(cells0, minlength=grid_size * grid_size)
    total1 = np.bincount(cells1, minlength=grid_size * grid_size)
    rescued = 0
    # Freeze the target order from the initial Top-K anchor. An earlier repair
    # may also fill a later target through the other view; that later target is
    # still processed, matching the experiments reported in the paper.
    missing: list[tuple[int, int]] = []
    for view in check_views:
        total, selected_counts = (total0, counts0) if view == 0 else (total1, counts1)
        missing.extend((view, int(cell)) for cell in np.flatnonzero((total > 0) & (selected_counts == 0)))

    for view, cell in missing:
        source_cells = cells0 if view == 0 else cells1
        rejected = [int(i) for i in order if not selected_mask[i] and source_cells[i] == cell]
        if not rejected:
            continue
        rescue = rejected[0]
        removable: list[int] = []
        for position, index in enumerate(selected):
            keep0 = counts0[cells0[index]] > 1
            keep1 = counts1[cells1[index]] > 1
            if (view == 0 and keep0 and (0 not in check_views or keep1)) or (
                view == 1 and keep1 and (1 not in check_views or keep0)
            ) or (view not in (0, 1) and keep0 and keep1):
                removable.append(position)
        if not removable:
            continue
        remove_position = min(removable, key=lambda position: (values[selected[position]], position))
        remove = int(selected[remove_position])
        if values[rescue] + max_score_drop < values[remove]:
            continue
        selected[remove_position] = rescue
        selected_mask[remove] = False
        selected_mask[rescue] = True
        counts0[cells0[remove]] -= 1
        counts1[cells1[remove]] -= 1
        counts0[cells0[rescue]] += 1
        counts1[cells1[rescue]] += 1
        rescued += 1
    return selected, _diagnostics(values, selected, cells0, cells1, grid_size, ratio, target, rescued)


def select_csmr(
    points0: np.ndarray,
    points1: np.ndarray,
    scores: np.ndarray,
    image_size0: np.ndarray,
    image_size1: np.ndarray,
    *,
    ratio: float = 0.95,
    grid_size: int = 4,
    min_matches: int = 20,
    max_score_drop: float = 0.10,
) -> tuple[np.ndarray, SelectionDiagnostics]:
    """CSMR: rescue newly empty cells while checking both image views."""
    p0, p1, values = _validate_inputs(points0, points1, scores)
    return _conditional_core(
        p0, p1, values, image_size0, image_size1,
        ratio=ratio, grid_size=grid_size, min_matches=min_matches,
        max_score_drop=max_score_drop, check_views=(0, 1),
    )


def select_single_view_csmr(
    points0: np.ndarray,
    points1: np.ndarray,
    scores: np.ndarray,
    image_size0: np.ndarray,
    image_size1: np.ndarray,
    *,
    ratio: float = 0.95,
    grid_size: int = 4,
    min_matches: int = 20,
    max_score_drop: float = 0.10,
) -> tuple[np.ndarray, SelectionDiagnostics]:
    """Ablation that repairs occupancy only in the first/source image."""
    p0, p1, values = _validate_inputs(points0, points1, scores)
    return _conditional_core(
        p0, p1, values, image_size0, image_size1,
        ratio=ratio, grid_size=grid_size, min_matches=min_matches,
        max_score_drop=max_score_drop, check_views=(0,),
    )


def select_unconditional_spatial(
    points0: np.ndarray,
    points1: np.ndarray,
    scores: np.ndarray,
    image_size0: np.ndarray,
    image_size1: np.ndarray,
    *,
    ratio: float = 0.95,
    grid_size: int = 4,
    min_matches: int = 20,
) -> np.ndarray:
    """Fixed-ratio spatial selector used as the unconditional control.

    This is the paper's unconditional control: reserve the strongest 20% of
    matches and the highest-confidence match in every occupied cell of either
    view, then fill the remaining budget by confidence. Unlike Grid-RR, it
    does not cycle through cells round-robin and therefore provides a distinct
    same-budget baseline.
    """
    p0, p1, values = _validate_inputs(points0, points1, scores)
    target = _budget(len(values), ratio, min_matches)
    cells0 = cell_ids(p0, image_size0, grid_size)
    cells1 = cell_ids(p1, image_size1, grid_size)
    if target == 0:
        return np.empty(0, dtype=np.int64)

    order = np.argsort(-values, kind="mergesort")
    protected_count = min(target, max(0, int(np.ceil(len(values) * 0.20))))
    reserved: set[int] = {int(index) for index in order[:protected_count]}
    for cells in (cells0, cells1):
        best_by_cell: dict[int, int] = {}
        for index in order:
            cell = int(cells[index])
            if cell not in best_by_cell:
                best_by_cell[cell] = int(index)
        reserved.update(best_by_cell.values())

    reserved_order = sorted(reserved, key=lambda index: (-values[index], index))
    selected = reserved_order[:target]
    selected_set = set(selected)
    for index in order:
        value = int(index)
        if len(selected) >= target:
            break
        if value not in selected_set:
            selected.append(value)
            selected_set.add(value)
    return np.asarray(selected, dtype=np.int64)
