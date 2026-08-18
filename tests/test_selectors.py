from __future__ import annotations

import unittest

import numpy as np

from csmr.selectors import (
    anms4d,
    cell_ids,
    grid_rr,
    select_csmr,
    select_single_view_csmr,
    select_top_confidence,
    select_unconditional_spatial,
)


class SelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.size0 = np.asarray([640.0, 480.0])
        self.size1 = np.asarray([640.0, 480.0])
        # Four occupied cells in each view, with deliberately nonuniform
        # confidence so spatial selectors have a meaningful choice.
        self.points0 = np.asarray(
            [[40, 40], [80, 60], [350, 60], [390, 80], [50, 350], [90, 370], [360, 350], [400, 380]],
            dtype=float,
        )
        self.points1 = self.points0 + np.asarray([3.0, 2.0])
        self.scores = np.asarray([.99, .98, .97, .96, .95, .94, .93, .92])

    def assert_same_budget(self, selected: np.ndarray, expected: int) -> None:
        self.assertEqual(len(selected), expected)
        self.assertEqual(len(np.unique(selected)), expected)
        self.assertTrue(np.all((selected >= 0) & (selected < len(self.scores))))

    def test_top_budget(self) -> None:
        selected = select_top_confidence(self.scores, ratio=.95, min_matches=0)
        self.assert_same_budget(selected, 8)
        selected = select_top_confidence(self.scores, ratio=.50, min_matches=0)
        self.assert_same_budget(selected, 4)

    def test_all_spatial_selectors_keep_same_k(self) -> None:
        expected = len(select_top_confidence(self.scores, ratio=.50, min_matches=0))
        grid = grid_rr(self.points0, self.scores, self.size0, ratio=.50, min_matches=0)
        anms = anms4d(self.points0, self.points1, self.scores, self.size0, self.size1, ratio=.50, min_matches=0)
        unconditional = select_unconditional_spatial(
            self.points0, self.points1, self.scores, self.size0, self.size1, ratio=.50, min_matches=0
        )
        self.assert_same_budget(grid, expected)
        self.assert_same_budget(anms, expected)
        self.assert_same_budget(unconditional, expected)

    def test_csmr_preserves_budget_and_unique_indices(self) -> None:
        expected = len(select_top_confidence(self.scores, ratio=.50, min_matches=0))
        selected, diagnostics = select_csmr(
            self.points0, self.points1, self.scores, self.size0, self.size1,
            ratio=.50, min_matches=0, max_score_drop=.10,
        )
        self.assert_same_budget(selected, expected)
        self.assertGreaterEqual(diagnostics.selected_matches, 0)
        self.assertLessEqual(diagnostics.rescued, expected)

    def test_single_view_variant_preserves_budget(self) -> None:
        selected, _ = select_single_view_csmr(
            self.points0, self.points1, self.scores, self.size0, self.size1,
            ratio=.50, min_matches=0,
        )
        self.assert_same_budget(selected, 4)

    def test_cell_ids_are_row_major_and_clipped(self) -> None:
        cells = cell_ids(np.asarray([[-1, -1], [639, 479], [320, 240]]), self.size0, 4)
        self.assertEqual(cells.tolist(), [0, 15, 10])

    def test_csmr_confidence_drop_bound(self) -> None:
        # The candidate in the empty source cell is far below the selected
        # scores, so the strict drop bound must reject the rescue.
        points0 = np.asarray([[20, 20], [30, 30], [610, 450]], dtype=float)
        points1 = points0.copy()
        scores = np.asarray([.99, .98, .10])
        selected, diagnostics = select_csmr(
            points0, points1, scores, self.size0, self.size1,
            ratio=2 / 3, min_matches=0, max_score_drop=.10,
        )
        self.assertEqual(diagnostics.rescued, 0)
        self.assertEqual(set(selected.tolist()), {0, 1})


if __name__ == "__main__":
    unittest.main()
