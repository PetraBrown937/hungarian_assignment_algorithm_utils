import unittest

from hungarian_assignment_algorithm_utils import AssignmentError, solve


class TestSolve(unittest.TestCase):
    def test_single_cell(self):
        self.assertEqual(solve([[7]]), [(0, 0)])

    def test_identity_is_cheapest(self):
        # The diagonal is cheap, so the optimal assignment is the diagonal.
        costs = [
            [1, 9, 9],
            [9, 1, 9],
            [9, 9, 1],
        ]
        self.assertEqual(sorted(solve(costs)), [(0, 0), (1, 1), (2, 2)])

    def test_off_diagonal_is_cheapest(self):
        costs = [
            [9, 1, 9],
            [9, 9, 1],
            [1, 9, 9],
        ]
        self.assertEqual(sorted(solve(costs)), [(0, 1), (1, 2), (2, 0)])

    def test_negative_costs(self):
        # Negative entries must not break the non-negative reduced-cost
        # invariant; the row-minimum initialisation handles this.
        costs = [
            [-5, -1],
            [-2, -8],
        ]
        self.assertEqual(sorted(solve(costs)), [(0, 0), (1, 1)])

    def test_more_columns_than_rows(self):
        costs = [
            [4, 1, 3],
            [2, 0, 5],
        ]
        # Row 0 takes column 1 (cost 1); row 1 takes column 0 (cost 2).
        self.assertEqual(sorted(solve(costs)), [(0, 1), (1, 0)])

    def test_more_rows_than_columns(self):
        costs = [
            [4, 1],
            [2, 0],
            [3, 3],
        ]
        # Column 0 is cheapest for row 1 (2); column 1 is cheapest for row 0 (1).
        self.assertEqual(sorted(solve(costs)), [(0, 1), (1, 0)])

    def test_flat_matrix(self):
        # All costs equal: any assignment is optimal, but it must be a
        # permutation and every row/column must appear exactly once.
        costs = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]
        result = sorted(solve(costs))
        self.assertEqual(result, [(0, 0), (1, 1), (2, 2)])

    def test_float_costs(self):
        costs = [
            [0.5, 2.0],
            [1.5, 0.25],
        ]
        self.assertEqual(sorted(solve(costs)), [(0, 0), (1, 1)])

    def test_empty_matrix_raises(self):
        with self.assertRaises(AssignmentError):
            solve([])

    def test_empty_row_raises(self):
        with self.assertRaises(AssignmentError):
            solve([[]])

    def test_ragged_matrix_raises(self):
        with self.assertRaises(AssignmentError):
            solve([[1, 2], [3]])

    def test_non_numeric_entry_raises(self):
        with self.assertRaises(AssignmentError):
            solve([[1, "two"]])

    def test_bool_entry_raises(self):
        # bool is a subclass of int in Python; reject it explicitly so the
        # caller does not get a surprising 0/1 cost.
        with self.assertRaises(AssignmentError):
            solve([[True, 2]])

    def test_none_raises(self):
        with self.assertRaises(AssignmentError):
            solve(None)

    def test_result_is_a_permutation(self):
        costs = [
            [10, 2, 8],
            [9, 7, 5],
            [6, 4, 3],
        ]
        result = solve(costs)
        rows = sorted(r for r, _ in result)
        cols = sorted(c for _, c in result)
        self.assertEqual(rows, [0, 1, 2])
        self.assertEqual(cols, [0, 1, 2])

    def test_result_is_optimal_for_known_case(self):
        # Brute force the optimal cost for a small case and compare.
        costs = [
            [10, 2, 8],
            [9, 7, 5],
            [6, 4, 3],
        ]
        import itertools

        best = min(
            sum(costs[i][perm[i]] for i in range(3))
            for perm in itertools.permutations(range(3))
        )
        result = solve(costs)
        got = sum(costs[r][c] for r, c in result)
        self.assertEqual(got, best)


if __name__ == "__main__":
    unittest.main()
