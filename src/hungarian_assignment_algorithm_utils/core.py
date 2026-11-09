"""Kuhn-Munkres (Hungarian) algorithm for the assignment problem.

The algorithm here is the O(n^3) augmenting-path variant that maintains a
potential (dual) for each row and column. Each augmentation runs a Dijkstra-
like search over the reduced costs, so no explicit shortest-path graph is
built. This is the standard formulation and is what most textbook treatments
of Kuhn-Munkres describe.

Design decisions worth knowing about:

* The input is a rectangular cost matrix given as a list of lists of numbers.
  Rows are the "workers", columns are the "jobs". Any numeric type that
  supports comparison and subtraction works; the result is returned as a list
  of (row, column) integer index pairs.

* If the number of rows exceeds the number of columns, the matrix is
transposed internally so that the algorithm always operates with
  rows <= columns. The caller never has to think about this.

* Costs may be negative. The dual potentials are initialised from the row
  minimum, which keeps the reduced costs non-negative even when the input has
  negative entries. This is the reason for the ``min`` pass in
  :func:`_solve_rectangular`.

* Ties are broken by the first index encountered, so the result is
  deterministic for a given input.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

__all__ = ["AssignmentError", "solve"]


class AssignmentError(ValueError):
    """Raised when the cost matrix is malformed or not solvable."""


Number = (int, float)


def solve(costs: Sequence[Sequence[float]]) -> List[Tuple[int, int]]:
    """Solve the minimum-cost assignment problem.

    ``costs`` is a rectangular matrix: ``costs[i][j]`` is the cost of
    assigning row ``i`` to column ``j``. The number of rows and columns may
    differ; the returned list has ``min(rows, columns)`` pairs.

    Returns a list of ``(row, column)`` pairs using the original indices of
    ``costs``. The order of the pairs is not specified; callers that need a
    stable order should sort by row.

    Raises :class:`AssignmentError` if the matrix is empty, ragged, or
    contains non-numeric entries.
    """
    if costs is None:
        raise AssignmentError("cost matrix must not be None")

    rows = list(costs)
    if not rows:
        raise AssignmentError("cost matrix must have at least one row")

    width = len(rows[0])
    if width == 0:
        raise AssignmentError("cost matrix must have at least one column")

    for i, row in enumerate(rows):
        if len(row) != width:
            raise AssignmentError(
                "row %d has length %d, expected %d" % (i, len(row), width)
            )
        for j, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, Number):
                raise AssignmentError(
                    "entry at (%d, %d) is not a number: %r" % (i, j, value)
                )

    height = len(rows)
    if height <= width:
        matrix = [list(row) for row in rows]
        transposed = False
    else:
        matrix = [[rows[i][j] for i in range(height)] for j in range(width)]
        transposed = True

    assignment = _solve_rectangular(matrix)

    if transposed:
        # We solved the transposed problem, so the roles of row and column
        # are swapped relative to the caller's view.
        return [(col, row) for (row, col) in assignment]
    return assignment


def _solve_rectangular(matrix: List[List[float]]) -> List[Tuple[int, int]]:
    """Solve a matrix with ``len(matrix) <= len(matrix[0])``.

    This is the hot loop of the library. The invariants maintained are:

    * ``u[i]`` and ``v[j]`` are the dual potentials for rows and columns.
    * ``u[i] + v[j] <= matrix[i][j]`` for every pair.
    * ``p[j]`` is the row currently assigned to column ``j``, or -1.
    * ``way[j]`` is the column that led to ``j`` in the current augmenting
      path; it is used to reconstruct the assignment after each search.
    """
    n = len(matrix)
    m = len(matrix[0])

    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [-1] * (m + 1)
    way = [0] * (m + 1)

    # Initialise the row potentials to the row minimum. This makes every
    # reduced cost non-negative, which is a precondition for the Dijkstra
    # search below. It also handles negative costs without a separate pass.
    for i in range(1, n + 1):
        u[i] = min(matrix[i - 1])

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (m + 1)
        used = [False] * (m + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = -1
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j

            if j1 == -1:
                # No augmenting column exists. With a rectangular matrix and
                # n <= m this should be unreachable, but guard anyway so a
                # future change to the invariants fails loudly.
                raise AssignmentError("no augmenting path found")

            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == -1:
                break

        # Reconstruct the augmenting path and flip the assignment along it.
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    result: List[Tuple[int, int]] = []
    for j in range(1, m + 1):
        if p[j] != -1:
            result.append((p[j] - 1, j - 1))
    return result
