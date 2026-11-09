# hungarian_assignment_algorithm_utils

Solves the minimum-cost assignment problem for two sets using the Kuhn-Munkres augmenting-path method.

## Usage

```python
from hungarian_assignment_algorithm_utils import solve

costs = [
    [10, 2, 8],
    [9, 7, 5],
    [6, 4, 3],
]

pairs = solve(costs)
# pairs is a list of (row, column) index pairs, e.g. [(0, 1), (1, 2), (2, 0)]
total = sum(costs[r][c] for r, c in pairs)
print(total)  # 2 + 5 + 6 = 13
```

## Why this exists

Assigning workers to jobs, tasks to machines, or rows to columns when the
number of each side is equal is a classic problem, but the textbook
implementations are usually written for square matrices and silently return
nonsense for rectangular ones. This library takes a rectangular cost matrix
and returns one pair per row of the smaller side, transposing internally when
there are more rows than columns. The trade-off is that the result is not
sorted: the algorithm emits pairs in column order of the transposed problem,
so callers who need a stable order should sort by row themselves. That keeps
the core loop simple and avoids a second allocation.

## The awkward edge

The matrix must be rectangular. Ragged input, empty input, and non-numeric
entries raise `AssignmentError`, which is a subclass of `ValueError`. Booleans
are rejected even though `bool` is a subclass of `int` in Python, because a
cost of `True` is almost always a bug in the caller's code rather than an
intentional cost of 1. Costs may be negative; the potentials are initialised
from the row minimum so the reduced costs stay non-negative.

## API

- `solve(costs)` — returns a list of `(row, column)` pairs. `costs` is a
  sequence of equal-length numeric sequences.
- `AssignmentError` — raised for malformed input.
