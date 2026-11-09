"""Utilities for solving the minimum-cost assignment problem.

The public surface is intentionally tiny: :func:`solve` and the
:class:`AssignmentError` it raises for malformed input.
"""

from .core import AssignmentError, solve

__all__ = ["AssignmentError", "solve"]
