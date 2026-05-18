"""Dataclass for logging confusion matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class ConfusionMatrix:
    """
    A confusion matrix with class labels.

    Rows represent true classes; columns represent predicted classes.

    Parameters
    ----------
    matrix : list[list[int]]
        Square integer matrix of shape (N, N).
    labels : list[str]
        Class names, length N.
    """

    matrix: list[list[int]]
    labels: list[str]

    def __post_init__(self) -> None:
        n = len(self.labels)
        if len(self.matrix) != n:
            raise ValueError(f"matrix rows ({len(self.matrix)}) != labels length ({n})")
        for i, row in enumerate(self.matrix):
            if len(row) != n:
                raise ValueError(f"matrix row {i} has length {len(row)}, expected {n}")

    @classmethod
    def from_array(cls, matrix: Any, labels: Sequence[str]) -> ConfusionMatrix:
        """
        Build from anything with a ``.tolist()`` method (numpy, torch).

        Parameters
        ----------
        matrix : array-like
            A 2D array with a ``.tolist()`` method, or a nested sequence.
        labels : Sequence[str]
            Class names, length N.
        """
        if hasattr(matrix, "tolist"):
            matrix = matrix.tolist()
        return cls(matrix=[list(row) for row in matrix], labels=list(labels))
