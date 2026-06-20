"""Dataclass for logging 2D charts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class Chart:
    """
    A 2D chart: an ordered series of ``(x, y)`` points with axis names.

    Parameters
    ----------
    points : list[tuple[float, float]]
        Ordered ``(x, y)`` coordinates.
    axis_names : tuple[str, str]
        The ``(x_axis, y_axis)`` names.
    """

    points: list[tuple[float, float]]
    axis_names: tuple[str, str]

    def __post_init__(self) -> None:
        if len(self.axis_names) != 2:
            raise ValueError(
                f"axis_names must be a 2-tuple, got length {len(self.axis_names)}"
            )
        for i, point in enumerate(self.points):
            if len(point) != 2:
                raise ValueError(
                    f"point {i} has length {len(point)}, expected 2 (x, y)"
                )

    @classmethod
    def from_arrays(cls, xs: Any, ys: Any, axis_names: Sequence[str]) -> Chart:
        """
        Build from two parallel arrays of x and y values.

        Parameters
        ----------
        xs : array-like
            X values; anything with a ``.tolist()`` method (numpy, torch) or a
            sequence.
        ys : array-like
            Y values, same length as *xs*.
        axis_names : Sequence[str]
            The ``(x_axis, y_axis)`` names, length 2.
        """
        if hasattr(xs, "tolist"):
            xs = xs.tolist()
        if hasattr(ys, "tolist"):
            ys = ys.tolist()
        xs = list(xs)
        ys = list(ys)
        if len(xs) != len(ys):
            raise ValueError(f"xs length ({len(xs)}) != ys length ({len(ys)})")
        names = tuple(axis_names)
        if len(names) != 2:
            raise ValueError("axis_names must have length 2")
        return cls(
            points=[(float(x), float(y)) for x, y in zip(xs, ys)],
            axis_names=(names[0], names[1]),
        )
