"""Dataclasses for logging examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Example:
    """
    A single prompt with grouped responses.

    Parameters
    ----------
    prompt : str
        The input prompt.
    responses : list[str]
        List of response variants for this prompt.
    """

    prompt: str
    responses: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"prompt": [self.prompt], "response": [self.responses]}


@dataclass(frozen=True)
class BatchExample:
    """
    A batch of prompts with grouped responses.

    Parameters
    ----------
    prompts : list[str]
        List of input prompts (batch).
    responses : list[list[str]]
        For each prompt, a list of response variants.
        Must have same length as prompts.
    """

    prompts: list[str]
    responses: list[list[str]]

    def __post_init__(self) -> None:
        if len(self.prompts) != len(self.responses):
            raise ValueError(
                f"prompts length ({len(self.prompts)}) != "
                f"responses length ({len(self.responses)})"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"prompt": self.prompts, "response": self.responses}
