"""Dataclasses for logging examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

Reward = float | dict[str, float]


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
    rewards : list[Reward] | None
        Optional reward for each response. Can be a float or a dict mapping
        reward component names to their values.
    groundtruth : str | None
        Optional reference/correct answer for this prompt.
    """

    prompt: str
    responses: list[str]
    rewards: list[Reward] | None = None
    groundtruth: str | None = None

    def __post_init__(self) -> None:
        if self.rewards is not None and len(self.rewards) != len(self.responses):
            raise ValueError(
                f"rewards length ({len(self.rewards)}) != "
                f"responses length ({len(self.responses)})"
            )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"prompt": [self.prompt], "response": [self.responses]}
        if self.rewards is not None:
            result["reward"] = [self.rewards]
        if self.groundtruth is not None:
            result["groundtruth"] = [self.groundtruth]
        return result


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
    rewards : list[list[Reward]] | None
        Optional rewards for each response of each prompt.
        Outer list must match prompts length, inner lists must match
        corresponding responses lengths.
    groundtruth : list[str] | None
        Optional reference/correct answers, one per prompt.
        Must have same length as prompts.
    """

    prompts: list[str]
    responses: list[list[str]]
    rewards: list[list[Reward]] | None = None
    groundtruth: list[str] | None = None

    def __post_init__(self) -> None:
        if len(self.prompts) != len(self.responses):
            raise ValueError(
                f"prompts length ({len(self.prompts)}) != "
                f"responses length ({len(self.responses)})"
            )
        if self.rewards is not None:
            if len(self.rewards) != len(self.prompts):
                raise ValueError(
                    f"rewards length ({len(self.rewards)}) != "
                    f"prompts length ({len(self.prompts)})"
                )
            for i, (resp_group, reward_group) in enumerate(
                zip(self.responses, self.rewards)
            ):
                if len(reward_group) != len(resp_group):
                    raise ValueError(
                        f"rewards[{i}] length ({len(reward_group)}) != "
                        f"responses[{i}] length ({len(resp_group)})"
                    )
        if self.groundtruth is not None and len(self.groundtruth) != len(self.prompts):
            raise ValueError(
                f"groundtruth length ({len(self.groundtruth)}) != "
                f"prompts length ({len(self.prompts)})"
            )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"prompt": self.prompts, "response": self.responses}
        if self.rewards is not None:
            result["reward"] = self.rewards
        if self.groundtruth is not None:
            result["groundtruth"] = self.groundtruth
        return result
