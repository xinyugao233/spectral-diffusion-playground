"""Shared utilities used by experiment entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from typing import Final

SCAFFOLD_STATUS: Final[str] = "Scaffold only; experiment implementation pending."


@dataclass(frozen=True, slots=True)
class ExperimentStub:
    """Experiment metadata for a not-yet-implemented script."""

    script_name: str
    title: str
    question: str


def format_experiment_stub(spec: ExperimentStub) -> str:
    """Render a consistent message for an unimplemented experiment."""
    return dedent(
        f"""
        [{spec.script_name}] {spec.title}
        Question: {spec.question}
        Status: {SCAFFOLD_STATUS}
        """
    ).strip()


def run_experiment_stub(spec: ExperimentStub) -> int:
    """Print a consistent message and return a success code."""
    print(format_experiment_stub(spec))
    return 0
