"""Historical ``run_sr_snake`` compatibility facade.

Canonical callers use :func:`pipelines.static_adapt.ra_adapt.run_ra_adapt`.
This module retains the old qualified name solely for source compatibility
and delegates every execution to the canonical facade.
"""

from __future__ import annotations

from pipelines.contracts.problem import ResolvedProblemContext
from pipelines.static_adapt.sr_snake.contracts import SRRunRequest, SRRunResult


def run_sr_snake(
    problem: ResolvedProblemContext,
    request: SRRunRequest | None = None,
) -> SRRunResult:
    """Delegate the historical facade to single-Pauli-word RA-ADAPT."""

    from pipelines.static_adapt.ra_adapt.adapters import (
        SinglePauliWordCandidateAdapter,
    )
    from pipelines.static_adapt.ra_adapt.contracts import RAAdaptRequest
    from pipelines.static_adapt.ra_adapt.engine import run_ra_adapt

    normalized = SRRunRequest() if request is None else request
    if not isinstance(normalized, SRRunRequest):
        raise TypeError("request must be an SRRunRequest or None.")
    return run_ra_adapt(
        problem,
        RAAdaptRequest(
            adapter=SinglePauliWordCandidateAdapter(),
            method=normalized.method,
            execution=normalized.execution,
            observation=normalized.observation,
        ),
    ).run


__all__ = ["run_sr_snake"]
