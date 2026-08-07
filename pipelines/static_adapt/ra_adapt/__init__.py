"""Canonical deep public seams for Paper-I RA-ADAPT and Append-ADAPT."""

from __future__ import annotations

from typing import Any

from pipelines.static_adapt.ra_adapt.adapters import (
    CandidateRepresentationAdapter,
    GlobalSinglePauliWordCandidateAdapter,
    H2OLinearFDSinglePauliWordCandidateAdapter,
    H2OLinearFDSymmetryCompleteCandidateAdapter,
    MacroCandidateAdapter,
    SinglePauliWordCandidateAdapter,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    AppendAdaptRequest,
    AppendAdaptResult,
    CandidateInventoryLineageReceipt,
    CandidateInventoryLineageRow,
    CandidateLineageReceipt,
    PhaseIIIMultiplierContract,
    PhaseIIIStabilizationReceipt,
    PolicyEchoReceipt,
    PoolInventoryReceipt,
    RAAdaptOperationalControls,
    RAAdaptRequest,
    RAAdaptResult,
    ResolvedRAAdaptProtocol,
)
from pipelines.static_adapt.ra_adapt.engine import run_ra_adapt
from pipelines.static_adapt.ra_adapt.campaign import (
    PaperICampaignContractError,
    PaperICampaignPlan,
    PaperILocalExecutionAuthorization,
    authorize_paper_i_campaign,
    execute_paper_i_campaign,
    materialize_paper_i_campaign,
    retry_paper_i_campaign_qiskit_observation,
)


def run_append_adapt(problem: Any, request: Any = None) -> AppendAdaptResult:
    """Load the independent conventional selector only when requested."""

    from pipelines.static_adapt.ra_adapt.append import (
        run_append_adapt as _run_append_adapt,
    )

    return _run_append_adapt(problem, request)


__all__ = [
    "AppendAdaptRequest",
    "AppendAdaptResult",
    "CandidateInventoryLineageReceipt",
    "CandidateInventoryLineageRow",
    "CandidateLineageReceipt",
    "CandidateRepresentationAdapter",
    "GlobalSinglePauliWordCandidateAdapter",
    "H2OLinearFDSinglePauliWordCandidateAdapter",
    "H2OLinearFDSymmetryCompleteCandidateAdapter",
    "MacroCandidateAdapter",
    "PaperICampaignContractError",
    "PaperICampaignPlan",
    "PaperILocalExecutionAuthorization",
    "PhaseIIIMultiplierContract",
    "PhaseIIIStabilizationReceipt",
    "PolicyEchoReceipt",
    "PoolInventoryReceipt",
    "RAAdaptOperationalControls",
    "RAAdaptRequest",
    "RAAdaptResult",
    "ResolvedRAAdaptProtocol",
    "SinglePauliWordCandidateAdapter",
    "authorize_paper_i_campaign",
    "execute_paper_i_campaign",
    "materialize_paper_i_campaign",
    "retry_paper_i_campaign_qiskit_observation",
    "run_append_adapt",
    "run_ra_adapt",
]
