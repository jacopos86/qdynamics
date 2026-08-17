"""Cycle-neutral identities shared by adaptive phase selection boundaries."""

from __future__ import annotations


ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1 = (
    "phase_iii_no_positive_feasible_candidate_v1"
)
ADAPTIVE_PHASE3_NO_POSITIVE_POLICY_RAISE_V1 = (
    "raise_no_positive_feasible_candidate_v1"
)
ADAPTIVE_PHASE3_NO_POSITIVE_POLICY_TYPED_TERMINAL_V1 = (
    "typed_natural_terminal_v1"
)
ADAPTIVE_PHASE3_NO_POSITIVE_POLICY_FORCED_ADMISSION_V1 = (
    "forced_admission_exact_horizon_v1"
)
ADAPTIVE_HORIZON_POLICY_EXACT_TARGET_V1 = (
    "exact_requested_accepted_controller_rounds_v1"
)
ADAPTIVE_HORIZON_POLICY_MAXIMUM_V1 = (
    "maximum_accepted_controller_rounds_v1"
)


__all__ = [
    "ADAPTIVE_HORIZON_POLICY_EXACT_TARGET_V1",
    "ADAPTIVE_HORIZON_POLICY_MAXIMUM_V1",
    "ADAPTIVE_PHASE3_NO_POSITIVE_POLICY_FORCED_ADMISSION_V1",
    "ADAPTIVE_PHASE3_NO_POSITIVE_POLICY_RAISE_V1",
    "ADAPTIVE_PHASE3_NO_POSITIVE_POLICY_TYPED_TERMINAL_V1",
    "ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1",
]
