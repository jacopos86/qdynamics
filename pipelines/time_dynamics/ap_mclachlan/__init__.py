"""Canonical AP-McLachlan namespace."""

from __future__ import annotations

import importlib
from typing import Any


_LAZY_EXPORTS = {
    "APMcLachlanState": ("state", "APMcLachlanState"),
    "APPEND_LADDER_PREFILTER_POLICY_V1": (
        "adaptive_trajectory",
        "APPEND_LADDER_PREFILTER_POLICY_V1",
    ),
    "APPEND_LADDER_SELECTION_POLICY_V1": (
        "adaptive_trajectory",
        "APPEND_LADDER_SELECTION_POLICY_V1",
    ),
    "AP_PRUNE_RANK_SCORE_KIND_V1": (
        "prune_cost",
        "AP_PRUNE_RANK_SCORE_KIND_V1",
    ),
    "ActiveSupportAtom": ("support_atoms", "ActiveSupportAtom"),
    "AppendControllerConfig": ("adaptive_trajectory", "AppendControllerConfig"),
    "AppendMclachlanTrajectory": ("adaptive_trajectory", "AppendMclachlanTrajectory"),
    "FixedMcLachlanStep": ("fixed_step", "FixedMcLachlanStep"),
    "FixedMclachlanTrajectory": ("trajectory", "FixedMclachlanTrajectory"),
    "INTEGRATOR_EULER": ("integrators", "INTEGRATOR_EULER"),
    "INTEGRATOR_RK4": ("integrators", "INTEGRATOR_RK4"),
    "GeometryEvaluation": ("geometry_eval", "GeometryEvaluation"),
    "McLachlanGeometry": ("geometry", "McLachlanGeometry"),
    "McLachlanInversePolicy": ("inverse", "McLachlanInversePolicy"),
    "PatchActionProposal": ("support_decision", "PatchActionProposal"),
    "PRUNE_LADDER_PREFILTER_POLICY_V1": (
        "adaptive_trajectory",
        "PRUNE_LADDER_PREFILTER_POLICY_V1",
    ),
    "PRUNE_LADDER_SELECTION_POLICY_V1": (
        "adaptive_trajectory",
        "PRUNE_LADDER_SELECTION_POLICY_V1",
    ),
    "PruneCostSettings": ("prune_cost", "PruneCostSettings"),
    "PruneCostTelemetry": ("prune_cost", "PruneCostTelemetry"),
    "RungDiagnostics": ("support_decision", "RungDiagnostics"),
    "SolveGuardReport": ("fixed_step", "SolveGuardReport"),
    "SolveRepairAttempt": ("fixed_step", "SolveRepairAttempt"),
    "SolveRepairConfig": ("fixed_step", "SolveRepairConfig"),
    "SolveRepairUnsupportedError": ("fixed_step", "SolveRepairUnsupportedError"),
    "StateSpaceSolveMetrics": ("geometry", "StateSpaceSolveMetrics"),
    "SupportAtom": ("support_atoms", "SupportAtom"),
    "SupportPatchControllerConfig": ("adaptive_trajectory", "SupportPatchControllerConfig"),
    "SupportPatchDecisionContext": ("support_decision", "SupportPatchDecisionContext"),
    "TimeDependentHamiltonian": ("hamiltonian", "TimeDependentHamiltonian"),
    "active_support_atoms": ("support_atoms", "active_support_atoms"),
    "candidate_append_atoms": ("support_atoms", "candidate_append_atoms"),
    "evaluate_mclachlan_geometry": ("geometry_eval", "evaluate_mclachlan_geometry"),
    "gamma_for_support": ("inverse", "gamma_for_support"),
    "integrate_theta_step": ("integrators", "integrate_theta_step"),
    "load_ap_mclachlan_state": ("state", "load_ap_mclachlan_state"),
    "run_append_mclachlan_trajectory": ("adaptive_trajectory", "run_append_mclachlan_trajectory"),
    "state_from_scaffold_runtime_input": ("state", "state_from_scaffold_runtime_input"),
    "state_with_appended_terms": ("state", "state_with_appended_terms"),
    "state_with_appended_atoms": ("support_atoms", "state_with_appended_atoms"),
    "state_with_appended_runtime_coordinates": (
        "state",
        "state_with_appended_runtime_coordinates",
    ),
    "state_with_support_patch_atoms": ("support_atoms", "state_with_support_patch_atoms"),
    "run_fixed_mclachlan_trajectory": ("trajectory", "run_fixed_mclachlan_trajectory"),
    "solve_fixed_mclachlan_step": ("fixed_step", "solve_fixed_mclachlan_step"),
    "solve_fixed_mclachlan_step_with_repair": (
        "fixed_step",
        "solve_fixed_mclachlan_step_with_repair",
    ),
    "solve_theta_dot": ("inverse", "solve_theta_dot"),
    "time_dependent_hamiltonian_from_runtime_input": (
        "hamiltonian",
        "time_dependent_hamiltonian_from_runtime_input",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    module = importlib.import_module(f"{__name__}.{module_name}")
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = sorted(_LAZY_EXPORTS)
