"""Drive-aligned variational scaffold augmentation for AP-McLachlan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from pipelines.time_dynamics.ap_mclachlan.state import (
    APMcLachlanState,
    state_with_appended_terms,
)
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


DRIVE_ALIGNED_AUGMENTATION_SCHEMA_V1 = "ap_mclachlan_drive_aligned_augmentation_v1"


@dataclass(frozen=True)
class DriveAlignedAugmentation:
    """Record for a zero-angle ansatz block aligned with the drive operator."""

    state: APMcLachlanState
    applied: bool
    reason: str
    label: str | None
    runtime_delta: int
    logical_delta: int

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema": DRIVE_ALIGNED_AUGMENTATION_SCHEMA_V1,
            "applied": bool(self.applied),
            "reason": str(self.reason),
            "label": self.label,
            "parameterization_mode": str(self.state.parameterization_mode),
            "parameterization_label": str(self.state.parameterization_label),
            "runtime_delta": int(self.runtime_delta),
            "logical_delta": int(self.logical_delta),
            "active_parameter_count_after": int(self.state.active_parameter_count),
            "runtime_parameter_count_after": int(self.state.runtime_parameter_count),
            "runtime_pauli_parameter_count_after": int(self.state.runtime_pauli_parameter_count),
            "logical_parameter_count_after": int(self.state.logical_parameter_count),
        }


def augment_state_with_drive_aligned_generator(
    state: APMcLachlanState,
    *,
    hamiltonian: Any,
    enabled: bool = True,
) -> DriveAlignedAugmentation:
    """Append the resolved drive operator as a zero-angle ansatz block.

    The Hamiltonian is not modified.  This only enlarges the variational tangent
    space so McLachlan propagation can project onto the driven direction.
    """

    if not bool(enabled):
        return DriveAlignedAugmentation(
            state=state,
            applied=False,
            reason="disabled",
            label=None,
            runtime_delta=0,
            logical_delta=0,
        )
    drive_model = getattr(hamiltonian, "drive_model", None)
    if drive_model is None:
        return DriveAlignedAugmentation(
            state=state,
            applied=False,
            reason="no_drive_model",
            label=None,
            runtime_delta=0,
            logical_delta=0,
        )
    label = drive_aligned_generator_label(drive_model)
    existing_labels = {str(getattr(term, "label", "")) for term in state.terms}
    if str(label) in existing_labels:
        return DriveAlignedAugmentation(
            state=state,
            applied=False,
            reason="already_present",
            label=str(label),
            runtime_delta=0,
            logical_delta=0,
        )
    drive_poly = getattr(drive_model, "drive_poly", None)
    if drive_poly is None:
        return DriveAlignedAugmentation(
            state=state,
            applied=False,
            reason="drive_model_missing_poly",
            label=str(label),
            runtime_delta=0,
            logical_delta=0,
        )
    old_runtime = int(state.runtime_parameter_count)
    old_logical = int(state.logical_parameter_count)
    augmented = state_with_appended_terms(
        state,
        (AnsatzTerm(label=str(label), polynomial=drive_poly),),
    )
    return DriveAlignedAugmentation(
        state=augmented,
        applied=True,
        reason="appended_zero_angle_drive_generator",
        label=str(label),
        runtime_delta=int(augmented.runtime_parameter_count) - old_runtime,
        logical_delta=int(augmented.logical_parameter_count) - old_logical,
    )


def drive_aligned_generator_label(drive_model: Any) -> str:
    profile = _profile_payload(drive_model)
    family_key = str(getattr(drive_model, "family_key", profile.get("family_key", "")))
    operator_label = str(
        getattr(drive_model, "operator_label", profile.get("operator_label", "drive_operator"))
    )
    pattern = str(profile.get("pattern", "none"))
    if family_key == "hh" and operator_label == "hh_spinful_onsite_density":
        return f"drive_aligned_density(pattern={pattern})"
    if operator_label.endswith("onsite_density") or "density" in operator_label:
        return f"drive_aligned_density(pattern={pattern},operator={operator_label})"
    return f"drive_aligned_operator(operator={operator_label},pattern={pattern})"


def _profile_payload(drive_model: Any) -> Mapping[str, Any]:
    payload = getattr(drive_model, "profile_payload", {}) or {}
    return dict(payload) if isinstance(payload, Mapping) else {}


__all__ = [
    "DRIVE_ALIGNED_AUGMENTATION_SCHEMA_V1",
    "DriveAlignedAugmentation",
    "augment_state_with_drive_aligned_generator",
    "drive_aligned_generator_label",
]
