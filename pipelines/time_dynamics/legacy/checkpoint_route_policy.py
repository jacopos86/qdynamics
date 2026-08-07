from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pipelines.time_dynamics.legacy.checkpoint_types import (
    normalize_reference_mode,
    normalize_realtime_controller_mode,
)
from pipelines.time_dynamics.adapters.hamiltonian import (
    BOSON_CHAIN_FAMILIES,
    SPINFUL_LATTICE_FAMILIES,
    SPINLESS_LATTICE_FAMILIES,
)


STRICT_QPU_FAITHFUL_FLAG = "--checkpoint-controller-strict-qpu-faithful"
LEGACY_STRICT_QPU_HH_FLAG = "--checkpoint-controller-strict-qpu-hh"
STRICT_QPU_FLAG_LABEL = f"{STRICT_QPU_FAITHFUL_FLAG} (legacy {LEGACY_STRICT_QPU_HH_FLAG})"


@dataclass(frozen=True)
class RealtimeFamilyRoutePolicy:
    family_key: str
    supports_static_off: bool
    supports_static_exact_v1: bool
    supports_static_strict_oracle_v1: bool
    supports_drive_off: bool
    supports_drive_exact_v1: bool
    supports_drive_strict_oracle_v1: bool
    exact_v1_requires_reference_mode: str | None
    strict_requires_reference_mode: str
    exact_v1_default_append_pool_family: str | None
    forbid_drive_include_identity: bool = False
    drive_requires_num_sites: int | None = None


@dataclass(frozen=True)
class ValidatedRealtimeRoute:
    family_key: str
    strict_qpu_faithful: bool
    drive_requested: bool
    effective_append_pool_family: str
    controller_mode: str
    reference_mode: str


_SPINFUL_LATTICE_FAMILIES = tuple(sorted(SPINFUL_LATTICE_FAMILIES))
_SPINLESS_LATTICE_FAMILIES = tuple(sorted(SPINLESS_LATTICE_FAMILIES))
_BOSON_CHAIN_FAMILIES = tuple(sorted(BOSON_CHAIN_FAMILIES))


def _policy_table() -> dict[str, RealtimeFamilyRoutePolicy]:
    policies: dict[str, RealtimeFamilyRoutePolicy] = {
        "hh": RealtimeFamilyRoutePolicy(
            family_key="hh",
            supports_static_off=True,
            supports_static_exact_v1=True,
            supports_static_strict_oracle_v1=True,
            supports_drive_off=False,
            supports_drive_exact_v1=True,
            supports_drive_strict_oracle_v1=True,
            exact_v1_requires_reference_mode=None,
            strict_requires_reference_mode="off",
            exact_v1_default_append_pool_family=None,
        ),
        "spin_boson": RealtimeFamilyRoutePolicy(
            family_key="spin_boson",
            supports_static_off=True,
            supports_static_exact_v1=True,
            supports_static_strict_oracle_v1=True,
            supports_drive_off=True,
            supports_drive_exact_v1=True,
            supports_drive_strict_oracle_v1=True,
            exact_v1_requires_reference_mode="benchmark_exact",
            strict_requires_reference_mode="off",
            exact_v1_default_append_pool_family="full_meta",
            forbid_drive_include_identity=True,
            drive_requires_num_sites=1,
        ),
        "molecular_vibronic_h2": RealtimeFamilyRoutePolicy(
            family_key="molecular_vibronic_h2",
            supports_static_off=True,
            supports_static_exact_v1=True,
            supports_static_strict_oracle_v1=True,
            supports_drive_off=True,
            supports_drive_exact_v1=True,
            supports_drive_strict_oracle_v1=True,
            exact_v1_requires_reference_mode="benchmark_exact",
            strict_requires_reference_mode="off",
            exact_v1_default_append_pool_family="full_meta",
        ),
    }
    for family_key in _SPINFUL_LATTICE_FAMILIES + _SPINLESS_LATTICE_FAMILIES:
        policies[family_key] = RealtimeFamilyRoutePolicy(
            family_key=family_key,
            supports_static_off=True,
            supports_static_exact_v1=True,
            supports_static_strict_oracle_v1=True,
            supports_drive_off=False,
            supports_drive_exact_v1=True,
            supports_drive_strict_oracle_v1=True,
            exact_v1_requires_reference_mode="benchmark_exact",
            strict_requires_reference_mode="off",
            exact_v1_default_append_pool_family="full_meta",
        )
    for family_key in _BOSON_CHAIN_FAMILIES:
        policies[family_key] = RealtimeFamilyRoutePolicy(
            family_key=family_key,
            supports_static_off=True,
            supports_static_exact_v1=False,
            supports_static_strict_oracle_v1=True,
            supports_drive_off=False,
            supports_drive_exact_v1=True,
            supports_drive_strict_oracle_v1=True,
            exact_v1_requires_reference_mode="benchmark_exact",
            strict_requires_reference_mode="off",
            exact_v1_default_append_pool_family="full_meta",
        )
    return policies


_ROUTE_POLICIES = _policy_table()


def strict_qpu_faithful_requested(args: Any) -> bool:
    """Return true for the generic strict flag or the legacy HH strict alias."""

    return bool(
        getattr(args, "checkpoint_controller_strict_qpu_faithful", False)
        or getattr(args, "checkpoint_controller_strict_qpu_hh", False)
    )


def policy_for_family(family_key: str) -> RealtimeFamilyRoutePolicy:
    normalized = str(family_key).strip().lower()
    try:
        return _ROUTE_POLICIES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_ROUTE_POLICIES))
        raise ValueError(
            f"Realtime route policy does not support problem family {family_key!r}; "
            f"supported families: {supported}."
        ) from exc


def _append_pool_request(raw: str | None) -> str:
    text = "match_replay" if raw is None else str(raw).strip().lower()
    return "match_replay" if text == "" else text


def validate_realtime_route_request(
    *,
    family_key: str,
    controller_mode: str,
    reference_mode: str,
    drive_requested: bool,
    strict_qpu_faithful: bool,
    append_pool_family: str | None = "match_replay",
    num_sites: int | None = None,
    drive_include_identity: bool = False,
    primary_density_mode: str = "auto",
) -> ValidatedRealtimeRoute:
    """Validate problem-family realtime route policy without constructing a controller.

    This helper owns the family/mode/reference/drive/strict admission table.  It
    intentionally does not validate controller runtime invariants such as dense
    Hamiltonian availability or measurement-bundle support; those remain at the
    controller boundary in later slices.
    """

    policy = policy_for_family(family_key)
    mode = normalize_realtime_controller_mode(controller_mode)
    ref_mode = normalize_reference_mode(reference_mode)
    append_request = _append_pool_request(append_pool_family)
    primary_density = str(primary_density_mode).strip().lower()
    strict = bool(strict_qpu_faithful)
    driven = bool(drive_requested)

    if driven and policy.drive_requires_num_sites is not None:
        if num_sites is None or int(num_sites) != int(policy.drive_requires_num_sites):
            raise ValueError(
                f"Driven {policy.family_key} realtime requires num_sites == "
                f"{int(policy.drive_requires_num_sites)}."
            )
    if driven and policy.forbid_drive_include_identity and bool(drive_include_identity):
        raise ValueError(
            f"Driven {policy.family_key} realtime does not support --drive-include-identity."
        )

    if strict:
        if mode not in {"observable_v1", "oracle_v1"}:
            raise ValueError(
                f"{STRICT_QPU_FLAG_LABEL} requires --checkpoint-controller-mode "
                "observable_v1 or oracle_v1."
            )
        required_ref = normalize_reference_mode(policy.strict_requires_reference_mode)
        if ref_mode != required_ref:
            raise ValueError(
                f"{STRICT_QPU_FLAG_LABEL} requires controller exact inputs {required_ref} "
                "(--checkpoint-controller-reference-mode/--checkpoint-controller-exact-input-mode off)."
            )
        supported = (
            policy.supports_drive_strict_oracle_v1
            if driven
            else policy.supports_static_strict_oracle_v1
        )
        if not supported:
            drive_label = "driven" if driven else "static"
            raise ValueError(
                f"{drive_label.capitalize()} {policy.family_key} realtime does not support strict QPU-faithful realtime."
            )
        return ValidatedRealtimeRoute(
            family_key=policy.family_key,
            strict_qpu_faithful=True,
            drive_requested=driven,
            effective_append_pool_family=append_request,
            controller_mode=mode,
            reference_mode=ref_mode,
        )

    if mode == "oracle_v1":
        raise ValueError(
            "--checkpoint-controller-mode oracle_v1 is reserved for strict QPU-faithful "
            f"routes; pass {STRICT_QPU_FAITHFUL_FLAG} or use a non-oracle mode."
        )
    if mode == "observable_v1":
        raise ValueError(
            "--checkpoint-controller-mode observable_v1 is reserved for strict QPU-faithful "
            f"routes; pass {STRICT_QPU_FAITHFUL_FLAG} or use a diagnostic mode."
        )

    if mode == "off":
        supported = policy.supports_drive_off if driven else policy.supports_static_off
        if not supported:
            drive_label = "driven" if driven else "static"
            raise ValueError(
                f"{drive_label.capitalize()} {policy.family_key} realtime does not support controller mode off."
            )
        if ref_mode == "benchmark_exact" and primary_density != "auto":
            raise ValueError(
                f"{policy.family_key} benchmark_exact diagnostics currently require "
                "--checkpoint-controller-exact-forecast-primary-density-target-mode auto."
            )
        return ValidatedRealtimeRoute(
            family_key=policy.family_key,
            strict_qpu_faithful=False,
            drive_requested=driven,
            effective_append_pool_family=append_request,
            controller_mode=mode,
            reference_mode=ref_mode,
        )

    if mode != "exact_v1":
        raise ValueError(f"Unsupported realtime route controller mode {mode!r}.")

    supported = policy.supports_drive_exact_v1 if driven else policy.supports_static_exact_v1
    if not supported:
        drive_label = "driven" if driven else "static"
        raise ValueError(
            f"{drive_label.capitalize()} {policy.family_key} realtime does not support exact_v1."
        )
    required_ref = policy.exact_v1_requires_reference_mode
    if required_ref is not None and ref_mode != normalize_reference_mode(required_ref):
        raise ValueError(
            f"{policy.family_key} exact_v1 realtime requires "
            f"--checkpoint-controller-reference-mode {normalize_reference_mode(required_ref)}."
        )
    if required_ref is not None and primary_density != "auto":
        raise ValueError(
            f"{policy.family_key} exact_v1 benchmark_exact currently requires "
            "--checkpoint-controller-exact-forecast-primary-density-target-mode auto."
        )

    effective_append = append_request
    promoted_append = policy.exact_v1_default_append_pool_family
    if promoted_append is not None:
        if append_request in {"", "match_replay"}:
            effective_append = str(promoted_append)
        elif append_request != str(promoted_append):
            raise ValueError(
                f"{policy.family_key} exact_v1 realtime currently supports only the "
                f"default/match_replay route, which promotes to {promoted_append}, or "
                f"--append-pool-family {promoted_append}."
            )

    return ValidatedRealtimeRoute(
        family_key=policy.family_key,
        strict_qpu_faithful=False,
        drive_requested=driven,
        effective_append_pool_family=effective_append,
        controller_mode=mode,
        reference_mode=ref_mode,
    )


__all__ = [
    "LEGACY_STRICT_QPU_HH_FLAG",
    "STRICT_QPU_FAITHFUL_FLAG",
    "STRICT_QPU_FLAG_LABEL",
    "RealtimeFamilyRoutePolicy",
    "ValidatedRealtimeRoute",
    "policy_for_family",
    "strict_qpu_faithful_requested",
    "validate_realtime_route_request",
]
