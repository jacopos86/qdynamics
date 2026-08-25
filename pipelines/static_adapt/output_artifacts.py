from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from docs.reports.pdf_utils import (
    HAS_MATPLOTLIB,
    get_PdfPages,
    get_plt,
    render_command_page,
    render_text_page,
    require_matplotlib,
)
from docs.reports.report_pages import (
    render_executive_summary_page,
    render_manifest_overview_page,
    render_section_divider_page,
)
from pipelines.scaffold.handoff_state_bundle import build_statevector_manifest
from pipelines.static_adapt.adapt_candidate_record_cache import _candidate_record_cache_jsonable
from pipelines.static_adapt.historical_formal_manifold_provenance import (
    FORMAL_MANIFOLD_ROUTE,
    FORMAL_MANIFOLD_ROUTE_FAMILY,
    FORMAL_MANIFOLD_SR_SELECTOR_FAMILY,
    FormalManifoldRouteComposition,
)
from pipelines.static_adapt.sr_snake_escape_controller import (
    SR_CONTROLLER_ABLATION_CONTRACT_OFF,
    SR_COORDINATE_SOLVE_SCOPE_PHASE2_AND_PHASE3_V1,
    SR_POWELL_COORDINATE_CHART_AUTO,
    SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1,
    SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1,
    SR_POWELL_COORDINATE_CHART_POLICY_CHOICES,
    SR_ROUTE_FAMILY,
    SR_ROUTE_PROFILE_DISABLED,
    SR_ROUTE_PROFILE_PHASE2_PHASE3_WHITENED,
    SR_ROUTE_PROFILE_REDUCED_POWELL,
    SR_ROUTE_PROFILE_SADDLE_ONLY,
    SR_ROUTE_PROFILE_SADDLE_PLUS_MODELED_MINIMUM,
    SR_ROUTE_PROFILE_CONFORMANCE_UNPROMOTED_EXPLICIT_ABLATION,
)
from pipelines.static_adapt.sr_snake_route_profile import (
    HISTORICAL_SR_SNAKE_PHASE12_ENERGY_MODEL_SETTINGS,
    HISTORICAL_SR_SNAKE_RESPONSE_SCOPE_SETTINGS,
    SR_ROUTE_PROFILE_CANONICAL_V1,
    SR_ROUTE_PROFILE_CANDIDATE_V4,
    SR_ROUTE_PROFILE_CONVENTIONAL_V2,
    SR_ROUTE_PROFILE_CONVENTIONAL_V3,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1,
    SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1,
    SR_ROUTE_PROFILE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1,
    SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_V2,
    SR_ROUTE_PROFILE_REQUEST_OFF,
    normalize_phase3_response_coordinate_scope,
    normalize_sr_route_profile_request,
    validate_sr_route_profile_contract,
)
from pipelines.static_adapt.sr_snake_phase12_policy import (
    normalize_phase1_energy_model,
    normalize_phase1_score_mode_policy,
    normalize_phase2_cheap_curvature_proxy_policy,
    normalize_phase2_curvature_policy,
)
from src.quantum.adapt_spsa_refit import (
    ADAPT_SPSA_REFIT_ENGINE_ENV,
    resolve_adapt_spsa_refit_engine_label,
)

plt = get_plt() if HAS_MATPLOTLIB else None  # type: ignore[assignment]
PdfPages = get_PdfPages() if HAS_MATPLOTLIB else type("PdfPages", (), {})  # type: ignore[misc]

_SR_ROUTE_PROFILE_TO_POWELL_CHART = {
    SR_ROUTE_PROFILE_DISABLED: (
        SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
    ),
    SR_ROUTE_PROFILE_CONVENTIONAL_V2: (
        SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
    ),
    SR_ROUTE_PROFILE_CONVENTIONAL_V3: (
        SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
    ),
    SR_ROUTE_PROFILE_CANDIDATE_V4: (
        SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
    ),
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1: (
        SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
    ),
    SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1: (
        SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
    ),
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1: (
        SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
    ),
    SR_ROUTE_PROFILE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1: (
        SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
    ),
    SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_V2: (
        SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
    ),
    SR_ROUTE_PROFILE_REDUCED_POWELL: (
        SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
    ),
    SR_ROUTE_PROFILE_PHASE2_PHASE3_WHITENED: (
        SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
    ),
    SR_ROUTE_PROFILE_SADDLE_ONLY: (
        SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
    ),
    SR_ROUTE_PROFILE_SADDLE_PLUS_MODELED_MINIMUM: (
        SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
    ),
}


def _resolved_output_phase3_response_coordinate_scope(
    *,
    args: Any,
    adapt_payload: Mapping[str, Any],
) -> tuple[str, str]:
    """Cross-check and return the executed Phase-III response scope.

    Canonical v3 output must carry runtime telemetry; the output writer is not
    allowed to recover the scope merely from a window size or refit cadence.
    Historical v1/v2 requests retain their versioned legacy contract.
    """

    candidates: dict[str, str] = {}

    def _record(label: str, raw: Any) -> None:
        if raw in {None, ""}:
            return
        try:
            candidates[label] = normalize_phase3_response_coordinate_scope(raw)
        except ValueError as exc:
            raise ValueError(
                f"{label} has an unknown Phase-III response-coordinate scope: "
                f"{raw!r}."
            ) from exc

    _record(
        "adapt_vqe.phase3_response_coordinate_scope",
        adapt_payload.get("phase3_response_coordinate_scope"),
    )
    for label, block in (
        ("adapt_vqe.static_route_identity", adapt_payload.get("static_route_identity")),
        (
            "adapt_vqe.historical_singleton_coordinate_trust_overlay",
            adapt_payload.get("historical_singleton_coordinate_trust_overlay"),
        ),
        (
            "adapt_vqe.terminal_active_prefix_checkpoint",
            adapt_payload.get("terminal_active_prefix_checkpoint"),
        ),
    ):
        if isinstance(block, Mapping):
            _record(
                f"{label}.phase3_response_coordinate_scope",
                block.get("phase3_response_coordinate_scope"),
            )
    distinct = sorted(set(candidates.values()))
    if len(distinct) > 1:
        raise ValueError(
            "Result telemetry has conflicting Phase-III response-coordinate "
            "scopes: "
            + json.dumps(candidates, sort_keys=True)
        )

    requested_profile = normalize_sr_route_profile_request(
        getattr(args, "sr_route_profile_request", SR_ROUTE_PROFILE_REQUEST_OFF)
    )
    if requested_profile in {
        SR_ROUTE_PROFILE_CONVENTIONAL_V3,
        SR_ROUTE_PROFILE_CANDIDATE_V4,
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1,
        SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1,
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1,
        SR_ROUTE_PROFILE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1,
        SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_V2,
    }:
        expected = str(
            getattr(args, "phase3_response_coordinate_scope", "")
        ).strip().lower()
        if not distinct:
            raise ValueError(
                "Full-response SR-SNAKE result telemetry is missing "
                "phase3_response_coordinate_scope."
            )
        if distinct[0] != expected:
            raise ValueError(
                "Full-response SR-SNAKE runtime response scope disagrees with "
                f"the normalized CLI: runtime={distinct[0]!r}, cli={expected!r}."
            )
        return distinct[0], "resolved_runtime_telemetry"

    if requested_profile in {
        SR_ROUTE_PROFILE_CANONICAL_V1,
        SR_ROUTE_PROFILE_CONVENTIONAL_V2,
    }:
        expected = HISTORICAL_SR_SNAKE_RESPONSE_SCOPE_SETTINGS[
            "phase3_response_coordinate_scope"
        ]
        if distinct and distinct[0] != expected:
            raise ValueError(
                "Historical SR-SNAKE runtime response scope drifted: "
                f"runtime={distinct[0]!r}, required={expected!r}."
            )
        return expected, (
            "resolved_runtime_telemetry" if distinct else "versioned_historical_contract"
        )

    requested = normalize_phase3_response_coordinate_scope(
        getattr(args, "phase3_response_coordinate_scope", None)
    )
    if distinct and distinct[0] != requested:
        raise ValueError(
            "Runtime Phase-III response scope disagrees with the CLI: "
            f"runtime={distinct[0]!r}, cli={requested!r}."
        )
    return requested, "resolved_runtime_telemetry" if distinct else "explicit_cli"


def _resolved_output_phase12_energy_model_policies(
    *,
    args: Any,
    adapt_payload: Mapping[str, Any],
) -> tuple[dict[str, str], str]:
    normalizers = {
        "phase1_score_mode": normalize_phase1_score_mode_policy,
        "phase1_energy_model": normalize_phase1_energy_model,
        "phase2_curvature_policy": normalize_phase2_curvature_policy,
        "phase2_cheap_curvature_proxy_policy": (
            normalize_phase2_cheap_curvature_proxy_policy
        ),
    }
    candidates: dict[str, dict[str, str]] = {
        key: {} for key in normalizers
    }

    def _record_block(label: str, block: Mapping[str, Any]) -> None:
        for key, normalizer in normalizers.items():
            raw = block.get(key)
            if raw in {None, ""}:
                continue
            try:
                candidates[key][label] = normalizer(raw)
            except ValueError as exc:
                raise ValueError(
                    f"{label}.{key} has an unknown policy value: {raw!r}."
                ) from exc

    _record_block("adapt_vqe", adapt_payload)
    phase12_telemetry = adapt_payload.get("phase12_energy_model_telemetry")
    if isinstance(phase12_telemetry, Mapping):
        _record_block("phase12_energy_model_telemetry", phase12_telemetry)
    for label, block in (
        ("static_route_identity", adapt_payload.get("static_route_identity")),
        (
            "historical_singleton_coordinate_trust_overlay",
            adapt_payload.get("historical_singleton_coordinate_trust_overlay"),
        ),
        (
            "terminal_active_prefix_checkpoint",
            adapt_payload.get("terminal_active_prefix_checkpoint"),
        ),
    ):
        if isinstance(block, Mapping):
            _record_block(label, block)

    for key, source_values in candidates.items():
        if len(set(source_values.values())) > 1:
            raise ValueError(
                f"Result telemetry has conflicting {key} values: "
                + json.dumps(source_values, sort_keys=True)
            )

    requested_profile = normalize_sr_route_profile_request(
        getattr(args, "sr_route_profile_request", SR_ROUTE_PROFILE_REQUEST_OFF)
    )
    if requested_profile in {
        SR_ROUTE_PROFILE_CANDIDATE_V4,
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1,
        SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1,
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1,
    }:
        expected = {
            key: normalizer(getattr(args, key, None))
            for key, normalizer in normalizers.items()
        }
        missing = [key for key, values in candidates.items() if not values]
        if missing:
            raise ValueError(
                "Strict Phase-I/II SR-SNAKE result telemetry is missing policy "
                "fields: " + ",".join(missing)
            )
        resolved = {
            key: next(iter(values.values()))
            for key, values in candidates.items()
        }
        if resolved != expected:
            raise ValueError(
                "Strict Phase-I/II SR-SNAKE runtime policies disagree with the "
                f"normalized CLI: runtime={resolved!r}, cli={expected!r}."
            )
        return resolved, "resolved_runtime_telemetry"

    if requested_profile in {
        SR_ROUTE_PROFILE_CANONICAL_V1,
        SR_ROUTE_PROFILE_CONVENTIONAL_V2,
        SR_ROUTE_PROFILE_CONVENTIONAL_V3,
        SR_ROUTE_PROFILE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1,
        SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_V2,
    }:
        expected = dict(HISTORICAL_SR_SNAKE_PHASE12_ENERGY_MODEL_SETTINGS)
        for key, values in candidates.items():
            if values and next(iter(values.values())) != expected[key]:
                raise ValueError(
                    f"Historical SR-SNAKE runtime {key} drifted from its "
                    "versioned replay policy."
                )
        return expected, (
            "resolved_runtime_telemetry"
            if all(bool(values) for values in candidates.values())
            else "versioned_historical_contract"
        )

    explicit = {
        key: normalizer(getattr(args, key, None))
        for key, normalizer in normalizers.items()
    }
    for key, values in candidates.items():
        if values and next(iter(values.values())) != explicit[key]:
            raise ValueError(
                f"Runtime {key} disagrees with the normalized CLI."
            )
    return explicit, (
        "resolved_runtime_telemetry"
        if all(bool(values) for values in candidates.values())
        else "explicit_cli"
    )


@dataclass(frozen=True)
class AdaptEnergyMetrics:
    energy: float | None
    exact_gs_energy: float | None
    abs_delta_e: float | None


def _finite_float_or_none(raw: Any) -> float | None:
    try:
        value = float(raw)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _optional_int(raw: Any) -> int | None:
    if raw in {None, ""}:
        return None
    return int(raw)


def _optional_float(raw: Any) -> float | None:
    if raw in {None, ""}:
        return None
    return float(raw)


def _resolved_output_formal_manifold_route_composition(
    adapt_payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Normalize and cross-check every serialized FM composition alias."""

    candidates: list[tuple[str, Mapping[str, Any]]] = []
    direct = adapt_payload.get("formal_manifold_route_composition")
    if isinstance(direct, Mapping) and (
        str(direct.get("route_family", ""))
        == FORMAL_MANIFOLD_ROUTE_FAMILY
        or str(direct.get("adapt_reoptimization_route", ""))
        == FORMAL_MANIFOLD_ROUTE
    ):
        candidates.append(("adapt_vqe.formal_manifold_route_composition", direct))
    static_route = adapt_payload.get("static_route_identity")
    if isinstance(static_route, Mapping) and (
        str(static_route.get("route_family", "")) == FORMAL_MANIFOLD_ROUTE_FAMILY
        or str(static_route.get("adapt_reoptimization_route", ""))
        == FORMAL_MANIFOLD_ROUTE
    ):
        candidates.append(("adapt_vqe.static_route_identity", static_route))
    if not candidates and (
        str(adapt_payload.get("route_family", "")) == FORMAL_MANIFOLD_ROUTE_FAMILY
        or str(adapt_payload.get("adapt_reoptimization_route", ""))
        == FORMAL_MANIFOLD_ROUTE
    ):
        candidates.append(("adapt_vqe", adapt_payload))
    if not candidates:
        return None
    normalized: list[tuple[str, dict[str, Any]]] = []
    for field_path, candidate in candidates:
        try:
            composition = FormalManifoldRouteComposition.from_mapping(
                candidate
            ).as_dict()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid Formal-Manifold route composition at {field_path}: {exc}"
            ) from exc
        normalized.append((field_path, composition))
    reference_path, reference = normalized[0]
    conflicts = [
        field_path
        for field_path, candidate in normalized[1:]
        if candidate != reference
    ]
    if conflicts:
        raise ValueError(
            "Formal-Manifold route composition aliases disagree: "
            + ", ".join([reference_path, *conflicts])
        )
    return dict(reference)


def _resolved_output_powell_coordinate_chart_policy(
    *, args: Any, adapt_payload: Mapping[str, Any]
) -> tuple[str | None, str]:
    """Resolve the already-executed chart without inferring inside Powell.

    SR-SNAKE result artifacts must serialize a concrete chart identity.  The
    runtime telemetry is authoritative because the CLI may contain ``auto``;
    multiple telemetry aliases are accepted only when they agree.
    """

    candidates: dict[str, str] = {}
    static_route = adapt_payload.get("static_route_identity")
    overlay = adapt_payload.get("historical_singleton_coordinate_trust_overlay")
    optimizer_chart = adapt_payload.get("optimizer_coordinate_chart")
    for label, block in (
        ("adapt_vqe.static_route_identity", static_route),
        ("adapt_vqe.historical_singleton_coordinate_trust_overlay", overlay),
        ("adapt_vqe.optimizer_coordinate_chart", optimizer_chart),
    ):
        if not isinstance(block, Mapping):
            continue
        raw = block.get("powell_coordinate_chart_policy")
        if raw is None or raw == "":
            continue
        value = str(raw).strip().lower()
        if value not in SR_POWELL_COORDINATE_CHART_POLICY_CHOICES:
            raise ValueError(
                f"{label}.powell_coordinate_chart_policy is unknown: {raw!r}."
            )
        candidates[f"{label}.powell_coordinate_chart_policy"] = value
    formal_composition = _resolved_output_formal_manifold_route_composition(
        adapt_payload
    )
    if formal_composition is not None:
        selector = formal_composition.get("singleton_response_selector")
        if isinstance(selector, Mapping):
            raw = selector.get("sr_powell_coordinate_chart_policy")
            if raw not in {None, ""}:
                value = str(raw).strip().lower()
                if value not in SR_POWELL_COORDINATE_CHART_POLICY_CHOICES:
                    raise ValueError(
                        "Formal-Manifold selector Powell coordinate-chart "
                        f"policy is unknown: {raw!r}."
                    )
                candidates[
                    "adapt_vqe.formal_manifold_route_composition."
                    "singleton_response_selector.sr_powell_coordinate_chart_policy"
                ] = value
    distinct = sorted(set(candidates.values()))
    if len(distinct) > 1:
        raise ValueError(
            "Result telemetry has conflicting Powell coordinate-chart policies: "
            + json.dumps(candidates, sort_keys=True)
        )
    route_profile_fields: dict[str, str] = {}
    for label, block in (
        ("adapt_vqe.static_route_identity", static_route),
        ("adapt_vqe.historical_singleton_coordinate_trust_overlay", overlay),
    ):
        if not isinstance(block, Mapping):
            continue
        for key in ("route_profile", "sr_route_profile"):
            raw = block.get(key)
            if raw in {None, ""}:
                continue
            profile = str(raw).strip().lower()
            if profile in _SR_ROUTE_PROFILE_TO_POWELL_CHART:
                route_profile_fields[f"{label}.{key}"] = profile
    profile_expected_charts = {
        _SR_ROUTE_PROFILE_TO_POWELL_CHART[profile]
        for profile in route_profile_fields.values()
    }
    if len(profile_expected_charts) > 1:
        raise ValueError(
            "Result telemetry has SR route profiles with incompatible Powell "
            "coordinate-chart policies: "
            + json.dumps(route_profile_fields, sort_keys=True)
        )
    route_profile_expected = (
        next(iter(profile_expected_charts)) if profile_expected_charts else None
    )
    conformance_fields: dict[str, str] = {}
    scope_fields: dict[str, str] = {}
    for label, block in (
        ("adapt_vqe.static_route_identity", static_route),
        ("adapt_vqe.historical_singleton_coordinate_trust_overlay", overlay),
    ):
        if not isinstance(block, Mapping):
            continue
        raw_conformance = block.get("route_profile_conformance")
        if raw_conformance not in {None, ""}:
            conformance_fields[f"{label}.route_profile_conformance"] = str(
                raw_conformance
            ).strip().lower()
        raw_scope = block.get("coordinate_solve_scope")
        if raw_scope not in {None, ""}:
            scope_fields[f"{label}.coordinate_solve_scope"] = str(
                raw_scope
            ).strip().lower()
    conformance_values = sorted(set(conformance_fields.values()))
    scope_values = sorted(set(scope_fields.values()))
    if len(conformance_values) > 1:
        raise ValueError(
            "Result telemetry has conflicting SR route-profile conformance markers: "
            + json.dumps(conformance_fields, sort_keys=True)
        )
    if len(scope_values) > 1:
        raise ValueError(
            "Result telemetry has conflicting SR coordinate-solve scopes: "
            + json.dumps(scope_fields, sort_keys=True)
        )
    route_profile_conformance = (
        conformance_values[0] if conformance_values else None
    )
    coordinate_solve_scope = scope_values[0] if scope_values else None

    def _allowed_unpromoted_expanded_ablation(chart: str) -> bool:
        return bool(
            route_profile_conformance
            == SR_ROUTE_PROFILE_CONFORMANCE_UNPROMOTED_EXPLICIT_ABLATION
            and coordinate_solve_scope
            == SR_COORDINATE_SOLVE_SCOPE_PHASE2_AND_PHASE3_V1
            and chart
            == SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
        )

    if distinct:
        resolved = distinct[0]
        if (
            route_profile_expected is not None
            and resolved != route_profile_expected
            and not _allowed_unpromoted_expanded_ablation(resolved)
        ):
            raise ValueError(
                "Result telemetry has a route-profile/Powell-chart mismatch: "
                f"profiles={json.dumps(route_profile_fields, sort_keys=True)}, "
                f"chart={resolved!r}, expected={route_profile_expected!r}."
            )
        return resolved, "resolved_runtime_telemetry"

    requested = str(
        getattr(
            args,
            "sr_powell_coordinate_chart_policy",
            SR_POWELL_COORDINATE_CHART_AUTO,
        )
    ).strip().lower()
    if requested in SR_POWELL_COORDINATE_CHART_POLICY_CHOICES:
        if (
            route_profile_expected is not None
            and requested != route_profile_expected
            and not _allowed_unpromoted_expanded_ablation(requested)
        ):
            raise ValueError(
                "Result telemetry has a route-profile/Powell-chart mismatch: "
                f"profiles={json.dumps(route_profile_fields, sort_keys=True)}, "
                f"chart={requested!r}, expected={route_profile_expected!r}."
            )
        return requested, "explicit_cli"
    if requested != SR_POWELL_COORDINATE_CHART_AUTO:
        raise ValueError(
            "sr_powell_coordinate_chart_policy is unknown: "
            f"{requested!r}."
        )
    route_family = (
        str(static_route.get("route_family", "")).strip().lower()
        if isinstance(static_route, Mapping)
        else ""
    )
    formal_selector_family = (
        None
        if formal_composition is None
        else formal_composition.get("candidate_selector_family")
    )
    if (
        route_family == str(SR_ROUTE_FAMILY).strip().lower()
        or str(formal_selector_family or "").strip().lower()
        == str(FORMAL_MANIFOLD_SR_SELECTOR_FAMILY).strip().lower()
    ):
        raise ValueError(
            "SR-SNAKE result telemetry is missing its resolved Powell "
            "coordinate-chart policy; refusing to serialize 'auto'."
        )
    return None, "not_applicable"


def extract_adapt_energy_metrics(payload: Mapping[str, Any]) -> AdaptEnergyMetrics:
    adapt_payload = payload.get("adapt_vqe", payload) if isinstance(payload, Mapping) else {}
    if not isinstance(adapt_payload, Mapping):
        return AdaptEnergyMetrics(energy=None, exact_gs_energy=None, abs_delta_e=None)
    energy = _finite_float_or_none(adapt_payload.get("energy"))
    exact_gs_energy = _finite_float_or_none(adapt_payload.get("exact_gs_energy"))
    abs_delta_e = _finite_float_or_none(adapt_payload.get("abs_delta_e"))
    if abs_delta_e is None and energy is not None and exact_gs_energy is not None:
        abs_delta_e = abs(float(energy) - float(exact_gs_energy))
    return AdaptEnergyMetrics(
        energy=energy,
        exact_gs_energy=exact_gs_energy,
        abs_delta_e=abs_delta_e,
    )


def _model_name_for_problem(problem: str) -> str:
    problem_key = str(problem).strip().lower()
    if problem_key == "hh":
        return "Hubbard-Holstein"
    if problem_key == "molecular_restricted_closed_shell":
        return "Restricted closed-shell molecular"
    if problem_key == "molecular_vibronic_h2":
        return "Molecular-vibronic H2"
    if problem_key == "molecular_vibronic_h2o":
        return "Molecular-vibronic H2O"
    if problem_key == "molecular_vibronic_h2o_linear_fd":
        return "Molecular-vibronic H2O linear-FD"
    if problem_key == "ionic_hubbard":
        return "Ionic Hubbard"
    if problem_key == "extended_hubbard":
        return "Extended Hubbard"
    if problem_key == "ttprime_hubbard":
        return "t-t'-U Hubbard"
    if problem_key == "spinless_tv":
        return "Spinless t-V"
    if problem_key == "spin_boson":
        return "Spin-boson / generalized Rabi"
    if problem_key == "bose_hubbard":
        return "Bose-Hubbard"
    if problem_key == "harmonic_kerr_chain":
        return "Harmonic / Kerr boson chain"
    return "Hubbard"


def _molecular_vibronic_h2_fixture_summary(metadata: Any) -> dict[str, Any] | None:
    if not isinstance(metadata, Mapping):
        return None
    provenance = metadata.get("provenance")
    if not isinstance(provenance, Mapping):
        provenance = {}
    projection = provenance.get("active_space_projection")
    if not isinstance(projection, Mapping):
        projection = {}
    summary = {
        "schema": metadata.get("schema"),
        "family_key": metadata.get("family_key"),
        "snake_runtime_hamiltonian_scope": provenance.get("snake_runtime_hamiltonian_scope"),
        "not_full_parent_hamiltonian": provenance.get("not_full_parent_hamiltonian"),
        "projection_kind": projection.get("kind"),
        "selected_spatial_orbital_indices": projection.get("selected_spatial_orbital_indices"),
        "source_artifact_id": projection.get("source_artifact_id"),
        "parent_basis": projection.get("parent_basis"),
    }
    return {str(k): v for k, v in summary.items() if v is not None}


def _molecular_vibronic_h2o_fixture_summary(metadata: Any) -> dict[str, Any] | None:
    if not isinstance(metadata, Mapping):
        return None
    provenance = metadata.get("provenance")
    if not isinstance(provenance, Mapping):
        provenance = {}
    model = metadata.get("model")
    if not isinstance(model, Mapping):
        model = {}
    summary = {
        "schema": metadata.get("schema"),
        "family_key": metadata.get("family_key"),
        "molecule": model.get("molecule"),
        "active_space_kind": model.get("active_space_kind"),
        "selected_spatial_orbital_indices": model.get("selected_spatial_orbital_indices"),
        "source_n_spatial_orbitals": model.get("source_n_spatial_orbitals"),
        "source_n_spin_orbitals": model.get("source_n_spin_orbitals"),
        "derivative_source": model.get("derivative_source"),
        "not_full_parent_hamiltonian": provenance.get("not_full_parent_hamiltonian"),
        "source_problem_json": provenance.get("source_problem_json"),
    }
    return {str(k): v for k, v in summary.items() if v is not None}


def _molecular_vibronic_h2o_linear_fd_fixture_summary(metadata: Any) -> dict[str, Any] | None:
    if not isinstance(metadata, Mapping):
        return None
    summary = {
        "schema": metadata.get("schema"),
        "family_key": metadata.get("family_key"),
        "model_role": metadata.get("model_role"),
        "production_status": metadata.get("production_status"),
        "derivative_source": metadata.get("derivative_source"),
        "mode_labels": metadata.get("mode_labels"),
        "mode_cutoffs": metadata.get("mode_cutoffs"),
    }
    return {str(k): v for k, v in summary.items() if v is not None}


def _write_pipeline_pdf(pdf_path: Path, payload: dict[str, Any], run_command: str) -> None:
    require_matplotlib()
    settings = payload.get("settings", {})
    adapt = payload.get("adapt_vqe", {})
    problem = settings.get("problem", "hubbard")
    model_name = _model_name_for_problem(str(problem))

    manifest_sections: list[tuple[str, list[tuple[str, Any]]]] = [
        (
            "Model and regime",
            [
                ("Model family", model_name),
                ("Ansatz type", f"ADAPT-VQE (pool: {settings.get('adapt_pool', '?')})"),
                ("Drive enabled", False),
                ("L", settings.get("L")),
                ("Boundary", settings.get("boundary")),
                ("Ordering", settings.get("ordering")),
            ],
        ),
        (
            "Core physical parameters",
            [
                ("t", settings.get("t")),
                ("U", settings.get("u")),
                ("dv", settings.get("dv")),
                ("V_nn", settings.get("v_nn")),
                ("t_prime", settings.get("t_prime")),
                ("n_fermions", settings.get("n_fermions")),
            ],
        ),
        (
            "ADAPT controls",
            [
                ("ADAPT max depth", settings.get("adapt_max_depth", "?")),
                ("ADAPT eps_grad", settings.get("adapt_eps_grad", "?")),
                ("ADAPT eps_energy", settings.get("adapt_eps_energy", "?")),
                ("Inner optimizer", settings.get("adapt_inner_optimizer", "?")),
                ("Reoptimization route", settings.get("adapt_reoptimization_route", "off")),
                ("Finite-angle fallback", settings.get("adapt_finite_angle_fallback", "?")),
                ("Finite-angle probe", settings.get("adapt_finite_angle", "?")),
            ],
        ),
        (
            "Trajectory settings",
            [
                ("trotter_steps", settings.get("trotter_steps")),
                ("t_final", settings.get("t_final")),
                ("Suzuki order", settings.get("suzuki_order")),
                ("Initial state source", settings.get("initial_state_source")),
            ],
        ),
    ]
    if problem == "hh":
        manifest_sections.append(
            (
                "Hubbard-Holstein parameters",
                [
                    ("omega0", settings.get("omega0")),
                    ("g_ep", settings.get("g_ep")),
                    ("n_ph_max", settings.get("n_ph_max")),
                    ("Boson encoding", settings.get("boson_encoding")),
                ],
            )
        )
    elif str(problem).strip().lower() == "spin_boson":
        manifest_sections.append(
            (
                "Spin-boson parameters",
                [
                    ("omega0", settings.get("omega0")),
                    ("u / transverse coupling", settings.get("u")),
                    ("g_ep / longitudinal coupling", settings.get("g_ep")),
                    ("n_ph_max", settings.get("n_ph_max")),
                    ("Boson encoding", settings.get("boson_encoding")),
                ],
            )
        )
    elif str(problem).strip().lower() in {"bose_hubbard", "harmonic_kerr_chain"}:
        manifest_sections.append(
            (
                "Boson-chain parameters",
                [
                    ("omega0", settings.get("omega0")),
                    ("U / Kerr", settings.get("u")),
                    ("t", settings.get("t")),
                    ("dv", settings.get("dv")),
                    ("n_ph_max", settings.get("n_ph_max")),
                    ("Boson encoding", settings.get("boson_encoding")),
                ],
            )
        )
    elif str(problem).strip().lower() == "molecular_restricted_closed_shell":
        manifest_sections.append(
            (
                "Molecular problem source",
                [
                    ("Problem JSON", settings.get("molecular_problem_json")),
                ],
            )
        )
    elif str(problem).strip().lower() == "molecular_vibronic_h2":
        manifest_sections.append(
            (
                "Molecular-vibronic H2 fixture source",
                [
                    ("Fixture JSON", settings.get("molecular_vibronic_h2_fixture_json")),
                    ("Runtime Hamiltonian scope", (settings.get("molecular_vibronic_h2_fixture_metadata") or {}).get("snake_runtime_hamiltonian_scope")),
                    ("Not full parent Hamiltonian", (settings.get("molecular_vibronic_h2_fixture_metadata") or {}).get("not_full_parent_hamiltonian")),
                    ("Projection kind", (settings.get("molecular_vibronic_h2_fixture_metadata") or {}).get("projection_kind")),
                    ("Selected spatial orbitals", (settings.get("molecular_vibronic_h2_fixture_metadata") or {}).get("selected_spatial_orbital_indices")),
                ],
            )
        )
    elif str(problem).strip().lower() == "molecular_vibronic_h2o":
        manifest_sections.append(
            (
                "Molecular-vibronic H2O fixture source",
                [
                    ("Fixture JSON", settings.get("molecular_vibronic_h2o_fixture_json")),
                    ("Molecule", (settings.get("molecular_vibronic_h2o_fixture_metadata") or {}).get("molecule")),
                    ("Active-space kind", (settings.get("molecular_vibronic_h2o_fixture_metadata") or {}).get("active_space_kind")),
                    ("Selected spatial orbitals", (settings.get("molecular_vibronic_h2o_fixture_metadata") or {}).get("selected_spatial_orbital_indices")),
                    ("Derivative source", (settings.get("molecular_vibronic_h2o_fixture_metadata") or {}).get("derivative_source")),
                    ("Not full parent Hamiltonian", (settings.get("molecular_vibronic_h2o_fixture_metadata") or {}).get("not_full_parent_hamiltonian")),
                ],
            )
        )
    elif str(problem).strip().lower() == "molecular_vibronic_h2o_linear_fd":
        manifest_sections.append(
            (
                "Molecular-vibronic H2O linear-FD fixture source",
                [
                    ("Fixture JSON", settings.get("molecular_vibronic_h2o_linear_fd_fixture_json")),
                    ("Schema", (settings.get("molecular_vibronic_h2o_linear_fd_fixture_metadata") or {}).get("schema")),
                    ("Model role", (settings.get("molecular_vibronic_h2o_linear_fd_fixture_metadata") or {}).get("model_role")),
                    ("Production status", (settings.get("molecular_vibronic_h2o_linear_fd_fixture_metadata") or {}).get("production_status")),
                    ("Derivative source", (settings.get("molecular_vibronic_h2o_linear_fd_fixture_metadata") or {}).get("derivative_source")),
                    ("Mode labels", (settings.get("molecular_vibronic_h2o_linear_fd_fixture_metadata") or {}).get("mode_labels")),
                    ("Mode cutoffs", (settings.get("molecular_vibronic_h2o_linear_fd_fixture_metadata") or {}).get("mode_cutoffs")),
                ],
            )
        )
    if str(settings.get("adapt_inner_optimizer", "")).strip().upper() == "SPSA":
        adapt_spsa = settings.get("adapt_spsa", {})
        if isinstance(adapt_spsa, dict):
            manifest_sections.append(
                (
                    "SPSA settings",
                    [
                        ("refit_engine", adapt_spsa.get("refit_engine")),
                        ("a", adapt_spsa.get("a")),
                        ("c", adapt_spsa.get("c")),
                        ("A", adapt_spsa.get("A")),
                        ("alpha", adapt_spsa.get("alpha")),
                        ("gamma", adapt_spsa.get("gamma")),
                        ("eval_repeats", adapt_spsa.get("eval_repeats")),
                        ("eval_agg", adapt_spsa.get("eval_agg")),
                        ("avg_last", adapt_spsa.get("avg_last")),
                    ],
                )
            )

    summary_sections: list[tuple[str, list[tuple[str, Any]]]] = [
        (
            "ADAPT outcome",
            [
                ("ADAPT-VQE energy", adapt.get("energy")),
                ("Exact GS energy", adapt.get("exact_gs_energy")),
                ("|ΔE|", adapt.get("abs_delta_e")),
                ("Ansatz depth", adapt.get("ansatz_depth")),
                ("Pool size", adapt.get("pool_size")),
            ],
        ),
        (
            "Optimization summary",
            [
                ("Stop reason", adapt.get("stop_reason")),
                ("Total nfev", adapt.get("nfev_total")),
                ("Elapsed (s)", adapt.get("elapsed_s")),
                ("Inner optimizer", settings.get("adapt_inner_optimizer")),
                ("Reoptimization route", settings.get("adapt_reoptimization_route", "off")),
            ],
        ),
        (
            "Trajectory grid",
            [
                ("trotter_steps", settings.get("trotter_steps")),
                ("t_final", settings.get("t_final")),
                ("Initial state source", settings.get("initial_state_source")),
            ],
        ),
    ]

    operator_lines = [
        "Selected operators",
        "",
        f"Ansatz depth: {adapt.get('ansatz_depth')}",
        f"Pool size: {adapt.get('pool_size')}",
        f"Stop reason: {adapt.get('stop_reason')}",
        "",
    ]
    for op_label in (adapt.get("operators") or []):
        operator_lines.append(f"  {op_label}")

    with PdfPages(str(pdf_path)) as pdf:
        render_manifest_overview_page(
            pdf,
            title=f"{model_name} ADAPT-VQE report — L={settings.get('L')}",
            experiment_statement="ADAPT-VQE state preparation followed by exact-versus-Trotter trajectory diagnostics.",
            sections=manifest_sections,
            notes=[
                "The full operator list and executed command are moved to the appendix.",
            ],
        )
        render_executive_summary_page(
            pdf,
            title="Executive summary",
            experiment_statement="Prepared-state quality and convergence summary before trajectory pages.",
            sections=summary_sections,
            notes=[
                "Trajectory pages show fidelity, energy, occupations, and doublon from the ADAPT state.",
            ],
        )
        render_section_divider_page(
            pdf,
            title="Trajectory diagnostics",
            summary="Main result pages compare exact and Trotter trajectories starting from the ADAPT-prepared state.",
            bullets=[
                "Fidelity and energy.",
                "Site-0 occupations and doublon.",
            ],
        )

        rows = payload.get("trajectory", [])
        if rows:
            times = np.array([r["time"] for r in rows])
            fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), sharex=True)
            ax_f, ax_e = axes[0]
            ax_n, ax_d = axes[1]

            ax_f.plot(times, [r["fidelity"] for r in rows], color="#0b3d91")
            ax_f.set_title("Fidelity (Trotter vs Exact)")
            ax_f.set_ylabel("F(t)")
            ax_f.grid(alpha=0.25)

            ax_e.plot(times, [r["energy_trotter"] for r in rows], label="Trotter", color="#d62728")
            ax_e.plot(times, [r["energy_exact"] for r in rows], label="Exact", color="#111111", linestyle="--")
            ax_e.set_title("Energy")
            ax_e.set_ylabel("E(t)")
            ax_e.legend(fontsize=8)
            ax_e.grid(alpha=0.25)

            if str(problem).strip().lower() == "spin_boson":
                ax_n.plot(times, [r["n_up_site0_trotter"] for r in rows], label="g occ trot", color="#17becf")
                ax_n.plot(times, [r["n_dn_site0_trotter"] for r in rows], label="e occ trot", color="#9467bd")
                ax_n.set_title("Emitter occupations (Trotter)")
                ax_n.set_xlabel("Time")
                ax_n.legend(fontsize=8)
                ax_n.grid(alpha=0.25)

                ax_d.plot(times, [r.get("boson_number_trotter", 0.0) for r in rows], label="Trotter", color="#e377c2")
                ax_d.plot(times, [r.get("boson_number_exact", 0.0) for r in rows], label="Exact", color="#111111", linestyle="--")
                ax_d.set_title("Boson number")
                ax_d.set_xlabel("Time")
                ax_d.legend(fontsize=8)
                ax_d.grid(alpha=0.25)
            elif str(problem).strip().lower() in {"bose_hubbard", "harmonic_kerr_chain"}:
                ax_n.plot(times, [r["n_up_site0_trotter"] for r in rows], label="site0 boson trot", color="#17becf")
                ax_n.plot(times, [r["n_up_site0_exact"] for r in rows], label="site0 boson exact", color="#111111", linestyle="--")
                ax_n.set_title("Site-0 boson number")
                ax_n.set_xlabel("Time")
                ax_n.legend(fontsize=8)
                ax_n.grid(alpha=0.25)

                ax_d.plot(times, [r.get("boson_number_trotter", 0.0) for r in rows], label="Trotter", color="#e377c2")
                ax_d.plot(times, [r.get("boson_number_exact", 0.0) for r in rows], label="Exact", color="#111111", linestyle="--")
                ax_d.set_title("Total boson number")
                ax_d.set_xlabel("Time")
                ax_d.legend(fontsize=8)
                ax_d.grid(alpha=0.25)
            else:
                ax_n.plot(times, [r["n_up_site0_trotter"] for r in rows], label="n_up trot", color="#17becf")
                ax_n.plot(times, [r["n_dn_site0_trotter"] for r in rows], label="n_dn trot", color="#9467bd")
                ax_n.set_title("Site-0 Occupations (Trotter)")
                ax_n.set_xlabel("Time")
                ax_n.legend(fontsize=8)
                ax_n.grid(alpha=0.25)

                ax_d.plot(times, [r["doublon_trotter"] for r in rows], label="Trotter", color="#e377c2")
                ax_d.plot(times, [r["doublon_exact"] for r in rows], label="Exact", color="#111111", linestyle="--")
                ax_d.set_title("Doublon")
                ax_d.set_xlabel("Time")
                ax_d.legend(fontsize=8)
                ax_d.grid(alpha=0.25)

            fig.suptitle(f"Hardcoded ADAPT-VQE Pipeline L={settings.get('L')}", fontsize=13)
            fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
            pdf.savefig(fig)
            plt.close(fig)

        render_section_divider_page(
            pdf,
            title="Technical appendix",
            summary="Detailed operator provenance and full reproducibility material.",
            bullets=[
                "Selected operator list.",
                "Executed command.",
            ],
        )
        render_text_page(pdf, operator_lines)
        render_command_page(
            pdf,
            run_command,
            script_name="pipelines/static_adapt/adapt_pipeline.py",
        )


def build_output_payload(
    *,
    args: Any,
    cli_adapt_continuation_mode: str,
    adapt_payload: dict[str, Any],
    ordered_labels_exyz: Sequence[str],
    coeff_map_exyz: Mapping[str, complex],
    hmat: np.ndarray | None,
    gs_energy_exact: float,
    gs_energy_source: str,
    psi0: np.ndarray,
    ansatz_input_state_for_adapt: np.ndarray,
    ansatz_input_state_source: str,
    ansatz_input_state_kind: str | None,
    trajectory: Sequence[Mapping[str, Any]],
    adapt_ref_import: dict[str, Any] | None,
    dense_eigh_enabled: bool,
    hilbert_dim: int,
    adapt_ref_base_depth: int,
    initial_state_source_resolved: str,
    initial_state_kind_resolved: str,
    resolved_problem_context: Any | None = None,
    exact_reference_import: dict[str, Any] | None = None,
) -> dict[str, Any]:
    adapt_payload = dict(adapt_payload)
    output_sr_route_contract = validate_sr_route_profile_contract(
        profile_request=getattr(
            args, "sr_route_profile_request", SR_ROUTE_PROFILE_REQUEST_OFF
        ),
        contract=getattr(args, "sr_route_profile_contract", None),
        contract_sha256=getattr(
            args, "sr_route_profile_contract_sha256", None
        ),
    )
    if output_sr_route_contract is not None:
        adapt_contract = validate_sr_route_profile_contract(
            profile_request=adapt_payload.get("sr_route_profile_request"),
            contract=adapt_payload.get("sr_route_profile_contract"),
            contract_sha256=adapt_payload.get(
                "sr_route_profile_contract_sha256"
            ),
        )
        if adapt_contract != output_sr_route_contract:
            raise ValueError(
                "Output SR-SNAKE route contract disagrees with runtime telemetry."
            )
    formal_manifold_route_composition = (
        _resolved_output_formal_manifold_route_composition(adapt_payload)
    )
    if formal_manifold_route_composition is not None:
        adapt_payload["formal_manifold_route_composition"] = dict(
            formal_manifold_route_composition
        )
    if ordered_labels_exyz:
        num_qubits = int(len(ordered_labels_exyz[0]))
    elif hmat is not None:
        num_qubits = int(round(math.log2(hmat.shape[0])))
    else:
        num_qubits = 0

    continuation_block = adapt_payload.get("continuation", {})
    hardware_resolution_resolved = (
        dict(continuation_block.get("hardware_resolution", {}))
        if isinstance(continuation_block, Mapping)
        and isinstance(continuation_block.get("hardware_resolution", None), Mapping)
        else None
    )
    runtime_data = getattr(resolved_problem_context, "runtime_data", {}) if resolved_problem_context is not None else {}
    fixture_summary = _molecular_vibronic_h2_fixture_summary(
        runtime_data.get("vibronic_h2_fixture_metadata") if isinstance(runtime_data, Mapping) else None
    )
    h2o_fixture_summary = _molecular_vibronic_h2o_fixture_summary(
        runtime_data.get("vibronic_h2o_fixture_metadata") if isinstance(runtime_data, Mapping) else None
    )
    h2o_linear_fd_fixture_summary = _molecular_vibronic_h2o_linear_fd_fixture_summary(
        runtime_data.get("vibronic_h2o_linear_fd_fixture_metadata") if isinstance(runtime_data, Mapping) else None
    )
    (
        resolved_powell_coordinate_chart_policy,
        resolved_powell_coordinate_chart_policy_source,
    ) = _resolved_output_powell_coordinate_chart_policy(
        args=args,
        adapt_payload=adapt_payload,
    )
    (
        resolved_phase3_response_coordinate_scope,
        resolved_phase3_response_coordinate_scope_source,
    ) = _resolved_output_phase3_response_coordinate_scope(
        args=args,
        adapt_payload=adapt_payload,
    )
    (
        resolved_phase12_energy_model_policies,
        resolved_phase12_energy_model_policy_source,
    ) = _resolved_output_phase12_energy_model_policies(
        args=args,
        adapt_payload=adapt_payload,
    )

    payload: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "hardcoded_adapt",
        "settings": {
            "L": int(args.L),
            "t": float(args.t),
            "u": float(args.u),
            "problem": str(args.problem),
            "sr_route_profile_request": str(
                getattr(
                    args,
                    "sr_route_profile_request",
                    SR_ROUTE_PROFILE_REQUEST_OFF,
                )
            ),
            "sr_route_profile_resolved": getattr(
                args, "sr_route_profile_resolved", None
            ),
            "sr_route_profile_contract": (
                None
                if output_sr_route_contract is None
                else dict(output_sr_route_contract)
            ),
            "sr_route_profile_contract_sha256": getattr(
                args, "sr_route_profile_contract_sha256", None
            ),
            "phase3_response_coordinate_scope_requested": str(
                getattr(args, "phase3_response_coordinate_scope", "")
            ),
            "phase3_response_coordinate_scope": (
                resolved_phase3_response_coordinate_scope
            ),
            "phase3_response_coordinate_scope_source": (
                resolved_phase3_response_coordinate_scope_source
            ),
            **dict(resolved_phase12_energy_model_policies),
            "phase12_energy_model_policy_source": str(
                resolved_phase12_energy_model_policy_source
            ),
            "molecular_problem_json": (
                None
                if getattr(args, "molecular_problem_json", None) is None
                else str(args.molecular_problem_json)
            ),
            "molecular_vibronic_h2_fixture_json": (
                None
                if getattr(args, "molecular_vibronic_h2_fixture_json", None) is None
                else str(args.molecular_vibronic_h2_fixture_json)
            ),
            "molecular_vibronic_h2_fixture_metadata": fixture_summary,
            "molecular_vibronic_h2o_fixture_json": (
                None
                if getattr(args, "molecular_vibronic_h2o_fixture_json", None) is None
                else str(args.molecular_vibronic_h2o_fixture_json)
            ),
            "molecular_vibronic_h2o_fixture_metadata": h2o_fixture_summary,
            "molecular_vibronic_h2o_linear_fd_fixture_json": (
                None
                if getattr(args, "molecular_vibronic_h2o_linear_fd_fixture_json", None) is None
                else str(args.molecular_vibronic_h2o_linear_fd_fixture_json)
            ),
            "molecular_vibronic_h2o_linear_fd_fixture_metadata": h2o_linear_fd_fixture_summary,
            "v_nn": float(getattr(args, "v_nn", 0.0)),
            "t_prime": float(getattr(args, "t_prime", 0.0)),
            "n_fermions": (
                None
                if getattr(args, "n_fermions", None) is None
                else int(args.n_fermions)
            ),
            "omega0": float(args.omega0),
            "g_ep": float(args.g_ep),
            "n_ph_max": int(args.n_ph_max),
            "boson_encoding": str(args.boson_encoding),
            "dv": float(args.dv),
            "boundary": str(args.boundary),
            "ordering": str(args.ordering),
            "t_final": float(args.t_final),
            "num_times": int(args.num_times),
            "suzuki_order": int(args.suzuki_order),
            "trotter_steps": int(args.trotter_steps),
            "term_order": str(args.term_order),
            "dense_eigh_max_dim": int(args.dense_eigh_max_dim),
            "dense_eigh_enabled": bool(dense_eigh_enabled),
            "hilbert_dim": int(hilbert_dim),
            "adapt_pool": (
                str(args.adapt_pool)
                if args.adapt_pool is not None
                else (
                    adapt_payload.get("pool_type")
                    if adapt_payload.get("pool_type") not in {None, ""}
                    else None
                )
            ),
            "adapt_pool_requested": (
                str(args.adapt_pool) if args.adapt_pool is not None else None
            ),
            "adapt_pool_class_filter_json": (
                str(args.adapt_pool_class_filter_json)
                if args.adapt_pool_class_filter_json is not None
                else None
            ),
            "adapt_pool_label_filter_json": (
                str(args.adapt_pool_label_filter_json)
                if args.adapt_pool_label_filter_json is not None
                else None
            ),
            "adapt_pool_class_filter_classifier_version": (
                adapt_payload.get("adapt_pool_class_filter_classifier_version")
            ),
            "adapt_pool_class_filter_keep_classes": (
                adapt_payload.get("adapt_pool_class_filter_keep_classes")
            ),
            "adapt_pool_label_filter_classifier_version": (
                adapt_payload.get("adapt_pool_label_filter_classifier_version")
            ),
            "adapt_pool_label_filter_drop_labels": (
                adapt_payload.get("adapt_pool_label_filter_drop_labels")
            ),
            "adapt_pool_label_filter_drop_prefixes": (
                adapt_payload.get("adapt_pool_label_filter_drop_prefixes")
            ),
            "adapt_continuation_mode": str(cli_adapt_continuation_mode),
            "adapt_max_depth": int(args.adapt_max_depth),
            "adapt_eps_grad": float(args.adapt_eps_grad),
            "adapt_eps_energy": float(args.adapt_eps_energy),
            "adapt_exact_gs_override": (
                None
                if getattr(args, "adapt_exact_gs_override", None) is None
                else float(getattr(args, "adapt_exact_gs_override"))
            ),
            "adapt_exact_gs_reference_json": (
                None
                if getattr(args, "adapt_exact_gs_reference_json", None) is None
                else str(getattr(args, "adapt_exact_gs_reference_json"))
            ),
            "adapt_inner_optimizer": str(args.adapt_inner_optimizer),
            "adapt_reoptimization_route": str(
                getattr(args, "adapt_reoptimization_route", "off")
            ),
            "adapt_scipy_maxfev": int(getattr(args, "adapt_scipy_maxfev", 0)),
            "adapt_state_backend": str(args.adapt_state_backend),
            "adapt_finite_angle_fallback": bool(args.adapt_finite_angle_fallback),
            "adapt_finite_angle": float(args.adapt_finite_angle),
            "adapt_finite_angle_min_improvement": float(args.adapt_finite_angle_min_improvement),
            "adapt_drop_floor": (float(args.adapt_drop_floor) if args.adapt_drop_floor is not None else None),
            "adapt_drop_patience": (int(args.adapt_drop_patience) if args.adapt_drop_patience is not None else None),
            "adapt_drop_min_depth": (int(args.adapt_drop_min_depth) if args.adapt_drop_min_depth is not None else None),
            "adapt_grad_floor": (float(args.adapt_grad_floor) if args.adapt_grad_floor is not None else None),
            "adapt_drop_floor_resolved": adapt_payload.get("adapt_drop_floor_resolved"),
            "adapt_drop_patience_resolved": adapt_payload.get("adapt_drop_patience_resolved"),
            "adapt_drop_min_depth_resolved": adapt_payload.get("adapt_drop_min_depth_resolved"),
            "adapt_grad_floor_resolved": adapt_payload.get("adapt_grad_floor_resolved"),
            "adapt_drop_floor_source": adapt_payload.get("adapt_drop_floor_source"),
            "adapt_drop_patience_source": adapt_payload.get("adapt_drop_patience_source"),
            "adapt_drop_min_depth_source": adapt_payload.get("adapt_drop_min_depth_source"),
            "adapt_grad_floor_source": adapt_payload.get("adapt_grad_floor_source"),
            "adapt_drop_policy_source": adapt_payload.get("adapt_drop_policy_source"),
            "hardware_resolution_mode": str(getattr(args, "hardware_resolution_mode", "ideal")),
            "gradient_hw_floor": float(getattr(args, "gradient_hw_floor", 0.0)),
            "gradient_drift_floor": float(getattr(args, "gradient_drift_floor", 0.0)),
            "hardware_resolution_profile_json": (
                None
                if getattr(args, "hardware_resolution_profile_json", None) is None
                else str(getattr(args, "hardware_resolution_profile_json"))
            ),
            "hardware_resolution_profile_name": (
                None
                if getattr(args, "hardware_resolution_profile_name", None) in {None, ""}
                else str(getattr(args, "hardware_resolution_profile_name"))
            ),
            "hardware_resolution_resolved": hardware_resolution_resolved,
            "adapt_eps_energy_min_extra_depth": int(args.adapt_eps_energy_min_extra_depth),
            "adapt_eps_energy_patience": int(args.adapt_eps_energy_patience),
            "adapt_ref_base_depth": int(adapt_ref_base_depth),
            "adapt_gradient_parity_check": bool(args.adapt_gradient_parity_check),
            "adapt_analytic_noise_std": float(args.adapt_analytic_noise_std),
            "adapt_analytic_noise_seed": (
                None
                if args.adapt_analytic_noise_seed is None
                else int(args.adapt_analytic_noise_seed)
            ),
            "adapt_seed": int(args.adapt_seed),
            "adapt_reopt_policy": str(args.adapt_reopt_policy),
            "adapt_window_size": int(args.adapt_window_size),
            "adapt_window_topk": int(args.adapt_window_topk),
            "adapt_full_refit_every": int(args.adapt_full_refit_every),
            "adapt_final_full_refit": str(args.adapt_final_full_refit),
            "adapt_final_refit_maxiter": int(getattr(args, "adapt_final_refit_maxiter", 0)),
            "phase1_lambda_compile": float(args.phase1_lambda_compile),
            "phase1_lambda_measure": float(args.phase1_lambda_measure),
            "phase1_lambda_leak": float(args.phase1_lambda_leak),
            "phase1_score_z_alpha": float(args.phase1_score_z_alpha),
            "phase1_score_mode": str(getattr(args, "phase1_score_mode", "trust_region_v1")),
            "phase1_depth_ref": float(args.phase1_depth_ref),
            "phase1_group_ref": float(args.phase1_group_ref),
            "phase1_shot_ref": float(args.phase1_shot_ref),
            "phase1_family_ref": float(args.phase1_family_ref),
            "phase1_compile_cx_proxy_weight": float(args.phase1_compile_cx_proxy_weight),
            "phase1_compile_sq_proxy_weight": float(args.phase1_compile_sq_proxy_weight),
            "phase1_compile_rotation_step_weight": float(args.phase1_compile_rotation_step_weight),
            "phase1_compile_position_shift_weight": float(args.phase1_compile_position_shift_weight),
            "phase1_compile_refit_active_weight": float(args.phase1_compile_refit_active_weight),
            "phase1_measure_groups_weight": float(args.phase1_measure_groups_weight),
            "phase1_measure_shots_weight": float(args.phase1_measure_shots_weight),
            "phase1_measure_reuse_weight": float(args.phase1_measure_reuse_weight),
            "phase1_opt_dim_cost_scale": float(args.phase1_opt_dim_cost_scale),
            "phase1_family_repeat_cost_scale": float(args.phase1_family_repeat_cost_scale),
            "phase1_shortlist_size": int(args.phase1_shortlist_size),
            "phase1_probe_max_positions": int(args.phase1_probe_max_positions),
            "phase1_plateau_patience": int(args.phase1_plateau_patience),
            "phase1_trough_margin_ratio": float(args.phase1_trough_margin_ratio),
            "phase2_shortlist_fraction": float(args.phase2_shortlist_fraction),
            "phase2_shortlist_size": int(args.phase2_shortlist_size),
            "phase2_lambda_H": float(args.phase2_lambda_H),
            "phase2_rho": float(args.phase2_rho),
            "phase2_score_z_alpha": (
                float(args.phase2_score_z_alpha)
                if args.phase2_score_z_alpha is not None
                else None
            ),
            "phase2_depth_ref": float(args.phase2_depth_ref),
            "phase2_group_ref": float(args.phase2_group_ref),
            "phase2_shot_ref": float(args.phase2_shot_ref),
            "phase2_optdim_ref": float(args.phase2_optdim_ref),
            "phase2_reuse_ref": float(args.phase2_reuse_ref),
            "phase2_family_ref": float(args.phase2_family_ref),
            "deferred_gram_fallback_ridge": float(
                getattr(args, "deferred_gram_fallback_ridge", 1e-6)
            ),
            "phase2_selector_gain_mode": str(getattr(args, "phase2_selector_gain_mode", "trust_region_v1")),
            "phase2_cheap_score_eps": float(args.phase2_cheap_score_eps),
            "phase2_metric_floor": float(args.phase2_metric_floor),
            "phase2_reduced_metric_collapse_rel_tol": float(
                args.phase2_reduced_metric_collapse_rel_tol
            ),
            "adapt_schur_warm_start_mode": str(getattr(args, "adapt_schur_warm_start_mode", "off")),
            "phase2_ridge_growth_factor": float(args.phase2_ridge_growth_factor),
            "phase2_ridge_max_steps": int(args.phase2_ridge_max_steps),
            "phase2_leakage_cap": float(args.phase2_leakage_cap),
            "phase2_compile_cx_proxy_weight": float(args.phase2_compile_cx_proxy_weight),
            "phase2_compile_sq_proxy_weight": float(args.phase2_compile_sq_proxy_weight),
            "phase2_compile_rotation_step_weight": float(args.phase2_compile_rotation_step_weight),
            "phase2_compile_position_shift_weight": float(args.phase2_compile_position_shift_weight),
            "phase2_compile_refit_active_weight": float(args.phase2_compile_refit_active_weight),
            "phase2_measure_groups_weight": float(args.phase2_measure_groups_weight),
            "phase2_measure_shots_weight": float(args.phase2_measure_shots_weight),
            "phase2_measure_reuse_weight": float(args.phase2_measure_reuse_weight),
            "phase2_opt_dim_cost_scale": float(args.phase2_opt_dim_cost_scale),
            "phase2_family_repeat_cost_scale": float(args.phase2_family_repeat_cost_scale),
            "phase2_w_depth": float(args.phase2_w_depth),
            "phase2_w_group": float(args.phase2_w_group),
            "phase2_w_shot": float(args.phase2_w_shot),
            "phase2_w_optdim": float(args.phase2_w_optdim),
            "phase2_w_reuse": float(args.phase2_w_reuse),
            "phase2_w_lifetime": float(args.phase2_w_lifetime),
            "phase2_eta_L": float(args.phase2_eta_L),
            "phase2_motif_bonus_weight": float(args.phase2_motif_bonus_weight),
            "phase2_duplicate_penalty_weight": float(args.phase2_duplicate_penalty_weight),
            "phase2_frontier_ratio": float(args.phase2_frontier_ratio),
            "phase3_frontier_ratio": float(args.phase3_frontier_ratio),
            "phase3_tie_beam_score_ratio": float(args.phase3_tie_beam_score_ratio),
            "phase3_tie_beam_abs_tol": float(args.phase3_tie_beam_abs_tol),
            "phase3_tie_beam_max_branches": int(args.phase3_tie_beam_max_branches),
            "phase3_tie_beam_max_late_coordinate": float(args.phase3_tie_beam_max_late_coordinate),
            "phase3_tie_beam_min_depth_left": int(args.phase3_tie_beam_min_depth_left),
            "phase2_remaining_evaluations_proxy_mode": str(
                args.phase2_remaining_evaluations_proxy_mode
            ),
            "phase3_motif_source_json": (
                str(args.phase3_motif_source_json)
                if args.phase3_motif_source_json is not None
                else None
            ),
            "phase3_symmetry_mitigation_mode": str(args.phase3_symmetry_mitigation_mode),
            "phase3_enable_rescue": bool(args.phase3_enable_rescue),
            "phase3_lifetime_cost_mode": str(args.phase3_lifetime_cost_mode),
            "phase3_hardware_cost_normalization_mode": str(args.phase3_hardware_cost_normalization_mode),
            "phase3_shadow_damping_policy": str(
                getattr(args, "phase3_shadow_damping_policy", "off")
            ),
            "phase3_source_lock_preferred_sequence": str(
                getattr(args, "phase3_source_lock_preferred_sequence", "")
            ),
            "phase3_runtime_split_mode": str(args.phase3_runtime_split_mode),
            "phase3_runtime_split_selection_mode": str(args.phase3_runtime_split_selection_mode),
            "phase3_runtime_split_max_subset_size": int(args.phase3_runtime_split_max_subset_size),
            "phase3_runtime_split_child_set_symmetry_policy": str(
                getattr(args, "phase3_runtime_split_child_set_symmetry_policy", "parent")
            ),
            "adapt_child_pool_expansion_mode": str(
                getattr(args, "adapt_child_pool_expansion_mode", "off")
            ),
            "adapt_child_pool_expansion_symmetry_policy": str(
                getattr(args, "adapt_child_pool_expansion_symmetry_policy", "off")
            ),
            "adapt_child_pool_expansion_max_subset_size": int(
                getattr(args, "adapt_child_pool_expansion_max_subset_size", 3)
            ),
            "shared_pauli_pool_mode": str(getattr(args, "shared_pauli_pool_mode", "off")),
            "shared_pauli_pool_symmetry_policy": str(
                getattr(args, "shared_pauli_pool_symmetry_policy", "off")
            ),
            "shared_pauli_pool_max_subset_size": int(
                getattr(args, "shared_pauli_pool_max_subset_size", 3)
            ),
            "phase3_selector_geometry_mode": str(args.phase3_selector_geometry_mode),
            "phase3_parent_collapse_debug_max_depth": int(args.phase3_parent_collapse_debug_max_depth),
            "phase3_backend_cost_mode": str(args.phase3_backend_cost_mode),
            "phase3_backend_name": (
                None if args.phase3_backend_name in {None, ""} else str(args.phase3_backend_name)
            ),
            "phase3_backend_shortlist": (
                []
                if args.phase3_backend_shortlist in {None, ""}
                else [str(tok).strip() for tok in str(args.phase3_backend_shortlist).split(",") if str(tok).strip() != ""]
            ),
            "phase3_backend_transpile_seed": int(args.phase3_backend_transpile_seed),
            "phase3_backend_optimization_level": int(args.phase3_backend_optimization_level),
            "phase3_oracle_inner_objective_mode": str(
                adapt_payload.get(
                    "phase3_oracle_inner_objective_mode",
                    args.phase3_oracle_inner_objective_mode,
                )
            ),
            "phase3_oracle_inner_objective_mode_requested": str(
                adapt_payload.get(
                    "phase3_oracle_inner_objective_mode_requested",
                    args.phase3_oracle_inner_objective_mode,
                )
            ),
            "phase3_oracle_inner_objective_runtime_guard_reason": (
                adapt_payload.get("phase3_oracle_inner_objective_runtime_guard_reason")
            ),
            "adapt_ref_json": (str(args.adapt_ref_json) if args.adapt_ref_json is not None else None),
            "adapt_resume_scaffold_json": (
                str(args.adapt_resume_scaffold_json)
                if getattr(args, "adapt_resume_scaffold_json", None) is not None
                else None
            ),
            "adapt_resume_mode": str(getattr(args, "adapt_resume_mode", "scaffold_v1")),
            "adapt_resume_boundary_refit_policy": str(
                getattr(args, "adapt_resume_boundary_refit_policy", "required")
            ),
            "adapt_segment_id": (
                None
                if getattr(args, "adapt_segment_id", None) in {None, ""}
                else str(args.adapt_segment_id)
            ),
            "adapt_segment_target_depth": (
                None
                if getattr(args, "adapt_segment_target_depth", None) is None
                else int(args.adapt_segment_target_depth)
            ),
            "adapt_segment_target_controller_round": (
                None
                if getattr(args, "adapt_segment_target_controller_round", None)
                is None
                else int(args.adapt_segment_target_controller_round)
            ),
            "adapt_segment_max_new_admissions": (
                None
                if getattr(args, "adapt_segment_max_new_admissions", None) is None
                else int(args.adapt_segment_max_new_admissions)
            ),
            "adapt_segment_wallclock_cap_s": (
                None
                if getattr(args, "adapt_segment_wallclock_cap_s", None) is None
                else float(args.adapt_segment_wallclock_cap_s)
            ),
            "adapt_resume_compile_smoke": str(
                getattr(args, "adapt_resume_compile_smoke", "auto")
            ),
            "adapt_resume_smoke_backend": str(
                getattr(args, "adapt_resume_smoke_backend", "FakeMarrakesh")
            ),
            "paop_r": int(args.paop_r),
            "paop_split_paulis": bool(args.paop_split_paulis),
            "paop_prune_eps": float(args.paop_prune_eps),
            "paop_normalization": str(args.paop_normalization),
            "static_route_id": str(
                getattr(args, "static_route_id", "unspecified")
            ),
            "static_meta_feature_profile": str(
                getattr(args, "static_meta_feature_profile", "off")
            ),
            "static_lane_route": str(
                getattr(args, "static_lane_route", "algebraic")
            ),
            "phase1_lane_retention_enabled": bool(
                getattr(args, "phase1_lane_retention_enabled", True)
            ),
            "historical_singleton_coordinate_solve_policy": str(
                getattr(
                    args,
                    "historical_singleton_coordinate_solve_policy",
                    "archival_reduced_scalar_v1",
                )
            ),
            "historical_singleton_coordinate_solve_scope": str(
                getattr(
                    args,
                    "historical_singleton_coordinate_solve_scope",
                    "phase3_only_v1",
                )
            ),
            "sr_controller_ablation_contract": str(
                getattr(
                    args,
                    "sr_controller_ablation_contract",
                    SR_CONTROLLER_ABLATION_CONTRACT_OFF,
                )
            ),
            "phase2_gram_novelty_policy": str(
                getattr(args, "phase2_gram_novelty_policy", "off")
            ),
            "phase3_gram_novelty_policy": str(
                getattr(args, "phase3_gram_novelty_policy", "off")
            ),
            "deferred_gram_all_models_infeasible_fallback_enabled": bool(
                str(
                    getattr(args, "phase2_gram_novelty_policy", "off")
                ).strip().lower()
                == "fallback_only_v1"
                and str(
                    getattr(args, "phase3_gram_novelty_policy", "off")
                ).strip().lower()
                == "fallback_only_v1"
            ),
            "historical_singleton_trust_region_update_policy": str(
                getattr(
                    args,
                    "historical_singleton_trust_region_update_policy",
                    "fixed",
                )
            ),
            "sr_escape_mode": str(
                getattr(args, "sr_escape_mode", "disabled")
            ),
            "sr_powell_coordinate_chart_policy_requested": str(
                getattr(
                    args,
                    "sr_powell_coordinate_chart_policy",
                    SR_POWELL_COORDINATE_CHART_AUTO,
                )
            ),
            "sr_powell_coordinate_chart_policy": (
                resolved_powell_coordinate_chart_policy
            ),
            "sr_powell_coordinate_chart_policy_source": (
                resolved_powell_coordinate_chart_policy_source
            ),
            "route_family": (
                formal_manifold_route_composition.get("route_family")
                if formal_manifold_route_composition is not None
                else (
                    adapt_payload.get("static_route_identity", {}).get(
                        "route_family"
                    )
                    if isinstance(
                        adapt_payload.get("static_route_identity"), Mapping
                    )
                    else None
                )
            ),
            "route_profile": (
                formal_manifold_route_composition.get("route_profile")
                if formal_manifold_route_composition is not None
                else (
                    adapt_payload.get("static_route_identity", {}).get(
                        "route_profile"
                    )
                    if isinstance(
                        adapt_payload.get("static_route_identity"), Mapping
                    )
                    else None
                )
            ),
            "route_profile_conformance": (
                adapt_payload.get("static_route_identity", {}).get(
                    "route_profile_conformance"
                )
                if isinstance(
                    adapt_payload.get("static_route_identity"), Mapping
                )
                else None
            ),
            "candidate_selector_family": (
                None
                if formal_manifold_route_composition is None
                else formal_manifold_route_composition.get(
                    "candidate_selector_family"
                )
            ),
            "candidate_selector_profile": (
                None
                if formal_manifold_route_composition is None
                else formal_manifold_route_composition.get(
                    "candidate_selector_profile"
                )
            ),
            "adapt_reoptimization_route": (
                adapt_payload.get("adapt_reoptimization_route")
                if formal_manifold_route_composition is None
                else formal_manifold_route_composition.get(
                    "adapt_reoptimization_route"
                )
            ),
            "formal_manifold_route_composition": (
                None
                if formal_manifold_route_composition is None
                else dict(formal_manifold_route_composition)
            ),
            "formal_manifold_route_composition_sha256": (
                None
                if formal_manifold_route_composition is None
                else str(formal_manifold_route_composition["sha256"])
            ),
            "initial_state_source": str(args.initial_state_source),
        },
        "hamiltonian": {
            "num_qubits": int(num_qubits),
            "num_terms": int(len(coeff_map_exyz)),
            "coefficients_exyz": [
                {
                    "label_exyz": lbl,
                    "coeff": {"re": float(np.real(coeff_map_exyz[lbl])), "im": float(np.imag(coeff_map_exyz[lbl]))},
                }
                for lbl in ordered_labels_exyz
            ],
        },
        "ground_state": {
            "exact_energy": float(gs_energy_exact),
            "exact_energy_source": str(gs_energy_source),
            "method": (
                "python_matrix_eigendecomposition"
                if hmat is not None
                else "sector_exact_only_no_dense_eigh"
            ),
        },
        "adapt_vqe": adapt_payload,
        "formal_manifold_route_composition": (
            None
            if formal_manifold_route_composition is None
            else dict(formal_manifold_route_composition)
        ),
        "initial_state": build_statevector_manifest(
            psi_state=np.asarray(psi0, dtype=complex).reshape(-1),
            source=initial_state_source_resolved,
            handoff_state_kind=initial_state_kind_resolved,
            amplitude_cutoff=1e-12,
        ),
        "ansatz_input_state": build_statevector_manifest(
            psi_state=np.asarray(ansatz_input_state_for_adapt, dtype=complex).reshape(-1),
            source=str(ansatz_input_state_source),
            handoff_state_kind=ansatz_input_state_kind,
            amplitude_cutoff=1e-12,
        ),
        "trajectory": trajectory,
    }
    if str(args.adapt_inner_optimizer).strip().upper() == "SPSA":
        adapt_spsa_payload = adapt_payload.get("adapt_spsa", {})
        adapt_spsa_engine = (
            adapt_spsa_payload.get("refit_engine")
            if isinstance(adapt_spsa_payload, Mapping)
            else None
        )
        payload["settings"]["adapt_spsa"] = {
            "refit_engine": (
                str(adapt_spsa_engine)
                if adapt_spsa_engine not in {None, ""}
                else resolve_adapt_spsa_refit_engine_label()
            ),
            "refit_engine_env": ADAPT_SPSA_REFIT_ENGINE_ENV,
            "a": float(args.adapt_spsa_a),
            "c": float(args.adapt_spsa_c),
            "alpha": float(args.adapt_spsa_alpha),
            "gamma": float(args.adapt_spsa_gamma),
            "A": float(args.adapt_spsa_A),
            "avg_last": int(args.adapt_spsa_avg_last),
            "eval_repeats": int(args.adapt_spsa_eval_repeats),
            "eval_agg": str(args.adapt_spsa_eval_agg),
            "callback_every": int(args.adapt_spsa_callback_every),
            "progress_every_s": float(args.adapt_spsa_progress_every_s),
            "parallel_evaluations": int(getattr(args, "adapt_spsa_parallel_evaluations", 1)),
        }
    if adapt_ref_import is not None:
        adapt_ref_import["ansatz_input_state_persisted"] = True
        payload["adapt_ref_import"] = adapt_ref_import
    if exact_reference_import is not None:
        payload["exact_reference_import"] = dict(exact_reference_import)
    if isinstance(adapt_payload.get("adapt_resume_import"), Mapping):
        payload["adapt_resume_import"] = dict(adapt_payload["adapt_resume_import"])
    if isinstance(adapt_payload.get("adapt_segment"), Mapping):
        payload["adapt_segment"] = dict(adapt_payload["adapt_segment"])
    credential_audit = adapt_payload.get("credential_audit")
    if isinstance(credential_audit, Mapping):
        payload["credential_audit"] = dict(credential_audit)
    else:
        payload["credential_audit"] = {
            "schema_version": "static_hh_adapt_runtime_audit_v1",
            "cli_accepts_credentials": False,
            "environment_serialized": False,
            "runtime_credentials_serialized": False,
            "no_credentials_serialized": True,
        }
    return payload


def persist_output_artifacts(
    *,
    output_json: Path,
    output_pdf: Path,
    payload: Mapping[str, Any],
    run_command: str,
    skip_pdf: bool,
    ai_log: Callable[..., None] | None = None,
    safe_stdout_print: Callable[..., bool] | None = None,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    persisted_payload = _candidate_record_cache_jsonable(payload)
    output_json.write_text(json.dumps(persisted_payload, indent=2), encoding="utf-8")
    if not bool(skip_pdf):
        _write_pipeline_pdf(output_pdf, dict(persisted_payload), run_command)

    if ai_log is not None:
        settings = persisted_payload.get("settings", {}) if isinstance(persisted_payload, Mapping) else {}
        adapt_payload = persisted_payload.get("adapt_vqe", {}) if isinstance(persisted_payload, Mapping) else {}
        static_route_payload = (
            adapt_payload.get("static_route_identity")
            if isinstance(adapt_payload, Mapping)
            else None
        )
        ai_log(
            "hardcoded_adapt_main_done",
            L=int(settings.get("L", 0) or 0),
            problem=str(settings.get("problem", "")),
            output_json=str(output_json),
            output_pdf=(str(output_pdf) if not bool(skip_pdf) else None),
            adapt_energy=(
                adapt_payload.get("energy")
                if isinstance(adapt_payload, Mapping)
                else None
            ),
            stop_reason=(
                adapt_payload.get("stop_reason")
                if isinstance(adapt_payload, Mapping)
                else None
            ),
            ansatz_depth=(
                adapt_payload.get("ansatz_depth")
                if isinstance(adapt_payload, Mapping)
                else None
            ),
            abs_delta_e=(
                adapt_payload.get("abs_delta_e")
                if isinstance(adapt_payload, Mapping)
                else None
            ),
            success=(
                adapt_payload.get("success")
                if isinstance(adapt_payload, Mapping)
                else None
            ),
            static_route_id=(
                static_route_payload.get("route_id")
                if isinstance(static_route_payload, Mapping)
                else None
            ),
        )

    if safe_stdout_print is not None:
        safe_stdout_print(f"Wrote JSON: {output_json}")
        if not bool(skip_pdf):
            safe_stdout_print(f"Wrote PDF:  {output_pdf}")
