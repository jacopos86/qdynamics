
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


from docs.reports.pdf_utils import (
    HAS_MATPLOTLIB,
    get_PdfPages,
    get_plt,
)
from pipelines.static_adapt.historical_formal_manifold_provenance import (
    FORMAL_MANIFOLD_ROUTE,
    FORMAL_MANIFOLD_ROUTE_FAMILY,
    FORMAL_MANIFOLD_SR_SELECTOR_FAMILY,
    FormalManifoldRouteComposition,
)
from pipelines.static_adapt.sr_snake_escape_controller import (
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
)
from pipelines.static_adapt.sr_snake_phase12_policy import (
    normalize_phase1_energy_model,
    normalize_phase1_score_mode_policy,
    normalize_phase2_cheap_curvature_proxy_policy,
    normalize_phase2_curvature_policy,
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














