#!/usr/bin/env python3
"""Detect-only sidecar replay for exported ADAPT artifacts."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pipelines.error_protected.contracts import DetectionReplayInput, ErrorDetectionConfig
from pipelines.error_protected.raw_acquisition import (
    acquire_raw_bundle,
    build_energy_observable,
    build_full_register_z_observable,
    build_parameterized_plan_from_runtime_input,
    build_raw_oracle_config,
    load_replay_runtime_input,
    validate_detect_only_request,
)
from pipelines.error_protected.reporting import (
    build_detect_only_sidecar_payload,
    write_detect_only_sidecar_json,
)
from pipelines.error_protected.shot_postprocess import (
    evaluate_sector_filter_observability,
    sector_audit_unavailable,
    sector_filter_unavailable,
    summarize_sector_audit,
    validate_correction_mode_off,
)


@dataclass(frozen=True)
class ResolvedDetectionContext:
    problem: str
    num_sites: int | None
    num_qubits: int | None
    fermion_qubits: int | None
    boson_qubits: int | None
    ordering: str | None
    ordering_source: str
    sector_n_up: int | None
    sector_n_dn: int | None
    sector_source: str
    supported_family: bool
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-json", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--noise-mode", choices=["runtime", "backend_scheduled"], required=True)
    p.add_argument("--execution-surface", choices=["raw_measurement_v1"], default="raw_measurement_v1")
    p.add_argument("--backend-name", type=str, default=None)
    p.add_argument("--use-fake-backend", action="store_true")
    p.add_argument("--shots", type=int, default=2048)
    p.add_argument("--oracle-repeats", type=int, default=1)
    p.add_argument("--oracle-aggregate", choices=["mean", "median"], default="mean")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--seed-transpiler", type=int, default=None)
    p.add_argument("--transpile-optimization-level", type=int, default=1)
    p.add_argument("--raw-transport", type=str, default="auto")
    p.add_argument(
        "--raw-grouping-mode",
        choices=["qwc_basis_cover_reuse"],
        default="qwc_basis_cover_reuse",
    )
    p.add_argument("--raw-store-memory", dest="raw_store_memory", action="store_true", default=True)
    p.add_argument("--no-raw-store-memory", dest="raw_store_memory", action="store_false")
    p.add_argument("--runtime-profile", type=str, default="legacy_runtime_v0")
    p.add_argument("--runtime-session", type=str, default="prefer_session")
    p.add_argument("--raw-artifact-path", type=Path, default=None)
    p.add_argument(
        "--detection-mode",
        choices=["off", "sector_audit", "sector_filter", "sector_detect"],
        default="sector_audit",
    )
    p.set_defaults(strict=True)
    p.add_argument("--strict", dest="strict", action="store_true")
    p.add_argument("--no-strict", dest="strict", action="store_false")
    p.add_argument("--min-accepted-shots", type=int, default=256)
    p.add_argument("--min-acceptance-rate", type=float, default=None)
    p.add_argument("--sector-n-up", type=int, default=None)
    p.add_argument("--sector-n-dn", type=int, default=None)
    p.add_argument("--ordering", choices=["blocked", "interleaved"], default=None)
    p.add_argument("--symmetry-mode", choices=["off", "verify_only"], default="verify_only")
    p.add_argument("--correction-mode", choices=["off"], default="off")
    p.add_argument("--loader-mode", type=str, default=None)
    return p


def _normalize_detection_mode(mode: str) -> str:
    mode_norm = str(mode).strip().lower()
    if mode_norm == "sector_detect":
        return "sector_audit"
    return mode_norm


def resolve_detection_replay_input(args: argparse.Namespace) -> DetectionReplayInput:
    min_acceptance_rate = args.min_acceptance_rate
    if min_acceptance_rate is not None and not (0.0 <= float(min_acceptance_rate) <= 1.0):
        raise ValueError("min_acceptance_rate must be in [0, 1] when provided.")
    detection = ErrorDetectionConfig(
        mode=_normalize_detection_mode(str(args.detection_mode)),  # type: ignore[arg-type]
        strict=bool(args.strict),
        min_accepted_shots=int(args.min_accepted_shots),
        min_acceptance_rate=None if min_acceptance_rate is None else float(min_acceptance_rate),
        sector_n_up=args.sector_n_up,
        sector_n_dn=args.sector_n_dn,
        ordering=args.ordering,
        symmetry_mode=str(args.symmetry_mode),  # type: ignore[arg-type]
        postprocess_correction_mode=str(args.correction_mode),  # type: ignore[arg-type]
    )
    request = DetectionReplayInput(
        artifact_json=Path(args.artifact_json),
        output_json=Path(args.output_json),
        noise_mode=str(args.noise_mode),  # type: ignore[arg-type]
        execution_surface=str(args.execution_surface),  # type: ignore[arg-type]
        backend_name=args.backend_name,
        use_fake_backend=bool(args.use_fake_backend),
        shots=int(args.shots),
        oracle_repeats=int(args.oracle_repeats),
        oracle_aggregate=str(args.oracle_aggregate),  # type: ignore[arg-type]
        seed=int(args.seed),
        seed_transpiler=args.seed_transpiler,
        transpile_optimization_level=int(args.transpile_optimization_level),
        raw_transport=str(args.raw_transport),
        raw_grouping_mode=str(args.raw_grouping_mode),  # type: ignore[arg-type]
        raw_store_memory=bool(args.raw_store_memory),
        runtime_profile=args.runtime_profile,
        runtime_session=args.runtime_session,
        raw_artifact_path=args.raw_artifact_path,
        detection=detection,
    )
    validate_correction_mode_off(str(detection.postprocess_correction_mode))
    validate_detect_only_request(request)
    return request


def _normalize_ordering(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    ordering = str(value).strip().lower()
    if ordering in {"blocked", "interleaved"}:
        return ordering
    return str(value)


def _state_num_qubits(state: Any) -> tuple[int | None, str | None]:
    size = int(np.asarray(state).size)
    if size <= 0:
        return None, "reference_state_empty"
    nq = int(round(float(np.log2(size))))
    if 2**nq != size:
        return None, "reference_state_size_not_power_of_two"
    return int(nq), None


def _as_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_artifact_sector(resolved_problem: Any) -> tuple[int, int] | None:
    for candidate in (
        getattr(getattr(resolved_problem, "sector", None), "num_particles", None),
        getattr(resolved_problem, "default_num_particles", None),
    ):
        if candidate is None:
            continue
        try:
            values = tuple(candidate)
        except TypeError:
            continue
        if len(values) == 2:
            return int(values[0]), int(values[1])
    return None


def resolve_detection_context(
    runtime_input: Any,
    detection: ErrorDetectionConfig,
) -> ResolvedDetectionContext:
    resolved_problem = getattr(runtime_input, "resolved_problem", None)
    request = getattr(resolved_problem, "request", None)
    layout = getattr(resolved_problem, "layout", None)

    problem = str(
        getattr(resolved_problem, "family_key", getattr(request, "problem_key", ""))
    ).strip().lower()
    num_sites = _as_int(getattr(request, "num_sites", None))
    layout_total_qubits = _as_int(getattr(layout, "total_qubits", None))
    fermion_qubits = _as_int(getattr(layout, "fermion_qubits", None))
    boson_qubits = _as_int(getattr(layout, "boson_qubits", None))
    state_qubits, state_error = _state_num_qubits(getattr(runtime_input, "psi_ref", ()))
    num_qubits = layout_total_qubits if layout_total_qubits is not None else state_qubits

    warnings: list[str] = []
    errors: list[str] = []
    if state_error is not None:
        errors.append(state_error)
    if layout_total_qubits is not None and state_qubits is not None and layout_total_qubits != state_qubits:
        errors.append("layout_total_qubits_conflicts_with_reference_state_size")
    if num_sites is None:
        errors.append("missing_num_sites")
    elif fermion_qubits is not None and fermion_qubits != 2 * int(num_sites):
        errors.append("fermion_qubits_conflicts_with_two_num_sites")
    if num_sites is not None and num_qubits is not None and 2 * int(num_sites) > int(num_qubits):
        errors.append("fermion_register_exceeds_total_qubits")

    artifact_ordering = _normalize_ordering(
        getattr(request, "ordering", None) or getattr(layout, "ordering", None)
    )
    cli_ordering = _normalize_ordering(detection.ordering)
    if artifact_ordering is not None and artifact_ordering not in {"blocked", "interleaved"}:
        errors.append("artifact_ordering_unsupported")
    if cli_ordering is not None and cli_ordering not in {"blocked", "interleaved"}:
        errors.append("cli_ordering_unsupported")
    if artifact_ordering is not None:
        ordering = artifact_ordering
        ordering_source = "artifact"
        if cli_ordering is not None and cli_ordering != artifact_ordering:
            errors.append("cli_ordering_conflicts_with_artifact_ordering")
    else:
        ordering = cli_ordering
        ordering_source = "cli" if cli_ordering is not None else "missing"
        if ordering is None:
            errors.append("missing_ordering")

    artifact_sector = _extract_artifact_sector(resolved_problem)
    cli_sector = None
    if detection.sector_n_up is not None or detection.sector_n_dn is not None:
        if detection.sector_n_up is None or detection.sector_n_dn is None:
            errors.append("partial_cli_sector")
        else:
            cli_sector = (int(detection.sector_n_up), int(detection.sector_n_dn))
    if artifact_sector is not None:
        sector = artifact_sector
        sector_source = "artifact"
        if cli_sector is not None and cli_sector != artifact_sector:
            errors.append("cli_sector_conflicts_with_artifact_sector")
    else:
        sector = cli_sector
        sector_source = "cli" if cli_sector is not None else "missing"
        if sector is None:
            errors.append("missing_target_sector")

    supported_family = problem == "hh"
    if not supported_family:
        errors.append("unsupported_family_for_hh_sector_decoder")

    return ResolvedDetectionContext(
        problem=problem,
        num_sites=num_sites,
        num_qubits=num_qubits,
        fermion_qubits=fermion_qubits,
        boson_qubits=boson_qubits,
        ordering=ordering,
        ordering_source=ordering_source,
        sector_n_up=None if sector is None else int(sector[0]),
        sector_n_dn=None if sector is None else int(sector[1]),
        sector_source=sector_source,
        supported_family=bool(supported_family),
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def _request_with_resolved_detection(
    request: DetectionReplayInput,
    context: ResolvedDetectionContext,
) -> DetectionReplayInput:
    return replace(
        request,
        detection=replace(
            request.detection,
            sector_n_up=context.sector_n_up,
            sector_n_dn=context.sector_n_dn,
            ordering=context.ordering,
        ),
    )


def _physics_point(runtime_input: Any, context: ResolvedDetectionContext | None = None) -> dict[str, Any]:
    request = getattr(getattr(runtime_input, "resolved_problem", None), "request", None)
    resolved_problem = getattr(runtime_input, "resolved_problem", None)
    if request is None:
        return {"problem": "" if context is None else context.problem}
    out: dict[str, Any] = {
        "problem": str(getattr(request, "problem_key", getattr(resolved_problem, "family_key", ""))),
        "num_sites": getattr(request, "num_sites", None),
        "L": getattr(request, "num_sites", None),
    }
    for key in ("t", "u", "dv", "omega0", "g_ep", "n_ph_max", "ordering", "boson_encoding"):
        if hasattr(request, key):
            out[key] = getattr(request, key)
    sector = getattr(resolved_problem, "sector", None)
    if sector is not None:
        out["sector_label"] = getattr(sector, "label", None)
        out["comparison_space_label"] = getattr(sector, "comparison_space_label", None)
    if context is not None:
        out.update(
            {
                "layout_total_qubits": context.num_qubits,
                "fermion_qubits": context.fermion_qubits,
                "boson_qubits": context.boson_qubits,
                "sector_n_up": context.sector_n_up,
                "sector_n_dn": context.sector_n_dn,
            }
        )
    return out


def _sector_context_failure_reason(context: ResolvedDetectionContext) -> str | None:
    if context.errors:
        return ",".join(context.errors)
    if context.num_sites is None:
        return "missing_num_sites"
    if context.ordering is None:
        return "missing_ordering"
    if context.sector_n_up is None or context.sector_n_dn is None:
        return "missing_target_sector"
    if context.num_qubits is None:
        return "missing_num_qubits"
    return None


def run_detect_only_replay(
    request: DetectionReplayInput,
    *,
    loader_mode: str | None = None,
) -> dict[str, Any]:
    validate_detect_only_request(request)
    runtime_input = load_replay_runtime_input(request.artifact_json, loader_mode=loader_mode)
    detection_context = resolve_detection_context(runtime_input, request.detection)
    resolved_request = _request_with_resolved_detection(request, detection_context)

    plan = build_parameterized_plan_from_runtime_input(runtime_input)
    theta_runtime = np.asarray(runtime_input.theta_runtime, dtype=float).reshape(-1)

    energy_bundle = None
    sector_audit_estimate = None
    sector_filter_estimate = None
    if str(resolved_request.detection.mode) in {"off", "sector_audit", "sector_filter"}:
        energy_bundle = acquire_raw_bundle(
            request=resolved_request,
            plan=plan,
            theta_runtime=theta_runtime,
            observable=build_energy_observable(runtime_input),
            observable_family="energy_raw",
        )
    if str(resolved_request.detection.mode) in {"sector_audit", "sector_filter"}:
        sector_failure = _sector_context_failure_reason(detection_context)
        if sector_failure is not None:
            sector_audit_estimate = sector_audit_unavailable(
                strict=bool(resolved_request.detection.strict),
                reason=sector_failure,
                gate=asdict(detection_context),
            )
        else:
            try:
                full_z_bundle = acquire_raw_bundle(
                    request=resolved_request,
                    plan=plan,
                    theta_runtime=theta_runtime,
                    observable=build_full_register_z_observable(
                        int(getattr(plan, "nq", detection_context.num_qubits))
                    ),
                    observable_family="sector_full_register_z",
                )
                sector_audit_estimate = summarize_sector_audit(
                    full_z_bundle.records,
                    num_sites=int(detection_context.num_sites),  # type: ignore[arg-type]
                    ordering=str(detection_context.ordering),
                    sector_n_up=int(detection_context.sector_n_up),  # type: ignore[arg-type]
                    sector_n_dn=int(detection_context.sector_n_dn),  # type: ignore[arg-type]
                    min_accepted_shots=int(resolved_request.detection.min_accepted_shots),
                    strict=bool(resolved_request.detection.strict),
                )
            except Exception as exc:  # pragma: no cover - exercised via orchestrator tests.
                sector_audit_estimate = sector_audit_unavailable(
                    strict=bool(resolved_request.detection.strict),
                    reason="sector_audit_acquisition_or_summary_failed",
                    warning=f"Sector audit failed: {exc}",
                    gate=asdict(detection_context),
                )
    if str(resolved_request.detection.mode) == "sector_filter":
        filter_context_failure = _sector_context_failure_reason(detection_context)
        if filter_context_failure is not None:
            gate = {
                "context_valid": False,
                "detector_observability_checked": False,
                "detector_observability_present": False,
                "filter_application_enabled": False,
                "reason": filter_context_failure,
            }
        else:
            gate = evaluate_sector_filter_observability(
                () if energy_bundle is None else energy_bundle.records,
                num_sites=int(detection_context.num_sites),  # type: ignore[arg-type]
            )
        sector_filter_estimate = sector_filter_unavailable(
            strict=bool(resolved_request.detection.strict),
            gate=gate,
        )

    oracle_cfg = build_raw_oracle_config(resolved_request)
    base_layout = runtime_input.base_layout
    summary = build_detect_only_sidecar_payload(
        request=resolved_request,
        physics_point=_physics_point(runtime_input, detection_context),
        ansatz_identity={
            "plan_digest": str(getattr(plan, "plan_digest", "")),
            "structure_digest": str(getattr(plan, "structure_digest", "")),
            "runtime_parameter_count": int(getattr(base_layout, "runtime_parameter_count", len(theta_runtime))),
            "logical_parameter_count": int(getattr(base_layout, "logical_parameter_count", 0)),
        },
        oracle_request=asdict(oracle_cfg),
        energy_bundle=energy_bundle,
        sector_audit_estimate=sector_audit_estimate,
        sector_filter_estimate=sector_filter_estimate,
        exact_energy=runtime_input.exact_energy,
        diagnostics={
            "loader_mode": loader_mode,
            "detection_context": asdict(detection_context),
        },
    )
    write_detect_only_sidecar_json(resolved_request.output_json, summary)
    return asdict(summary)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    request = resolve_detection_replay_input(args)
    run_detect_only_replay(request, loader_mode=args.loader_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
