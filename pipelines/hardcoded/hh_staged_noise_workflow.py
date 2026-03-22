#!/usr/bin/env python3
"""Noise-capable staged HH workflow built on the shared staged core."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from docs.reports.pdf_utils import (
    get_PdfPages,
    get_plt,
    render_command_page,
    render_compact_table,
    render_parameter_manifest,
    render_text_page,
    require_matplotlib,
)
from pipelines.exact_bench import hh_noise_robustness_seq_report as noise_report
from pipelines.exact_bench.noise_oracle_runtime import (
    BACKEND_SCHEDULED_ATTRIBUTION_SLICES,
    normalize_mitigation_config,
    normalize_symmetry_mitigation_config,
)
from pipelines.hardcoded import hh_staged_workflow as base_wf
from pipelines.hardcoded.imported_artifact_resolution import (
    resolve_default_lean_pareto_l2_artifact_json,
    resolve_imported_artifact_path,
)


_ALLOWED_NOISY_METHODS = {"suzuki2", "cfqm4", "cfqm6"}
_ALLOWED_NOISE_MODES = {"ideal", "shots", "aer_noise", "runtime"}
_ALLOWED_AUDIT_NOISE_MODES = {"ideal", "shots", "aer_noise", "runtime", "backend_scheduled"}


@dataclass(frozen=True)
class NoiseConfig:
    methods: tuple[str, ...]
    modes: tuple[str, ...]
    audit_modes: tuple[str, ...]
    shots: int
    oracle_repeats: int
    oracle_aggregate: str
    mitigation_config: dict[str, Any]
    symmetry_mitigation_config: dict[str, Any]
    local_readout_strategy: str | None
    seed: int
    backend_name: str | None
    use_fake_backend: bool
    allow_aer_fallback: bool
    omp_shm_workaround: bool
    noisy_mode_timeout_s: int
    benchmark_active_coeff_tol: float
    include_final_audit: bool
    include_full_circuit_audit: bool


@dataclass(frozen=True)
class FixedLeanNoisyReplayConfig:
    enabled: bool
    reps: int
    method: str
    seed: int
    maxiter: int
    wallclock_cap_s: int
    spsa_a: float
    spsa_c: float
    spsa_alpha: float
    spsa_gamma: float
    spsa_A: float
    spsa_avg_last: int
    spsa_eval_repeats: int
    spsa_eval_agg: str
    noise_mode: str
    mitigation_config: dict[str, Any]
    symmetry_mitigation_config: dict[str, Any]


@dataclass(frozen=True)
class FixedLeanNoiseAttributionConfig:
    enabled: bool
    slices: tuple[str, ...]
    noise_mode: str
    mitigation_config: dict[str, Any]
    symmetry_mitigation_config: dict[str, Any]


@dataclass(frozen=True)
class StagedNoiseSourceConfig:
    mode: str
    requested_json: Path | None
    resolved_json: Path | None
    source_kind: str | None
    default_subject: bool = False


@dataclass(frozen=True)
class StagedHHNoiseConfig:
    staged: base_wf.StagedHHConfig
    noise: NoiseConfig
    source: StagedNoiseSourceConfig
    fixed_lean_replay: FixedLeanNoisyReplayConfig
    fixed_lean_attribution: FixedLeanNoiseAttributionConfig


def _parse_csv(raw: str | Sequence[str], *, allowed: set[str], label: str) -> tuple[str, ...]:
    if isinstance(raw, str):
        items = [x.strip().lower() for x in raw.split(",") if x.strip()]
    else:
        items = [str(x).strip().lower() for x in raw if str(x).strip()]
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in allowed:
            raise ValueError(f"Unsupported {label} '{item}'. Allowed: {sorted(allowed)}")
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    if not out:
        raise ValueError(f"Expected at least one value in {label} list.")
    return tuple(out)


def _retag_staged_artifacts(
    staged_cfg: base_wf.StagedHHConfig,
    *,
    tag: str,
    output_json_override: Path | None,
    output_pdf_override: Path | None,
) -> base_wf.StagedHHConfig:
    art = staged_cfg.artifacts
    new_art = replace(
        art,
        tag=str(tag),
        output_json=(Path(output_json_override) if output_json_override is not None else art.output_json.with_name(f"{tag}.json")),
        output_pdf=(Path(output_pdf_override) if output_pdf_override is not None else art.output_pdf.with_name(f"{tag}.pdf")),
        handoff_json=art.handoff_json.with_name(f"{tag}_adapt_handoff.json"),
        replay_output_json=art.replay_output_json.with_name(f"{tag}_replay.json"),
        replay_output_csv=art.replay_output_csv.with_name(f"{tag}_replay.csv"),
        replay_output_md=art.replay_output_md.with_name(f"{tag}_replay.md"),
        replay_output_log=art.replay_output_log.with_name(f"{tag}_replay.log"),
    )
    return replace(staged_cfg, artifacts=new_art)


def _default_full_circuit_import_json() -> tuple[Path | None, bool]:
    return resolve_default_lean_pareto_l2_artifact_json()


def _resolve_import_source(
    *,
    requested_json: Path | None,
    require_default_import_source: bool,
) -> StagedNoiseSourceConfig:
    requested = Path(requested_json) if requested_json is not None else None
    default_subject = False
    if requested is None and bool(require_default_import_source):
        requested, default_subject = _default_full_circuit_import_json()
    resolution = resolve_imported_artifact_path(
        requested_json=requested,
        require_default_import_source=False,
    )
    return StagedNoiseSourceConfig(
        mode=str(resolution.mode),
        requested_json=resolution.requested_json,
        resolved_json=resolution.resolved_json,
        source_kind=resolution.source_kind,
        default_subject=bool(default_subject or resolution.default_subject),
    )


def resolve_staged_hh_noise_config(args: Any) -> StagedHHNoiseConfig:
    staged_cfg = base_wf.resolve_staged_hh_config(args)
    if getattr(args, "tag", None) is None:
        base_tag = str(staged_cfg.artifacts.tag)
        noise_tag = (
            base_tag.replace("hh_staged_", "hh_staged_noise_", 1)
            if base_tag.startswith("hh_staged_")
            else f"{base_tag}_noise"
        )
        staged_cfg = _retag_staged_artifacts(
            staged_cfg,
            tag=noise_tag,
            output_json_override=getattr(args, "output_json", None),
            output_pdf_override=getattr(args, "output_pdf", None),
        )

    methods = _parse_csv(getattr(args, "noisy_methods"), allowed=_ALLOWED_NOISY_METHODS, label="noisy method")
    modes = _parse_csv(getattr(args, "noise_modes"), allowed=_ALLOWED_NOISE_MODES, label="noise mode")
    include_full_circuit_audit = bool(getattr(args, "include_full_circuit_audit", False))
    include_fixed_lean_noisy_replay = bool(getattr(args, "include_fixed_lean_noisy_replay", False))
    include_fixed_lean_noise_attribution = bool(
        getattr(args, "include_fixed_lean_noise_attribution", False)
    )
    source_cfg = _resolve_import_source(
        requested_json=getattr(args, "fixed_final_state_json", None),
        require_default_import_source=bool(
            include_full_circuit_audit
            or include_fixed_lean_noisy_replay
            or include_fixed_lean_noise_attribution
        ),
    )
    raw_audit_modes = getattr(args, "audit_noise_modes", None)
    if raw_audit_modes in {None, ""}:
        if str(source_cfg.mode) == "imported_artifact":
            audit_modes = ("ideal", "backend_scheduled")
        else:
            audit_modes = tuple(modes)
    else:
        audit_modes = _parse_csv(raw_audit_modes, allowed=_ALLOWED_AUDIT_NOISE_MODES, label="audit noise mode")
    backend_name_raw = getattr(args, "backend_name", None)
    backend_name = None if backend_name_raw in {None, "", "none"} else str(backend_name_raw)
    use_fake_backend = bool(getattr(args, "use_fake_backend"))
    if str(source_cfg.mode) == "imported_artifact" and (
        "backend_scheduled" in audit_modes
        or bool(include_fixed_lean_noisy_replay)
        or bool(include_fixed_lean_noise_attribution)
    ):
        if backend_name is None:
            backend_name = "FakeGuadalupeV2"
        use_fake_backend = True
    noise_cfg = NoiseConfig(
        methods=methods,
        modes=modes,
        audit_modes=tuple(audit_modes),
        shots=int(getattr(args, "shots")),
        oracle_repeats=int(getattr(args, "oracle_repeats")),
        oracle_aggregate=str(getattr(args, "oracle_aggregate")),
        mitigation_config=normalize_mitigation_config(
            {
                "mode": str(getattr(args, "mitigation")),
                "zne_scales": getattr(args, "zne_scales", None),
                "dd_sequence": getattr(args, "dd_sequence", None),
                "local_readout_strategy": getattr(args, "local_readout_strategy", None),
            }
        ),
        symmetry_mitigation_config=normalize_symmetry_mitigation_config(
            {
                "mode": str(getattr(args, "symmetry_mitigation_mode")),
                "num_sites": int(staged_cfg.physics.L),
                "ordering": str(staged_cfg.physics.ordering),
                "sector_n_up": int(staged_cfg.physics.sector_n_up),
                "sector_n_dn": int(staged_cfg.physics.sector_n_dn),
            }
        ),
        local_readout_strategy=(None if getattr(args, "local_readout_strategy", None) in {None, "", "none"} else str(getattr(args, "local_readout_strategy"))),
        seed=int(getattr(args, "noise_seed")),
        backend_name=backend_name,
        use_fake_backend=bool(use_fake_backend),
        allow_aer_fallback=bool(getattr(args, "allow_aer_fallback")),
        omp_shm_workaround=bool(getattr(args, "omp_shm_workaround")),
        noisy_mode_timeout_s=int(getattr(args, "noisy_mode_timeout_s")),
        benchmark_active_coeff_tol=float(getattr(args, "benchmark_active_coeff_tol")),
        include_final_audit=bool(getattr(args, "include_final_audit")),
        include_full_circuit_audit=bool(include_full_circuit_audit),
    )
    fixed_lean_replay_enabled = bool(include_fixed_lean_noisy_replay)
    if fixed_lean_replay_enabled:
        if str(source_cfg.mode) != "imported_artifact":
            raise ValueError(
                "fixed lean noisy replay requires an imported artifact source; pass "
                "--fixed-final-state-json or enable the default imported lean subject."
            )
        fixed_lean_method = str(staged_cfg.replay.method)
        if fixed_lean_method.strip().upper() not in {"SPSA", "POWELL"}:
            raise ValueError(
                "fixed lean noisy replay currently supports only final replay optimizer SPSA or Powell."
            )
    else:
        fixed_lean_method = str(staged_cfg.replay.method)
    fixed_lean_replay_cfg = FixedLeanNoisyReplayConfig(
        enabled=bool(fixed_lean_replay_enabled),
        reps=1,
        method=str(fixed_lean_method),
        seed=int(staged_cfg.replay.seed),
        maxiter=int(staged_cfg.replay.maxiter),
        wallclock_cap_s=int(staged_cfg.replay.wallclock_cap_s),
        spsa_a=float(staged_cfg.replay.spsa_a),
        spsa_c=float(staged_cfg.replay.spsa_c),
        spsa_alpha=float(staged_cfg.replay.spsa_alpha),
        spsa_gamma=float(staged_cfg.replay.spsa_gamma),
        spsa_A=float(staged_cfg.replay.spsa_A),
        spsa_avg_last=int(staged_cfg.replay.spsa_avg_last),
        spsa_eval_repeats=int(staged_cfg.replay.spsa_eval_repeats),
        spsa_eval_agg=str(staged_cfg.replay.spsa_eval_agg),
        noise_mode="backend_scheduled",
        mitigation_config=normalize_mitigation_config(
            {"mode": "readout", "local_readout_strategy": "mthree"}
        ),
        symmetry_mitigation_config=normalize_symmetry_mitigation_config(
            {"mode": "off"}
        ),
    )
    fixed_lean_attribution_enabled = bool(include_fixed_lean_noise_attribution)
    if fixed_lean_attribution_enabled and str(source_cfg.mode) != "imported_artifact":
        raise ValueError(
            "fixed lean noise attribution requires an imported artifact source; pass "
            "--fixed-final-state-json or enable the default imported lean subject."
        )
    fixed_lean_attribution_cfg = FixedLeanNoiseAttributionConfig(
        enabled=bool(fixed_lean_attribution_enabled),
        slices=tuple(BACKEND_SCHEDULED_ATTRIBUTION_SLICES),
        noise_mode="backend_scheduled",
        mitigation_config=normalize_mitigation_config({"mode": "none"}),
        symmetry_mitigation_config=normalize_symmetry_mitigation_config({"mode": "off"}),
    )
    return StagedHHNoiseConfig(
        staged=staged_cfg,
        noise=noise_cfg,
        source=source_cfg,
        fixed_lean_replay=fixed_lean_replay_cfg,
        fixed_lean_attribution=fixed_lean_attribution_cfg,
    )


def _drive_profile_from_staged_cfg(staged_cfg: base_wf.StagedHHConfig) -> dict[str, Any] | None:
    if not bool(staged_cfg.dynamics.enable_drive):
        return None
    return {
        "A": float(staged_cfg.dynamics.drive_A),
        "omega": float(staged_cfg.dynamics.drive_omega),
        "tbar": float(staged_cfg.dynamics.drive_tbar),
        "phi": float(staged_cfg.dynamics.drive_phi),
        "pattern": str(staged_cfg.dynamics.drive_pattern),
        "custom_weights": base_wf._parse_drive_custom_weights(staged_cfg.dynamics.drive_custom_s),
        "include_identity": bool(staged_cfg.dynamics.drive_include_identity),
        "time_sampling": str(staged_cfg.dynamics.drive_time_sampling),
        "t0": float(staged_cfg.dynamics.drive_t0),
    }


def _run_single_noisy_mode(
    *,
    stage_result: base_wf.StageExecutionResult,
    staged_cfg: base_wf.StagedHHConfig,
    noise_cfg: NoiseConfig,
    drive_profile: dict[str, Any] | None,
    method: str,
    mode: str,
) -> dict[str, Any]:
    kwargs = {
        "L": int(staged_cfg.physics.L),
        "ordering": str(staged_cfg.physics.ordering),
        "psi_seed": np.asarray(stage_result.psi_final, dtype=complex).reshape(-1),
        "ordered_labels_exyz": list(stage_result.ordered_labels_exyz),
        "static_coeff_map_exyz": dict(stage_result.coeff_map_exyz),
        "t_final": float(staged_cfg.dynamics.t_final),
        "num_times": int(staged_cfg.dynamics.num_times),
        "trotter_steps": int(staged_cfg.dynamics.trotter_steps),
        "drive_profile": drive_profile,
        "noise_mode": str(mode),
        "shots": int(noise_cfg.shots),
        "seed": int(noise_cfg.seed),
        "oracle_repeats": int(noise_cfg.oracle_repeats),
        "oracle_aggregate": str(noise_cfg.oracle_aggregate),
        "mitigation_config": dict(noise_cfg.mitigation_config),
        "symmetry_mitigation_config": dict(noise_cfg.symmetry_mitigation_config),
        "backend_name": noise_cfg.backend_name,
        "use_fake_backend": bool(noise_cfg.use_fake_backend),
        "allow_aer_fallback": bool(noise_cfg.allow_aer_fallback),
        "omp_shm_workaround": bool(noise_cfg.omp_shm_workaround),
        "method": str(method),
        "benchmark_active_coeff_tol": float(noise_cfg.benchmark_active_coeff_tol),
        "cfqm_coeff_drop_abs_tol": float(staged_cfg.dynamics.cfqm_coeff_drop_abs_tol),
    }
    return noise_report._run_noisy_mode_isolated(
        kwargs=kwargs,
        timeout_s=int(noise_cfg.noisy_mode_timeout_s),
    )


def _run_final_audit_mode(
    *,
    stage_result: base_wf.StageExecutionResult,
    staged_cfg: base_wf.StagedHHConfig,
    noise_cfg: NoiseConfig,
    drive_profile: dict[str, Any] | None,
    mode: str,
) -> dict[str, Any]:
    kwargs = {
        "L": int(staged_cfg.physics.L),
        "ordering": str(staged_cfg.physics.ordering),
        "psi_seed": np.asarray(stage_result.psi_final, dtype=complex).reshape(-1),
        "ordered_labels_exyz": list(stage_result.ordered_labels_exyz),
        "static_coeff_map_exyz": dict(stage_result.coeff_map_exyz),
        "drive_profile": drive_profile,
        "noise_mode": str(mode),
        "shots": int(noise_cfg.shots),
        "seed": int(noise_cfg.seed),
        "oracle_repeats": int(noise_cfg.oracle_repeats),
        "oracle_aggregate": str(noise_cfg.oracle_aggregate),
        "mitigation_config": dict(noise_cfg.mitigation_config),
        "symmetry_mitigation_config": dict(noise_cfg.symmetry_mitigation_config),
        "backend_name": noise_cfg.backend_name,
        "use_fake_backend": bool(noise_cfg.use_fake_backend),
        "allow_aer_fallback": bool(noise_cfg.allow_aer_fallback),
        "omp_shm_workaround": bool(noise_cfg.omp_shm_workaround),
    }
    return noise_report._run_noisy_audit_mode_isolated(
        kwargs=kwargs,
        timeout_s=int(noise_cfg.noisy_mode_timeout_s),
    )


def run_noisy_profiles(
    stage_result: base_wf.StageExecutionResult,
    cfg: StagedHHNoiseConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    staged_cfg = cfg.staged
    noise_cfg = cfg.noise
    drive_profile = _drive_profile_from_staged_cfg(staged_cfg)
    profile_specs: list[tuple[str, dict[str, Any] | None]] = [("static", None)]
    if drive_profile is not None:
        profile_specs.append(("drive", drive_profile))

    noisy_profiles: dict[str, Any] = {}
    audit_profiles: dict[str, Any] = {}
    for profile_name, profile_drive in profile_specs:
        methods_payload: dict[str, Any] = {}
        for method in noise_cfg.methods:
            modes_payload: dict[str, Any] = {}
            for mode in noise_cfg.modes:
                modes_payload[str(mode)] = _run_single_noisy_mode(
                    stage_result=stage_result,
                    staged_cfg=staged_cfg,
                    noise_cfg=noise_cfg,
                    drive_profile=profile_drive,
                    method=str(method),
                    mode=str(mode),
                )
            methods_payload[str(method)] = {"modes": modes_payload}
        noisy_profiles[str(profile_name)] = {
            "drive_enabled": bool(profile_drive is not None),
            "drive_profile": profile_drive,
            "methods": methods_payload,
        }
        if bool(noise_cfg.include_final_audit):
            audit_modes = {
                str(mode): _run_final_audit_mode(
                    stage_result=stage_result,
                    staged_cfg=staged_cfg,
                    noise_cfg=noise_cfg,
                    drive_profile=profile_drive,
                    mode=str(mode),
                )
                for mode in noise_cfg.modes
            }
            audit_profiles[str(profile_name)] = {
                "drive_enabled": bool(profile_drive is not None),
                "drive_profile": profile_drive,
                "modes": audit_modes,
            }

    dynamics_noisy = {"profiles": noisy_profiles}
    noisy_final_audit = {"profiles": audit_profiles}
    dynamics_benchmarks = {"rows": noise_report._collect_noisy_benchmark_rows(dynamics_noisy)}
    return dynamics_noisy, noisy_final_audit, dynamics_benchmarks


def _run_imported_prepared_state_audit_mode(
    *,
    source_cfg: StagedNoiseSourceConfig,
    noise_cfg: NoiseConfig,
    mode: str,
) -> dict[str, Any]:
    if source_cfg.resolved_json is None:
        return {"success": False, "available": False, "reason": "missing_import_source"}
    kwargs = {
        "artifact_json": str(source_cfg.resolved_json),
        "noise_mode": str(mode),
        "shots": int(noise_cfg.shots),
        "seed": int(noise_cfg.seed),
        "oracle_repeats": int(noise_cfg.oracle_repeats),
        "oracle_aggregate": str(noise_cfg.oracle_aggregate),
        "mitigation_config": dict(noise_cfg.mitigation_config),
        "symmetry_mitigation_config": dict(noise_cfg.symmetry_mitigation_config),
        "backend_name": noise_cfg.backend_name,
        "use_fake_backend": bool(noise_cfg.use_fake_backend),
        "allow_aer_fallback": bool(noise_cfg.allow_aer_fallback),
        "omp_shm_workaround": bool(noise_cfg.omp_shm_workaround),
    }
    return noise_report._run_imported_prepared_state_audit_mode_isolated(
        kwargs=kwargs,
        timeout_s=int(noise_cfg.noisy_mode_timeout_s),
    )


def _run_imported_full_circuit_audit_mode(
    *,
    source_cfg: StagedNoiseSourceConfig,
    noise_cfg: NoiseConfig,
    mode: str,
) -> dict[str, Any]:
    if source_cfg.resolved_json is None:
        return {"success": False, "available": False, "reason": "missing_import_source"}
    kwargs = {
        "artifact_json": str(source_cfg.resolved_json),
        "noise_mode": str(mode),
        "shots": int(noise_cfg.shots),
        "seed": int(noise_cfg.seed),
        "oracle_repeats": int(noise_cfg.oracle_repeats),
        "oracle_aggregate": str(noise_cfg.oracle_aggregate),
        "mitigation_config": dict(noise_cfg.mitigation_config),
        "symmetry_mitigation_config": dict(noise_cfg.symmetry_mitigation_config),
        "backend_name": noise_cfg.backend_name,
        "use_fake_backend": bool(noise_cfg.use_fake_backend),
        "allow_aer_fallback": bool(noise_cfg.allow_aer_fallback),
        "omp_shm_workaround": bool(noise_cfg.omp_shm_workaround),
    }
    return noise_report._run_imported_full_circuit_audit_mode_isolated(
        kwargs=kwargs,
        timeout_s=int(noise_cfg.noisy_mode_timeout_s),
    )


def _run_fixed_lean_noisy_replay_mode(
    *,
    source_cfg: StagedNoiseSourceConfig,
    noise_cfg: NoiseConfig,
    fixed_cfg: FixedLeanNoisyReplayConfig,
) -> dict[str, Any]:
    if source_cfg.resolved_json is None:
        return {"success": False, "available": False, "reason": "missing_import_source"}
    if not bool(fixed_cfg.enabled):
        return {"success": False, "available": False, "reason": "fixed_lean_noisy_replay_disabled"}
    kwargs = {
        "artifact_json": str(source_cfg.resolved_json),
        "shots": int(noise_cfg.shots),
        "seed": int(noise_cfg.seed),
        "oracle_repeats": int(noise_cfg.oracle_repeats),
        "oracle_aggregate": str(noise_cfg.oracle_aggregate),
        "backend_name": noise_cfg.backend_name,
        "use_fake_backend": bool(noise_cfg.use_fake_backend),
        "allow_aer_fallback": bool(noise_cfg.allow_aer_fallback),
        "omp_shm_workaround": bool(noise_cfg.omp_shm_workaround),
        "optimizer_method": str(fixed_cfg.method),
        "optimizer_seed": int(fixed_cfg.seed),
        "optimizer_maxiter": int(fixed_cfg.maxiter),
        "optimizer_wallclock_cap_s": int(fixed_cfg.wallclock_cap_s),
        "spsa_a": float(fixed_cfg.spsa_a),
        "spsa_c": float(fixed_cfg.spsa_c),
        "spsa_alpha": float(fixed_cfg.spsa_alpha),
        "spsa_gamma": float(fixed_cfg.spsa_gamma),
        "spsa_A": float(fixed_cfg.spsa_A),
        "spsa_avg_last": int(fixed_cfg.spsa_avg_last),
        "spsa_eval_repeats": int(fixed_cfg.spsa_eval_repeats),
        "spsa_eval_agg": str(fixed_cfg.spsa_eval_agg),
        "mitigation_config": dict(fixed_cfg.mitigation_config),
        "symmetry_mitigation_config": dict(fixed_cfg.symmetry_mitigation_config),
    }
    return noise_report._run_imported_fixed_lean_noisy_replay_mode_isolated(
        kwargs=kwargs,
        timeout_s=int(noise_cfg.noisy_mode_timeout_s),
    )


def _run_fixed_lean_noise_attribution_mode(
    *,
    source_cfg: StagedNoiseSourceConfig,
    noise_cfg: NoiseConfig,
    attribution_cfg: FixedLeanNoiseAttributionConfig,
) -> dict[str, Any]:
    if source_cfg.resolved_json is None:
        return {"success": False, "available": False, "reason": "missing_import_source"}
    if not bool(attribution_cfg.enabled):
        return {"success": False, "available": False, "reason": "fixed_lean_noise_attribution_disabled"}
    kwargs = {
        "artifact_json": str(source_cfg.resolved_json),
        "shots": int(noise_cfg.shots),
        "seed": int(noise_cfg.seed),
        "oracle_repeats": int(noise_cfg.oracle_repeats),
        "oracle_aggregate": str(noise_cfg.oracle_aggregate),
        "backend_name": noise_cfg.backend_name,
        "use_fake_backend": bool(noise_cfg.use_fake_backend),
        "allow_aer_fallback": bool(noise_cfg.allow_aer_fallback),
        "omp_shm_workaround": bool(noise_cfg.omp_shm_workaround),
        "mitigation_config": dict(attribution_cfg.mitigation_config),
        "symmetry_mitigation_config": dict(attribution_cfg.symmetry_mitigation_config),
        "slices": list(attribution_cfg.slices),
    }
    return noise_report._run_imported_fixed_lean_noise_attribution_mode_isolated(
        kwargs=kwargs,
        timeout_s=int(noise_cfg.noisy_mode_timeout_s),
    )


def _build_noise_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    stage_pipeline = payload.get("stage_pipeline", {}) if isinstance(payload, Mapping) else {}
    warm = stage_pipeline.get("warm_start", {}) if isinstance(stage_pipeline, Mapping) else {}
    adapt = stage_pipeline.get("adapt_vqe", {}) if isinstance(stage_pipeline, Mapping) else {}
    final = stage_pipeline.get("conventional_replay", {}) if isinstance(stage_pipeline, Mapping) else {}
    noisy_profiles = payload.get("dynamics_noisy", {}).get("profiles", {}) if isinstance(payload, Mapping) else {}
    completed = 0
    total = 0
    failures: list[str] = []
    for profile_name, profile_payload in noisy_profiles.items():
        methods = profile_payload.get("methods", {}) if isinstance(profile_payload, Mapping) else {}
        for method_name, method_payload in methods.items():
            modes = method_payload.get("modes", {}) if isinstance(method_payload, Mapping) else {}
            for mode_name, mode_payload in modes.items():
                total += 1
                if bool(isinstance(mode_payload, Mapping) and mode_payload.get("success", False)):
                    completed += 1
                else:
                    reason = "unknown"
                    if isinstance(mode_payload, Mapping):
                        reason = str(mode_payload.get("reason", mode_payload.get("error", "unknown")))
                    failures.append(f"{profile_name}:{method_name}:{mode_name}:{reason}")
    imported_prepared = payload.get("imported_prepared_state_audit", {}) if isinstance(payload, Mapping) else {}
    imported_full = payload.get("full_circuit_import_audit", {}) if isinstance(payload, Mapping) else {}
    fixed_lean_replay = payload.get("fixed_lean_scaffold_noisy_replay", {}) if isinstance(payload, Mapping) else {}
    fixed_lean_attribution = payload.get("fixed_lean_noise_attribution", {}) if isinstance(payload, Mapping) else {}
    imported_prepared_modes = imported_prepared.get("modes", {}) if isinstance(imported_prepared, Mapping) else {}
    imported_full_modes = imported_full.get("modes", {}) if isinstance(imported_full, Mapping) else {}
    imported_prepared_completed = sum(
        1 for rec in imported_prepared_modes.values() if isinstance(rec, Mapping) and bool(rec.get("success", False))
    )
    imported_full_completed = sum(
        1 for rec in imported_full_modes.values() if isinstance(rec, Mapping) and bool(rec.get("success", False))
    )
    fixed_lean_success = bool(isinstance(fixed_lean_replay, Mapping) and fixed_lean_replay.get("success", False))
    if isinstance(fixed_lean_replay, Mapping) and not fixed_lean_success and bool(fixed_lean_replay):
        failures.append(
            "fixed_lean_scaffold_noisy_replay:"
            + str(fixed_lean_replay.get("reason", fixed_lean_replay.get("error", "unknown")))
        )
    fixed_lean_attribution_success = bool(
        isinstance(fixed_lean_attribution, Mapping) and fixed_lean_attribution.get("success", False)
    )
    if isinstance(fixed_lean_attribution, Mapping) and not fixed_lean_attribution_success and bool(fixed_lean_attribution):
        failures.append(
            "fixed_lean_noise_attribution:"
            + str(fixed_lean_attribution.get("reason", fixed_lean_attribution.get("error", "unknown")))
        )
    attribution_slices = fixed_lean_attribution.get("slices", {}) if isinstance(fixed_lean_attribution, Mapping) else {}
    attribution_slices_completed = sum(
        1 for rec in attribution_slices.values() if isinstance(rec, Mapping) and bool(rec.get("success", False))
    )
    for slice_name, rec in attribution_slices.items() if isinstance(attribution_slices, Mapping) else []:
        if isinstance(rec, Mapping) and not bool(rec.get("success", False)):
            failures.append(
                "fixed_lean_noise_attribution:"
                + str(slice_name)
                + ":"
                + str(rec.get("reason", rec.get("error", "unknown")))
            )
    best_noisy_minus_ideal = float("nan")
    if isinstance(fixed_lean_replay, Mapping):
        energies = fixed_lean_replay.get("energies", {})
        if isinstance(energies, Mapping):
            best_noisy_minus_ideal = float(energies.get("best_noisy_minus_ideal", float("nan")))
    attribution_full_delta = float("nan")
    attribution_readout_delta = float("nan")
    attribution_gate_delta = float("nan")
    if isinstance(attribution_slices, Mapping):
        full_rec = attribution_slices.get("full", {})
        readout_rec = attribution_slices.get("readout_only", {})
        gate_rec = attribution_slices.get("gate_stateprep_only", {})
        if isinstance(full_rec, Mapping):
            attribution_full_delta = float(full_rec.get("delta_mean", float("nan")))
        if isinstance(readout_rec, Mapping):
            attribution_readout_delta = float(readout_rec.get("delta_mean", float("nan")))
        if isinstance(gate_rec, Mapping):
            attribution_gate_delta = float(gate_rec.get("delta_mean", float("nan")))
    return {
        "warm_delta_abs": float(warm.get("delta_abs", float("nan"))),
        "adapt_delta_abs": float(adapt.get("delta_abs", float("nan"))),
        "final_delta_abs": float(final.get("delta_abs", float("nan"))),
        "noisy_method_modes_completed": int(completed),
        "noisy_method_modes_total": int(total),
        "imported_prepared_state_audit_completed": int(imported_prepared_completed),
        "imported_prepared_state_audit_total": int(len(imported_prepared_modes)),
        "full_circuit_import_audit_completed": int(imported_full_completed),
        "full_circuit_import_audit_total": int(len(imported_full_modes)),
        "fixed_lean_scaffold_noisy_replay_completed": int(1 if fixed_lean_success else 0),
        "fixed_lean_scaffold_noisy_replay_total": int(1 if bool(fixed_lean_replay) else 0),
        "fixed_lean_scaffold_best_noisy_minus_ideal": float(best_noisy_minus_ideal),
        "fixed_lean_noise_attribution_completed": int(1 if fixed_lean_attribution_success else 0),
        "fixed_lean_noise_attribution_total": int(1 if bool(fixed_lean_attribution) else 0),
        "fixed_lean_noise_attribution_slices_completed": int(attribution_slices_completed),
        "fixed_lean_noise_attribution_slices_total": int(
            len(attribution_slices) if isinstance(attribution_slices, Mapping) else 0
        ),
        "fixed_lean_noise_attribution_full_minus_ideal": float(attribution_full_delta),
        "fixed_lean_noise_attribution_readout_only_minus_ideal": float(attribution_readout_delta),
        "fixed_lean_noise_attribution_gate_stateprep_only_minus_ideal": float(attribution_gate_delta),
        "failure_samples": failures[:10],
    }


def write_staged_hh_noise_pdf(payload: Mapping[str, Any], cfg: StagedHHNoiseConfig, run_command: str) -> None:
    staged_cfg = cfg.staged
    if bool(staged_cfg.artifacts.skip_pdf):
        return
    require_matplotlib()
    pdf_path = Path(staged_cfg.artifacts.output_pdf)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    PdfPages = get_PdfPages()
    plt = get_plt()

    stage_pipeline = payload.get("stage_pipeline", {}) if isinstance(payload, Mapping) else {}
    summary = payload.get("summary", {}) if isinstance(payload, Mapping) else {}
    noise_settings = payload.get("settings", {}).get("noise", {}) if isinstance(payload.get("settings", {}), Mapping) else {}
    import_source = payload.get("import_source", {}) if isinstance(payload, Mapping) else {}
    imported_full = payload.get("full_circuit_import_audit", {}) if isinstance(payload, Mapping) else {}
    imported_prepared = payload.get("imported_prepared_state_audit", {}) if isinstance(payload, Mapping) else {}
    fixed_lean_replay = payload.get("fixed_lean_scaffold_noisy_replay", {}) if isinstance(payload, Mapping) else {}
    fixed_lean_attribution = payload.get("fixed_lean_noise_attribution", {}) if isinstance(payload, Mapping) else {}
    import_mode = bool(isinstance(import_source, Mapping) and str(import_source.get("mode", "")) == "imported_artifact")

    with PdfPages(pdf_path) as pdf:
        render_parameter_manifest(
            pdf,
            model="Hubbard-Holstein",
            ansatz=(
                "imported lean ADAPT circuit audit"
                if import_mode
                else "warm: hh_hva_ptw; ADAPT: staged HH; final: matched-family replay; noisy final-only dynamics"
            ),
            drive_enabled=bool(staged_cfg.dynamics.enable_drive),
            t=float(staged_cfg.physics.t),
            U=float(staged_cfg.physics.u),
            dv=float(staged_cfg.physics.dv),
            extra={
                "L": int(staged_cfg.physics.L),
                "omega0": float(staged_cfg.physics.omega0),
                "g_ep": float(staged_cfg.physics.g_ep),
                "n_ph_max": int(staged_cfg.physics.n_ph_max),
                "noisy_methods": ",".join(cfg.noise.methods),
                "noise_modes": ",".join(cfg.noise.modes),
                "audit_noise_modes": ",".join(cfg.noise.audit_modes),
                "shots": int(cfg.noise.shots),
                "mitigation": dict(cfg.noise.mitigation_config),
                "symmetry_mitigation": dict(cfg.noise.symmetry_mitigation_config),
                "import_source": dict(import_source) if isinstance(import_source, Mapping) else {},
            },
            command=str(run_command),
        )
        render_text_page(
            pdf,
            [
                "HH staged noise workflow summary",
                "",
                f"Warm |ΔE|: {summary.get('warm_delta_abs')}",
                f"ADAPT |ΔE|: {summary.get('adapt_delta_abs')}",
                f"Replay |ΔE|: {summary.get('final_delta_abs')}",
                f"Noisy method/mode completion: {summary.get('noisy_method_modes_completed')} / {summary.get('noisy_method_modes_total')}",
                f"Imported prepared-state audit: {summary.get('imported_prepared_state_audit_completed')} / {summary.get('imported_prepared_state_audit_total')}",
                f"Imported full-circuit audit: {summary.get('full_circuit_import_audit_completed')} / {summary.get('full_circuit_import_audit_total')}",
                f"Fixed lean noisy replay: {summary.get('fixed_lean_scaffold_noisy_replay_completed')} / {summary.get('fixed_lean_scaffold_noisy_replay_total')}",
                f"Fixed lean best Δ(noisy-ideal): {summary.get('fixed_lean_scaffold_best_noisy_minus_ideal')}",
                f"Fixed lean attribution: {summary.get('fixed_lean_noise_attribution_completed')} / {summary.get('fixed_lean_noise_attribution_total')}",
                f"Fixed lean attribution slices: {summary.get('fixed_lean_noise_attribution_slices_completed')} / {summary.get('fixed_lean_noise_attribution_slices_total')}",
                f"Attribution full Δ(noisy-ideal): {summary.get('fixed_lean_noise_attribution_full_minus_ideal')}",
                f"Attribution readout-only Δ(noisy-ideal): {summary.get('fixed_lean_noise_attribution_readout_only_minus_ideal')}",
                f"Attribution gate-only Δ(noisy-ideal): {summary.get('fixed_lean_noise_attribution_gate_stateprep_only_minus_ideal')}",
                f"Failure samples: {summary.get('failure_samples', [])}",
                "",
                f"workflow_json: {staged_cfg.artifacts.output_json}",
                f"import_source_json: {payload.get('artifacts', {}).get('import_source_json')}",
                f"adapt_handoff_json: {staged_cfg.artifacts.handoff_json}",
                f"replay_json: {staged_cfg.artifacts.replay_output_json}",
                f"mitigation: {noise_settings.get('mitigation_config')}",
                f"symmetry: {noise_settings.get('symmetry_mitigation_config')}",
            ],
            fontsize=10,
            line_spacing=0.03,
        )

        fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.5))
        warm = stage_pipeline.get("warm_start", {}) if isinstance(stage_pipeline, Mapping) else {}
        adapt = stage_pipeline.get("adapt_vqe", {}) if isinstance(stage_pipeline, Mapping) else {}
        final = stage_pipeline.get("conventional_replay", {}) if isinstance(stage_pipeline, Mapping) else {}
        render_compact_table(
            axes[0],
            title="Stage metrics",
            col_labels=["Stage", "|ΔE|", "Gate/stop"],
            rows=[
                ["Warm", f"{float(warm.get('delta_abs', float('nan'))):.3e}", str(warm.get("ecut_1", {}))],
                ["ADAPT", f"{float(adapt.get('delta_abs', float('nan'))):.3e}", str(adapt.get("stop_reason", ""))],
                ["Replay", f"{float(final.get('delta_abs', float('nan'))):.3e}", str(final.get("ecut_2", {}))],
            ],
            fontsize=8,
        )
        noisy_rows: list[list[str]] = []
        for profile_name, methods in payload.get("comparisons", {}).get("noise_vs_noiseless_methods", {}).items():
            if not isinstance(methods, Mapping):
                continue
            for method_name, modes in methods.items():
                if not isinstance(modes, Mapping):
                    continue
                for mode_name, rec in modes.items():
                    if not isinstance(rec, Mapping):
                        continue
                    noisy_rows.append([
                        str(profile_name),
                        str(method_name),
                        str(mode_name),
                        str(rec.get("available", False)),
                        f"{float(rec.get('final_energy_total_delta_noisy_minus_ideal', float('nan'))):.3e}",
                    ])
        if not noisy_rows:
            for mode_name, rec in imported_full.get("modes", {}).items() if isinstance(imported_full, Mapping) else []:
                if not isinstance(rec, Mapping):
                    continue
                obs = rec.get("final_observables", {}) if isinstance(rec.get("final_observables", {}), Mapping) else {}
                energy = obs.get("energy_static", {}) if isinstance(obs.get("energy_static", {}), Mapping) else {}
                noisy_rows.append([
                    "imported_full_circuit",
                    "direct_adapt",
                    str(mode_name),
                    str(bool(rec.get("success", False))),
                    f"{float(energy.get('delta_mean', float('nan'))):.3e}",
                ])
        if isinstance(fixed_lean_replay, Mapping) and fixed_lean_replay:
            energies = fixed_lean_replay.get("energies", {}) if isinstance(fixed_lean_replay.get("energies", {}), Mapping) else {}
            noisy_rows.append([
                "fixed_lean_replay",
                "imported_locked_scaffold",
                "backend_scheduled",
                str(bool(fixed_lean_replay.get("success", False))),
                f"{float(energies.get('best_noisy_minus_ideal', float('nan'))):.3e}",
            ])
        if isinstance(fixed_lean_attribution, Mapping) and fixed_lean_attribution:
            slices = fixed_lean_attribution.get("slices", {}) if isinstance(fixed_lean_attribution.get("slices", {}), Mapping) else {}
            for slice_name, rec in slices.items():
                if not isinstance(rec, Mapping):
                    continue
                noisy_rows.append([
                    "fixed_lean_attribution",
                    "locked_imported_circuit",
                    str(slice_name),
                    str(bool(rec.get("success", False))),
                    f"{float(rec.get('delta_mean', float('nan'))):.3e}",
                ])
        if not noisy_rows:
            noisy_rows = [["(none)", "(none)", "(none)", "False", "nan"]]
        render_compact_table(
            axes[1],
            title="Noisy final deltas",
            col_labels=["Profile", "Method", "Mode", "Available", "Final Δ(noisy-ideal)"],
            rows=noisy_rows[:18],
            fontsize=7,
        )
        pdf.savefig(fig)
        plt.close(fig)

        render_command_page(
            pdf,
            str(run_command),
            script_name="pipelines/hardcoded/hh_staged_noise.py",
            extra_header_lines=[
                f"workflow_json: {staged_cfg.artifacts.output_json}",
                f"noisy methods: {','.join(cfg.noise.methods)}",
                f"noise modes: {','.join(cfg.noise.modes)}",
            ],
        )


def run_staged_hh_noise(cfg: StagedHHNoiseConfig, *, run_command: str | None = None) -> dict[str, Any]:
    run_command_str = base_wf.current_command_string() if run_command is None else str(run_command)
    staged_cfg = cfg.staged
    staged_cfg.artifacts.output_json.parent.mkdir(parents=True, exist_ok=True)
    staged_cfg.artifacts.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    staged_cfg.artifacts.handoff_json.parent.mkdir(parents=True, exist_ok=True)
    staged_cfg.artifacts.replay_output_json.parent.mkdir(parents=True, exist_ok=True)
    staged_cfg.artifacts.replay_output_csv.parent.mkdir(parents=True, exist_ok=True)
    staged_cfg.artifacts.replay_output_md.parent.mkdir(parents=True, exist_ok=True)
    staged_cfg.artifacts.replay_output_log.parent.mkdir(parents=True, exist_ok=True)
    if str(cfg.source.mode) == "imported_artifact":
        imported_prepared_state_audit = {
            "modes": {
                str(mode): _run_imported_prepared_state_audit_mode(
                    source_cfg=cfg.source,
                    noise_cfg=cfg.noise,
                    mode=str(mode),
                )
                for mode in cfg.noise.audit_modes
            }
        }
        full_circuit_import_audit = {
            "modes": (
                {
                    str(mode): _run_imported_full_circuit_audit_mode(
                        source_cfg=cfg.source,
                        noise_cfg=cfg.noise,
                        mode=str(mode),
                    )
                    for mode in cfg.noise.audit_modes
                }
                if bool(cfg.noise.include_full_circuit_audit)
                else {}
            )
        }
        fixed_lean_scaffold_noisy_replay = (
            _run_fixed_lean_noisy_replay_mode(
                source_cfg=cfg.source,
                noise_cfg=cfg.noise,
                fixed_cfg=cfg.fixed_lean_replay,
            )
            if bool(cfg.fixed_lean_replay.enabled)
            else {}
        )
        fixed_lean_noise_attribution = (
            _run_fixed_lean_noise_attribution_mode(
                source_cfg=cfg.source,
                noise_cfg=cfg.noise,
                attribution_cfg=cfg.fixed_lean_attribution,
            )
            if bool(cfg.fixed_lean_attribution.enabled)
            else {}
        )
        payload = {
            "generated_utc": base_wf._now_utc(),
            "pipeline": "hh_staged_noise",
            "workflow_contract": {
                "noise_extension": "imported_adapt_circuit_audit",
                "stage_chain": [],
                "default_subject_policy": (
                    "new full_circuit_import mode defaults to imported lean pareto_lean_l2 artifact only; "
                    "repo-wide staged-noise defaults unchanged"
                ),
                "imported_routes": {
                    "prepared_state_audit": True,
                    "full_circuit_audit": bool(cfg.noise.include_full_circuit_audit),
                    "fixed_lean_scaffold_noisy_replay": bool(cfg.fixed_lean_replay.enabled),
                    "fixed_lean_noise_attribution": bool(cfg.fixed_lean_attribution.enabled),
                },
                "fixed_lean_scaffold_noisy_replay_contract": (
                    {
                        "matched_family_replay": False,
                        "structure_locked": True,
                        "reps": int(cfg.fixed_lean_replay.reps),
                        "optimizer": str(cfg.fixed_lean_replay.method),
                        "noise_mode": str(cfg.fixed_lean_replay.noise_mode),
                        "mitigation": "readout/mthree",
                    }
                    if bool(cfg.fixed_lean_replay.enabled)
                    else {}
                ),
                "fixed_lean_noise_attribution_contract": (
                    {
                        "matched_family_replay": False,
                        "structure_locked": True,
                        "parameter_optimization": False,
                        "circuit_source": "imported_artifact_runtime_theta",
                        "noise_mode": str(cfg.fixed_lean_attribution.noise_mode),
                        "slices": [str(x) for x in cfg.fixed_lean_attribution.slices],
                        "shared_transpile": True,
                        "mitigation": "none",
                        "symmetry": "off",
                    }
                    if bool(cfg.fixed_lean_attribution.enabled)
                    else {}
                ),
            },
            "settings": {
                **base_wf._jsonable(asdict(staged_cfg)),
                "noise": base_wf._jsonable(asdict(cfg.noise)),
                "fixed_lean_noisy_replay": base_wf._jsonable(asdict(cfg.fixed_lean_replay)),
                "fixed_lean_noise_attribution": base_wf._jsonable(asdict(cfg.fixed_lean_attribution)),
            },
            "artifacts": {
                **base_wf._payload_artifacts(staged_cfg),
                "import_source_json": (
                    None if cfg.source.resolved_json is None else str(cfg.source.resolved_json)
                ),
            },
            "command": str(run_command_str),
            "import_source": base_wf._jsonable(asdict(cfg.source)),
            "dynamics_noiseless": {"profiles": {}},
            "dynamics_noisy": {"profiles": {}},
            "noisy_final_audit": {"profiles": {}},
            "dynamics_benchmarks": {"rows": []},
            "imported_prepared_state_audit": imported_prepared_state_audit,
            "full_circuit_import_audit": full_circuit_import_audit,
            "fixed_lean_scaffold_noisy_replay": fixed_lean_scaffold_noisy_replay,
            "fixed_lean_noise_attribution": fixed_lean_noise_attribution,
            "comparisons": {},
        }
        payload["summary"] = _build_noise_summary(payload)
        base_wf._write_json(staged_cfg.artifacts.output_json, payload)
        write_staged_hh_noise_pdf(payload, cfg, run_command_str)
        return payload
    stage_result = base_wf.run_stage_pipeline(staged_cfg)
    dynamics_noiseless = base_wf.run_noiseless_profiles(stage_result, staged_cfg)
    dynamics_noisy, noisy_final_audit, dynamics_benchmarks = run_noisy_profiles(stage_result, cfg)

    payload = base_wf.assemble_payload(
        cfg=staged_cfg,
        stage_result=stage_result,
        dynamics_noiseless=dynamics_noiseless,
        run_command=run_command_str,
    )
    payload["pipeline"] = "hh_staged_noise"
    if isinstance(payload.get("workflow_contract"), dict):
        payload["workflow_contract"]["noise_extension"] = "final_only_noisy_dynamics"
        payload["workflow_contract"]["imported_routes"] = {
            "prepared_state_audit": False,
            "full_circuit_audit": False,
            "fixed_lean_scaffold_noisy_replay": False,
            "fixed_lean_noise_attribution": False,
        }
    if isinstance(payload.get("settings"), dict):
        payload["settings"]["noise"] = base_wf._jsonable(asdict(cfg.noise))
        payload["settings"]["fixed_lean_noisy_replay"] = base_wf._jsonable(asdict(cfg.fixed_lean_replay))
        payload["settings"]["fixed_lean_noise_attribution"] = base_wf._jsonable(asdict(cfg.fixed_lean_attribution))
    payload["import_source"] = base_wf._jsonable(asdict(cfg.source))
    payload["dynamics_noisy"] = dynamics_noisy
    payload["noisy_final_audit"] = noisy_final_audit
    payload["dynamics_benchmarks"] = dynamics_benchmarks
    payload["fixed_lean_scaffold_noisy_replay"] = {}
    payload["fixed_lean_noise_attribution"] = {}
    payload["comparisons"] = {
        **base_wf._compute_comparisons(payload),
        **noise_report._compute_comparisons(payload),
    }
    payload["summary"] = _build_noise_summary(payload)

    staged_cfg.artifacts.output_json.parent.mkdir(parents=True, exist_ok=True)
    base_wf._write_json(staged_cfg.artifacts.output_json, payload)
    write_staged_hh_noise_pdf(payload, cfg, run_command_str)
    return payload
