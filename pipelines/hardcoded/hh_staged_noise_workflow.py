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
    normalize_mitigation_config,
    normalize_symmetry_mitigation_config,
)
from pipelines.hardcoded import hh_staged_workflow as base_wf


_ALLOWED_NOISY_METHODS = {"suzuki2", "cfqm4", "cfqm6"}
_ALLOWED_NOISE_MODES = {"ideal", "shots", "aer_noise", "runtime"}


@dataclass(frozen=True)
class NoiseConfig:
    methods: tuple[str, ...]
    modes: tuple[str, ...]
    shots: int
    oracle_repeats: int
    oracle_aggregate: str
    mitigation_config: dict[str, Any]
    symmetry_mitigation_config: dict[str, Any]
    seed: int
    backend_name: str | None
    use_fake_backend: bool
    allow_aer_fallback: bool
    omp_shm_workaround: bool
    noisy_mode_timeout_s: int
    benchmark_active_coeff_tol: float
    include_final_audit: bool


@dataclass(frozen=True)
class StagedHHNoiseConfig:
    staged: base_wf.StagedHHConfig
    noise: NoiseConfig


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
    noise_cfg = NoiseConfig(
        methods=methods,
        modes=modes,
        shots=int(getattr(args, "shots")),
        oracle_repeats=int(getattr(args, "oracle_repeats")),
        oracle_aggregate=str(getattr(args, "oracle_aggregate")),
        mitigation_config=normalize_mitigation_config(
            {
                "mode": str(getattr(args, "mitigation")),
                "zne_scales": getattr(args, "zne_scales", None),
                "dd_sequence": getattr(args, "dd_sequence", None),
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
        seed=int(getattr(args, "noise_seed")),
        backend_name=(None if getattr(args, "backend_name", None) in {None, "", "none"} else str(getattr(args, "backend_name"))),
        use_fake_backend=bool(getattr(args, "use_fake_backend")),
        allow_aer_fallback=bool(getattr(args, "allow_aer_fallback")),
        omp_shm_workaround=bool(getattr(args, "omp_shm_workaround")),
        noisy_mode_timeout_s=int(getattr(args, "noisy_mode_timeout_s")),
        benchmark_active_coeff_tol=float(getattr(args, "benchmark_active_coeff_tol")),
        include_final_audit=bool(getattr(args, "include_final_audit")),
    )
    return StagedHHNoiseConfig(staged=staged_cfg, noise=noise_cfg)


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
    return {
        "warm_delta_abs": float(warm.get("delta_abs", float("nan"))),
        "adapt_delta_abs": float(adapt.get("delta_abs", float("nan"))),
        "final_delta_abs": float(final.get("delta_abs", float("nan"))),
        "noisy_method_modes_completed": int(completed),
        "noisy_method_modes_total": int(total),
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

    with PdfPages(pdf_path) as pdf:
        render_parameter_manifest(
            pdf,
            model="Hubbard-Holstein",
            ansatz="warm: hh_hva_ptw; ADAPT: staged HH; final: matched-family replay; noisy final-only dynamics",
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
                "shots": int(cfg.noise.shots),
                "mitigation": dict(cfg.noise.mitigation_config),
                "symmetry_mitigation": dict(cfg.noise.symmetry_mitigation_config),
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
                f"Failure samples: {summary.get('failure_samples', [])}",
                "",
                f"workflow_json: {staged_cfg.artifacts.output_json}",
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
    if isinstance(payload.get("settings"), dict):
        payload["settings"]["noise"] = base_wf._jsonable(asdict(cfg.noise))
    payload["dynamics_noisy"] = dynamics_noisy
    payload["noisy_final_audit"] = noisy_final_audit
    payload["dynamics_benchmarks"] = dynamics_benchmarks
    payload["comparisons"] = {
        **base_wf._compute_comparisons(payload),
        **noise_report._compute_comparisons(payload),
    }
    payload["summary"] = _build_noise_summary(payload)

    staged_cfg.artifacts.output_json.parent.mkdir(parents=True, exist_ok=True)
    base_wf._write_json(staged_cfg.artifacts.output_json, payload)
    write_staged_hh_noise_pdf(payload, cfg, run_command_str)
    return payload
