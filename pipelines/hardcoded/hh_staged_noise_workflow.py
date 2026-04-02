#!/usr/bin/env python3
"""Noise-capable staged HH workflow built on the shared staged core."""

from __future__ import annotations

import json
import multiprocessing as mp
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
    OracleConfig,
    normalize_mitigation_config,
    normalize_symmetry_mitigation_config,
)
from pipelines.hardcoded import hh_staged_workflow as base_wf
from pipelines.hardcoded.hh_realtime_checkpoint_controller import RealtimeCheckpointController
from pipelines.hardcoded.hh_realtime_measurement import validate_controller_oracle_base_config
from pipelines.hardcoded.imported_artifact_resolution import (
    resolve_default_hh_marrakesh_runtime_candidate_artifact_json,
    resolve_default_hh_nighthawk_fixed_scaffold_artifact_json,
    resolve_default_lean_pareto_l2_artifact_json,
    resolve_default_lean_pareto_l2_circuit_ready_artifact_json,
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
    controller_noise_mode: str | None
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
    controller_timeout_s: int | None
    controller_progress_every_s: float
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
class FixedLeanCompileControlScoutConfig:
    enabled: bool
    baseline_transpile_optimization_level: int
    baseline_seed_transpiler: int
    scout_transpile_optimization_levels: tuple[int, ...]
    scout_seed_transpilers: tuple[int, ...]
    noise_mode: str
    mitigation_config: dict[str, Any]
    symmetry_mitigation_config: dict[str, Any]
    rank_policy: str


@dataclass(frozen=True)
class FixedScaffoldNoisyReplayConfig:
    enabled: bool
    subject_kind: str
    reps: int
    method: str
    seed: int
    maxiter: int
    wallclock_cap_s: int
    progress_every_s: float
    spsa_a: float
    spsa_c: float
    spsa_alpha: float
    spsa_gamma: float
    spsa_A: float
    spsa_avg_last: int
    spsa_eval_repeats: int
    spsa_eval_agg: str
    local_dd_probe_sequence: str | None
    noise_mode: str
    mitigation_config: dict[str, Any]
    symmetry_mitigation_config: dict[str, Any]
    runtime_profile_config: dict[str, Any]
    runtime_session_config: dict[str, Any]
    transpile_optimization_level: int
    seed_transpiler: int | None
    include_dd_probe: bool
    include_final_zne_audit: bool


@dataclass(frozen=True)
class FixedScaffoldRuntimeEnergyOnlyConfig:
    enabled: bool
    subject_kind: str
    noise_mode: str
    mitigation_config: dict[str, Any]
    symmetry_mitigation_config: dict[str, Any]
    runtime_profile_config: dict[str, Any]
    runtime_session_config: dict[str, Any]
    transpile_optimization_level: int
    seed_transpiler: int | None
    include_dd_probe: bool
    include_final_zne_audit: bool


@dataclass(frozen=True)
class FixedScaffoldRuntimeRawBaselineConfig:
    enabled: bool
    subject_kind: str
    noise_mode: str
    mitigation_config: dict[str, Any]
    symmetry_mitigation_config: dict[str, Any]
    runtime_profile_config: dict[str, Any]
    runtime_session_config: dict[str, Any]
    transpile_optimization_level: int
    seed_transpiler: int | None
    raw_transport: str
    raw_store_memory: bool
    raw_artifact_path: str | None


@dataclass(frozen=True)
class FixedScaffoldCompileControlScoutConfig:
    enabled: bool
    subject_kind: str
    baseline_transpile_optimization_level: int
    baseline_seed_transpiler: int
    scout_transpile_optimization_levels: tuple[int, ...]
    scout_seed_transpilers: tuple[int, ...]
    noise_mode: str
    mitigation_config: dict[str, Any]
    symmetry_mitigation_config: dict[str, Any]
    rank_policy: str


@dataclass(frozen=True)
class FixedScaffoldSavedThetaMitigationMatrixConfig:
    enabled: bool
    subject_kind: str
    noise_mode: str
    compile_presets: tuple[dict[str, Any], ...]
    zne_scales: tuple[float, ...]
    suppression_labels: tuple[str, ...]
    selected_cells: tuple[str, ...]
    mitigation_config_base: dict[str, Any]
    symmetry_mitigation_config: dict[str, Any]
    rank_policy: str
    raw_artifact_root: str | None


@dataclass(frozen=True)
class FixedScaffoldNoiseAttributionConfig:
    enabled: bool
    subject_kind: str
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
    default_subject_kind: str | None = None
    default_subject: bool = False


@dataclass(frozen=True)
class StagedHHNoiseConfig:
    staged: base_wf.StagedHHConfig
    noise: NoiseConfig
    source: StagedNoiseSourceConfig
    fixed_lean_replay: FixedLeanNoisyReplayConfig
    fixed_lean_attribution: FixedLeanNoiseAttributionConfig
    fixed_lean_compile_control_scout: FixedLeanCompileControlScoutConfig
    fixed_scaffold_replay: FixedScaffoldNoisyReplayConfig
    fixed_scaffold_runtime_energy_only: FixedScaffoldRuntimeEnergyOnlyConfig
    fixed_scaffold_runtime_raw_baseline: FixedScaffoldRuntimeRawBaselineConfig
    fixed_scaffold_compile_control_scout: FixedScaffoldCompileControlScoutConfig
    fixed_scaffold_saved_theta_mitigation_matrix: FixedScaffoldSavedThetaMitigationMatrixConfig
    fixed_scaffold_attribution: FixedScaffoldNoiseAttributionConfig


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


def _parse_fixed_scaffold_matrix_compile_presets(raw: str | None) -> tuple[dict[str, Any], ...]:
    if raw in {None, ""}:
        return ()
    presets: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for item in [x.strip() for x in str(raw).split(",") if x.strip()]:
        parts = [x.strip() for x in item.split(":")]
        if len(parts) != 3:
            raise ValueError(
                "fixed scaffold matrix compile presets must use label:optimization_level:seed_transpiler"
            )
        label, opt_raw, seed_raw = parts
        if label == "":
            raise ValueError("fixed scaffold matrix compile preset label must be non-empty")
        if label in seen_labels:
            raise ValueError(f"Duplicate fixed scaffold matrix compile preset label '{label}'")
        try:
            opt_level = int(opt_raw)
            seed_transpiler = int(seed_raw)
        except ValueError as exc:
            raise ValueError(
                f"Invalid fixed scaffold matrix compile preset '{item}': {exc}"
            ) from exc
        if opt_level < 0 or seed_transpiler < 0:
            raise ValueError(
                f"Invalid fixed scaffold matrix compile preset '{item}': values must be >= 0"
            )
        presets.append(
            {
                "label": str(label),
                "transpile_optimization_level": int(opt_level),
                "seed_transpiler": int(seed_transpiler),
            }
        )
        seen_labels.add(label)
    return tuple(presets)


def _parse_fixed_scaffold_matrix_selected_cells(raw: str | None) -> tuple[str, ...]:
    if raw in {None, ""}:
        return ()
    allowed_suffixes = {"twirl", "dd", "twirl_dd"}
    selected: list[str] = []
    seen: set[str] = set()
    for item in [x.strip() for x in str(raw).split(",") if x.strip()]:
        if item in seen:
            continue
        if "__zne_" not in item:
            raise ValueError(
                "fixed scaffold matrix selected cells must use preset__zne_on|off__twirl|dd|twirl_dd"
            )
        prefix, suffix = item.split("__zne_", 1)
        if prefix.strip() == "":
            raise ValueError(
                "fixed scaffold matrix selected cells must use preset__zne_on|off__twirl|dd|twirl_dd"
            )
        zne_part, sep, stack_part = suffix.partition("__")
        if sep == "" or zne_part not in {"on", "off"} or stack_part not in allowed_suffixes:
            raise ValueError(
                "fixed scaffold matrix selected cells must use preset__zne_on|off__twirl|dd|twirl_dd"
            )
        selected.append(str(item))
        seen.add(str(item))
    if not selected:
        raise ValueError("Expected at least one fixed scaffold matrix selected cell.")
    return tuple(selected)


def _planned_fixed_scaffold_matrix_cell_labels(
    *,
    compile_presets: Sequence[Mapping[str, Any]],
    suppression_labels: Sequence[str],
) -> tuple[str, ...]:
    short_labels = {
        "readout_plus_gate_twirling": "twirl",
        "readout_plus_local_dd": "dd",
        "readout_plus_gate_twirling_plus_local_dd": "twirl_dd",
    }
    labels: list[str] = []
    for preset in compile_presets:
        preset_label = str(preset.get("label", "")).strip()
        if preset_label == "":
            preset_label = (
                f"opt{int(preset.get('transpile_optimization_level'))}"
                f"_seed{int(preset.get('seed_transpiler'))}"
            )
        for zne_enabled in (False, True):
            for suppression_label in suppression_labels:
                short_label = short_labels.get(str(suppression_label))
                if short_label is None:
                    raise ValueError(
                        f"Unsupported fixed scaffold matrix suppression label '{suppression_label}'"
                    )
                labels.append(
                    f"{preset_label}__zne_{'on' if bool(zne_enabled) else 'off'}__{short_label}"
                )
    return tuple(labels)


def _build_fixed_scaffold_matrix_base_mitigation_config(base_mode: str) -> dict[str, Any]:
    mode_key = str(base_mode).strip().lower()
    if mode_key == "readout":
        return normalize_mitigation_config(
            {
                "mode": "readout",
                "local_readout_strategy": "mthree",
            }
        )
    if mode_key == "none":
        return normalize_mitigation_config({"mode": "none"})
    raise ValueError(
        "fixed scaffold matrix base mitigation mode must be one of: readout, none"
    )


def _repo_relative_str(path_like: Path | str | None) -> str | None:
    if path_like in {None, ""}:
        return None
    path = Path(path_like)
    repo_root = Path(__file__).resolve().parents[2]
    if path.is_absolute():
        try:
            return str(path.resolve().relative_to(repo_root))
        except Exception:
            return str(path.resolve())
    return str(path)


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
    return resolve_default_lean_pareto_l2_circuit_ready_artifact_json()


def _default_fixed_scaffold_import_json() -> tuple[Path | None, bool]:
    return resolve_default_hh_nighthawk_fixed_scaffold_artifact_json()


def _default_fixed_scaffold_runtime_import_json() -> tuple[Path | None, bool]:
    return resolve_default_hh_marrakesh_runtime_candidate_artifact_json()


def _default_fixed_scaffold_local_replay_import_json() -> tuple[Path | None, bool]:
    return resolve_default_hh_marrakesh_runtime_candidate_artifact_json()


def _resolve_import_source(
    *,
    requested_json: Path | None,
    require_default_import_source: bool,
    default_subject_kind: str | None = None,
) -> StagedNoiseSourceConfig:
    requested = Path(requested_json) if requested_json is not None else None
    default_subject = False
    if requested is None and bool(require_default_import_source):
        if str(default_subject_kind) == "hh_marrakesh_gate_pruned_6term_drop_eyezee_v1":
            requested, default_subject = _default_fixed_scaffold_runtime_import_json()
        elif str(default_subject_kind) in {
            "hh_nighthawk_gate_pruned_7term_v1",
            "hh_nighthawk_circuit_optimized_7term_v1",
        }:
            requested, default_subject = _default_fixed_scaffold_import_json()
        else:
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
        default_subject_kind=(None if default_subject_kind in {None, ""} else str(default_subject_kind)),
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
    include_fixed_lean_compile_control_scout = bool(
        getattr(args, "include_fixed_lean_compile_control_scout", False)
    )
    include_fixed_scaffold_noisy_replay = bool(
        getattr(args, "include_fixed_scaffold_noisy_replay", False)
    )
    include_fixed_scaffold_runtime_energy_only = bool(
        getattr(args, "include_fixed_scaffold_runtime_energy_only_baseline", False)
    )
    include_fixed_scaffold_runtime_raw_baseline = bool(
        getattr(args, "include_fixed_scaffold_runtime_raw_baseline", False)
    )
    include_fixed_scaffold_compile_control_scout = bool(
        getattr(args, "include_fixed_scaffold_compile_control_scout", False)
    )
    include_fixed_scaffold_saved_theta_mitigation_matrix = bool(
        getattr(args, "include_fixed_scaffold_saved_theta_mitigation_matrix", False)
    )
    include_fixed_scaffold_noise_attribution = bool(
        getattr(args, "include_fixed_scaffold_noise_attribution", False)
    )
    any_fixed_lean_route = bool(
        include_fixed_lean_noisy_replay
        or include_fixed_lean_noise_attribution
        or include_fixed_lean_compile_control_scout
    )
    any_fixed_scaffold_route = bool(
        include_fixed_scaffold_noisy_replay
        or include_fixed_scaffold_runtime_energy_only
        or include_fixed_scaffold_runtime_raw_baseline
        or include_fixed_scaffold_compile_control_scout
        or include_fixed_scaffold_saved_theta_mitigation_matrix
        or include_fixed_scaffold_noise_attribution
    )
    if bool(include_fixed_scaffold_runtime_energy_only) and bool(include_fixed_scaffold_runtime_raw_baseline):
        raise ValueError(
            "fixed scaffold runtime energy-only and raw-baseline routes cannot be enabled together in one invocation."
        )
    if any_fixed_lean_route and any_fixed_scaffold_route:
        raise ValueError(
            "fixed lean imported routes and fixed scaffold imported routes cannot be enabled together "
            "in one invocation."
        )
    default_subject_kind = None
    if bool(
        include_fixed_scaffold_runtime_energy_only
        or include_fixed_scaffold_runtime_raw_baseline
        or include_fixed_scaffold_noisy_replay
        or include_fixed_scaffold_compile_control_scout
        or include_fixed_scaffold_saved_theta_mitigation_matrix
    ):
        default_subject_kind = "hh_marrakesh_gate_pruned_6term_drop_eyezee_v1"
    elif any_fixed_scaffold_route:
        default_subject_kind = "hh_nighthawk_gate_pruned_7term_v1"
    elif bool(include_full_circuit_audit or any_fixed_lean_route):
        default_subject_kind = "lean_pareto_l2"
    source_cfg = _resolve_import_source(
        requested_json=getattr(args, "fixed_final_state_json", None),
        require_default_import_source=bool(
            include_full_circuit_audit
            or include_fixed_lean_noisy_replay
            or include_fixed_lean_noise_attribution
            or include_fixed_lean_compile_control_scout
            or include_fixed_scaffold_noisy_replay
            or include_fixed_scaffold_runtime_energy_only
            or include_fixed_scaffold_runtime_raw_baseline
            or include_fixed_scaffold_compile_control_scout
            or include_fixed_scaffold_saved_theta_mitigation_matrix
            or include_fixed_scaffold_noise_attribution
        ),
        default_subject_kind=default_subject_kind,
    )
    raw_audit_modes = getattr(args, "audit_noise_modes", None)
    needs_backend_scheduled_audit_default = bool(
        str(source_cfg.mode) == "imported_artifact"
        and (
            include_full_circuit_audit
            or include_fixed_lean_noise_attribution
            or include_fixed_scaffold_noise_attribution
        )
    )
    if raw_audit_modes in {None, ""}:
        if needs_backend_scheduled_audit_default:
            audit_modes = ("ideal", "backend_scheduled")
        else:
            audit_modes = tuple(modes)
    else:
        audit_modes = _parse_csv(raw_audit_modes, allowed=_ALLOWED_AUDIT_NOISE_MODES, label="audit noise mode")
    backend_name_raw = getattr(args, "backend_name", None)
    backend_name = None if backend_name_raw in {None, "", "none"} else str(backend_name_raw)
    use_fake_backend = bool(getattr(args, "use_fake_backend"))
    if str(source_cfg.mode) == "imported_artifact" and str(staged_cfg.realtime_checkpoint.mode) != "off":
        raise ValueError(
            "adaptive realtime checkpoint controller is currently supported only for fresh-stage noisy runs; disable --checkpoint-controller-mode for imported-artifact routes."
        )
    if (
        str(source_cfg.mode) == "imported_artifact"
        and staged_cfg.adapt.phase3_oracle_gradient_config is not None
    ):
        raise ValueError(
            "phase3 oracle-gradient knobs are only valid for fresh-stage staged noise runs; imported-artifact routes do not execute ADAPT."
        )
    imported_requires_fake_backend = bool(
        str(source_cfg.mode) == "imported_artifact"
        and (
            "backend_scheduled" in audit_modes
            or bool(any_fixed_lean_route)
            or bool(include_fixed_scaffold_compile_control_scout)
            or bool(include_fixed_scaffold_saved_theta_mitigation_matrix)
            or bool(include_fixed_scaffold_noise_attribution)
            or (bool(include_fixed_scaffold_runtime_raw_baseline) and bool(use_fake_backend))
            or (bool(include_fixed_scaffold_noisy_replay) and bool(use_fake_backend))
        )
    )
    if imported_requires_fake_backend:
        if bool(include_fixed_lean_compile_control_scout) and backend_name is None:
            raise ValueError(
                "fixed lean compile-control scout requires an explicit --backend-name so the local scout stays pinned to the intended Heron/Marrakesh-class target."
            )
        if bool(include_fixed_scaffold_compile_control_scout) and backend_name is None:
            raise ValueError(
                "fixed scaffold compile-control scout requires an explicit --backend-name so the local scout stays pinned to the intended Heron/Marrakesh-class target."
            )
        if backend_name is None:
            backend_name = (
                "FakeMarrakesh"
                if bool(
                    include_fixed_scaffold_noisy_replay
                    or include_fixed_scaffold_runtime_raw_baseline
                    or include_fixed_scaffold_saved_theta_mitigation_matrix
                )
                else ("FakeNighthawk" if bool(any_fixed_scaffold_route) else "FakeGuadalupeV2")
            )
        use_fake_backend = True
    controller_noise_mode: str | None = None
    controller_mode = str(staged_cfg.realtime_checkpoint.mode)
    if controller_mode in {"oracle_v1", "off"}:
        requested_controller_noise_mode = str(
            getattr(args, "checkpoint_controller_noise_mode", "inherit")
        ).strip().lower()
        if requested_controller_noise_mode != "inherit":
            controller_noise_mode = str(requested_controller_noise_mode)
        elif bool(use_fake_backend):
            controller_noise_mode = "backend_scheduled"
        elif len(modes) == 1:
            controller_noise_mode = str(modes[0])
        else:
            raise ValueError(
                "checkpoint controller off/oracle_v1 requires an explicit --checkpoint-controller-noise-mode when multiple --noise-modes are configured."
            )
        if str(controller_noise_mode) == "backend_scheduled" and not bool(use_fake_backend):
            raise ValueError(
                "checkpoint controller off/oracle_v1 backend_scheduled mode requires --use-fake-backend."
            )
    noise_cfg = NoiseConfig(
        methods=methods,
        modes=modes,
        audit_modes=tuple(audit_modes),
        controller_noise_mode=(None if controller_noise_mode is None else str(controller_noise_mode)),
        shots=int(getattr(args, "shots")),
        oracle_repeats=int(getattr(args, "oracle_repeats")),
        oracle_aggregate=str(getattr(args, "oracle_aggregate")),
        mitigation_config=normalize_mitigation_config(
            {
                "mode": str(getattr(args, "mitigation")),
                "zne_scales": getattr(args, "zne_scales", None),
                "dd_sequence": getattr(args, "dd_sequence", None),
                "local_readout_strategy": getattr(args, "local_readout_strategy", None),
                "local_gate_twirling": bool(getattr(args, "local_gate_twirling", False)),
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
        controller_timeout_s=(
            None
            if getattr(args, "checkpoint_controller_timeout_s", None) is None
            else int(getattr(args, "checkpoint_controller_timeout_s"))
        ),
        controller_progress_every_s=float(getattr(args, "checkpoint_controller_progress_every_s")),
        benchmark_active_coeff_tol=float(getattr(args, "benchmark_active_coeff_tol")),
        include_final_audit=bool(getattr(args, "include_final_audit")),
        include_full_circuit_audit=bool(include_full_circuit_audit),
    )
    if controller_mode in {"oracle_v1", "off"} and noise_cfg.controller_noise_mode is not None:
        validate_controller_oracle_base_config(_controller_oracle_config_from_noise_cfg(noise_cfg))
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
        noise_mode=("backend_scheduled" if bool(noise_cfg.use_fake_backend) else "runtime"),
        mitigation_config=normalize_mitigation_config(
            {
                "mode": "readout",
                "local_readout_strategy": "mthree",
                "local_gate_twirling": bool(getattr(args, "local_gate_twirling", False)),
            }
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
    fixed_lean_compile_control_scout_enabled = bool(include_fixed_lean_compile_control_scout)
    if fixed_lean_compile_control_scout_enabled and str(source_cfg.mode) != "imported_artifact":
        raise ValueError(
            "fixed lean compile-control scout requires an imported artifact source; pass "
            "--fixed-final-state-json or enable the default imported lean subject."
        )
    fixed_lean_compile_control_scout_cfg = FixedLeanCompileControlScoutConfig(
        enabled=bool(fixed_lean_compile_control_scout_enabled),
        baseline_transpile_optimization_level=1,
        baseline_seed_transpiler=7,
        scout_transpile_optimization_levels=(1, 2),
        scout_seed_transpilers=(0, 1, 2, 3, 4),
        noise_mode="backend_scheduled",
        mitigation_config=normalize_mitigation_config(
            {
                "mode": "readout",
                "local_readout_strategy": "mthree",
                "local_gate_twirling": bool(getattr(args, "local_gate_twirling", False)),
            }
        ),
        symmetry_mitigation_config=normalize_symmetry_mitigation_config({"mode": "off"}),
        rank_policy="delta_mean_then_two_qubit_then_depth_then_size",
    )
    runtime_profile_cfg = noise_report.normalize_runtime_estimator_profile_config(
        {"name": str(getattr(args, "fixed_scaffold_runtime_profile"))}
    )
    runtime_raw_profile_cfg = noise_report.normalize_runtime_estimator_profile_config(
        {"name": "legacy_runtime_v0"}
    )
    runtime_session_cfg = noise_report.normalize_runtime_session_policy_config(
        {"mode": str(getattr(args, "fixed_scaffold_runtime_session_policy"))}
    )
    runtime_transpile_opt = int(getattr(args, "fixed_scaffold_runtime_transpile_optimization_level"))
    runtime_seed_transpiler = int(getattr(args, "fixed_scaffold_runtime_seed_transpiler"))
    fixed_scaffold_runtime_subject_kind = "hh_marrakesh_gate_pruned_6term_drop_eyezee_v1"
    fixed_scaffold_runtime_energy_only_enabled = bool(include_fixed_scaffold_runtime_energy_only)
    fixed_scaffold_runtime_raw_baseline_enabled = bool(include_fixed_scaffold_runtime_raw_baseline)
    if fixed_scaffold_runtime_energy_only_enabled:
        if str(source_cfg.mode) != "imported_artifact":
            raise ValueError(
                "fixed scaffold runtime energy-only baseline requires an imported artifact source; pass "
                "--fixed-final-state-json or enable the default Marrakesh/Heron runtime candidate."
            )
        if bool(noise_cfg.use_fake_backend):
            raise ValueError(
                "fixed scaffold runtime energy-only baseline requires a real Runtime backend; omit --use-fake-backend."
            )
        if backend_name is None:
            raise ValueError(
                "fixed scaffold runtime energy-only baseline requires --backend-name <ibm_backend>."
            )
    if fixed_scaffold_runtime_raw_baseline_enabled:
        if str(source_cfg.mode) != "imported_artifact":
            raise ValueError(
                "fixed scaffold runtime raw baseline requires an imported artifact source; pass "
                "--fixed-final-state-json or enable the default Marrakesh/Heron runtime candidate."
            )
        if not bool(noise_cfg.use_fake_backend) and backend_name is None:
            raise ValueError(
                "fixed scaffold runtime raw baseline requires --backend-name <ibm_backend>."
            )
        if bool(getattr(args, "include_fixed_scaffold_runtime_dd_probe", False)):
            raise ValueError(
                "fixed scaffold runtime raw baseline does not support the legacy Runtime DD probe flag."
            )
        if bool(getattr(args, "include_fixed_scaffold_runtime_final_zne_audit", False)):
            raise ValueError(
                "fixed scaffold runtime raw baseline does not support the legacy Runtime final ZNE audit flag."
            )
        if bool(noise_cfg.use_fake_backend) and bool(include_full_circuit_audit):
            requested_local_postprocessing = (
                str(noise_cfg.mitigation_config.get("mode", "none")) != "none"
                or str(noise_cfg.symmetry_mitigation_config.get("mode", "off")) != "off"
            )
            if requested_local_postprocessing:
                raise ValueError(
                    "fixed scaffold runtime raw baseline local diagonal postprocessing cannot be combined "
                    "with full-circuit audit in one imported-artifact invocation."
                )
    if bool(noise_cfg.use_fake_backend) and bool(fixed_scaffold_runtime_raw_baseline_enabled):
        runtime_raw_mitigation_cfg = dict(noise_cfg.mitigation_config)
        runtime_raw_symmetry_cfg = dict(noise_cfg.symmetry_mitigation_config)
        if str(runtime_raw_mitigation_cfg.get("mode", "none")) == "readout" and runtime_raw_mitigation_cfg.get(
            "local_readout_strategy", None
        ) in {None, "", "none"}:
            runtime_raw_mitigation_cfg = normalize_mitigation_config(
                {
                    **dict(runtime_raw_mitigation_cfg),
                    "local_readout_strategy": "mthree",
                }
            )
    else:
        runtime_raw_mitigation_cfg = normalize_mitigation_config({"mode": "none"})
        runtime_raw_symmetry_cfg = normalize_symmetry_mitigation_config({"mode": "off"})
    fixed_scaffold_runtime_energy_only_cfg = FixedScaffoldRuntimeEnergyOnlyConfig(
        enabled=bool(fixed_scaffold_runtime_energy_only_enabled),
        subject_kind="hh_marrakesh_gate_pruned_6term_drop_eyezee_v1",
        noise_mode="runtime",
        mitigation_config=normalize_mitigation_config({"mode": "readout"}),
        symmetry_mitigation_config=normalize_symmetry_mitigation_config({"mode": "off"}),
        runtime_profile_config=dict(runtime_profile_cfg),
        runtime_session_config=dict(runtime_session_cfg),
        transpile_optimization_level=int(runtime_transpile_opt),
        seed_transpiler=int(runtime_seed_transpiler),
        include_dd_probe=bool(getattr(args, "include_fixed_scaffold_runtime_dd_probe", False)),
        include_final_zne_audit=bool(
            getattr(args, "include_fixed_scaffold_runtime_final_zne_audit", False)
        ),
    )
    fixed_scaffold_runtime_raw_baseline_cfg = FixedScaffoldRuntimeRawBaselineConfig(
        enabled=bool(fixed_scaffold_runtime_raw_baseline_enabled),
        subject_kind="hh_marrakesh_gate_pruned_6term_drop_eyezee_v1",
        noise_mode=("backend_scheduled" if bool(noise_cfg.use_fake_backend) else "runtime"),
        mitigation_config=dict(runtime_raw_mitigation_cfg),
        symmetry_mitigation_config=dict(runtime_raw_symmetry_cfg),
        runtime_profile_config=dict(runtime_raw_profile_cfg),
        runtime_session_config=dict(runtime_session_cfg),
        transpile_optimization_level=int(runtime_transpile_opt),
        seed_transpiler=int(runtime_seed_transpiler),
        raw_transport=str(getattr(args, "fixed_scaffold_runtime_raw_transport")),
        raw_store_memory=bool(getattr(args, "fixed_scaffold_runtime_raw_store_memory", False)),
        raw_artifact_path=(
            None
            if getattr(args, "fixed_scaffold_runtime_raw_artifact_path", None) in {None, ""}
            else str(getattr(args, "fixed_scaffold_runtime_raw_artifact_path"))
        ),
    )
    fixed_scaffold_replay_enabled = bool(include_fixed_scaffold_noisy_replay)
    if fixed_scaffold_replay_enabled:
        if str(source_cfg.mode) != "imported_artifact":
            raise ValueError(
                "fixed scaffold noisy replay requires an imported artifact source; pass "
                "--fixed-final-state-json or enable the default fixed-scaffold subject."
            )
        if not bool(noise_cfg.use_fake_backend):
            raise ValueError(
                "fixed scaffold noisy replay is local-only on the fake-backend path; pass --use-fake-backend."
            )
        if bool(getattr(args, "include_fixed_scaffold_runtime_dd_probe", False)):
            raise ValueError(
                "fixed scaffold noisy replay does not use the runtime DD probe flag; use --dd-sequence XpXm for the saved-theta local DD probe."
            )
        if bool(getattr(args, "include_fixed_scaffold_runtime_final_zne_audit", False)):
            raise ValueError(
                "fixed scaffold noisy replay is local-only and does not support the runtime final ZNE audit flag."
            )
        fixed_scaffold_method = str(staged_cfg.replay.method)
        if fixed_scaffold_method.strip().upper() not in {"SPSA", "POWELL"}:
            raise ValueError(
                "fixed scaffold noisy replay currently supports only final replay optimizer SPSA or Powell."
            )
        requested_dd_sequence_raw = getattr(args, "dd_sequence", None)
        if requested_dd_sequence_raw in {None, "", "none"}:
            fixed_scaffold_local_dd_probe_sequence = None
        else:
            fixed_scaffold_local_dd_probe_sequence = str(requested_dd_sequence_raw).strip().upper()
            if fixed_scaffold_local_dd_probe_sequence != "XPXM":
                raise ValueError(
                    "fixed scaffold noisy replay local DD probe currently supports only --dd-sequence XpXm."
                )
        fixed_scaffold_requested_mitigation = dict(noise_cfg.mitigation_config)
        fixed_scaffold_requested_symmetry = dict(noise_cfg.symmetry_mitigation_config)
        if str(fixed_scaffold_requested_mitigation.get("mode", "none")).strip().lower() == "dd":
            raise ValueError(
                "fixed scaffold noisy replay keeps DD as a saved-theta local probe only; use --dd-sequence XpXm with readout / zne / none."
            )
        fixed_scaffold_is_compat_default = (
            str(fixed_scaffold_requested_mitigation.get("mode", "none")).strip().lower() == "none"
            and not bool(fixed_scaffold_requested_mitigation.get("zne_scales", []))
            and fixed_scaffold_requested_mitigation.get("dd_sequence", None) in {None, "", "none"}
            and fixed_scaffold_requested_mitigation.get("local_readout_strategy", None) in {None, "", "none"}
            and str(fixed_scaffold_requested_symmetry.get("mode", "off")).strip().lower() == "off"
        )
        if fixed_scaffold_is_compat_default:
            fixed_scaffold_replay_mitigation_cfg = normalize_mitigation_config(
                {
                    "mode": "readout",
                    "local_readout_strategy": "mthree",
                    "local_gate_twirling": bool(getattr(args, "local_gate_twirling", False)),
                }
            )
        else:
            fixed_scaffold_replay_mitigation_payload = dict(fixed_scaffold_requested_mitigation)
            fixed_scaffold_replay_mitigation_payload["dd_sequence"] = None
            if (
                str(fixed_scaffold_replay_mitigation_payload.get("mode", "none")).strip().lower() == "readout"
                and fixed_scaffold_replay_mitigation_payload.get("local_readout_strategy", None) in {None, "", "none"}
            ):
                fixed_scaffold_replay_mitigation_payload["local_readout_strategy"] = "mthree"
            fixed_scaffold_replay_mitigation_cfg = normalize_mitigation_config(
                fixed_scaffold_replay_mitigation_payload
            )
        fixed_scaffold_replay_symmetry_cfg = normalize_symmetry_mitigation_config(
            fixed_scaffold_requested_symmetry
        )
    else:
        fixed_scaffold_method = str(staged_cfg.replay.method)
        fixed_scaffold_local_dd_probe_sequence = None
        fixed_scaffold_replay_mitigation_cfg = normalize_mitigation_config(
            {
                "mode": "readout",
                "local_readout_strategy": "mthree",
                "local_gate_twirling": bool(getattr(args, "local_gate_twirling", False)),
            }
        )
        fixed_scaffold_replay_symmetry_cfg = normalize_symmetry_mitigation_config({"mode": "off"})
    fixed_scaffold_replay_cfg = FixedScaffoldNoisyReplayConfig(
        enabled=bool(fixed_scaffold_replay_enabled),
        subject_kind=str(fixed_scaffold_runtime_subject_kind),
        reps=1,
        method=str(fixed_scaffold_method),
        seed=int(staged_cfg.replay.seed),
        maxiter=int(staged_cfg.replay.maxiter),
        wallclock_cap_s=int(staged_cfg.replay.wallclock_cap_s),
        progress_every_s=float(staged_cfg.replay.progress_every_s),
        spsa_a=float(staged_cfg.replay.spsa_a),
        spsa_c=float(staged_cfg.replay.spsa_c),
        spsa_alpha=float(staged_cfg.replay.spsa_alpha),
        spsa_gamma=float(staged_cfg.replay.spsa_gamma),
        spsa_A=float(staged_cfg.replay.spsa_A),
        spsa_avg_last=int(staged_cfg.replay.spsa_avg_last),
        spsa_eval_repeats=int(staged_cfg.replay.spsa_eval_repeats),
        spsa_eval_agg=str(staged_cfg.replay.spsa_eval_agg),
        local_dd_probe_sequence=(
            None if fixed_scaffold_local_dd_probe_sequence is None else str(fixed_scaffold_local_dd_probe_sequence)
        ),
        noise_mode=("backend_scheduled" if bool(noise_cfg.use_fake_backend) else "runtime"),
        mitigation_config=dict(fixed_scaffold_replay_mitigation_cfg),
        symmetry_mitigation_config=dict(fixed_scaffold_replay_symmetry_cfg),
        runtime_profile_config=dict(runtime_profile_cfg),
        runtime_session_config=dict(runtime_session_cfg),
        transpile_optimization_level=int(runtime_transpile_opt),
        seed_transpiler=int(runtime_seed_transpiler),
        include_dd_probe=False,
        include_final_zne_audit=False,
    )
    fixed_scaffold_compile_control_scout_enabled = bool(include_fixed_scaffold_compile_control_scout)
    if fixed_scaffold_compile_control_scout_enabled and str(source_cfg.mode) != "imported_artifact":
        raise ValueError(
            "fixed scaffold compile-control scout requires an imported artifact source; pass "
            "--fixed-final-state-json or enable the default Marrakesh/Heron runtime candidate."
        )
    fixed_scaffold_compile_control_scout_cfg = FixedScaffoldCompileControlScoutConfig(
        enabled=bool(fixed_scaffold_compile_control_scout_enabled),
        subject_kind="hh_marrakesh_gate_pruned_6term_drop_eyezee_v1",
        baseline_transpile_optimization_level=int(runtime_transpile_opt),
        baseline_seed_transpiler=int(runtime_seed_transpiler),
        scout_transpile_optimization_levels=(1, 2),
        scout_seed_transpilers=(0, 1, 2, 3, 4),
        noise_mode="backend_scheduled",
        mitigation_config=normalize_mitigation_config(
            {
                "mode": "readout",
                "local_readout_strategy": "mthree",
                "local_gate_twirling": bool(getattr(args, "local_gate_twirling", False)),
            }
        ),
        symmetry_mitigation_config=normalize_symmetry_mitigation_config({"mode": "off"}),
        rank_policy="delta_mean_then_two_qubit_then_depth_then_size",
    )
    fixed_scaffold_saved_theta_mitigation_matrix_enabled = bool(
        include_fixed_scaffold_saved_theta_mitigation_matrix
    )
    if fixed_scaffold_saved_theta_mitigation_matrix_enabled and str(source_cfg.mode) != "imported_artifact":
        raise ValueError(
            "fixed scaffold saved-theta mitigation matrix requires an imported artifact source; pass "
            "--fixed-final-state-json or enable the default Marrakesh/Heron runtime candidate."
        )
    if fixed_scaffold_saved_theta_mitigation_matrix_enabled:
        if bool(getattr(args, "include_fixed_scaffold_runtime_dd_probe", False)):
            raise ValueError(
                "fixed scaffold saved-theta mitigation matrix does not support the legacy Runtime DD probe flag."
            )
        if bool(getattr(args, "include_fixed_scaffold_runtime_final_zne_audit", False)):
            raise ValueError(
                "fixed scaffold saved-theta mitigation matrix does not support the legacy Runtime final ZNE audit flag."
            )
    fixed_scaffold_matrix_zne_scales = tuple(
        float(x)
        for x in normalize_mitigation_config(
            {
                "mode": "zne",
                "zne_scales": getattr(args, "fixed_scaffold_matrix_zne_scales", "1.0,3.0,5.0"),
            }
        ).get("zne_scales", [])
    )
    fixed_scaffold_matrix_compile_presets = _parse_fixed_scaffold_matrix_compile_presets(
        getattr(args, "fixed_scaffold_matrix_compile_presets", None)
    )
    if not fixed_scaffold_matrix_compile_presets:
        fixed_scaffold_matrix_compile_presets = (
            {
                "label": "opt1_seed4",
                "transpile_optimization_level": 1,
                "seed_transpiler": 4,
            },
            {
                "label": "opt2_seed0",
                "transpile_optimization_level": 2,
                "seed_transpiler": 0,
            },
        )
    fixed_scaffold_matrix_suppression_labels = (
        "readout_plus_gate_twirling",
        "readout_plus_local_dd",
        "readout_plus_gate_twirling_plus_local_dd",
    )
    fixed_scaffold_matrix_selected_cells = _parse_fixed_scaffold_matrix_selected_cells(
        getattr(args, "fixed_scaffold_matrix_selected_cells", None)
    )
    fixed_scaffold_matrix_base_mitigation_mode = str(
        getattr(args, "fixed_scaffold_matrix_base_mitigation_mode", "readout")
    ).strip().lower()
    if fixed_scaffold_matrix_selected_cells:
        planned_labels = set(
            _planned_fixed_scaffold_matrix_cell_labels(
                compile_presets=fixed_scaffold_matrix_compile_presets,
                suppression_labels=fixed_scaffold_matrix_suppression_labels,
            )
        )
        unsupported_selected_cells = [
            str(label)
            for label in fixed_scaffold_matrix_selected_cells
            if str(label) not in planned_labels
        ]
        if unsupported_selected_cells:
            raise ValueError(
                "Unsupported fixed scaffold matrix selected cells: "
                f"{unsupported_selected_cells}"
            )
    fixed_scaffold_saved_theta_mitigation_matrix_cfg = FixedScaffoldSavedThetaMitigationMatrixConfig(
        enabled=bool(fixed_scaffold_saved_theta_mitigation_matrix_enabled),
        subject_kind="hh_marrakesh_gate_pruned_6term_drop_eyezee_v1",
        noise_mode="backend_scheduled",
        compile_presets=fixed_scaffold_matrix_compile_presets,
        zne_scales=fixed_scaffold_matrix_zne_scales,
        suppression_labels=fixed_scaffold_matrix_suppression_labels,
        selected_cells=fixed_scaffold_matrix_selected_cells,
        mitigation_config_base=_build_fixed_scaffold_matrix_base_mitigation_config(
            fixed_scaffold_matrix_base_mitigation_mode
        ),
        symmetry_mitigation_config=normalize_symmetry_mitigation_config(
            dict(noise_cfg.symmetry_mitigation_config)
        ),
        rank_policy="delta_mean_then_two_qubit_then_depth_then_size",
        raw_artifact_root=None,
    )
    fixed_scaffold_attribution_enabled = bool(include_fixed_scaffold_noise_attribution)
    if fixed_scaffold_attribution_enabled and str(source_cfg.mode) != "imported_artifact":
        raise ValueError(
            "fixed scaffold noise attribution requires an imported artifact source; pass "
            "--fixed-final-state-json or enable the default Nighthawk gate-pruned scaffold."
        )
    fixed_scaffold_attribution_cfg = FixedScaffoldNoiseAttributionConfig(
        enabled=bool(fixed_scaffold_attribution_enabled),
        subject_kind="hh_nighthawk_gate_pruned_7term_v1",
        slices=tuple(BACKEND_SCHEDULED_ATTRIBUTION_SLICES),
        noise_mode="backend_scheduled",
        mitigation_config=normalize_mitigation_config({"mode": "none"}),
        symmetry_mitigation_config=normalize_symmetry_mitigation_config({"mode": "off"}),
    )
    if bool(fixed_scaffold_replay_enabled):
        noise_cfg = replace(
            noise_cfg,
            mitigation_config=normalize_mitigation_config(
                {
                    **dict(noise_cfg.mitigation_config),
                    "dd_sequence": None,
                }
            ),
        )
    return StagedHHNoiseConfig(
        staged=staged_cfg,
        noise=noise_cfg,
        source=source_cfg,
        fixed_lean_replay=fixed_lean_replay_cfg,
        fixed_lean_attribution=fixed_lean_attribution_cfg,
        fixed_lean_compile_control_scout=fixed_lean_compile_control_scout_cfg,
        fixed_scaffold_replay=fixed_scaffold_replay_cfg,
        fixed_scaffold_runtime_energy_only=fixed_scaffold_runtime_energy_only_cfg,
        fixed_scaffold_runtime_raw_baseline=fixed_scaffold_runtime_raw_baseline_cfg,
        fixed_scaffold_compile_control_scout=fixed_scaffold_compile_control_scout_cfg,
        fixed_scaffold_saved_theta_mitigation_matrix=fixed_scaffold_saved_theta_mitigation_matrix_cfg,
        fixed_scaffold_attribution=fixed_scaffold_attribution_cfg,
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


def _controller_oracle_config_from_noise_cfg(noise_cfg: NoiseConfig) -> OracleConfig:
    resolved_mode = str(noise_cfg.controller_noise_mode or "").strip().lower()
    if resolved_mode in {"", "none"}:
        raise ValueError("checkpoint controller oracle_v1 requires a resolved controller noise mode.")
    return OracleConfig(
        noise_mode=str(resolved_mode),
        shots=int(noise_cfg.shots),
        seed=int(noise_cfg.seed),
        oracle_repeats=int(noise_cfg.oracle_repeats),
        oracle_aggregate=str(noise_cfg.oracle_aggregate),
        backend_name=(None if noise_cfg.backend_name is None else str(noise_cfg.backend_name)),
        use_fake_backend=bool(noise_cfg.use_fake_backend),
        allow_aer_fallback=bool(noise_cfg.allow_aer_fallback),
        omp_shm_workaround=bool(noise_cfg.omp_shm_workaround),
        mitigation=dict(noise_cfg.mitigation_config),
        symmetry_mitigation=dict(noise_cfg.symmetry_mitigation_config),
    )


def _write_controller_progress(path: Path | None, payload: Mapping[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(base_wf._jsonable(dict(payload)), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _read_controller_progress(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"read_error": f"{type(exc).__name__}: {exc}"}
    return data if isinstance(data, dict) else {"raw_payload_type": type(data).__name__}


def _read_controller_partial_payload(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"read_error": f"{type(exc).__name__}: {exc}"}
    return data if isinstance(data, dict) else {"raw_payload_type": type(data).__name__}


def _adaptive_realtime_checkpoint_env_blocked_payload(
    *,
    mode: str,
    scaffold_acceptance_payload: Mapping[str, Any],
    decision_noise_mode: str | None,
    reason: str,
    exitcode: int | None = None,
    error: str | None = None,
    progress_path: Path | None = None,
    progress_snapshot: Mapping[str, Any] | None = None,
    partial_payload_path: Path | None = None,
    partial_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    partial_data = {} if partial_payload is None else dict(partial_payload)
    partial_trajectory = list(partial_data.get("trajectory", []) or [])
    partial_ledger = list(partial_data.get("ledger", []) or [])
    partial_reference = dict(partial_data.get("reference", {}) or {})
    partial_summary = dict(partial_data.get("summary", {}) or {})
    executed_backends = sorted({str(row.get("decision_backend", "exact")) for row in partial_ledger})
    append_count = int(sum(1 for row in partial_ledger if str(row.get("action_kind")) == "append_candidate"))
    stay_count = int(sum(1 for row in partial_ledger if str(row.get("action_kind")) == "stay"))
    exact_decision_checkpoints = int(sum(1 for row in partial_ledger if str(row.get("decision_backend")) == "exact"))
    oracle_decision_checkpoints = int(sum(1 for row in partial_ledger if str(row.get("decision_backend")) == "oracle"))
    oracle_attempted_checkpoints = int(sum(1 for row in partial_ledger if bool(row.get("oracle_attempted", False))))
    degraded_checkpoints = int(sum(1 for row in partial_ledger if row.get("degraded_reason") not in {None, ""}))
    final_logical_block_count = int(
        partial_summary.get(
            "final_logical_block_count",
            (partial_ledger[-1].get("logical_block_count_after", 0) if partial_ledger else 0),
        )
    )
    final_runtime_parameter_count = int(
        partial_summary.get(
            "final_runtime_parameter_count",
            (partial_ledger[-1].get("runtime_parameter_count_after", 0) if partial_ledger else 0),
        )
    )
    final_fidelity_exact = float(
        partial_summary.get(
            "final_fidelity_exact",
            (partial_ledger[-1].get("fidelity_exact", float("nan")) if partial_ledger else float("nan")),
        )
    )
    final_abs_energy_total_error = float(
        partial_summary.get(
            "final_abs_energy_total_error",
            (partial_ledger[-1].get("abs_energy_total_error", float("nan")) if partial_ledger else float("nan")),
        )
    )
    summary = {
        "mode": str(mode),
        "status": "env_blocked",
        "requested_decision_backend": ("oracle" if str(mode) == "oracle_v1" else "exact"),
        "decision_backend": "env_blocked",
        "executed_decision_backends": list(executed_backends),
        "decision_noise_mode": (None if decision_noise_mode is None else str(decision_noise_mode)),
        "oracle_estimate_kind": (
            None if decision_noise_mode is None else f"oracle_{str(decision_noise_mode).strip().lower()}"
        ),
        "append_count": int(partial_summary.get("append_count", append_count)),
        "stay_count": int(partial_summary.get("stay_count", stay_count)),
        "exact_decision_checkpoints": int(partial_summary.get("exact_decision_checkpoints", exact_decision_checkpoints)),
        "oracle_decision_checkpoints": int(partial_summary.get("oracle_decision_checkpoints", oracle_decision_checkpoints)),
        "oracle_attempted_checkpoints": int(partial_summary.get("oracle_attempted_checkpoints", oracle_attempted_checkpoints)),
        "degraded_checkpoints": int(partial_summary.get("degraded_checkpoints", degraded_checkpoints)),
        "final_logical_block_count": int(final_logical_block_count),
        "final_runtime_parameter_count": int(final_runtime_parameter_count),
        "final_fidelity_exact": float(final_fidelity_exact),
        "final_abs_energy_total_error": float(final_abs_energy_total_error),
        "planning_audit": dict(partial_summary.get("planning_audit", {}) or {}),
        "temporal_measurement_ledger": dict(partial_summary.get("temporal_measurement_ledger", {}) or {}),
        "env_blocked": True,
        "env_blocked_reason": str(reason),
        "exitcode": (None if exitcode is None else int(exitcode)),
        "error": (None if error is None else str(error)),
        "progress_path": _repo_relative_str(progress_path),
        "partial_payload_path": _repo_relative_str(partial_payload_path),
        "last_progress": (None if progress_snapshot is None else base_wf._jsonable(dict(progress_snapshot))),
    }
    return {
        "mode": str(mode),
        "status": "env_blocked",
        "reason": str(reason),
        "exitcode": (None if exitcode is None else int(exitcode)),
        "error": (None if error is None else str(error)),
        "scaffold_acceptance": dict(scaffold_acceptance_payload),
        "reference": partial_reference,
        "trajectory": partial_trajectory,
        "ledger": partial_ledger,
        "summary": summary,
        "progress_path": _repo_relative_str(progress_path),
        "partial_payload_path": _repo_relative_str(partial_payload_path),
        "progress_snapshot": (None if progress_snapshot is None else base_wf._jsonable(dict(progress_snapshot))),
    }


def _adaptive_realtime_checkpoint_oracle_worker_entry(queue: Any, kwargs: dict[str, Any]) -> None:
    progress_path_raw = kwargs.get("progress_path")
    progress_path = None if progress_path_raw in {None, ""} else Path(progress_path_raw)
    _write_controller_progress(
        progress_path,
        {
            "status": "worker_start",
            "stage": "worker_start",
            "mode": str(getattr(kwargs.get("cfg", None), "mode", "unknown")),
        },
    )
    try:
        controller = RealtimeCheckpointController(**kwargs)
        _write_controller_progress(
            progress_path,
            {
                "status": "controller_initialized",
                "stage": "controller_initialized",
                "mode": str(getattr(kwargs.get("cfg", None), "mode", "unknown")),
            },
        )
        artifacts = controller.run()
        _write_controller_progress(
            progress_path,
            {
                "status": "completed",
                "stage": "worker_complete",
                "mode": str(getattr(kwargs.get("cfg", None), "mode", "unknown")),
                "trajectory_points": int(len(artifacts.trajectory)),
                "ledger_entries": int(len(artifacts.ledger)),
                "summary": dict(artifacts.summary),
            },
        )
        queue.put(
            {
                "ok": True,
                "payload": {
                    "trajectory": list(artifacts.trajectory),
                    "ledger": list(artifacts.ledger),
                    "summary": dict(artifacts.summary),
                    "reference": dict(artifacts.reference),
                },
            }
        )
    except Exception as exc:  # pragma: no cover - subprocess fault path
        _write_controller_progress(
            progress_path,
            {
                "status": "worker_exception",
                "stage": "worker_exception",
                "mode": str(getattr(kwargs.get("cfg", None), "mode", "unknown")),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _run_adaptive_realtime_checkpoint_profile_noisy_isolated(
    *,
    controller_kwargs: dict[str, Any],
    mode: str,
    scaffold_acceptance_payload: Mapping[str, Any],
    decision_noise_mode: str | None,
    timeout_s: int,
) -> dict[str, Any]:
    progress_path_raw = controller_kwargs.get("progress_path")
    progress_path = None if progress_path_raw in {None, ""} else Path(progress_path_raw)
    partial_payload_path_raw = controller_kwargs.get("partial_payload_path")
    partial_payload_path = None if partial_payload_path_raw in {None, ""} else Path(partial_payload_path_raw)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_adaptive_realtime_checkpoint_oracle_worker_entry,
        args=(queue, controller_kwargs),
        daemon=False,
    )
    try:
        proc.start()
    except Exception as exc:
        return _adaptive_realtime_checkpoint_env_blocked_payload(
            mode=str(mode),
            scaffold_acceptance_payload=scaffold_acceptance_payload,
            decision_noise_mode=decision_noise_mode,
            reason="subprocess_start_failed",
            error=f"{type(exc).__name__}: {exc}",
            progress_path=progress_path,
            progress_snapshot=_read_controller_progress(progress_path),
            partial_payload_path=partial_payload_path,
            partial_payload=_read_controller_partial_payload(partial_payload_path),
        )
    proc.join(timeout=float(timeout_s))

    if proc.is_alive():
        proc.terminate()
        proc.join(5.0)
        return _adaptive_realtime_checkpoint_env_blocked_payload(
            mode=str(mode),
            scaffold_acceptance_payload=scaffold_acceptance_payload,
            decision_noise_mode=decision_noise_mode,
            reason=f"timeout_after_{int(timeout_s)}s",
            exitcode=proc.exitcode,
            progress_path=progress_path,
            progress_snapshot=_read_controller_progress(progress_path),
            partial_payload_path=partial_payload_path,
            partial_payload=_read_controller_partial_payload(partial_payload_path),
        )

    if int(proc.exitcode or 0) != 0:
        return _adaptive_realtime_checkpoint_env_blocked_payload(
            mode=str(mode),
            scaffold_acceptance_payload=scaffold_acceptance_payload,
            decision_noise_mode=decision_noise_mode,
            reason="subprocess_nonzero_exit",
            exitcode=int(proc.exitcode or 0),
            progress_path=progress_path,
            progress_snapshot=_read_controller_progress(progress_path),
            partial_payload_path=partial_payload_path,
            partial_payload=_read_controller_partial_payload(partial_payload_path),
        )

    if queue.empty():
        return _adaptive_realtime_checkpoint_env_blocked_payload(
            mode=str(mode),
            scaffold_acceptance_payload=scaffold_acceptance_payload,
            decision_noise_mode=decision_noise_mode,
            reason="subprocess_completed_without_payload",
            exitcode=int(proc.exitcode or 0),
            progress_path=progress_path,
            progress_snapshot=_read_controller_progress(progress_path),
            partial_payload_path=partial_payload_path,
            partial_payload=_read_controller_partial_payload(partial_payload_path),
        )

    msg = queue.get()
    if not bool(msg.get("ok", False)):
        return _adaptive_realtime_checkpoint_env_blocked_payload(
            mode=str(mode),
            scaffold_acceptance_payload=scaffold_acceptance_payload,
            decision_noise_mode=decision_noise_mode,
            reason="worker_exception",
            exitcode=int(proc.exitcode or 0),
            error=str(msg.get("error", "unknown")),
            progress_path=progress_path,
            progress_snapshot=_read_controller_progress(progress_path),
            partial_payload_path=partial_payload_path,
            partial_payload=_read_controller_partial_payload(partial_payload_path),
        )

    payload = dict(msg.get("payload", {}))
    summary = dict(payload.get("summary", {}) or {})
    progress_snapshot = _read_controller_progress(progress_path)
    summary.setdefault("progress_path", _repo_relative_str(progress_path))
    summary.setdefault("partial_payload_path", _repo_relative_str(partial_payload_path))
    summary.setdefault(
        "last_progress",
        None if progress_snapshot is None else base_wf._jsonable(dict(progress_snapshot)),
    )
    return {
        "mode": str(mode),
        "status": str(summary.get("status", "completed")),
        "scaffold_acceptance": dict(scaffold_acceptance_payload),
        "reference": dict(payload.get("reference", {}) or {}),
        "trajectory": list(payload.get("trajectory", []) or []),
        "ledger": list(payload.get("ledger", []) or []),
        "summary": summary,
        "progress_path": _repo_relative_str(progress_path),
        "partial_payload_path": _repo_relative_str(partial_payload_path),
        "progress_snapshot": (
            None if progress_snapshot is None else base_wf._jsonable(dict(progress_snapshot))
        ),
    }


def _prepare_checkpoint_controller_inputs(
    stage_result: base_wf.StageExecutionResult,
    staged_cfg: base_wf.StagedHHConfig,
) -> tuple[Any, Any, Sequence[float]]:
    replay_cfg = base_wf._build_replay_run_config(staged_cfg)
    replay_context = base_wf.replay_mod.build_replay_scaffold_context(
        replay_cfg,
        h_poly=stage_result.h_poly,
    )
    acceptance = base_wf.validate_scaffold_acceptance(replay_context.payload_in)
    if not bool(acceptance.accepted):
        raise ValueError(
            f"checkpoint controller rejected scaffold ownership: {acceptance.reason} ({acceptance.source_kind})."
        )
    best_state = stage_result.replay_payload.get("best_state", {})
    if not isinstance(best_state, Mapping):
        raise ValueError("Replay payload missing best_state block.")
    best_theta = best_state.get("best_theta", None)
    if not isinstance(best_theta, Sequence):
        raise ValueError(
            "Replay payload missing best_state.best_theta; checkpoint controller requires replay runtime parameters."
        )
    return replay_context, acceptance, best_theta


def _run_adaptive_realtime_checkpoint_profile_local(
    *,
    stage_result: base_wf.StageExecutionResult,
    cfg: StagedHHNoiseConfig,
    replay_context: Any,
    acceptance: Any,
    best_theta: Sequence[float],
) -> dict[str, Any]:
    logs_dir = Path(cfg.staged.artifacts.output_json).parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    progress_path = logs_dir / "controller_progress.json"
    partial_payload_path = logs_dir / "controller_partial.json"
    if progress_path.exists():
        progress_path.unlink()
    if partial_payload_path.exists():
        partial_payload_path.unlink()

    controller_timeout_s = (
        None
        if cfg.noise.controller_timeout_s is None
        else int(cfg.noise.controller_timeout_s)
    )
    controller = RealtimeCheckpointController(
        cfg=cfg.staged.realtime_checkpoint,
        replay_context=replay_context,
        h_poly=stage_result.h_poly,
        hmat=np.asarray(stage_result.hmat, dtype=complex),
        psi_initial=np.asarray(stage_result.psi_final, dtype=complex).reshape(-1),
        best_theta=np.asarray(best_theta, dtype=float).reshape(-1),
        allow_repeats=bool(cfg.staged.adapt.allow_repeats),
        t_final=float(cfg.staged.dynamics.t_final),
        num_times=int(cfg.staged.dynamics.num_times),
        drive_config=base_wf._controller_drive_config_from_cfg(cfg.staged),
        oracle_base_config=(
            None
            if cfg.noise.controller_noise_mode is None
            else _controller_oracle_config_from_noise_cfg(cfg.noise)
        ),
        wallclock_cap_s=controller_timeout_s,
        progress_path=progress_path,
        partial_payload_path=partial_payload_path,
        progress_every_s=float(cfg.noise.controller_progress_every_s),
    )
    artifacts = controller.run()
    progress_snapshot = _read_controller_progress(progress_path)
    summary = dict(artifacts.summary)
    summary.setdefault("progress_path", _repo_relative_str(progress_path))
    summary.setdefault("partial_payload_path", _repo_relative_str(partial_payload_path))
    summary.setdefault(
        "last_progress",
        None if progress_snapshot is None else base_wf._jsonable(dict(progress_snapshot)),
    )
    return {
        "mode": str(cfg.staged.realtime_checkpoint.mode),
        "status": str(summary.get("status", "completed")),
        "scaffold_acceptance": base_wf._jsonable(asdict(acceptance)),
        "reference": dict(artifacts.reference),
        "trajectory": list(artifacts.trajectory),
        "ledger": list(artifacts.ledger),
        "summary": summary,
        "progress_path": _repo_relative_str(progress_path),
        "partial_payload_path": _repo_relative_str(partial_payload_path),
        "progress_snapshot": (
            None if progress_snapshot is None else base_wf._jsonable(dict(progress_snapshot))
        ),
    }


def run_adaptive_realtime_checkpoint_profile_noisy(
    stage_result: base_wf.StageExecutionResult,
    cfg: StagedHHNoiseConfig,
) -> dict[str, Any] | None:
    mode = str(cfg.staged.realtime_checkpoint.mode)
    if mode not in {"off", "exact_v1", "oracle_v1"}:
        raise ValueError(f"Unsupported checkpoint controller mode {mode!r} for staged noise workflow.")
    replay_context, acceptance, best_theta = _prepare_checkpoint_controller_inputs(stage_result, cfg.staged)
    if mode in {"off", "exact_v1"}:
        return _run_adaptive_realtime_checkpoint_profile_local(
            stage_result=stage_result,
            cfg=cfg,
            replay_context=replay_context,
            acceptance=acceptance,
            best_theta=best_theta,
        )
    scaffold_acceptance_payload = base_wf._jsonable(asdict(acceptance))
    logs_dir = Path(cfg.staged.artifacts.output_json).parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    progress_path = logs_dir / "controller_progress.json"
    partial_payload_path = logs_dir / "controller_partial.json"
    if progress_path.exists():
        progress_path.unlink()
    if partial_payload_path.exists():
        partial_payload_path.unlink()
    controller_timeout_s = int(
        cfg.noise.noisy_mode_timeout_s
        if cfg.noise.controller_timeout_s is None
        else cfg.noise.controller_timeout_s
    )
    controller_kwargs = {
        "cfg": cfg.staged.realtime_checkpoint,
        "replay_context": replay_context,
        "h_poly": stage_result.h_poly,
        "hmat": np.asarray(stage_result.hmat, dtype=complex),
        "psi_initial": np.asarray(stage_result.psi_final, dtype=complex).reshape(-1),
        "best_theta": np.asarray(best_theta, dtype=float).reshape(-1),
        "allow_repeats": bool(cfg.staged.adapt.allow_repeats),
        "t_final": float(cfg.staged.dynamics.t_final),
        "num_times": int(cfg.staged.dynamics.num_times),
        "drive_config": base_wf._controller_drive_config_from_cfg(cfg.staged),
        "oracle_base_config": _controller_oracle_config_from_noise_cfg(cfg.noise),
        "wallclock_cap_s": int(controller_timeout_s),
        "progress_path": progress_path,
        "partial_payload_path": partial_payload_path,
        "progress_every_s": float(cfg.noise.controller_progress_every_s),
    }
    return _run_adaptive_realtime_checkpoint_profile_noisy_isolated(
        controller_kwargs=controller_kwargs,
        mode=str(cfg.staged.realtime_checkpoint.mode),
        scaffold_acceptance_payload=scaffold_acceptance_payload,
        decision_noise_mode=cfg.noise.controller_noise_mode,
        timeout_s=int(controller_timeout_s),
    )


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


def _run_imported_ansatz_input_state_audit_mode(
    *,
    source_cfg: StagedNoiseSourceConfig,
    noise_cfg: NoiseConfig,
    mode: str,
    seed_transpiler: int | None = None,
    transpile_optimization_level: int | None = None,
    compile_request_source: str | None = None,
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
        "seed_transpiler": (None if seed_transpiler is None else int(seed_transpiler)),
        "transpile_optimization_level": (
            None if transpile_optimization_level is None else int(transpile_optimization_level)
        ),
        "compile_request_source": (
            None if compile_request_source in {None, ""} else str(compile_request_source)
        ),
    }
    return noise_report._run_imported_ansatz_input_state_audit_mode_isolated(
        kwargs=kwargs,
        timeout_s=int(noise_cfg.noisy_mode_timeout_s),
    )


def _run_imported_full_circuit_audit_mode(
    *,
    source_cfg: StagedNoiseSourceConfig,
    noise_cfg: NoiseConfig,
    mode: str,
    seed_transpiler: int | None = None,
    transpile_optimization_level: int | None = None,
    compile_request_source: str | None = None,
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
        "seed_transpiler": (None if seed_transpiler is None else int(seed_transpiler)),
        "transpile_optimization_level": (
            None if transpile_optimization_level is None else int(transpile_optimization_level)
        ),
        "compile_request_source": (
            None if compile_request_source in {None, ""} else str(compile_request_source)
        ),
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


def _run_fixed_lean_compile_control_scout_mode(
    *,
    source_cfg: StagedNoiseSourceConfig,
    noise_cfg: NoiseConfig,
    scout_cfg: FixedLeanCompileControlScoutConfig,
) -> dict[str, Any]:
    if source_cfg.resolved_json is None:
        return {"success": False, "available": False, "reason": "missing_import_source"}
    if not bool(scout_cfg.enabled):
        return {"success": False, "available": False, "reason": "fixed_lean_compile_control_scout_disabled"}
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
        "mitigation_config": dict(scout_cfg.mitigation_config),
        "symmetry_mitigation_config": dict(scout_cfg.symmetry_mitigation_config),
        "baseline_transpile_optimization_level": int(scout_cfg.baseline_transpile_optimization_level),
        "baseline_seed_transpiler": int(scout_cfg.baseline_seed_transpiler),
        "scout_transpile_optimization_levels": list(scout_cfg.scout_transpile_optimization_levels),
        "scout_seed_transpilers": list(scout_cfg.scout_seed_transpilers),
        "rank_policy": str(scout_cfg.rank_policy),
    }
    return noise_report._run_imported_fixed_lean_compile_control_scout_mode_isolated(
        kwargs=kwargs,
        timeout_s=int(noise_cfg.noisy_mode_timeout_s),
    )


def _run_fixed_scaffold_compile_control_scout_mode(
    *,
    source_cfg: StagedNoiseSourceConfig,
    noise_cfg: NoiseConfig,
    scout_cfg: FixedScaffoldCompileControlScoutConfig,
) -> dict[str, Any]:
    if source_cfg.resolved_json is None:
        return {"success": False, "available": False, "reason": "missing_import_source"}
    if not bool(scout_cfg.enabled):
        return {
            "success": False,
            "available": False,
            "reason": "fixed_scaffold_compile_control_scout_disabled",
        }
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
        "mitigation_config": dict(scout_cfg.mitigation_config),
        "symmetry_mitigation_config": dict(scout_cfg.symmetry_mitigation_config),
        "baseline_transpile_optimization_level": int(scout_cfg.baseline_transpile_optimization_level),
        "baseline_seed_transpiler": int(scout_cfg.baseline_seed_transpiler),
        "scout_transpile_optimization_levels": list(scout_cfg.scout_transpile_optimization_levels),
        "scout_seed_transpilers": list(scout_cfg.scout_seed_transpilers),
        "rank_policy": str(scout_cfg.rank_policy),
    }
    return noise_report._run_imported_fixed_scaffold_compile_control_scout_mode_isolated(
        kwargs=kwargs,
        timeout_s=int(noise_cfg.noisy_mode_timeout_s),
    )


def _run_fixed_scaffold_saved_theta_mitigation_matrix_mode(
    *,
    source_cfg: StagedNoiseSourceConfig,
    noise_cfg: NoiseConfig,
    matrix_cfg: FixedScaffoldSavedThetaMitigationMatrixConfig,
) -> dict[str, Any]:
    if source_cfg.resolved_json is None:
        return {"success": False, "available": False, "reason": "missing_import_source"}
    if not bool(matrix_cfg.enabled):
        return {
            "success": False,
            "available": False,
            "reason": "fixed_scaffold_saved_theta_mitigation_matrix_disabled",
        }
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
        "noise_mode": str(matrix_cfg.noise_mode),
        "compile_presets": [dict(x) for x in matrix_cfg.compile_presets],
        "zne_scales": [float(x) for x in matrix_cfg.zne_scales],
        "suppression_labels": [str(x) for x in matrix_cfg.suppression_labels],
        "selected_cells": [str(x) for x in matrix_cfg.selected_cells],
        "mitigation_config_base": dict(matrix_cfg.mitigation_config_base),
        "symmetry_mitigation_config": dict(matrix_cfg.symmetry_mitigation_config),
        "rank_policy": str(matrix_cfg.rank_policy),
        "raw_artifact_root": matrix_cfg.raw_artifact_root,
    }
    return noise_report._run_imported_fixed_scaffold_saved_theta_mitigation_matrix_mode_isolated(
        kwargs=kwargs,
        timeout_s=int(noise_cfg.noisy_mode_timeout_s),
    )


def _run_fixed_scaffold_runtime_energy_only_mode(
    *,
    source_cfg: StagedNoiseSourceConfig,
    noise_cfg: NoiseConfig,
    runtime_cfg: FixedScaffoldRuntimeEnergyOnlyConfig,
) -> dict[str, Any]:
    if source_cfg.resolved_json is None:
        return {"success": False, "available": False, "reason": "missing_import_source"}
    if not bool(runtime_cfg.enabled):
        return {
            "success": False,
            "available": False,
            "reason": "fixed_scaffold_runtime_energy_only_disabled",
        }
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
        "mitigation_config": dict(runtime_cfg.mitigation_config),
        "symmetry_mitigation_config": dict(runtime_cfg.symmetry_mitigation_config),
        "runtime_profile_config": dict(runtime_cfg.runtime_profile_config),
        "runtime_session_config": dict(runtime_cfg.runtime_session_config),
        "transpile_optimization_level": int(runtime_cfg.transpile_optimization_level),
        "seed_transpiler": runtime_cfg.seed_transpiler,
        "include_dd_probe": bool(runtime_cfg.include_dd_probe),
        "include_final_zne_audit": bool(runtime_cfg.include_final_zne_audit),
    }
    return noise_report._run_imported_fixed_scaffold_runtime_energy_only_mode_isolated(
        kwargs=kwargs,
        timeout_s=int(noise_cfg.noisy_mode_timeout_s),
    )


def _run_fixed_scaffold_runtime_raw_baseline_mode(
    *,
    source_cfg: StagedNoiseSourceConfig,
    noise_cfg: NoiseConfig,
    runtime_cfg: FixedScaffoldRuntimeRawBaselineConfig,
) -> dict[str, Any]:
    if source_cfg.resolved_json is None:
        return {"success": False, "available": False, "reason": "missing_import_source"}
    if not bool(runtime_cfg.enabled):
        return {
            "success": False,
            "available": False,
            "reason": "fixed_scaffold_runtime_raw_baseline_disabled",
        }
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
        "mitigation_config": dict(runtime_cfg.mitigation_config),
        "symmetry_mitigation_config": dict(runtime_cfg.symmetry_mitigation_config),
        "runtime_profile_config": dict(runtime_cfg.runtime_profile_config),
        "runtime_session_config": dict(runtime_cfg.runtime_session_config),
        "transpile_optimization_level": int(runtime_cfg.transpile_optimization_level),
        "seed_transpiler": runtime_cfg.seed_transpiler,
        "raw_transport": str(runtime_cfg.raw_transport),
        "raw_store_memory": bool(runtime_cfg.raw_store_memory),
        "raw_artifact_path": runtime_cfg.raw_artifact_path,
    }
    return noise_report._run_imported_fixed_scaffold_runtime_raw_baseline_mode_isolated(
        kwargs=kwargs,
        timeout_s=int(noise_cfg.noisy_mode_timeout_s),
    )


def _run_fixed_scaffold_noisy_replay_mode(
    *,
    source_cfg: StagedNoiseSourceConfig,
    noise_cfg: NoiseConfig,
    fixed_cfg: FixedScaffoldNoisyReplayConfig,
) -> dict[str, Any]:
    if source_cfg.resolved_json is None:
        return {"success": False, "available": False, "reason": "missing_import_source"}
    if not bool(fixed_cfg.enabled):
        return {"success": False, "available": False, "reason": "fixed_scaffold_noisy_replay_disabled"}
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
        "progress_every_s": float(fixed_cfg.progress_every_s),
        "spsa_a": float(fixed_cfg.spsa_a),
        "spsa_c": float(fixed_cfg.spsa_c),
        "spsa_alpha": float(fixed_cfg.spsa_alpha),
        "spsa_gamma": float(fixed_cfg.spsa_gamma),
        "spsa_A": float(fixed_cfg.spsa_A),
        "spsa_avg_last": int(fixed_cfg.spsa_avg_last),
        "spsa_eval_repeats": int(fixed_cfg.spsa_eval_repeats),
        "spsa_eval_agg": str(fixed_cfg.spsa_eval_agg),
        "local_dd_probe_sequence": fixed_cfg.local_dd_probe_sequence,
        "mitigation_config": dict(fixed_cfg.mitigation_config),
        "symmetry_mitigation_config": dict(fixed_cfg.symmetry_mitigation_config),
        "transpile_optimization_level": int(fixed_cfg.transpile_optimization_level),
        "seed_transpiler": fixed_cfg.seed_transpiler,
    }
    return noise_report._run_imported_fixed_scaffold_noisy_replay_mode_isolated(
        kwargs=kwargs,
        timeout_s=int(noise_cfg.noisy_mode_timeout_s),
    )


def _run_fixed_scaffold_noise_attribution_mode(
    *,
    source_cfg: StagedNoiseSourceConfig,
    noise_cfg: NoiseConfig,
    attribution_cfg: FixedScaffoldNoiseAttributionConfig,
) -> dict[str, Any]:
    if source_cfg.resolved_json is None:
        return {"success": False, "available": False, "reason": "missing_import_source"}
    if not bool(attribution_cfg.enabled):
        return {"success": False, "available": False, "reason": "fixed_scaffold_noise_attribution_disabled"}
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
    return noise_report._run_imported_fixed_scaffold_noise_attribution_mode_isolated(
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
    imported_ansatz_input = (
        payload.get("imported_ansatz_input_state_audit", {}) if isinstance(payload, Mapping) else {}
    )
    imported_full = payload.get("full_circuit_import_audit", {}) if isinstance(payload, Mapping) else {}
    fixed_lean_replay = payload.get("fixed_lean_scaffold_noisy_replay", {}) if isinstance(payload, Mapping) else {}
    fixed_lean_attribution = payload.get("fixed_lean_noise_attribution", {}) if isinstance(payload, Mapping) else {}
    fixed_lean_compile_control_scout = (
        payload.get("fixed_lean_compile_control_scout", {}) if isinstance(payload, Mapping) else {}
    )
    fixed_scaffold_runtime_energy_only = (
        payload.get("fixed_scaffold_runtime_energy_only", {}) if isinstance(payload, Mapping) else {}
    )
    fixed_scaffold_runtime_raw_baseline = (
        payload.get("fixed_scaffold_runtime_raw_baseline", {}) if isinstance(payload, Mapping) else {}
    )
    fixed_scaffold_replay = payload.get("fixed_scaffold_noisy_replay", {}) if isinstance(payload, Mapping) else {}
    fixed_scaffold_compile_control_scout = (
        payload.get("fixed_scaffold_compile_control_scout", {}) if isinstance(payload, Mapping) else {}
    )
    fixed_scaffold_saved_theta_mitigation_matrix = (
        payload.get("fixed_scaffold_saved_theta_mitigation_matrix", {})
        if isinstance(payload, Mapping)
        else {}
    )
    fixed_scaffold_attribution = payload.get("fixed_scaffold_noise_attribution", {}) if isinstance(payload, Mapping) else {}
    adaptive_rt = payload.get("adaptive_realtime_checkpoint", {}) if isinstance(payload, Mapping) else {}
    imported_prepared_modes = imported_prepared.get("modes", {}) if isinstance(imported_prepared, Mapping) else {}
    imported_ansatz_input_modes = (
        imported_ansatz_input.get("modes", {}) if isinstance(imported_ansatz_input, Mapping) else {}
    )
    imported_full_modes = imported_full.get("modes", {}) if isinstance(imported_full, Mapping) else {}
    imported_prepared_completed = sum(
        1 for rec in imported_prepared_modes.values() if isinstance(rec, Mapping) and bool(rec.get("success", False))
    )
    imported_ansatz_input_completed = sum(
        1
        for rec in imported_ansatz_input_modes.values()
        if isinstance(rec, Mapping) and bool(rec.get("success", False))
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
    fixed_lean_compile_control_success = bool(
        isinstance(fixed_lean_compile_control_scout, Mapping)
        and fixed_lean_compile_control_scout.get("success", False)
    )
    if (
        isinstance(fixed_lean_compile_control_scout, Mapping)
        and not fixed_lean_compile_control_success
        and bool(fixed_lean_compile_control_scout)
    ):
        failures.append(
            "fixed_lean_compile_control_scout:"
            + str(
                fixed_lean_compile_control_scout.get(
                    "reason",
                    fixed_lean_compile_control_scout.get("error", "unknown"),
                )
            )
        )
    fixed_lean_compile_control_candidates = (
        fixed_lean_compile_control_scout.get("candidate_counts", {})
        if isinstance(fixed_lean_compile_control_scout, Mapping)
        else {}
    )
    fixed_lean_compile_control_best = (
        fixed_lean_compile_control_scout.get("best_candidate", {})
        if isinstance(fixed_lean_compile_control_scout, Mapping)
        else {}
    )
    fixed_lean_compile_control_best_delta = float("nan")
    fixed_lean_compile_control_best_two_qubit = float("nan")
    fixed_lean_compile_control_best_depth = float("nan")
    if isinstance(fixed_lean_compile_control_best, Mapping):
        fixed_lean_compile_control_best_delta = float(
            fixed_lean_compile_control_best.get("delta_mean", float("nan"))
        )
        fixed_lean_compile_control_best_two_qubit = float(
            fixed_lean_compile_control_best.get("compiled_two_qubit_count", float("nan"))
        )
        fixed_lean_compile_control_best_depth = float(
            fixed_lean_compile_control_best.get("compiled_depth", float("nan"))
        )

    fixed_scaffold_replay_success = bool(
        isinstance(fixed_scaffold_replay, Mapping) and fixed_scaffold_replay.get("success", False)
    )
    fixed_scaffold_replay_execution_mode = (
        None
        if not isinstance(fixed_scaffold_replay, Mapping)
        else fixed_scaffold_replay.get("execution_mode", None)
    )
    fixed_scaffold_replay_theta_source = (
        None
        if not isinstance(fixed_scaffold_replay, Mapping)
        else fixed_scaffold_replay.get("theta_source", None)
    )
    fixed_scaffold_replay_local_mitigation_label = (
        None
        if not isinstance(fixed_scaffold_replay, Mapping)
        else fixed_scaffold_replay.get("local_mitigation_label", None)
    )
    fixed_scaffold_replay_backend_name = (
        None
        if not isinstance(fixed_scaffold_replay, Mapping)
        else (
            fixed_scaffold_replay.get("backend_info", {}).get("backend_name", None)
            if isinstance(fixed_scaffold_replay.get("backend_info", {}), Mapping)
            else None
        )
    )
    fixed_scaffold_saved_theta_probe = (
        fixed_scaffold_replay.get("saved_theta_local_mitigation_ablation", {})
        if isinstance(fixed_scaffold_replay, Mapping)
        else {}
    )
    fixed_scaffold_saved_theta_probe_results = (
        fixed_scaffold_saved_theta_probe.get("results", {})
        if isinstance(fixed_scaffold_saved_theta_probe, Mapping)
        else {}
    )
    def _probe_delta(key: str) -> float:
        rec = fixed_scaffold_saved_theta_probe_results.get(key, {}) if isinstance(fixed_scaffold_saved_theta_probe_results, Mapping) else {}
        return float(rec.get("delta_mean", float("nan"))) if isinstance(rec, Mapping) else float("nan")
    fixed_scaffold_saved_theta_probe_dd_available = bool(
        isinstance(fixed_scaffold_saved_theta_probe_results, Mapping)
        and isinstance(fixed_scaffold_saved_theta_probe_results.get("readout_plus_local_dd", {}), Mapping)
        and fixed_scaffold_saved_theta_probe_results.get("readout_plus_local_dd", {}).get("available", False)
    )
    fixed_scaffold_runtime_energy_only_success = bool(
        isinstance(fixed_scaffold_runtime_energy_only, Mapping)
        and fixed_scaffold_runtime_energy_only.get("success", False)
    )
    if (
        isinstance(fixed_scaffold_runtime_energy_only, Mapping)
        and not fixed_scaffold_runtime_energy_only_success
        and bool(fixed_scaffold_runtime_energy_only)
    ):
        failures.append(
            "fixed_scaffold_runtime_energy_only:"
            + str(
                fixed_scaffold_runtime_energy_only.get(
                    "reason",
                    fixed_scaffold_runtime_energy_only.get("error", "unknown"),
                )
            )
        )
    if isinstance(fixed_scaffold_replay, Mapping) and not fixed_scaffold_replay_success and bool(fixed_scaffold_replay):
        failures.append(
            "fixed_scaffold_noisy_replay:"
            + str(fixed_scaffold_replay.get("reason", fixed_scaffold_replay.get("error", "unknown")))
        )
    fixed_scaffold_runtime_energy_only_main_delta = float("nan")
    if isinstance(fixed_scaffold_runtime_energy_only, Mapping):
        audits = fixed_scaffold_runtime_energy_only.get("energy_audits", {})
        main = audits.get("main", {}) if isinstance(audits, Mapping) else {}
        evaluation = main.get("evaluation", {}) if isinstance(main, Mapping) else {}
        if isinstance(evaluation, Mapping):
            fixed_scaffold_runtime_energy_only_main_delta = float(
                evaluation.get("delta_mean", float("nan"))
            )
    fixed_scaffold_runtime_raw_baseline_success = bool(
        isinstance(fixed_scaffold_runtime_raw_baseline, Mapping)
        and fixed_scaffold_runtime_raw_baseline.get("success", False)
    )
    if (
        isinstance(fixed_scaffold_runtime_raw_baseline, Mapping)
        and not fixed_scaffold_runtime_raw_baseline_success
        and bool(fixed_scaffold_runtime_raw_baseline)
    ):
        failures.append(
            "fixed_scaffold_runtime_raw_baseline:"
            + str(
                fixed_scaffold_runtime_raw_baseline.get(
                    "reason",
                    fixed_scaffold_runtime_raw_baseline.get("error", "unknown"),
                )
            )
        )
    fixed_scaffold_runtime_raw_baseline_main_delta = float("nan")
    fixed_scaffold_runtime_raw_baseline_noise_mode: str | None = None
    fixed_scaffold_runtime_raw_baseline_execution_surface: str | None = None
    fixed_scaffold_runtime_raw_baseline_raw_transport: str | None = None
    fixed_scaffold_runtime_raw_baseline_raw_artifact_path: str | None = None
    fixed_scaffold_runtime_raw_baseline_raw_store_memory = False
    fixed_scaffold_runtime_raw_baseline_diagonal_postprocessing_available = False
    fixed_scaffold_runtime_raw_baseline_requested_seed_transpiler: int | None = None
    fixed_scaffold_runtime_raw_baseline_requested_transpile_optimization_level: int | None = None
    fixed_scaffold_runtime_raw_baseline_observed_seed_transpiler: int | None = None
    fixed_scaffold_runtime_raw_baseline_observed_transpile_optimization_level: int | None = None
    fixed_scaffold_runtime_raw_baseline_compile_request_matched: bool | None = None
    if isinstance(fixed_scaffold_runtime_raw_baseline, Mapping):
        audits = fixed_scaffold_runtime_raw_baseline.get("energy_audits", {}) 
        main = audits.get("main", {}) if isinstance(audits, Mapping) else {}
        evaluation = main.get("evaluation", {}) if isinstance(main, Mapping) else {}
        if isinstance(evaluation, Mapping):
            fixed_scaffold_runtime_raw_baseline_main_delta = float(
                evaluation.get("delta_mean", float("nan"))
            )
        noise_config = (
            fixed_scaffold_runtime_raw_baseline.get("noise_config", {})
            if isinstance(fixed_scaffold_runtime_raw_baseline.get("noise_config", {}), Mapping)
            else {}
        )
        fixed_scaffold_runtime_raw_baseline_noise_mode = (
            None
            if noise_config.get("noise_mode", None) in {None, ""}
            else str(noise_config.get("noise_mode"))
        )
        backend_info = (
            fixed_scaffold_runtime_raw_baseline.get("backend_info", {})
            if isinstance(fixed_scaffold_runtime_raw_baseline.get("backend_info", {}), Mapping)
            else {}
        )
        backend_details = (
            backend_info.get("details", {})
            if isinstance(backend_info.get("details", {}), Mapping)
            else {}
        )
        fixed_scaffold_runtime_raw_baseline_execution_surface = (
            None
            if backend_details.get("execution_surface", noise_config.get("execution_surface")) in {None, ""}
            else str(backend_details.get("execution_surface", noise_config.get("execution_surface")))
        )
        fixed_scaffold_runtime_raw_baseline_raw_transport = (
            None
            if backend_details.get("transport", noise_config.get("raw_transport")) in {None, ""}
            else str(backend_details.get("transport", noise_config.get("raw_transport")))
        )
        fixed_scaffold_runtime_raw_baseline_raw_artifact_path = (
            None
            if backend_details.get("raw_artifact_path", noise_config.get("raw_artifact_path")) in {None, ""}
            else str(backend_details.get("raw_artifact_path", noise_config.get("raw_artifact_path")))
        )
        fixed_scaffold_runtime_raw_baseline_raw_store_memory = bool(
            noise_config.get("raw_store_memory", False)
        )
        fixed_scaffold_runtime_raw_baseline_diagonal_postprocessing_available = bool(
            main.get("diagonal_postprocessing", {}).get("available", False)
            if isinstance(main.get("diagonal_postprocessing", {}), Mapping)
            else False
        )
        compile_control = (
            fixed_scaffold_runtime_raw_baseline.get("compile_control", {})
            if isinstance(fixed_scaffold_runtime_raw_baseline.get("compile_control", {}), Mapping)
            else {}
        )
        compile_observation = (
            fixed_scaffold_runtime_raw_baseline.get("compile_observation", {})
            if isinstance(fixed_scaffold_runtime_raw_baseline.get("compile_observation", {}), Mapping)
            else {}
        )
        observed_compile = (
            compile_observation.get("observed", {})
            if isinstance(compile_observation.get("observed", {}), Mapping)
            else {}
        )
        if compile_control.get("seed_transpiler", None) is not None:
            fixed_scaffold_runtime_raw_baseline_requested_seed_transpiler = int(
                compile_control.get("seed_transpiler")
            )
        if compile_control.get("transpile_optimization_level", None) is not None:
            fixed_scaffold_runtime_raw_baseline_requested_transpile_optimization_level = int(
                compile_control.get("transpile_optimization_level")
            )
        if observed_compile.get("seed_transpiler", None) is not None:
            fixed_scaffold_runtime_raw_baseline_observed_seed_transpiler = int(
                observed_compile.get("seed_transpiler")
            )
        if observed_compile.get("transpile_optimization_level", None) is not None:
            fixed_scaffold_runtime_raw_baseline_observed_transpile_optimization_level = int(
                observed_compile.get("transpile_optimization_level")
            )
        if compile_observation.get("matches_requested", None) is not None:
            fixed_scaffold_runtime_raw_baseline_compile_request_matched = bool(
                compile_observation.get("matches_requested")
            )
    fixed_scaffold_attribution_success = bool(
        isinstance(fixed_scaffold_attribution, Mapping) and fixed_scaffold_attribution.get("success", False)
    )
    if isinstance(fixed_scaffold_attribution, Mapping) and not fixed_scaffold_attribution_success and bool(fixed_scaffold_attribution):
        failures.append(
            "fixed_scaffold_noise_attribution:"
            + str(fixed_scaffold_attribution.get("reason", fixed_scaffold_attribution.get("error", "unknown")))
        )
    fixed_scaffold_slices = fixed_scaffold_attribution.get("slices", {}) if isinstance(fixed_scaffold_attribution, Mapping) else {}
    fixed_scaffold_slices_completed = sum(
        1 for rec in fixed_scaffold_slices.values() if isinstance(rec, Mapping) and bool(rec.get("success", False))
    )
    for slice_name, rec in fixed_scaffold_slices.items() if isinstance(fixed_scaffold_slices, Mapping) else []:
        if isinstance(rec, Mapping) and not bool(rec.get("success", False)):
            failures.append(
                "fixed_scaffold_noise_attribution:"
                + str(slice_name)
                + ":"
                + str(rec.get("reason", rec.get("error", "unknown")))
            )
    fixed_scaffold_best_noisy_minus_ideal = float("nan")
    if isinstance(fixed_scaffold_replay, Mapping):
        energies = fixed_scaffold_replay.get("energies", {})
        if isinstance(energies, Mapping):
            fixed_scaffold_best_noisy_minus_ideal = float(
                energies.get("best_noisy_minus_ideal", float("nan"))
            )
    fixed_scaffold_compile_control_success = bool(
        isinstance(fixed_scaffold_compile_control_scout, Mapping)
        and fixed_scaffold_compile_control_scout.get("success", False)
    )
    if (
        isinstance(fixed_scaffold_compile_control_scout, Mapping)
        and not fixed_scaffold_compile_control_success
        and bool(fixed_scaffold_compile_control_scout)
    ):
        failures.append(
            "fixed_scaffold_compile_control_scout:"
            + str(
                fixed_scaffold_compile_control_scout.get(
                    "reason",
                    fixed_scaffold_compile_control_scout.get("error", "unknown"),
                )
            )
        )
    fixed_scaffold_compile_control_candidates = (
        fixed_scaffold_compile_control_scout.get("candidate_counts", {})
        if isinstance(fixed_scaffold_compile_control_scout, Mapping)
        else {}
    )
    fixed_scaffold_compile_control_best = (
        fixed_scaffold_compile_control_scout.get("best_candidate", {})
        if isinstance(fixed_scaffold_compile_control_scout, Mapping)
        else {}
    )
    fixed_scaffold_compile_control_best_delta = float("nan")
    fixed_scaffold_compile_control_best_two_qubit = float("nan")
    fixed_scaffold_compile_control_best_depth = float("nan")
    if isinstance(fixed_scaffold_compile_control_best, Mapping):
        fixed_scaffold_compile_control_best_delta = float(
            fixed_scaffold_compile_control_best.get("delta_mean", float("nan"))
        )
        fixed_scaffold_compile_control_best_two_qubit = float(
            fixed_scaffold_compile_control_best.get("compiled_two_qubit_count", float("nan"))
        )
        fixed_scaffold_compile_control_best_depth = float(
            fixed_scaffold_compile_control_best.get("compiled_depth", float("nan"))
        )
    fixed_scaffold_mitigation_matrix_success = bool(
        isinstance(fixed_scaffold_saved_theta_mitigation_matrix, Mapping)
        and fixed_scaffold_saved_theta_mitigation_matrix.get("success", False)
    )
    if (
        isinstance(fixed_scaffold_saved_theta_mitigation_matrix, Mapping)
        and not fixed_scaffold_mitigation_matrix_success
        and bool(fixed_scaffold_saved_theta_mitigation_matrix)
    ):
        failures.append(
            "fixed_scaffold_saved_theta_mitigation_matrix:"
            + str(
                fixed_scaffold_saved_theta_mitigation_matrix.get(
                    "reason",
                    fixed_scaffold_saved_theta_mitigation_matrix.get("error", "unknown"),
                )
            )
        )
    fixed_scaffold_mitigation_matrix_counts = (
        fixed_scaffold_saved_theta_mitigation_matrix.get("cell_counts", {})
        if isinstance(fixed_scaffold_saved_theta_mitigation_matrix, Mapping)
        else {}
    )
    fixed_scaffold_mitigation_matrix_best = (
        fixed_scaffold_saved_theta_mitigation_matrix.get("best_cell", {})
        if isinstance(fixed_scaffold_saved_theta_mitigation_matrix, Mapping)
        else {}
    )
    fixed_scaffold_mitigation_matrix_best_label: str | None = None
    fixed_scaffold_mitigation_matrix_best_delta = float("nan")
    fixed_scaffold_mitigation_matrix_best_two_qubit = float("nan")
    fixed_scaffold_mitigation_matrix_best_depth = float("nan")
    fixed_scaffold_mitigation_matrix_best_compile_preset: str | None = None
    fixed_scaffold_mitigation_matrix_best_suppression_stack: str | None = None
    fixed_scaffold_mitigation_matrix_best_zne_enabled: bool | None = None
    fixed_scaffold_mitigation_matrix_best_mitigation_mode: str | None = None
    if isinstance(fixed_scaffold_mitigation_matrix_best, Mapping):
        fixed_scaffold_mitigation_matrix_best_label = (
            None
            if fixed_scaffold_mitigation_matrix_best.get("label", None) in {None, ""}
            else str(fixed_scaffold_mitigation_matrix_best.get("label"))
        )
        fixed_scaffold_mitigation_matrix_best_delta = float(
            fixed_scaffold_mitigation_matrix_best.get("delta_mean", float("nan"))
        )
        fixed_scaffold_mitigation_matrix_best_two_qubit = float(
            fixed_scaffold_mitigation_matrix_best.get("compiled_two_qubit_count", float("nan"))
        )
        fixed_scaffold_mitigation_matrix_best_depth = float(
            fixed_scaffold_mitigation_matrix_best.get("compiled_depth", float("nan"))
        )
        fixed_scaffold_mitigation_matrix_best_compile_preset = (
            None
            if fixed_scaffold_mitigation_matrix_best.get("compile_preset_label", None) in {None, ""}
            else str(fixed_scaffold_mitigation_matrix_best.get("compile_preset_label"))
        )
        fixed_scaffold_mitigation_matrix_best_suppression_stack = (
            None
            if fixed_scaffold_mitigation_matrix_best.get("suppression_stack", None) in {None, ""}
            else str(fixed_scaffold_mitigation_matrix_best.get("suppression_stack"))
        )
        if fixed_scaffold_mitigation_matrix_best.get("zne_enabled", None) is not None:
            fixed_scaffold_mitigation_matrix_best_zne_enabled = bool(
                fixed_scaffold_mitigation_matrix_best.get("zne_enabled")
            )
        best_mitigation_cfg = fixed_scaffold_mitigation_matrix_best.get("mitigation_config", {})
        if isinstance(best_mitigation_cfg, Mapping):
            fixed_scaffold_mitigation_matrix_best_mitigation_mode = (
                None
                if best_mitigation_cfg.get("mode", None) in {None, ""}
                else str(best_mitigation_cfg.get("mode"))
            )
    fixed_scaffold_full_delta = float("nan")
    fixed_scaffold_readout_delta = float("nan")
    fixed_scaffold_gate_delta = float("nan")
    if isinstance(fixed_scaffold_slices, Mapping):
        full_rec = fixed_scaffold_slices.get("full", {})
        readout_rec = fixed_scaffold_slices.get("readout_only", {})
        gate_rec = fixed_scaffold_slices.get("gate_stateprep_only", {})
        if isinstance(full_rec, Mapping):
            fixed_scaffold_full_delta = float(full_rec.get("delta_mean", float("nan")))
        if isinstance(readout_rec, Mapping):
            fixed_scaffold_readout_delta = float(readout_rec.get("delta_mean", float("nan")))
        if isinstance(gate_rec, Mapping):
            fixed_scaffold_gate_delta = float(gate_rec.get("delta_mean", float("nan")))
    return {
        "warm_delta_abs": float(warm.get("delta_abs", float("nan"))),
        "adapt_delta_abs": float(adapt.get("delta_abs", float("nan"))),
        "final_delta_abs": float(final.get("delta_abs", float("nan"))),
        "adapt_oracle_execution_surface": (
            None if adapt.get("oracle_execution_surface") in {None, "", "off"} else str(adapt.get("oracle_execution_surface"))
        ) if isinstance(adapt, Mapping) else None,
        "adapt_oracle_raw_transport": (
            None if adapt.get("oracle_raw_transport") in {None, ""} else str(adapt.get("oracle_raw_transport"))
        ) if isinstance(adapt, Mapping) else None,
        "adapt_oracle_raw_artifact_path": (
            None if adapt.get("oracle_gradient_raw_artifact_path") in {None, ""} else str(adapt.get("oracle_gradient_raw_artifact_path"))
        ) if isinstance(adapt, Mapping) else None,
        "adapt_oracle_raw_records_total": int(adapt.get("oracle_gradient_raw_records_total", 0)) if isinstance(adapt, Mapping) else 0,
        "noisy_method_modes_completed": int(completed),
        "noisy_method_modes_total": int(total),
        "imported_prepared_state_audit_completed": int(imported_prepared_completed),
        "imported_prepared_state_audit_total": int(len(imported_prepared_modes)),
        "imported_ansatz_input_state_audit_completed": int(imported_ansatz_input_completed),
        "imported_ansatz_input_state_audit_total": int(len(imported_ansatz_input_modes)),
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
        "fixed_lean_compile_control_scout_completed": int(1 if fixed_lean_compile_control_success else 0),
        "fixed_lean_compile_control_scout_total": int(
            1 if bool(fixed_lean_compile_control_scout) else 0
        ),
        "fixed_lean_compile_control_scout_candidates_total": int(
            fixed_lean_compile_control_candidates.get("total", 0)
        )
        if isinstance(fixed_lean_compile_control_candidates, Mapping)
        else 0,
        "fixed_lean_compile_control_scout_candidates_successful": int(
            fixed_lean_compile_control_candidates.get("successful", 0)
        )
        if isinstance(fixed_lean_compile_control_candidates, Mapping)
        else 0,
        "fixed_lean_compile_control_scout_best_delta_mean": float(
            fixed_lean_compile_control_best_delta
        ),
        "fixed_lean_compile_control_scout_best_two_qubit_count": float(
            fixed_lean_compile_control_best_two_qubit
        ),
        "fixed_lean_compile_control_scout_best_depth": float(
            fixed_lean_compile_control_best_depth
        ),
        "fixed_scaffold_noisy_replay_completed": int(1 if fixed_scaffold_replay_success else 0),
        "fixed_scaffold_noisy_replay_total": int(1 if bool(fixed_scaffold_replay) else 0),
        "fixed_scaffold_best_noisy_minus_ideal": float(fixed_scaffold_best_noisy_minus_ideal),
        "fixed_scaffold_noisy_replay_execution_mode": fixed_scaffold_replay_execution_mode,
        "fixed_scaffold_noisy_replay_theta_source": fixed_scaffold_replay_theta_source,
        "fixed_scaffold_noisy_replay_local_mitigation_label": fixed_scaffold_replay_local_mitigation_label,
        "fixed_scaffold_noisy_replay_backend_name": fixed_scaffold_replay_backend_name,
        "fixed_scaffold_saved_theta_probe_readout_only_delta_mean": float(_probe_delta("readout_only")),
        "fixed_scaffold_saved_theta_probe_gate_twirling_delta_mean": float(
            _probe_delta("readout_plus_gate_twirling")
        ),
        "fixed_scaffold_saved_theta_probe_local_dd_delta_mean": float(
            _probe_delta("readout_plus_local_dd")
        ),
        "fixed_scaffold_saved_theta_probe_local_dd_available": bool(
            fixed_scaffold_saved_theta_probe_dd_available
        ),
        "fixed_scaffold_runtime_energy_only_completed": int(
            1 if fixed_scaffold_runtime_energy_only_success else 0
        ),
        "fixed_scaffold_runtime_energy_only_total": int(
            1 if bool(fixed_scaffold_runtime_energy_only) else 0
        ),
        "fixed_scaffold_runtime_energy_only_main_delta_mean": float(
            fixed_scaffold_runtime_energy_only_main_delta
        ),
        "fixed_scaffold_runtime_raw_baseline_completed": int(
            1 if fixed_scaffold_runtime_raw_baseline_success else 0
        ),
        "fixed_scaffold_runtime_raw_baseline_total": int(
            1 if bool(fixed_scaffold_runtime_raw_baseline) else 0
        ),
        "fixed_scaffold_runtime_raw_baseline_main_delta_mean": float(
            fixed_scaffold_runtime_raw_baseline_main_delta
        ),
        "fixed_scaffold_runtime_raw_baseline_noise_mode": (
            fixed_scaffold_runtime_raw_baseline_noise_mode
        ),
        "fixed_scaffold_runtime_raw_baseline_execution_surface": (
            fixed_scaffold_runtime_raw_baseline_execution_surface
        ),
        "fixed_scaffold_runtime_raw_baseline_raw_transport": (
            fixed_scaffold_runtime_raw_baseline_raw_transport
        ),
        "fixed_scaffold_runtime_raw_baseline_raw_store_memory": bool(
            fixed_scaffold_runtime_raw_baseline_raw_store_memory
        ),
        "fixed_scaffold_runtime_raw_baseline_raw_artifact_path": (
            fixed_scaffold_runtime_raw_baseline_raw_artifact_path
        ),
        "fixed_scaffold_runtime_raw_baseline_diagonal_postprocessing_available": bool(
            fixed_scaffold_runtime_raw_baseline_diagonal_postprocessing_available
        ),
        "fixed_scaffold_runtime_raw_baseline_requested_seed_transpiler": (
            fixed_scaffold_runtime_raw_baseline_requested_seed_transpiler
        ),
        "fixed_scaffold_runtime_raw_baseline_requested_transpile_optimization_level": (
            fixed_scaffold_runtime_raw_baseline_requested_transpile_optimization_level
        ),
        "fixed_scaffold_runtime_raw_baseline_observed_seed_transpiler": (
            fixed_scaffold_runtime_raw_baseline_observed_seed_transpiler
        ),
        "fixed_scaffold_runtime_raw_baseline_observed_transpile_optimization_level": (
            fixed_scaffold_runtime_raw_baseline_observed_transpile_optimization_level
        ),
        "fixed_scaffold_runtime_raw_baseline_compile_request_matched": (
            fixed_scaffold_runtime_raw_baseline_compile_request_matched
        ),
        "fixed_scaffold_compile_control_scout_completed": int(
            1 if fixed_scaffold_compile_control_success else 0
        ),
        "fixed_scaffold_compile_control_scout_total": int(
            1 if bool(fixed_scaffold_compile_control_scout) else 0
        ),
        "fixed_scaffold_compile_control_scout_candidates_total": int(
            fixed_scaffold_compile_control_candidates.get("total", 0)
        )
        if isinstance(fixed_scaffold_compile_control_candidates, Mapping)
        else 0,
        "fixed_scaffold_compile_control_scout_candidates_successful": int(
            fixed_scaffold_compile_control_candidates.get("successful", 0)
        )
        if isinstance(fixed_scaffold_compile_control_candidates, Mapping)
        else 0,
        "fixed_scaffold_compile_control_scout_best_delta_mean": float(
            fixed_scaffold_compile_control_best_delta
        ),
        "fixed_scaffold_compile_control_scout_best_two_qubit_count": float(
            fixed_scaffold_compile_control_best_two_qubit
        ),
        "fixed_scaffold_compile_control_scout_best_depth": float(
            fixed_scaffold_compile_control_best_depth
        ),
        "fixed_scaffold_saved_theta_mitigation_matrix_completed": int(
            1 if fixed_scaffold_mitigation_matrix_success else 0
        ),
        "fixed_scaffold_saved_theta_mitigation_matrix_total": int(
            1 if bool(fixed_scaffold_saved_theta_mitigation_matrix) else 0
        ),
        "fixed_scaffold_saved_theta_mitigation_matrix_cells_total": int(
            fixed_scaffold_mitigation_matrix_counts.get("total", 0)
        )
        if isinstance(fixed_scaffold_mitigation_matrix_counts, Mapping)
        else 0,
        "fixed_scaffold_saved_theta_mitigation_matrix_cells_completed": int(
            fixed_scaffold_mitigation_matrix_counts.get("completed", 0)
        )
        if isinstance(fixed_scaffold_mitigation_matrix_counts, Mapping)
        else 0,
        "fixed_scaffold_saved_theta_mitigation_matrix_cells_failed": int(
            fixed_scaffold_mitigation_matrix_counts.get("failed", 0)
        )
        if isinstance(fixed_scaffold_mitigation_matrix_counts, Mapping)
        else 0,
        "fixed_scaffold_saved_theta_mitigation_matrix_best_label": (
            fixed_scaffold_mitigation_matrix_best_label
        ),
        "fixed_scaffold_saved_theta_mitigation_matrix_best_delta_mean": float(
            fixed_scaffold_mitigation_matrix_best_delta
        ),
        "fixed_scaffold_saved_theta_mitigation_matrix_best_two_qubit_count": float(
            fixed_scaffold_mitigation_matrix_best_two_qubit
        ),
        "fixed_scaffold_saved_theta_mitigation_matrix_best_depth": float(
            fixed_scaffold_mitigation_matrix_best_depth
        ),
        "fixed_scaffold_saved_theta_mitigation_matrix_best_compile_preset": (
            fixed_scaffold_mitigation_matrix_best_compile_preset
        ),
        "fixed_scaffold_saved_theta_mitigation_matrix_best_suppression_stack": (
            fixed_scaffold_mitigation_matrix_best_suppression_stack
        ),
        "fixed_scaffold_saved_theta_mitigation_matrix_best_zne_enabled": (
            fixed_scaffold_mitigation_matrix_best_zne_enabled
        ),
        "fixed_scaffold_saved_theta_mitigation_matrix_best_mitigation_mode": (
            fixed_scaffold_mitigation_matrix_best_mitigation_mode
        ),
        "fixed_scaffold_noise_attribution_completed": int(1 if fixed_scaffold_attribution_success else 0),
        "fixed_scaffold_noise_attribution_total": int(1 if bool(fixed_scaffold_attribution) else 0),
        "fixed_scaffold_noise_attribution_slices_completed": int(fixed_scaffold_slices_completed),
        "fixed_scaffold_noise_attribution_slices_total": int(
            len(fixed_scaffold_slices) if isinstance(fixed_scaffold_slices, Mapping) else 0
        ),
        "fixed_scaffold_noise_attribution_full_minus_ideal": float(fixed_scaffold_full_delta),
        "fixed_scaffold_noise_attribution_readout_only_minus_ideal": float(fixed_scaffold_readout_delta),
        "fixed_scaffold_noise_attribution_gate_stateprep_only_minus_ideal": float(fixed_scaffold_gate_delta),
        "adaptive_realtime_checkpoint_mode": (
            None if not isinstance(adaptive_rt, Mapping) else adaptive_rt.get("mode", None)
        ),
        "adaptive_realtime_checkpoint_status": (
            None if not isinstance(adaptive_rt, Mapping) else adaptive_rt.get("status", None)
        ),
        "adaptive_realtime_checkpoint_append_count": (
            0 if not isinstance(adaptive_rt, Mapping) else int(adaptive_rt.get("summary", {}).get("append_count", 0))
        ),
        "adaptive_realtime_checkpoint_stay_count": (
            0 if not isinstance(adaptive_rt, Mapping) else int(adaptive_rt.get("summary", {}).get("stay_count", 0))
        ),
        "adaptive_realtime_checkpoint_decision_noise_mode": (
            None if not isinstance(adaptive_rt, Mapping) else adaptive_rt.get("summary", {}).get("decision_noise_mode", None)
        ),
        "adaptive_realtime_checkpoint_degraded_checkpoints": (
            0 if not isinstance(adaptive_rt, Mapping) else int(adaptive_rt.get("summary", {}).get("degraded_checkpoints", 0))
        ),
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
    imported_prepared = payload.get("imported_prepared_state_audit", {}) if isinstance(payload, Mapping) else {}
    imported_ansatz_input = (
        payload.get("imported_ansatz_input_state_audit", {}) if isinstance(payload, Mapping) else {}
    )
    imported_full = payload.get("full_circuit_import_audit", {}) if isinstance(payload, Mapping) else {}
    fixed_lean_replay = payload.get("fixed_lean_scaffold_noisy_replay", {}) if isinstance(payload, Mapping) else {}
    fixed_lean_attribution = payload.get("fixed_lean_noise_attribution", {}) if isinstance(payload, Mapping) else {}
    fixed_lean_compile_control_scout = (
        payload.get("fixed_lean_compile_control_scout", {}) if isinstance(payload, Mapping) else {}
    )
    fixed_scaffold_replay = payload.get("fixed_scaffold_noisy_replay", {}) if isinstance(payload, Mapping) else {}
    fixed_scaffold_saved_theta_mitigation_matrix = (
        payload.get("fixed_scaffold_saved_theta_mitigation_matrix", {})
        if isinstance(payload, Mapping)
        else {}
    )
    fixed_scaffold_runtime_raw_baseline = (
        payload.get("fixed_scaffold_runtime_raw_baseline", {}) if isinstance(payload, Mapping) else {}
    )
    fixed_scaffold_compile_control_scout = (
        payload.get("fixed_scaffold_compile_control_scout", {}) if isinstance(payload, Mapping) else {}
    )
    fixed_scaffold_attribution = payload.get("fixed_scaffold_noise_attribution", {}) if isinstance(payload, Mapping) else {}
    import_mode = bool(isinstance(import_source, Mapping) and str(import_source.get("mode", "")) == "imported_artifact")
    if bool(fixed_scaffold_runtime_raw_baseline):
        ansatz_label = "imported fixed-scaffold Marrakesh/Heron 6-term Runtime raw-shot baseline"
    elif bool(payload.get("fixed_scaffold_runtime_energy_only", {})):
        ansatz_label = "imported fixed-scaffold Marrakesh/Heron 6-term Runtime audit"
    elif bool(fixed_scaffold_saved_theta_mitigation_matrix):
        ansatz_label = "imported fixed-scaffold Marrakesh/Heron 6-term saved-theta mitigation matrix"
    elif bool(fixed_scaffold_replay) or bool(fixed_scaffold_compile_control_scout) or bool(fixed_scaffold_attribution):
        ansatz_label = "imported fixed-scaffold Marrakesh/Heron 6-term local replay / scout"
    elif bool(fixed_lean_replay) or bool(fixed_lean_attribution) or bool(fixed_lean_compile_control_scout):
        ansatz_label = "imported lean ADAPT compile-control scout / fixed-circuit audit"
    elif import_mode:
        ansatz_label = "imported lean ADAPT circuit audit"
    else:
        ansatz_label = "warm: hh_hva_ptw; ADAPT: staged HH; final: matched-family replay; noisy final-only dynamics"

    with PdfPages(pdf_path) as pdf:
        render_parameter_manifest(
            pdf,
            model="Hubbard-Holstein",
            ansatz=ansatz_label,
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
                f"ADAPT oracle surface/transport/path/records: {summary.get('adapt_oracle_execution_surface')} / {summary.get('adapt_oracle_raw_transport')} / {summary.get('adapt_oracle_raw_artifact_path')} / {summary.get('adapt_oracle_raw_records_total')}",
                f"Replay |ΔE|: {summary.get('final_delta_abs')}",
                f"Checkpoint controller: mode={summary.get('adaptive_realtime_checkpoint_mode')} status={summary.get('adaptive_realtime_checkpoint_status')} append={summary.get('adaptive_realtime_checkpoint_append_count')} stay={summary.get('adaptive_realtime_checkpoint_stay_count')} decision_noise_mode={summary.get('adaptive_realtime_checkpoint_decision_noise_mode')} degraded={summary.get('adaptive_realtime_checkpoint_degraded_checkpoints')}",
                f"Noisy method/mode completion: {summary.get('noisy_method_modes_completed')} / {summary.get('noisy_method_modes_total')}",
                f"Imported prepared-state audit: {summary.get('imported_prepared_state_audit_completed')} / {summary.get('imported_prepared_state_audit_total')}",
                f"Imported ansatz-input-state audit: {summary.get('imported_ansatz_input_state_audit_completed')} / {summary.get('imported_ansatz_input_state_audit_total')}",
                f"Imported full-circuit audit: {summary.get('full_circuit_import_audit_completed')} / {summary.get('full_circuit_import_audit_total')}",
                f"Fixed lean noisy replay: {summary.get('fixed_lean_scaffold_noisy_replay_completed')} / {summary.get('fixed_lean_scaffold_noisy_replay_total')}",
                f"Fixed lean best Δ(noisy-ideal): {summary.get('fixed_lean_scaffold_best_noisy_minus_ideal')}",
                f"Fixed lean attribution: {summary.get('fixed_lean_noise_attribution_completed')} / {summary.get('fixed_lean_noise_attribution_total')}",
                f"Fixed lean attribution slices: {summary.get('fixed_lean_noise_attribution_slices_completed')} / {summary.get('fixed_lean_noise_attribution_slices_total')}",
                f"Attribution full Δ(noisy-ideal): {summary.get('fixed_lean_noise_attribution_full_minus_ideal')}",
                f"Attribution readout-only Δ(noisy-ideal): {summary.get('fixed_lean_noise_attribution_readout_only_minus_ideal')}",
                f"Attribution gate-only Δ(noisy-ideal): {summary.get('fixed_lean_noise_attribution_gate_stateprep_only_minus_ideal')}",
                f"Fixed lean compile-control scout: {summary.get('fixed_lean_compile_control_scout_completed')} / {summary.get('fixed_lean_compile_control_scout_total')}",
                f"Compile-control successful candidates: {summary.get('fixed_lean_compile_control_scout_candidates_successful')} / {summary.get('fixed_lean_compile_control_scout_candidates_total')}",
                f"Compile-control best Δ(noisy-ideal): {summary.get('fixed_lean_compile_control_scout_best_delta_mean')}",
                f"Compile-control best 2Q count: {summary.get('fixed_lean_compile_control_scout_best_two_qubit_count')}",
                f"Compile-control best depth: {summary.get('fixed_lean_compile_control_scout_best_depth')}",
                f"Fixed scaffold noisy replay: {summary.get('fixed_scaffold_noisy_replay_completed')} / {summary.get('fixed_scaffold_noisy_replay_total')}",
                f"Fixed scaffold best Δ(noisy-ideal): {summary.get('fixed_scaffold_best_noisy_minus_ideal')}",
                f"Fixed scaffold replay exec/theta/backend: {summary.get('fixed_scaffold_noisy_replay_execution_mode')} / {summary.get('fixed_scaffold_noisy_replay_theta_source')} / {summary.get('fixed_scaffold_noisy_replay_backend_name')}",
                f"Fixed scaffold replay mitigation label: {summary.get('fixed_scaffold_noisy_replay_local_mitigation_label')}",
                f"Saved-theta probe readout-only Δ(noisy-ideal): {summary.get('fixed_scaffold_saved_theta_probe_readout_only_delta_mean')}",
                f"Saved-theta probe twirling Δ(noisy-ideal): {summary.get('fixed_scaffold_saved_theta_probe_gate_twirling_delta_mean')}",
                f"Saved-theta probe DD available / Δ(noisy-ideal): {summary.get('fixed_scaffold_saved_theta_probe_local_dd_available')} / {summary.get('fixed_scaffold_saved_theta_probe_local_dd_delta_mean')}",
                f"Fixed scaffold runtime energy-only: {summary.get('fixed_scaffold_runtime_energy_only_completed')} / {summary.get('fixed_scaffold_runtime_energy_only_total')}",
                f"Fixed scaffold runtime main Δ(noisy-ideal): {summary.get('fixed_scaffold_runtime_energy_only_main_delta_mean')}",
                f"Fixed scaffold runtime raw baseline: {summary.get('fixed_scaffold_runtime_raw_baseline_completed')} / {summary.get('fixed_scaffold_runtime_raw_baseline_total')}",
                f"Fixed scaffold runtime raw baseline Δ(surface/transport/store/path): {summary.get('fixed_scaffold_runtime_raw_baseline_main_delta_mean')} / {summary.get('fixed_scaffold_runtime_raw_baseline_execution_surface')} / {summary.get('fixed_scaffold_runtime_raw_baseline_raw_transport')} / {summary.get('fixed_scaffold_runtime_raw_baseline_raw_store_memory')} / {summary.get('fixed_scaffold_runtime_raw_baseline_raw_artifact_path')}",
                f"Fixed scaffold runtime raw baseline compile request/observed/match: {summary.get('fixed_scaffold_runtime_raw_baseline_requested_transpile_optimization_level')}:{summary.get('fixed_scaffold_runtime_raw_baseline_requested_seed_transpiler')} / {summary.get('fixed_scaffold_runtime_raw_baseline_observed_transpile_optimization_level')}:{summary.get('fixed_scaffold_runtime_raw_baseline_observed_seed_transpiler')} / {summary.get('fixed_scaffold_runtime_raw_baseline_compile_request_matched')}",
                f"Fixed scaffold compile-control scout: {summary.get('fixed_scaffold_compile_control_scout_completed')} / {summary.get('fixed_scaffold_compile_control_scout_total')}",
                f"Fixed scaffold compile-control successful candidates: {summary.get('fixed_scaffold_compile_control_scout_candidates_successful')} / {summary.get('fixed_scaffold_compile_control_scout_candidates_total')}",
                f"Fixed scaffold compile-control best Δ(noisy-ideal): {summary.get('fixed_scaffold_compile_control_scout_best_delta_mean')}",
                f"Fixed scaffold compile-control best 2Q count: {summary.get('fixed_scaffold_compile_control_scout_best_two_qubit_count')}",
                f"Fixed scaffold compile-control best depth: {summary.get('fixed_scaffold_compile_control_scout_best_depth')}",
                f"Fixed scaffold saved-theta mitigation matrix: {summary.get('fixed_scaffold_saved_theta_mitigation_matrix_completed')} / {summary.get('fixed_scaffold_saved_theta_mitigation_matrix_total')}",
                f"Fixed scaffold mitigation-matrix cells completed/failed/total: {summary.get('fixed_scaffold_saved_theta_mitigation_matrix_cells_completed')} / {summary.get('fixed_scaffold_saved_theta_mitigation_matrix_cells_failed')} / {summary.get('fixed_scaffold_saved_theta_mitigation_matrix_cells_total')}",
                f"Fixed scaffold mitigation-matrix best label/Δ: {summary.get('fixed_scaffold_saved_theta_mitigation_matrix_best_label')} / {summary.get('fixed_scaffold_saved_theta_mitigation_matrix_best_delta_mean')}",
                f"Fixed scaffold mitigation-matrix best compile/zne/stack/mode: {summary.get('fixed_scaffold_saved_theta_mitigation_matrix_best_compile_preset')} / {summary.get('fixed_scaffold_saved_theta_mitigation_matrix_best_zne_enabled')} / {summary.get('fixed_scaffold_saved_theta_mitigation_matrix_best_suppression_stack')} / {summary.get('fixed_scaffold_saved_theta_mitigation_matrix_best_mitigation_mode')}",
                f"Fixed scaffold attribution: {summary.get('fixed_scaffold_noise_attribution_completed')} / {summary.get('fixed_scaffold_noise_attribution_total')}",
                f"Fixed scaffold attribution slices: {summary.get('fixed_scaffold_noise_attribution_slices_completed')} / {summary.get('fixed_scaffold_noise_attribution_slices_total')}",
                f"Fixed scaffold attribution full Δ(noisy-ideal): {summary.get('fixed_scaffold_noise_attribution_full_minus_ideal')}",
                f"Fixed scaffold attribution readout-only Δ(noisy-ideal): {summary.get('fixed_scaffold_noise_attribution_readout_only_minus_ideal')}",
                f"Fixed scaffold attribution gate-only Δ(noisy-ideal): {summary.get('fixed_scaffold_noise_attribution_gate_stateprep_only_minus_ideal')}",
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
        if isinstance(fixed_lean_compile_control_scout, Mapping) and fixed_lean_compile_control_scout:
            best = (
                fixed_lean_compile_control_scout.get("best_candidate", {})
                if isinstance(fixed_lean_compile_control_scout.get("best_candidate", {}), Mapping)
                else {}
            )
            noisy_rows.append([
                "fixed_lean_compile_control_scout",
                "locked_imported_circuit",
                str(best.get("label", "best")),
                str(bool(fixed_lean_compile_control_scout.get("success", False))),
                f"{float(best.get('delta_mean', float('nan'))):.3e}",
            ])
        if isinstance(fixed_scaffold_replay, Mapping) and fixed_scaffold_replay:
            energies = fixed_scaffold_replay.get("energies", {}) if isinstance(fixed_scaffold_replay.get("energies", {}), Mapping) else {}
            noisy_rows.append([
                "fixed_scaffold_replay",
                "imported_locked_scaffold",
                str(fixed_scaffold_replay.get("noise_config", {}).get("noise_mode", "backend_scheduled")) if isinstance(fixed_scaffold_replay.get("noise_config", {}), Mapping) else "backend_scheduled",
                str(bool(fixed_scaffold_replay.get("success", False))),
                f"{float(energies.get('best_noisy_minus_ideal', float('nan'))):.3e}",
            ])
        if isinstance(payload.get("fixed_scaffold_runtime_energy_only", {}), Mapping) and payload.get("fixed_scaffold_runtime_energy_only", {}):
            runtime_energy_only = payload.get("fixed_scaffold_runtime_energy_only", {})
            audits = runtime_energy_only.get("energy_audits", {}) if isinstance(runtime_energy_only, Mapping) else {}
            main = audits.get("main", {}) if isinstance(audits, Mapping) else {}
            evaluation = main.get("evaluation", {}) if isinstance(main, Mapping) else {}
            noisy_rows.append([
                "fixed_scaffold_runtime_energy_only",
                "imported_locked_scaffold",
                "runtime_main",
                str(bool(runtime_energy_only.get("success", False))),
                f"{float(evaluation.get('delta_mean', float('nan'))):.3e}",
            ])
        if isinstance(fixed_scaffold_runtime_raw_baseline, Mapping) and fixed_scaffold_runtime_raw_baseline:
            audits = fixed_scaffold_runtime_raw_baseline.get("energy_audits", {}) if isinstance(fixed_scaffold_runtime_raw_baseline, Mapping) else {}
            main = audits.get("main", {}) if isinstance(audits, Mapping) else {}
            evaluation = main.get("evaluation", {}) if isinstance(main, Mapping) else {}
            noisy_rows.append([
                "fixed_scaffold_runtime_raw_baseline",
                "imported_locked_scaffold",
                str(summary.get("fixed_scaffold_runtime_raw_baseline_raw_transport") or "runtime_raw"),
                str(bool(fixed_scaffold_runtime_raw_baseline.get("success", False))),
                f"{float(evaluation.get('delta_mean', float('nan'))):.3e}",
            ])
        if isinstance(fixed_scaffold_compile_control_scout, Mapping) and fixed_scaffold_compile_control_scout:
            best = (
                fixed_scaffold_compile_control_scout.get("best_candidate", {})
                if isinstance(fixed_scaffold_compile_control_scout.get("best_candidate", {}), Mapping)
                else {}
            )
            noisy_rows.append([
                "fixed_scaffold_compile_control_scout",
                "locked_imported_circuit",
                str(best.get("label", "best")),
                str(bool(fixed_scaffold_compile_control_scout.get("success", False))),
                f"{float(best.get('delta_mean', float('nan'))):.3e}",
            ])
        if isinstance(fixed_scaffold_saved_theta_mitigation_matrix, Mapping) and fixed_scaffold_saved_theta_mitigation_matrix:
            best = (
                fixed_scaffold_saved_theta_mitigation_matrix.get("best_cell", {})
                if isinstance(fixed_scaffold_saved_theta_mitigation_matrix.get("best_cell", {}), Mapping)
                else {}
            )
            noisy_rows.append([
                "fixed_scaffold_saved_theta_mitigation_matrix",
                "locked_imported_circuit",
                str(best.get("label", "best")),
                str(bool(fixed_scaffold_saved_theta_mitigation_matrix.get("success", False))),
                f"{float(best.get('delta_mean', float('nan'))):.3e}",
            ])
        if isinstance(fixed_scaffold_attribution, Mapping) and fixed_scaffold_attribution:
            slices = fixed_scaffold_attribution.get("slices", {}) if isinstance(fixed_scaffold_attribution.get("slices", {}), Mapping) else {}
            for slice_name, rec in slices.items():
                if not isinstance(rec, Mapping):
                    continue
                noisy_rows.append([
                    "fixed_scaffold_attribution",
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
        if str(cfg.staged.realtime_checkpoint.mode) != "off":
            raise ValueError(
                "adaptive realtime checkpoint controller is currently supported only for fresh-stage noisy runs; disable --checkpoint-controller-mode for imported-artifact routes."
            )
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
        imported_ansatz_input_state_audit = {
            "modes": (
                {
                    str(mode): _run_imported_ansatz_input_state_audit_mode(
                        source_cfg=cfg.source,
                        noise_cfg=cfg.noise,
                        mode=str(mode),
                        seed_transpiler=cfg.fixed_scaffold_runtime_raw_baseline.seed_transpiler,
                        transpile_optimization_level=cfg.fixed_scaffold_runtime_raw_baseline.transpile_optimization_level,
                        compile_request_source="fixed_scaffold_runtime_transpile_cli",
                    )
                    for mode in cfg.noise.audit_modes
                }
                if bool(cfg.noise.include_full_circuit_audit)
                else {}
            )
        }
        full_circuit_import_audit = {
            "modes": (
                {
                    str(mode): _run_imported_full_circuit_audit_mode(
                        source_cfg=cfg.source,
                        noise_cfg=cfg.noise,
                        mode=str(mode),
                        seed_transpiler=cfg.fixed_scaffold_runtime_raw_baseline.seed_transpiler,
                        transpile_optimization_level=cfg.fixed_scaffold_runtime_raw_baseline.transpile_optimization_level,
                        compile_request_source="fixed_scaffold_runtime_transpile_cli",
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
        fixed_lean_compile_control_scout = (
            _run_fixed_lean_compile_control_scout_mode(
                source_cfg=cfg.source,
                noise_cfg=cfg.noise,
                scout_cfg=cfg.fixed_lean_compile_control_scout,
            )
            if bool(cfg.fixed_lean_compile_control_scout.enabled)
            else {}
        )
        fixed_scaffold_compile_control_scout = (
            _run_fixed_scaffold_compile_control_scout_mode(
                source_cfg=cfg.source,
                noise_cfg=cfg.noise,
                scout_cfg=cfg.fixed_scaffold_compile_control_scout,
            )
            if bool(cfg.fixed_scaffold_compile_control_scout.enabled)
            else {}
        )
        fixed_scaffold_saved_theta_mitigation_matrix = (
            _run_fixed_scaffold_saved_theta_mitigation_matrix_mode(
                source_cfg=cfg.source,
                noise_cfg=cfg.noise,
                matrix_cfg=cfg.fixed_scaffold_saved_theta_mitigation_matrix,
            )
            if bool(cfg.fixed_scaffold_saved_theta_mitigation_matrix.enabled)
            else {}
        )
        fixed_scaffold_runtime_energy_only = (
            _run_fixed_scaffold_runtime_energy_only_mode(
                source_cfg=cfg.source,
                noise_cfg=cfg.noise,
                runtime_cfg=cfg.fixed_scaffold_runtime_energy_only,
            )
            if bool(cfg.fixed_scaffold_runtime_energy_only.enabled)
            else {}
        )
        fixed_scaffold_runtime_raw_baseline = (
            _run_fixed_scaffold_runtime_raw_baseline_mode(
                source_cfg=cfg.source,
                noise_cfg=cfg.noise,
                runtime_cfg=cfg.fixed_scaffold_runtime_raw_baseline,
            )
            if bool(cfg.fixed_scaffold_runtime_raw_baseline.enabled)
            else {}
        )
        fixed_scaffold_noisy_replay = (
            _run_fixed_scaffold_noisy_replay_mode(
                source_cfg=cfg.source,
                noise_cfg=cfg.noise,
                fixed_cfg=cfg.fixed_scaffold_replay,
            )
            if bool(cfg.fixed_scaffold_replay.enabled)
            else {}
        )
        fixed_scaffold_noise_attribution = (
            _run_fixed_scaffold_noise_attribution_mode(
                source_cfg=cfg.source,
                noise_cfg=cfg.noise,
                attribution_cfg=cfg.fixed_scaffold_attribution,
            )
            if bool(cfg.fixed_scaffold_attribution.enabled)
            else {}
        )
        fixed_scaffold_runtime_energy_only_sidecar_json = staged_cfg.artifacts.output_json.with_name(
            f"{staged_cfg.artifacts.tag}_fixed_scaffold_runtime_energy_only.json"
        )
        fixed_scaffold_saved_theta_mitigation_matrix_sidecar_json = staged_cfg.artifacts.output_json.with_name(
            f"{staged_cfg.artifacts.tag}_fixed_scaffold_saved_theta_mitigation_matrix.json"
        )
        fixed_scaffold_runtime_raw_baseline_sidecar_json = staged_cfg.artifacts.output_json.with_name(
            f"{staged_cfg.artifacts.tag}_fixed_scaffold_runtime_raw_baseline.json"
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
                    "ansatz_input_state_audit": bool(cfg.noise.include_full_circuit_audit),
                    "full_circuit_audit": bool(cfg.noise.include_full_circuit_audit),
                    "fixed_lean_scaffold_noisy_replay": bool(cfg.fixed_lean_replay.enabled),
                    "fixed_lean_noise_attribution": bool(cfg.fixed_lean_attribution.enabled),
                    "fixed_lean_compile_control_scout": bool(
                        cfg.fixed_lean_compile_control_scout.enabled
                    ),
                    "fixed_scaffold_runtime_energy_only": bool(
                        cfg.fixed_scaffold_runtime_energy_only.enabled
                    ),
                    "fixed_scaffold_runtime_raw_baseline": bool(
                        cfg.fixed_scaffold_runtime_raw_baseline.enabled
                    ),
                    "fixed_scaffold_noisy_replay": bool(cfg.fixed_scaffold_replay.enabled),
                    "fixed_scaffold_compile_control_scout": bool(
                        cfg.fixed_scaffold_compile_control_scout.enabled
                    ),
                    "fixed_scaffold_saved_theta_mitigation_matrix": bool(
                        cfg.fixed_scaffold_saved_theta_mitigation_matrix.enabled
                    ),
                    "fixed_scaffold_noise_attribution": bool(cfg.fixed_scaffold_attribution.enabled),
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
                "fixed_lean_compile_control_scout_contract": (
                    {
                        "matched_family_replay": False,
                        "structure_locked": True,
                        "parameter_optimization": False,
                        "objective": "energy_only",
                        "circuit_source": "imported_artifact_runtime_theta",
                        "noise_mode": str(cfg.fixed_lean_compile_control_scout.noise_mode),
                        "mitigation": "readout/mthree",
                        "symmetry": "off",
                        "compile_control_grid": {
                            "baseline_transpile_optimization_level": int(
                                cfg.fixed_lean_compile_control_scout.baseline_transpile_optimization_level
                            ),
                            "baseline_seed_transpiler": int(
                                cfg.fixed_lean_compile_control_scout.baseline_seed_transpiler
                            ),
                            "scout_transpile_optimization_levels": [
                                int(x) for x in cfg.fixed_lean_compile_control_scout.scout_transpile_optimization_levels
                            ],
                            "scout_seed_transpilers": [
                                int(x) for x in cfg.fixed_lean_compile_control_scout.scout_seed_transpilers
                            ],
                            "rank_policy": str(cfg.fixed_lean_compile_control_scout.rank_policy),
                        },
                    }
                    if bool(cfg.fixed_lean_compile_control_scout.enabled)
                    else {}
                ),
                "fixed_scaffold_noisy_replay_contract": (
                    {
                        "matched_family_replay": False,
                        "structure_locked": True,
                        "subject_kind": str(cfg.fixed_scaffold_replay.subject_kind),
                        "reps": int(cfg.fixed_scaffold_replay.reps),
                        "optimizer": str(cfg.fixed_scaffold_replay.method),
                        "noise_mode": str(cfg.fixed_scaffold_replay.noise_mode),
                        "theta_source": "imported_theta_runtime",
                        "execution_mode": "backend_scheduled",
                        "local_only": True,
                        "mitigation": "readout/mthree",
                        "local_gate_twirling": bool(
                            cfg.fixed_scaffold_replay.mitigation_config.get("local_gate_twirling", False)
                        ),
                        "local_dd_probe_sequence": cfg.fixed_scaffold_replay.local_dd_probe_sequence,
                    }
                    if bool(cfg.fixed_scaffold_replay.enabled)
                    else {}
                ),
                "fixed_scaffold_runtime_energy_only_contract": (
                    {
                        "matched_family_replay": False,
                        "structure_locked": True,
                        "subject_kind": str(cfg.fixed_scaffold_runtime_energy_only.subject_kind),
                        "parameter_optimization": False,
                        "objective": "energy_only",
                        "noise_mode": str(cfg.fixed_scaffold_runtime_energy_only.noise_mode),
                        "mitigation": "runtime_builtin_readout+twirling",
                        "session_policy": dict(cfg.fixed_scaffold_runtime_energy_only.runtime_session_config),
                        "runtime_profile": dict(cfg.fixed_scaffold_runtime_energy_only.runtime_profile_config),
                    }
                    if bool(cfg.fixed_scaffold_runtime_energy_only.enabled)
                    else {}
                ),
                "fixed_scaffold_runtime_raw_baseline_contract": (
                    {
                        "matched_family_replay": False,
                        "structure_locked": True,
                        "subject_kind": str(cfg.fixed_scaffold_runtime_raw_baseline.subject_kind),
                        "parameter_optimization": False,
                        "objective": "energy_only",
                        "noise_mode": str(cfg.fixed_scaffold_runtime_raw_baseline.noise_mode),
                        "execution_surface": "raw_measurement_v1",
                        "mitigation": "none",
                        "symmetry": "off",
                        "requested_diagonal_postprocessing": {
                            "mitigation": dict(
                                cfg.fixed_scaffold_runtime_raw_baseline.mitigation_config
                            ),
                            "symmetry_mitigation": dict(
                                cfg.fixed_scaffold_runtime_raw_baseline.symmetry_mitigation_config
                            ),
                            "order": "readout_then_symmetry",
                            "observable_scope": "full_register_diagonal_only",
                        },
                        "session_policy": dict(cfg.fixed_scaffold_runtime_raw_baseline.runtime_session_config),
                        "runtime_profile": dict(cfg.fixed_scaffold_runtime_raw_baseline.runtime_profile_config),
                        "raw_transport": str(cfg.fixed_scaffold_runtime_raw_baseline.raw_transport),
                        "raw_store_memory": bool(cfg.fixed_scaffold_runtime_raw_baseline.raw_store_memory),
                        "raw_artifact_path": cfg.fixed_scaffold_runtime_raw_baseline.raw_artifact_path,
                        "dd_probe": False,
                        "final_audit_zne": False,
                    }
                    if bool(cfg.fixed_scaffold_runtime_raw_baseline.enabled)
                    else {}
                ),
                "fixed_scaffold_compile_control_scout_contract": (
                    {
                        "matched_family_replay": False,
                        "structure_locked": True,
                        "subject_kind": str(cfg.fixed_scaffold_compile_control_scout.subject_kind),
                        "parameter_optimization": False,
                        "objective": "energy_only",
                        "circuit_source": "imported_artifact_runtime_theta",
                        "noise_mode": str(cfg.fixed_scaffold_compile_control_scout.noise_mode),
                        "mitigation": "readout/mthree",
                        "symmetry": "off",
                        "compile_control_grid": {
                            "baseline_transpile_optimization_level": int(
                                cfg.fixed_scaffold_compile_control_scout.baseline_transpile_optimization_level
                            ),
                            "baseline_seed_transpiler": int(
                                cfg.fixed_scaffold_compile_control_scout.baseline_seed_transpiler
                            ),
                            "scout_transpile_optimization_levels": [
                                int(x)
                                for x in cfg.fixed_scaffold_compile_control_scout.scout_transpile_optimization_levels
                            ],
                            "scout_seed_transpilers": [
                                int(x)
                                for x in cfg.fixed_scaffold_compile_control_scout.scout_seed_transpilers
                            ],
                            "rank_policy": str(cfg.fixed_scaffold_compile_control_scout.rank_policy),
                        },
                    }
                    if bool(cfg.fixed_scaffold_compile_control_scout.enabled)
                    else {}
                ),
                "fixed_scaffold_saved_theta_mitigation_matrix_contract": (
                    {
                        "matched_family_replay": False,
                        "structure_locked": True,
                        "subject_kind": str(cfg.fixed_scaffold_saved_theta_mitigation_matrix.subject_kind),
                        "parameter_optimization": False,
                        "objective": "fixed_theta_energy_matrix",
                        "circuit_source": "imported_artifact_runtime_theta",
                        "noise_mode": str(cfg.fixed_scaffold_saved_theta_mitigation_matrix.noise_mode),
                        "execution_mode": "backend_scheduled",
                        "local_only": True,
                        "readout_base": dict(
                            cfg.fixed_scaffold_saved_theta_mitigation_matrix.mitigation_config_base
                        ),
                        "symmetry": "off",
                        "compile_presets": [
                            dict(x) for x in cfg.fixed_scaffold_saved_theta_mitigation_matrix.compile_presets
                        ],
                        "zne_scales": [
                            float(x) for x in cfg.fixed_scaffold_saved_theta_mitigation_matrix.zne_scales
                        ],
                        "suppression_labels": [
                            str(x) for x in cfg.fixed_scaffold_saved_theta_mitigation_matrix.suppression_labels
                        ],
                        "selected_cells": [
                            str(x) for x in cfg.fixed_scaffold_saved_theta_mitigation_matrix.selected_cells
                        ],
                        "rank_policy": str(cfg.fixed_scaffold_saved_theta_mitigation_matrix.rank_policy),
                    }
                    if bool(cfg.fixed_scaffold_saved_theta_mitigation_matrix.enabled)
                    else {}
                ),
                "fixed_scaffold_noise_attribution_contract": (
                    {
                        "matched_family_replay": False,
                        "structure_locked": True,
                        "subject_kind": str(cfg.fixed_scaffold_attribution.subject_kind),
                        "parameter_optimization": False,
                        "circuit_source": "imported_artifact_runtime_theta",
                        "noise_mode": str(cfg.fixed_scaffold_attribution.noise_mode),
                        "slices": [str(x) for x in cfg.fixed_scaffold_attribution.slices],
                        "shared_transpile": True,
                        "mitigation": "none",
                        "symmetry": "off",
                    }
                    if bool(cfg.fixed_scaffold_attribution.enabled)
                    else {}
                ),
            },
            "settings": {
                **base_wf._jsonable(asdict(staged_cfg)),
                "noise": base_wf._jsonable(asdict(cfg.noise)),
                "fixed_lean_noisy_replay": base_wf._jsonable(asdict(cfg.fixed_lean_replay)),
                "fixed_lean_noise_attribution": base_wf._jsonable(asdict(cfg.fixed_lean_attribution)),
                "fixed_lean_compile_control_scout": base_wf._jsonable(
                    asdict(cfg.fixed_lean_compile_control_scout)
                ),
                "fixed_scaffold_noisy_replay": base_wf._jsonable(asdict(cfg.fixed_scaffold_replay)),
                "fixed_scaffold_runtime_energy_only": base_wf._jsonable(
                    asdict(cfg.fixed_scaffold_runtime_energy_only)
                ),
                "fixed_scaffold_runtime_raw_baseline": base_wf._jsonable(
                    asdict(cfg.fixed_scaffold_runtime_raw_baseline)
                ),
                "fixed_scaffold_compile_control_scout": base_wf._jsonable(
                    asdict(cfg.fixed_scaffold_compile_control_scout)
                ),
                "fixed_scaffold_saved_theta_mitigation_matrix": base_wf._jsonable(
                    asdict(cfg.fixed_scaffold_saved_theta_mitigation_matrix)
                ),
                "fixed_scaffold_noise_attribution": base_wf._jsonable(asdict(cfg.fixed_scaffold_attribution)),
            },
            "artifacts": {
                **base_wf._payload_artifacts(staged_cfg),
                "import_source_json": (
                    None if cfg.source.resolved_json is None else str(cfg.source.resolved_json)
                ),
                "fixed_scaffold_runtime_energy_only_json": (
                    str(fixed_scaffold_runtime_energy_only_sidecar_json)
                    if bool(cfg.fixed_scaffold_runtime_energy_only.enabled)
                    else None
                ),
                "fixed_scaffold_saved_theta_mitigation_matrix_json": (
                    str(fixed_scaffold_saved_theta_mitigation_matrix_sidecar_json)
                    if bool(cfg.fixed_scaffold_saved_theta_mitigation_matrix.enabled)
                    else None
                ),
                "fixed_scaffold_runtime_raw_baseline_json": (
                    str(fixed_scaffold_runtime_raw_baseline_sidecar_json)
                    if bool(cfg.fixed_scaffold_runtime_raw_baseline.enabled)
                    else None
                ),
            },
            "command": str(run_command_str),
            "import_source": base_wf._jsonable(asdict(cfg.source)),
            "dynamics_noiseless": {"profiles": {}},
            "dynamics_noisy": {"profiles": {}},
            "noisy_final_audit": {"profiles": {}},
            "dynamics_benchmarks": {"rows": []},
            "imported_prepared_state_audit": imported_prepared_state_audit,
            "imported_ansatz_input_state_audit": imported_ansatz_input_state_audit,
            "full_circuit_import_audit": full_circuit_import_audit,
            "fixed_lean_scaffold_noisy_replay": fixed_lean_scaffold_noisy_replay,
            "fixed_lean_noise_attribution": fixed_lean_noise_attribution,
            "fixed_lean_compile_control_scout": fixed_lean_compile_control_scout,
            "fixed_scaffold_noisy_replay": fixed_scaffold_noisy_replay,
            "fixed_scaffold_runtime_energy_only": fixed_scaffold_runtime_energy_only,
            "fixed_scaffold_runtime_raw_baseline": fixed_scaffold_runtime_raw_baseline,
            "fixed_scaffold_compile_control_scout": fixed_scaffold_compile_control_scout,
            "fixed_scaffold_saved_theta_mitigation_matrix": fixed_scaffold_saved_theta_mitigation_matrix,
            "fixed_scaffold_noise_attribution": fixed_scaffold_noise_attribution,
            "comparisons": {},
        }
        payload["summary"] = _build_noise_summary(payload)
        if bool(cfg.fixed_scaffold_runtime_energy_only.enabled):
            sidecar_payload = {
                "generated_utc": base_wf._now_utc(),
                "pipeline": "hh_fixed_scaffold_energy_only_runtime_eval_v1",
                "candidate_artifact_json": _repo_relative_str(cfg.source.resolved_json),
                "workflow_json": str(staged_cfg.artifacts.output_json),
                "backend_name": cfg.noise.backend_name,
                "result": base_wf._jsonable(fixed_scaffold_runtime_energy_only),
                "settings": {
                    "runtime": base_wf._jsonable(asdict(cfg.fixed_scaffold_runtime_energy_only)),
                },
            }
            base_wf._write_json(fixed_scaffold_runtime_energy_only_sidecar_json, sidecar_payload)
        if bool(cfg.fixed_scaffold_runtime_raw_baseline.enabled):
            sidecar_payload = {
                "generated_utc": base_wf._now_utc(),
                "pipeline": "hh_fixed_scaffold_runtime_raw_baseline_eval_v1",
                "candidate_artifact_json": _repo_relative_str(cfg.source.resolved_json),
                "workflow_json": str(staged_cfg.artifacts.output_json),
                "backend_name": cfg.noise.backend_name,
                "result": base_wf._jsonable(fixed_scaffold_runtime_raw_baseline),
                "settings": {
                    "runtime_raw_baseline": base_wf._jsonable(
                        asdict(cfg.fixed_scaffold_runtime_raw_baseline)
                    ),
                },
            }
            base_wf._write_json(fixed_scaffold_runtime_raw_baseline_sidecar_json, sidecar_payload)
        if bool(cfg.fixed_scaffold_saved_theta_mitigation_matrix.enabled):
            sidecar_payload = {
                "generated_utc": base_wf._now_utc(),
                "pipeline": "hh_fixed_scaffold_saved_theta_mitigation_matrix_eval_v1",
                "candidate_artifact_json": _repo_relative_str(cfg.source.resolved_json),
                "workflow_json": str(staged_cfg.artifacts.output_json),
                "backend_name": cfg.noise.backend_name,
                "result": base_wf._jsonable(fixed_scaffold_saved_theta_mitigation_matrix),
                "settings": {
                    "fixed_scaffold_saved_theta_mitigation_matrix": base_wf._jsonable(
                        asdict(cfg.fixed_scaffold_saved_theta_mitigation_matrix)
                    ),
                },
            }
            base_wf._write_json(fixed_scaffold_saved_theta_mitigation_matrix_sidecar_json, sidecar_payload)
        base_wf._write_json(staged_cfg.artifacts.output_json, payload)
        write_staged_hh_noise_pdf(payload, cfg, run_command_str)
        return payload
    stage_result = base_wf.run_stage_pipeline(staged_cfg)
    adaptive_realtime_checkpoint = run_adaptive_realtime_checkpoint_profile_noisy(stage_result, cfg)
    dynamics_noiseless = base_wf.run_noiseless_profiles(stage_result, staged_cfg)
    dynamics_noisy, noisy_final_audit, dynamics_benchmarks = run_noisy_profiles(stage_result, cfg)

    payload = base_wf.assemble_payload(
        cfg=staged_cfg,
        stage_result=stage_result,
        dynamics_noiseless=dynamics_noiseless,
        adaptive_realtime_checkpoint=adaptive_realtime_checkpoint,
        run_command=run_command_str,
    )
    payload["pipeline"] = "hh_staged_noise"
    if isinstance(payload.get("workflow_contract"), dict):
        payload["workflow_contract"]["noise_extension"] = "final_only_noisy_dynamics"
        payload["workflow_contract"]["imported_routes"] = {
            "prepared_state_audit": False,
            "ansatz_input_state_audit": False,
            "full_circuit_audit": False,
            "fixed_lean_scaffold_noisy_replay": False,
            "fixed_lean_noise_attribution": False,
            "fixed_lean_compile_control_scout": False,
            "fixed_scaffold_noisy_replay": False,
            "fixed_scaffold_runtime_energy_only": False,
            "fixed_scaffold_runtime_raw_baseline": False,
            "fixed_scaffold_compile_control_scout": False,
            "fixed_scaffold_saved_theta_mitigation_matrix": False,
            "fixed_scaffold_noise_attribution": False,
        }
    if isinstance(payload.get("settings"), dict):
        payload["settings"]["noise"] = base_wf._jsonable(asdict(cfg.noise))
        payload["settings"]["fixed_lean_noisy_replay"] = base_wf._jsonable(asdict(cfg.fixed_lean_replay))
        payload["settings"]["fixed_lean_noise_attribution"] = base_wf._jsonable(asdict(cfg.fixed_lean_attribution))
        payload["settings"]["fixed_lean_compile_control_scout"] = base_wf._jsonable(
            asdict(cfg.fixed_lean_compile_control_scout)
        )
        payload["settings"]["fixed_scaffold_noisy_replay"] = base_wf._jsonable(asdict(cfg.fixed_scaffold_replay))
        payload["settings"]["fixed_scaffold_runtime_energy_only"] = base_wf._jsonable(
            asdict(cfg.fixed_scaffold_runtime_energy_only)
        )
        payload["settings"]["fixed_scaffold_runtime_raw_baseline"] = base_wf._jsonable(
            asdict(cfg.fixed_scaffold_runtime_raw_baseline)
        )
        payload["settings"]["fixed_scaffold_compile_control_scout"] = base_wf._jsonable(
            asdict(cfg.fixed_scaffold_compile_control_scout)
        )
        payload["settings"]["fixed_scaffold_saved_theta_mitigation_matrix"] = base_wf._jsonable(
            asdict(cfg.fixed_scaffold_saved_theta_mitigation_matrix)
        )
        payload["settings"]["fixed_scaffold_noise_attribution"] = base_wf._jsonable(asdict(cfg.fixed_scaffold_attribution))
    payload["import_source"] = base_wf._jsonable(asdict(cfg.source))
    payload["dynamics_noisy"] = dynamics_noisy
    payload["noisy_final_audit"] = noisy_final_audit
    payload["dynamics_benchmarks"] = dynamics_benchmarks
    payload["fixed_lean_scaffold_noisy_replay"] = {}
    payload["fixed_lean_noise_attribution"] = {}
    payload["fixed_lean_compile_control_scout"] = {}
    payload["fixed_scaffold_noisy_replay"] = {}
    payload["fixed_scaffold_runtime_energy_only"] = {}
    payload["fixed_scaffold_runtime_raw_baseline"] = {}
    payload["fixed_scaffold_compile_control_scout"] = {}
    payload["fixed_scaffold_saved_theta_mitigation_matrix"] = {}
    payload["fixed_scaffold_noise_attribution"] = {}
    payload["comparisons"] = {
        **base_wf._compute_comparisons(payload),
        **noise_report._compute_comparisons(payload),
    }
    payload["summary"] = _build_noise_summary(payload)

    staged_cfg.artifacts.output_json.parent.mkdir(parents=True, exist_ok=True)
    base_wf._write_json(staged_cfg.artifacts.output_json, payload)
    write_staged_hh_noise_pdf(payload, cfg, run_command_str)
    return payload
