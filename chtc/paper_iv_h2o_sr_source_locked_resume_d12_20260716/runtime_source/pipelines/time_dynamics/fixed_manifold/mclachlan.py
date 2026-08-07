#!/usr/bin/env python3
"""Fixed-manifold HH McLachlan runner.

V1 contract:
- static or driven fixed-manifold McLachlan
- local exact geometry only
- stay-only controller behavior via large miss threshold
- two loader routes:
  1. replay-family ADAPT artifacts
  2. locked fixed-scaffold exports
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_realtime_checkpoint_controller import (  # noqa: E402
    ControllerDriveConfig,
    RealtimeCheckpointController,
)
from pipelines.hardcoded.hh_realtime_checkpoint_types import (  # noqa: E402
    RealtimeCheckpointConfig,
)
from pipelines.contracts.scaffold import ReplayScaffoldContext  # noqa: E402
from pipelines.scaffold.hh_fixed_manifold_loader import (  # noqa: E402
    FixedManifoldRunSpec,
    LoadedRunContext,
    _lock_replay_context_to_fixed_manifold,
    _make_replay_run_cfg,
    _statevector_from_named_payload_state,
    _validate_prepared_state_consistency,
    build_fixed_scaffold_context_from_payload,
    normalize_replay_payload,
)
from pipelines.scaffold.hh_vqe_from_adapt_family import (  # noqa: E402
    RunConfig as ReplayRunConfig,
    _build_hh_hamiltonian,
    build_replay_scaffold_context,
)
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix  # noqa: E402


DEFAULT_PARETO_ARTIFACT = Path(
    "artifacts/json/adapt_hh_L2_ecut1_pareto_lean_l2_phase3_powell_rerun_with_ansatz_input_20260321T214822Z.json"
)
DEFAULT_LOCKED_7TERM_ARTIFACT = Path("artifacts/json/hh_prune_nighthawk_gate_pruned_7term.json")


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _statevector_sha256_for_contract(psi: np.ndarray | Sequence[complex]) -> str:
    arr = np.asarray(psi, dtype=np.complex128).reshape(-1)
    payload = np.ascontiguousarray(
        np.stack([arr.real, arr.imag], axis=1).astype("<f8", copy=False)
    )
    return hashlib.sha256(payload.tobytes()).hexdigest()


def _strict_state_prep_contract_from_payload(
    payload: Mapping[str, Any],
    *,
    psi_ref: np.ndarray,
    psi_initial: np.ndarray,
) -> dict[str, Any]:
    ansatz_payload = payload.get("ansatz_input_state", {})
    initial_payload = payload.get("initial_state", {})
    if not isinstance(ansatz_payload, Mapping):
        ansatz_payload = {}
    if not isinstance(initial_payload, Mapping):
        initial_payload = {}
    ansatz_source = str(ansatz_payload.get("source", "payload"))
    initial_source = str(initial_payload.get("source", "payload"))
    return {
        "version": "strict_state_prep_v1",
        "state_prep_role": "prepared_seed_state_only",
        "exact_target_or_reference_trajectory": False,
        "feeds_controller_decisions": "prepared_ansatz_observables_only",
        "ansatz_input_state": {
            "role": "ansatz_input_state",
            "source": ansatz_source,
            "source_location": "payload.ansatz_input_state",
            "handoff_state_kind": str(
                ansatz_payload.get("handoff_state_kind", "reference_state")
            ),
            "source_allowlist": [ansatz_source],
            "state_sha256": _statevector_sha256_for_contract(psi_ref),
        },
        "initial_state": {
            "role": "prepared_ansatz_state",
            "source": initial_source,
            "source_location": "payload.initial_state",
            "handoff_state_kind": str(
                initial_payload.get("handoff_state_kind", "prepared_state")
            ),
            "source_allowlist": [initial_source],
            "state_sha256": _statevector_sha256_for_contract(psi_initial),
        },
    }


def _default_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_fixed_mclachlan_dual")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object payload at {path}.")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_drive_custom_weights(raw: str | None) -> tuple[float, ...] | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "":
        return None
    if text.startswith("["):
        vals = json.loads(text)
    else:
        vals = [float(x) for x in text.split(",") if x.strip()]
    return tuple(float(x) for x in vals)


def _controller_drive_config(
    *,
    loaded: LoadedRunContext,
    enable_drive: bool,
    drive_A: float,
    drive_omega: float,
    drive_tbar: float,
    drive_phi: float,
    drive_pattern: str,
    drive_custom_s: str | None,
    drive_include_identity: bool,
    drive_time_sampling: str,
    drive_t0: float,
    exact_steps_multiplier: int,
) -> ControllerDriveConfig | None:
    if not bool(enable_drive):
        return None
    custom_weights = None
    if str(drive_pattern).strip().lower() == "custom":
        custom_weights = _parse_drive_custom_weights(drive_custom_s)
        if custom_weights is None:
            raise ValueError("--drive-custom-s is required when --drive-pattern custom.")
    return ControllerDriveConfig(
        enabled=True,
        n_sites=int(loaded.cfg.L),
        ordering=str(loaded.cfg.ordering),
        drive_A=float(drive_A),
        drive_omega=float(drive_omega),
        drive_tbar=float(drive_tbar),
        drive_phi=float(drive_phi),
        drive_pattern=str(drive_pattern),
        drive_custom_weights=custom_weights,
        drive_include_identity=bool(drive_include_identity),
        drive_time_sampling=str(drive_time_sampling),
        drive_t0=float(drive_t0),
        exact_steps_multiplier=int(exact_steps_multiplier),
    )


def load_run_context(
    spec: FixedManifoldRunSpec,
    *,
    tag: str,
    lock_fixed_manifold: bool = True,
) -> LoadedRunContext:
    payload = _read_json(spec.artifact_json)
    psi_initial = _statevector_from_named_payload_state(payload, "initial_state")

    settings = payload.get("settings", {})
    problem_key = ""
    if isinstance(settings, Mapping):
        problem_key = str(settings.get("problem", "")).strip().lower()
    if problem_key and problem_key != "hh":
        from pipelines.scaffold.runtime_loader import load_scaffold_runtime_input_from_payload

        cfg = _make_replay_run_cfg(
            payload,
            artifact_json=spec.artifact_json,
            tag=tag,
            generator_family=str(spec.generator_family),
            fallback_family=str(spec.fallback_family),
            append_pool_family=str(spec.append_pool_family),
        )
        runtime_input = load_scaffold_runtime_input_from_payload(
            payload,
            artifact_json=spec.artifact_json,
            loader_mode=spec.loader_mode,
            tag=tag,
            generator_family=str(spec.generator_family),
            fallback_family=str(spec.fallback_family),
        )
        pool_source = runtime_input.candidate_pool_source
        family_info = {
            "requested": str(spec.generator_family),
            "resolved": str(pool_source.pool_key or spec.generator_family),
            "resolution_source": str(
                runtime_input.provenance.get("resolution_source", "generic_runtime_loader")
            ),
            "fallback_family": str(spec.fallback_family),
            "fallback_used": False,
            "warning": None,
        }
        pool_meta = {
            "family": str(pool_source.pool_key or spec.generator_family),
            "candidate_pool_complete": bool(pool_source.candidate_pool_complete),
            "family_pool_origin": str(
                runtime_input.extensions.get("generic_loader_summary", {}).get(
                    "family_pool_origin",
                    "generic_runtime_loader",
                )
            ),
        }
        if isinstance(pool_source.filter_payload, Mapping):
            pool_meta.update(dict(pool_source.filter_payload))
        payload_for_replay = dict(payload)
        payload_for_replay.setdefault(
            "strict_state_prep_contract",
            _strict_state_prep_contract_from_payload(
                payload,
                psi_ref=np.asarray(runtime_input.psi_ref, dtype=complex).reshape(-1),
                psi_initial=np.asarray(runtime_input.psi_initial, dtype=complex).reshape(-1),
            ),
        )
        replay_context = ReplayScaffoldContext(
            cfg=cfg,
            h_poly=runtime_input.resolved_problem.hamiltonian,
            psi_ref=np.asarray(runtime_input.psi_ref, dtype=complex).reshape(-1),
            payload_in=payload_for_replay,
            family_info=family_info,
            family_pool=tuple(runtime_input.candidate_pool_terms),
            pool_meta=pool_meta,
            replay_terms=tuple(runtime_input.selected_terms),
            base_layout=runtime_input.base_layout,
            adapt_theta_runtime=np.asarray(runtime_input.theta_runtime, dtype=float).reshape(-1),
            adapt_theta_logical=(
                np.asarray(runtime_input.theta_logical, dtype=float).reshape(-1)
                if runtime_input.theta_logical is not None
                else np.asarray([], dtype=float)
            ),
            adapt_depth=int(len(runtime_input.selected_terms)),
            handoff_state_kind=str(
                payload.get("initial_state", {}).get("handoff_state_kind", "prepared_state")
                if isinstance(payload.get("initial_state", None), Mapping)
                else "prepared_state"
            ),
            provenance_source="generic_runtime_loader",
            family_terms_count=int(len(runtime_input.candidate_pool_terms)),
            append_family_info=dict(family_info),
            append_family_pool=tuple(runtime_input.candidate_pool_terms),
            append_pool_meta=dict(pool_meta),
            append_family_terms_count=int(len(runtime_input.candidate_pool_terms)),
        )
        loader_summary = {
            "loader_mode": "replay_family",
            "input_artifact_json": str(spec.artifact_json),
            "problem_family": str(runtime_input.resolved_problem.family_key),
            "resolved_family": str(family_info["resolved"]),
            "resolution_source": str(family_info["resolution_source"]),
            "append_pool_family_requested": str(spec.append_pool_family),
            "append_resolved_family": str(family_info["resolved"]),
            "append_resolution_source": "generic_runtime_loader",
            "candidate_pool_complete": bool(pool_source.candidate_pool_complete),
            "replay_candidate_pool_complete": bool(pool_source.candidate_pool_complete),
            "append_candidate_pool_complete": bool(pool_source.candidate_pool_complete),
            "fixed_manifold_locked": False,
            "lock_fixed_manifold_requested": bool(lock_fixed_manifold),
            "family_pool_origin": str(pool_meta.get("family_pool_origin", "generic_runtime_loader")),
            "logical_operator_count": int(runtime_input.base_layout.logical_parameter_count),
            "runtime_parameter_count": int(runtime_input.base_layout.runtime_parameter_count),
            "initial_state_source": str(
                runtime_input.extensions.get("generic_loader_summary", {}).get(
                    "initial_state_source",
                    "payload",
                )
            ),
            "prepared_state_reconstruction_error": float(
                runtime_input.extensions.get("generic_loader_summary", {}).get(
                    "prepared_state_reconstruction_error",
                    0.0,
                )
            ),
        }
        return LoadedRunContext(
            spec=spec,
            cfg=cfg,
            payload=payload_for_replay,
            replay_context=replay_context,
            psi_initial=np.asarray(runtime_input.psi_initial, dtype=complex).reshape(-1),
            loader_summary=loader_summary,
            runtime_input=runtime_input,
        )

    if str(spec.loader_mode) == "replay_family":
        normalized = normalize_replay_payload(payload)
        continuation_lifted = (
            not isinstance(payload.get("continuation", None), Mapping)
            and isinstance(normalized.get("continuation", None), Mapping)
        )
        cfg = _make_replay_run_cfg(
            normalized,
            artifact_json=spec.artifact_json,
            tag=tag,
            generator_family=str(spec.generator_family),
            fallback_family=str(spec.fallback_family),
            append_pool_family=str(spec.append_pool_family),
        )
        psi_ref = _statevector_from_named_payload_state(normalized, "ansatz_input_state")
        h_poly = _build_hh_hamiltonian(cfg)
        replay_context = build_replay_scaffold_context(
            cfg,
            h_poly=h_poly,
            psi_ref=psi_ref,
            payload_in=normalized,
        )
        if bool(lock_fixed_manifold):
            replay_context = _lock_replay_context_to_fixed_manifold(replay_context)
        loader_summary = {
            "loader_mode": "replay_family",
            "input_artifact_json": str(spec.artifact_json),
            "normalized_continuation_lifted": bool(continuation_lifted),
            "resolved_family": str(replay_context.family_info.get("resolved", "")),
            "resolution_source": str(replay_context.family_info.get("resolution_source", "")),
            "append_pool_family_requested": str(spec.append_pool_family),
            "append_resolved_family": str(
                (replay_context.append_family_info or {}).get("resolved", "")
            ),
            "append_resolution_source": str(
                (replay_context.append_family_info or {}).get("resolution_source", "")
            ),
            "candidate_pool_complete": bool(
                (
                    replay_context.pool_meta
                    if replay_context.append_pool_meta is None
                    else replay_context.append_pool_meta
                ).get(
                    "candidate_pool_complete",
                    False,
                )
            ),
            "replay_candidate_pool_complete": bool(
                replay_context.pool_meta.get("candidate_pool_complete", False)
            ),
            "append_candidate_pool_complete": bool(
                ({} if replay_context.append_pool_meta is None else replay_context.append_pool_meta).get(
                    "candidate_pool_complete",
                    False,
                )
            ),
            "fixed_manifold_locked": bool(replay_context.pool_meta.get("fixed_manifold_locked", False)),
            "lock_fixed_manifold_requested": bool(lock_fixed_manifold),
            "family_pool_origin": replay_context.pool_meta.get("family_pool_origin", None),
            "append_pool_source": (
                {} if replay_context.append_pool_meta is None else replay_context.append_pool_meta
            ).get(
                "append_pool_source",
                None,
            ),
            "append_family_terms_count": (
                None
                if replay_context.append_family_terms_count is None
                else int(replay_context.append_family_terms_count)
            ),
            "logical_operator_count": int(replay_context.base_layout.logical_parameter_count),
            "runtime_parameter_count": int(replay_context.base_layout.runtime_parameter_count),
        }
        payload_used = normalized
    elif str(spec.loader_mode) == "fixed_scaffold":
        cfg = _make_replay_run_cfg(
            payload,
            artifact_json=spec.artifact_json,
            tag=tag,
            generator_family="fixed_scaffold_locked",
            fallback_family=str(spec.fallback_family),
            append_pool_family=str(spec.append_pool_family),
        )
        replay_context = build_fixed_scaffold_context_from_payload(payload, cfg=cfg)
        if bool(lock_fixed_manifold):
            replay_context = _lock_replay_context_to_fixed_manifold(replay_context)
        loader_summary = {
            "loader_mode": "fixed_scaffold",
            "input_artifact_json": str(spec.artifact_json),
            "fixed_scaffold_kind": replay_context.pool_meta.get("fixed_scaffold_kind", None),
            "structure_locked": bool(replay_context.pool_meta.get("structure_locked", True)),
            "route_family": replay_context.pool_meta.get("route_family", None),
            "candidate_pool_complete": bool(
                (
                    replay_context.pool_meta
                    if replay_context.append_pool_meta is None
                    else replay_context.append_pool_meta
                ).get(
                    "candidate_pool_complete",
                    False,
                )
            ),
            "append_pool_family_requested": str(spec.append_pool_family),
            "append_resolved_family": str(
                (replay_context.append_family_info or {}).get("resolved", "")
            ),
            "append_resolution_source": str(
                (replay_context.append_family_info or {}).get("resolution_source", "")
            ),
            "replay_candidate_pool_complete": bool(
                replay_context.pool_meta.get("candidate_pool_complete", False)
            ),
            "append_candidate_pool_complete": bool(
                ({} if replay_context.append_pool_meta is None else replay_context.append_pool_meta).get(
                    "candidate_pool_complete",
                    False,
                )
            ),
            "fixed_manifold_locked": bool(replay_context.pool_meta.get("fixed_manifold_locked", False)),
            "lock_fixed_manifold_requested": bool(lock_fixed_manifold),
            "family_pool_origin": replay_context.pool_meta.get("family_pool_origin", None),
            "append_pool_source": (
                {} if replay_context.append_pool_meta is None else replay_context.append_pool_meta
            ).get(
                "append_pool_source",
                None,
            ),
            "append_family_terms_count": (
                None
                if replay_context.append_family_terms_count is None
                else int(replay_context.append_family_terms_count)
            ),
            "logical_operator_count": int(replay_context.base_layout.logical_parameter_count),
            "runtime_parameter_count": int(replay_context.base_layout.runtime_parameter_count),
        }
        payload_used = dict(payload)
    else:
        raise ValueError(f"Unsupported loader_mode {spec.loader_mode!r}.")

    reconstruction_error = _validate_prepared_state_consistency(
        replay_context,
        psi_initial,
        tol=1.0e-10,
    )
    loader_summary["prepared_state_reconstruction_error"] = float(reconstruction_error)

    return LoadedRunContext(
        spec=spec,
        cfg=cfg,
        payload=dict(payload_used),
        replay_context=replay_context,
        psi_initial=np.asarray(psi_initial, dtype=complex).reshape(-1),
        loader_summary=dict(loader_summary),
    )


def summarize_result_artifact(
    *,
    trajectory: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    fidelity_vals = [float(row.get("fidelity_exact", float("nan"))) for row in trajectory]
    rho_vals = [float(row.get("rho_miss", float("nan"))) for row in trajectory]
    energy_err_vals = [
        float(row.get("abs_energy_total_error", float("nan"))) for row in trajectory
    ]
    condition_vals = [
        float(row.get("baseline_geometry", {}).get("condition_number", float("nan")))
        for row in trajectory
    ]

    def _nanmax(vals: Sequence[float]) -> float:
        arr = np.asarray(list(vals), dtype=float)
        return float(np.nanmax(arr)) if arr.size > 0 else float("nan")

    def _nanmin(vals: Sequence[float]) -> float:
        arr = np.asarray(list(vals), dtype=float)
        return float(np.nanmin(arr)) if arr.size > 0 else float("nan")

    return {
        "trajectory_points": int(len(list(trajectory))),
        "fidelity_min": _nanmin(fidelity_vals),
        "fidelity_max": _nanmax(fidelity_vals),
        "rho_miss_max": _nanmax(rho_vals),
        "abs_energy_total_error_max": _nanmax(energy_err_vals),
        "condition_number_max": _nanmax(condition_vals),
        "condition_number_final": (
            float(condition_vals[-1]) if len(condition_vals) > 0 else float("nan")
        ),
        "final_logical_block_count": int(summary.get("final_logical_block_count", 0)),
        "final_runtime_parameter_count": int(summary.get("final_runtime_parameter_count", 0)),
    }


def run_fixed_manifold_exact(
    spec: FixedManifoldRunSpec,
    *,
    tag: str,
    output_dir: Path,
    t_final: float,
    num_times: int,
    miss_threshold: float,
    gain_ratio_threshold: float,
    append_margin_abs: float,
    enable_drive: bool = False,
    drive_A: float = 0.0,
    drive_omega: float = 1.0,
    drive_tbar: float = 1.0,
    drive_phi: float = 0.0,
    drive_pattern: str = "staggered",
    drive_custom_s: str | None = None,
    drive_include_identity: bool = False,
    drive_time_sampling: str = "midpoint",
    drive_t0: float = 0.0,
    exact_steps_multiplier: int = 1,
) -> dict[str, Any]:
    loaded = load_run_context(spec, tag=tag)
    controller_drive_cfg = _controller_drive_config(
        loaded=loaded,
        enable_drive=bool(enable_drive),
        drive_A=float(drive_A),
        drive_omega=float(drive_omega),
        drive_tbar=float(drive_tbar),
        drive_phi=float(drive_phi),
        drive_pattern=str(drive_pattern),
        drive_custom_s=drive_custom_s,
        drive_include_identity=bool(drive_include_identity),
        drive_time_sampling=str(drive_time_sampling),
        drive_t0=float(drive_t0),
        exact_steps_multiplier=int(exact_steps_multiplier),
    )
    controller_cfg = RealtimeCheckpointConfig(
        mode="exact_v1",
        miss_threshold=float(miss_threshold),
        gain_ratio_threshold=float(gain_ratio_threshold),
        append_margin_abs=float(append_margin_abs),
    )
    hmat = np.asarray(hamiltonian_matrix(loaded.replay_context.h_poly), dtype=complex)
    controller = RealtimeCheckpointController(
        cfg=controller_cfg,
        replay_context=loaded.replay_context,
        h_poly=loaded.replay_context.h_poly,
        hmat=hmat,
        psi_initial=loaded.psi_initial,
        best_theta=loaded.replay_context.adapt_theta_runtime,
        allow_repeats=False,
        t_final=float(t_final),
        num_times=int(num_times),
        drive_config=controller_drive_cfg,
    )
    result = controller.run()
    extra_summary = summarize_result_artifact(
        trajectory=result.trajectory,
        summary=result.summary,
    )

    settings = loaded.payload.get("settings", {})
    if not isinstance(settings, Mapping):
        settings = {}

    run_payload = {
        "generated_utc": _now_utc(),
        "pipeline": "hh_fixed_manifold_exact_mclachlan_v1",
        "run_name": str(spec.name),
        "input_artifact_json": str(spec.artifact_json),
        "loader": dict(loaded.loader_summary),
        "manifest": {
            "model_family": "Hubbard-Holstein",
            "ansatz_type": str(spec.name),
            "drive_enabled": bool(controller_drive_cfg is not None),
            "t": float(settings.get("t", loaded.cfg.t)),
            "U": float(settings.get("u", loaded.cfg.u)),
            "dv": float(settings.get("dv", loaded.cfg.dv)),
            "omega0": float(settings.get("omega0", loaded.cfg.omega0)),
            "g_ep": float(settings.get("g_ep", loaded.cfg.g_ep)),
            "n_ph_max": int(settings.get("n_ph_max", loaded.cfg.n_ph_max)),
            "L": int(settings.get("L", loaded.cfg.L)),
        },
        "run_config": {
            "t_final": float(t_final),
            "num_times": int(num_times),
            "allow_repeats": False,
            "decision_mode": "exact_v1",
            "structure_policy": "fixed_manifold_locked_pool",
            "controller": asdict(controller_cfg),
            "effective_pool_kind": "replay_terms_only",
            "projection_time_sampling": (
                "left"
                if controller_drive_cfg is None
                else str(controller_drive_cfg.drive_time_sampling)
            ),
        },
        "drive_profile": dict(result.reference.get("drive_profile", {})) if controller_drive_cfg is not None else None,
        "summary": dict(result.summary),
        "extra_summary": dict(extra_summary),
        "reference": dict(result.reference),
        "trajectory": [dict(row) for row in result.trajectory],
        "ledger": [dict(row) for row in result.ledger],
    }
    output_path = Path(output_dir) / f"{spec.name}.json"
    _write_json(output_path, run_payload)

    return {
        "name": str(spec.name),
        "status": "completed",
        "input_artifact_json": str(spec.artifact_json),
        "output_json": str(output_path),
        "loader_mode": str(loaded.loader_summary.get("loader_mode", "")),
        "resolved_family": str(loaded.replay_context.family_info.get("resolved", "")),
        "manifest": dict(run_payload["manifest"]),
        "drive_profile": (None if run_payload["drive_profile"] is None else dict(run_payload["drive_profile"])),
        "summary": dict(result.summary),
        "extra_summary": dict(extra_summary),
        "loader": dict(loaded.loader_summary),
    }


def build_compare_summary(
    *,
    run_records: Sequence[Mapping[str, Any]],
    tag: str,
    output_dir: Path,
    t_final: float,
    num_times: int,
    miss_threshold: float,
    drive_enabled: bool = False,
    drive_profile: Mapping[str, Any] | None = None,
    projection_time_sampling: str = "left",
    reference_steps_multiplier: int = 1,
) -> dict[str, Any]:
    successes = [dict(row) for row in run_records if str(row.get("status", "")) == "completed"]
    failures = [dict(row) for row in run_records if str(row.get("status", "")) != "completed"]

    best_min_fidelity = None
    leanest_runtime = None
    if successes:
        best_min_fidelity = max(
            successes,
            key=lambda row: float(row.get("extra_summary", {}).get("fidelity_min", float("-inf"))),
        )
        leanest_runtime = min(
            successes,
            key=lambda row: int(
                row.get("extra_summary", {}).get("final_runtime_parameter_count", 10**9)
            ),
        )

    return {
        "generated_utc": _now_utc(),
        "pipeline": "hh_fixed_manifold_exact_mclachlan_compare_v1",
        "tag": str(tag),
        "output_dir": str(output_dir),
        "manifest": {
            "model_family": "Hubbard-Holstein",
            "drive_enabled": bool(drive_enabled),
            "decision_mode": "exact_v1",
            "structure_policy": "fixed_manifold_locked_pool",
            "effective_pool_kind": "replay_terms_only",
            "t_final": float(t_final),
            "num_times": int(num_times),
            "miss_threshold": float(miss_threshold),
            "projection_time_sampling": str(projection_time_sampling),
            "reference_steps_multiplier": int(reference_steps_multiplier),
        },
        "drive_profile": (None if drive_profile is None else dict(drive_profile)),
        "completed_runs": int(len(successes)),
        "failed_runs": int(len(failures)),
        "runs": [dict(row) for row in run_records],
        "frontier_summary": {
            "best_min_fidelity_run": (None if best_min_fidelity is None else str(best_min_fidelity["name"])),
            "leanest_runtime_run": (None if leanest_runtime is None else str(leanest_runtime["name"])),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed-manifold HH McLachlan exact/local comparisons.",
    )
    parser.add_argument("--tag", type=str, default=_default_tag())
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--compare-summary-json", type=str, default=None)
    parser.add_argument("--t-final", type=float, default=10.0)
    parser.add_argument("--num-times", type=int, default=135)
    parser.add_argument(
        "--miss-threshold",
        type=float,
        default=1.0e9,
        help="Recorded controller knob; default append pool remains locked to replay_terms.",
    )
    parser.add_argument(
        "--gain-ratio-threshold",
        type=float,
        default=1.0e-9,
        help="Recorded controller knob; default append pool remains locked to replay_terms.",
    )
    parser.add_argument(
        "--append-margin-abs",
        type=float,
        default=1.0e-12,
        help="Recorded controller knob; default append pool remains locked to replay_terms.",
    )
    parser.add_argument(
        "--append-pool-family",
        "--candidate-pool-family",
        dest="append_pool_family",
        default="match_replay",
        help="Append/candidate-pool family forwarded to each run spec.",
    )
    parser.add_argument(
        "--pareto-artifact-json",
        type=str,
        default=str(DEFAULT_PARETO_ARTIFACT),
    )
    parser.add_argument(
        "--locked-7term-artifact-json",
        type=str,
        default=str(DEFAULT_LOCKED_7TERM_ARTIFACT),
    )
    parser.add_argument("--enable-drive", action="store_true", help="Enable time-dependent onsite density drive.")
    parser.add_argument("--drive-A", type=float, default=0.0)
    parser.add_argument("--drive-omega", type=float, default=1.0)
    parser.add_argument("--drive-tbar", type=float, default=1.0)
    parser.add_argument("--drive-phi", type=float, default=0.0)
    parser.add_argument(
        "--drive-pattern",
        choices=["dimer_bias", "staggered", "custom"],
        default="staggered",
    )
    parser.add_argument("--drive-custom-s", type=str, default=None)
    parser.add_argument("--drive-include-identity", action="store_true")
    parser.add_argument(
        "--drive-time-sampling",
        choices=["midpoint", "left", "right"],
        default="midpoint",
    )
    parser.add_argument("--drive-t0", type=float, default=0.0)
    parser.add_argument("--exact-steps-multiplier", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("artifacts/agent_runs") / str(args.tag)
    )
    run_specs = [
        FixedManifoldRunSpec(
            name="pareto_lean_l2",
            artifact_json=Path(args.pareto_artifact_json),
            loader_mode="replay_family",
            generator_family="match_adapt",
            fallback_family="full_meta",
            append_pool_family=str(args.append_pool_family),
        ),
        FixedManifoldRunSpec(
            name="locked_7term",
            artifact_json=Path(args.locked_7term_artifact_json),
            loader_mode="fixed_scaffold",
            generator_family="fixed_scaffold_locked",
            fallback_family="full_meta",
            append_pool_family=str(args.append_pool_family),
        ),
    ]

    drive_profile = None
    if bool(args.enable_drive):
        drive_profile = {
            "A": float(args.drive_A),
            "omega": float(args.drive_omega),
            "tbar": float(args.drive_tbar),
            "phi": float(args.drive_phi),
            "pattern": str(args.drive_pattern),
            "custom_weights": (
                None
                if str(args.drive_pattern).strip().lower() != "custom"
                else list(_parse_drive_custom_weights(args.drive_custom_s) or [])
            ),
            "include_identity": bool(args.drive_include_identity),
            "time_sampling": str(args.drive_time_sampling),
            "t0": float(args.drive_t0),
        }
    run_records: list[dict[str, Any]] = []
    failures = 0
    for spec in run_specs:
        try:
            run_records.append(
                run_fixed_manifold_exact(
                    spec,
                    tag=str(args.tag),
                    output_dir=output_dir,
                    t_final=float(args.t_final),
                    num_times=int(args.num_times),
                    miss_threshold=float(args.miss_threshold),
                    gain_ratio_threshold=float(args.gain_ratio_threshold),
                    append_margin_abs=float(args.append_margin_abs),
                    enable_drive=bool(args.enable_drive),
                    drive_A=float(args.drive_A),
                    drive_omega=float(args.drive_omega),
                    drive_tbar=float(args.drive_tbar),
                    drive_phi=float(args.drive_phi),
                    drive_pattern=str(args.drive_pattern),
                    drive_custom_s=args.drive_custom_s,
                    drive_include_identity=bool(args.drive_include_identity),
                    drive_time_sampling=str(args.drive_time_sampling),
                    drive_t0=float(args.drive_t0),
                    exact_steps_multiplier=int(args.exact_steps_multiplier),
                )
            )
        except Exception as exc:
            failures += 1
            run_records.append(
                {
                    "name": str(spec.name),
                    "status": "failed",
                    "input_artifact_json": str(spec.artifact_json),
                    "loader_mode": str(spec.loader_mode),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    summary_payload = build_compare_summary(
        run_records=run_records,
        tag=str(args.tag),
        output_dir=output_dir,
        t_final=float(args.t_final),
        num_times=int(args.num_times),
        miss_threshold=float(args.miss_threshold),
        drive_enabled=bool(args.enable_drive),
        drive_profile=drive_profile,
        projection_time_sampling=(str(args.drive_time_sampling) if bool(args.enable_drive) else "left"),
        reference_steps_multiplier=(int(args.exact_steps_multiplier) if bool(args.enable_drive) else 1),
    )
    summary_path = (
        Path(args.compare_summary_json)
        if args.compare_summary_json is not None
        else output_dir / "summary.json"
    )
    _write_json(summary_path, summary_payload)
    return 1 if int(failures) > 0 else 0


__all__ = [
    "DEFAULT_LOCKED_7TERM_ARTIFACT",
    "DEFAULT_PARETO_ARTIFACT",
    "FixedManifoldRunSpec",
    "LoadedRunContext",
    "build_compare_summary",
    "build_fixed_scaffold_context_from_payload",
    "load_run_context",
    "main",
    "normalize_replay_payload",
    "run_fixed_manifold_exact",
    "summarize_result_artifact",
]


if __name__ == "__main__":
    raise SystemExit(main())
