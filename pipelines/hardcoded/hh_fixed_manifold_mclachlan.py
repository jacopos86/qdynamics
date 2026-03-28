#!/usr/bin/env python3
"""Fixed-manifold HH McLachlan runner.

V1 contract:
- static / no-drive only
- local exact geometry only
- stay-only controller behavior via large miss threshold
- two loader routes:
  1. replay-family ADAPT artifacts
  2. locked fixed-scaffold exports
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_continuation_generators import (  # noqa: E402
    rebuild_polynomial_from_serialized_terms,
)
from pipelines.hardcoded.hh_realtime_checkpoint_controller import (  # noqa: E402
    RealtimeCheckpointController,
)
from pipelines.hardcoded.hh_realtime_checkpoint_types import (  # noqa: E402
    RealtimeCheckpointConfig,
)
from pipelines.hardcoded.hh_vqe_from_adapt_family import (  # noqa: E402
    ReplayScaffoldContext,
    RunConfig as ReplayRunConfig,
    _amplitudes_qn_to_q0_to_statevector,
    _build_hh_hamiltonian,
    build_replay_scaffold_context,
)
from src.quantum.ansatz_parameterization import (  # noqa: E402
    deserialize_layout,
    project_runtime_theta_block_mean,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor  # noqa: E402
from src.quantum.vqe_latex_python_pairs import (  # noqa: E402
    AnsatzTerm,
    hamiltonian_matrix,
)


DEFAULT_PARETO_ARTIFACT = Path(
    "artifacts/json/adapt_hh_L2_ecut1_pareto_lean_l2_phase3_powell_rerun_with_ansatz_input_20260321T214822Z.json"
)
DEFAULT_LOCKED_7TERM_ARTIFACT = Path("artifacts/json/hh_prune_nighthawk_gate_pruned_7term.json")


@dataclass(frozen=True)
class FixedManifoldRunSpec:
    name: str
    artifact_json: Path
    loader_mode: str
    generator_family: str = "match_adapt"
    fallback_family: str = "full_meta"


@dataclass(frozen=True)
class LoadedRunContext:
    spec: FixedManifoldRunSpec
    cfg: ReplayRunConfig
    payload: dict[str, Any]
    replay_context: ReplayScaffoldContext
    psi_initial: np.ndarray
    loader_summary: dict[str, Any]


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _boolish(raw: Any, default: bool) -> bool:
    if raw is None:
        return bool(default)
    if isinstance(raw, bool):
        return bool(raw)
    if isinstance(raw, (int, np.integer)):
        return bool(int(raw))
    if isinstance(raw, str):
        lowered = str(raw).strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(raw)


def _half_filled_particles(num_sites: int) -> tuple[int, int]:
    return ((int(num_sites) + 1) // 2, int(num_sites) // 2)


def _resolve_sector_counts(
    settings: Mapping[str, Any],
    adapt: Mapping[str, Any],
    *,
    L: int,
) -> tuple[int, int]:
    n_up_default, n_dn_default = _half_filled_particles(L)
    settings_up = settings.get("sector_n_up", None)
    settings_dn = settings.get("sector_n_dn", None)

    num_particles = adapt.get("num_particles", {})
    if not isinstance(num_particles, Mapping):
        num_particles = {}
    adapt_up = num_particles.get("n_up", None)
    adapt_dn = num_particles.get("n_dn", None)

    if settings_up is not None and adapt_up is not None and int(settings_up) != int(adapt_up):
        raise ValueError(
            f"Sector mismatch for n_up: settings={int(settings_up)} vs adapt_vqe.num_particles={int(adapt_up)}."
        )
    if settings_dn is not None and adapt_dn is not None and int(settings_dn) != int(adapt_dn):
        raise ValueError(
            f"Sector mismatch for n_dn: settings={int(settings_dn)} vs adapt_vqe.num_particles={int(adapt_dn)}."
        )

    n_up = (
        int(settings_up)
        if settings_up is not None
        else (int(adapt_up) if adapt_up is not None else int(n_up_default))
    )
    n_dn = (
        int(settings_dn)
        if settings_dn is not None
        else (int(adapt_dn) if adapt_dn is not None else int(n_dn_default))
    )
    return int(n_up), int(n_dn)


def _statevector_from_named_payload_state(payload: Mapping[str, Any], state_key: str) -> np.ndarray:
    state_payload = payload.get(state_key, None)
    if not isinstance(state_payload, Mapping):
        raise ValueError(f"Input JSON missing '{state_key}' state payload.")
    amps = state_payload.get("amplitudes_qn_to_q0", None)
    if not isinstance(amps, Mapping):
        raise ValueError(f"Input JSON missing {state_key}.amplitudes_qn_to_q0.")
    nq_total_raw = state_payload.get("nq_total", None)
    nq_total = int(nq_total_raw) if nq_total_raw is not None else len(next(iter(amps.keys()), ""))
    if int(nq_total) <= 0:
        raise ValueError(f"Could not infer nq_total from {state_key} amplitudes.")
    return _amplitudes_qn_to_q0_to_statevector(amps, nq=int(nq_total))


def normalize_replay_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(payload))
    top_level = out.get("continuation", None)
    adapt = out.get("adapt_vqe", None)
    if not isinstance(adapt, Mapping):
        return out
    nested = adapt.get("continuation", None)
    if isinstance(top_level, Mapping) and isinstance(nested, Mapping) and dict(top_level) != dict(nested):
        raise ValueError(
            "Replay payload has conflicting continuation objects at top-level and adapt_vqe.continuation."
        )
    if isinstance(top_level, Mapping):
        return out
    if isinstance(nested, Mapping):
        out["continuation"] = copy.deepcopy(dict(nested))
    return out


def _aggregate_serialized_terms(serialized_terms: Sequence[Mapping[str, Any]]) -> dict[str, complex]:
    out: dict[str, complex] = {}
    for raw in serialized_terms:
        if not isinstance(raw, Mapping):
            continue
        label = str(raw.get("pauli_exyz", "")).strip()
        nq = int(raw.get("nq", 0))
        if label == "" or int(nq) <= 0:
            continue
        coeff = complex(float(raw.get("coeff_re", 0.0)), float(raw.get("coeff_im", 0.0)))
        out[label] = complex(out.get(label, 0.0 + 0.0j) + coeff)
    return out


def _aggregate_polynomial_terms(poly: Any, *, tol: float = 1.0e-12) -> dict[str, complex]:
    out: dict[str, complex] = {}
    for term in poly.return_polynomial():
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tol):
            continue
        out[label] = complex(out.get(label, 0.0 + 0.0j) + coeff)
    return out


def _validate_locked_scaffold_payload(payload: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    adapt = payload.get("adapt_vqe", None)
    if not isinstance(adapt, Mapping):
        raise ValueError("Input JSON missing adapt_vqe for fixed scaffold import.")
    fixed_meta = adapt.get("fixed_scaffold_metadata", {})
    if not isinstance(fixed_meta, Mapping):
        raise ValueError("Locked fixed scaffold import requires adapt_vqe.fixed_scaffold_metadata.")
    pool_type = str(adapt.get("pool_type", "")).strip().lower()
    if pool_type != "fixed_scaffold_locked":
        raise ValueError(
            f"Locked fixed scaffold import requires adapt_vqe.pool_type='fixed_scaffold_locked', got {pool_type!r}."
        )
    if not bool(adapt.get("structure_locked", False)):
        raise ValueError("Locked fixed scaffold import requires adapt_vqe.structure_locked=true.")
    route_family = str(fixed_meta.get("route_family", "")).strip().lower()
    if route_family != "locked_imported_scaffold_v1":
        raise ValueError(
            "Locked fixed scaffold import requires "
            "adapt_vqe.fixed_scaffold_metadata.route_family='locked_imported_scaffold_v1'."
        )
    return dict(adapt), dict(fixed_meta)


def _lock_replay_context_to_fixed_manifold(
    replay_context: ReplayScaffoldContext,
) -> ReplayScaffoldContext:
    pool_meta = dict(replay_context.pool_meta)
    pool_meta["candidate_pool_complete"] = True
    pool_meta["fixed_manifold_locked"] = True
    pool_meta["family_pool_origin"] = "replay_terms_only"
    return replace(
        replay_context,
        family_pool=tuple(replay_context.replay_terms),
        pool_meta=dict(pool_meta),
        family_terms_count=int(len(replay_context.replay_terms)),
    )


def _validate_prepared_state_consistency(
    replay_context: ReplayScaffoldContext,
    psi_initial: np.ndarray,
    *,
    tol: float = 1.0e-10,
) -> float:
    layout = replay_context.base_layout
    executor = CompiledAnsatzExecutor(
        list(replay_context.replay_terms),
        coefficient_tolerance=float(layout.coefficient_tolerance),
        ignore_identity=bool(layout.ignore_identity),
        sort_terms=(str(layout.term_order).strip().lower() == "sorted"),
        parameterization_mode="per_pauli_term",
        parameterization_layout=layout,
    )
    psi_reconstructed = executor.prepare_state(
        np.asarray(replay_context.adapt_theta_runtime, dtype=float).reshape(-1),
        np.asarray(replay_context.psi_ref, dtype=complex).reshape(-1),
    )
    err = float(
        np.linalg.norm(
            np.asarray(psi_reconstructed, dtype=complex).reshape(-1)
            - np.asarray(psi_initial, dtype=complex).reshape(-1)
        )
    )
    if err > float(tol):
        raise ValueError(
            f"Prepared-state reconstruction mismatch: ||psi_reconstructed - psi_initial||={err:.3e} > {tol:.3e}."
        )
    return float(err)


def _make_replay_run_cfg(
    payload: Mapping[str, Any],
    *,
    artifact_json: Path,
    tag: str,
    generator_family: str,
    fallback_family: str,
) -> ReplayRunConfig:
    settings = payload.get("settings", {})
    if not isinstance(settings, Mapping):
        raise ValueError("Input JSON missing settings object.")
    adapt = payload.get("adapt_vqe", {})
    if not isinstance(adapt, Mapping):
        adapt = {}

    L = int(settings.get("L", 0))
    if L <= 0:
        raise ValueError("settings.L must be present and positive.")
    n_up, n_dn = _resolve_sector_counts(settings, adapt, L=L)

    paop_r_raw = settings.get("paop_r", 1)
    paop_split_raw = settings.get("paop_split_paulis", False)
    paop_prune_eps_raw = settings.get("paop_prune_eps", 0.0)
    paop_normalization_raw = settings.get("paop_normalization", "none")

    scratch_dir = Path("artifacts/agent_runs") / tag / "_scratch"
    stem = str(Path(artifact_json).stem)

    return ReplayRunConfig(
        adapt_input_json=Path(artifact_json),
        output_json=scratch_dir / f"{stem}.json",
        output_csv=scratch_dir / f"{stem}.csv",
        output_md=scratch_dir / f"{stem}.md",
        output_log=scratch_dir / f"{stem}.log",
        tag=f"{tag}_{stem}",
        generator_family=str(generator_family),
        fallback_family=str(fallback_family),
        legacy_paop_key="paop_lf_full",
        replay_seed_policy="auto",
        replay_continuation_mode="phase3_v1",
        L=int(L),
        t=float(settings.get("t", 1.0)),
        u=float(settings.get("u", settings.get("U", 0.0))),
        dv=float(settings.get("dv", 0.0)),
        omega0=float(settings.get("omega0", 1.0)),
        g_ep=float(settings.get("g_ep", 0.0)),
        n_ph_max=int(settings.get("n_ph_max", 0)),
        boson_encoding=str(settings.get("boson_encoding", "binary")),
        ordering=str(settings.get("ordering", "blocked")),
        boundary=str(settings.get("boundary", "open")),
        sector_n_up=int(n_up),
        sector_n_dn=int(n_dn),
        reps=1,
        restarts=1,
        maxiter=1,
        method="POWELL",
        seed=7,
        energy_backend="dense",
        progress_every_s=30.0,
        wallclock_cap_s=3600,
        paop_r=(1 if paop_r_raw is None else int(paop_r_raw)),
        paop_split_paulis=_boolish(paop_split_raw, False),
        paop_prune_eps=(0.0 if paop_prune_eps_raw is None else float(paop_prune_eps_raw)),
        paop_normalization=("none" if paop_normalization_raw in {None, ""} else str(paop_normalization_raw)),
        spsa_a=0.1,
        spsa_c=0.1,
        spsa_alpha=0.602,
        spsa_gamma=0.101,
        spsa_A=10.0,
        spsa_avg_last=1,
        spsa_eval_repeats=1,
        spsa_eval_agg="mean",
        replay_freeze_fraction=0.0,
        replay_unfreeze_fraction=0.0,
        replay_full_fraction=1.0,
        replay_qn_spsa_refresh_every=0,
        replay_qn_spsa_refresh_mode="never",
        phase3_symmetry_mitigation_mode="none",
    )


def build_fixed_scaffold_context_from_payload(
    payload: Mapping[str, Any],
    *,
    cfg: ReplayRunConfig,
) -> ReplayScaffoldContext:
    adapt, fixed_meta = _validate_locked_scaffold_payload(payload)
    parameterization = adapt.get("parameterization", None)
    if not isinstance(parameterization, Mapping):
        raise ValueError("Fixed scaffold import requires adapt_vqe.parameterization.")

    layout = deserialize_layout(parameterization)
    raw_blocks = parameterization.get("blocks", [])
    if not isinstance(raw_blocks, Sequence):
        raise ValueError("adapt_vqe.parameterization.blocks must be a sequence.")
    if len(raw_blocks) != int(layout.logical_parameter_count):
        raise ValueError(
            f"Fixed scaffold parameterization block count mismatch: {len(raw_blocks)} vs {layout.logical_parameter_count}."
        )

    replay_terms: list[AnsatzTerm] = []
    for idx, raw_block in enumerate(raw_blocks):
        if not isinstance(raw_block, Mapping):
            raise ValueError("Fixed scaffold parameterization block must be a mapping.")
        serialized_terms = raw_block.get("runtime_terms_exyz", None)
        if not isinstance(serialized_terms, Sequence):
            raise ValueError(
                f"Fixed scaffold parameterization block {idx} missing runtime_terms_exyz sequence."
            )
        polynomial = rebuild_polynomial_from_serialized_terms(
            serialized_terms,
            drop_abs_tol=float(layout.coefficient_tolerance),
        )
        serialized_sig = _aggregate_serialized_terms(serialized_terms)
        rebuilt_sig = _aggregate_polynomial_terms(polynomial)
        if set(serialized_sig.keys()) != set(rebuilt_sig.keys()):
            raise ValueError(
                f"Fixed scaffold block {idx} reconstruction changed Pauli support: "
                f"{sorted(serialized_sig.keys())} vs {sorted(rebuilt_sig.keys())}."
            )
        for label in sorted(serialized_sig.keys()):
            if abs(complex(serialized_sig[label]) - complex(rebuilt_sig[label])) > 1.0e-10:
                raise ValueError(
                    f"Fixed scaffold block {idx} reconstruction mismatch for {label}: "
                    f"{serialized_sig[label]} vs {rebuilt_sig[label]}."
                )
        replay_terms.append(
            AnsatzTerm(
                label=str(layout.blocks[idx].candidate_label),
                polynomial=polynomial,
            )
        )

    psi_ref = _statevector_from_named_payload_state(payload, "ansatz_input_state")
    h_poly = _build_hh_hamiltonian(cfg)

    optimal_point = np.asarray(adapt.get("optimal_point", []), dtype=float).reshape(-1)
    if int(optimal_point.size) != int(layout.runtime_parameter_count):
        raise ValueError(
            f"Fixed scaffold runtime theta mismatch: {optimal_point.size} vs {layout.runtime_parameter_count}."
        )

    logical_optimal_raw = adapt.get("logical_optimal_point", None)
    if isinstance(logical_optimal_raw, Sequence):
        logical_optimal = np.asarray([float(x) for x in logical_optimal_raw], dtype=float).reshape(-1)
        if int(logical_optimal.size) != int(layout.logical_parameter_count):
            raise ValueError(
                "Fixed scaffold logical theta mismatch: "
                f"{logical_optimal.size} vs {layout.logical_parameter_count}."
            )
    else:
        logical_optimal = np.asarray(
            project_runtime_theta_block_mean(optimal_point, layout),
            dtype=float,
        ).reshape(-1)

    init_state = payload.get("initial_state", {})
    if not isinstance(init_state, Mapping):
        init_state = {}

    family_info = {
        "requested": "fixed_scaffold_locked",
        "resolved": "fixed_scaffold_locked",
        "resolution_source": (
            "adapt_vqe.fixed_scaffold_metadata.route_family"
            if fixed_meta.get("route_family", None) is not None
            else "adapt_vqe.pool_type"
        ),
        "fallback_family": "full_meta",
        "fallback_used": False,
        "warning": None,
    }
    pool_meta = {
        "family": "fixed_scaffold_locked",
        "candidate_pool_complete": True,
        "structure_locked": bool(adapt.get("structure_locked", True)),
        "fixed_scaffold_kind": adapt.get("fixed_scaffold_kind", None),
        "route_family": fixed_meta.get("route_family", None),
        "subject_kind": fixed_meta.get("subject_kind", None),
        "source_artifact_json": fixed_meta.get("source_artifact_json", None),
    }

    handoff_state_kind = str(init_state.get("handoff_state_kind", "prepared_state"))
    provenance_source = (
        "explicit" if "handoff_state_kind" in init_state else "fixed_scaffold_export"
    )

    return ReplayScaffoldContext(
        cfg=cfg,
        h_poly=h_poly,
        psi_ref=np.asarray(psi_ref, dtype=complex).reshape(-1),
        payload_in=dict(payload),
        family_info=dict(family_info),
        family_pool=tuple(replay_terms),
        pool_meta=dict(pool_meta),
        replay_terms=tuple(replay_terms),
        base_layout=layout,
        adapt_theta_runtime=np.asarray(optimal_point, dtype=float).reshape(-1),
        adapt_theta_logical=np.asarray(logical_optimal, dtype=float).reshape(-1),
        adapt_depth=int(len(replay_terms)),
        handoff_state_kind=str(handoff_state_kind),
        provenance_source=str(provenance_source),
        family_terms_count=int(len(replay_terms)),
    )


def load_run_context(
    spec: FixedManifoldRunSpec,
    *,
    tag: str,
    lock_fixed_manifold: bool = True,
) -> LoadedRunContext:
    payload = _read_json(spec.artifact_json)
    psi_initial = _statevector_from_named_payload_state(payload, "initial_state")

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
            "candidate_pool_complete": bool(
                replay_context.pool_meta.get("candidate_pool_complete", False)
            ),
            "fixed_manifold_locked": bool(replay_context.pool_meta.get("fixed_manifold_locked", False)),
            "lock_fixed_manifold_requested": bool(lock_fixed_manifold),
            "family_pool_origin": replay_context.pool_meta.get("family_pool_origin", None),
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
                replay_context.pool_meta.get("candidate_pool_complete", False)
            ),
            "fixed_manifold_locked": bool(replay_context.pool_meta.get("fixed_manifold_locked", False)),
            "lock_fixed_manifold_requested": bool(lock_fixed_manifold),
            "family_pool_origin": replay_context.pool_meta.get("family_pool_origin", None),
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
) -> dict[str, Any]:
    loaded = load_run_context(spec, tag=tag)
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
            "drive_enabled": False,
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
        },
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
            "drive_enabled": False,
            "decision_mode": "exact_v1",
            "structure_policy": "fixed_manifold_locked_pool",
            "effective_pool_kind": "replay_terms_only",
            "t_final": float(t_final),
            "num_times": int(num_times),
            "miss_threshold": float(miss_threshold),
        },
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
        help="Recorded controller knob. Append is still impossible here because the family pool is locked to replay_terms.",
    )
    parser.add_argument(
        "--gain-ratio-threshold",
        type=float,
        default=1.0e-9,
        help="Recorded controller knob. Append is still impossible here because the family pool is locked to replay_terms.",
    )
    parser.add_argument(
        "--append-margin-abs",
        type=float,
        default=1.0e-12,
        help="Recorded controller knob. Append is still impossible here because the family pool is locked to replay_terms.",
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
        ),
        FixedManifoldRunSpec(
            name="locked_7term",
            artifact_json=Path(args.locked_7term_artifact_json),
            loader_mode="fixed_scaffold",
            generator_family="fixed_scaffold_locked",
            fallback_family="full_meta",
        ),
    ]

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
