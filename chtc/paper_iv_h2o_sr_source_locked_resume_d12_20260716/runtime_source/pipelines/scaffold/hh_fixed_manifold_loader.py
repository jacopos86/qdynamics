"""Scaffold-owned helpers for HH fixed-manifold runtime loading.

These helpers normalize legacy replay/fixed-scaffold payloads and build the
`ReplayScaffoldContext` shape consumed by scaffold runtime loading and Paper-II
fixed-manifold dynamics. They intentionally do not import time-dynamics runner
or controller modules; Paper II imports this bridge instead.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.contracts.scaffold import ReplayScaffoldContext
from pipelines.scaffold.hh_continuation_generators import rebuild_polynomial_from_serialized_terms
from pipelines.scaffold.hh_vqe_from_adapt_family import (
    RunConfig as ReplayRunConfig,
    _amplitudes_qn_to_q0_to_statevector,
    _build_hh_hamiltonian,
    _build_pool_for_family,
    _resolve_append_family,
)
from src.quantum.ansatz_parameterization import (
    deserialize_layout,
    project_runtime_theta_block_mean,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


@dataclass(frozen=True)
class FixedManifoldRunSpec:
    name: str
    artifact_json: Path
    loader_mode: str
    generator_family: str = "match_adapt"
    fallback_family: str = "full_meta"
    append_pool_family: str = "match_replay"


@dataclass(frozen=True)
class LoadedRunContext:
    spec: FixedManifoldRunSpec
    cfg: ReplayRunConfig
    payload: dict[str, Any]
    replay_context: ReplayScaffoldContext
    psi_initial: np.ndarray
    loader_summary: dict[str, Any]
    runtime_input: Any | None = None


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
    append_info = dict(replay_context.append_family_info or {})
    append_meta_src = replay_context.append_pool_meta
    append_meta = dict(replay_context.pool_meta if append_meta_src is None else append_meta_src)
    append_pool = replay_context.append_family_pool
    append_family_terms_count = replay_context.append_family_terms_count
    if bool(append_info.get("uses_replay_pool", True)):
        append_meta["candidate_pool_complete"] = True
        append_meta["fixed_manifold_locked"] = True
        append_meta["append_pool_source"] = "locked_replay_terms_only"
        append_pool = tuple(replay_context.replay_terms)
        append_family_terms_count = int(len(replay_context.replay_terms))
    return replace(
        replay_context,
        family_pool=tuple(replay_context.replay_terms),
        pool_meta=dict(pool_meta),
        family_terms_count=int(len(replay_context.replay_terms)),
        append_family_pool=(
            None if append_pool is None else tuple(append_pool)
        ),
        append_pool_meta=dict(append_meta),
        append_family_terms_count=(
            None if append_family_terms_count is None else int(append_family_terms_count)
        ),
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
    append_pool_family: str = "match_replay",
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
        append_pool_family=str(append_pool_family),
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
    append_family_info = _resolve_append_family(cfg, payload, family_info)
    append_resolved = str(append_family_info.get("resolved", "fixed_scaffold_locked"))
    if bool(append_family_info.get("uses_replay_pool", False)):
        append_pool = list(replay_terms)
        append_pool_meta = dict(pool_meta)
        append_pool_meta["append_pool_source"] = "fixed_scaffold_replay_terms"
        append_family_terms_count = int(len(append_pool))
    else:
        append_pool, append_pool_meta = _build_pool_for_family(
            cfg,
            family=append_resolved,
            h_poly=h_poly,
        )
        append_pool_meta = dict(append_pool_meta)
        append_pool_meta.update(
            {
                "candidate_pool_complete": True,
                "append_pool_source": "append_pool_family",
                "replay_family": "fixed_scaffold_locked",
                "append_family": str(append_resolved),
            }
        )
        append_family_terms_count = int(len(append_pool))

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
        append_family_info=dict(append_family_info),
        append_family_pool=tuple(append_pool),
        append_pool_meta=dict(append_pool_meta),
        append_family_terms_count=int(append_family_terms_count),
    )


__all__ = [
    "FixedManifoldRunSpec",
    "LoadedRunContext",
    "_lock_replay_context_to_fixed_manifold",
    "_make_replay_run_cfg",
    "_statevector_from_named_payload_state",
    "_validate_prepared_state_consistency",
    "build_fixed_scaffold_context_from_payload",
    "normalize_replay_payload",
]
