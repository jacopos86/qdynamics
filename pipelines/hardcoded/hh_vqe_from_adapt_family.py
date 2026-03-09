#!/usr/bin/env python3
"""Conventional HH VQE replay seeded from ADAPT state, matching ADAPT family.

Default behavior:
- Resolve generator family from input ADAPT/checkpoint JSON metadata.
- Fall back to full_meta if family cannot be resolved.
- Use SPSA by default.

This path is intended for ADAPT-seeded VQE replay where the variational
generator family should match ADAPT provenance, not a fixed hh_hva_* ansatz.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.adapt_pipeline import (
    _build_hh_termwise_augmented_pool,
    _build_hh_full_meta_pool,
    _build_hh_uccsd_fermion_lifted_pool,
    _build_hva_pool,
    _build_paop_pool,
    _deduplicate_pool_terms,
    _deduplicate_pool_terms_lightweight,
)
from pipelines.hardcoded.hh_continuation_generators import rebuild_polynomial_from_serialized_terms
from src.quantum.operator_pools.polaron_paop import _make_paop_core
from src.quantum.hubbard_latex_python_pairs import (
    boson_qubits_per_site,
    build_hubbard_holstein_hamiltonian,
)
from src.quantum.vqe_latex_python_pairs import (
    AnsatzTerm,
    apply_exp_pauli_polynomial,
    expval_pauli_polynomial,
    exact_ground_energy_sector_hh,
    vqe_minimize,
)
from pipelines.hardcoded.hh_continuation_replay import (
    ReplayControllerConfig,
    run_phase1_replay,
    run_phase2_replay,
    run_phase3_replay,
)


EXPLICIT_FAMILIES = {
    "full_meta",
    "uccsd_paop_lf_full",
    "hva",
    "paop",
    "paop_min",
    "paop_std",
    "paop_full",
    "paop_lf",
    "paop_lf_std",
    "paop_lf2_std",
    "paop_lf_full",
    "pool_a",
    "pool_b",
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical_family(raw: Any) -> str | None:
    if raw is None:
        return None
    val = str(raw).strip().lower()
    return val if val in EXPLICIT_FAMILIES else None


def _extract_nested(payload: Mapping[str, Any], *keys: str) -> Any:
    cur: Any = payload
    for k in keys:
        if not isinstance(cur, Mapping) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _resolve_exact_energy_from_payload(payload: Mapping[str, Any]) -> float | None:
    candidates = (
        _extract_nested(payload, "exact", "E_exact_sector"),
        _extract_nested(payload, "ground_state", "exact_energy_filtered"),
        _extract_nested(payload, "adapt_vqe", "exact_gs_energy"),
        _extract_nested(payload, "vqe", "exact_energy"),
    )
    for raw in candidates:
        if raw is None:
            continue
        try:
            val = float(raw)
        except Exception:
            continue
        if np.isfinite(val):
            return float(val)
    return None


def _resolve_family_from_metadata(payload: Mapping[str, Any]) -> tuple[str | None, str | None]:
    cand = _canonical_family(_extract_nested(payload, "adapt_vqe", "pool_type"))
    if cand is not None:
        return cand, "adapt_vqe.pool_type"

    cand = _canonical_family(_extract_nested(payload, "settings", "adapt_pool"))
    if cand is not None:
        return cand, "settings.adapt_pool"

    meta_pool_variant = _extract_nested(payload, "meta", "pool_variant")
    if meta_pool_variant is not None:
        pv = str(meta_pool_variant).strip().upper()
        if pv == "A":
            return "pool_a", "meta.pool_variant"
        if pv == "B":
            return "pool_b", "meta.pool_variant"

    source = _extract_nested(payload, "initial_state", "source")
    if source is not None:
        source_s = str(source).strip().lower()
        if "a_probe" in source_s or "a_medium" in source_s or source_s.startswith("a_"):
            return "pool_a", "initial_state.source"
        if "b_probe" in source_s or "b_medium" in source_s or source_s.startswith("b_"):
            return "pool_b", "initial_state.source"

    return None, None


def _parse_int_setting(settings: Mapping[str, Any], key: str, fallback_key: str | None = None) -> int | None:
    raw = settings.get(key, None)
    if raw is None and fallback_key is not None:
        raw = settings.get(fallback_key, None)
    if raw is None:
        return None
    return int(raw)


def _parse_float_setting(settings: Mapping[str, Any], key: str, fallback_key: str | None = None) -> float | None:
    raw = settings.get(key, None)
    if raw is None and fallback_key is not None:
        raw = settings.get(fallback_key, None)
    if raw is None:
        return None
    return float(raw)


def _half_filled_particles(num_sites: int) -> tuple[int, int]:
    return ((int(num_sites) + 1) // 2, int(num_sites) // 2)


def _require(name: str, val: Any) -> Any:
    if val is None:
        raise ValueError(f"Missing required setting '{name}' in CLI and input JSON metadata.")
    return val


def _amplitudes_qn_to_q0_to_statevector(payload: Mapping[str, Any], *, nq: int) -> np.ndarray:
    vec = np.zeros(1 << int(nq), dtype=complex)
    for bit, coeff in payload.items():
        bit_s = str(bit)
        if len(bit_s) != int(nq) or set(bit_s) - {"0", "1"}:
            continue
        if isinstance(coeff, Mapping):
            re = float(coeff.get("re", 0.0))
            im = float(coeff.get("im", 0.0))
        else:
            re = 0.0
            im = 0.0
        vec[int(bit_s, 2)] = re + 1.0j * im
    nrm = float(np.linalg.norm(vec))
    if nrm <= 0.0:
        raise ValueError("Imported statevector has zero norm.")
    return vec / nrm


def _statevector_to_amplitudes_qn_to_q0(
    vec: np.ndarray, *, cutoff: float = 1e-14
) -> dict[str, dict[str, float]]:
    arr = np.asarray(vec, dtype=complex).reshape(-1)
    nq = int(round(math.log2(int(arr.size))))
    out: dict[str, dict[str, float]] = {}
    for idx, amp in enumerate(arr):
        if abs(amp) < float(cutoff):
            continue
        bit = format(idx, f"0{nq}b")
        out[bit] = {"re": float(np.real(amp)), "im": float(np.imag(amp))}
    return out


@dataclass(frozen=True)
class RunConfig:
    adapt_input_json: Path
    output_json: Path
    output_csv: Path
    output_md: Path
    output_log: Path
    tag: str
    generator_family: str
    fallback_family: str
    legacy_paop_key: str
    replay_seed_policy: str
    replay_continuation_mode: str
    L: int
    t: float
    u: float
    dv: float
    omega0: float
    g_ep: float
    n_ph_max: int
    boson_encoding: str
    ordering: str
    boundary: str
    sector_n_up: int
    sector_n_dn: int
    reps: int
    restarts: int
    maxiter: int
    method: str
    seed: int
    energy_backend: str
    progress_every_s: float
    wallclock_cap_s: int
    paop_r: int
    paop_split_paulis: bool
    paop_prune_eps: float
    paop_normalization: str
    spsa_a: float
    spsa_c: float
    spsa_alpha: float
    spsa_gamma: float
    spsa_A: float
    spsa_avg_last: int
    spsa_eval_repeats: int
    spsa_eval_agg: str
    replay_freeze_fraction: float
    replay_unfreeze_fraction: float
    replay_full_fraction: float
    replay_qn_spsa_refresh_every: int
    replay_qn_spsa_refresh_mode: str
    phase3_symmetry_mitigation_mode: str


class RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def log(self, msg: str) -> None:
        line = f"[{_now_utc()}] {msg}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


class PoolTermwiseAnsatz:
    """Fixed ansatz built from a list of generators, repeated by layers."""

    def __init__(self, *, terms: list[AnsatzTerm], reps: int, nq: int) -> None:
        if int(reps) < 1:
            raise ValueError("reps must be >= 1.")
        self.base_terms = list(terms)
        self.reps = int(reps)
        self.nq = int(nq)
        self.num_parameters = int(len(self.base_terms)) * int(self.reps)

    def prepare_state(
        self,
        theta: np.ndarray,
        psi_ref: np.ndarray,
        *,
        ignore_identity: bool = True,
        coefficient_tolerance: Optional[float] = None,
        sort_terms: Optional[bool] = None,
    ) -> np.ndarray:
        arr = np.asarray(theta, dtype=float).reshape(-1)
        if int(arr.size) != int(self.num_parameters):
            raise ValueError(
                f"theta length {int(arr.size)} does not match num_parameters={int(self.num_parameters)}"
            )
        psi = np.asarray(psi_ref, dtype=complex).reshape(-1)
        if int(psi.size) != (1 << int(self.nq)):
            raise ValueError(f"psi_ref length {int(psi.size)} does not match 2^nq={1 << int(self.nq)}")

        out = np.array(psi, copy=True)
        tol = 1e-12 if coefficient_tolerance is None else float(coefficient_tolerance)
        sflag = True if sort_terms is None else bool(sort_terms)
        idx = 0
        for _ in range(self.reps):
            for term in self.base_terms:
                out = apply_exp_pauli_polynomial(
                    out,
                    term.polynomial,
                    float(arr[idx]),
                    ignore_identity=ignore_identity,
                    coefficient_tolerance=tol,
                    sort_terms=sflag,
                )
                idx += 1
        return out


def _build_hh_hamiltonian(cfg: RunConfig) -> Any:
    return build_hubbard_holstein_hamiltonian(
        dims=int(cfg.L),
        J=float(cfg.t),
        U=float(cfg.u),
        omega0=float(cfg.omega0),
        g=float(cfg.g_ep),
        n_ph_max=int(cfg.n_ph_max),
        boson_encoding=str(cfg.boson_encoding),
        v0=float(cfg.dv),
        repr_mode="JW",
        indexing=str(cfg.ordering),
        pbc=(str(cfg.boundary).strip().lower() == "periodic"),
        include_zero_point=True,
    )


def _dedup_terms(terms: list[AnsatzTerm], *, n_ph_max: int) -> list[AnsatzTerm]:
    if int(n_ph_max) >= 2:
        return _deduplicate_pool_terms_lightweight(terms)
    return _deduplicate_pool_terms(terms)


def _build_pool_for_family(cfg: RunConfig, *, family: str, h_poly: Any) -> tuple[list[AnsatzTerm], dict[str, Any]]:
    family_key = str(family).strip().lower()
    num_particles = (int(cfg.sector_n_up), int(cfg.sector_n_dn))
    n_sites = int(cfg.L)
    base_meta: dict[str, Any] = {"family": family_key}

    if family_key == "full_meta":
        pool, meta = _build_hh_full_meta_pool(
            h_poly=h_poly,
            num_sites=n_sites,
            t=float(cfg.t),
            u=float(cfg.u),
            omega0=float(cfg.omega0),
            g_ep=float(cfg.g_ep),
            dv=float(cfg.dv),
            n_ph_max=int(cfg.n_ph_max),
            boson_encoding=str(cfg.boson_encoding),
            ordering=str(cfg.ordering),
            boundary=str(cfg.boundary),
            paop_r=int(cfg.paop_r),
            paop_split_paulis=bool(cfg.paop_split_paulis),
            paop_prune_eps=float(cfg.paop_prune_eps),
            paop_normalization=str(cfg.paop_normalization),
            num_particles=num_particles,
        )
        out_meta = dict(base_meta)
        out_meta["raw_sizes"] = dict(meta)
        out_meta["dedup_total"] = int(len(pool))
        return pool, out_meta

    if family_key == "uccsd_paop_lf_full":
        uccsd = _build_hh_uccsd_fermion_lifted_pool(
            n_sites,
            int(cfg.n_ph_max),
            str(cfg.boson_encoding),
            str(cfg.ordering),
            str(cfg.boundary),
            num_particles=num_particles,
        )
        paop = _build_paop_pool(
            n_sites,
            int(cfg.n_ph_max),
            str(cfg.boson_encoding),
            str(cfg.ordering),
            str(cfg.boundary),
            "paop_lf_full",
            int(cfg.paop_r),
            bool(cfg.paop_split_paulis),
            float(cfg.paop_prune_eps),
            str(cfg.paop_normalization),
            num_particles,
        )
        dedup = _dedup_terms(list(uccsd) + list(paop), n_ph_max=int(cfg.n_ph_max))
        out_meta = dict(base_meta)
        out_meta["raw_sizes"] = {"raw_uccsd_lifted": int(len(uccsd)), "raw_paop_lf_full": int(len(paop))}
        out_meta["dedup_total"] = int(len(dedup))
        return dedup, out_meta

    if family_key == "hva":
        pool = _build_hva_pool(
            n_sites,
            float(cfg.t),
            float(cfg.u),
            float(cfg.omega0),
            float(cfg.g_ep),
            float(cfg.dv),
            int(cfg.n_ph_max),
            str(cfg.boson_encoding),
            str(cfg.ordering),
            str(cfg.boundary),
        )
        out_meta = dict(base_meta)
        out_meta["dedup_total"] = int(len(pool))
        return list(pool), out_meta

    if family_key in {"paop", "paop_min", "paop_std", "paop_full", "paop_lf", "paop_lf_std", "paop_lf2_std", "paop_lf_full"}:
        pool = _build_paop_pool(
            n_sites,
            int(cfg.n_ph_max),
            str(cfg.boson_encoding),
            str(cfg.ordering),
            str(cfg.boundary),
            family_key,
            int(cfg.paop_r),
            bool(cfg.paop_split_paulis),
            float(cfg.paop_prune_eps),
            str(cfg.paop_normalization),
            num_particles,
        )
        out_meta = dict(base_meta)
        out_meta["dedup_total"] = int(len(pool))
        return list(pool), out_meta

    if family_key == "pool_a":
        uccsd = _build_hh_uccsd_fermion_lifted_pool(
            n_sites,
            int(cfg.n_ph_max),
            str(cfg.boson_encoding),
            str(cfg.ordering),
            str(cfg.boundary),
            num_particles=num_particles,
        )
        paop = _build_paop_pool(
            n_sites,
            int(cfg.n_ph_max),
            str(cfg.boson_encoding),
            str(cfg.ordering),
            str(cfg.boundary),
            str(cfg.legacy_paop_key),
            int(cfg.paop_r),
            bool(cfg.paop_split_paulis),
            float(cfg.paop_prune_eps),
            str(cfg.paop_normalization),
            num_particles,
        )
        dedup = _dedup_terms(list(uccsd) + list(paop), n_ph_max=int(cfg.n_ph_max))
        out_meta = dict(base_meta)
        out_meta["legacy_mapping_note"] = f"pool_a -> uccsd_lifted + {cfg.legacy_paop_key}"
        out_meta["raw_sizes"] = {"raw_uccsd_lifted": int(len(uccsd)), f"raw_{cfg.legacy_paop_key}": int(len(paop))}
        out_meta["dedup_total"] = int(len(dedup))
        return dedup, out_meta

    if family_key == "pool_b":
        uccsd = _build_hh_uccsd_fermion_lifted_pool(
            n_sites,
            int(cfg.n_ph_max),
            str(cfg.boson_encoding),
            str(cfg.ordering),
            str(cfg.boundary),
            num_particles=num_particles,
        )
        hva = _build_hva_pool(
            n_sites,
            float(cfg.t),
            float(cfg.u),
            float(cfg.omega0),
            float(cfg.g_ep),
            float(cfg.dv),
            int(cfg.n_ph_max),
            str(cfg.boson_encoding),
            str(cfg.ordering),
            str(cfg.boundary),
        )
        paop = _build_paop_pool(
            n_sites,
            int(cfg.n_ph_max),
            str(cfg.boson_encoding),
            str(cfg.ordering),
            str(cfg.boundary),
            str(cfg.legacy_paop_key),
            int(cfg.paop_r),
            bool(cfg.paop_split_paulis),
            float(cfg.paop_prune_eps),
            str(cfg.paop_normalization),
            num_particles,
        )
        dedup = _dedup_terms(list(uccsd) + list(hva) + list(paop), n_ph_max=int(cfg.n_ph_max))
        out_meta = dict(base_meta)
        out_meta["legacy_mapping_note"] = f"pool_b -> uccsd_lifted + hva + {cfg.legacy_paop_key}"
        out_meta["raw_sizes"] = {
            "raw_uccsd_lifted": int(len(uccsd)),
            "raw_hva": int(len(hva)),
            f"raw_{cfg.legacy_paop_key}": int(len(paop)),
        }
        out_meta["dedup_total"] = int(len(dedup))
        return dedup, out_meta

    raise ValueError(f"Unsupported generator family: {family_key}")


def _read_input_state_and_payload(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    init = payload.get("initial_state", {})
    if not isinstance(init, Mapping):
        raise ValueError("Input JSON missing 'initial_state' payload.")
    amps = init.get("amplitudes_qn_to_q0", None)
    if not isinstance(amps, Mapping):
        raise ValueError("Input JSON missing initial_state.amplitudes_qn_to_q0.")

    nq_total_raw = init.get("nq_total", None)
    nq_total = int(nq_total_raw) if nq_total_raw is not None else len(next(iter(amps.keys()), ""))
    if int(nq_total) <= 0:
        raise ValueError("Could not infer nq_total from input JSON state amplitudes.")
    state = _amplitudes_qn_to_q0_to_statevector(amps, nq=int(nq_total))
    return state, payload


def _extract_adapt_operator_theta_sequence(payload: Mapping[str, Any]) -> tuple[list[str], np.ndarray]:
    adapt = payload.get("adapt_vqe", None)
    if not isinstance(adapt, Mapping):
        raise ValueError(
            "Input JSON missing object key 'adapt_vqe'; replay requires "
            "adapt_vqe.operators and adapt_vqe.optimal_point."
        )
    ops_raw = adapt.get("operators", None)
    theta_raw = adapt.get("optimal_point", None)

    if not isinstance(ops_raw, list) or len(ops_raw) == 0:
        raise ValueError("Input JSON missing non-empty list adapt_vqe.operators.")
    if not isinstance(theta_raw, list) or len(theta_raw) == 0:
        raise ValueError("Input JSON missing non-empty list adapt_vqe.optimal_point.")
    if len(ops_raw) != len(theta_raw):
        raise ValueError(
            "Length mismatch between adapt_vqe.operators and adapt_vqe.optimal_point: "
            f"{len(ops_raw)} vs {len(theta_raw)}."
        )

    labels: list[str] = []
    for i, raw_label in enumerate(ops_raw):
        label = str(raw_label).strip()
        if len(label) == 0:
            raise ValueError(f"Invalid empty operator label at adapt_vqe.operators[{i}].")
        labels.append(label)

    theta_vals: list[float] = []
    for i, raw_theta in enumerate(theta_raw):
        try:
            val = float(raw_theta)
        except Exception as exc:
            raise ValueError(
                f"Invalid theta value at adapt_vqe.optimal_point[{i}]={raw_theta!r}; expected float."
            ) from exc
        if not np.isfinite(val):
            raise ValueError(
                f"Non-finite theta value at adapt_vqe.optimal_point[{i}]={raw_theta!r}."
            )
        theta_vals.append(float(val))

    theta = np.asarray(theta_vals, dtype=float)
    return labels, theta


def _inject_replay_terms_from_payload(
    label_to_term: dict[str, AnsatzTerm],
    payload: Mapping[str, Any] | None,
) -> None:
    if not isinstance(payload, Mapping):
        return
    continuation = payload.get("continuation", {})
    if not isinstance(continuation, Mapping):
        return
    raw_selected_meta = continuation.get("selected_generator_metadata", [])
    if not isinstance(raw_selected_meta, Sequence):
        return
    for raw_meta in raw_selected_meta:
        if not isinstance(raw_meta, Mapping):
            continue
        lbl = str(raw_meta.get("candidate_label", "")).strip()
        if lbl == "" or lbl in label_to_term:
            continue
        compile_meta = raw_meta.get("compile_metadata", {})
        serialized_terms = (
            compile_meta.get("serialized_terms_exyz", [])
            if isinstance(compile_meta, Mapping)
            else []
        )
        if not isinstance(serialized_terms, Sequence):
            continue
        try:
            poly = rebuild_polynomial_from_serialized_terms(serialized_terms)
        except Exception:
            continue
        label_to_term[lbl] = AnsatzTerm(label=str(lbl), polynomial=poly)


def _build_replay_terms_from_adapt_labels(
    family_pool: Sequence[AnsatzTerm],
    adapt_labels: Sequence[str],
    payload: Mapping[str, Any] | None = None,
) -> list[AnsatzTerm]:
    label_to_term: dict[str, AnsatzTerm] = {}
    duplicate_labels: list[str] = []
    for term in family_pool:
        lbl = str(term.label)
        if lbl in label_to_term:
            duplicate_labels.append(lbl)
            continue
        label_to_term[lbl] = term
    if duplicate_labels:
        dup_preview = sorted(set(duplicate_labels))[:8]
        raise ValueError(
            "Resolved family pool has duplicate operator labels; replay mapping is ambiguous. "
            f"Examples: {dup_preview}"
        )

    _inject_replay_terms_from_payload(label_to_term, payload)

    replay_terms: list[AnsatzTerm] = []
    missing: list[str] = []
    for lbl in adapt_labels:
        term = label_to_term.get(str(lbl), None)
        if term is None:
            missing.append(str(lbl))
            continue
        replay_terms.append(term)
    if missing:
        miss_preview = missing[:8]
        raise ValueError(
            "ADAPT operators are not present in the resolved replay family pool. "
            f"Missing examples: {miss_preview}"
        )
    return replay_terms


def _build_replay_seed_theta(adapt_theta: np.ndarray, *, reps: int) -> np.ndarray:
    if int(reps) < 1:
        raise ValueError("reps must be >= 1 for replay seed construction.")
    base = np.asarray(adapt_theta, dtype=float).reshape(-1)
    if int(base.size) == 0:
        raise ValueError("adapt_theta must be non-empty for replay seed construction.")
    if not np.all(np.isfinite(base)):
        raise ValueError("adapt_theta contains non-finite values.")
    return np.tile(base, int(reps)).astype(float, copy=False)


# ── Replay seed policy (v2 contract) ────────────────────────────────────
REPLAY_SEED_POLICIES = {"auto", "scaffold_plus_zero", "residual_only", "tile_adapt"}
REPLAY_CONTRACT_VERSION = 2

# Handoff-state kind constants
_PREPARED_STATE = "prepared_state"
_REFERENCE_STATE = "reference_state"

# Sources that map unambiguously to reference_state in legacy payloads.
_LEGACY_REFERENCE_SOURCES = {"hf", "exact"}
# Source suffixes that map to prepared_state in legacy staged exports.
_LEGACY_PREPARED_FINAL_SUFFIXES = ("_final",)
# Source values that map to prepared_state in legacy payloads.
_LEGACY_PREPARED_SOURCES = {"adapt_vqe"}


def _infer_handoff_state_kind(
    payload: Mapping[str, Any],
) -> tuple[str, str]:
    """Return (handoff_state_kind, provenance_source).

    provenance_source is one of:
      "explicit"          – ``initial_state.handoff_state_kind`` was present.
      "inferred_source"   – inferred from ``initial_state.source`` legacy field.
      "ambiguous"         – could not resolve; caller must raise.
    """
    init = payload.get("initial_state", {})
    if not isinstance(init, Mapping):
        init = {}

    explicit = init.get("handoff_state_kind", None)
    if isinstance(explicit, str) and explicit in {_PREPARED_STATE, _REFERENCE_STATE}:
        return str(explicit), "explicit"

    # Legacy inference from initial_state.source
    source_raw = init.get("source", None)
    if isinstance(source_raw, str):
        source = str(source_raw).strip().lower()
        if source in _LEGACY_REFERENCE_SOURCES:
            return _REFERENCE_STATE, "inferred_source"
        if source in _LEGACY_PREPARED_SOURCES:
            return _PREPARED_STATE, "inferred_source"
        # Staged final exports use suffixes like A_probe_final, B_medium_final
        for suffix in _LEGACY_PREPARED_FINAL_SUFFIXES:
            if source.endswith(suffix):
                return _PREPARED_STATE, "inferred_source"
        # Warm-start exports
        if "warm_start" in source:
            return _PREPARED_STATE, "inferred_source"

    return "ambiguous", "ambiguous"


def _build_replay_seed_theta_policy(
    adapt_theta: np.ndarray,
    *,
    reps: int,
    policy: str,
    handoff_state_kind: str,
) -> tuple[np.ndarray, str]:
    """Build replay seed theta according to the given policy.

    Returns (seed_theta, resolved_policy_name).
    """
    base = np.asarray(adapt_theta, dtype=float).reshape(-1)
    if int(base.size) == 0:
        raise ValueError("adapt_theta must be non-empty for replay seed construction.")
    if not np.all(np.isfinite(base)):
        raise ValueError("adapt_theta contains non-finite values.")
    if int(reps) < 1:
        raise ValueError("reps must be >= 1 for replay seed construction.")

    adapt_depth = int(base.size)
    total_params = adapt_depth * int(reps)

    if policy == "auto":
        if handoff_state_kind == _PREPARED_STATE:
            resolved = "residual_only"
        elif handoff_state_kind == _REFERENCE_STATE:
            resolved = "scaffold_plus_zero"
        else:
            raise ValueError(
                "Cannot resolve replay seed policy 'auto': handoff_state_kind is "
                f"'{handoff_state_kind}'. Provide an explicit --replay-seed-policy or "
                "ensure the input JSON has initial_state.handoff_state_kind."
            )
    else:
        resolved = str(policy)

    if resolved == "tile_adapt":
        seed = np.tile(base, int(reps)).astype(float, copy=False)
    elif resolved == "scaffold_plus_zero":
        seed = np.zeros(total_params, dtype=float)
        seed[:adapt_depth] = base
    elif resolved == "residual_only":
        seed = np.zeros(total_params, dtype=float)
    else:
        raise ValueError(f"Unknown replay seed policy: '{resolved}'")

    return seed, resolved


def _build_full_meta_replay_terms_sparse(
    cfg: RunConfig,
    *,
    h_poly: Any,
    adapt_labels: Sequence[str],
    payload: Mapping[str, Any] | None = None,
) -> tuple[list[AnsatzTerm], dict[str, Any]]:
    """Resolve only needed full_meta labels to reduce peak memory for n_ph_max>=2."""
    needed_order = [str(lbl) for lbl in adapt_labels]
    needed_set = set(needed_order)
    selected: dict[str, AnsatzTerm] = {}
    raw_sizes: dict[str, int] = {
        "raw_uccsd_lifted": 0,
        "raw_hva": 0,
        "raw_hh_termwise_augmented": 0,
        "raw_paop_full": 0,
        "raw_paop_lf_full": 0,
    }
    num_particles = (int(cfg.sector_n_up), int(cfg.sector_n_dn))
    n_sites = int(cfg.L)

    def _consume(component_key: str, terms: Sequence[AnsatzTerm], *, raw_count: int | None = None) -> None:
        raw_sizes[component_key] = int(raw_count if raw_count is not None else len(terms))
        for term in terms:
            lbl = str(term.label)
            if lbl in needed_set and lbl not in selected:
                selected[lbl] = term

    def _build_paop_subset(pool_name: str) -> tuple[list[AnsatzTerm], int]:
        prefix = f"{pool_name}:"
        local_needed = {lbl[len(prefix):] for lbl in needed_set if lbl.startswith(prefix)}
        if not local_needed:
            return [], 0

        known_prefixes = (
            "paop_disp(",
            "paop_dbl(",
            "paop_hopdrag(",
            "paop_curdrag(",
            "paop_hop2(",
            "paop_cloud_p(",
            "paop_cloud_x(",
            "paop_dbl_p(",
            "paop_dbl_x(",
        )
        unknown = [lbl for lbl in local_needed if not any(lbl.startswith(pfx) for pfx in known_prefixes)]
        if unknown:
            full_pool = _build_paop_pool(
                n_sites,
                int(cfg.n_ph_max),
                str(cfg.boson_encoding),
                str(cfg.ordering),
                str(cfg.boundary),
                pool_name,
                int(cfg.paop_r),
                bool(cfg.paop_split_paulis),
                float(cfg.paop_prune_eps),
                str(cfg.paop_normalization),
                num_particles,
            )
            return list(full_pool), int(len(full_pool))

        include_disp = any(lbl.startswith("paop_disp(") for lbl in local_needed)
        include_doublon = any(lbl.startswith("paop_dbl(") for lbl in local_needed)
        include_hopdrag = any(lbl.startswith("paop_hopdrag(") for lbl in local_needed)
        include_curdrag = any(lbl.startswith("paop_curdrag(") for lbl in local_needed)
        include_hop2 = any(lbl.startswith("paop_hop2(") for lbl in local_needed)
        include_cloud_p = any(lbl.startswith("paop_cloud_p(") for lbl in local_needed)
        include_cloud_x = any(lbl.startswith("paop_cloud_x(") for lbl in local_needed)
        include_dbl_p = any(lbl.startswith("paop_dbl_p(") for lbl in local_needed)
        include_dbl_x = any(lbl.startswith("paop_dbl_x(") for lbl in local_needed)
        include_extended_cloud = bool(include_cloud_p or include_cloud_x)
        cloud_radius = int(cfg.paop_r)
        if (include_extended_cloud or include_dbl_p or include_dbl_x) and cloud_radius == 0:
            cloud_radius = 1

        specs = _make_paop_core(
            num_sites=n_sites,
            n_ph_max=int(cfg.n_ph_max),
            boson_encoding=str(cfg.boson_encoding),
            ordering=str(cfg.ordering),
            boundary=str(cfg.boundary),
            num_particles=num_particles,
            include_disp=bool(include_disp),
            include_doublon=bool(include_doublon),
            include_hopdrag=bool(include_hopdrag),
            include_curdrag=bool(include_curdrag),
            include_hop2=bool(include_hop2),
            drop_hop2_phonon_identity=bool(include_hop2),
            include_extended_cloud=bool(include_extended_cloud),
            cloud_radius=int(cloud_radius),
            include_cloud_x=bool(include_cloud_x),
            include_doublon_translation_p=bool(include_dbl_p),
            include_doublon_translation_x=bool(include_dbl_x),
            split_paulis=bool(cfg.paop_split_paulis),
            prune_eps=float(cfg.paop_prune_eps),
            normalization=str(cfg.paop_normalization),
            pool_name=pool_name,
        )
        needed_prefixed = {f"{pool_name}:{lbl}" for lbl in local_needed}
        subset = [AnsatzTerm(label=label, polynomial=poly) for label, poly in specs if label in needed_prefixed]
        return subset, int(len(specs))

    uccsd = _build_hh_uccsd_fermion_lifted_pool(
        n_sites,
        int(cfg.n_ph_max),
        str(cfg.boson_encoding),
        str(cfg.ordering),
        str(cfg.boundary),
        num_particles=num_particles,
    )
    _consume("raw_uccsd_lifted", uccsd)
    del uccsd
    gc.collect()

    hva = _build_hva_pool(
        n_sites,
        float(cfg.t),
        float(cfg.u),
        float(cfg.omega0),
        float(cfg.g_ep),
        float(cfg.dv),
        int(cfg.n_ph_max),
        str(cfg.boson_encoding),
        str(cfg.ordering),
        str(cfg.boundary),
    )
    _consume("raw_hva", hva)
    del hva
    gc.collect()

    if any(lbl.startswith("hh_termwise_") for lbl in needed_set):
        termwise_aug = [
            AnsatzTerm(label=f"hh_termwise_{term.label}", polynomial=term.polynomial)
            for term in _build_hh_termwise_augmented_pool(h_poly)
        ]
        _consume("raw_hh_termwise_augmented", termwise_aug)
        del termwise_aug
        gc.collect()

    if any(lbl.startswith("paop_full:") for lbl in needed_set):
        paop_full, paop_full_raw = _build_paop_subset("paop_full")
        _consume("raw_paop_full", paop_full, raw_count=paop_full_raw)
        del paop_full
        gc.collect()

    if any(lbl.startswith("paop_lf_full:") for lbl in needed_set):
        paop_lf_full, paop_lf_full_raw = _build_paop_subset("paop_lf_full")
        _consume("raw_paop_lf_full", paop_lf_full, raw_count=paop_lf_full_raw)
        del paop_lf_full
        gc.collect()

    _inject_replay_terms_from_payload(selected, payload)

    missing = [lbl for lbl in needed_order if lbl not in selected]
    if missing:
        miss_preview = missing[:8]
        raise ValueError(
            "ADAPT operators are not present in sparse full_meta component pools or serialized replay metadata. "
            f"Missing examples: {miss_preview}"
        )

    replay_terms = [selected[lbl] for lbl in needed_order]
    meta: dict[str, Any] = {
        "family": "full_meta",
        "selection_mode": "sparse_label_lookup",
        "raw_sizes": dict(raw_sizes),
        "raw_total": int(sum(raw_sizes.values())),
        "selected_unique_labels": int(len(set(needed_order))),
        "replay_terms": int(len(replay_terms)),
    }
    return replay_terms, meta


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(row.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerow(dict(row))


def _write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_cfg(args: argparse.Namespace, payload: Mapping[str, Any]) -> RunConfig:
    settings = payload.get("settings", {})
    if not isinstance(settings, Mapping):
        settings = {}

    L = int(_require("L", args.L if args.L is not None else _parse_int_setting(settings, "L")))
    t = float(_require("t", args.t if args.t is not None else _parse_float_setting(settings, "t")))
    u = float(_require("u", args.u if args.u is not None else _parse_float_setting(settings, "u", fallback_key="U")))
    dv = float(_require("dv", args.dv if args.dv is not None else _parse_float_setting(settings, "dv")))
    omega0 = float(_require("omega0", args.omega0 if args.omega0 is not None else _parse_float_setting(settings, "omega0")))
    g_ep = float(_require("g_ep", args.g_ep if args.g_ep is not None else _parse_float_setting(settings, "g_ep")))
    n_ph_max = int(_require("n_ph_max", args.n_ph_max if args.n_ph_max is not None else _parse_int_setting(settings, "n_ph_max")))
    boson_encoding = str(args.boson_encoding if args.boson_encoding is not None else settings.get("boson_encoding", "binary"))
    ordering = str(args.ordering if args.ordering is not None else settings.get("ordering", "blocked"))
    boundary = str(args.boundary if args.boundary is not None else settings.get("boundary", "open"))

    n_up_default, n_dn_default = _half_filled_particles(int(L))
    n_up = int(args.sector_n_up if args.sector_n_up is not None else settings.get("sector_n_up", n_up_default))
    n_dn = int(args.sector_n_dn if args.sector_n_dn is not None else settings.get("sector_n_dn", n_dn_default))

    reps = int(args.reps if args.reps is not None else int(L))
    tag = str(args.tag if args.tag is not None else f"hh_adapt_family_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")

    out_json = Path(args.output_json) if args.output_json is not None else Path(f"artifacts/json/{tag}.json")
    out_csv = Path(args.output_csv) if args.output_csv is not None else Path(f"artifacts/json/{tag}.csv")
    out_md = Path(args.output_md) if args.output_md is not None else Path(f"artifacts/useful/L{int(L)}/{tag}.md")
    out_log = Path(args.output_log) if args.output_log is not None else Path(f"artifacts/logs/{tag}.log")

    return RunConfig(
        adapt_input_json=Path(args.adapt_input_json),
        output_json=out_json,
        output_csv=out_csv,
        output_md=out_md,
        output_log=out_log,
        tag=tag,
        generator_family=str(args.generator_family),
        fallback_family=str(args.fallback_family),
        legacy_paop_key=str(args.legacy_paop_key),
        replay_seed_policy=str(args.replay_seed_policy),
        replay_continuation_mode=str(args.replay_continuation_mode),
        L=int(L),
        t=float(t),
        u=float(u),
        dv=float(dv),
        omega0=float(omega0),
        g_ep=float(g_ep),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        ordering=str(ordering),
        boundary=str(boundary),
        sector_n_up=int(n_up),
        sector_n_dn=int(n_dn),
        reps=int(reps),
        restarts=int(args.restarts),
        maxiter=int(args.maxiter),
        method=str(args.method),
        seed=int(args.seed),
        energy_backend=str(args.energy_backend),
        progress_every_s=float(args.progress_every_s),
        wallclock_cap_s=int(args.wallclock_cap_s),
        paop_r=int(args.paop_r),
        paop_split_paulis=bool(args.paop_split_paulis),
        paop_prune_eps=float(args.paop_prune_eps),
        paop_normalization=str(args.paop_normalization),
        spsa_a=float(args.spsa_a),
        spsa_c=float(args.spsa_c),
        spsa_alpha=float(args.spsa_alpha),
        spsa_gamma=float(args.spsa_gamma),
        spsa_A=float(args.spsa_A),
        spsa_avg_last=int(args.spsa_avg_last),
        spsa_eval_repeats=int(args.spsa_eval_repeats),
        spsa_eval_agg=str(args.spsa_eval_agg),
        replay_freeze_fraction=float(args.replay_freeze_fraction),
        replay_unfreeze_fraction=float(args.replay_unfreeze_fraction),
        replay_full_fraction=float(args.replay_full_fraction),
        replay_qn_spsa_refresh_every=int(args.replay_qn_spsa_refresh_every),
        replay_qn_spsa_refresh_mode=str(args.replay_qn_spsa_refresh_mode),
        phase3_symmetry_mitigation_mode=str(args.phase3_symmetry_mitigation_mode),
    )


def _resolve_family(cfg: RunConfig, payload: Mapping[str, Any]) -> dict[str, Any]:
    requested = str(cfg.generator_family).strip().lower()
    warning: str | None = None
    source: str | None = None
    fallback_used = False
    resolved: str

    if requested == "match_adapt":
        from_meta, source = _resolve_family_from_metadata(payload)
        if from_meta is not None:
            resolved = from_meta
        else:
            fallback_used = True
            resolved = str(cfg.fallback_family).strip().lower()
            warning = (
                "Could not resolve family from input metadata; using fallback family "
                f"'{resolved}'."
            )
            source = "fallback_family"
    else:
        cand = _canonical_family(requested)
        if cand is None:
            fallback_used = True
            resolved = str(cfg.fallback_family).strip().lower()
            warning = (
                f"Requested generator family '{requested}' unsupported; "
                f"using fallback '{resolved}'."
            )
            source = "fallback_family"
        else:
            resolved = cand
            source = "cli.generator_family"

    fallback = _canonical_family(cfg.fallback_family)
    if fallback is None:
        raise ValueError(f"Invalid --fallback-family '{cfg.fallback_family}'.")

    return {
        "requested": requested,
        "resolved": resolved,
        "resolution_source": source,
        "fallback_family": fallback,
        "fallback_used": bool(fallback_used),
        "warning": warning,
    }


def _resolve_replay_continuation_mode(raw: str | None) -> str:
    mode = "legacy" if raw is None else str(raw).strip().lower()
    if mode == "":
        return "legacy"
    if mode not in {"legacy", "phase1_v1", "phase2_v1", "phase3_v1"}:
        raise ValueError("replay_continuation_mode must be one of {'legacy','phase1_v1','phase2_v1','phase3_v1'}.")
    return mode


def run(cfg: RunConfig) -> dict[str, Any]:
    logger = RunLogger(cfg.output_log)
    logger.log(f"Loading ADAPT input JSON: {cfg.adapt_input_json}")
    psi_ref, payload_in = _read_input_state_and_payload(cfg.adapt_input_json)

    family_info = _resolve_family(cfg, payload_in)
    if family_info.get("warning"):
        logger.log(f"FAMILY WARNING: {family_info['warning']}")
    logger.log(
        f"Generator family resolved: requested={family_info['requested']} "
        f"resolved={family_info['resolved']} source={family_info['resolution_source']}"
    )

    logger.log("Building HH Hamiltonian.")
    h_poly = _build_hh_hamiltonian(cfg)
    e_exact_payload = _resolve_exact_energy_from_payload(payload_in)
    if e_exact_payload is not None:
        e_exact = float(e_exact_payload)
        logger.log(f"Using exact sector energy from input payload: E_exact={e_exact:.12f}")
    else:
        logger.log("Computing exact sector energy via ED (payload exact unavailable).")
        e_exact = float(
            exact_ground_energy_sector_hh(
                h_poly,
                num_sites=int(cfg.L),
                num_particles=(int(cfg.sector_n_up), int(cfg.sector_n_dn)),
                n_ph_max=int(cfg.n_ph_max),
                boson_encoding=str(cfg.boson_encoding),
                indexing=str(cfg.ordering),
            )
        )
        logger.log(f"Computed exact sector energy via ED: E_exact={e_exact:.12f}")

    adapt_labels, adapt_theta = _extract_adapt_operator_theta_sequence(payload_in)

    # ── Provenance resolution ────────────────────────────────────────
    handoff_state_kind, provenance_source = _infer_handoff_state_kind(payload_in)
    if provenance_source == "ambiguous" and str(cfg.replay_seed_policy) == "auto":
        raise ValueError(
            "Cannot resolve replay seed policy 'auto': input JSON has no "
            "initial_state.handoff_state_kind and initial_state.source could not "
            "be mapped unambiguously to reference_state or prepared_state. "
            "Use an explicit --replay-seed-policy (scaffold_plus_zero, residual_only, "
            "or tile_adapt) to proceed."
        )
    if provenance_source != "explicit":
        logger.log(
            f"PROVENANCE WARNING: handoff_state_kind inferred as '{handoff_state_kind}' "
            f"from legacy metadata (source='{provenance_source}'). "
            "Consider regenerating input JSON with explicit handoff_state_kind."
        )
    logger.log(
        f"Provenance: handoff_state_kind={handoff_state_kind} "
        f"provenance_source={provenance_source} "
        f"replay_seed_policy={cfg.replay_seed_policy}"
    )

    seed_theta, resolved_seed_policy = _build_replay_seed_theta_policy(
        adapt_theta,
        reps=int(cfg.reps),
        policy=str(cfg.replay_seed_policy),
        handoff_state_kind=str(handoff_state_kind),
    )

    family_resolved = str(family_info["resolved"])
    if family_resolved == "full_meta" and int(cfg.n_ph_max) >= 2:
        replay_terms, pool_meta = _build_full_meta_replay_terms_sparse(
            cfg,
            h_poly=h_poly,
            adapt_labels=adapt_labels,
            payload=payload_in,
        )
        family_terms_count = int(pool_meta.get("raw_total", 0))
    else:
        pool, pool_meta = _build_pool_for_family(cfg, family=family_resolved, h_poly=h_poly)
        replay_terms = _build_replay_terms_from_adapt_labels(pool, adapt_labels, payload=payload_in)
        family_terms_count = int(len(pool))
    nq = int(2 * int(cfg.L) + int(cfg.L) * int(boson_qubits_per_site(int(cfg.n_ph_max), str(cfg.boson_encoding))))
    ansatz = PoolTermwiseAnsatz(terms=replay_terms, reps=int(cfg.reps), nq=nq)
    if int(seed_theta.size) != int(ansatz.num_parameters):
        raise ValueError(
            "Internal replay parameter mismatch: "
            f"seed size {int(seed_theta.size)} != ansatz.num_parameters {int(ansatz.num_parameters)}."
        )
    logger.log(
        f"Pool built: family={family_info['resolved']} family_terms={family_terms_count} "
        f"adapt_depth={len(adapt_labels)} replay_terms={len(replay_terms)} npar={ansatz.num_parameters}"
    )
    psi_seed = np.asarray(ansatz.prepare_state(seed_theta, psi_ref), dtype=complex).reshape(-1)
    seed_energy = float(expval_pauli_polynomial(psi_seed, h_poly))
    seed_delta_abs = float(abs(seed_energy - float(e_exact)))
    seed_relative_abs = float(seed_delta_abs / max(abs(float(e_exact)), 1e-14))
    logger.log(
        f"Seed baseline (policy={resolved_seed_policy}): "
        f"E={seed_energy:.12f} |DeltaE|={seed_delta_abs:.6e}"
    )

    progress_tail: list[dict[str, Any]] = []
    run_t0 = time.perf_counter()
    wall_hit = False
    replay_mode = _resolve_replay_continuation_mode(str(cfg.replay_continuation_mode))
    incoming_optimizer_memory = None
    incoming_generator_metadata = None
    incoming_motif_library = None
    if isinstance(payload_in.get("continuation"), Mapping):
        incoming_optimizer_memory = payload_in.get("continuation", {}).get("optimizer_memory", None)
        incoming_generator_metadata = payload_in.get("continuation", {}).get("selected_generator_metadata", None)
        incoming_motif_library = payload_in.get("continuation", {}).get("motif_library", None)

    def _progress_logger(ev: dict[str, Any]) -> None:
        nonlocal progress_tail
        row = dict(ev)
        e_cur = row.get("energy_current", None)
        e_best = row.get("energy_best_global", None)
        if isinstance(e_cur, (int, float)):
            row["delta_abs_current"] = float(abs(float(e_cur) - float(e_exact)))
        if isinstance(e_best, (int, float)):
            row["delta_abs_best"] = float(abs(float(e_best) - float(e_exact)))
        progress_tail.append(row)
        if len(progress_tail) > 200:
            progress_tail = progress_tail[-200:]
        if str(row.get("event", "")) in {"heartbeat", "restart_end", "run_end", "early_stop_triggered"}:
            logger.log(
                f"VQE {row.get('event')} elapsed_s={float(row.get('elapsed_s', 0.0)):.1f} "
                f"nfev={int(row.get('nfev_so_far', 0))} "
                f"delta_abs_best={row.get('delta_abs_best')}"
            )

    def _early_stop_checker(ev: dict[str, Any]) -> bool:
        nonlocal wall_hit
        elapsed = float(ev.get("elapsed_s", 0.0))
        if elapsed >= float(cfg.wallclock_cap_s):
            wall_hit = True
            return True
        return False

    common_opt_kwargs = {
        "spsa_a": float(cfg.spsa_a),
        "spsa_c": float(cfg.spsa_c),
        "spsa_alpha": float(cfg.spsa_alpha),
        "spsa_gamma": float(cfg.spsa_gamma),
        "spsa_A": float(cfg.spsa_A),
        "spsa_avg_last": int(cfg.spsa_avg_last),
        "spsa_eval_repeats": int(cfg.spsa_eval_repeats),
        "spsa_eval_agg": str(cfg.spsa_eval_agg),
        "energy_backend": str(cfg.energy_backend),
    }
    replay_phase_history: list[dict[str, Any]] = []
    replay_phase_config: dict[str, Any] = {}
    if replay_mode == "legacy":
        vqe_res = vqe_minimize(
            h_poly,
            ansatz,
            psi_ref,
            restarts=int(cfg.restarts),
            seed=int(cfg.seed),
            initial_point=seed_theta,
            use_initial_point_first_restart=True,
            method=str(cfg.method),
            maxiter=int(cfg.maxiter),
            progress_logger=_progress_logger,
            progress_every_s=float(cfg.progress_every_s),
            progress_label="hh_vqe_from_adapt_family",
            track_history=False,
            emit_theta_in_progress=False,
            return_best_on_keyboard_interrupt=True,
            early_stop_checker=_early_stop_checker,
            **common_opt_kwargs,
        )
        theta_best = np.asarray(vqe_res.theta, dtype=float)
        e_best = float(vqe_res.energy)
        vqe_success = bool(vqe_res.success)
        vqe_message = str(vqe_res.message)
        vqe_nfev = int(vqe_res.nfev)
        vqe_nit = int(vqe_res.nit)
        best_restart = int(vqe_res.best_restart)
    elif replay_mode == "phase1_v1":
        replay_cfg = ReplayControllerConfig(
            freeze_fraction=float(cfg.replay_freeze_fraction),
            unfreeze_fraction=float(cfg.replay_unfreeze_fraction),
            full_fraction=float(cfg.replay_full_fraction),
        )
        theta_best, replay_phase_history, replay_meta = run_phase1_replay(
            vqe_minimize_fn=vqe_minimize,
            h_poly=h_poly,
            ansatz=ansatz,
            psi_ref=psi_ref,
            seed_theta=np.asarray(seed_theta, dtype=float),
            scaffold_block_size=int(len(adapt_labels)),
            seed_policy_resolved=str(resolved_seed_policy),
            handoff_state_kind=str(handoff_state_kind),
            cfg=replay_cfg,
            restarts=int(cfg.restarts),
            seed=int(cfg.seed),
            maxiter=int(cfg.maxiter),
            method=str(cfg.method),
            progress_every_s=float(cfg.progress_every_s),
            exact_energy=float(e_exact),
            kwargs=common_opt_kwargs,
        )
        replay_phase_config = dict(replay_meta.get("replay_phase_config", {}))
        e_best = float(replay_meta.get("result", {}).get("energy", float("nan")))
        vqe_success = bool(replay_meta.get("result", {}).get("success", False))
        vqe_message = str(replay_meta.get("result", {}).get("message", ""))
        vqe_nfev = int(replay_meta.get("result", {}).get("nfev", 0))
        vqe_nit = int(replay_meta.get("result", {}).get("nit", 0))
        best_restart = 0
    elif replay_mode == "phase2_v1":
        replay_cfg = ReplayControllerConfig(
            freeze_fraction=float(cfg.replay_freeze_fraction),
            unfreeze_fraction=float(cfg.replay_unfreeze_fraction),
            full_fraction=float(cfg.replay_full_fraction),
            qn_spsa_refresh_every=int(max(0, cfg.replay_qn_spsa_refresh_every)),
            qn_spsa_refresh_mode=str(cfg.replay_qn_spsa_refresh_mode),
            symmetry_mitigation_mode="off",
        )
        theta_best, replay_phase_history, replay_meta = run_phase2_replay(
            vqe_minimize_fn=vqe_minimize,
            h_poly=h_poly,
            ansatz=ansatz,
            psi_ref=psi_ref,
            seed_theta=np.asarray(seed_theta, dtype=float),
            scaffold_block_size=int(len(adapt_labels)),
            seed_policy_resolved=str(resolved_seed_policy),
            handoff_state_kind=str(handoff_state_kind),
            cfg=replay_cfg,
            restarts=int(cfg.restarts),
            seed=int(cfg.seed),
            maxiter=int(cfg.maxiter),
            method=str(cfg.method),
            progress_every_s=float(cfg.progress_every_s),
            exact_energy=float(e_exact),
            kwargs=common_opt_kwargs,
            incoming_optimizer_memory=(
                dict(incoming_optimizer_memory)
                if isinstance(incoming_optimizer_memory, Mapping)
                else None
            ),
        )
        replay_phase_config = dict(replay_meta.get("replay_phase_config", {}))
        e_best = float(replay_meta.get("result", {}).get("energy", float("nan")))
        vqe_success = bool(replay_meta.get("result", {}).get("success", False))
        vqe_message = str(replay_meta.get("result", {}).get("message", ""))
        vqe_nfev = int(replay_meta.get("result", {}).get("nfev", 0))
        vqe_nit = int(replay_meta.get("result", {}).get("nit", 0))
        best_restart = 0
    else:
        replay_cfg = ReplayControllerConfig(
            freeze_fraction=float(cfg.replay_freeze_fraction),
            unfreeze_fraction=float(cfg.replay_unfreeze_fraction),
            full_fraction=float(cfg.replay_full_fraction),
            qn_spsa_refresh_every=int(max(0, cfg.replay_qn_spsa_refresh_every)),
            qn_spsa_refresh_mode=str(cfg.replay_qn_spsa_refresh_mode),
            symmetry_mitigation_mode=str(cfg.phase3_symmetry_mitigation_mode),
        )
        theta_best, replay_phase_history, replay_meta = run_phase3_replay(
            vqe_minimize_fn=vqe_minimize,
            h_poly=h_poly,
            ansatz=ansatz,
            psi_ref=psi_ref,
            seed_theta=np.asarray(seed_theta, dtype=float),
            scaffold_block_size=int(len(adapt_labels)),
            seed_policy_resolved=str(resolved_seed_policy),
            handoff_state_kind=str(handoff_state_kind),
            cfg=replay_cfg,
            restarts=int(cfg.restarts),
            seed=int(cfg.seed),
            maxiter=int(cfg.maxiter),
            method=str(cfg.method),
            progress_every_s=float(cfg.progress_every_s),
            exact_energy=float(e_exact),
            kwargs=common_opt_kwargs,
            incoming_optimizer_memory=(
                dict(incoming_optimizer_memory)
                if isinstance(incoming_optimizer_memory, Mapping)
                else None
            ),
            generator_ids=[
                str(meta.get("generator_id", ""))
                for meta in incoming_generator_metadata
                if isinstance(meta, Mapping)
            ] if isinstance(incoming_generator_metadata, Sequence) else None,
            motif_reference_ids=[
                str(rec.get("motif_id", ""))
                for rec in incoming_motif_library.get("records", [])
                if isinstance(rec, Mapping)
            ] if isinstance(incoming_motif_library, Mapping) and isinstance(incoming_motif_library.get("records", []), Sequence) else None,
        )
        replay_phase_config = dict(replay_meta.get("replay_phase_config", {}))
        e_best = float(replay_meta.get("result", {}).get("energy", float("nan")))
        vqe_success = bool(replay_meta.get("result", {}).get("success", False))
        vqe_message = str(replay_meta.get("result", {}).get("message", ""))
        vqe_nfev = int(replay_meta.get("result", {}).get("nfev", 0))
        vqe_nit = int(replay_meta.get("result", {}).get("nit", 0))
        best_restart = 0

    runtime_s = float(time.perf_counter() - run_t0)
    psi_best = np.asarray(ansatz.prepare_state(theta_best, psi_ref), dtype=complex).reshape(-1)
    psi_best = psi_best / np.linalg.norm(psi_best)
    delta_abs = float(abs(e_best - float(e_exact)))
    rel_abs = float(delta_abs / max(abs(float(e_exact)), 1e-14))
    gate_pass = bool(delta_abs <= 1e-2)
    stop_reason = "wallclock_cap" if bool(wall_hit) else ("converged" if bool(vqe_success) else str(vqe_message))

    result = {
        "generated_utc": _now_utc(),
        "pipeline": "hh_vqe_from_adapt_family",
        "settings": {
            "problem": "hh",
            "L": int(cfg.L),
            "t": float(cfg.t),
            "u": float(cfg.u),
            "dv": float(cfg.dv),
            "omega0": float(cfg.omega0),
            "g_ep": float(cfg.g_ep),
            "n_ph_max": int(cfg.n_ph_max),
            "boson_encoding": str(cfg.boson_encoding),
            "ordering": str(cfg.ordering),
            "boundary": str(cfg.boundary),
            "sector_n_up": int(cfg.sector_n_up),
            "sector_n_dn": int(cfg.sector_n_dn),
            "reps": int(cfg.reps),
            "restarts": int(cfg.restarts),
            "maxiter": int(cfg.maxiter),
            "method": str(cfg.method),
            "seed": int(cfg.seed),
            "energy_backend": str(cfg.energy_backend),
            "progress_every_s": float(cfg.progress_every_s),
            "wallclock_cap_s": int(cfg.wallclock_cap_s),
            "paop_r": int(cfg.paop_r),
            "paop_split_paulis": bool(cfg.paop_split_paulis),
            "paop_prune_eps": float(cfg.paop_prune_eps),
            "paop_normalization": str(cfg.paop_normalization),
            "replay_seed_policy": str(cfg.replay_seed_policy),
            "replay_continuation_mode": str(replay_mode),
            "replay_freeze_fraction": float(cfg.replay_freeze_fraction),
            "replay_unfreeze_fraction": float(cfg.replay_unfreeze_fraction),
            "replay_full_fraction": float(cfg.replay_full_fraction),
            "replay_qn_spsa_refresh_every": int(cfg.replay_qn_spsa_refresh_every),
            "replay_qn_spsa_refresh_mode": str(cfg.replay_qn_spsa_refresh_mode),
            "phase3_symmetry_mitigation_mode": str(cfg.phase3_symmetry_mitigation_mode),
        },
        "generator_family": family_info,
        "pool": pool_meta,
        "replay_contract": {
            "contract_version": int(REPLAY_CONTRACT_VERSION),
            "continuation_mode": str(replay_mode),
            "replay_block_source": "adapt_vqe.operators",
            "seed_source": "adapt_vqe.optimal_point",
            "seed_policy_requested": str(cfg.replay_seed_policy),
            "seed_policy_resolved": str(resolved_seed_policy),
            "handoff_state_kind": str(handoff_state_kind),
            "provenance_source": str(provenance_source),
            "adapt_depth": int(len(adapt_labels)),
            "reps": int(cfg.reps),
            "derived_num_parameters_formula": "adapt_depth * reps",
            "derived_num_parameters": int(ansatz.num_parameters),
        },
        "seed_baseline": {
            "theta_policy": str(resolved_seed_policy),
            "energy": float(seed_energy),
            "abs_delta_e": float(seed_delta_abs),
            "relative_error_abs": float(seed_relative_abs),
        },
        "exact": {"E_exact_sector": float(e_exact)},
        "vqe": {
            "success": bool(vqe_success),
            "message": str(vqe_message),
            "energy": float(e_best),
            "abs_delta_e": float(delta_abs),
            "relative_error_abs": float(rel_abs),
            "best_restart": int(best_restart),
            "nfev": int(vqe_nfev),
            "nit": int(vqe_nit),
            "num_parameters": int(ansatz.num_parameters),
            "runtime_s": float(runtime_s),
            "stop_reason": str(stop_reason),
            "gate_pass_1e2": bool(gate_pass),
        },
        "initial_state": {
            "source": "adapt_input_json",
            "input_json_path": str(cfg.adapt_input_json),
            "nq_total": int(nq),
            "amplitudes_qn_to_q0": _statevector_to_amplitudes_qn_to_q0(psi_ref),
        },
        "best_state": {
            "amplitudes_qn_to_q0": _statevector_to_amplitudes_qn_to_q0(psi_best),
            "best_theta": [float(x) for x in theta_best.tolist()],
        },
        "progress_events_tail": progress_tail[-40:],
    }
    if replay_mode in {"phase1_v1", "phase2_v1", "phase3_v1"}:
        phase_payload = {
            "replay_phase_config": dict(replay_phase_config),
            "replay_phase_history": [dict(x) for x in replay_phase_history],
            "trust_radius_schedule": list(replay_phase_config.get("trust_radius_schedule", [])),
            "optimizer_memory_reused": bool(replay_phase_config.get("optimizer_memory_reused", False)),
            "qn_spsa_refresh": dict(
                replay_phase_config.get("qn_spsa_refresh", {"enabled": False, "refresh_points": []})
            ),
        }
        if replay_mode in {"phase2_v1", "phase3_v1"}:
            phase_payload.update(
                {
                    "optimizer_memory_source": str(replay_phase_config.get("optimizer_memory_source", "unavailable")),
                    "optimizer_memory": dict(replay_phase_config.get("optimizer_memory", {})),
                }
            )
        if replay_mode == "phase3_v1":
            phase_payload.update(
                {
                    "symmetry_mitigation_mode": str(replay_phase_config.get("symmetry_mitigation_mode", "off")),
                    "generator_ids": [str(x) for x in replay_phase_config.get("generator_ids", [])],
                    "motif_reference_ids": [str(x) for x in replay_phase_config.get("motif_reference_ids", [])],
                }
            )
        result.update(phase_payload)

    _write_json(cfg.output_json, result)
    _write_csv(
        cfg.output_csv,
        {
            "tag": str(cfg.tag),
            "family_requested": str(family_info["requested"]),
            "family_resolved": str(family_info["resolved"]),
            "method": str(cfg.method),
            "reps": int(cfg.reps),
            "delta_abs": float(delta_abs),
            "relative_error_abs": float(rel_abs),
            "gate_pass_1e2": bool(gate_pass),
            "runtime_s": float(runtime_s),
            "stop_reason": str(stop_reason),
            "output_json": str(cfg.output_json),
            "output_log": str(cfg.output_log),
        },
    )
    _write_md(
        cfg.output_md,
        [
            "# HH VQE Replay From ADAPT Family",
            "",
            f"- generated_utc: {_now_utc()}",
            f"- tag: {cfg.tag}",
            f"- adapt_input_json: {cfg.adapt_input_json}",
            "",
            "## Family",
            f"- requested: {family_info['requested']}",
            f"- resolved: {family_info['resolved']}",
            f"- resolution_source: {family_info['resolution_source']}",
            f"- fallback_used: {family_info['fallback_used']}",
            f"- warning: {family_info['warning']}",
            "",
            "## Metrics",
            f"- E_exact_sector: {e_exact}",
            f"- E_best: {e_best}",
            f"- abs_delta_e: {delta_abs}",
            f"- relative_error_abs: {rel_abs}",
            f"- replay_adapt_depth: {len(adapt_labels)}",
            f"- replay_reps: {cfg.reps}",
            f"- replay_npar: {ansatz.num_parameters} (= adapt_depth * reps)",
            f"- seed_policy_requested: {cfg.replay_seed_policy}",
            f"- seed_policy_resolved: {resolved_seed_policy}",
            f"- handoff_state_kind: {handoff_state_kind}",
            f"- provenance_source: {provenance_source}",
            f"- replay_contract_version: {REPLAY_CONTRACT_VERSION}",
            f"- gate_pass_1e2: {gate_pass}",
            f"- nfev: {int(vqe_nfev)}",
            f"- nit: {int(vqe_nit)}",
            f"- runtime_s: {runtime_s}",
            f"- stop_reason: {stop_reason}",
            "",
            "## Artifacts",
            f"- output_json: {cfg.output_json}",
            f"- output_csv: {cfg.output_csv}",
            f"- output_log: {cfg.output_log}",
        ],
    )
    logger.log(
        f"DONE family={family_info['resolved']} method={cfg.method} "
        f"abs_delta_e={delta_abs:.6e} runtime_s={runtime_s:.1f}"
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay HH VQE from ADAPT state with ADAPT-family generator contract.")
    p.add_argument("--adapt-input-json", type=Path, required=True)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-csv", type=Path, default=None)
    p.add_argument("--output-md", type=Path, default=None)
    p.add_argument("--output-log", type=Path, default=None)
    p.add_argument("--tag", type=str, default=None)

    p.add_argument("--generator-family", type=str, default="match_adapt")
    p.add_argument("--fallback-family", type=str, default="full_meta")
    p.add_argument("--legacy-paop-key", type=str, default="paop_lf_std")
    p.add_argument(
        "--replay-seed-policy",
        type=str,
        default="auto",
        choices=sorted(REPLAY_SEED_POLICIES),
        help=(
            "Replay seed initialization policy.  "
            "'auto' (default): branch on initial_state.handoff_state_kind — "
            "prepared_state -> residual_only, reference_state -> scaffold_plus_zero.  "
            "'scaffold_plus_zero': first replay block = adapt theta, rest zero.  "
            "'residual_only': all replay blocks start at zero.  "
            "'tile_adapt': legacy tiled seed [theta*, theta*, ...]."
        ),
    )
    p.add_argument(
        "--replay-continuation-mode",
        type=str,
        default="legacy",
        choices=["legacy", "phase1_v1", "phase2_v1", "phase3_v1"],
        help="Replay continuation mode (default: legacy). phase1_v1 is staged replay; phase2_v1 adds memory reuse/refresh; phase3_v1 adds generator/motif/symmetry telemetry.",
    )
    p.add_argument(
        "--phase3-symmetry-mitigation-mode",
        choices=["off", "verify_only", "postselect_diag_v1", "projector_renorm_v1"],
        default="off",
        help="Optional Phase 3 replay symmetry mitigation hook. verify_only preserves current behavior; active symmetry modes remain metadata/telemetry in replay and are enforced in the noise oracle path.",
    )

    p.add_argument("--L", type=int, default=None)
    p.add_argument("--t", type=float, default=None)
    p.add_argument("--u", type=float, default=None)
    p.add_argument("--dv", type=float, default=None)
    p.add_argument("--omega0", type=float, default=None)
    p.add_argument("--g-ep", type=float, default=None, dest="g_ep")
    p.add_argument("--n-ph-max", type=int, default=None, dest="n_ph_max")
    p.add_argument("--boson-encoding", type=str, default=None)
    p.add_argument("--ordering", type=str, default=None)
    p.add_argument("--boundary", type=str, default=None)
    p.add_argument("--sector-n-up", type=int, default=None)
    p.add_argument("--sector-n-dn", type=int, default=None)

    p.add_argument("--reps", type=int, default=None)
    p.add_argument("--restarts", type=int, default=16)
    p.add_argument("--maxiter", type=int, default=12000)
    p.add_argument("--method", type=str, default="SPSA")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--energy-backend", type=str, default="one_apply_compiled", choices=["legacy", "one_apply_compiled"])
    p.add_argument("--progress-every-s", type=float, default=60.0)
    p.add_argument("--wallclock-cap-s", type=int, default=43200)

    p.add_argument("--paop-r", type=int, default=1)
    p.add_argument("--paop-split-paulis", action="store_true")
    p.add_argument("--no-paop-split-paulis", dest="paop_split_paulis", action="store_false")
    p.set_defaults(paop_split_paulis=False)
    p.add_argument("--paop-prune-eps", type=float, default=0.0)
    p.add_argument("--paop-normalization", type=str, default="none", choices=["none", "fro", "maxcoeff"])

    p.add_argument("--spsa-a", type=float, default=0.2)
    p.add_argument("--spsa-c", type=float, default=0.1)
    p.add_argument("--spsa-alpha", type=float, default=0.602)
    p.add_argument("--spsa-gamma", type=float, default=0.101)
    p.add_argument("--spsa-A", type=float, default=10.0)
    p.add_argument("--spsa-avg-last", type=int, default=0)
    p.add_argument("--spsa-eval-repeats", type=int, default=1)
    p.add_argument("--spsa-eval-agg", type=str, default="mean", choices=["mean", "median"])
    p.add_argument("--replay-freeze-fraction", type=float, default=0.2)
    p.add_argument("--replay-unfreeze-fraction", type=float, default=0.3)
    p.add_argument("--replay-full-fraction", type=float, default=0.5)
    p.add_argument("--replay-qn-spsa-refresh-every", type=int, default=5)
    p.add_argument("--replay-qn-spsa-refresh-mode", type=str, default="diag_rms_grad", choices=["diag_rms_grad"])

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = json.loads(Path(args.adapt_input_json).read_text(encoding="utf-8"))
    cfg = _build_cfg(args, payload)
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
