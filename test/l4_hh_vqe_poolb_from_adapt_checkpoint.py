#!/usr/bin/env python3
"""Run seeded fixed-ansatz VQE from an ADAPT B checkpoint state.

- Builds ADAPT pool B: uccsd_lifted + paop + hva
- Initializes from a checkpoint JSON containing amplitudes_qn_to_q0
- Writes: run JSON, CSV, MD summary, heartbeat checkpoint JSON, human log
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.hubbard_latex_python_pairs import (
    SPIN_DN,
    SPIN_UP,
    boson_qubits_per_site,
    build_hubbard_holstein_hamiltonian,
    mode_index,
)
from src.quantum.operator_pools import make_pool as make_paop_pool
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import (
    AnsatzTerm,
    HardcodedUCCSDAnsatz,
    HubbardHolsteinPhysicalTermwiseAnsatz,
    apply_exp_pauli_polynomial,
    expval_pauli_polynomial,
    exact_ground_energy_sector_hh,
    vqe_minimize,
)


AMPTYPE_CUTOFF = 1e-14


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def log(self, msg: str) -> None:
        ts = _now_utc()
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


@dataclass(frozen=True)
class RunConfig:
    adapt_input_json: Path
    tag: str
    output_json: Path
    output_csv: Path
    output_md: Path
    output_log: Path
    resume_json: Optional[Path]
    L: int
    t: float
    U: float
    dv: float
    omega0: float
    g_ep: float
    n_ph_max: int
    boson_encoding: str
    ordering: str
    boundary: str
    sector_n_up: int
    sector_n_dn: int
    paop_key: str
    paop_r: int
    paop_split_paulis: bool
    paop_prune_eps: float
    paop_normalization: str
    include_hva: bool
    reps: int
    restarts: int
    seed: int
    maxiter: int
    method: str
    progress_every_s: float
    wallclock_cap_s: int
    initial_state_name: str


def _normalize_state(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=complex).reshape(-1)
    nrm = float(np.linalg.norm(arr))
    if nrm <= 0.0:
        raise ValueError("Statevector has zero norm")
    return arr / nrm


def _statevector_to_amplitudes_qn_to_q0(vec: np.ndarray, *, cutoff: float = AMPTYPE_CUTOFF) -> dict[str, dict[str, float]]:
    nq = int(round(math.log2(int(vec.size))))
    out: dict[str, dict[str, float]] = {}
    for idx, amp in enumerate(np.asarray(vec, dtype=complex).reshape(-1)):
        if abs(amp) < float(cutoff):
            continue
        bit = format(idx, f"0{nq}b")
        out[bit] = {"re": float(np.real(amp)), "im": float(np.imag(amp))}
    return out


def _amplitudes_qn_to_q0_to_statevector(payload: Mapping[str, Any], *, nq: int) -> np.ndarray:
    vec = np.zeros(1 << int(nq), dtype=complex)
    for bit, coeff in payload.items():
        if bit not in str(bit):
            continue
        amp_re = float((coeff or {}).get("re", 0.0)) if isinstance(coeff, dict) else 0.0
        amp_im = float((coeff or {}).get("im", 0.0)) if isinstance(coeff, dict) else 0.0
        idx = int(str(bit), 2)
        vec[idx] = amp_re + 1.0j * amp_im
    return _normalize_state(vec)


# ────────────────────────────────────────────────────────────────────────────
# Pool builders
# ────────────────────────────────────────────────────────────────────────────

def _build_uccsd_ferm_only_lifted_pool(cfg: RunConfig) -> list[AnsatzTerm]:
    n_sites = int(cfg.L)
    base = HardcodedUCCSDAnsatz(
        dims=n_sites,
        num_particles=(int(cfg.sector_n_up), int(cfg.sector_n_dn)),
        reps=1,
        repr_mode="JW",
        indexing=str(cfg.ordering),
        include_singles=True,
        include_doubles=True,
    )
    ferm_nq = 2 * n_sites
    boson_bits = n_sites * int(boson_qubits_per_site(int(cfg.n_ph_max), str(cfg.boson_encoding)))
    nq_total = ferm_nq + boson_bits
    lifted_pool: list[AnsatzTerm] = []
    for op in base.base_terms:
        lifted = PauliPolynomial("JW")
        for term in op.polynomial.return_polynomial():
            coeff = complex(term.p_coeff)
            if abs(coeff) <= 1e-15:
                continue
            if abs(coeff.imag) > 1e-10:
                raise ValueError(f"Imaginary UCCSD coeff in {op.label}: {coeff}")
            ferm_ps = str(term.pw2strng())
            if len(ferm_ps) != ferm_nq:
                raise ValueError(f"Unexpected UCCSD Pauli length {len(ferm_ps)} != ferm_nq {ferm_nq}")
            full_ps = ("e" * boson_bits) + ferm_ps
            lifted.add_term(PauliTerm(nq_total, ps=full_ps, pc=float(coeff.real)))
        lifted_pool.append(AnsatzTerm(label=f"uccsd_ferm_lifted::{op.label}", polynomial=lifted))
    return lifted_pool


def _build_paop_pool(cfg: RunConfig) -> list[AnsatzTerm]:
    specs = make_paop_pool(
        str(cfg.paop_key),
        num_sites=int(cfg.L),
        num_particles=(int(cfg.sector_n_up), int(cfg.sector_n_dn)),
        n_ph_max=int(cfg.n_ph_max),
        boson_encoding=str(cfg.boson_encoding),
        ordering=str(cfg.ordering),
        boundary=str(cfg.boundary),
        paop_r=int(cfg.paop_r),
        paop_split_paulis=bool(cfg.paop_split_paulis),
        paop_prune_eps=float(cfg.paop_prune_eps),
        paop_normalization=str(cfg.paop_normalization),
    )
    return [AnsatzTerm(label=str(lbl), polynomial=poly) for lbl, poly in specs]


def _build_hva_pool(cfg: RunConfig) -> list[AnsatzTerm]:
    ptw = HubbardHolsteinPhysicalTermwiseAnsatz(
        dims=int(cfg.L),
        J=float(cfg.t),
        U=float(cfg.U),
        omega0=float(cfg.omega0),
        g=float(cfg.g_ep),
        n_ph_max=int(cfg.n_ph_max),
        boson_encoding=str(cfg.boson_encoding),
        v=float(cfg.dv),
        reps=1,
        repr_mode="JW",
        indexing=str(cfg.ordering),
        pbc=(str(cfg.boundary).strip().lower() == "periodic"),
    )
    return list(ptw.base_terms)


def _poly_signature(poly: Any, tol: float = 1e-12) -> tuple[tuple[str, float], ...]:
    sig: list[tuple[str, float]] = []
    for term in poly.return_polynomial():
        coeff = complex(term.p_coeff)
        if abs(coeff) <= tol:
            continue
        if abs(coeff.imag) > 1e-10:
            raise ValueError(f"Non-negligible imaginary coefficient {coeff}")
        sig.append((str(term.pw2strng()), round(float(coeff.real), 12)))
    sig.sort()
    return tuple(sig)


def _make_dedup_pool(
    ordered_components: list[tuple[str, list[AnsatzTerm]]],
) -> tuple[list[AnsatzTerm], dict[tuple[tuple[str, float], ...], dict[str, bool]], dict[str, Any]]:
    source_by_sig: dict[tuple[tuple[str, float], ...], dict[str, bool]] = {}
    raw_sizes: dict[str, int] = {}
    for source, ops in ordered_components:
        raw_sizes[source] = int(len(ops))
        for op in ops:
            sig = _poly_signature(op.polynomial)
            source_by_sig.setdefault(sig, {"uccsd": False, "paop": False, "hva": False})[source] = True

    dedup: list[AnsatzTerm] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    for _, ops in ordered_components:
        for op in ops:
            sig = _poly_signature(op.polynomial)
            if sig in seen:
                continue
            seen.add(sig)
            dedup.append(op)

    overlap_count = 0
    dedup_source_counts = {"uccsd": 0, "paop": 0, "hva": 0}
    for op in dedup:
        sig = _poly_signature(op.polynomial)
        flags = source_by_sig.get(sig, {"uccsd": False, "paop": False, "hva": False})
        sources = [k for k in ("uccsd", "paop", "hva") if bool(flags.get(k))]
        if len(sources) >= 2:
            overlap_count += 1
        for k in sources:
            dedup_source_counts[k] += 1

    meta = {
        "raw_sizes": raw_sizes,
        "dedup_total": int(len(dedup)),
        "dedup_source_presence_counts": dedup_source_counts,
        "overlap_count": int(overlap_count),
    }
    return dedup, source_by_sig, meta


class PoolTermwiseAnsatz:
    """Fixed ansatz from a list of Pauli operator terms (list reused each layer)."""

    def __init__(self, *, terms: list[AnsatzTerm], reps: int, nq: int) -> None:
        if int(reps) < 1:
            raise ValueError("reps must be >= 1")
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
        if int(theta.size) != int(self.num_parameters):
            raise ValueError(
                f"theta has wrong length {int(theta.size)}; expected {int(self.num_parameters)}"
            )
        if int(psi_ref.size) != (1 << int(self.nq)):
            raise ValueError(f"psi_ref length {int(psi_ref.size)} != 2^nq={1 << int(self.nq)}")
        theta_arr = np.asarray(theta, dtype=float)
        psi = np.array(np.asarray(psi_ref, dtype=complex).reshape(-1), copy=True)
        k = 0
        coeff_tol = 1e-12 if coefficient_tolerance is None else float(coefficient_tolerance)
        sort_flag = True if sort_terms is None else bool(sort_terms)

        for _ in range(self.reps):
            for term in self.base_terms:
                psi = apply_exp_pauli_polynomial(
                    psi,
                    term.polynomial,
                    float(theta_arr[k]),
                    ignore_identity=ignore_identity,
                    coefficient_tolerance=coeff_tol,
                    sort_terms=sort_flag,
                )
                k += 1
        return psi


def _load_checkpoint_state(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    settings = obj.get("settings", {})
    init = obj.get("initial_state", {})
    nq = int(settings.get("nq_total", settings.get("nq", 0) or 0))
    if nq <= 0:
        amp_obj = init.get("amplitudes_qn_to_q0", {})
        nq = len(next(iter(amp_obj.keys()))) if amp_obj else 0
        if nq == 0:
            raise ValueError("Cannot infer nq_total from checkpoint initial state amplitudes")
        # infer nq from bitstring key length
        # fallback to first key length
        nq = len(next(iter(amp_obj.keys()))) if amp_obj else nq
    amp_obj = init.get("amplitudes_qn_to_q0", {})
    psi = _amplitudes_qn_to_q0_to_statevector(
        {str(k): v for k, v in amp_obj.items()} if isinstance(amp_obj, dict) else {}, nq=nq
    )
    return psi, obj


def _read_resume_theta(path: Path, npar: int) -> Optional[np.ndarray]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    for key in ("best_theta", "vqe_best_theta", "theta_best", "last_theta", "checkpoint"):
        if key in raw and isinstance(raw[key], list):
            arr = np.asarray(raw[key], dtype=float)
            if arr.size == npar:
                return _normalize_point(arr)
    # nested forms
    for key in ("result", "payload", "adapt_vqe", "vqe"):
        node = raw.get(key)
        if isinstance(node, dict):
            for k2 in ("best_theta", "theta_best", "theta"):
                vals = node.get(k2)
                if isinstance(vals, list):
                    arr = np.asarray(vals, dtype=float)
                    if arr.size == npar:
                        return _normalize_point(arr)
    return None


def _normalize_point(theta: np.ndarray) -> np.ndarray:
    arr = np.asarray(theta, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    if not np.all(np.isfinite(arr)):
        raise ValueError("Resume theta contains non-finite values")
    return arr


def _extract_best_theta_from_checkpoint(payload: Mapping[str, Any], npar: int) -> Optional[np.ndarray]:
    for key in ("vqe", "adapt_vqe", "payload", "result"):
        node = payload.get(key)
        if isinstance(node, dict):
            cand = node.get("theta_best")
            if isinstance(cand, list):
                arr = np.asarray(cand, dtype=float)
                if arr.size == npar:
                    return _normalize_point(arr)
    for key in ("best_theta", "theta_restart_best"):
        if isinstance(payload.get(key), list):
            arr = np.asarray(payload[key], dtype=float)
            if arr.size == npar:
                return _normalize_point(arr)
    return None


def _json_dump(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _write_csv(path: Path, row: dict[str, Any]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(row.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerow(row)


def _write_md(path: Path, *, cfg: RunConfig, metrics: Mapping[str, Any], artifacts: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# L4 HH ADAPT-B-to-Pool-B VQE Summary",
        "",
        f"Generated UTC: { _now_utc() }",
        f"Tag: {cfg.tag}",
        "",
        "## Run Contract",
        "- Problem: HH",
        f"- L: {cfg.L}, boundary: {cfg.boundary}, ordering: {cfg.ordering}",
        f"- sector: ({cfg.sector_n_up},{cfg.sector_n_dn})",
        f"- boson_encoding: {cfg.boson_encoding}, n_ph_max: {cfg.n_ph_max}",
        f"- t={cfg.t}, U={cfg.U}, dv={cfg.dv}, omega0={cfg.omega0}, g_ep={cfg.g_ep}",
        "- seed state: Adapt-B checkpoint seed", 
        "",
        "## VQE Settings",
        f"- method: {cfg.method}",
        f"- reps: {cfg.reps}",
        f"- restarts: {cfg.restarts}",
        f"- seed: {cfg.seed}",
        f"- maxiter: {cfg.maxiter}",
        f"- progress_every_s: {cfg.progress_every_s}",
        f"- wallclock_cap_s: {cfg.wallclock_cap_s}",
        "- pool: uccsd_lifted + paop_lf_std + hva (deduped)",
        "",
        "## Key Metrics",
        f"- Best |ΔE|: {metrics.get('delta_E_abs_best')}",
        f"- Best relative_error: {metrics.get('relative_error_best')}",
        f"- Runtime (s): {metrics.get('runtime_s')}",
        f"- nfev_total: {metrics.get('nfev_total')}",
        f"- nit_total: {metrics.get('nit_total')}",
        f"- npar: {metrics.get('npar')}",
        f"- Best energy: {metrics.get('energy_best')}",
        f"- Exact sector energy: {metrics.get('exact_energy')}",
        f"- Stop reason: {metrics.get('stop_reason')}",
        f"- Gate pass (1e-2): {metrics.get('gate_pass')}",
        "",
        "## Artifacts",
        f"- state JSON: {artifacts.get('output_json')}",
        f"- checkpoint JSON: {artifacts.get('checkpoint_json')}",
        f"- CSV: {artifacts.get('output_csv')}",
        f"- log: {artifacts.get('output_log')}",
        f"- resume input: {artifacts.get('adapt_input_json')}",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _slope_last(points: list[float], times: list[float], window: int) -> Optional[float]:
    if len(points) < max(2, window):
        return None
    i0 = -window
    y0 = float(points[i0])
    t0 = float(times[i0])
    y1 = float(points[-1])
    t1 = float(times[-1])
    if not math.isfinite(t1 - t0) or (t1 - t0) == 0:
        return None
    return (y1 - y0) / (t1 - t0)


def _load_pool_meta(payload: Mapping[str, Any]) -> tuple[list[AnsatzTerm], int]:
    # placeholder: not needed; signature kept for call symmetry
    return [], 0


def run(cfg: RunConfig, logger: RunLogger) -> dict[str, Any]:
    logger.log(f"Loading seed state from {cfg.adapt_input_json}")
    start_vec, adapt_payload = _load_checkpoint_state(cfg.adapt_input_json)

    h_poly = build_hubbard_holstein_hamiltonian(
        dims=int(cfg.L),
        J=float(cfg.t),
        U=float(cfg.U),
        omega0=float(cfg.omega0),
        g=float(cfg.g_ep),
        n_ph_max=int(cfg.n_ph_max),
        boson_encoding=str(cfg.boson_encoding),
        repr_mode="JW",
        indexing=str(cfg.ordering),
        pbc=(str(cfg.boundary).strip().lower() == "periodic"),
    )

    e_exact = exact_ground_energy_sector_hh(
        h_poly,
        num_sites=int(cfg.L),
        num_particles=(int(cfg.sector_n_up), int(cfg.sector_n_dn)),
        n_ph_max=int(cfg.n_ph_max),
        boson_encoding=str(cfg.boson_encoding),
        indexing=str(cfg.ordering),
    )

    uccsd_pool = _build_uccsd_ferm_only_lifted_pool(cfg)
    paop_pool = _build_paop_pool(cfg)
    hva_pool = _build_hva_pool(cfg) if bool(cfg.include_hva) else []

    pool, source_by_sig, pool_meta = _make_dedup_pool([
        ("uccsd", uccsd_pool),
        ("paop", paop_pool),
        ("hva", hva_pool),
    ])

    ansatz = PoolTermwiseAnsatz(
        terms=pool,
        reps=int(cfg.reps),
        nq=int(
            2 * int(cfg.L)
            + int(cfg.L) * int(boson_qubits_per_site(int(cfg.n_ph_max), str(cfg.boson_encoding)))
        ),
    )

    npar = int(ansatz.num_parameters)
    logger.log(f"Pool-B fixed ansatz built: dedup={len(pool)} terms, npar={npar}")

    resume_theta: Optional[np.ndarray] = None
    if cfg.resume_json is not None:
        try:
            resume_theta = _read_resume_theta(cfg.resume_json, npar)
        except Exception:
            resume_theta = None
    else:
        # try to recover from last vqe state in adapt payload if embedded
        resume_theta = _extract_best_theta_from_checkpoint(adapt_payload, npar)

    if resume_theta is not None:
        logger.log(f"Resuming VQE from supplied theta with npar={npar}")

    best_delta_trace: list[float] = []
    best_time_trace: list[float] = []
    progress_events: list[dict[str, Any]] = []

    checkpoint_path = cfg.output_json.with_name(f"{cfg.output_json.stem}_checkpoint_state.json")
    run_t0 = time.perf_counter()
    wallhit = {"hit": False, "t": None}

    def _progress_logger(event: dict[str, Any]) -> None:
        ev = dict(event)
        ev_label = str(ev.get("event", ""))
        if "energy_restart_best" in ev and isinstance(ev["energy_restart_best"], (int, float)) and math.isfinite(ev["energy_restart_best"]):
            d_abs = abs(float(ev["energy_restart_best"]) - float(e_exact))
            elapsed = float(ev.get("elapsed_s", 0.0))
            best_delta_trace.append(d_abs)
            best_time_trace.append(elapsed)
            ev["delta_abs_best"] = float(d_abs)

        if ev_label in {"heartbeat", "run_end", "run_interrupted", "early_stop_triggered", "restart_end"}:
            now_s = float(ev.get("elapsed_s", 0.0))
            if math.isfinite(now_s):
                message = {
                    "event": ev_label,
                    "elapsed_s": now_s,
                    "restarts_total": int(ev.get("restarts_total", 0)),
                    "nfev_so_far": int(ev.get("nfev_so_far", 0)) if isinstance(ev.get("nfev_so_far"), (int, float)) else 0,
                    "nfev_restart": int(ev.get("nfev_restart", 0)) if isinstance(ev.get("nfev_restart"), (int, float)) else 0,
                    "energy_current": ev.get("energy_current"),
                    "delta_abs_best": ev.get("delta_abs_best"),
                    "best_restart": ev.get("best_restart"),
                    "method": ev.get("method"),
                    "message": ev.get("message", ""),
                }
                if "theta_restart_best" in ev and ev.get("theta_restart_best"):
                    message["theta_restart_best"] = ev["theta_restart_best"]
                progress_events.append(message)

        if ev_label in {"run_end", "run_interrupted", "early_stop_triggered", "restart_end"}:
            logger.log(f"VQE {ev_label}: {ev}")

        if ev_label in {"run_end", "run_interrupted", "early_stop_triggered"}:
            now = float(ev.get("elapsed_s", 0.0))
            payload = {
                "generated_utc": _now_utc(),
                "settings": vars(cfg),
                "meta": {
                    "event": str(ev_label),
                    "npar": npar,
                    "elapsed_s": now,
                    "run_id": str(cfg.tag),
                },
                "initial_state": {
                    "source": "adapt_checkpoint",
                    "amplitudes_qn_to_q0": _statevector_to_amplitudes_qn_to_q0(start_vec),
                    "nq_total": int(round(math.log2(int(start_vec.size)))),
                },
                "adapt_vqe": {
                    "energy": float(ev.get("energy_best", ev.get("energy_current", float("nan")))),
                    "abs_delta_e": float(abs(float(ev.get("energy_best", ev.get("energy_current", float("nan")))) - float(e_exact)),) if _is_finite_or_nan(ev.get("energy_current", float("nan"))) else None,
                    "relative_error_abs": float(
                        abs(float(ev.get("energy_best", ev.get("energy_current", float("nan"))) - float(e_exact))
                            / max(abs(float(e_exact)), 1e-14)
                        )
                        if _is_finite_or_nan(ev.get("energy_current", float("nan"))) else None,
                    ),
                },
                "vqe": {
                    "npar": npar,
                    "nfev_total": int(ev.get("nfev_total", 0) or 0),
                    "nit_total": int(ev.get("nit_total", 0) or 0),
                },
                "runtime_s": float(time.perf_counter() - run_t0),
                "pool": pool_meta,
            }
            ckpt = dict(payload)
            if "theta_restart_best" in ev and isinstance(ev.get("theta_restart_best"), list):
                ckpt["best_theta"] = [float(x) for x in ev["theta_restart_best"]]
            _json_dump(checkpoint_path, ckpt)

    def _early_stop_checker(payload: dict[str, Any]) -> bool:
        elapsed = float(payload.get("elapsed_s", 0.0))
        if elapsed >= float(cfg.wallclock_cap_s):
            wallhit["hit"] = True
            wallhit["t"] = elapsed
            logger.log(f"Wallclock cap hit at {elapsed:.1f}s; triggering early stop")
            return True
        return False

    if resume_theta is not None:
        logger.log(
            "Loaded resume theta from checkpoint, but this backend does not accept "
            "an explicit initial optimizer point; continuing with seeded restarts."
        )

    logger.log("Starting pool-B conventional VQE")
    start_t = time.perf_counter()
    vqe_res = vqe_minimize(
        h_poly,
        ansatz,
        start_vec,
        restarts=int(cfg.restarts),
        seed=int(cfg.seed),
        method=str(cfg.method),
        maxiter=int(cfg.maxiter),
        progress_logger=_progress_logger,
        progress_every_s=float(cfg.progress_every_s),
        progress_label="l4_hh_vqe_poolb_from_adapt",
        track_history=False,
        emit_theta_in_progress=True,
        return_best_on_keyboard_interrupt=True,
        early_stop_checker=_early_stop_checker,
    )
    runtime_s = float(time.perf_counter() - start_t)

    theta_best = np.asarray(vqe_res.theta, dtype=float)
    psi_best = _normalize_state(ansatz.prepare_state(theta_best, start_vec))

    e_best = float(vqe_res.energy)
    delta_best = abs(e_best - float(e_exact))
    rel_best = float(delta_best / max(abs(float(e_exact)), 1e-14))
    slope_last5 = _slope_last(
        points=[float(x) for x in best_delta_trace],
        times=[float(x) for x in best_time_trace],
        window=5,
    )
    slope_last10 = _slope_last(
        points=[float(x) for x in best_delta_trace],
        times=[float(x) for x in best_time_trace],
        window=10,
    )

    gate_pass = delta_best <= 1e-2
    status = {
        "generated_utc": _now_utc(),
        "settings": {
            "L": int(cfg.L),
            "problem": "hh",
            "t": float(cfg.t),
            "u": float(cfg.U),
            "dv": float(cfg.dv),
            "omega0": float(cfg.omega0),
            "g_ep": float(cfg.g_ep),
            "n_ph_max": int(cfg.n_ph_max),
            "boson_encoding": str(cfg.boson_encoding),
            "ordering": str(cfg.ordering),
            "boundary": str(cfg.boundary),
            "sector": [int(cfg.sector_n_up), int(cfg.sector_n_dn)],
            "pool_variant": "pool_b",
            "reps": int(cfg.reps),
            "restarts": int(cfg.restarts),
            "seed": int(cfg.seed),
            "maxiter": int(cfg.maxiter),
            "method": str(cfg.method),
            "progress_every_s": float(cfg.progress_every_s),
            "wallclock_cap_s": int(cfg.wallclock_cap_s),
            "paop_key": str(cfg.paop_key),
            "paop_r": int(cfg.paop_r),
            "paop_split_paulis": bool(cfg.paop_split_paulis),
            "include_hva": bool(cfg.include_hva),
            "run_id": str(cfg.tag),
            "initial_state_name": str(cfg.initial_state_name),
        },
        "exact": {
            "E_exact_sector": float(e_exact),
        },
        "adapt_vqe": {
            "energy": float(e_best),
            "abs_delta_e": float(delta_best),
            "relative_error_abs": float(rel_best),
            "npar": npar,
            "nfev": int(vqe_res.nfev),
            "nit": int(vqe_res.nit),
            "best_restart": int(vqe_res.best_restart),
            "success": bool(vqe_res.success),
            "message": str(vqe_res.message),
            "runtime_s_run": float(runtime_s),
            "runtime_s_elapsed_total": float(time.perf_counter() - run_t0),
            "stop_reason": (
                "wallclock_cap" if bool(wallhit["hit"]) else
                ("converged" if bool(vqe_res.success) else str(vqe_res.message))
            ),
        },
        "pool": pool_meta,
        "initial_state": {
            "source": "adapt_b_checkpoint",
            "nq_total": int(start_vec.size.bit_length() - 1),
            "amplitudes_qn_to_q0": _statevector_to_amplitudes_qn_to_q0(start_vec),
            "norm": float(np.linalg.norm(start_vec)),
            "amplitude_cutoff": AMPTYPE_CUTOFF,
        },
        "initial_state_export": str(cfg.adapt_input_json),
        "best_theta": [float(x) for x in theta_best.tolist()],
        "best_state_amplitudes_qn_to_q0": _statevector_to_amplitudes_qn_to_q0(psi_best),
        "best_delta_traces": {
            "slope_last5_per_s": slope_last5,
            "slope_last10_per_s": slope_last10,
        },
        "progress_events_count": int(len(progress_events)),
        "progress_events_tail": [
            {
                k: v for k, v in ev.items() if k not in {"theta_restart_best"}
            }
            for ev in progress_events[-20:]
        ],
    }

    _json_dump(cfg.output_json, status)
    gate = "pass" if gate_pass else "fail"
    csv_row = {
        "attempt_id": str(cfg.tag),
        "stage": "vqe_pool_b_from_adapt",
        "delta_E_abs_best": delta_best,
        "relative_error_best": rel_best,
        "gate_pass": bool(gate_pass),
        "runtime_s": runtime_s,
        "stop_reason": status["adapt_vqe"]["stop_reason"],
        "artifact_json": str(cfg.output_json),
        "artifact_log": str(cfg.output_log),
    }
    _write_csv(cfg.output_csv, csv_row)

    _write_md(
        cfg.output_md,
        cfg=cfg,
        metrics={
            "delta_E_abs_best": delta_best,
            "relative_error_best": rel_best,
            "runtime_s": runtime_s,
            "nfev_total": int(vqe_res.nfev),
            "nit_total": int(vqe_res.nit),
            "npar": npar,
            "energy_best": e_best,
            "exact_energy": float(e_exact),
            "stop_reason": status["adapt_vqe"]["stop_reason"],
            "gate_pass": bool(gate_pass),
        },
        artifacts={
            "output_json": str(cfg.output_json),
            "output_csv": str(cfg.output_csv),
            "checkpoint_json": str(checkpoint_path),
            "output_log": str(cfg.output_log),
            "adapt_input_json": str(cfg.adapt_input_json),
            "checkpoint_input_json": str(cfg.resume_json) if cfg.resume_json is not None else "",
        },
    )

    ckpt_payload = dict(status)
    ckpt_payload["meta"] = {
        "event": "final_state",
        "run_id": str(cfg.tag),
        "runtime_s": runtime_s,
        "stop_reason": status["adapt_vqe"]["stop_reason"],
        "best_energy": float(e_best),
    }
    ckpt_payload["best_state_amplitudes_qn_to_q0"] = _statevector_to_amplitudes_qn_to_q0(psi_best)
    ckpt_payload["best_theta"] = [float(x) for x in theta_best.tolist()]
    ckpt_payload["slope_last5_per_s"] = slope_last5
    ckpt_payload["slope_last10_per_s"] = slope_last10
    _json_dump(checkpoint_path, ckpt_payload)

    logger.log(
        f"VQE done: gate={gate}, best|ΔE|={delta_best:.6e}, rel={rel_best:.6e}, "
        f"runtime_s={runtime_s:.1f}, stop={status['adapt_vqe']['stop_reason']}"
    )
    return status


def parse_args() -> RunConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapt-input-json", type=Path, required=True)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--output-json", type=Path, default=None)
    ap.add_argument("--output-csv", type=Path, default=None)
    ap.add_argument("--output-md", type=Path, default=None)
    ap.add_argument("--output-log", type=Path, default=None)
    ap.add_argument("--resume-json", type=Path, default=None)

    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--U", type=float, default=4.0)
    ap.add_argument("--dv", type=float, default=0.0)
    ap.add_argument("--omega0", type=float, default=1.0)
    ap.add_argument("--g-ep", type=float, default=0.5)
    ap.add_argument("--n-ph-max", type=int, default=1)
    ap.add_argument("--boson-encoding", type=str, default="binary")
    ap.add_argument("--ordering", type=str, default="blocked")
    ap.add_argument("--boundary", type=str, default="open")
    ap.add_argument("--sector", type=str, default="2,2")

    ap.add_argument("--paop-key", type=str, default="paop_lf_std")
    ap.add_argument("--paop-r", type=int, default=1)
    ap.add_argument("--paop-split-paulis", action="store_true")
    ap.add_argument("--no-paop-split-paulis", dest="paop_split_paulis", action="store_false")
    ap.set_defaults(paop_split_paulis=False)
    ap.add_argument("--paop-prune-eps", type=float, default=0.0)
    ap.add_argument("--paop-normalization", type=str, default="none")
    ap.add_argument("--no-hva", dest="include_hva", action="store_false")
    ap.set_defaults(include_hva=True)

    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--restarts", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--maxiter", type=int, default=12000)
    ap.add_argument("--method", type=str, default="COBYLA")
    ap.add_argument("--progress-every-s", type=float, default=60.0)
    ap.add_argument("--wallclock-cap-s", type=int, default=43200)
    ap.add_argument("--initial-state-name", type=str, default="adapt_B_checkpoint")
    args = ap.parse_args()

    if args.output_json is None:
        args.output_json = Path(f"artifacts/useful/L4/{args.tag}_poolb_vqe_from_adaptB_state.json")
    if args.output_csv is None:
        args.output_csv = Path(f"artifacts/useful/L4/{args.tag}_poolb_vqe_from_adaptB.csv")
    if args.output_md is None:
        args.output_md = Path(f"artifacts/useful/L4/{args.tag}_poolb_vqe_from_adaptB.md")
    if args.output_log is None:
        args.output_log = Path(f"artifacts/useful/L4/{args.tag}_poolb_vqe_from_adaptB.log")

    nup, ndn = [int(x.strip()) for x in str(args.sector).split(",")]

    return RunConfig(
        adapt_input_json=args.adapt_input_json,
        tag=args.tag,
        output_json=args.output_json,
        output_csv=args.output_csv,
        output_md=args.output_md,
        output_log=args.output_log,
        resume_json=args.resume_json,
        L=int(args.L),
        t=float(args.t),
        U=float(args.U),
        dv=float(args.dv),
        omega0=float(args.omega0),
        g_ep=float(args.g_ep),
        n_ph_max=int(args.n_ph_max),
        boson_encoding=str(args.boson_encoding),
        ordering=str(args.ordering),
        boundary=str(args.boundary),
        sector_n_up=int(nup),
        sector_n_dn=int(ndn),
        paop_key=str(args.paop_key),
        paop_r=int(args.paop_r),
        paop_split_paulis=bool(args.paop_split_paulis),
        paop_prune_eps=float(args.paop_prune_eps),
        paop_normalization=str(args.paop_normalization),
        include_hva=bool(args.include_hva),
        reps=int(args.reps),
        restarts=int(args.restarts),
        seed=int(args.seed),
        maxiter=int(args.maxiter),
        method=str(args.method),
        progress_every_s=float(args.progress_every_s),
        wallclock_cap_s=int(args.wallclock_cap_s),
        initial_state_name=str(args.initial_state_name),
    )


def _is_finite_or_nan(x: Any) -> bool:
    try:
        return isinstance(x, (int, float)) and math.isfinite(float(x))
    except Exception:
        return False


def main() -> int:
    cfg = parse_args()
    logger = RunLogger(cfg.output_log)

    try:
        run(cfg, logger)
    except Exception as exc:
        logger.log(f"FATAL: {exc}")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
