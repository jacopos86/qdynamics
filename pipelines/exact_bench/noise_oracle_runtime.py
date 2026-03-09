#!/usr/bin/env python3
"""Noise/runtime expectation oracle utilities for HH/Hubbard validation.

This module stays in wrapper/benchmark space. It does not modify core operator
algebra modules and only adapts existing PauliPolynomial + ansatz objects to
Qiskit primitives at the boundary.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import SuzukiTrotter


@dataclass(frozen=True)
class OracleConfig:
    noise_mode: str = "ideal"  # ideal | shots | aer_noise | runtime
    shots: int = 2048
    seed: int = 7
    oracle_repeats: int = 1
    oracle_aggregate: str = "mean"  # mean | median
    backend_name: str | None = None
    use_fake_backend: bool = False
    approximation: bool = False
    abelian_grouping: bool = True
    allow_aer_fallback: bool = True
    aer_fallback_mode: str = "sampler_shots"
    omp_shm_workaround: bool = True
    mitigation: dict[str, Any] | str = "none"
    symmetry_mitigation: dict[str, Any] | str = "off"


@dataclass(frozen=True)
class MitigationConfig:
    mode: str = "none"  # none | readout | zne | dd
    zne_scales: tuple[float, ...] = ()
    dd_sequence: str | None = None


@dataclass(frozen=True)
class OracleEstimate:
    mean: float
    std: float
    stdev: float
    stderr: float
    n_samples: int
    raw_values: list[float]
    aggregate: str


@dataclass(frozen=True)
class SymmetryMitigationConfig:
    mode: str = "off"  # off | verify_only | postselect_diag_v1 | projector_renorm_v1
    num_sites: int | None = None
    ordering: str = "blocked"
    sector_n_up: int | None = None
    sector_n_dn: int | None = None


_MITIGATION_MODES = {"none", "readout", "zne", "dd"}
_SYMMETRY_MITIGATION_MODES = {"off", "verify_only", "postselect_diag_v1", "projector_renorm_v1"}


def _parse_zne_scales(raw: Any) -> list[float]:
    if raw is None:
        return []
    if isinstance(raw, str):
        tokens = [tok.strip() for tok in str(raw).split(",")]
        vals = [tok for tok in tokens if tok]
    elif isinstance(raw, Sequence):
        vals = [str(v).strip() for v in list(raw)]
        vals = [tok for tok in vals if tok]
    else:
        vals = [str(raw).strip()]
    out: list[float] = []
    for tok in vals:
        value = float(tok)
        if (not np.isfinite(value)) or (value <= 0.0):
            raise ValueError(f"Invalid mitigation zne scale {tok!r}; expected finite > 0.")
        out.append(float(value))
    return out


def normalize_mitigation_config(mitigation: Any) -> dict[str, Any]:
    mode = "none"
    zne_scales: list[float] = []
    dd_sequence: str | None = None

    if mitigation is None:
        pass
    elif isinstance(mitigation, MitigationConfig):
        mode = str(mitigation.mode).strip().lower() or "none"
        zne_scales = _parse_zne_scales(list(mitigation.zne_scales))
        dd_sequence = None if mitigation.dd_sequence is None else str(mitigation.dd_sequence)
    elif isinstance(mitigation, str):
        mode = str(mitigation).strip().lower() or "none"
    elif isinstance(mitigation, Mapping):
        mode = str(mitigation.get("mode", mitigation.get("mitigation", "none"))).strip().lower() or "none"
        zne_raw = mitigation.get("zne_scales", mitigation.get("zneScales", []))
        zne_scales = _parse_zne_scales(zne_raw)
        dd_raw = mitigation.get("dd_sequence", mitigation.get("ddSequence", None))
        dd_sequence = None if dd_raw is None else str(dd_raw)
    else:
        raise ValueError(
            "Unsupported mitigation config type; expected str, dict, MitigationConfig, or None."
        )

    if mode not in _MITIGATION_MODES:
        raise ValueError(
            f"Unsupported mitigation mode {mode!r}; expected one of {sorted(_MITIGATION_MODES)}."
        )

    return {
        "mode": str(mode),
        "zne_scales": [float(x) for x in zne_scales],
        "dd_sequence": dd_sequence,
    }


def normalize_symmetry_mitigation_config(symmetry_mitigation: Any) -> dict[str, Any]:
    mode = "off"
    num_sites: int | None = None
    ordering = "blocked"
    sector_n_up: int | None = None
    sector_n_dn: int | None = None

    if symmetry_mitigation is None:
        pass
    elif isinstance(symmetry_mitigation, SymmetryMitigationConfig):
        mode = str(symmetry_mitigation.mode).strip().lower() or "off"
        num_sites = (
            None if symmetry_mitigation.num_sites is None else int(symmetry_mitigation.num_sites)
        )
        ordering = str(symmetry_mitigation.ordering).strip().lower() or "blocked"
        sector_n_up = (
            None if symmetry_mitigation.sector_n_up is None else int(symmetry_mitigation.sector_n_up)
        )
        sector_n_dn = (
            None if symmetry_mitigation.sector_n_dn is None else int(symmetry_mitigation.sector_n_dn)
        )
    elif isinstance(symmetry_mitigation, str):
        mode = str(symmetry_mitigation).strip().lower() or "off"
    elif isinstance(symmetry_mitigation, Mapping):
        mode = str(
            symmetry_mitigation.get("mode", symmetry_mitigation.get("symmetry_mitigation", "off"))
        ).strip().lower() or "off"
        num_sites_raw = symmetry_mitigation.get("num_sites", symmetry_mitigation.get("L", None))
        ordering_raw = symmetry_mitigation.get("ordering", "blocked")
        n_up_raw = symmetry_mitigation.get("sector_n_up", symmetry_mitigation.get("n_up", None))
        n_dn_raw = symmetry_mitigation.get("sector_n_dn", symmetry_mitigation.get("n_dn", None))
        num_sites = None if num_sites_raw is None else int(num_sites_raw)
        ordering = str(ordering_raw).strip().lower() or "blocked"
        sector_n_up = None if n_up_raw is None else int(n_up_raw)
        sector_n_dn = None if n_dn_raw is None else int(n_dn_raw)
    else:
        raise ValueError(
            "Unsupported symmetry mitigation config type; expected str, dict, SymmetryMitigationConfig, or None."
        )

    if mode not in _SYMMETRY_MITIGATION_MODES:
        raise ValueError(
            f"Unsupported symmetry mitigation mode {mode!r}; expected one of {sorted(_SYMMETRY_MITIGATION_MODES)}."
        )

    return {
        "mode": str(mode),
        "num_sites": (None if num_sites is None else int(num_sites)),
        "ordering": str(ordering),
        "sector_n_up": (None if sector_n_up is None else int(sector_n_up)),
        "sector_n_dn": (None if sector_n_dn is None else int(sector_n_dn)),
    }


def normalize_ideal_reference_symmetry_mitigation(
    symmetry_mitigation: Any,
    *,
    noise_mode: str,
) -> dict[str, Any]:
    cfg = normalize_symmetry_mitigation_config(symmetry_mitigation)
    if str(noise_mode).strip().lower() == "runtime" and str(cfg.get("mode", "off")) not in {"off", "verify_only"}:
        downgraded = dict(cfg)
        downgraded["mode"] = "verify_only"
        return downgraded
    return cfg


@dataclass(frozen=True)
class NoiseBackendInfo:
    noise_mode: str
    estimator_kind: str
    backend_name: str | None = None
    using_fake_backend: bool = False
    details: dict[str, Any] = field(default_factory=dict)


def _to_ixyz(label_exyz: str) -> str:
    return (
        str(label_exyz)
        .replace("e", "I")
        .replace("x", "X")
        .replace("y", "Y")
        .replace("z", "Z")
    )


_PAULI_POLY_TO_QOP_MATH = "H = sum_j c_j P_j  ->  SparsePauliOp([(P_j, c_j)])"


def _pauli_poly_to_sparse_pauli_op(poly: Any, tol: float = 1e-12) -> SparsePauliOp:
    """Convert repo PauliPolynomial (exyz labels) into SparsePauliOp."""
    terms = list(poly.return_polynomial())
    if not terms:
        return SparsePauliOp.from_list([("I", 0.0)])

    nq = int(terms[0].nqubit())
    coeff_map: dict[str, complex] = {}
    for term in terms:
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tol):
            continue
        lbl = _to_ixyz(str(term.pw2strng()))
        coeff_map[lbl] = coeff_map.get(lbl, 0.0 + 0.0j) + coeff

    cleaned = [(lbl, c) for lbl, c in coeff_map.items() if abs(c) > float(tol)]
    if not cleaned:
        cleaned = [("I" * nq, 0.0 + 0.0j)]
    return SparsePauliOp.from_list(cleaned).simplify(atol=float(tol))


def _ansatz_terms_with_parameters(ansatz: Any, theta: np.ndarray) -> list[tuple[Any, float]]:
    theta = np.asarray(theta, dtype=float).reshape(-1)
    num_parameters = int(getattr(ansatz, "num_parameters", -1))
    if num_parameters < 0:
        raise ValueError("ansatz is missing num_parameters")
    if int(theta.size) != num_parameters:
        raise ValueError(f"theta length {int(theta.size)} does not match ansatz.num_parameters={num_parameters}")

    reps = int(getattr(ansatz, "reps", 1))
    out: list[tuple[Any, float]] = []
    k = 0

    layer_term_groups = getattr(ansatz, "layer_term_groups", None)
    if isinstance(layer_term_groups, list) and layer_term_groups:
        for _ in range(reps):
            for _name, terms in layer_term_groups:
                val = float(theta[k])
                for term in terms:
                    out.append((term.polynomial, val))
                k += 1
    else:
        base_terms = list(getattr(ansatz, "base_terms", []))
        if not base_terms:
            raise ValueError("ansatz has no base_terms/layer_term_groups")
        for _ in range(reps):
            for term in base_terms:
                out.append((term.polynomial, float(theta[k])))
                k += 1

    if k != int(theta.size):
        raise RuntimeError(
            f"ansatz parameter traversal consumed {k}, expected {int(theta.size)}"
        )
    return out


def _append_reference_state(circuit: QuantumCircuit, reference_state: np.ndarray) -> None:
    ref = np.asarray(reference_state, dtype=complex).reshape(-1)
    dim = int(1 << int(circuit.num_qubits))
    if ref.size != dim:
        raise ValueError(
            f"reference_state dimension {ref.size} does not match num_qubits={circuit.num_qubits}"
        )
    nrm = float(np.linalg.norm(ref))
    if nrm <= 0.0:
        raise ValueError("reference_state has zero norm")
    ref = ref / nrm

    nz = np.where(np.abs(ref) > 1e-12)[0]
    if nz.size == 1:
        idx = int(nz[0])
        phase = ref[idx]
        if abs(abs(phase) - 1.0) <= 1e-10:
            bit = format(idx, f"0{circuit.num_qubits}b")
            for q in range(circuit.num_qubits):
                if bit[circuit.num_qubits - 1 - q] == "1":
                    circuit.x(q)
            return

    circuit.initialize(ref, list(range(circuit.num_qubits)))


def _ansatz_to_circuit(
    ansatz: Any,
    theta: np.ndarray,
    *,
    num_qubits: int,
    reference_state: np.ndarray | None = None,
    coefficient_tolerance: float = 1e-12,
) -> QuantumCircuit:
    """Convert existing hardcoded ansatz object into a Qiskit circuit."""
    qc = QuantumCircuit(int(num_qubits))
    if reference_state is not None:
        _append_reference_state(qc, np.asarray(reference_state, dtype=complex))

    terms = _ansatz_terms_with_parameters(ansatz, np.asarray(theta, dtype=float))
    synthesis = SuzukiTrotter(order=2, reps=1, preserve_order=True)

    for poly, angle in terms:
        qop = _pauli_poly_to_sparse_pauli_op(poly, tol=float(coefficient_tolerance))
        coeffs = np.asarray(qop.coeffs, dtype=complex).reshape(-1)
        if coeffs.size == 0 or np.max(np.abs(coeffs)) <= float(coefficient_tolerance):
            continue
        gate = PauliEvolutionGate(qop, time=float(angle), synthesis=synthesis)
        qc.append(gate, list(range(int(num_qubits))))
    return qc


def _load_fake_backend(name: str | None) -> tuple[Any, str]:
    try:
        from qiskit_ibm_runtime import fake_provider
    except Exception as exc:
        raise RuntimeError(
            "Unable to import qiskit_ibm_runtime.fake_provider; install qiskit-ibm-runtime."
        ) from exc

    class_name = str(name).strip() if name is not None else "FakeManilaV2"
    if class_name and not class_name.startswith("Fake"):
        class_name = f"Fake{class_name.replace('-', '_').replace(' ', '').title().replace('_', '')}V2"
    backend_cls = getattr(fake_provider, class_name, None)
    if backend_cls is None:
        available = sorted([x for x in dir(fake_provider) if x.startswith("Fake") and x.endswith("V2")])
        sample = ", ".join(available[:8])
        raise ValueError(
            f"Unknown fake backend '{class_name}'. Available examples: {sample}"
        )
    return backend_cls(), class_name


def _resolve_noise_backend(cfg: OracleConfig) -> tuple[Any, str, bool]:
    if bool(cfg.use_fake_backend):
        backend, name = _load_fake_backend(cfg.backend_name)
        return backend, name, True

    if cfg.backend_name is None:
        backend, name = _load_fake_backend("FakeManilaV2")
        return backend, name, True

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except Exception as exc:
        raise RuntimeError(
            "qiskit_ibm_runtime is required for real backend lookup. "
            "Use --use-fake-backend or install/configure qiskit-ibm-runtime."
        ) from exc

    try:
        service = QiskitRuntimeService()
        backend = service.backend(str(cfg.backend_name))
        return backend, str(cfg.backend_name), False
    except Exception as exc:
        raise RuntimeError(
            f"Unable to resolve runtime backend '{cfg.backend_name}'. "
            "Check IBM Runtime credentials, backend name, or pass --use-fake-backend."
        ) from exc


_OMP_SHM_MARKERS = (
    "OMP: Error #178",
    "Can't open SHM2",
    "Function Can't open SHM2 failed",
    "OMP: System error",
)
_AER_PREFLIGHT_OK_CACHE: set[tuple[str, int, int | None, bool, bool]] = set()


def _tail_text(text: str, max_chars: int = 2400) -> str:
    cleaned = str(text).strip()
    if not cleaned:
        return "<no output captured>"
    if len(cleaned) <= int(max_chars):
        return cleaned
    return "..." + cleaned[-int(max_chars):]


def _looks_like_openmp_shm_abort(text: str) -> bool:
    lowered = str(text).lower()
    if not lowered:
        return False
    return any(str(marker).lower() in lowered for marker in _OMP_SHM_MARKERS)


def _apply_omp_env_workaround(cfg: OracleConfig) -> bool:
    if not bool(cfg.omp_shm_workaround):
        return False
    changed = False
    if os.environ.get("KMP_USE_SHM") != "0":
        os.environ["KMP_USE_SHM"] = "0"
        changed = True
    if os.environ.get("OMP_NUM_THREADS") != "1":
        os.environ["OMP_NUM_THREADS"] = "1"
        changed = True
    return changed


def _preflight_aer_environment(cfg: OracleConfig, mode: str) -> None:
    key = (
        str(mode),
        int(cfg.shots),
        (None if cfg.seed is None else int(cfg.seed)),
        bool(cfg.approximation),
        bool(cfg.abelian_grouping),
    )
    if key in _AER_PREFLIGHT_OK_CACHE:
        return

    payload = {
        "mode": str(mode),
        "shots": int(cfg.shots),
        "seed": (None if cfg.seed is None else int(cfg.seed)),
        "approximation": bool(cfg.approximation),
        "abelian_grouping": bool(cfg.abelian_grouping),
    }
    script = r"""
import json
import sys

cfg = json.loads(sys.argv[1])
mode = str(cfg.get("mode", "shots")).strip().lower()

from qiskit_aer.primitives import Estimator as AerEstimator

backend_options = {}
if mode == "aer_noise":
    from qiskit_aer.noise import NoiseModel
    backend_options["noise_model"] = NoiseModel()

run_options = {"shots": int(cfg["shots"])}
seed = cfg.get("seed", None)
if seed is not None:
    run_options["seed"] = int(seed)
    run_options["seed_simulator"] = int(seed)

_ = AerEstimator(
    backend_options=backend_options if backend_options else None,
    run_options=run_options,
    approximation=bool(cfg.get("approximation", False)),
    abelian_grouping=bool(cfg.get("abelian_grouping", True)),
)
print("AER_PREFLIGHT_OK")
"""
    env = None
    if bool(cfg.omp_shm_workaround):
        env = dict(os.environ)
        env["KMP_USE_SHM"] = "0"
        env["OMP_NUM_THREADS"] = "1"

    result = subprocess.run(
        [sys.executable, "-c", script, json.dumps(payload, sort_keys=True)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    if int(result.returncode) != 0:
        combined = f"{result.stdout}\n{result.stderr}"
        detail_tail = _tail_text(combined)
        if _looks_like_openmp_shm_abort(combined):
            raise RuntimeError(
                "Aer preflight failed due to OpenMP shared-memory restrictions in this environment "
                "(detected OMP/SHM2 failure). This is an environment-level crash path, not a script logic "
                "error. Modes 'shots' and 'aer_noise' are local/offline and do not require IBM Runtime "
                "credentials. Run this command in a shell/runtime with working shared-memory support "
                "(for example, a non-sandbox terminal with functional /dev/shm or equivalent). "
                f"Preflight stderr/stdout tail:\n{detail_tail}"
            )
        raise RuntimeError(
            "Aer preflight failed before noisy execution started. "
            f"Preflight stderr/stdout tail:\n{detail_tail}"
        )

    _AER_PREFLIGHT_OK_CACHE.add(key)


def _build_estimator(
    cfg: OracleConfig,
) -> tuple[Any, Any | None, NoiseBackendInfo]:
    mode = str(cfg.noise_mode).strip().lower()
    mitigation_cfg = normalize_mitigation_config(getattr(cfg, "mitigation", "none"))
    symmetry_cfg = normalize_symmetry_mitigation_config(getattr(cfg, "symmetry_mitigation", "off"))
    if mode not in {"ideal", "shots", "aer_noise", "runtime"}:
        raise ValueError(f"Unsupported noise_mode: {mode}")

    if mode == "ideal":
        try:
            from qiskit.primitives import StatevectorEstimator
        except Exception as exc:
            raise RuntimeError(
                "Failed to import StatevectorEstimator. Ensure qiskit primitives are available."
            ) from exc
        estimator = StatevectorEstimator()
        info = NoiseBackendInfo(
            noise_mode=mode,
            estimator_kind="qiskit.primitives.StatevectorEstimator",
            backend_name="statevector_simulator",
            using_fake_backend=False,
            details={
                "shots": None,
                "mitigation": dict(mitigation_cfg),
                "symmetry_mitigation": dict(symmetry_cfg),
            },
        )
        return estimator, None, info

    if mode in {"shots", "aer_noise"}:
        if str(os.environ.get("HH_FORCE_SAMPLER_FALLBACK", "0")).strip() == "1":
            raise RuntimeError(
                "OMP: Error #178: Forced sampler fallback via HH_FORCE_SAMPLER_FALLBACK=1."
            )
        env_workaround_applied = _apply_omp_env_workaround(cfg)
        _preflight_aer_environment(cfg, mode)
        try:
            from qiskit_aer.primitives import Estimator as AerEstimator
        except Exception as exc:
            raise RuntimeError(
                "Failed to import qiskit_aer.primitives.Estimator. Install qiskit-aer."
            ) from exc

        backend_options: dict[str, Any] = {}
        backend_name = "aer_simulator"
        using_fake = False
        details: dict[str, Any] = {
            "shots": int(cfg.shots),
            "aer_failed": False,
            "fallback_used": False,
            "fallback_mode": str(cfg.aer_fallback_mode),
            "fallback_reason": "",
            "env_workaround_applied": bool(cfg.omp_shm_workaround or env_workaround_applied),
            "mitigation": dict(mitigation_cfg),
            "symmetry_mitigation": dict(symmetry_cfg),
        }

        if mode == "aer_noise":
            try:
                from qiskit_aer.noise import NoiseModel
            except Exception as exc:
                raise RuntimeError(
                    "Failed to import qiskit_aer.noise.NoiseModel for aer_noise mode."
                ) from exc
            backend_obj, backend_name, using_fake = _resolve_noise_backend(cfg)
            noise_model = NoiseModel.from_backend(backend_obj)
            backend_options["noise_model"] = noise_model
            details["noise_model_basis_gates"] = list(getattr(noise_model, "basis_gates", []))

        run_options: dict[str, Any] = {"shots": int(cfg.shots)}
        if cfg.seed is not None:
            run_options["seed"] = int(cfg.seed)
            run_options["seed_simulator"] = int(cfg.seed)
        estimator = AerEstimator(
            backend_options=backend_options if backend_options else None,
            run_options=run_options,
            approximation=bool(cfg.approximation),
            abelian_grouping=bool(cfg.abelian_grouping),
        )
        info = NoiseBackendInfo(
            noise_mode=mode,
            estimator_kind="qiskit_aer.primitives.Estimator",
            backend_name=backend_name,
            using_fake_backend=using_fake,
            details=details,
        )
        return estimator, None, info

    # mode == "runtime"
    try:
        from qiskit_ibm_runtime import (
            QiskitRuntimeService,
            Session,
            EstimatorV2 as RuntimeEstimatorV2,
        )
    except Exception as exc:
        raise RuntimeError(
            "runtime mode requires qiskit-ibm-runtime. Install and configure IBM Runtime."
        ) from exc

    if cfg.backend_name is None:
        raise RuntimeError(
            "runtime mode requires --backend-name <ibm_backend>."
        )

    try:
        service = QiskitRuntimeService()
        backend = service.backend(str(cfg.backend_name))
        session = Session(service=service, backend=backend)
        estimator = RuntimeEstimatorV2(mode=session)
        info = NoiseBackendInfo(
            noise_mode=mode,
            estimator_kind="qiskit_ibm_runtime.EstimatorV2",
            backend_name=str(cfg.backend_name),
            using_fake_backend=False,
            details={
                "shots": int(cfg.shots),
                "mitigation": dict(mitigation_cfg),
                "symmetry_mitigation": dict(symmetry_cfg),
            },
        )
        return estimator, session, info
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize IBM Runtime Estimator. "
            "Verify IBM credentials (`QISKIT_IBM_TOKEN`), backend availability, and account access."
        ) from exc


def _extract_expectation_value(result: Any) -> float:
    if hasattr(result, "values"):
        vals = np.asarray(getattr(result, "values"), dtype=float).reshape(-1)
        if vals.size > 0:
            return float(vals[0])

    try:
        first = result[0]
    except Exception:
        first = None

    if first is not None:
        data = getattr(first, "data", None)
        if data is not None and hasattr(data, "evs"):
            evs = np.asarray(getattr(data, "evs"), dtype=float).reshape(-1)
            if evs.size > 0:
                return float(evs[0])
        if hasattr(first, "value"):
            return float(np.real(getattr(first, "value")))
        if hasattr(first, "evs"):
            evs = np.asarray(getattr(first, "evs"), dtype=float).reshape(-1)
            if evs.size > 0:
                return float(evs[0])

    if hasattr(result, "evs"):
        evs = np.asarray(getattr(result, "evs"), dtype=float).reshape(-1)
        if evs.size > 0:
            return float(evs[0])

    raise RuntimeError(
        f"Unable to extract expectation value from estimator result type: {type(result)!r}"
    )


def _run_estimator_job(estimator: Any, circuit: QuantumCircuit, observable: SparsePauliOp) -> float:
    errors: list[Exception] = []

    # V2-style tuple(pub) invocation
    for pub in (
        [(circuit, observable)],
        [(circuit, [observable])],
    ):
        try:
            job = estimator.run(pub)
            result = job.result()
            return float(np.real(_extract_expectation_value(result)))
        except Exception as exc:
            errors.append(exc)

    # V1-style invocation
    try:
        job = estimator.run([circuit], [observable])
        result = job.result()
        return float(np.real(_extract_expectation_value(result)))
    except Exception as exc:
        errors.append(exc)

    msg = "; ".join(f"{type(e).__name__}: {e}" for e in errors)
    raise RuntimeError(f"Estimator execution failed across known call paths. Details: {msg}")


def _term_measurement_circuit(base: QuantumCircuit, pauli_label_ixyz: str) -> QuantumCircuit:
    """Rotate into Pauli measurement basis and measure all qubits."""
    label = str(pauli_label_ixyz).upper()
    n = int(base.num_qubits)
    if len(label) != n:
        raise ValueError(f"Pauli label length {len(label)} does not match circuit qubits {n}")

    qc = base.copy()
    for q in range(n):
        op = label[n - 1 - q]  # left-to-right is q_(n-1)..q_0; q0 rightmost
        if op == "X":
            qc.h(q)
        elif op == "Y":
            qc.sdg(q)
            qc.h(q)
        elif op in {"I", "Z"}:
            continue
        else:
            raise ValueError(f"Unsupported Pauli op '{op}' in '{label}'")
    qc.measure_all()
    return qc


def _pauli_parity_from_bitstring(bitstr_raw: str, pauli_label_ixyz: str, n_qubits: int) -> float:
    label = str(pauli_label_ixyz).upper()
    if len(label) != int(n_qubits):
        raise ValueError(f"Pauli label length {len(label)} does not match n_qubits={n_qubits}")
    bitstr = str(bitstr_raw).replace(" ", "")
    if len(bitstr) < int(n_qubits):
        bitstr = bitstr.zfill(int(n_qubits))
    parity = 1.0
    for q in range(int(n_qubits)):
        if label[int(n_qubits) - 1 - q] == "I":
            continue
        bit = bitstr[-1 - int(q)]
        parity *= (-1.0 if bit == "1" else 1.0)
    return float(parity)


def _pauli_expectation_from_counts(counts: dict[str, int], pauli_label_ixyz: str, n_qubits: int) -> float:
    label = str(pauli_label_ixyz).upper()
    if len(label) != int(n_qubits):
        raise ValueError(f"Pauli label length {len(label)} does not match n_qubits={n_qubits}")
    active_q = [q for q in range(int(n_qubits)) if label[int(n_qubits) - 1 - q] != "I"]
    if not active_q:
        return 1.0
    shots = int(sum(int(v) for v in counts.values()))
    if shots <= 0:
        raise RuntimeError("Sampler returned zero total shots.")

    acc = 0.0
    for bitstr_raw, ct in counts.items():
        acc += _pauli_parity_from_bitstring(str(bitstr_raw), label, int(n_qubits)) * float(ct)
    return float(acc / float(shots))


def _observable_is_diagonal(observable: SparsePauliOp) -> bool:
    for label, _coeff in observable.to_list():
        if any(ch not in {"I", "Z"} for ch in str(label).upper()):
            return False
    return True


def _spin_orbital_index_sets(num_sites: int, ordering: str) -> tuple[list[int], list[int]]:
    if str(ordering).strip().lower() == "interleaved":
        return [2 * i for i in range(int(num_sites))], [2 * i + 1 for i in range(int(num_sites))]
    return list(range(int(num_sites))), list(range(int(num_sites), 2 * int(num_sites)))


def _bitstring_passes_sector(
    bitstr_raw: str,
    *,
    n_qubits: int,
    num_sites: int,
    ordering: str,
    sector_n_up: int,
    sector_n_dn: int,
) -> bool:
    bitstr = str(bitstr_raw).replace(" ", "")
    if len(bitstr) < int(n_qubits):
        bitstr = bitstr.zfill(int(n_qubits))
    alpha_indices, beta_indices = _spin_orbital_index_sets(int(num_sites), str(ordering))
    n_up = sum(1 for idx in alpha_indices if bitstr[-1 - int(idx)] == "1")
    n_dn = sum(1 for idx in beta_indices if bitstr[-1 - int(idx)] == "1")
    return int(n_up) == int(sector_n_up) and int(n_dn) == int(sector_n_dn)


def _diagonal_expectation_from_counts(counts: dict[str, int], observable: SparsePauliOp) -> float:
    total = 0.0 + 0.0j
    n = int(observable.num_qubits)
    for label, coeff in observable.to_list():
        total += complex(coeff) * complex(_pauli_expectation_from_counts(counts, str(label), n), 0.0)
    return float(np.real(total))


def _exact_postselected_diagonal_expectation(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    symmetry_cfg: Mapping[str, Any],
) -> tuple[float, float]:
    psi = np.asarray(Statevector.from_instruction(circuit).data, dtype=complex).reshape(-1)
    n_qubits = int(circuit.num_qubits)
    kept_prob = 0.0
    total = 0.0 + 0.0j
    for idx, amp in enumerate(psi):
        prob = float(abs(complex(amp)) ** 2)
        if prob <= 1e-18:
            continue
        bitstr = format(int(idx), f"0{n_qubits}b")
        if not _bitstring_passes_sector(
            bitstr,
            n_qubits=int(n_qubits),
            num_sites=int(symmetry_cfg.get("num_sites", 0)),
            ordering=str(symmetry_cfg.get("ordering", "blocked")),
            sector_n_up=int(symmetry_cfg.get("sector_n_up", 0)),
            sector_n_dn=int(symmetry_cfg.get("sector_n_dn", 0)),
        ):
            continue
        kept_prob += prob
        for label, coeff in observable.to_list():
            total += complex(coeff) * complex(
                _pauli_parity_from_bitstring(bitstr, str(label), int(n_qubits)) * prob,
                0.0,
            )
    if kept_prob <= 0.0:
        raise RuntimeError("Symmetry postselection retained zero probability mass.")
    return float(np.real(total) / kept_prob), float(kept_prob)


def _exact_projector_renorm_diagonal_expectation(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    symmetry_cfg: Mapping[str, Any],
) -> tuple[float, float]:
    psi = np.asarray(Statevector.from_instruction(circuit).data, dtype=complex).reshape(-1)
    n_qubits = int(circuit.num_qubits)
    sector_prob = 0.0
    numerator = 0.0 + 0.0j
    for idx, amp in enumerate(psi):
        prob = float(abs(complex(amp)) ** 2)
        if prob <= 1e-18:
            continue
        bitstr = format(int(idx), f"0{n_qubits}b")
        in_sector = _bitstring_passes_sector(
            bitstr,
            n_qubits=int(n_qubits),
            num_sites=int(symmetry_cfg.get("num_sites", 0)),
            ordering=str(symmetry_cfg.get("ordering", "blocked")),
            sector_n_up=int(symmetry_cfg.get("sector_n_up", 0)),
            sector_n_dn=int(symmetry_cfg.get("sector_n_dn", 0)),
        )
        if not in_sector:
            continue
        sector_prob += prob
        for label, coeff in observable.to_list():
            numerator += complex(coeff) * complex(
                _pauli_parity_from_bitstring(bitstr, str(label), int(n_qubits)) * prob,
                0.0,
            )
    if sector_prob <= 0.0:
        raise RuntimeError("Projector renormalization retained zero probability mass.")
    return float(np.real(numerator) / sector_prob), float(sector_prob)


def _sample_measurement_counts(
    circuit: QuantumCircuit,
    cfg: OracleConfig,
    *,
    repeat_idx: int,
) -> dict[str, int]:
    measured = circuit.copy()
    measured.measure_all()
    mode = str(cfg.noise_mode).strip().lower()
    if mode == "shots" or mode == "ideal":
        from qiskit.primitives import StatevectorSampler

        sampler = StatevectorSampler(
            default_shots=int(cfg.shots),
            seed=int(cfg.seed) + int(repeat_idx),
        )
        job = sampler.run([measured])
        result = job.result()
        return dict(result[0].join_data().get_counts())

    if mode == "aer_noise":
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel

        backend_obj, _backend_name, _using_fake = _resolve_noise_backend(cfg)
        noise_model = NoiseModel.from_backend(backend_obj)
        sim = AerSimulator(noise_model=noise_model, seed_simulator=int(cfg.seed) + int(repeat_idx))
        compiled = transpile(measured, sim, optimization_level=0)
        result = sim.run(compiled, shots=int(cfg.shots)).result()
        counts = result.get_counts()
        if isinstance(counts, list):
            counts = counts[0]
        return dict(counts)

    raise RuntimeError(f"Counts-based symmetry mitigation is unavailable for noise_mode={mode!r}.")


def _postselected_counts_and_fraction(
    counts: Mapping[str, int],
    *,
    n_qubits: int,
    symmetry_cfg: Mapping[str, Any],
) -> tuple[dict[str, int], float]:
    kept: dict[str, int] = {}
    total = int(sum(int(v) for v in counts.values()))
    if total <= 0:
        raise RuntimeError("Counts-based symmetry mitigation received zero total shots.")
    kept_total = 0
    for bitstr_raw, ct_raw in counts.items():
        ct = int(ct_raw)
        if ct <= 0:
            continue
        if _bitstring_passes_sector(
            str(bitstr_raw),
            n_qubits=int(n_qubits),
            num_sites=int(symmetry_cfg.get("num_sites", 0)),
            ordering=str(symmetry_cfg.get("ordering", "blocked")),
            sector_n_up=int(symmetry_cfg.get("sector_n_up", 0)),
            sector_n_dn=int(symmetry_cfg.get("sector_n_dn", 0)),
        ):
            kept[str(bitstr_raw)] = kept.get(str(bitstr_raw), 0) + int(ct)
            kept_total += int(ct)
    if kept_total <= 0:
        raise RuntimeError("Symmetry postselection retained zero shots.")
    return kept, float(kept_total) / float(total)


def _projector_renorm_diagonal_expectation_from_counts(
    counts: Mapping[str, int],
    observable: SparsePauliOp,
    *,
    n_qubits: int,
    symmetry_cfg: Mapping[str, Any],
) -> tuple[float, float]:
    total_shots = int(sum(int(v) for v in counts.values()))
    if total_shots <= 0:
        raise RuntimeError("Counts-based projector renormalization received zero total shots.")
    sector_shots = 0
    numerator = 0.0 + 0.0j
    for bitstr_raw, ct_raw in counts.items():
        ct = int(ct_raw)
        if ct <= 0:
            continue
        if not _bitstring_passes_sector(
            str(bitstr_raw),
            n_qubits=int(n_qubits),
            num_sites=int(symmetry_cfg.get("num_sites", 0)),
            ordering=str(symmetry_cfg.get("ordering", "blocked")),
            sector_n_up=int(symmetry_cfg.get("sector_n_up", 0)),
            sector_n_dn=int(symmetry_cfg.get("sector_n_dn", 0)),
        ):
            continue
        sector_shots += int(ct)
        for label, coeff in observable.to_list():
            numerator += complex(coeff) * complex(
                _pauli_parity_from_bitstring(str(bitstr_raw), str(label), int(n_qubits)) * float(ct),
                0.0,
            )
    if sector_shots <= 0:
        raise RuntimeError("Projector renormalization retained zero shots.")
    sector_prob = float(sector_shots) / float(total_shots)
    numerator_expectation = numerator / float(total_shots)
    return float(np.real(numerator_expectation) / sector_prob), float(sector_prob)


def _run_sampler_fallback_job(
    sampler: Any,
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
) -> float:
    total = 0.0 + 0.0j
    n = int(observable.num_qubits)
    for label, coeff in observable.to_list():
        lbl = str(label).upper()
        meas_qc = _term_measurement_circuit(circuit, lbl)
        job = sampler.run([meas_qc])
        result = job.result()
        counts = result[0].join_data().get_counts()
        exp_lbl = _pauli_expectation_from_counts(counts, lbl, n)
        total += complex(coeff) * complex(exp_lbl, 0.0)
    return float(np.real(total))


class ExpectationOracle:
    """Shared expectation-value oracle for ideal/noisy/runtime execution."""

    def __init__(self, config: OracleConfig):
        self.config = OracleConfig(
            noise_mode=str(config.noise_mode).strip().lower(),
            shots=int(config.shots),
            seed=int(config.seed),
            oracle_repeats=max(1, int(config.oracle_repeats)),
            oracle_aggregate=str(config.oracle_aggregate).strip().lower(),
            backend_name=(None if config.backend_name is None else str(config.backend_name)),
            use_fake_backend=bool(config.use_fake_backend),
            approximation=bool(config.approximation),
            abelian_grouping=bool(config.abelian_grouping),
            allow_aer_fallback=bool(config.allow_aer_fallback),
            aer_fallback_mode=str(config.aer_fallback_mode).strip().lower(),
            omp_shm_workaround=bool(config.omp_shm_workaround),
            mitigation=normalize_mitigation_config(getattr(config, "mitigation", "none")),
            symmetry_mitigation=normalize_symmetry_mitigation_config(
                getattr(config, "symmetry_mitigation", "off")
            ),
        )
        if self.config.oracle_aggregate not in {"mean", "median"}:
            raise ValueError(
                f"Unsupported oracle_aggregate={self.config.oracle_aggregate}; use mean or median."
            )
        if self.config.aer_fallback_mode not in {"sampler_shots"}:
            raise ValueError(
                f"Unsupported aer_fallback_mode={self.config.aer_fallback_mode}; use sampler_shots."
            )

        self._sampler_fallback = None
        self._fallback_reason = ""
        self._estimator = None
        self._session = None
        self.backend_info = NoiseBackendInfo(
            noise_mode=str(self.config.noise_mode),
            estimator_kind="unknown",
            backend_name=None,
            using_fake_backend=bool(self.config.use_fake_backend),
            details={},
        )

        try:
            self._estimator, self._session, self.backend_info = _build_estimator(self.config)
        except Exception as exc:
            if self._can_fallback_from_error(exc):
                self._activate_sampler_fallback(reason=str(exc), aer_failed=True)
            else:
                raise
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
        self._closed = True

    def __enter__(self) -> "ExpectationOracle":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _update_backend_details(self, **updates: Any) -> None:
        details = dict(getattr(self.backend_info, "details", {}))
        details.update(updates)
        self.backend_info = NoiseBackendInfo(
            noise_mode=str(self.backend_info.noise_mode),
            estimator_kind=str(self.backend_info.estimator_kind),
            backend_name=self.backend_info.backend_name,
            using_fake_backend=bool(self.backend_info.using_fake_backend),
            details=details,
        )

    def _set_symmetry_mitigation_details(self, details_map: Mapping[str, Any]) -> None:
        self._update_backend_details(symmetry_mitigation=dict(details_map))

    def _maybe_evaluate_symmetry_mitigated(
        self,
        circuit: QuantumCircuit,
        observable: SparsePauliOp,
    ) -> OracleEstimate | None:
        symmetry_cfg = normalize_symmetry_mitigation_config(
            getattr(self.config, "symmetry_mitigation", "off")
        )
        requested_mode = str(symmetry_cfg.get("mode", "off"))
        details: dict[str, Any] = {
            "requested_mode": str(requested_mode),
            "applied_mode": str(requested_mode),
            "executed": False,
            "eligible": False,
            "fallback_reason": "",
            "retained_fraction_mean": None,
            "retained_fraction_samples": [],
            "sector_probability_mean": None,
            "sector_probability_samples": [],
            "sector_values": {
                "sector_n_up": symmetry_cfg.get("sector_n_up", None),
                "sector_n_dn": symmetry_cfg.get("sector_n_dn", None),
            },
            "estimator_form": "none",
        }
        if requested_mode in {"off", "verify_only"}:
            self._set_symmetry_mitigation_details(details)
            return None
        if not _observable_is_diagonal(observable):
            details["applied_mode"] = "verify_only"
            details["fallback_reason"] = "observable_not_diagonal"
            self._set_symmetry_mitigation_details(details)
            return None
        if str(self.config.noise_mode) == "runtime":
            details["applied_mode"] = "verify_only"
            details["fallback_reason"] = "runtime_counts_path_unavailable"
            self._set_symmetry_mitigation_details(details)
            return None
        required_keys = ("num_sites", "sector_n_up", "sector_n_dn")
        if any(symmetry_cfg.get(key, None) is None for key in required_keys):
            details["applied_mode"] = "verify_only"
            details["fallback_reason"] = "incomplete_sector_config"
            self._set_symmetry_mitigation_details(details)
            return None
        vals: list[float] = []
        retained: list[float] = []
        repeats = max(1, int(self.config.oracle_repeats))
        try:
            for rep in range(repeats):
                if str(self.config.noise_mode) == "ideal":
                    if requested_mode == "projector_renorm_v1":
                        val, retained_fraction = _exact_projector_renorm_diagonal_expectation(
                            circuit,
                            observable,
                            symmetry_cfg,
                        )
                    else:
                        val, retained_fraction = _exact_postselected_diagonal_expectation(
                            circuit,
                            observable,
                            symmetry_cfg,
                        )
                else:
                    counts = _sample_measurement_counts(circuit, self.config, repeat_idx=int(rep))
                    if requested_mode == "projector_renorm_v1":
                        val, retained_fraction = _projector_renorm_diagonal_expectation_from_counts(
                            counts,
                            observable,
                            n_qubits=int(circuit.num_qubits),
                            symmetry_cfg=symmetry_cfg,
                        )
                    else:
                        kept_counts, retained_fraction = _postselected_counts_and_fraction(
                            counts,
                            n_qubits=int(circuit.num_qubits),
                            symmetry_cfg=symmetry_cfg,
                        )
                        val = _diagonal_expectation_from_counts(kept_counts, observable)
                vals.append(float(val))
                retained.append(float(retained_fraction))
        except Exception as exc:
            if self._can_fallback_from_error(exc):
                self._activate_sampler_fallback(reason=str(exc), aer_failed=True)
            details["applied_mode"] = "verify_only"
            details["fallback_reason"] = str(exc)
            self._set_symmetry_mitigation_details(details)
            return None
        arr = np.asarray(vals, dtype=float)
        stdev = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        stderr = float(stdev / np.sqrt(float(arr.size))) if arr.size > 0 else float("nan")
        agg = float(np.median(arr)) if self.config.oracle_aggregate == "median" else float(np.mean(arr))
        details.update(
            {
                "applied_mode": str(requested_mode),
                "executed": True,
                "eligible": True,
                "fallback_reason": "",
                "retained_fraction_mean": (float(np.mean(retained)) if retained else None),
                "retained_fraction_samples": [float(x) for x in retained],
                "sector_probability_mean": (float(np.mean(retained)) if retained else None),
                "sector_probability_samples": [float(x) for x in retained],
                "estimator_form": (
                    "postselected_bitstring_average"
                    if requested_mode == "postselect_diag_v1"
                    else "projector_ratio_diag_v1"
                ),
            }
        )
        self._set_symmetry_mitigation_details(details)
        return OracleEstimate(
            mean=agg,
            std=stdev,
            stdev=stdev,
            stderr=stderr,
            n_samples=int(arr.size),
            raw_values=[float(x) for x in arr.tolist()],
            aggregate=self.config.oracle_aggregate,
        )

    def _fallback_allowed_for_mode(self) -> bool:
        return (
            str(self.config.noise_mode) in {"shots", "aer_noise"}
            and bool(self.config.allow_aer_fallback)
            and str(self.config.aer_fallback_mode) == "sampler_shots"
        )

    def _can_fallback_from_error(self, exc: Exception) -> bool:
        if not self._fallback_allowed_for_mode():
            return False
        return _looks_like_openmp_shm_abort(str(exc))

    def _activate_sampler_fallback(self, *, reason: str, aer_failed: bool) -> None:
        if self._sampler_fallback is None:
            try:
                from qiskit.primitives import StatevectorSampler
            except Exception as exc:
                raise RuntimeError(
                    "Failed to activate sampler fallback (`StatevectorSampler` unavailable)."
                ) from exc
            self._sampler_fallback = StatevectorSampler(
                default_shots=int(self.config.shots),
                seed=int(self.config.seed),
            )
        self._fallback_reason = str(reason)
        old = self.backend_info
        details = dict(getattr(old, "details", {}))
        details["aer_failed"] = bool(aer_failed)
        details["fallback_used"] = True
        details["fallback_mode"] = str(self.config.aer_fallback_mode)
        details["fallback_reason"] = str(reason)
        details["env_workaround_applied"] = bool(self.config.omp_shm_workaround)
        details["mitigation"] = normalize_mitigation_config(getattr(self.config, "mitigation", "none"))
        details["symmetry_mitigation"] = normalize_symmetry_mitigation_config(
            getattr(self.config, "symmetry_mitigation", "off")
        )
        self.backend_info = NoiseBackendInfo(
            noise_mode=str(self.config.noise_mode),
            estimator_kind="qiskit.primitives.StatevectorSampler(fallback)",
            backend_name=(old.backend_name or "statevector_sampler_fallback"),
            using_fake_backend=bool(old.using_fake_backend),
            details=details,
        )
        self._estimator = None

    def evaluate(self, circuit: QuantumCircuit, observable: SparsePauliOp) -> OracleEstimate:
        if self._closed:
            raise RuntimeError("ExpectationOracle is closed.")

        symmetry_est = self._maybe_evaluate_symmetry_mitigated(circuit, observable)
        if symmetry_est is not None:
            return symmetry_est

        vals: list[float] = []
        repeats = max(1, int(self.config.oracle_repeats))
        for _ in range(repeats):
            if self._sampler_fallback is not None:
                val = _run_sampler_fallback_job(self._sampler_fallback, circuit, observable)
                vals.append(float(np.real(val)))
                continue
            try:
                val = _run_estimator_job(self._estimator, circuit, observable)
                vals.append(float(np.real(val)))
            except Exception as exc:
                if self._can_fallback_from_error(exc):
                    self._activate_sampler_fallback(reason=str(exc), aer_failed=True)
                    val = _run_sampler_fallback_job(self._sampler_fallback, circuit, observable)
                    vals.append(float(np.real(val)))
                else:
                    raise

        arr = np.asarray(vals, dtype=float)
        stdev = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        stderr = float(stdev / np.sqrt(float(arr.size))) if arr.size > 0 else float("nan")
        if self.config.oracle_aggregate == "median":
            agg = float(np.median(arr))
        else:
            agg = float(np.mean(arr))

        return OracleEstimate(
            mean=agg,
            std=stdev,
            stdev=stdev,
            stderr=stderr,
            n_samples=int(arr.size),
            raw_values=[float(x) for x in arr.tolist()],
            aggregate=self.config.oracle_aggregate,
        )


_NUMBER_OPERATOR_MATH = "n_p = (I - Z_p) / 2"


def _number_operator_qop(num_qubits: int, index: int) -> SparsePauliOp:
    if index < 0 or index >= int(num_qubits):
        raise ValueError(f"index {index} out of range for num_qubits={num_qubits}")
    chars = ["I"] * int(num_qubits)
    chars[int(num_qubits) - 1 - int(index)] = "Z"
    z_label = "".join(chars)
    return SparsePauliOp.from_list(
        [
            ("I" * int(num_qubits), 0.5),
            (z_label, -0.5),
        ]
    ).simplify(atol=1e-12)


_DOUBLON_OPERATOR_MATH = "D_i = n_{i,up} n_{i,dn} = (I - Z_up - Z_dn + Z_up Z_dn) / 4"


def _doublon_site_qop(num_qubits: int, up_index: int, dn_index: int) -> SparsePauliOp:
    if up_index == dn_index:
        raise ValueError("up_index and dn_index must differ")
    chars_up = ["I"] * int(num_qubits)
    chars_dn = ["I"] * int(num_qubits)
    chars_both = ["I"] * int(num_qubits)
    chars_up[int(num_qubits) - 1 - int(up_index)] = "Z"
    chars_dn[int(num_qubits) - 1 - int(dn_index)] = "Z"
    chars_both[int(num_qubits) - 1 - int(up_index)] = "Z"
    chars_both[int(num_qubits) - 1 - int(dn_index)] = "Z"
    return SparsePauliOp.from_list(
        [
            ("I" * int(num_qubits), 0.25),
            ("".join(chars_up), -0.25),
            ("".join(chars_dn), -0.25),
            ("".join(chars_both), 0.25),
        ]
    ).simplify(atol=1e-12)


def _ordered_qop_from_exyz(
    ordered_labels_exyz: Sequence[str],
    coeff_map_exyz: dict[str, complex],
    *,
    tol: float = 1e-12,
) -> SparsePauliOp:
    terms: list[tuple[str, complex]] = []
    for lbl in ordered_labels_exyz:
        coeff = complex(coeff_map_exyz[lbl])
        if abs(coeff) <= float(tol):
            continue
        terms.append((_to_ixyz(lbl), coeff))
    if not terms:
        nq = len(ordered_labels_exyz[0]) if ordered_labels_exyz else 1
        terms = [("I" * nq, 0.0 + 0.0j)]
    return SparsePauliOp.from_list(terms).simplify(atol=float(tol))
