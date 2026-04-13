#!/usr/bin/env python3
"""Measured/oracle fixed-manifold McLachlan for locked HH scaffolds."""

from __future__ import annotations

import argparse
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

from pipelines.exact_bench.noise_oracle_runtime import (  # noqa: E402
    ExpectationOracle,
    OracleConfig,
)
from pipelines.hardcoded.hh_fixed_manifold_mclachlan import (  # noqa: E402
    DEFAULT_PARETO_ARTIFACT,
    DEFAULT_LOCKED_7TERM_ARTIFACT,
    FixedManifoldRunSpec,
    LoadedRunContext,
    load_run_context,
)
from pipelines.hardcoded.hh_fixed_manifold_observables import (  # noqa: E402
    CheckpointObservablePlan,
    ObservableSpec,
    build_checkpoint_observable_plan,
)
from pipelines.hardcoded.hh_vqe_from_adapt_family import AnsatzTerm  # noqa: E402
from src.quantum.ansatz_parameterization import build_parameter_layout  # noqa: E402
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor  # noqa: E402
from src.quantum.drives_time_potential import (  # noqa: E402
    build_gaussian_sinusoid_density_drive,
    default_spatial_weights,
    reference_method_name,
)
from src.quantum.hubbard_latex_python_pairs import build_hubbard_holstein_drive  # noqa: E402
from src.quantum.pauli_polynomial_class import PauliPolynomial  # noqa: E402
from src.quantum.qubitization_module import PauliTerm  # noqa: E402
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix  # noqa: E402

try:  # noqa: E402
    from scipy.linalg import expm as _dense_expm
    from scipy.sparse import csc_matrix as _csc_matrix
    from scipy.sparse.linalg import expm_multiply as _expm_multiply
except ImportError:  # pragma: no cover
    _dense_expm = None
    _csc_matrix = None
    _expm_multiply = None

_SPARSE_EMPTY_OBSERVABLE_GUARD = 1.0e-11


@dataclass(frozen=True)
class FixedManifoldMeasuredConfig:
    regularization_lambda: float = 1.0e-8
    pinv_rcond: float = 1.0e-10
    observable_drop_abs_tol: float = 1.0e-12
    observable_hermiticity_tol: float = 1.0e-10
    observable_max_terms: int = 512
    variance_floor: float = 0.0
    g_symmetrize_tol: float = 1.0e-12


@dataclass(frozen=True)
class FixedManifoldDriveConfig:
    enable_drive: bool = False
    drive_A: float = 0.0
    drive_omega: float = 1.0
    drive_tbar: float = 1.0
    drive_phi: float = 0.0
    drive_pattern: str = "staggered"
    drive_custom_s: str | None = None
    drive_include_identity: bool = False
    drive_time_sampling: str = "midpoint"
    drive_t0: float = 0.0
    exact_steps_multiplier: int = 1


@dataclass(frozen=True)
class FixedManifoldAugmentationConfig:
    drive_generator_mode: str = "none"


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _default_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_fixed_mclachlan_7term_measured")


def _parse_drive_custom_weights(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "":
        return None
    if text.startswith("["):
        vals = json.loads(text)
    else:
        vals = [float(x) for x in text.split(",") if x.strip()]
    return [float(x) for x in vals]


"""
psi_ref = computational_basis_state(bitstring_hf)
"""
def _validate_basis_reference_state(payload: Mapping[str, Any]) -> dict[str, Any]:
    ansatz_input = payload.get("ansatz_input_state", None)
    if not isinstance(ansatz_input, Mapping):
        raise ValueError("Measured fixed-manifold runner requires ansatz_input_state.")
    source = str(ansatz_input.get("source", "")).strip().lower()
    if source != "hf":
        raise ValueError(
            f"Measured fixed-manifold v1 requires ansatz_input_state.source='hf'; got {source!r}."
        )
    amps = ansatz_input.get("amplitudes_qn_to_q0", None)
    if not isinstance(amps, Mapping):
        raise ValueError("ansatz_input_state.amplitudes_qn_to_q0 must be present.")
    nz: list[tuple[str, complex]] = []
    for bit, coeff_payload in amps.items():
        if not isinstance(coeff_payload, Mapping):
            continue
        coeff = complex(
            float(coeff_payload.get("re", 0.0)),
            float(coeff_payload.get("im", 0.0)),
        )
        if abs(coeff) > 1.0e-12:
            nz.append((str(bit), coeff))
    if len(nz) != 1:
        raise ValueError(
            "Measured fixed-manifold v1 requires a one-hot computational-basis HF reference state."
        )
    bitstring, amplitude = nz[0]
    if abs(abs(amplitude) - 1.0) > 1.0e-10:
        raise ValueError(
            "Measured fixed-manifold v1 requires a unit-magnitude computational-basis HF reference state."
        )
    return {
        "source": str(source),
        "bitstring_qn_to_q0": str(bitstring),
        "nq_total": int(ansatz_input.get("nq_total", len(bitstring))),
    }


def _build_exact_reference(psi_initial: np.ndarray, hmat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    evals, evecs = np.linalg.eigh(np.asarray(hmat, dtype=complex))
    coeffs0 = np.asarray(np.conjugate(evecs).T @ np.asarray(psi_initial, dtype=complex), dtype=complex)
    return np.asarray(evals, dtype=float), np.asarray(evecs, dtype=complex) @ np.diag(coeffs0)


def _exact_state_at(
    time_value: float,
    *,
    evals: np.ndarray,
    evecs_times_coeffs0: np.ndarray,
) -> np.ndarray:
    phases = np.exp(-1.0j * np.asarray(evals, dtype=float) * float(time_value))
    return np.asarray(evecs_times_coeffs0 @ phases, dtype=complex).reshape(-1)


def _poly_from_real_coeff_map(
    coeff_map: Mapping[str, complex],
    *,
    nq: int,
    drop_abs_tol: float,
    hermiticity_tol: float,
    context: str,
) -> PauliPolynomial:
    poly = PauliPolynomial("JW")
    for label in sorted(coeff_map.keys()):
        coeff = complex(coeff_map[label])
        if abs(coeff) <= float(drop_abs_tol):
            continue
        if abs(coeff.imag) > float(hermiticity_tol):
            raise ValueError(f"{context} produced non-Hermitian coefficient for {label}: {coeff}.")
        poly.add_term(PauliTerm(int(nq), ps=str(label), pc=float(coeff.real)))
    return poly


"""
psi(theta,0_extra) = psi(theta) for zero-angle appended generators
"""
def _augment_loaded_context_with_drive_generator(
    loaded: LoadedRunContext,
    *,
    drive_cfg: FixedManifoldDriveConfig,
    aug_cfg: FixedManifoldAugmentationConfig,
) -> LoadedRunContext:
    mode = str(aug_cfg.drive_generator_mode).strip().lower()
    if mode in {"", "none"}:
        return loaded
    if mode != "aligned_density":
        raise ValueError(f"Unknown drive_generator_mode={mode!r}.")
    if not bool(drive_cfg.enable_drive):
        raise ValueError("drive_generator_mode requires enable_drive=True.")

    custom_weights = _parse_drive_custom_weights(drive_cfg.drive_custom_s)
    weights = default_spatial_weights(
        int(loaded.cfg.L),
        mode=str(drive_cfg.drive_pattern),
        custom=custom_weights,
    )
    nq_total = int(round(np.log2(int(np.asarray(loaded.replay_context.psi_ref).size))))
    drive_poly = build_hubbard_holstein_drive(
        dims=int(loaded.cfg.L),
        v_t=[float(x) for x in np.asarray(weights, dtype=float).tolist()],
        v0=[0.0] * int(loaded.cfg.L),
        repr_mode="JW",
        indexing=str(loaded.cfg.ordering),
        nq_override=int(nq_total),
    )
    extra_term = AnsatzTerm(
        label=f"drive_aligned_density(pattern={str(drive_cfg.drive_pattern)})",
        polynomial=drive_poly,
    )

    old_layout = loaded.replay_context.base_layout
    new_terms = tuple(loaded.replay_context.replay_terms) + (extra_term,)
    new_layout = build_parameter_layout(
        list(new_terms),
        ignore_identity=bool(old_layout.ignore_identity),
        coefficient_tolerance=float(old_layout.coefficient_tolerance),
        sort_terms=(str(old_layout.term_order).strip().lower() == "sorted"),
    )
    old_blocks = tuple(old_layout.blocks)
    new_blocks = tuple(new_layout.blocks)
    if new_blocks[: len(old_blocks)] != old_blocks:
        raise ValueError("Drive-generator augmentation changed imported runtime layout prefix.")

    new_theta_runtime = np.concatenate(
        [
            np.asarray(loaded.replay_context.adapt_theta_runtime, dtype=float).reshape(-1),
            np.zeros(
                int(new_layout.runtime_parameter_count) - int(old_layout.runtime_parameter_count),
                dtype=float,
            ),
        ]
    )
    new_theta_logical = np.concatenate(
        [
            np.asarray(loaded.replay_context.adapt_theta_logical, dtype=float).reshape(-1),
            np.zeros(1, dtype=float),
        ]
    )
    pool_meta = dict(loaded.replay_context.pool_meta)
    pool_meta["drive_generator_mode"] = str(mode)
    pool_meta["family_pool_origin"] = "replay_terms_plus_drive_augmented"
    family_info = dict(loaded.replay_context.family_info)
    family_info["drive_generator_mode"] = str(mode)
    replay_context = replace(
        loaded.replay_context,
        family_info=family_info,
        family_pool=tuple(new_terms),
        pool_meta=pool_meta,
        replay_terms=tuple(new_terms),
        base_layout=new_layout,
        adapt_theta_runtime=np.asarray(new_theta_runtime, dtype=float),
        adapt_theta_logical=np.asarray(new_theta_logical, dtype=float),
        adapt_depth=int(len(new_terms)),
        family_terms_count=int(len(new_terms)),
    )
    executor = CompiledAnsatzExecutor(
        list(replay_context.replay_terms),
        coefficient_tolerance=float(new_layout.coefficient_tolerance),
        ignore_identity=bool(new_layout.ignore_identity),
        sort_terms=(str(new_layout.term_order).strip().lower() == "sorted"),
        parameterization_mode="per_pauli_term",
        parameterization_layout=new_layout,
    )
    psi_reconstructed = executor.prepare_state(
        np.asarray(replay_context.adapt_theta_runtime, dtype=float).reshape(-1),
        np.asarray(replay_context.psi_ref, dtype=complex).reshape(-1),
    )
    err = float(np.linalg.norm(np.asarray(psi_reconstructed, dtype=complex) - np.asarray(loaded.psi_initial, dtype=complex)))
    if err > 1.0e-10:
        raise ValueError(f"Drive-generator augmentation changed prepared state: reconstruction error {err:.3e}.")
    loader_summary = dict(loaded.loader_summary)
    loader_summary["drive_generator_mode"] = str(mode)
    loader_summary["family_pool_origin"] = "replay_terms_plus_drive_augmented"
    loader_summary["logical_operator_count"] = int(replay_context.base_layout.logical_parameter_count)
    loader_summary["runtime_parameter_count"] = int(replay_context.base_layout.runtime_parameter_count)
    loader_summary["augmentation_runtime_parameter_delta"] = int(
        replay_context.base_layout.runtime_parameter_count - old_layout.runtime_parameter_count
    )
    loader_summary["prepared_state_reconstruction_error"] = float(err)
    return replace(loaded, replay_context=replay_context, loader_summary=loader_summary)


def _build_driven_hamiltonian(
    *,
    h_poly_static: Any,
    hmat_static: np.ndarray,
    drive_coeff_provider_exyz: Any | None,
    physical_time: float,
    nq: int,
    geom_cfg: FixedManifoldMeasuredConfig,
    drive_drop_abs_tol: float = 1.0e-15,
) -> tuple[Any, np.ndarray, dict[str, complex]]:
    if drive_coeff_provider_exyz is None:
        return h_poly_static, np.asarray(hmat_static, dtype=complex), {}
    drive_coeff_map = {
        str(label): complex(coeff)
        for label, coeff in dict(drive_coeff_provider_exyz(float(physical_time))).items()
    }
    drive_poly = _poly_from_real_coeff_map(
        drive_coeff_map,
        nq=int(nq),
        drop_abs_tol=float(drive_drop_abs_tol),
        hermiticity_tol=float(geom_cfg.observable_hermiticity_tol),
        context="drive Hamiltonian",
    )
    drive_terms = list(drive_poly.return_polynomial())
    if not drive_terms:
        return h_poly_static, np.asarray(hmat_static, dtype=complex), {}
    h_poly_total = h_poly_static + drive_poly
    hmat_drive = np.asarray(hamiltonian_matrix(drive_poly), dtype=complex)
    hmat_total = np.asarray(np.asarray(hmat_static, dtype=complex) + hmat_drive, dtype=complex)
    kept_map = {
        str(term.pw2strng()): complex(term.p_coeff)
        for term in drive_terms
    }
    return h_poly_total, hmat_total, kept_map


def _apply_reference_step(psi: np.ndarray, hmat: np.ndarray, *, dt: float) -> np.ndarray:
    op = (-1.0j * float(dt)) * np.asarray(hmat, dtype=complex)
    if _csc_matrix is not None and _expm_multiply is not None:
        return np.asarray(_expm_multiply(_csc_matrix(op), np.asarray(psi, dtype=complex)), dtype=complex).reshape(-1)
    if _dense_expm is None:  # pragma: no cover
        raise RuntimeError("Drive reference evolution requires SciPy expm or expm_multiply.")
    return np.asarray(_dense_expm(op) @ np.asarray(psi, dtype=complex), dtype=complex).reshape(-1)


def _build_driven_reference_states(
    *,
    psi_initial: np.ndarray,
    times: Sequence[float],
    hmat_static: np.ndarray,
    h_poly_static: Any,
    drive_coeff_provider_exyz: Any,
    drive_cfg: FixedManifoldDriveConfig,
    geom_cfg: FixedManifoldMeasuredConfig,
) -> list[np.ndarray]:
    multiplier = int(drive_cfg.exact_steps_multiplier)
    if multiplier < 1:
        raise ValueError("--exact-steps-multiplier must be >= 1.")
    if len(times) <= 0:
        return []
    nq = int(round(np.log2(int(np.asarray(psi_initial).size))))
    sampling = str(drive_cfg.drive_time_sampling).strip().lower()
    if sampling not in {"midpoint", "left", "right"}:
        raise ValueError(f"Unsupported drive_time_sampling {drive_cfg.drive_time_sampling!r}.")
    psi = np.asarray(psi_initial, dtype=complex).reshape(-1)
    out = [np.array(psi, copy=True)]
    for idx in range(len(times) - 1):
        t_start = float(times[idx])
        t_stop = float(times[idx + 1])
        dt_interval = float(t_stop - t_start)
        dt_micro = float(dt_interval) / float(multiplier)
        for micro in range(multiplier):
            base = float(t_start) + float(micro) * float(dt_micro)
            if sampling == "midpoint":
                t_sample = float(drive_cfg.drive_t0) + float(base) + 0.5 * float(dt_micro)
            elif sampling == "left":
                t_sample = float(drive_cfg.drive_t0) + float(base)
            else:
                t_sample = float(drive_cfg.drive_t0) + float(base) + float(dt_micro)
            _h_poly_step, hmat_step, _drive_map = _build_driven_hamiltonian(
                h_poly_static=h_poly_static,
                hmat_static=hmat_static,
                drive_coeff_provider_exyz=drive_coeff_provider_exyz,
                physical_time=float(t_sample),
                nq=int(nq),
                geom_cfg=geom_cfg,
                drive_drop_abs_tol=1.0e-15,
            )
            psi = _apply_reference_step(psi, hmat_step, dt=float(dt_micro))
        out.append(np.array(psi, copy=True))
    return out


def _projection_sample_time(
    *,
    time_start: float,
    time_stop: float | None,
    drive_cfg: FixedManifoldDriveConfig,
) -> float:
    if not bool(drive_cfg.enable_drive):
        return float(time_start)
    sampling = str(drive_cfg.drive_time_sampling).strip().lower()
    if sampling not in {"midpoint", "left", "right"}:
        raise ValueError(f"Unsupported drive_time_sampling {drive_cfg.drive_time_sampling!r}.")
    if time_stop is None:
        sample_time = float(time_start)
    elif sampling == "midpoint":
        sample_time = 0.5 * (float(time_start) + float(time_stop))
    elif sampling == "left":
        sample_time = float(time_start)
    else:
        sample_time = float(time_stop)
    return float(sample_time) + float(drive_cfg.drive_t0)


def _compiled_executor(loaded: Any) -> CompiledAnsatzExecutor:
    layout = loaded.replay_context.base_layout
    return CompiledAnsatzExecutor(
        list(loaded.replay_context.replay_terms),
        coefficient_tolerance=float(layout.coefficient_tolerance),
        ignore_identity=bool(layout.ignore_identity),
        sort_terms=(str(layout.term_order).strip().lower() == "sorted"),
        parameterization_mode="per_pauli_term",
        parameterization_layout=layout,
    )


def _evaluate_observable(
    oracle: ExpectationOracle,
    plan: CheckpointObservablePlan,
    spec: ObservableSpec,
    *,
    counters: dict[str, int],
) -> float:
    if bool(spec.is_zero):
        counters["zero_observable_skips_total"] = int(counters.get("zero_observable_skips_total", 0) + 1)
        return 0.0
    coeffs = None
    if spec.sparse_op is not None and hasattr(spec.sparse_op, "coeffs"):
        coeffs = np.asarray(getattr(spec.sparse_op, "coeffs"), dtype=complex).reshape(-1)
        if coeffs.size == 0 or float(np.max(np.abs(coeffs))) <= float(_SPARSE_EMPTY_OBSERVABLE_GUARD):
            counters["tiny_observable_skips_total"] = int(counters.get("tiny_observable_skips_total", 0) + 1)
            return 0.0
    try:
        est = oracle.evaluate(plan.circuit, spec.sparse_op)
    except RuntimeError as exc:
        msg = str(exc)
        if "Empty observable was detected" in msg and coeffs is not None:
            if coeffs.size == 0 or float(np.max(np.abs(coeffs))) <= float(10.0 * _SPARSE_EMPTY_OBSERVABLE_GUARD):
                counters["tiny_observable_skips_total"] = int(counters.get("tiny_observable_skips_total", 0) + 1)
                return 0.0
        raise
    counters["oracle_evaluations_total"] = int(counters.get("oracle_evaluations_total", 0) + 1)
    return float(est.mean)


"""
G_ii = c_i^2 (1 - a_i^2),  G_ij = c_i c_j (s_ij - a_i a_j),  f_i = c_i (h_i - E a_i)
"""
def assemble_measured_geometry(
    *,
    plan: CheckpointObservablePlan,
    energy: float,
    h2: float,
    generator_means: Sequence[float],
    pair_expectations: Mapping[tuple[int, int], float],
    force_expectations: Sequence[float],
    geom_cfg: FixedManifoldMeasuredConfig,
) -> dict[str, Any]:
    m = int(len(plan.runtime_rotations))
    a = np.asarray(generator_means, dtype=float).reshape(-1)
    h = np.asarray(force_expectations, dtype=float).reshape(-1)
    if int(a.size) != int(m):
        raise ValueError(f"generator_means length mismatch: {a.size} vs {m}.")
    if int(h.size) != int(m):
        raise ValueError(f"force_expectations length mismatch: {h.size} vs {m}.")

    coeffs = np.asarray([float(rot.coeff_real) for rot in plan.runtime_rotations], dtype=float)
    G = np.zeros((m, m), dtype=float)
    for i in range(m):
        G[i, i] = float((coeffs[i] ** 2) * max(0.0, 1.0 - float(a[i] * a[i])))
        for j in range(i + 1, m):
            sij = float(pair_expectations[(int(i), int(j))])
            gij = float(coeffs[i] * coeffs[j] * (sij - float(a[i] * a[j])))
            G[i, j] = gij
            G[j, i] = gij
    asym = float(np.max(np.abs(G - G.T))) if int(m) > 0 else 0.0
    if asym > float(geom_cfg.g_symmetrize_tol):
        G = 0.5 * (G + G.T)

    f = np.asarray(coeffs * (h - float(energy) * a), dtype=float).reshape(-1)
    variance = float(max(float(geom_cfg.variance_floor), float(h2) - float(energy * energy)))
    G_pinv = np.linalg.pinv(G, rcond=float(geom_cfg.pinv_rcond)) if int(m) > 0 else np.zeros((0, 0), dtype=float)
    K = np.asarray(
        G + float(geom_cfg.regularization_lambda) * np.eye(int(m), dtype=float),
        dtype=float,
    )
    K_pinv = np.linalg.pinv(K, rcond=float(geom_cfg.pinv_rcond)) if int(m) > 0 else np.zeros((0, 0), dtype=float)
    theta_dot_proj = np.asarray(G_pinv @ f, dtype=float).reshape(-1) if int(m) > 0 else np.zeros(0, dtype=float)
    theta_dot_step = np.asarray(K_pinv @ f, dtype=float).reshape(-1) if int(m) > 0 else np.zeros(0, dtype=float)
    epsilon_proj_sq = float(max(0.0, float(variance) - float(f @ theta_dot_proj))) if int(m) > 0 else float(variance)
    epsilon_step_sq = (
        float(max(0.0, float(variance) - 2.0 * float(f @ theta_dot_step) + float(theta_dot_step @ G @ theta_dot_step)))
        if int(m) > 0
        else float(variance)
    )
    rho_miss = float(epsilon_proj_sq / max(float(variance), 1.0e-14))
    rank = int(np.linalg.matrix_rank(K, tol=float(geom_cfg.pinv_rcond))) if int(m) > 0 else 0
    cond = float(np.linalg.cond(K)) if int(m) > 0 else 1.0

    return {
        "G": np.asarray(G, dtype=float),
        "f": np.asarray(f, dtype=float),
        "K": np.asarray(K, dtype=float),
        "theta_dot_proj": np.asarray(theta_dot_proj, dtype=float),
        "theta_dot_step": np.asarray(theta_dot_step, dtype=float),
        "energy": float(energy),
        "h2": float(h2),
        "variance": float(variance),
        "epsilon_proj_sq": float(epsilon_proj_sq),
        "epsilon_step_sq": float(epsilon_step_sq),
        "rho_miss": float(rho_miss),
        "matrix_rank": int(rank),
        "condition_number": float(cond),
        "symmetry_violation_max": float(asym),
        "generator_means": [float(x) for x in a.tolist()],
        "force_expectations": [float(x) for x in h.tolist()],
    }


def _trajectory_metric(
    trajectory: Sequence[Mapping[str, Any]],
    *path: str,
) -> list[float]:
    values: list[float] = []
    for row in trajectory:
        node: Any = row
        for key in path:
            if not isinstance(node, Mapping):
                node = None
                break
            node = node.get(key, None)
        if node is None:
            continue
        values.append(float(node))
    return values


def run_fixed_manifold_measured(
    spec: FixedManifoldRunSpec,
    *,
    tag: str,
    output_json: Path,
    t_final: float,
    num_times: int,
    oracle_cfg: OracleConfig,
    geom_cfg: FixedManifoldMeasuredConfig,
    drive_cfg: FixedManifoldDriveConfig | None = None,
    aug_cfg: FixedManifoldAugmentationConfig | None = None,
) -> dict[str, Any]:
    loader_mode = str(spec.loader_mode).strip().lower()
    if loader_mode not in {"fixed_scaffold", "replay_family"}:
        raise ValueError(
            "Measured fixed-manifold v1 supports only loader_mode in "
            "{'fixed_scaffold','replay_family'}."
        )
    drive_cfg = FixedManifoldDriveConfig() if drive_cfg is None else drive_cfg
    aug_cfg = FixedManifoldAugmentationConfig() if aug_cfg is None else aug_cfg
    if str(oracle_cfg.noise_mode).strip().lower() != "ideal":
        raise ValueError("Measured fixed-manifold v1 currently supports only noise_mode='ideal'.")
    if int(oracle_cfg.oracle_repeats) != 1:
        raise ValueError("Measured fixed-manifold v1 currently requires oracle_repeats=1.")
    if str(oracle_cfg.oracle_aggregate).strip().lower() != "mean":
        raise ValueError("Measured fixed-manifold v1 requires oracle_aggregate='mean'.")

    loaded = load_run_context(spec, tag=tag)
    if not bool(loaded.loader_summary.get("fixed_manifold_locked", False)):
        raise ValueError("Measured fixed-manifold v1 requires fixed_manifold_locked=true.")
    loaded = _augment_loaded_context_with_drive_generator(loaded, drive_cfg=drive_cfg, aug_cfg=aug_cfg)
    reference_summary = _validate_basis_reference_state(loaded.payload)
    hmat = np.asarray(hamiltonian_matrix(loaded.replay_context.h_poly), dtype=complex)
    drive_profile: dict[str, Any] | None = None
    drive_coeff_provider_exyz = None
    if bool(drive_cfg.enable_drive):
        custom_weights = None
        if str(drive_cfg.drive_pattern).strip().lower() == "custom":
            custom_weights = _parse_drive_custom_weights(drive_cfg.drive_custom_s)
            if custom_weights is None:
                raise ValueError("--drive-custom-s is required when --drive-pattern custom.")
        drive = build_gaussian_sinusoid_density_drive(
            n_sites=int(loaded.cfg.L),
            nq_total=int(reference_summary["nq_total"]),
            indexing=str(loaded.cfg.ordering),
            A=float(drive_cfg.drive_A),
            omega=float(drive_cfg.drive_omega),
            tbar=float(drive_cfg.drive_tbar),
            phi=float(drive_cfg.drive_phi),
            pattern_mode=str(drive_cfg.drive_pattern),
            custom_weights=custom_weights,
            include_identity=bool(drive_cfg.drive_include_identity),
            coeff_tol=0.0,
        )
        drive_coeff_provider_exyz = drive.coeff_map_exyz
        drive_profile = {
            "A": float(drive_cfg.drive_A),
            "omega": float(drive_cfg.drive_omega),
            "tbar": float(drive_cfg.drive_tbar),
            "phi": float(drive_cfg.drive_phi),
            "pattern": str(drive_cfg.drive_pattern),
            "custom_weights": custom_weights,
            "include_identity": bool(drive_cfg.drive_include_identity),
            "time_sampling": str(drive_cfg.drive_time_sampling),
            "t0": float(drive_cfg.drive_t0),
        }
    else:
        evals, evecs = np.linalg.eigh(hmat)
        coeffs0 = np.asarray(np.conjugate(evecs).T @ np.asarray(loaded.psi_initial, dtype=complex), dtype=complex)
    executor = _compiled_executor(loaded)

    times = np.linspace(0.0, float(t_final), int(num_times), dtype=float)
    if bool(drive_cfg.enable_drive):
        reference_states = _build_driven_reference_states(
            psi_initial=np.asarray(loaded.psi_initial, dtype=complex),
            times=times,
            hmat_static=hmat,
            h_poly_static=loaded.replay_context.h_poly,
            drive_coeff_provider_exyz=drive_coeff_provider_exyz,
            drive_cfg=drive_cfg,
            geom_cfg=geom_cfg,
        )
    else:
        reference_states = []
    theta_current = np.asarray(loaded.replay_context.adapt_theta_runtime, dtype=float).reshape(-1)
    trajectory: list[dict[str, Any]] = []
    counters: dict[str, int] = {
        "oracle_evaluations_total": 0,
        "zero_observable_skips_total": 0,
        "tiny_observable_skips_total": 0,
    }
    max_observable_terms = 0
    max_generator_terms = 0
    max_pair_terms = 0
    max_force_terms = 0
    oracle = ExpectationOracle(oracle_cfg)
    try:
        for checkpoint_index, time_value in enumerate(times):
            time_stop = (
                None
                if int(checkpoint_index) + 1 >= int(len(times))
                else float(times[int(checkpoint_index) + 1])
            )
            physical_time = _projection_sample_time(
                time_start=float(time_value),
                time_stop=time_stop,
                drive_cfg=drive_cfg,
            )
            h_poly_step, hmat_step, drive_coeff_map = _build_driven_hamiltonian(
                h_poly_static=loaded.replay_context.h_poly,
                hmat_static=hmat,
                drive_coeff_provider_exyz=drive_coeff_provider_exyz,
                physical_time=float(physical_time),
                nq=int(reference_summary["nq_total"]),
                geom_cfg=geom_cfg,
                drive_drop_abs_tol=1.0e-15,
            )
            plan = build_checkpoint_observable_plan(
                loaded.replay_context,
                theta_current,
                h_poly=h_poly_step,
                drop_abs_tol=float(geom_cfg.observable_drop_abs_tol),
                hermiticity_tol=float(geom_cfg.observable_hermiticity_tol),
                max_observable_terms=int(geom_cfg.observable_max_terms),
            )
            max_observable_terms = max(int(max_observable_terms), int(plan.stats["max_observable_terms_any"]))
            max_generator_terms = max(int(max_generator_terms), int(plan.stats["max_generator_terms"]))
            max_pair_terms = max(int(max_pair_terms), int(plan.stats["max_pair_terms"]))
            max_force_terms = max(int(max_force_terms), int(plan.stats["max_force_terms"]))

            energy = _evaluate_observable(oracle, plan, plan.energy, counters=counters)
            h2 = _evaluate_observable(oracle, plan, plan.variance_h2, counters=counters)
            generator_means = [
                _evaluate_observable(oracle, plan, spec_i, counters=counters)
                for spec_i in plan.generator_means
            ]
            pair_expectations = {
                tuple(pair): _evaluate_observable(oracle, plan, pair_spec, counters=counters)
                for pair, pair_spec in plan.pair_anticommutators.items()
            }
            force_expectations = [
                _evaluate_observable(oracle, plan, spec_i, counters=counters)
                for spec_i in plan.force_anticommutators
            ]
            geom = assemble_measured_geometry(
                plan=plan,
                energy=float(energy),
                h2=float(h2),
                generator_means=generator_means,
                pair_expectations=pair_expectations,
                force_expectations=force_expectations,
                geom_cfg=geom_cfg,
            )

            psi_ansatz = executor.prepare_state(theta_current, loaded.replay_context.psi_ref)
            if bool(drive_cfg.enable_drive):
                psi_exact = np.asarray(reference_states[int(checkpoint_index)], dtype=complex).reshape(-1)
            else:
                psi_exact = np.asarray(
                    evecs @ (np.exp(-1.0j * np.asarray(evals, dtype=float) * float(time_value)) * coeffs0),
                    dtype=complex,
                ).reshape(-1)
            energy_ansatz_exact = float(np.real(np.vdot(psi_ansatz, hmat_step @ psi_ansatz)))
            energy_reference_exact = float(np.real(np.vdot(psi_exact, hmat_step @ psi_exact)))
            fidelity_exact_audit = float(abs(np.vdot(psi_exact, psi_ansatz)) ** 2)
            abs_energy_total_error_exact_audit = float(abs(energy_ansatz_exact - energy_reference_exact))

            trajectory.append(
                {
                    "checkpoint_index": int(checkpoint_index),
                    "time": float(time_value),
                    "physical_time": float(physical_time),
                    "geometry": {
                        "energy": float(geom["energy"]),
                        "h2": float(geom["h2"]),
                        "variance": float(geom["variance"]),
                        "epsilon_proj_sq": float(geom["epsilon_proj_sq"]),
                        "epsilon_step_sq": float(geom["epsilon_step_sq"]),
                        "rho_miss": float(geom["rho_miss"]),
                        "matrix_rank": int(geom["matrix_rank"]),
                        "condition_number": float(geom["condition_number"]),
                        "symmetry_violation_max": float(geom["symmetry_violation_max"]),
                        "theta_dot_l2": float(np.linalg.norm(np.asarray(geom["theta_dot_step"], dtype=float))),
                        "generator_means": [float(x) for x in geom["generator_means"]],
                        "force_expectations": [float(x) for x in geom["force_expectations"]],
                        "drive_term_count": int(len(drive_coeff_map)),
                    },
                    "audit": {
                        "fidelity_exact_audit": float(fidelity_exact_audit),
                        "energy_ansatz_exact_audit": float(energy_ansatz_exact),
                        "energy_reference_exact_audit": float(energy_reference_exact),
                        "abs_energy_total_error_exact_audit": float(abs_energy_total_error_exact_audit),
                    },
                    "observable_plan": dict(plan.stats),
                }
            )

            if int(checkpoint_index) + 1 < int(len(times)):
                dt = float(times[int(checkpoint_index) + 1] - float(time_value))
                theta_current = np.asarray(
                    theta_current + float(dt) * np.asarray(geom["theta_dot_step"], dtype=float),
                    dtype=float,
                ).reshape(-1)
    finally:
        oracle.close()

    final_row = trajectory[-1] if trajectory else {}
    fidelity_values = _trajectory_metric(trajectory, "audit", "fidelity_exact_audit")
    energy_error_values = _trajectory_metric(trajectory, "audit", "abs_energy_total_error_exact_audit")
    rho_values = _trajectory_metric(trajectory, "geometry", "rho_miss")
    cond_values = _trajectory_metric(trajectory, "geometry", "condition_number")
    theta_dot_values = _trajectory_metric(trajectory, "geometry", "theta_dot_l2")
    effective_pool_kind = str(loaded.loader_summary.get("family_pool_origin", "replay_terms_only"))
    drive_generator_mode = str(loaded.loader_summary.get("drive_generator_mode", "none"))
    run_name = str(spec.name) if drive_generator_mode in {"", "none"} else f"{spec.name}_{drive_generator_mode}"

    summary = {
        "status": "completed",
        "noise_mode": str(oracle_cfg.noise_mode),
        "oracle_evaluations_total": int(counters["oracle_evaluations_total"]),
        "zero_observable_skips_total": int(counters["zero_observable_skips_total"]),
        "tiny_observable_skips_total": int(counters["tiny_observable_skips_total"]),
        "final_fidelity_exact_audit": float(final_row.get("audit", {}).get("fidelity_exact_audit", float("nan"))),
        "min_fidelity_exact_audit": float(min(fidelity_values)) if fidelity_values else float("nan"),
        "final_abs_energy_total_error_exact_audit": float(
            final_row.get("audit", {}).get("abs_energy_total_error_exact_audit", float("nan"))
        ),
        "max_abs_energy_total_error_exact_audit": float(max(energy_error_values)) if energy_error_values else float("nan"),
        "final_rho_miss": float(final_row.get("geometry", {}).get("rho_miss", float("nan"))),
        "max_rho_miss": float(max(rho_values)) if rho_values else float("nan"),
        "final_condition_number": float(final_row.get("geometry", {}).get("condition_number", float("nan"))),
        "max_condition_number": float(max(cond_values)) if cond_values else float("nan"),
        "final_theta_dot_l2": float(final_row.get("geometry", {}).get("theta_dot_l2", float("nan"))),
        "max_theta_dot_l2": float(max(theta_dot_values)) if theta_dot_values else float("nan"),
        "max_observable_terms": int(max_observable_terms),
        "max_conjugated_generator_terms": int(max_generator_terms),
        "max_pair_observable_terms": int(max_pair_terms),
        "max_force_observable_terms": int(max_force_terms),
        "runtime_parameter_count": int(loaded.replay_context.base_layout.runtime_parameter_count),
        "logical_block_count": int(loaded.replay_context.base_layout.logical_parameter_count),
    }
    payload = {
        "generated_utc": _now_utc(),
        "pipeline": "hh_fixed_manifold_measured_mclachlan_v1",
        "run_name": str(run_name),
        "input_artifact_json": str(spec.artifact_json),
        "manifest": {
            "model_family": "Hubbard-Holstein",
            "geometry_backend": "oracle",
            "reference_audit_backend": (
                "dense_piecewise_constant_reference"
                if bool(drive_cfg.enable_drive)
                else "exact_dense_static"
            ),
            "reference_audit_method": (
                str(reference_method_name(str(drive_cfg.drive_time_sampling)))
                if bool(drive_cfg.enable_drive)
                else "eigendecomposition"
            ),
            "structure_policy": "fixed_manifold_locked_pool",
            "effective_pool_kind": str(effective_pool_kind),
            "drive_enabled": bool(drive_cfg.enable_drive),
            "noise_mode": str(oracle_cfg.noise_mode),
            "fixed_scaffold_kind": loaded.loader_summary.get("fixed_scaffold_kind", None),
            "drive_generator_mode": str(drive_generator_mode),
        },
        "reference_state": dict(reference_summary),
        "loader": dict(loaded.loader_summary),
        "augmentation_config": asdict(aug_cfg),
        "oracle_config": {
            "noise_mode": str(oracle_cfg.noise_mode),
            "shots": int(oracle_cfg.shots),
            "seed": int(oracle_cfg.seed),
            "oracle_repeats": int(oracle_cfg.oracle_repeats),
            "oracle_aggregate": str(oracle_cfg.oracle_aggregate),
        },
        "geometry_config": asdict(geom_cfg),
        "run_config": {
            "t_final": float(t_final),
            "num_times": int(num_times),
        },
        "drive_profile": (dict(drive_profile) if drive_profile is not None else None),
        "reference_config": {
            "method": (
                str(reference_method_name(str(drive_cfg.drive_time_sampling)))
                if bool(drive_cfg.enable_drive)
                else "eigendecomposition"
            ),
            "reference_steps_multiplier": (
                int(drive_cfg.exact_steps_multiplier) if bool(drive_cfg.enable_drive) else 1
            ),
            "reference_steps": (
                int(max(0, int(num_times) - 1) * int(drive_cfg.exact_steps_multiplier))
                if bool(drive_cfg.enable_drive)
                else 0
            ),
            "geometry_sample_time_policy": (
                f"interval_{str(drive_cfg.drive_time_sampling).strip().lower()}_plus_t0_with_final_endpoint_fallback"
                if bool(drive_cfg.enable_drive)
                else "checkpoint_time"
            ),
        },
        "projection_config": {
            "integrator": "explicit_euler",
            "time_sampling": (
                str(drive_cfg.drive_time_sampling) if bool(drive_cfg.enable_drive) else "left"
            ),
            "geometry_sample_time_policy": (
                f"interval_{str(drive_cfg.drive_time_sampling).strip().lower()}_plus_t0_with_final_endpoint_fallback"
                if bool(drive_cfg.enable_drive)
                else "checkpoint_time"
            ),
        },
        "summary": dict(summary),
        "trajectory": trajectory,
    }
    _write_json(Path(output_json), payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measured/oracle fixed-manifold McLachlan for fixed HH manifolds.",
    )
    parser.add_argument("--tag", type=str, default=_default_tag())
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument(
        "--manifold",
        type=str,
        choices=["locked_7term", "pareto_lean_l2"],
        default="locked_7term",
        help="Fixed manifold to load.",
    )
    parser.add_argument(
        "--locked-artifact-json",
        type=str,
        default=str(DEFAULT_LOCKED_7TERM_ARTIFACT),
    )
    parser.add_argument(
        "--pareto-artifact-json",
        type=str,
        default=str(DEFAULT_PARETO_ARTIFACT),
    )
    parser.add_argument("--t-final", type=float, default=10.0)
    parser.add_argument("--num-times", type=int, default=135)
    parser.add_argument("--noise-mode", type=str, default="ideal", choices=["ideal"])
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--oracle-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
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
    parser.add_argument(
        "--augment-drive-generator-mode",
        choices=["none", "aligned_density"],
        default="none",
    )
    parser.add_argument("--regularization-lambda", type=float, default=1.0e-8)
    parser.add_argument("--pinv-rcond", type=float, default=1.0e-10)
    parser.add_argument("--observable-drop-abs-tol", type=float, default=1.0e-12)
    parser.add_argument("--observable-hermiticity-tol", type=float, default=1.0e-10)
    parser.add_argument("--observable-max-terms", type=int, default=512)
    parser.add_argument("--variance-floor", type=float, default=0.0)
    parser.add_argument("--g-symmetrize-tol", type=float, default=1.0e-12)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if str(args.manifold) == "pareto_lean_l2":
        spec = FixedManifoldRunSpec(
            name="pareto_lean_l2",
            artifact_json=Path(args.pareto_artifact_json),
            loader_mode="replay_family",
            generator_family="match_adapt",
            fallback_family="full_meta",
        )
    else:
        spec = FixedManifoldRunSpec(
            name="locked_7term",
            artifact_json=Path(args.locked_artifact_json),
            loader_mode="fixed_scaffold",
            generator_family="fixed_scaffold_locked",
            fallback_family="full_meta",
        )
    output_json = (
        Path(args.output_json)
        if args.output_json is not None
        else Path("artifacts/agent_runs") / str(args.tag) / f"{spec.name}_measured.json"
    )
    oracle_cfg = OracleConfig(
        noise_mode=str(args.noise_mode),
        shots=int(args.shots),
        seed=int(args.seed),
        oracle_repeats=int(args.oracle_repeats),
        oracle_aggregate="mean",
    )
    geom_cfg = FixedManifoldMeasuredConfig(
        regularization_lambda=float(args.regularization_lambda),
        pinv_rcond=float(args.pinv_rcond),
        observable_drop_abs_tol=float(args.observable_drop_abs_tol),
        observable_hermiticity_tol=float(args.observable_hermiticity_tol),
        observable_max_terms=int(args.observable_max_terms),
        variance_floor=float(args.variance_floor),
        g_symmetrize_tol=float(args.g_symmetrize_tol),
    )
    drive_cfg = FixedManifoldDriveConfig(
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
    aug_cfg = FixedManifoldAugmentationConfig(
        drive_generator_mode=str(args.augment_drive_generator_mode),
    )
    run_fixed_manifold_measured(
        spec,
        tag=str(args.tag),
        output_json=output_json,
        t_final=float(args.t_final),
        num_times=int(args.num_times),
        oracle_cfg=oracle_cfg,
        geom_cfg=geom_cfg,
        drive_cfg=drive_cfg,
        aug_cfg=aug_cfg,
    )
    return 0


__all__ = [
    "FixedManifoldAugmentationConfig",
    "FixedManifoldDriveConfig",
    "FixedManifoldMeasuredConfig",
    "assemble_measured_geometry",
    "parse_args",
    "run_fixed_manifold_measured",
    "_augment_loaded_context_with_drive_generator",
]


if __name__ == "__main__":
    raise SystemExit(main())
