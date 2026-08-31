"""CFQM macro-step propagation utilities.

This module implements one CFQM macro-step for statevector propagation:

    psi(t + dt) = exp(-i dt * Omega_1) ... exp(-i dt * Omega_s) psi(t)

with stage operators Omega_k assembled from static and sampled drive
coefficient maps.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class CFQMConfig:
    """Configuration for CFQM stage propagation backends."""

    backend: str = "expm_multiply_sparse"
    coeff_drop_abs_tol: float = 0.0
    normalize: bool = False
    sparse_min_dim: int = 64
    norm_drift_logger: Callable[..., None] | None = None
    emit_inner_order_warning: bool = True


def _cfg_get(config: object, key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _build_dense_stage_matrix_via_repo_utility(
    stage_coeff_map: Mapping[str, complex | float],
    ordered_labels: list[str],
) -> np.ndarray:
    """Build dense stage matrix using existing repo Pauli-sum utility."""
    from src.quantum.pauli_polynomial_class import PauliPolynomial
    from src.quantum.qubitization_module import PauliTerm
    from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix

    if not ordered_labels:
        raise ValueError("ordered_labels must be non-empty.")
    nq = len(ordered_labels[0])
    for label in ordered_labels:
        if len(label) != nq:
            raise ValueError("All ordered_labels must have equal length.")
    ordered_set = set(ordered_labels)
    for label in stage_coeff_map:
        if label not in ordered_set:
            raise ValueError(f"Stage label {label!r} absent from ordered_labels.")

    pol = PauliPolynomial("JW")
    any_nonzero = False
    for label in ordered_labels:
        coeff = stage_coeff_map.get(label)
        if coeff is None:
            continue
        coeff_c = complex(coeff)
        if coeff_c == 0.0:
            continue
        pol.add_term(PauliTerm(nq, ps=label, pc=coeff_c))
        any_nonzero = True

    if not any_nonzero:
        dim = 1 << nq
        return np.zeros((dim, dim), dtype=complex)
    return hamiltonian_matrix(pol, tol=0.0)


def _build_sparse_stage_matrix_via_compiled_actions(
    stage_coeff_map: Mapping[str, complex | float],
    ordered_labels: list[str],
):
    """Build sparse stage matrix without dense materialization.

    Reuses the existing compiled Pauli-action utility from the hardcoded
    pipeline to preserve Pauli conventions exactly.
    """
    from scipy.sparse import csc_matrix, coo_matrix
    from src.quantum.pauli_actions import compile_pauli_action_exyz

    if not ordered_labels:
        raise ValueError("ordered_labels must be non-empty.")
    nq = len(ordered_labels[0])
    for label in ordered_labels:
        if len(label) != nq:
            raise ValueError("All ordered_labels must have equal length.")
    ordered_set = set(ordered_labels)
    for label in stage_coeff_map:
        if label not in ordered_set:
            raise ValueError(f"Stage label {label!r} absent from ordered_labels.")

    dim = 1 << nq
    row_base = np.arange(dim, dtype=np.int64)
    row_chunks: list[np.ndarray] = []
    col_chunks: list[np.ndarray] = []
    data_chunks: list[np.ndarray] = []

    for label in ordered_labels:
        coeff = stage_coeff_map.get(label)
        if coeff is None:
            continue
        coeff_c = complex(coeff)
        if coeff_c == 0.0:
            continue
        action = compile_pauli_action_exyz(label, nq)
        row_chunks.append(row_base)
        col_chunks.append(np.asarray(action.perm, dtype=np.int64))
        data_chunks.append(coeff_c * np.asarray(action.phase, dtype=complex))

    if not data_chunks:
        return csc_matrix((dim, dim), dtype=complex)

    rows = np.concatenate(row_chunks)
    cols = np.concatenate(col_chunks)
    data = np.concatenate(data_chunks)
    h_csc = coo_matrix((data, (rows, cols)), shape=(dim, dim)).tocsc()
    h_csc.sum_duplicates()
    h_csc.eliminate_zeros()
    return h_csc


def _apply_stage_pauli_suzuki2(
    psi: np.ndarray,
    stage_coeff_map: Mapping[str, complex | float],
    dt: float,
    ordered_labels: list[str],
) -> np.ndarray:
    """Apply symmetric Suzuki-2 over Pauli terms in deterministic order."""
    from src.quantum.pauli_actions import apply_exp_term, compile_pauli_action_exyz

    nq = len(ordered_labels[0])
    compiled = {label: compile_pauli_action_exyz(label, nq) for label in ordered_labels}
    half = 0.5 * float(dt)
    out = np.array(psi, copy=True)

    for label in ordered_labels:
        alpha = complex(stage_coeff_map.get(label, 0.0 + 0.0j))
        if alpha == 0.0:
            continue
        out = apply_exp_term(out, compiled[label], alpha, half)
    for label in reversed(ordered_labels):
        alpha = complex(stage_coeff_map.get(label, 0.0 + 0.0j))
        if alpha == 0.0:
            continue
        out = apply_exp_term(out, compiled[label], alpha, half)
    return out


def apply_stage_exponential(
    psi: np.ndarray,
    stage_coeff_map: dict[str, complex | float],
    dt: float,
    ordered_labels: list[str],
    backend_config: object,
) -> np.ndarray:
    """Apply exp(-i * dt * H_stage) to a statevector."""
    if abs(float(dt)) <= 1e-15 or not stage_coeff_map:
        return np.array(psi, copy=True)

    backend_key = str(_cfg_get(backend_config, "backend", "expm_multiply_sparse")).strip().lower()
    scheme_name = str(_cfg_get(backend_config, "scheme_name", "")).strip().lower()

    if backend_key in {"dense", "dense_expm"}:
        from scipy.linalg import expm

        h_stage = _build_dense_stage_matrix_via_repo_utility(stage_coeff_map, ordered_labels)
        return np.asarray(expm((-1j * float(dt)) * h_stage) @ psi, dtype=complex)

    if backend_key in {"expm_multiply_sparse", "sparse_expm_multiply", "sparse"}:
        from scipy.sparse.linalg import expm_multiply

        h_stage = _build_sparse_stage_matrix_via_compiled_actions(stage_coeff_map, ordered_labels)
        return np.asarray(expm_multiply((-1j * float(dt)) * h_stage, psi), dtype=complex)

    if backend_key == "pauli_suzuki2":
        emit_warning = bool(_cfg_get(backend_config, "emit_inner_order_warning", True))
        if emit_warning and (scheme_name in {"cf4:2", "cf6:5opt"} or scheme_name.startswith("cfqm")):
            warnings.warn(
                "Inner Suzuki-2 makes overall method 2nd order; use expm_multiply_sparse/dense_expm for true CFQM order.",
                RuntimeWarning,
                stacklevel=2,
            )
        return np.asarray(
            _apply_stage_pauli_suzuki2(
                psi=psi,
                stage_coeff_map=stage_coeff_map,
                dt=float(dt),
                ordered_labels=ordered_labels,
            ),
            dtype=complex,
        )

    if backend_key == "auto":
        from scipy.sparse.linalg import expm_multiply

        sparse_min_dim = int(_cfg_get(backend_config, "sparse_min_dim", 64))
        if int(psi.size) < sparse_min_dim:
            from scipy.linalg import expm

            h_stage_dense = _build_dense_stage_matrix_via_repo_utility(stage_coeff_map, ordered_labels)
            return np.asarray(expm((-1j * float(dt)) * h_stage_dense) @ psi, dtype=complex)
        h_sparse = _build_sparse_stage_matrix_via_compiled_actions(stage_coeff_map, ordered_labels)
        return np.asarray(expm_multiply((-1j * float(dt)) * h_sparse, psi), dtype=complex)

    raise ValueError(
        f"Unsupported backend {backend_key!r}. "
        "Use one of {'dense_expm','expm_multiply_sparse','pauli_suzuki2','auto'}."
    )


def cfqm_step(
    psi: np.ndarray,
    t_abs: float,
    dt: float,
    static_coeff_map: dict[str, complex | float],
    drive_coeff_provider: Callable[[float], Mapping[str, complex | float]] | None,
    ordered_labels: list[str],
    scheme: dict,
    config: object,
) -> np.ndarray:
    """Advance one CFQM macro-step.

    Drive-map labels not present in ``ordered_labels`` are ignored so that
    stage assembly remains deterministic and never introduces out-of-order
    terms.
    """
    if not ordered_labels:
        raise ValueError("ordered_labels must be non-empty.")
    dt_f = float(dt)
    if not np.isfinite(dt_f) or dt_f <= 0.0:
        raise ValueError(f"cfqm_step requires dt > 0 and finite; got dt={dt_f}.")

    dim = int(psi.size)
    nq = int(round(math.log2(dim)))
    if (1 << nq) != dim:
        raise ValueError("psi size must be a power of two.")
    for label in ordered_labels:
        if len(label) != nq:
            raise ValueError("All labels must match statevector qubit count.")

    c_nodes = [float(x) for x in scheme["c"]]
    a_rows = [[float(v) for v in row] for row in scheme["a"]]
    s_static = [float(v) for v in scheme["s_static"]]

    m_nodes = len(c_nodes)
    s_stages = len(a_rows)
    if len(s_static) != s_stages:
        raise ValueError("scheme.s_static length must match number of stage rows.")
    for k, row in enumerate(a_rows):
        if len(row) != m_nodes:
            raise ValueError(f"scheme.a row {k} length mismatch: expected {m_nodes}.")

    ordered_set = set(ordered_labels)
    static_map = {str(lbl): complex(coeff) for lbl, coeff in static_coeff_map.items()}
    for label in static_map:
        if label not in ordered_set:
            raise ValueError(f"Static label {label!r} is absent from ordered_labels.")

    unknown_label_policy = str(_cfg_get(config, "unknown_label_policy", "warn_ignore")).strip().lower()
    if unknown_label_policy not in {"warn_ignore", "ignore", "strict"}:
        raise ValueError(
            "unknown_label_policy must be one of {'warn_ignore','ignore','strict'}."
        )
    unknown_label_warn_abs_tol = max(
        0.0,
        float(_cfg_get(config, "unknown_label_warn_abs_tol", 1e-14)),
    )
    warned_labels_obj = _cfg_get(config, "unknown_label_warned_labels", None)
    if warned_labels_obj is None:
        warned_labels: set[str] = set()
    else:
        if not hasattr(warned_labels_obj, "__contains__") or not hasattr(warned_labels_obj, "add"):
            raise ValueError("unknown_label_warned_labels must support membership and add().")
        warned_labels = warned_labels_obj

    drive_maps: list[dict[str, complex]] = []
    for c_j in c_nodes:
        t_node = float(t_abs) + float(c_j) * dt_f
        if drive_coeff_provider is None:
            raw_map: Mapping[str, complex | float] = {}
        else:
            raw = drive_coeff_provider(float(t_node))
            raw_map = {} if raw is None else raw

        node_map: dict[str, complex] = {}
        for label, coeff in raw_map.items():
            lbl = str(label)
            coeff_c = complex(coeff)
            if not np.isfinite(coeff_c.real) or not np.isfinite(coeff_c.imag):
                raise ValueError(
                    "Non-finite drive coefficient detected: "
                    f"label={lbl!r}, time={float(t_node):.16g}, coeff={coeff_c!r}."
                )
            if lbl not in ordered_set:
                # Do not introduce labels outside deterministic ordered_labels.
                mag = abs(coeff_c)
                if mag <= unknown_label_warn_abs_tol:
                    continue
                if unknown_label_policy == "strict":
                    raise ValueError(
                        "Unknown drive label absent from ordered_labels: "
                        f"label={lbl!r}, time={float(t_node):.16g}, coeff={coeff_c!r}."
                    )
                if unknown_label_policy == "warn_ignore" and lbl not in warned_labels:
                    warnings.warn(
                        "Ignoring unknown drive label absent from ordered_labels: "
                        f"label={lbl!r}, time={float(t_node):.16g}, |coeff|={mag:.3e}.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    warned_labels.add(lbl)
                continue
            node_map[lbl] = coeff_c
        drive_maps.append(node_map)

    coeff_drop_tol = float(_cfg_get(config, "coeff_drop_abs_tol", 0.0))
    stage_maps: list[dict[str, complex]] = []

    for k in range(s_stages):
        stage_map: dict[str, complex] = {}

        w_static = float(s_static[k])
        for label in ordered_labels:
            coeff0 = static_map.get(label)
            if coeff0 is None:
                continue
            scaled = complex(w_static) * coeff0
            if scaled != 0.0:
                stage_map[label] = scaled

        for j in range(m_nodes):
            w = float(a_rows[k][j])
            if w == 0.0:
                continue
            for label, coeff_drive in drive_maps[j].items():
                incr = complex(w) * complex(coeff_drive)
                # A=0 invariance guard:
                # if increment is exactly zero, do not insert new labels.
                if incr == 0.0 and label not in stage_map:
                    continue
                stage_map[label] = stage_map.get(label, 0.0 + 0.0j) + incr

        if coeff_drop_tol > 0.0:
            for label in list(stage_map):
                if abs(stage_map[label]) < coeff_drop_tol:
                    del stage_map[label]

        stage_maps.append(stage_map)

    backend_cfg = {
        "backend": str(_cfg_get(config, "backend", "expm_multiply_sparse")),
        "sparse_min_dim": int(_cfg_get(config, "sparse_min_dim", 64)),
        "scheme_name": str(scheme.get("name", "")),
        "emit_inner_order_warning": bool(_cfg_get(config, "emit_inner_order_warning", True)),
    }

    psi_next = np.asarray(psi, dtype=complex)
    # Rightmost exponential acts first on statevectors => descending stage index.
    for k in range(s_stages - 1, -1, -1):
        psi_next = apply_stage_exponential(
            psi=psi_next,
            stage_coeff_map=stage_maps[k],
            dt=dt_f,
            ordered_labels=ordered_labels,
            backend_config=backend_cfg,
        )

    if bool(_cfg_get(config, "normalize", False)):
        norm_before = float(np.linalg.norm(psi_next))
        if norm_before <= 0.0:
            raise ValueError("Encountered zero-norm state in cfqm_step.")
        norm_drift = abs(norm_before - 1.0)
        logger = _cfg_get(config, "norm_drift_logger", None)
        if callable(logger):
            logger(
                event="cfqm_norm_drift",
                t_abs=float(t_abs),
                dt=dt_f,
                norm_before=norm_before,
                norm_drift=norm_drift,
            )
        psi_next = psi_next / norm_before

    return np.asarray(psi_next, dtype=complex)
