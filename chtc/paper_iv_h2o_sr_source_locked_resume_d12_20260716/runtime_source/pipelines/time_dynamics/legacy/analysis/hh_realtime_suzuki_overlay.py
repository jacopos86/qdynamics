#!/usr/bin/env python3
"""Suzuki/controller/exact overlay report for HH realtime artifacts.

This report is deliberately read-mostly: it reuses a completed Chapter 17A
controller JSON, rebuilds the same HH seed/context, then runs local
statevector Suzuki order-1 and order-2 controls on the source time grid.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import textwrap
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.hardcoded import hubbard_pipeline as hc_pipeline
from pipelines.hardcoded.adapt_circuit_execution import (
    append_pauli_rotation_exyz,
    build_ansatz_circuit,
)
from pipelines.time_dynamics.fixed_manifold.mclachlan import (
    FixedManifoldRunSpec,
    LoadedRunContext,
    load_run_context,
)
from pipelines.qiskit_backend_tools import (
    compile_circuit_for_backend,
    compiled_gate_stats,
    export_compiled_circuit_artifacts,
    rank_compile_rows,
    resolve_backend_targets,
    safe_circuit_depth,
)
from src.quantum.drives_time_potential import (
    build_gaussian_sinusoid_density_drive,
    reference_method_name,
)


DEFAULT_CONTROLLER_JSON = Path(
    "artifacts/agent_runs/"
    "20260421_hh_l2_t8_good_trial_compile_audit_regression_v1/result.json"
)
DEFAULT_SOURCE_PDF = Path("output/pdf/20260421_hh_l2_t8_good_trial_compile_audit_regression.pdf")
DEFAULT_PREFERRED_FAKES = ("FakeMarrakesh", "FakeNighthawk", "FakeFez")


@dataclass(frozen=True)
class SuzukiOverlayConfig:
    controller_json: Path
    output_json: Path | None
    output_pdf: Path | None
    source_pdf: Path
    trotter_steps: int | None
    suzuki_orders: tuple[int, ...]
    backend_name: str
    seed_transpiler: int
    optimization_level: int
    preferred_fake_backends: tuple[str, ...]
    export_compiled_circuits: bool = False
    compiled_circuit_dir: Path | None = None
    skip_pdf: bool = False


@dataclass(frozen=True)
class CircuitCostRow:
    method: str
    order: int | None
    scope: str
    trotter_steps: int | None
    includes_seed_prep: bool
    abstract_size: int | None
    abstract_depth: int | None
    compiled_count_2q: int | None
    compiled_depth: int | None
    compiled_size: int | None
    compiled_num_qubits: int | None
    backend_name: str | None
    seed_transpiler: int | None
    optimization_level: int | None
    transpile_status: str
    compiled_op_counts: dict[str, int]
    logical_to_physical: list[int]
    error: str | None = None
    compiled_circuit_artifacts: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class SuzukiOverlayResult:
    method: str
    order: int
    trajectory: list[dict[str, Any]]
    summary: dict[str, Any]
    final_state: np.ndarray


@dataclass(frozen=True)
class RebuiltOverlayContext:
    loaded: LoadedRunContext
    hmat: np.ndarray
    ordered_labels_exyz: list[str]
    coeff_map_exyz: dict[str, complex]
    psi_initial: np.ndarray
    nq: int
    drive_coeff_provider_exyz: Any | None
    drive_profile: dict[str, Any] | None
    drive_meta: dict[str, Any] | None


"Built Math: U_1(dt)=prod_j exp(-i dt c_j(t_k) P_j); U_2(dt)=prod_j exp(-i dt c_j(t_k) P_j/2) prod_j^rev exp(-i dt c_j(t_k) P_j/2)."
def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(x) for x in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"re": float(np.real(value)), "im": float(np.imag(value))}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object at {path}.")
    return dict(payload)


def _parse_int_tuple(raw: str | Sequence[int]) -> tuple[int, ...]:
    chunks: Sequence[Any]
    if isinstance(raw, str):
        chunks = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    else:
        chunks = list(raw)
    out: list[int] = []
    seen: set[int] = set()
    for chunk in chunks:
        order = int(chunk)
        if order not in {1, 2}:
            raise ValueError("Only Suzuki orders 1 and 2 are supported.")
        if order not in seen:
            seen.add(order)
            out.append(order)
    if not out:
        raise ValueError("At least one Suzuki order is required.")
    return tuple(out)


def _parse_string_tuple(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    out: list[str] = []
    seen: set[str] = set()
    for chunk in str(raw).split(","):
        token = chunk.strip()
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(token)
    return tuple(out)


def _default_output_paths(controller_json: Path, source_payload: Mapping[str, Any]) -> tuple[Path, Path]:
    run_dir_name = Path(controller_json).resolve().parent.name
    if run_dir_name in {"", "."}:
        run_dir_name = str(source_payload.get("run_tag", "hh_realtime_suzuki_overlay"))
    stem = f"{run_dir_name}_suzuki_overlay"
    return Path("artifacts/agent_runs") / stem / "result.json", Path("output/pdf") / f"{stem}.pdf"


def _display_path(raw: Any) -> str:
    path = Path(str(raw)).expanduser()
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(raw)


def _load_source_payload(path: Path) -> dict[str, Any]:
    payload = _read_json_object(path)
    if "trajectory" not in payload or not isinstance(payload.get("trajectory"), Sequence):
        raise ValueError("Controller JSON must contain a trajectory array.")
    if "artifact_json" not in payload:
        raise ValueError("Controller JSON must contain artifact_json for seed reconstruction.")
    return payload


def _state_sample_rows(source_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for raw in source_payload.get("trajectory", []):
        if not isinstance(raw, Mapping):
            continue
        if str(raw.get("trajectory_sample_kind", "state_sample")) == "repair_event":
            continue
        if raw.get("advances_time", True) is False:
            continue
        rows.append(dict(raw))
    if not rows:
        raise ValueError("Controller JSON contains no state-sample trajectory rows.")
    return rows


def _source_times(source_payload: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    reference = source_payload.get("reference", {})
    ref_times = reference.get("times", None) if isinstance(reference, Mapping) else None
    if isinstance(ref_times, Sequence) and not isinstance(ref_times, (str, bytes, bytearray)):
        times = np.asarray([float(x) for x in ref_times], dtype=float)
    else:
        times = np.asarray([float(row["time"]) for row in rows], dtype=float)
    if times.size < 2:
        raise ValueError("Need at least two time points for Suzuki overlay.")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("Source time grid must be strictly increasing.")
    if len(rows) != int(times.size):
        raise ValueError(f"Source trajectory rows ({len(rows)}) do not match time grid ({times.size}).")
    row_times = np.asarray([float(row["time"]) for row in rows], dtype=float)
    if not np.allclose(row_times, times, rtol=0.0, atol=1.0e-10):
        raise ValueError("Source trajectory row times do not match reference time grid.")
    return times


def _source_physical_times(
    rows: Sequence[Mapping[str, Any]],
    *,
    fallback_drive_t0: float,
) -> np.ndarray:
    return np.asarray(
        [
            float(row.get("physical_time", float(row["time"]) + float(fallback_drive_t0)))
            for row in rows
        ],
        dtype=float,
    )


def _uniform_dt(times: np.ndarray, trotter_steps: int) -> float:
    if int(trotter_steps) != int(times.size) - 1:
        raise ValueError(
            f"trotter_steps={int(trotter_steps)} must equal len(source_times)-1={int(times.size) - 1}."
        )
    diffs = np.diff(np.asarray(times, dtype=float))
    dt = float(diffs[0])
    if not np.allclose(diffs, dt, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("Source time grid must be uniform for fixed-step Suzuki overlay.")
    return dt


def _source_compile_defaults(source_payload: Mapping[str, Any]) -> dict[str, Any]:
    audit = source_payload.get("compile_audit", {})
    request = audit.get("request", {}) if isinstance(audit, Mapping) else {}
    observation = audit.get("observation", {}) if isinstance(audit, Mapping) else {}
    summary = source_payload.get("summary", {})
    summary_request = summary.get("oracle_compile_request", {}) if isinstance(summary, Mapping) else {}
    summary_observation = (
        summary.get("oracle_compile_observation", {}) if isinstance(summary, Mapping) else {}
    )
    req = request if isinstance(request, Mapping) and request else summary_request
    obs = observation if isinstance(observation, Mapping) and observation else summary_observation
    preferred = req.get("preferred_fake_backends", DEFAULT_PREFERRED_FAKES) if isinstance(req, Mapping) else DEFAULT_PREFERRED_FAKES
    return {
        "backend_name": str(
            (req.get("backend_name", None) if isinstance(req, Mapping) else None)
            or (obs.get("backend_name", None) if isinstance(obs, Mapping) else None)
            or "FakeMarrakesh"
        ),
        "seed_transpiler": int(
            (req.get("seed_transpiler", None) if isinstance(req, Mapping) else None)
            or (obs.get("seed_transpiler", None) if isinstance(obs, Mapping) else None)
            or 7
        ),
        "optimization_level": int(
            (req.get("optimization_level", None) if isinstance(req, Mapping) else None)
            or (obs.get("optimization_level", None) if isinstance(obs, Mapping) else None)
            or 2
        ),
        "preferred_fake_backends": tuple(str(x) for x in preferred),
    }


def _rebuild_context(source_payload: Mapping[str, Any]) -> RebuiltOverlayContext:
    loader_summary = (
        source_payload.get("loader_summary", {})
        if isinstance(source_payload.get("loader_summary", {}), Mapping)
        else {}
    )
    run_tag = str(source_payload.get("run_tag", "hh_realtime_suzuki_overlay"))
    spec = FixedManifoldRunSpec(
        name=f"{run_tag}_suzuki_overlay",
        artifact_json=Path(str(source_payload["artifact_json"])).expanduser().resolve(),
        loader_mode=str(source_payload.get("loader_mode", "replay_family")),
        generator_family=str(loader_summary.get("generator_family", "match_adapt")),
        fallback_family=str(loader_summary.get("fallback_family", "full_meta")),
        append_pool_family=str(loader_summary.get("append_pool_family_requested", "match_replay")),
    )
    loaded = load_run_context(spec, tag=f"{run_tag}_suzuki_overlay", lock_fixed_manifold=False)
    native_order, coeff_map = hc_pipeline._collect_hardcoded_terms_exyz(loaded.replay_context.h_poly)
    term_order = str(loaded.payload.get("settings", {}).get("term_order", "sorted"))
    ordered_labels = list(native_order) if term_order == "native" else sorted(coeff_map)
    hmat = hc_pipeline._build_hamiltonian_matrix(coeff_map)
    psi_initial = np.asarray(loaded.psi_initial, dtype=complex).reshape(-1)
    nq = int(round(np.log2(int(psi_initial.size))))
    drive_provider, drive_meta, drive_profile = _build_drive_provider(
        source_payload,
        loaded=loaded,
        nq=nq,
        ordered_labels_exyz=ordered_labels,
    )
    return RebuiltOverlayContext(
        loaded=loaded,
        hmat=np.asarray(hmat, dtype=complex),
        ordered_labels_exyz=list(ordered_labels),
        coeff_map_exyz=dict(coeff_map),
        psi_initial=np.asarray(psi_initial, dtype=complex),
        nq=int(nq),
        drive_coeff_provider_exyz=drive_provider,
        drive_profile=drive_profile,
        drive_meta=drive_meta,
    )


def _build_drive_provider(
    source_payload: Mapping[str, Any],
    *,
    loaded: LoadedRunContext,
    nq: int,
    ordered_labels_exyz: Sequence[str],
) -> tuple[Any | None, dict[str, Any] | None, dict[str, Any] | None]:
    drive_cfg = (
        source_payload.get("drive_config", {})
        if isinstance(source_payload.get("drive_config", {}), Mapping)
        else {}
    )
    if not bool(drive_cfg.get("enabled", False)):
        return None, None, None
    custom_weights = drive_cfg.get("drive_custom_weights", None)
    if custom_weights is not None:
        custom_weights = [float(x) for x in custom_weights]
    drive = build_gaussian_sinusoid_density_drive(
        n_sites=int(drive_cfg.get("n_sites", getattr(loaded.cfg, "L", 0))),
        nq_total=int(nq),
        indexing=str(drive_cfg.get("ordering", getattr(loaded.cfg, "ordering", "blocked"))),
        A=float(drive_cfg.get("drive_A", 0.0)),
        omega=float(drive_cfg.get("drive_omega", 1.0)),
        tbar=float(drive_cfg.get("drive_tbar", 1.0)),
        phi=float(drive_cfg.get("drive_phi", 0.0)),
        pattern_mode=str(drive_cfg.get("drive_pattern", "staggered")),
        custom_weights=custom_weights,
        include_identity=bool(drive_cfg.get("drive_include_identity", False)),
        coeff_tol=0.0,
    )
    drive_labels = set(drive.template.labels_exyz(include_identity=bool(drive.include_identity)))
    _validate_no_missing_drive_labels(drive_labels, ordered_labels_exyz)
    profile = {
        "enabled": True,
        "A": float(drive_cfg.get("drive_A", 0.0)),
        "omega": float(drive_cfg.get("drive_omega", 1.0)),
        "tbar": float(drive_cfg.get("drive_tbar", 1.0)),
        "phi": float(drive_cfg.get("drive_phi", 0.0)),
        "pattern": str(drive_cfg.get("drive_pattern", "staggered")),
        "custom_weights": custom_weights,
        "include_identity": bool(drive_cfg.get("drive_include_identity", False)),
        "time_sampling": str(drive_cfg.get("drive_time_sampling", "midpoint")),
        "t0": float(drive_cfg.get("drive_t0", 0.0)),
    }
    meta = {
        "reference_method": reference_method_name(str(profile["time_sampling"])),
        "drive_label_count": int(len(drive_labels)),
        "missing_drive_labels_added": 0,
    }
    return drive.coeff_map_exyz, meta, profile


def _validate_no_missing_drive_labels(
    drive_labels: set[str],
    ordered_labels_exyz: Sequence[str],
) -> None:
    missing = sorted(set(drive_labels).difference(str(x) for x in ordered_labels_exyz))
    if missing:
        raise ValueError(
            "Drive labels are missing from ordered_labels_exyz for Suzuki overlay; "
            f"refusing to insert labels in this report path: {missing}"
        )


def _sample_time_for_step(
    k: int,
    *,
    dt: float,
    drive_t0: float,
    time_sampling: str,
) -> float:
    sampling = str(time_sampling).strip().lower()
    if sampling == "midpoint":
        return float(drive_t0) + (float(k) + 0.5) * float(dt)
    if sampling == "left":
        return float(drive_t0) + float(k) * float(dt)
    if sampling == "right":
        return float(drive_t0) + (float(k) + 1.0) * float(dt)
    raise ValueError("drive_time_sampling must be one of midpoint, left, right.")


def _coeff_at_step(
    *,
    k: int,
    dt: float,
    coeff_map_exyz: Mapping[str, complex],
    ordered_labels_exyz: Sequence[str],
    drive_coeff_provider_exyz: Any | None,
    drive_t0: float,
    drive_time_sampling: str,
) -> dict[str, complex]:
    drive_map: Mapping[str, Any] = {}
    if drive_coeff_provider_exyz is not None:
        sample_time = _sample_time_for_step(
            int(k),
            dt=float(dt),
            drive_t0=float(drive_t0),
            time_sampling=str(drive_time_sampling),
        )
        drive_map = dict(drive_coeff_provider_exyz(float(sample_time)))
        extra = sorted(set(str(x) for x in drive_map).difference(str(x) for x in ordered_labels_exyz))
        if extra:
            raise ValueError(f"Drive provider returned labels outside ordered label set: {extra}")
    return {
        str(label): complex(coeff_map_exyz.get(str(label), 0.0 + 0.0j))
        + complex(drive_map.get(str(label), 0.0 + 0.0j))
        for label in ordered_labels_exyz
    }


def _apply_lie_step(
    psi: np.ndarray,
    *,
    ordered_labels_exyz: Sequence[str],
    compiled_actions: Mapping[str, Any],
    coeffs: Mapping[str, complex],
    dt: float,
) -> np.ndarray:
    out = np.asarray(psi, dtype=complex).reshape(-1)
    for label in ordered_labels_exyz:
        out = hc_pipeline._apply_exp_term(
            out,
            compiled_actions[str(label)],
            complex(coeffs[str(label)]),
            float(dt),
        )
    return hc_pipeline._normalize_state(out)


def _apply_strang_step(
    psi: np.ndarray,
    *,
    ordered_labels_exyz: Sequence[str],
    compiled_actions: Mapping[str, Any],
    coeffs: Mapping[str, complex],
    dt: float,
) -> np.ndarray:
    out = np.asarray(psi, dtype=complex).reshape(-1)
    half = 0.5 * float(dt)
    for label in ordered_labels_exyz:
        out = hc_pipeline._apply_exp_term(
            out,
            compiled_actions[str(label)],
            complex(coeffs[str(label)]),
            half,
        )
    for label in reversed(list(ordered_labels_exyz)):
        out = hc_pipeline._apply_exp_term(
            out,
            compiled_actions[str(label)],
            complex(coeffs[str(label)]),
            half,
        )
    return hc_pipeline._normalize_state(out)


def _hmat_total_at_observation(
    *,
    hmat_static: np.ndarray,
    drive_coeff_provider_exyz: Any | None,
    physical_time: float,
    nq: int,
) -> np.ndarray:
    if drive_coeff_provider_exyz is None:
        return np.asarray(hmat_static, dtype=complex)
    return np.asarray(hmat_static, dtype=complex) + hc_pipeline._build_drive_matrix_at_time(
        drive_coeff_provider_exyz,
        float(physical_time),
        int(nq),
    )


def _summarize_energy_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors = np.asarray([float(row.get("abs_energy_total_error", np.nan)) for row in rows], dtype=float)
    energies = np.asarray([float(row.get("energy_total", np.nan)) for row in rows], dtype=float)
    exact = np.asarray([float(row.get("energy_total_exact", np.nan)) for row in rows], dtype=float)
    return {
        "row_count": int(len(rows)),
        "final_energy_total": float(energies[-1]) if energies.size else float("nan"),
        "final_energy_total_exact": float(exact[-1]) if exact.size else float("nan"),
        "final_abs_energy_total_error": float(errors[-1]) if errors.size else float("nan"),
        "mean_abs_energy_total_error": float(np.nanmean(errors)) if errors.size else float("nan"),
        "max_abs_energy_total_error": float(np.nanmax(errors)) if errors.size else float("nan"),
    }


def _simulate_suzuki_order(
    *,
    order: int,
    psi_initial: np.ndarray,
    times: np.ndarray,
    exact_energy_total: Sequence[float] | None,
    observation_physical_times: Sequence[float],
    ordered_labels_exyz: Sequence[str],
    coeff_map_exyz: Mapping[str, complex],
    hmat_static: np.ndarray,
    drive_coeff_provider_exyz: Any | None,
    drive_t0: float,
    drive_time_sampling: str,
    nq: int,
) -> SuzukiOverlayResult:
    if int(order) not in {1, 2}:
        raise ValueError("Only Suzuki orders 1 and 2 are supported.")
    trotter_steps = int(len(times)) - 1
    dt = _uniform_dt(np.asarray(times, dtype=float), trotter_steps)
    compiled = {
        str(label): hc_pipeline._compile_pauli_action(str(label), int(nq))
        for label in ordered_labels_exyz
    }
    exact_arr = None if exact_energy_total is None else np.asarray(exact_energy_total, dtype=float)
    obs_physical = np.asarray(observation_physical_times, dtype=float)
    if obs_physical.size != int(times.size):
        raise ValueError("observation_physical_times must match source time grid.")
    if exact_arr is not None and exact_arr.size != int(times.size):
        raise ValueError("exact_energy_total must match source time grid.")

    rows: list[dict[str, Any]] = []
    psi = hc_pipeline._normalize_state(np.asarray(psi_initial, dtype=complex).reshape(-1))

    def _append_row(idx: int, state: np.ndarray) -> None:
        hmat_total = _hmat_total_at_observation(
            hmat_static=np.asarray(hmat_static, dtype=complex),
            drive_coeff_provider_exyz=drive_coeff_provider_exyz,
            physical_time=float(obs_physical[int(idx)]),
            nq=int(nq),
        )
        energy = float(hc_pipeline._expectation_hamiltonian(state, hmat_total))
        exact_energy = None if exact_arr is None else float(exact_arr[int(idx)])
        err = None if exact_energy is None else float(abs(energy - exact_energy))
        rows.append(
            {
                "checkpoint_index": int(idx),
                "time": float(times[int(idx)]),
                "physical_time": float(obs_physical[int(idx)]),
                "method": f"suzuki{int(order)}",
                "suzuki_order": int(order),
                "energy_total": float(energy),
                "energy_total_exact": exact_energy,
                "abs_energy_total_error": err,
                "state_norm": float(np.linalg.norm(state)),
            }
        )

    _append_row(0, psi)
    for k in range(trotter_steps):
        coeffs = _coeff_at_step(
            k=int(k),
            dt=float(dt),
            coeff_map_exyz=coeff_map_exyz,
            ordered_labels_exyz=ordered_labels_exyz,
            drive_coeff_provider_exyz=drive_coeff_provider_exyz,
            drive_t0=float(drive_t0),
            drive_time_sampling=str(drive_time_sampling),
        )
        if int(order) == 1:
            psi = _apply_lie_step(
                psi,
                ordered_labels_exyz=ordered_labels_exyz,
                compiled_actions=compiled,
                coeffs=coeffs,
                dt=float(dt),
            )
        else:
            psi = _apply_strang_step(
                psi,
                ordered_labels_exyz=ordered_labels_exyz,
                compiled_actions=compiled,
                coeffs=coeffs,
                dt=float(dt),
            )
        _append_row(k + 1, psi)

    method = f"suzuki{int(order)}"
    return SuzukiOverlayResult(
        method=method,
        order=int(order),
        trajectory=rows,
        summary=_summarize_energy_rows(rows),
        final_state=np.asarray(psi, dtype=complex).reshape(-1),
    )


def _append_trotter_step_to_circuit(
    qc: Any,
    *,
    order: int,
    ordered_labels_exyz: Sequence[str],
    coeffs: Mapping[str, complex],
    dt: float,
    coeff_tol: float = 1.0e-15,
) -> None:
    if int(order) == 1:
        for label in ordered_labels_exyz:
            coeff = complex(coeffs[str(label)])
            if abs(coeff.real) <= float(coeff_tol) and abs(coeff.imag) <= float(coeff_tol):
                continue
            if abs(coeff.imag) > 1.0e-12:
                raise ValueError(f"Imaginary coefficient for {label}: {coeff}")
            append_pauli_rotation_exyz(qc, label_exyz=str(label), angle=2.0 * float(dt) * float(coeff.real))
        return
    if int(order) == 2:
        half = 0.5 * float(dt)
        for label in ordered_labels_exyz:
            coeff = complex(coeffs[str(label)])
            if abs(coeff.real) <= float(coeff_tol) and abs(coeff.imag) <= float(coeff_tol):
                continue
            if abs(coeff.imag) > 1.0e-12:
                raise ValueError(f"Imaginary coefficient for {label}: {coeff}")
            append_pauli_rotation_exyz(qc, label_exyz=str(label), angle=2.0 * half * float(coeff.real))
        for label in reversed(list(ordered_labels_exyz)):
            coeff = complex(coeffs[str(label)])
            if abs(coeff.real) <= float(coeff_tol) and abs(coeff.imag) <= float(coeff_tol):
                continue
            if abs(coeff.imag) > 1.0e-12:
                raise ValueError(f"Imaginary coefficient for {label}: {coeff}")
            append_pauli_rotation_exyz(qc, label_exyz=str(label), angle=2.0 * half * float(coeff.real))
        return
    raise ValueError("Only Suzuki orders 1 and 2 are supported.")


def _build_trotter_circuit(
    *,
    order: int,
    nq: int,
    ordered_labels_exyz: Sequence[str],
    coeff_map_exyz: Mapping[str, complex],
    trotter_steps: int,
    dt: float,
    drive_coeff_provider_exyz: Any | None,
    drive_t0: float,
    drive_time_sampling: str,
    include_seed_prep: bool,
    seed_circuit: Any | None = None,
    step_index: int | None = None,
) -> Any:
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(int(nq))
    if bool(include_seed_prep):
        if seed_circuit is None:
            raise ValueError("include_seed_prep requires seed_circuit.")
        qc.compose(seed_circuit, inplace=True)
    step_indices = [int(step_index)] if step_index is not None else list(range(int(trotter_steps)))
    for k in step_indices:
        coeffs = _coeff_at_step(
            k=int(k),
            dt=float(dt),
            coeff_map_exyz=coeff_map_exyz,
            ordered_labels_exyz=ordered_labels_exyz,
            drive_coeff_provider_exyz=drive_coeff_provider_exyz,
            drive_t0=float(drive_t0),
            drive_time_sampling=str(drive_time_sampling),
        )
        _append_trotter_step_to_circuit(
            qc,
            order=int(order),
            ordered_labels_exyz=ordered_labels_exyz,
            coeffs=coeffs,
            dt=float(dt),
        )
    return qc


def _compile_one_circuit_cost(
    *,
    method: str,
    order: int | None,
    scope: str,
    trotter_steps: int | None,
    includes_seed_prep: bool,
    circuit: Any,
    backend_name: str,
    preferred_fake_backends: Sequence[str],
    seed_transpiler: int,
    optimization_level: int,
    export_circuit_dir: Path | None = None,
    export_stem: str | None = None,
) -> tuple[CircuitCostRow, list[dict[str, Any]]]:
    try:
        targets, resolution_audit = resolve_backend_targets(
            requested_names=(str(backend_name),),
            preferred_fake_backends=tuple(str(x) for x in preferred_fake_backends),
            allow_preferred_fallback=True,
            fallback_mode="single",
            allow_runtime_lookup=False,
        )
    except Exception as exc:
        return (
            CircuitCostRow(
                method=str(method),
                order=order,
                scope=str(scope),
                trotter_steps=trotter_steps,
                includes_seed_prep=bool(includes_seed_prep),
                abstract_size=int(circuit.size()),
                abstract_depth=int(safe_circuit_depth(circuit)),
                compiled_count_2q=None,
                compiled_depth=None,
                compiled_size=None,
                compiled_num_qubits=None,
                backend_name=str(backend_name),
                seed_transpiler=int(seed_transpiler),
                optimization_level=int(optimization_level),
                transpile_status="error",
                compiled_op_counts={},
                logical_to_physical=[],
                error=f"{type(exc).__name__}: {exc}",
            ),
            [],
        )

    rows: list[dict[str, Any]] = []
    compiled_by_backend: dict[str, Any] = {}
    for target in targets:
        row: dict[str, Any] = {
            "method": str(method),
            "order": order,
            "scope": str(scope),
            "trotter_steps": trotter_steps,
            "includes_seed_prep": bool(includes_seed_prep),
            "abstract_size": int(circuit.size()),
            "abstract_depth": int(safe_circuit_depth(circuit)),
            "requested_backend": str(target.requested_name),
            "transpile_backend": str(target.resolved_name),
            "backend_name": str(target.resolved_name),
            "resolution_kind": str(target.resolution_kind),
            "using_fake_backend": bool(target.using_fake_backend),
            "seed_transpiler": int(seed_transpiler),
            "optimization_level": int(optimization_level),
            "transpile_status": "not_run",
            "error": None,
        }
        try:
            compiled_info = compile_circuit_for_backend(
                circuit,
                target.backend_obj,
                seed_transpiler=int(seed_transpiler),
                optimization_level=int(optimization_level),
            )
            compiled = compiled_info["compiled"]
            compiled_by_backend[str(target.resolved_name)] = compiled
            row.update(
                {
                    "transpile_status": "ok",
                    "compiled_depth": int(safe_circuit_depth(compiled)),
                    "compiled_size": int(compiled.size()),
                    "compiled_num_qubits": int(compiled_info.get("compiled_num_qubits", compiled.num_qubits)),
                    "logical_to_physical": [int(x) for x in compiled_info.get("logical_to_physical", ())],
                }
            )
            row.update(dict(compiled_gate_stats(compiled)))
        except Exception as exc:
            row.update(
                {
                    "transpile_status": "error",
                    "compiled_depth": None,
                    "compiled_size": None,
                    "compiled_num_qubits": None,
                    "logical_to_physical": [],
                    "compiled_count_2q": None,
                    "compiled_op_counts": {},
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        rows.append(row)

    selected = rank_compile_rows(rows)
    compiled_circuit_artifacts: list[dict[str, Any]] = []
    if selected is None:
        cost = CircuitCostRow(
            method=str(method),
            order=order,
            scope=str(scope),
            trotter_steps=trotter_steps,
            includes_seed_prep=bool(includes_seed_prep),
            abstract_size=int(circuit.size()),
            abstract_depth=int(safe_circuit_depth(circuit)),
            compiled_count_2q=None,
            compiled_depth=None,
            compiled_size=None,
            compiled_num_qubits=None,
            backend_name=str(backend_name),
            seed_transpiler=int(seed_transpiler),
            optimization_level=int(optimization_level),
            transpile_status="error",
            compiled_op_counts={},
            logical_to_physical=[],
            error="no_compile_target_succeeded",
            compiled_circuit_artifacts=[],
        )
    else:
        selected_backend = str(selected.get("transpile_backend", selected.get("backend_name", "")))
        if export_circuit_dir is not None:
            compiled_selected = compiled_by_backend.get(selected_backend)
            if compiled_selected is not None:
                stem = export_stem or f"{method}_{scope}"
                artifact = export_compiled_circuit_artifacts(
                    compiled_selected,
                    output_dir=Path(export_circuit_dir),
                    stem=str(stem),
                    metadata={
                        "method": str(method),
                        "order": order,
                        "scope": str(scope),
                        "trotter_steps": trotter_steps,
                        "includes_seed_prep": bool(includes_seed_prep),
                        "backend_name": selected_backend,
                        "seed_transpiler": int(seed_transpiler),
                        "optimization_level": int(optimization_level),
                    },
                )
                compiled_circuit_artifacts.append(dict(artifact))
                selected["compiled_circuit_artifacts"] = [dict(artifact)]
        cost = CircuitCostRow(
            method=str(method),
            order=order,
            scope=str(scope),
            trotter_steps=trotter_steps,
            includes_seed_prep=bool(includes_seed_prep),
            abstract_size=int(selected.get("abstract_size", circuit.size())),
            abstract_depth=int(selected.get("abstract_depth", safe_circuit_depth(circuit))),
            compiled_count_2q=int(selected["compiled_count_2q"]),
            compiled_depth=int(selected["compiled_depth"]),
            compiled_size=int(selected["compiled_size"]),
            compiled_num_qubits=int(selected["compiled_num_qubits"]),
            backend_name=str(selected.get("transpile_backend", selected.get("backend_name", ""))),
            seed_transpiler=int(seed_transpiler),
            optimization_level=int(optimization_level),
            transpile_status="ok",
            compiled_op_counts={str(k): int(v) for k, v in dict(selected.get("compiled_op_counts", {})).items()},
            logical_to_physical=[int(x) for x in selected.get("logical_to_physical", [])],
            error=None,
            compiled_circuit_artifacts=list(compiled_circuit_artifacts),
        )
    return cost, _jsonable(rows + [{"resolution_audit": resolution_audit}])


def _source_controller_cost_row(source_payload: Mapping[str, Any]) -> CircuitCostRow | None:
    audit = source_payload.get("compile_audit", {})
    selected = audit.get("selected_backend", None) if isinstance(audit, Mapping) else None
    observation = audit.get("observation", None) if isinstance(audit, Mapping) else None
    row = selected if isinstance(selected, Mapping) else observation
    if not isinstance(row, Mapping):
        summary = source_payload.get("summary", {})
        obs = summary.get("oracle_compile_observation", None) if isinstance(summary, Mapping) else None
        row = obs if isinstance(obs, Mapping) else None
    if not isinstance(row, Mapping):
        return None
    logical = audit.get("logical_circuit", {}) if isinstance(audit, Mapping) else {}
    artifacts = row.get("compiled_circuit_artifacts", None)
    if artifacts is None and isinstance(audit, Mapping):
        artifacts = audit.get("compiled_circuit_artifacts", None)
    if artifacts is None:
        artifacts = []
    return CircuitCostRow(
        method="controller",
        order=None,
        scope="controller_final_scaffold_source",
        trotter_steps=None,
        includes_seed_prep=True,
        abstract_size=(
            None if not isinstance(logical, Mapping) or logical.get("abstract_size") is None else int(logical["abstract_size"])
        ),
        abstract_depth=(
            None if not isinstance(logical, Mapping) or logical.get("abstract_depth") is None else int(logical["abstract_depth"])
        ),
        compiled_count_2q=None if row.get("compiled_count_2q") is None else int(row["compiled_count_2q"]),
        compiled_depth=None if row.get("compiled_depth") is None else int(row["compiled_depth"]),
        compiled_size=None if row.get("compiled_size") is None else int(row["compiled_size"]),
        compiled_num_qubits=(
            None if row.get("compiled_num_qubits") is None else int(row["compiled_num_qubits"])
        ),
        backend_name=None if row.get("transpile_backend", row.get("backend_name")) is None else str(row.get("transpile_backend", row.get("backend_name"))),
        seed_transpiler=None if row.get("seed_transpiler") is None else int(row["seed_transpiler"]),
        optimization_level=None if row.get("optimization_level") is None else int(row["optimization_level"]),
        transpile_status=str(row.get("transpile_status", "ok")),
        compiled_op_counts={str(k): int(v) for k, v in dict(row.get("compiled_op_counts", {})).items()},
        logical_to_physical=[int(x) for x in row.get("logical_to_physical", [])],
        error=None if row.get("error") is None else str(row.get("error")),
        compiled_circuit_artifacts=[dict(x) for x in artifacts if isinstance(x, Mapping)],
    )


def _copy_compiled_artifacts_to_dir(
    artifacts: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    for raw in artifacts:
        item = dict(raw)
        for key in ("qpy_path", "ops_jsonl_path", "preview_text_path"):
            src_raw = item.get(key)
            if src_raw in {None, ""}:
                continue
            src = Path(str(src_raw))
            if not src.exists():
                continue
            dst = out_dir / src.name
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            item[f"source_{key}"] = str(src)
            item[key] = str(dst)
        copied.append(item)
    return copied


def _compile_cost_rows(
    *,
    config: SuzukiOverlayConfig,
    source_payload: Mapping[str, Any],
    context: RebuiltOverlayContext,
    times: np.ndarray,
    trotter_steps: int,
    dt: float,
    compiled_circuit_dir: Path | None = None,
) -> tuple[list[CircuitCostRow], dict[str, Any]]:
    rows: list[CircuitCostRow] = []
    raw_compile_rows: dict[str, Any] = {"seed": [], "suzuki": []}
    controller_row = _source_controller_cost_row(source_payload)
    if controller_row is not None:
        if compiled_circuit_dir is not None and controller_row.compiled_circuit_artifacts:
            controller_row = replace(
                controller_row,
                compiled_circuit_artifacts=_copy_compiled_artifacts_to_dir(
                    controller_row.compiled_circuit_artifacts,
                    Path(compiled_circuit_dir),
                ),
            )
        rows.append(controller_row)

    seed_circuit = build_ansatz_circuit(
        context.loaded.replay_context.base_layout,
        np.asarray(context.loaded.replay_context.adapt_theta_runtime, dtype=float).reshape(-1),
        int(context.nq),
        ref_state=np.asarray(context.loaded.replay_context.psi_ref, dtype=complex).reshape(-1),
    )
    seed_cost, seed_raw_rows = _compile_one_circuit_cost(
        method="seed",
        order=None,
        scope="seed_prep_only",
        trotter_steps=None,
        includes_seed_prep=True,
        circuit=seed_circuit,
        backend_name=str(config.backend_name),
        preferred_fake_backends=tuple(config.preferred_fake_backends),
        seed_transpiler=int(config.seed_transpiler),
        optimization_level=int(config.optimization_level),
        export_circuit_dir=compiled_circuit_dir,
        export_stem="adapt_seed_prep_only",
    )
    rows.append(seed_cost)
    raw_compile_rows["seed"].append(
        {
            "method": "seed",
            "scope": "seed_prep_only",
            "selected": _jsonable(seed_cost),
            "raw_rows": seed_raw_rows,
        }
    )
    drive_t0 = float((context.drive_profile or {}).get("t0", 0.0))
    drive_sampling = str((context.drive_profile or {}).get("time_sampling", "midpoint"))
    for order in config.suzuki_orders:
        per_step = _build_trotter_circuit(
            order=int(order),
            nq=int(context.nq),
            ordered_labels_exyz=context.ordered_labels_exyz,
            coeff_map_exyz=context.coeff_map_exyz,
            trotter_steps=int(trotter_steps),
            dt=float(dt),
            drive_coeff_provider_exyz=context.drive_coeff_provider_exyz,
            drive_t0=float(drive_t0),
            drive_time_sampling=str(drive_sampling),
            include_seed_prep=False,
            seed_circuit=None,
            step_index=0,
        )
        one_step_with_seed = _build_trotter_circuit(
            order=int(order),
            nq=int(context.nq),
            ordered_labels_exyz=context.ordered_labels_exyz,
            coeff_map_exyz=context.coeff_map_exyz,
            trotter_steps=int(trotter_steps),
            dt=float(dt),
            drive_coeff_provider_exyz=context.drive_coeff_provider_exyz,
            drive_t0=float(drive_t0),
            drive_time_sampling=str(drive_sampling),
            include_seed_prep=True,
            seed_circuit=seed_circuit,
            step_index=0,
        )
        full = _build_trotter_circuit(
            order=int(order),
            nq=int(context.nq),
            ordered_labels_exyz=context.ordered_labels_exyz,
            coeff_map_exyz=context.coeff_map_exyz,
            trotter_steps=int(trotter_steps),
            dt=float(dt),
            drive_coeff_provider_exyz=context.drive_coeff_provider_exyz,
            drive_t0=float(drive_t0),
            drive_time_sampling=str(drive_sampling),
            include_seed_prep=True,
            seed_circuit=seed_circuit,
            step_index=None,
        )
        for scope, circuit, includes_seed, steps in (
            ("per_step_evolution_only", per_step, False, 1),
            ("one_step_with_seed_prep", one_step_with_seed, True, 1),
            ("full_horizon_with_seed_prep", full, True, int(trotter_steps)),
        ):
            cost, raw_rows = _compile_one_circuit_cost(
                method=f"suzuki{int(order)}",
                order=int(order),
                scope=str(scope),
                trotter_steps=int(steps),
                includes_seed_prep=bool(includes_seed),
                circuit=circuit,
                backend_name=str(config.backend_name),
                preferred_fake_backends=tuple(config.preferred_fake_backends),
                seed_transpiler=int(config.seed_transpiler),
                optimization_level=int(config.optimization_level),
                export_circuit_dir=compiled_circuit_dir,
                export_stem=(
                    f"suzuki{int(order)}_per_step_evolution_only_step000"
                    if str(scope) == "per_step_evolution_only"
                    else (
                        f"suzuki{int(order)}_one_step_with_seed_prep_step000"
                        if str(scope) == "one_step_with_seed_prep"
                        else f"suzuki{int(order)}_full_horizon_with_seed_prep_t{_fmt_float(times[-1], digits=8).replace('.', 'p')}_steps{int(trotter_steps)}"
                    )
                ),
            )
            rows.append(cost)
            raw_compile_rows["suzuki"].append(
                {
                    "method": f"suzuki{int(order)}",
                    "scope": str(scope),
                    "selected": _jsonable(cost),
                    "raw_rows": raw_rows,
                }
            )
    raw_compile_rows["time_grid"] = {"trotter_steps": int(trotter_steps), "dt": float(dt), "points": int(times.size)}
    return rows, raw_compile_rows


def _method_cost(cost_rows: Sequence[CircuitCostRow], method: str, scope: str) -> CircuitCostRow | None:
    for row in cost_rows:
        if row.method == method and row.scope == scope:
            return row
    return None


def _format_cost_value(row: CircuitCostRow | None, attr: str) -> Any:
    if row is None:
        return None
    return getattr(row, attr)


def _controller_summary(source_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for raw in source_rows:
        exact = float(raw["energy_total_exact"])
        energy = float(raw.get("energy_total_controller", raw.get("energy_total")))
        rows.append(
            {
                "energy_total": energy,
                "energy_total_exact": exact,
                "abs_energy_total_error": abs(energy - exact),
            }
        )
    return _summarize_energy_rows(rows)


def _exact_summary(source_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [
        {
            "energy_total": float(row["energy_total_exact"]),
            "energy_total_exact": float(row["energy_total_exact"]),
            "abs_energy_total_error": 0.0,
        }
        for row in source_rows
    ]
    return _summarize_energy_rows(rows)


def _scoreboard_rows(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    results: Sequence[SuzukiOverlayResult],
    cost_rows: Sequence[CircuitCostRow],
) -> list[dict[str, Any]]:
    scoreboard: list[dict[str, Any]] = []
    summaries = {
        "exact": _exact_summary(source_rows),
        "controller": _controller_summary(source_rows),
    }
    for result in results:
        summaries[result.method] = dict(result.summary)
    for method in ["exact", "controller", *[result.method for result in results]]:
        full_scope = "controller_final_scaffold_source" if method == "controller" else "full_horizon_with_seed_prep"
        per_scope = "per_step_evolution_only"
        full_cost = _method_cost(cost_rows, method, full_scope)
        per_cost = _method_cost(cost_rows, method, per_scope)
        summary = summaries[method]
        scoreboard.append(
            {
                "method": method,
                "final_energy_total": summary["final_energy_total"],
                "final_energy_total_exact": summary["final_energy_total_exact"],
                "final_abs_energy_total_error": summary["final_abs_energy_total_error"],
                "mean_abs_energy_total_error": summary["mean_abs_energy_total_error"],
                "max_abs_energy_total_error": summary["max_abs_energy_total_error"],
                "full_2q": _format_cost_value(full_cost, "compiled_count_2q"),
                "full_depth": _format_cost_value(full_cost, "compiled_depth"),
                "full_scope": None if full_cost is None else full_cost.scope,
                "full_cost_basis": None if full_cost is None else _cost_scope_label(full_cost),
                "per_step_2q": _format_cost_value(per_cost, "compiled_count_2q"),
                "per_step_depth": _format_cost_value(per_cost, "compiled_depth"),
                "per_step_scope": None if per_cost is None else per_cost.scope,
                "per_step_cost_basis": None if per_cost is None else _cost_scope_label(per_cost),
            }
        )
    return scoreboard


def _cost_scope_label(row: CircuitCostRow) -> str:
    if row.scope == "controller_final_scaffold_source":
        return "controller final scaffold"
    if row.scope == "seed_prep_only":
        return "seed prep only"
    if row.scope == "one_step_with_seed_prep":
        return "seed prep + one evolution macro-step"
    if row.scope == "full_horizon_with_seed_prep":
        steps = "?" if row.trotter_steps is None else str(int(row.trotter_steps))
        return f"full t horizon + seed ({steps} steps)"
    if row.scope == "per_step_evolution_only":
        return "one evolution macro-step"
    return str(row.scope)


def _hardware_metric_definitions() -> dict[str, str]:
    return {
        "compiled_count_2q": (
            "Number of two-qubit gates in the selected transpiled circuit for the chosen backend. "
            "For this FakeMarrakesh compile, the counted two-qubit operation is cz."
        ),
        "compiled_depth": (
            "Qiskit circuit depth after backend transpilation, routing, and decomposition. "
            "This is an operation-layer count, not a calibrated wall-clock duration."
        ),
        "compiled_size": "Total operation count after backend transpilation.",
        "abstract_size": "Operation count before backend transpilation.",
        "abstract_depth": "Circuit depth before backend transpilation.",
        "controller_final_scaffold_source": (
            "The source controller compile audit: the final variational scaffold circuit "
            "with 15 logical blocks and 29 runtime parameters. It is not an unrolled 160-step "
            "time-evolution circuit."
        ),
        "controller_per_time_step": (
            "Derived per-time-step controller cost. The source trajectory used the same 15-block, "
            "29-parameter scaffold at every reported time, so the final-scaffold compile is the "
            "representative controller circuit executed per time step."
        ),
        "controller_interval_execution_budget": (
            "Derived 160-interval budget for repeatedly executing the controller scaffold. "
            "The two-qubit count is multiplied by interval count; the depth sum is a serial "
            "execution budget, not one transpiled circuit depth."
        ),
        "seed_prep_only": (
            "Circuit that prepares the ADAPT-seeded initial state used by the Suzuki evolution."
        ),
        "seed_plus_one_step_additive": (
            "Apples-to-apples state-at-time Suzuki estimate: seed-prep cost plus one dt evolution "
            "cost, using separately compiled rows. This is the comparison against one controller "
            "state scaffold."
        ),
        "one_step_with_seed_prep": (
            "Audit row for compiling seed prep and one dt evolution as one circuit. This can differ "
            "from the additive estimate because transpilation and routing are global."
        ),
        "full_horizon_with_seed_prep": (
            "For Suzuki rows, the seed-preparation circuit followed by every evolution macro-step "
            "over t=8, compiled as one full-horizon circuit."
        ),
        "per_step_evolution_only": (
            "For Suzuki rows, one representative dt=0.05 evolution macro-step with no seed prep. "
            "The controller has no analogous Suzuki-style step block."
        ),
    }


def _hardware_cost_rows(cost_rows: Sequence[CircuitCostRow]) -> list[dict[str, Any]]:
    rows = []
    for row in cost_rows:
        rows.append(
            {
                "method": row.method,
                "scope": row.scope,
                "basis": _cost_scope_label(row),
                "trotter_steps": row.trotter_steps,
                "includes_seed_prep": row.includes_seed_prep,
                "abstract_size": row.abstract_size,
                "abstract_depth": row.abstract_depth,
                "compiled_count_2q": row.compiled_count_2q,
                "compiled_depth": row.compiled_depth,
                "compiled_size": row.compiled_size,
                "compiled_op_counts": dict(row.compiled_op_counts),
                "compiled_circuit_artifacts": list(row.compiled_circuit_artifacts or []),
                "backend_name": row.backend_name,
                "seed_transpiler": row.seed_transpiler,
                "optimization_level": row.optimization_level,
                "transpile_status": row.transpile_status,
            }
        )
    return rows


def _compiled_circuit_artifacts(cost_rows: Sequence[CircuitCostRow]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for row in cost_rows:
        for raw in row.compiled_circuit_artifacts or []:
            if not isinstance(raw, Mapping):
                continue
            item = dict(raw)
            item.setdefault("method", row.method)
            item.setdefault("scope", row.scope)
            item.setdefault("order", row.order)
            artifacts.append(item)
    return artifacts


def _mul_optional_int(value: int | None, factor: int) -> int | None:
    return None if value is None else int(value) * int(factor)


def _sum_optional_int(*values: int | None) -> int | None:
    if any(value is None for value in values):
        return None
    return int(sum(int(value) for value in values if value is not None))


def _hardware_report_rows(
    cost_rows: Sequence[CircuitCostRow],
    *,
    source_rows: Sequence[Mapping[str, Any]],
    trotter_steps: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_count = int(len(source_rows))
    controller = _method_cost(cost_rows, "controller", "controller_final_scaffold_source")
    seed = _method_cost(cost_rows, "seed", "seed_prep_only")
    if controller is not None:
        rows.append(
            {
                "method": "controller",
                "group": "state_at_time",
                "scope": "controller_state_at_time",
                "basis": "variational state scaffold",
                "circuit_calls": 1,
                "state_samples": int(sample_count),
                "compiled_count_2q": controller.compiled_count_2q,
                "compiled_depth": controller.compiled_depth,
                "compiled_size": controller.compiled_size,
                "horizon_count_2q": None,
                "horizon_depth_serial": None,
                "basis_note": (
                    "One controller state at one time point. Derived from the source final-scaffold compile "
                    "because every source trajectory row "
                    "uses the same logical block count and runtime parameter count."
                ),
                "source_scope": controller.scope,
            }
        )
        rows.append(
            {
                "method": "controller",
                "group": "horizon",
                "scope": "controller_interval_execution_budget",
                "basis": "160 repeated state scaffolds",
                "circuit_calls": int(trotter_steps),
                "state_samples": int(sample_count),
                "compiled_count_2q": controller.compiled_count_2q,
                "compiled_depth": controller.compiled_depth,
                "compiled_size": controller.compiled_size,
                "horizon_count_2q": _mul_optional_int(controller.compiled_count_2q, int(trotter_steps)),
                "horizon_depth_serial": _mul_optional_int(controller.compiled_depth, int(trotter_steps)),
                "basis_note": (
                    "Repeated-execution budget for preparing a controller state at each interval. "
                    "The serial depth is a budget, not a single transpiled circuit depth."
                ),
                "source_scope": controller.scope,
            }
        )
    if seed is not None:
        rows.append(
            {
                "method": "suzuki2",
                "group": "state_at_time",
                "scope": "seed_prep_only",
                "basis": "seed prep only",
                "circuit_calls": 1,
                "state_samples": None,
                "compiled_count_2q": seed.compiled_count_2q,
                "compiled_depth": seed.compiled_depth,
                "compiled_size": seed.compiled_size,
                "horizon_count_2q": None,
                "horizon_depth_serial": None,
                "basis_note": "Initial ADAPT-seed state preparation before Suzuki evolution.",
                "source_scope": seed.scope,
            }
        )
    for row in cost_rows:
        if row.method in {"controller", "seed"}:
            continue
        if row.scope == "per_step_evolution_only":
            rows.append(
                {
                    "method": row.method,
                    "group": "state_at_time",
                    "scope": row.scope,
                    "basis": "dt evolution only",
                    "circuit_calls": 1,
                    "state_samples": None,
                    "compiled_count_2q": row.compiled_count_2q,
                    "compiled_depth": row.compiled_depth,
                    "compiled_size": row.compiled_size,
                    "horizon_count_2q": None,
                    "horizon_depth_serial": None,
                    "basis_note": "Representative one-step Suzuki evolution block with no seed prep.",
                    "source_scope": row.scope,
                }
            )
            if seed is not None:
                rows.append(
                    {
                        "method": row.method,
                        "group": "state_at_time",
                        "scope": "seed_plus_one_step_additive",
                        "basis": "seed + dt additive",
                        "circuit_calls": 1,
                        "state_samples": None,
                        "compiled_count_2q": _sum_optional_int(seed.compiled_count_2q, row.compiled_count_2q),
                        "compiled_depth": _sum_optional_int(seed.compiled_depth, row.compiled_depth),
                        "compiled_size": _sum_optional_int(seed.compiled_size, row.compiled_size),
                        "horizon_count_2q": None,
                        "horizon_depth_serial": None,
                        "basis_note": (
                            "Additive apples-to-apples estimate for preparing the seed state and then "
                            "applying one Suzuki dt step."
                        ),
                        "source_scope": "seed_prep_only + per_step_evolution_only",
                    }
                )
        elif row.scope == "full_horizon_with_seed_prep":
            rows.append(
                {
                    "method": row.method,
                    "group": "horizon",
                    "scope": row.scope,
                    "basis": "compiled seed + 160 dt steps",
                    "circuit_calls": 1,
                    "state_samples": None,
                    "compiled_count_2q": row.compiled_count_2q,
                    "compiled_depth": row.compiled_depth,
                    "compiled_size": row.compiled_size,
                    "horizon_count_2q": row.compiled_count_2q,
                    "horizon_depth_serial": row.compiled_depth,
                    "basis_note": "One compiled circuit containing seed prep plus all evolution macro-steps.",
                    "source_scope": row.scope,
                }
            )
    return rows


def _parameter_manifest(
    *,
    config: SuzukiOverlayConfig,
    source_payload: Mapping[str, Any],
    context: RebuiltOverlayContext,
    times: np.ndarray,
    trotter_steps: int,
    output_json: Path,
    output_pdf: Path | None,
) -> dict[str, Any]:
    settings = context.loaded.payload.get("settings", {})
    drive_cfg = source_payload.get("drive_config", {}) if isinstance(source_payload.get("drive_config", {}), Mapping) else {}
    reference = source_payload.get("reference", {}) if isinstance(source_payload.get("reference", {}), Mapping) else {}
    suzuki_labels = "; ".join(f"Suzuki order {int(order)}" for order in config.suzuki_orders)
    ansatz_types = "Chapter17A controller final scaffold"
    if suzuki_labels:
        ansatz_types = f"{ansatz_types}; {suzuki_labels}"
    return {
        "model_family_name": "Hubbard-Holstein",
        "problem": str(settings.get("problem", "hh")),
        "L": int(settings.get("L", getattr(context.loaded.cfg, "L", 0))),
        "boundary": str(settings.get("boundary", getattr(context.loaded.cfg, "boundary", "open"))),
        "ordering": str(settings.get("ordering", getattr(context.loaded.cfg, "ordering", "blocked"))),
        "boson_encoding": str(settings.get("boson_encoding", getattr(context.loaded.cfg, "boson_encoding", "binary"))),
        "ansatz_types": ansatz_types,
        "t": float(settings.get("t", 1.0)),
        "U": float(settings.get("u", settings.get("U", 0.0))),
        "dv": float(settings.get("dv", 0.0)),
        "omega0": float(settings.get("omega0", 1.0)),
        "g_ep": float(settings.get("g_ep", 0.0)),
        "n_ph_max": int(settings.get("n_ph_max", 0)),
        "drive_enabled": bool(drive_cfg.get("enabled", False)),
        "drive_A": drive_cfg.get("drive_A"),
        "drive_omega": drive_cfg.get("drive_omega"),
        "drive_tbar": drive_cfg.get("drive_tbar"),
        "drive_phi": drive_cfg.get("drive_phi"),
        "drive_pattern": drive_cfg.get("drive_pattern"),
        "drive_time_sampling": drive_cfg.get("drive_time_sampling"),
        "drive_t0": drive_cfg.get("drive_t0"),
        "t_final": float(times[-1]),
        "num_times": int(times.size),
        "trotter_steps": int(trotter_steps),
        "exact_reference_method": reference.get("reference_method"),
        "exact_steps_multiplier": drive_cfg.get("exact_steps_multiplier", reference.get("reference_steps_multiplier")),
        "compile_backend": str(config.backend_name),
        "compile_seed_transpiler": int(config.seed_transpiler),
        "compile_optimization_level": int(config.optimization_level),
        "controller_json": _display_path(config.controller_json),
        "source_pdf": _display_path(config.source_pdf),
        "seed_artifact_json": _display_path(source_payload.get("artifact_json")),
        "output_json": _display_path(output_json),
        "output_pdf": None if output_pdf is None else _display_path(output_pdf),
    }


def _build_overlay_payload(
    *,
    config: SuzukiOverlayConfig,
    source_payload: Mapping[str, Any],
    context: RebuiltOverlayContext,
    source_rows: Sequence[Mapping[str, Any]],
    times: np.ndarray,
    trotter_steps: int,
    results: Sequence[SuzukiOverlayResult],
    cost_rows: Sequence[CircuitCostRow],
    raw_compile_rows: Mapping[str, Any],
    output_json: Path,
    output_pdf: Path | None,
    command: str,
) -> dict[str, Any]:
    methods: dict[str, Any] = {
        "exact": {
            "trajectory": [
                {
                    "checkpoint_index": int(idx),
                    "time": float(row["time"]),
                    "physical_time": float(row.get("physical_time", row["time"])),
                    "energy_total": float(row["energy_total_exact"]),
                    "energy_total_exact": float(row["energy_total_exact"]),
                    "abs_energy_total_error": 0.0,
                }
                for idx, row in enumerate(source_rows)
            ],
            "summary": _exact_summary(source_rows),
        },
        "controller": {
            "trajectory": [
                {
                    "checkpoint_index": int(idx),
                    "time": float(row["time"]),
                    "physical_time": float(row.get("physical_time", row["time"])),
                    "energy_total": float(row.get("energy_total_controller", row.get("energy_total"))),
                    "energy_total_exact": float(row["energy_total_exact"]),
                    "abs_energy_total_error": float(row.get("abs_energy_total_error", abs(float(row.get("energy_total_controller", row.get("energy_total"))) - float(row["energy_total_exact"])))),
                }
                for idx, row in enumerate(source_rows)
            ],
            "summary": _controller_summary(source_rows),
        },
    }
    for result in results:
        methods[str(result.method)] = {
            "order": int(result.order),
            "trajectory": result.trajectory,
            "summary": result.summary,
        }
    scoreboard = _scoreboard_rows(source_rows=source_rows, results=results, cost_rows=cost_rows)
    return _jsonable(
        {
            "schema_version": "hh_realtime_suzuki_overlay_v1",
            "generated_utc": _now_utc(),
            "command": str(command),
            "source": {
                "controller_json": _display_path(config.controller_json),
                "source_pdf": _display_path(config.source_pdf),
                "run_tag": source_payload.get("run_tag"),
                "artifact_json": _display_path(source_payload.get("artifact_json")),
            },
            "parameter_manifest": _parameter_manifest(
                config=config,
                source_payload=source_payload,
                context=context,
                times=times,
                trotter_steps=int(trotter_steps),
                output_json=output_json,
                output_pdf=output_pdf,
            ),
            "config": {
                "suzuki_orders": [int(x) for x in config.suzuki_orders],
                "trotter_steps": int(trotter_steps),
                "cost_basis": [
                    "controller_final_scaffold_source",
                    "seed_prep_only",
                    "per_step_evolution_only",
                    "seed_plus_one_step_additive",
                    "one_step_with_seed_prep",
                    "full_horizon_with_seed_prep",
                ],
                "missing_drive_label_policy": "fail",
                "export_compiled_circuits": bool(config.export_compiled_circuits),
                "compiled_circuit_dir": None if config.compiled_circuit_dir is None else str(config.compiled_circuit_dir),
            },
            "drive_meta": context.drive_meta,
            "methods": methods,
            "scoreboard": scoreboard,
            "hardware_metric_definitions": _hardware_metric_definitions(),
            "hardware_cost_rows": _hardware_cost_rows(cost_rows),
            "compiled_circuit_artifacts": _compiled_circuit_artifacts(cost_rows),
            "hardware_report_rows": _hardware_report_rows(
                cost_rows,
                source_rows=source_rows,
                trotter_steps=int(trotter_steps),
            ),
            "circuit_costs": [_jsonable(row) for row in cost_rows],
            "raw_compile_rows": raw_compile_rows,
        }
    )


def _write_overlay_json(path: Path, payload: Mapping[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return output_path


def _wrap_line(line: str, *, width: int = 104, indent: str = "  ") -> list[str]:
    text = str(line)
    if len(text) <= int(width):
        return [text]
    return textwrap.wrap(
        text,
        width=int(width),
        subsequent_indent=str(indent),
        break_long_words=False,
        break_on_hyphens=False,
    )


def _wrap_lines(lines: Sequence[str], *, width: int = 104) -> list[str]:
    out: list[str] = []
    for line in lines:
        if str(line) == "":
            out.append("")
            continue
        out.extend(_wrap_line(str(line), width=int(width)))
    return out


def _fmt_float(value: Any, *, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    try:
        val = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(val):
        return "n/a"
    if abs(val) >= 1000 or (abs(val) > 0 and abs(val) < 0.001):
        return f"{val:.3e}"
    return f"{val:.{digits}g}"


def _line_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return _fmt_float(value, digits=8)
    return str(value)


def _pdf_hardware_basis(row: Mapping[str, Any]) -> str:
    scope = str(row.get("scope", ""))
    if scope == "controller_state_at_time":
        return "state scaffold"
    if scope == "controller_interval_execution_budget":
        return "160 states"
    if scope == "seed_prep_only":
        return "seed prep"
    if scope == "per_step_evolution_only":
        return "dt only"
    if scope == "seed_plus_one_step_additive":
        return "seed + dt add"
    if scope == "full_horizon_with_seed_prep":
        return "t=8 full"
    return str(row.get("basis", scope))


def _write_overlay_pdf(path: Path, payload: Mapping[str, Any]) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = payload.get("parameter_manifest", {})
    methods = payload.get("methods", {})
    scoreboard = list(payload.get("scoreboard", []))
    hardware_rows = list(payload.get("hardware_cost_rows", []))
    hardware_report_rows = list(payload.get("hardware_report_rows", hardware_rows))
    circuit_artifacts = list(payload.get("compiled_circuit_artifacts", []))
    metric_definitions = dict(payload.get("hardware_metric_definitions", {}))
    with PdfPages(str(output_path)) as pdf:
        fig = plt.figure(figsize=(8.5, 11.0))
        lines = _wrap_lines(
            [
                "HH L2 t=8 Suzuki/controller/exact overlay",
                "Parameter manifest",
                "",
                *[f"- {key}: {_line_value(value)}" for key, value in dict(manifest).items()],
            ],
            width=110,
        )
        fig.text(0.07, 0.95, "\n".join(lines), ha="left", va="top", fontsize=7.8, family="monospace")
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11.0, 8.5))
        ax.axis("off")
        energy_columns = [
            "method",
            "final E",
            "final |dE|",
            "mean |dE|",
            "max |dE|",
        ]
        energy_text = []
        for row in scoreboard:
            energy_text.append(
                [
                    str(row.get("method")),
                    _fmt_float(row.get("final_energy_total")),
                    _fmt_float(row.get("final_abs_energy_total_error")),
                    _fmt_float(row.get("mean_abs_energy_total_error")),
                    _fmt_float(row.get("max_abs_energy_total_error")),
                ]
            )
        energy_table = ax.table(
            cellText=energy_text,
            colLabels=energy_columns,
            bbox=[0.10, 0.74, 0.80, 0.17],
            cellLoc="center",
        )
        energy_table.auto_set_font_size(False)
        energy_table.set_fontsize(8.0)

        state_rows = [
            row for row in hardware_report_rows if str(row.get("group", "")) == "state_at_time"
        ]

        state_columns = ["method", "state-at-time basis", "2Q", "depth", "size"]
        state_text = []
        for row in state_rows:
            state_text.append(
                [
                    str(row.get("method")),
                    _pdf_hardware_basis(row),
                    _line_value(row.get("compiled_count_2q")),
                    _line_value(row.get("compiled_depth")),
                    _line_value(row.get("compiled_size")),
                ]
            )
        state_table = ax.table(
            cellText=state_text,
            colLabels=state_columns,
            bbox=[0.14, 0.39, 0.72, 0.28],
            cellLoc="center",
        )
        state_table.auto_set_font_size(False)
        state_table.set_fontsize(8.0)

        ax.set_title("Energy scoreboard and hardware cost basis", fontsize=14, pad=20)
        ax.text(
            0.14,
            0.31,
            "Primary hardware comparison: controller state scaffold vs Suzuki seed prep plus one dt evolution.",
            ha="left",
            va="top",
            fontsize=8.2,
            transform=ax.transAxes,
        )
        ax.text(
            0.14,
            0.24,
            "The bare Suzuki dt row is diagnostic only; seed + dt is the apples-to-apples state-at-time cost.",
            ha="left",
            va="top",
            fontsize=8.2,
            transform=ax.transAxes,
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11.0, 8.5))
        style = {
            "exact": {"color": "black", "linewidth": 2.4, "linestyle": "-"},
            "controller": {"color": "#1f77b4", "linewidth": 1.8, "linestyle": "-"},
            "suzuki1": {"color": "#d95f02", "linewidth": 1.6, "linestyle": "--"},
            "suzuki2": {"color": "#1b9e77", "linewidth": 1.6, "linestyle": "-."},
        }
        for name in ("exact", "controller", "suzuki1", "suzuki2"):
            data = methods.get(name, {})
            traj = data.get("trajectory", []) if isinstance(data, Mapping) else []
            if not traj:
                continue
            times = np.asarray([float(row["time"]) for row in traj], dtype=float)
            energy = np.asarray([float(row["energy_total"]) for row in traj], dtype=float)
            ax.plot(times, energy, label=name, **style.get(name, {}))
        ax.set_xlabel("time")
        ax.set_ylabel("instantaneous total energy")
        ax.set_title("Energy overlay: Suzuki over controller over exact")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11.0, 8.5))
        for name in ("controller", "suzuki1", "suzuki2"):
            data = methods.get(name, {})
            traj = data.get("trajectory", []) if isinstance(data, Mapping) else []
            if not traj:
                continue
            times = np.asarray([float(row["time"]) for row in traj], dtype=float)
            error = np.asarray([float(row["abs_energy_total_error"]) for row in traj], dtype=float)
            ax.plot(times, error, label=name, **style.get(name, {}))
        ax.set_xlabel("time")
        ax.set_ylabel("|energy - exact|")
        ax.set_title("Energy error against exact reference")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.5, 11.0))
        circuit_artifact_lines: list[str] = []
        circuit_dirs = [
            str(Path(str(item.get("qpy_path"))).parent)
            for item in circuit_artifacts
            if isinstance(item, Mapping) and item.get("qpy_path") not in {None, ""}
        ]
        common_circuit_dir = circuit_dirs[0] if circuit_dirs else None
        if common_circuit_dir is not None:
            circuit_artifact_lines.append(f"dir: {_display_path(common_circuit_dir)}")
        for item in circuit_artifacts:
            if not isinstance(item, Mapping):
                continue
            stem = str(item.get("stem", item.get("scope", "unknown")))
            qpy_name = Path(str(item.get("qpy_path"))).name if item.get("qpy_path") not in {None, ""} else "n/a"
            ops_name = Path(str(item.get("ops_jsonl_path"))).name if item.get("ops_jsonl_path") not in {None, ""} else "n/a"
            preview_name = Path(str(item.get("preview_text_path"))).name if item.get("preview_text_path") not in {None, ""} else "n/a"
            circuit_artifact_lines.extend(
                [
                    f"- {stem}",
                    f"  qpy: {qpy_name}",
                    f"  ops: {ops_name}",
                    f"  preview: {preview_name}",
                ]
            )
        appendix = _wrap_lines(
            [
                "Appendix: source and execution contract",
                "",
                f"controller_json: {payload.get('source', {}).get('controller_json')}",
                f"source_pdf: {payload.get('source', {}).get('source_pdf')}",
                f"seed_artifact_json: {payload.get('source', {}).get('artifact_json')}",
                f"command: {payload.get('command')}",
                "",
                "Circuit-cost basis:",
                f"- compiled_count_2q: {metric_definitions.get('compiled_count_2q', 'n/a')}",
                f"- compiled_depth: {metric_definitions.get('compiled_depth', 'n/a')}",
                f"- compiled_size: {metric_definitions.get('compiled_size', 'n/a')}",
                f"- controller_final_scaffold_source: {metric_definitions.get('controller_final_scaffold_source', 'n/a')}",
                f"- controller_per_time_step: {metric_definitions.get('controller_per_time_step', 'n/a')}",
                f"- controller_interval_execution_budget: {metric_definitions.get('controller_interval_execution_budget', 'n/a')}",
                f"- seed_prep_only: {metric_definitions.get('seed_prep_only', 'n/a')}",
                f"- seed_plus_one_step_additive: {metric_definitions.get('seed_plus_one_step_additive', 'n/a')}",
                f"- one_step_with_seed_prep: {metric_definitions.get('one_step_with_seed_prep', 'n/a')}",
                f"- full_horizon_with_seed_prep: {metric_definitions.get('full_horizon_with_seed_prep', 'n/a')}",
                f"- per_step_evolution_only: {metric_definitions.get('per_step_evolution_only', 'n/a')}",
                "- Missing drive labels are not inserted; this report path fails instead.",
                "",
                "Compiled circuit artifacts:",
                *circuit_artifact_lines,
            ],
            width=98,
        )
        fig.text(0.07, 0.95, "\n".join(appendix), ha="left", va="top", fontsize=6.8, family="monospace")
        pdf.savefig(fig)
        plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an HH Suzuki/controller/exact overlay PDF from a controller JSON."
    )
    parser.add_argument("--controller-json", type=Path, default=DEFAULT_CONTROLLER_JSON)
    parser.add_argument("--source-pdf", type=Path, default=DEFAULT_SOURCE_PDF)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-pdf", type=Path, default=None)
    parser.add_argument("--trotter-steps", type=int, default=None)
    parser.add_argument("--suzuki-orders", type=str, default="1,2")
    parser.add_argument("--compile-backend-name", type=str, default=None)
    parser.add_argument("--compile-seed-transpiler", type=int, default=None)
    parser.add_argument("--compile-optimization-level", type=int, default=None)
    parser.add_argument("--compile-preferred-fake-backends", type=str, default=None)
    parser.add_argument("--export-compiled-circuits", action="store_true")
    parser.add_argument("--compiled-circuit-dir", type=Path, default=None)
    parser.add_argument("--skip-pdf", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace, source_payload: Mapping[str, Any]) -> SuzukiOverlayConfig:
    defaults = _source_compile_defaults(source_payload)
    preferred = _parse_string_tuple(args.compile_preferred_fake_backends)
    if not preferred:
        preferred = tuple(defaults["preferred_fake_backends"])
    return SuzukiOverlayConfig(
        controller_json=Path(args.controller_json),
        output_json=None if args.output_json is None else Path(args.output_json),
        output_pdf=None if args.output_pdf is None else Path(args.output_pdf),
        source_pdf=Path(args.source_pdf),
        trotter_steps=None if args.trotter_steps is None else int(args.trotter_steps),
        suzuki_orders=_parse_int_tuple(args.suzuki_orders),
        backend_name=str(args.compile_backend_name or defaults["backend_name"]),
        seed_transpiler=int(args.compile_seed_transpiler if args.compile_seed_transpiler is not None else defaults["seed_transpiler"]),
        optimization_level=int(args.compile_optimization_level if args.compile_optimization_level is not None else defaults["optimization_level"]),
        preferred_fake_backends=tuple(str(x) for x in preferred),
        export_compiled_circuits=bool(args.export_compiled_circuits),
        compiled_circuit_dir=None if args.compiled_circuit_dir is None else Path(args.compiled_circuit_dir),
        skip_pdf=bool(args.skip_pdf),
    )


def run_overlay(config: SuzukiOverlayConfig, *, command: str = "") -> dict[str, Any]:
    source_payload = _load_source_payload(config.controller_json)
    source_rows = _state_sample_rows(source_payload)
    times = _source_times(source_payload, source_rows)
    output_json_default, output_pdf_default = _default_output_paths(config.controller_json, source_payload)
    output_json = output_json_default if config.output_json is None else Path(config.output_json)
    output_pdf = output_pdf_default if config.output_pdf is None else Path(config.output_pdf)
    if bool(config.skip_pdf):
        output_pdf = None
    compiled_circuit_dir = None
    if bool(config.export_compiled_circuits):
        compiled_circuit_dir = (
            Path(config.compiled_circuit_dir)
            if config.compiled_circuit_dir is not None
            else Path(output_json).parent / "compiled_circuits"
        )
    trotter_steps = int(config.trotter_steps if config.trotter_steps is not None else int(times.size) - 1)
    dt = _uniform_dt(times, trotter_steps)
    context = _rebuild_context(source_payload)
    drive_t0 = float((context.drive_profile or {}).get("t0", 0.0))
    drive_sampling = str((context.drive_profile or {}).get("time_sampling", "midpoint"))
    physical_times = _source_physical_times(source_rows, fallback_drive_t0=float(drive_t0))
    exact_energy = [float(row["energy_total_exact"]) for row in source_rows]
    results = [
        _simulate_suzuki_order(
            order=int(order),
            psi_initial=context.psi_initial,
            times=times,
            exact_energy_total=exact_energy,
            observation_physical_times=physical_times,
            ordered_labels_exyz=context.ordered_labels_exyz,
            coeff_map_exyz=context.coeff_map_exyz,
            hmat_static=context.hmat,
            drive_coeff_provider_exyz=context.drive_coeff_provider_exyz,
            drive_t0=float(drive_t0),
            drive_time_sampling=str(drive_sampling),
            nq=int(context.nq),
        )
        for order in config.suzuki_orders
    ]
    cost_rows, raw_compile_rows = _compile_cost_rows(
        config=config,
        source_payload=source_payload,
        context=context,
        times=times,
        trotter_steps=int(trotter_steps),
        dt=float(dt),
        compiled_circuit_dir=compiled_circuit_dir,
    )
    payload = _build_overlay_payload(
        config=config,
        source_payload=source_payload,
        context=context,
        source_rows=source_rows,
        times=times,
        trotter_steps=int(trotter_steps),
        results=results,
        cost_rows=cost_rows,
        raw_compile_rows=raw_compile_rows,
        output_json=output_json,
        output_pdf=output_pdf,
        command=command,
    )
    payload["written"] = {
        "output_json": str(output_json),
        "output_pdf": None if output_pdf is None else str(output_pdf),
        "compiled_circuit_dir": None if compiled_circuit_dir is None else str(compiled_circuit_dir),
    }
    _write_overlay_json(output_json, payload)
    if output_pdf is not None:
        _write_overlay_pdf(output_pdf, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    source_payload = _load_source_payload(Path(args.controller_json))
    config = _config_from_args(args, source_payload)
    command = "python -m pipelines.time_dynamics.legacy.analysis.hh_realtime_suzuki_overlay"
    if argv is None:
        command = " ".join([sys.executable, "-m", "pipelines.time_dynamics.legacy.analysis.hh_realtime_suzuki_overlay", *sys.argv[1:]])
    else:
        command = " ".join(["python", "-m", "pipelines.time_dynamics.legacy.analysis.hh_realtime_suzuki_overlay", *map(str, argv)])
    payload = run_overlay(config, command=command)
    written = payload.get("written", {})
    print(f"overlay_json={written.get('output_json')}")
    if written.get("output_pdf") is not None:
        print(f"overlay_pdf={written.get('output_pdf')}")
    if written.get("compiled_circuit_dir") is not None:
        print(f"compiled_circuit_dir={written.get('compiled_circuit_dir')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
