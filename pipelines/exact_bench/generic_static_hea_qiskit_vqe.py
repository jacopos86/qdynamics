#!/usr/bin/env python3
"""Generic static Qiskit HEA VQE benchmark runner.

This is an exact-bench-local external fixed-ansatz benchmark.  It does not call
the Phase3 ADAPT controller and does not use exact target data during optimizer
decisions.  Exact references are resolved only after the VQE optimizer returns.
"""

from __future__ import annotations

import importlib
import json
import math
import time
from argparse import Namespace
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.exact_bench.benchmark_decision_noise import (
    BenchmarkDecisionNoiseConfig,
    BenchmarkDecisionNoiseRecorder,
    coerce_config as coerce_benchmark_decision_noise_config,
    copy_decision_noise_metadata,
)
from pipelines.exact_bench.benchmark_metrics_proxy import write_proxy_sidecars
from pipelines.exact_bench.comparator_provenance import comparator_source_fields
from pipelines.exact_bench.molecular_vibronic_h2_fixture_override import (
    with_molecular_vibronic_h2_fixture_override,
)
from pipelines.exact_bench.paper_i_main_tables_spsa_profile import (
    PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID,
    normalize_paper_i_main_tables_spsa_profile,
)
from pipelines.exact_bench.qiskit_hea_adapter import (
    QiskitHeaUnavailable,
    build_qiskit_hea_ansatz,
    has_qiskit_hea_support,
)
from pipelines.static_adapt.builders.problem_registry import (
    ProblemRequest,
    ResolvedProblemContext,
    resolve_problem_context,
)
from pipelines.static_adapt.numerical_physical_integrity import (
    sector_probability,
)
from pipelines.exact_bench.static_benchmark_runtime import (
    HamiltonianBenchmarkSpec,
)
from pipelines.exact_bench.table_i_canonical_cases import (
    table_i_canonical_case_ids,
    table_i_canonical_spec_by_case_id,
)
from src.quantum.vqe_latex_python_pairs import VQEResult, expval_pauli_polynomial, vqe_minimize

SCHEMA_VERSION = "generic_static_hea_qiskit_vqe_v1"
_METHOD_ID = "static_hea_qiskit_vqe"
_RUNNER_MODULE = "pipelines.exact_bench.generic_static_hea_qiskit_vqe"
_DEFAULT_SHOTS_PER_PAULI_TERM_PROXY = 1024
_HEA_SHOT_PROXY_FORMULA = (
    "shots_total = shots_per_pauli_term_proxy * hamiltonian_pauli_term_count * energy_eval_count_proxy"
)
_COMPILED_BASIS_GATES = (
    "id",
    "x",
    "sx",
    "rx",
    "ry",
    "rz",
    "h",
    "s",
    "sdg",
    "cx",
    "cz",
)
_QISKIT_ALGORITHMS_SPSA_OPTIMIZER = "qiskit_algorithms.optimizers.SPSA"


def _positive_int(value: int | str | None, *, field: str) -> int:
    try:
        out = int(value)  # type: ignore[arg-type]
    except Exception as exc:
        raise ValueError(f"{field} must be a positive integer; got {value!r}.") from exc
    if out < 1:
        raise ValueError(f"{field} must be a positive integer; got {value!r}.")
    return int(out)


def _blank_to_none(value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def _positive_float(value: Any, *, field: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"{field} must be a positive finite float; got {value!r}.") from exc
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{field} must be a positive finite float; got {value!r}.")
    return float(out)


def _normalize_hea_optimizer_settings(
    *,
    optimizer: str,
    maxiter: int,
    seed: int,
    optimizer_profile: str | None = None,
    optimizer_profile_source: str | None = None,
    hea_optimizer: str | None = None,
    hea_spsa_maxiter: int | None = None,
    hea_spsa_seed: int | None = None,
    hea_spsa_learning_rate: float | str | None = None,
    hea_spsa_perturbation: float | str | None = None,
    optimizer_overlay_source: str | None = None,
) -> dict[str, Any]:
    learning_rate_raw = _blank_to_none(hea_spsa_learning_rate)
    perturbation_raw = _blank_to_none(hea_spsa_perturbation)
    profile = normalize_paper_i_main_tables_spsa_profile(optimizer_profile)
    if profile == PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID and hea_optimizer in {None, ""}:
        # Direct runner callers may select the profile without the generic-dispatch
        # overlay that supplies hea_optimizer=spsa.  Treat that profile as the
        # force-SPSA source rather than preserving the COBYLA default.
        requested = "spsa"
        optimizer_source = "optimizer_profile_default"
    else:
        requested = str(hea_optimizer if hea_optimizer not in {None, ""} else optimizer).strip()
        optimizer_source = "hea_optimizer" if hea_optimizer not in {None, ""} else "optimizer_arg"
    key = requested.lower()
    overlay_requested = profile is not None or hea_optimizer not in {None, ""}
    if profile == PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID and key != "spsa":
        raise ValueError(f"optimizer_profile={profile} requires HEA optimizer=spsa; got {requested!r}.")
    if overlay_requested and key not in {"cobyla", "spsa"}:
        raise ValueError(f"HEA optimizer overlay must be one of {{cobyla, spsa}}; got {requested!r}.")
    if (learning_rate_raw is None) != (perturbation_raw is None):
        raise ValueError(
            "HEA Qiskit SPSA schedule requires hea_spsa_learning_rate and "
            "hea_spsa_perturbation to be provided together."
        )
    if learning_rate_raw is not None and key != "spsa":
        raise ValueError(
            "HEA Qiskit SPSA schedule fields require HEA optimizer=spsa; "
            f"got {requested!r}."
        )
    effective_learning_rate: float | None = None
    effective_perturbation: float | None = None
    if key == "spsa":
        effective_maxiter = _positive_int(
            hea_spsa_maxiter if hea_spsa_maxiter is not None else maxiter,
            field="hea_spsa_maxiter" if hea_spsa_maxiter is not None else "maxiter",
        )
        effective_seed = _positive_int(
            hea_spsa_seed if hea_spsa_seed is not None else seed,
            field="hea_spsa_seed" if hea_spsa_seed is not None else "seed",
        )
        actual_optimizer = _QISKIT_ALGORITHMS_SPSA_OPTIMIZER
        if learning_rate_raw is not None:
            effective_learning_rate = _positive_float(learning_rate_raw, field="hea_spsa_learning_rate")
            effective_perturbation = _positive_float(perturbation_raw, field="hea_spsa_perturbation")
    else:
        effective_maxiter = _positive_int(maxiter, field="maxiter")
        effective_seed = _positive_int(seed, field="seed")
        actual_optimizer = requested.upper() if key == "cobyla" else requested
    return {
        "optimizer_kind": key,
        "optimizer": actual_optimizer,
        "optimizer_requested": requested,
        "optimizer_source": optimizer_source,
        "optimizer_profile": profile,
        "optimizer_profile_source": optimizer_profile_source if profile is not None else None,
        "optimizer_overlay_source": optimizer_overlay_source,
        "maxiter": int(effective_maxiter),
        "seed": int(effective_seed),
        "hea_spsa_maxiter": int(effective_maxiter) if key == "spsa" else None,
        "hea_spsa_seed": int(effective_seed) if key == "spsa" else None,
        "hea_spsa_learning_rate": effective_learning_rate,
        "hea_spsa_perturbation": effective_perturbation,
    }


def _load_qiskit_algorithms_spsa_class() -> Any:
    try:
        module = importlib.import_module("qiskit_algorithms.optimizers")
    except Exception as exc:
        raise RuntimeError(
            "HEA optimizer=SPSA requires qiskit_algorithms.optimizers.SPSA; "
            f"failed to import qiskit_algorithms.optimizers ({type(exc).__name__}: {exc})."
        ) from exc
    cls = getattr(module, "SPSA", None)
    if cls is None:
        raise RuntimeError("HEA optimizer=SPSA requires qiskit_algorithms.optimizers.SPSA; SPSA is missing.")
    return cls


def _set_qiskit_algorithms_seed(seed: int) -> bool:
    try:
        module = importlib.import_module("qiskit_algorithms.utils")
        algorithm_globals = getattr(module, "algorithm_globals", None)
        if algorithm_globals is None:
            return False
        setattr(algorithm_globals, "random_seed", int(seed))
        return True
    except Exception:
        return False


def _make_qiskit_algorithms_spsa(
    *,
    maxiter: int,
    learning_rate: float | None = None,
    perturbation: float | None = None,
) -> Any:
    cls = _load_qiskit_algorithms_spsa_class()
    if (learning_rate is None) != (perturbation is None):
        raise ValueError("Qiskit SPSA learning_rate and perturbation must be supplied together.")
    kwargs: dict[str, Any] = {"maxiter": int(maxiter)}
    if learning_rate is not None:
        kwargs["learning_rate"] = float(learning_rate)
        kwargs["perturbation"] = float(perturbation)
    try:
        optimizer = cls(**kwargs)
    except Exception as exc:
        schedule_suffix = ""
        if learning_rate is not None:
            schedule_suffix = (
                f", learning_rate={float(learning_rate)}, "
                f"perturbation={float(perturbation)}"
            )
        raise RuntimeError(
            "HEA optimizer=SPSA found qiskit_algorithms.optimizers.SPSA but could not construct it "
            f"with maxiter={int(maxiter)}{schedule_suffix} "
            f"({type(exc).__name__}: {exc})."
        ) from exc
    if not callable(getattr(optimizer, "minimize", None)):
        raise RuntimeError(
            "HEA optimizer=SPSA found qiskit_algorithms.optimizers.SPSA but the constructed object "
            "does not expose minimize(fun, x0, jac=None, bounds=None)."
        )
    return optimizer


def _qiskit_algorithms_spsa_vqe_minimize(
    H: Any,
    ansatz: Any,
    psi_ref: np.ndarray,
    *,
    restarts: int,
    seed: int,
    maxiter: int,
    learning_rate: float | None = None,
    perturbation: float | None = None,
    objective_value_transform: Any | None = None,
) -> VQEResult:
    npar = int(getattr(ansatz, "num_parameters", 0))
    if npar <= 0:
        raise ValueError("HEA optimizer=SPSA requires an ansatz with at least one parameter.")
    restart_count = max(1, int(restarts))
    rng = np.random.default_rng(int(seed))
    _set_qiskit_algorithms_seed(int(seed))
    bounds = [(-math.pi, math.pi) for _ in range(npar)]
    best_theta: np.ndarray | None = None
    best_energy = float("inf")
    best_success = False
    best_message = "no SPSA restart executed"
    best_restart = -1
    total_nfev = 0
    total_nit = 0
    restart_summaries: list[dict[str, Any]] = []

    for restart_index in range(restart_count):
        x0 = np.clip(rng.normal(loc=0.0, scale=0.3, size=npar), -math.pi, math.pi)
        restart_nfev = 0

        def objective(theta_vec: np.ndarray) -> float:
            nonlocal restart_nfev
            restart_nfev += 1
            theta = np.asarray(theta_vec, dtype=float).reshape(-1)
            psi = np.asarray(ansatz.prepare_state(theta, psi_ref), dtype=complex).reshape(-1)
            energy_ideal = float(expval_pauli_polynomial(psi, H))
            if objective_value_transform is not None:
                return float(
                    objective_value_transform(
                        {
                            "energy_ideal": energy_ideal,
                            "restart_index": int(restart_index + 1),
                            "nfev_restart": int(restart_nfev),
                            "nfev_total_estimate": int(total_nfev + restart_nfev),
                            "progress_label": "qiskit_algorithms_spsa",
                        }
                    )
                )
            return energy_ideal

        if learning_rate is None and perturbation is None:
            optimizer = _make_qiskit_algorithms_spsa(maxiter=int(maxiter))
        else:
            optimizer = _make_qiskit_algorithms_spsa(
                maxiter=int(maxiter),
                learning_rate=float(learning_rate),
                perturbation=float(perturbation),
            )
        try:
            result = optimizer.minimize(fun=objective, x0=x0, bounds=bounds)
        except TypeError as exc:
            raise RuntimeError(
                "HEA optimizer=SPSA requires qiskit_algorithms.optimizers.SPSA.minimize"
                "(fun, x0, jac=None, bounds=None); installed API rejected the expected call."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"HEA optimizer=SPSA failed during qiskit_algorithms.optimizers.SPSA.minimize: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not hasattr(result, "x"):
            raise RuntimeError("HEA optimizer=SPSA returned an incompatible result without an x field.")
        theta = np.asarray(getattr(result, "x"), dtype=float).reshape(-1)
        if int(theta.size) != npar:
            raise RuntimeError(
                f"HEA optimizer=SPSA returned theta with length {theta.size}; expected {npar}."
            )
        fun = getattr(result, "fun", None)
        decision_energy = float(objective(theta) if fun is None else fun)
        nfev = max(int(restart_nfev), int(getattr(result, "nfev", 0) or 0))
        nit = int(getattr(result, "nit", maxiter) or 0)
        success = bool(getattr(result, "success", math.isfinite(decision_energy)))
        message = str(getattr(result, "message", "qiskit_algorithms_spsa"))
        total_nfev += int(nfev)
        total_nit += int(nit)
        restart_summaries.append(
            {
                "restart_index": int(restart_index + 1),
                "energy": float(decision_energy),
                "success": bool(success),
                "message": message,
                "nfev": int(nfev),
                "nit": int(nit),
            }
        )
        if decision_energy < best_energy:
            best_theta = np.array(theta, copy=True)
            best_energy = float(decision_energy)
            best_success = bool(success)
            best_message = message
            best_restart = int(restart_index)

    if best_theta is None:
        raise RuntimeError("HEA optimizer=SPSA did not produce any restart result.")
    return VQEResult(
        energy=float(best_energy),
        theta=np.asarray(best_theta, dtype=float).reshape(-1),
        success=bool(best_success),
        message=best_message,
        nfev=int(total_nfev),
        nit=int(total_nit),
        best_restart=int(best_restart),
        restart_summaries=restart_summaries,
        optimizer_memory={
            "version": "generic_static_hea_qiskit_algorithms_spsa_v1",
            "optimizer": _QISKIT_ALGORITHMS_SPSA_OPTIMIZER,
            "seed": int(seed),
            "maxiter": int(maxiter),
            "restarts": int(restart_count),
        },
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        try:
            return dict(value.__dict__)
        except Exception:
            return str(value)
    return str(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_static_hea_case_ids(family: str) -> tuple[str, ...]:
    """Return canonical Paper-I Table-I HEA case IDs for a problem family."""
    return table_i_canonical_case_ids(family)


def _spec_by_case_id(family: str, case_id: str) -> HamiltonianBenchmarkSpec:
    family_key = str(family).strip()
    return with_molecular_vibronic_h2_fixture_override(
        table_i_canonical_spec_by_case_id(family_key, case_id),
        family=family_key,
    )


def _namespace_from_base_args(argv: Sequence[str]) -> Namespace:
    defaults: dict[str, Any] = {
        "problem": "hubbard",
        "L": 2,
        "t": 1.0,
        "u": 4.0,
        "dv": 0.0,
        "omega0": 1.0,
        "g_ep": 0.5,
        "n_ph_max": 1,
        "boson_encoding": "binary",
        "ordering": "blocked",
        "boundary": "periodic",
        "include_zero_point": True,
        "molecular_problem_json": None,
        "molecular_vibronic_h2_fixture_json": None,
        "v_nn": 0.0,
        "t_prime": 0.0,
        "n_fermions": None,
    }
    key_map = {
        "--problem": "problem",
        "--L": "L",
        "--t": "t",
        "--u": "u",
        "--dv": "dv",
        "--omega0": "omega0",
        "--g-ep": "g_ep",
        "--n-ph-max": "n_ph_max",
        "--boson-encoding": "boson_encoding",
        "--ordering": "ordering",
        "--boundary": "boundary",
        "--molecular-problem-json": "molecular_problem_json",
        "--molecular-vibronic-h2-fixture-json": "molecular_vibronic_h2_fixture_json",
        "--v-nn": "v_nn",
        "--t-prime": "t_prime",
        "--n-fermions": "n_fermions",
    }
    int_keys = {"L", "n_ph_max", "n_fermions"}
    float_keys = {"t", "u", "dv", "omega0", "g_ep", "v_nn", "t_prime"}
    values = dict(defaults)
    idx = 0
    argv_tuple = tuple(str(x) for x in argv)
    while idx < len(argv_tuple):
        token = argv_tuple[idx]
        if token == "--include-zero-point":
            values["include_zero_point"] = True
            idx += 1
            continue
        if token == "--no-include-zero-point":
            values["include_zero_point"] = False
            idx += 1
            continue
        if token not in key_map:
            idx += 1
            continue
        if idx + 1 >= len(argv_tuple):
            raise ValueError(f"Missing value for {token}")
        key = key_map[token]
        raw = argv_tuple[idx + 1]
        if key in int_keys and raw not in {"", "None", "none"}:
            values[key] = int(raw)
        elif key in float_keys:
            values[key] = float(raw)
        elif key == "n_fermions" and raw in {"", "None", "none"}:
            values[key] = None
        else:
            values[key] = raw
        idx += 2
    return Namespace(**values)


def _resolve_context_from_spec(spec: HamiltonianBenchmarkSpec) -> ResolvedProblemContext:
    request = ProblemRequest.from_namespace(_namespace_from_base_args(spec.base_pipeline_args))
    return resolve_problem_context(request)


def _safe_exact_energy(context: ResolvedProblemContext) -> float | None:
    try:
        return float(context.exact_target.resolve_energy(ai_log=None))
    except TypeError:
        try:
            return float(context.exact_target.resolve_energy())
        except Exception:
            return None
    except Exception:
        return None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _source_fields(**overrides: Any) -> dict[str, Any]:
    return comparator_source_fields(_METHOD_ID, runner_module=_RUNNER_MODULE, **overrides)


def _skip_payload(*, family: str, case_id: str, output_dir: Path, reason: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    finished = _utc_now()
    row = {
        "run_id": f"{case_id}::{_METHOD_ID}",
        "hamiltonian_id": case_id,
        "case_id": case_id,
        "family": family,
        "problem": family,
        "status": "skipped_optional_dependency",
        "method_id": _METHOD_ID,
        "method_kind": "fixed_ansatz_vqe",
        "ansatz_name": "qiskit_hea_linear_ryrz_cx",
        "algorithm_origin": "external_fixed_ansatz_qiskit_hea",
        "qiskit_available": False,
        "reason": reason,
        "uses_exact_for_decision": False,
        "exact_reference_usage": "not_resolved_for_dependency_skip",
        "phase3_controller_called": False,
        "qiskit_boundary": "pipelines.exact_bench_only",
        **_source_fields(),
    }
    payload = {
        "schema": SCHEMA_VERSION,
        "family": family,
        "case_id": case_id,
        "algorithm_id": _METHOD_ID,
        "method_id": _METHOD_ID,
        "status": "skipped_optional_dependency",
        "qiskit_available": False,
        "reason": reason,
        "table_i": {
            "tex_label": "tab:static_claims",
            "static_train_suite": True,
            "sweep_complete": False,
        },
        "guardrails": {
            "uses_exact_for_decision": False,
            "exact_reference_usage": "not_resolved_for_dependency_skip",
            "phase3_controller_called": False,
            "qiskit_boundary": "pipelines.exact_bench_only",
        },
        "comparator_source": _source_fields(),
        "rows": [row],
        "finished_utc": finished,
    }
    _write_json(output_dir / "result.json", payload)
    _write_json(output_dir / "rows.json", {"schema": f"{SCHEMA_VERSION}_rows", "rows": [row]})
    _write_json(
        output_dir / "manifest.json",
        {"schema": f"{SCHEMA_VERSION}_manifest", **{k: v for k, v in payload.items() if k != "schema"}},
    )
    _write_json(output_dir / "generic_static_single.json", payload)
    write_proxy_sidecars([row], output_dir, summary_extras={"schema_source": SCHEMA_VERSION})
    return payload


def _circuit_stats_from_qiskit_circuit(circuit: Any) -> dict[str, Any]:
    try:
        depth = int(circuit.depth())
    except Exception:
        depth = None
    try:
        op_counts = {str(k): int(v) for k, v in dict(circuit.count_ops()).items()}
    except Exception:
        op_counts = {}
    count_2q = 0
    try:
        for item in circuit.data:
            operation = getattr(item, "operation", None)
            if operation is None and isinstance(item, (tuple, list)) and item:
                operation = item[0]
            if int(getattr(operation, "num_qubits", 0)) == 2:
                count_2q += 1
    except Exception:
        count_2q = int(op_counts.get("cx", 0) + op_counts.get("cz", 0))
    try:
        from pipelines.qiskit_backend_tools import safe_two_qubit_depth

        depth_2q = int(safe_two_qubit_depth(circuit))
    except Exception:
        depth_2q = None
    return {"depth": depth, "depth_2q": depth_2q, "count_2q": int(count_2q), "op_counts": op_counts}


def _compiled_circuit_stats(circuit: Any | None) -> dict[str, Any]:
    empty = {
        "compiled_depth_total": None,
        "compiled_depth_2q_total": None,
        "compiled_depth_2q_semantics": None,
        "compiled_count_2q_total": None,
        "compiled_op_counts": None,
        "compiled_circuit_stats_status": "not_available",
        "compiled_circuit_stats_error": None,
        "compiled_basis_gates": list(_COMPILED_BASIS_GATES),
        "first_hit_cost_source_kind": "qiskit_compiled_terminal_only_fixed_ansatz",
        "compiled_resource_source_kind": "qiskit_compiled_terminal_only_fixed_ansatz",
        "compiled_resource_qiskit_validated": False,
        "qiskit_first_hit_cost_validated": False,
    }
    if circuit is None:
        out = dict(empty)
        out["compiled_circuit_stats_status"] = "not_available_no_circuit"
        return out
    try:
        from qiskit import transpile
    except Exception as exc:  # pragma: no cover - optional-dep failure varies
        out = dict(empty)
        out.update(
            {
                "compiled_circuit_stats_status": "qiskit_transpile_unavailable",
                "compiled_circuit_stats_error": str(exc),
            }
        )
        return out
    try:
        try:
            decomposed = circuit.decompose(reps=10)
        except Exception:
            decomposed = circuit
        compiled = transpile(
            decomposed,
            basis_gates=list(_COMPILED_BASIS_GATES),
            optimization_level=0,
        )
        stats = _circuit_stats_from_qiskit_circuit(compiled)
        return {
            "compiled_depth_total": stats["depth"],
            "compiled_depth_2q_total": stats["depth_2q"],
            "compiled_count_2q_total": stats["count_2q"],
            "compiled_op_counts": stats["op_counts"],
            "compiled_circuit_stats_status": "ok",
            "compiled_circuit_stats_error": None,
            "compiled_basis_gates": list(_COMPILED_BASIS_GATES),
            "first_hit_cost_source_kind": "qiskit_compiled_terminal_only_fixed_ansatz",
            "compiled_resource_source_kind": "qiskit_compiled_terminal_only_fixed_ansatz",
            "compiled_resource_qiskit_validated": True,
            "qiskit_first_hit_cost_validated": False,
            "compiled_depth_2q_semantics": "qiskit_compiled_two_qubit_layer_depth_ansatz_circuit",
            "depth_2q_semantics": "qiskit_compiled_two_qubit_layer_depth_ansatz_circuit",
        }
    except Exception as exc:
        out = dict(empty)
        out.update(
            {
                "compiled_circuit_stats_status": "failed",
                "compiled_circuit_stats_error": str(exc),
            }
        )
        return out


def _hamiltonian_pauli_term_count(hamiltonian: Any, *, tol: float = 1e-12) -> int:
    try:
        terms = list(hamiltonian.return_polynomial())
    except Exception:
        return 0
    seen: set[str] = set()
    for term in terms:
        try:
            coeff = complex(term.p_coeff)
            label = str(term.pw2strng()).lower()
        except Exception:
            continue
        if abs(coeff) <= float(tol):
            continue
        if not label or label == "e" * len(label):
            continue
        seen.add(label)
    return int(len(seen))


def _hea_shot_proxy_fields(
    *,
    hamiltonian_pauli_term_count: int,
    energy_eval_count: int | None,
    shots_per_pauli_term_proxy: int = _DEFAULT_SHOTS_PER_PAULI_TERM_PROXY,
) -> dict[str, Any]:
    h_count = max(0, int(hamiltonian_pauli_term_count))
    energy_count = max(1, int(energy_eval_count or 0))
    shots_per_term = max(0, int(shots_per_pauli_term_proxy))
    return {
        "shots_total": int(shots_per_term * h_count * energy_count),
        "static_shot_estimate_status": "deterministic_proxy_not_physical_shots",
        "shot_proxy_formula": _HEA_SHOT_PROXY_FORMULA,
        "shot_proxy_note": "Benchmark-table deterministic proxy only; it is not a hardware shot allocation.",
        "shots_per_pauli_term_proxy": shots_per_term,
        "hamiltonian_pauli_term_count": h_count,
        "energy_eval_count_proxy": energy_count,
    }


def _failure_payload(
    *,
    family: str,
    case_id: str,
    output_dir: Path,
    reason: str,
    exception_type: str,
    qiskit_available: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    finished = _utc_now()
    row = {
        "run_id": f"{case_id}::{_METHOD_ID}",
        "hamiltonian_id": case_id,
        "case_id": case_id,
        "family": family,
        "problem": family,
        "status": "failed",
        "method_id": _METHOD_ID,
        "method_kind": "fixed_ansatz_vqe",
        "ansatz_name": "qiskit_hea_linear_ryrz_cx",
        "algorithm_origin": "external_fixed_ansatz_qiskit_hea",
        "qiskit_available": bool(qiskit_available),
        "reason": reason,
        "exception_type": exception_type,
        "uses_exact_for_decision": False,
        "exact_reference_usage": "reporting_only_after_optimization_or_not_reached",
        "phase3_controller_called": False,
        "qiskit_boundary": "pipelines.exact_bench_only",
        **_source_fields(),
    }
    payload = {
        "schema": SCHEMA_VERSION,
        "family": family,
        "case_id": case_id,
        "algorithm_id": _METHOD_ID,
        "method_id": _METHOD_ID,
        "status": "failed",
        "qiskit_available": bool(qiskit_available),
        "reason": reason,
        "exception_type": exception_type,
        "table_i": {
            "tex_label": "tab:static_claims",
            "static_train_suite": True,
            "sweep_complete": False,
        },
        "guardrails": {
            "uses_exact_for_decision": False,
            "exact_reference_usage": "reporting_only_after_optimization_or_not_reached",
            "phase3_controller_called": False,
            "qiskit_boundary": "pipelines.exact_bench_only",
        },
        "comparator_source": _source_fields(),
        "rows": [row],
        "finished_utc": finished,
    }
    _write_json(output_dir / "result.json", payload)
    _write_json(output_dir / "rows.json", {"schema": f"{SCHEMA_VERSION}_rows", "rows": [row]})
    _write_json(
        output_dir / "manifest.json",
        {"schema": f"{SCHEMA_VERSION}_manifest", **{k: v for k, v in payload.items() if k != "schema"}},
    )
    _write_json(output_dir / "generic_static_single.json", payload)
    write_proxy_sidecars([row], output_dir, summary_extras={"schema_source": SCHEMA_VERSION})
    return payload


def _run_static_hea_qiskit_vqe_single_impl(
    *,
    family: str,
    case_id: str,
    output_dir: Path,
    reps: int = 2,
    restarts: int = 3,
    maxiter: int = 800,
    optimizer: str = "COBYLA",
    seed: int = 42,
    benchmark_decision_noise_config: BenchmarkDecisionNoiseConfig | Mapping[str, Any] | None = None,
    optimizer_profile: str | None = None,
    optimizer_profile_source: str | None = None,
    hea_optimizer: str | None = None,
    hea_spsa_maxiter: int | None = None,
    hea_spsa_seed: int | None = None,
    hea_spsa_learning_rate: float | str | None = None,
    hea_spsa_perturbation: float | str | None = None,
    optimizer_overlay_source: str | None = None,
) -> dict[str, Any]:
    """Run one generic Qiskit HEA VQE benchmark case."""
    family_key = str(family).strip()
    case_key = str(case_id).strip()
    decision_noise_config = coerce_benchmark_decision_noise_config(
        benchmark_decision_noise_config,
        family=family_key,
        case_id=case_key,
        algorithm_id=_METHOD_ID,
    )
    decision_noise_recorder = BenchmarkDecisionNoiseRecorder(
        decision_noise_config,
        base_scope={"family": family_key, "case_id": case_key, "algorithm_id": _METHOD_ID},
    )
    if case_key not in default_static_hea_case_ids(family_key):
        raise ValueError(f"static_hea_qiskit_vqe is not implemented for {family_key}/{case_key}")
    optimizer_settings = _normalize_hea_optimizer_settings(
        optimizer=optimizer,
        maxiter=int(maxiter),
        seed=int(seed),
        optimizer_profile=optimizer_profile,
        optimizer_profile_source=optimizer_profile_source,
        hea_optimizer=hea_optimizer,
        hea_spsa_maxiter=hea_spsa_maxiter,
        hea_spsa_seed=hea_spsa_seed,
        hea_spsa_learning_rate=hea_spsa_learning_rate,
        hea_spsa_perturbation=hea_spsa_perturbation,
        optimizer_overlay_source=optimizer_overlay_source,
    )
    output = Path(output_dir)
    started_utc = _utc_now()
    t0 = time.perf_counter()

    if not has_qiskit_hea_support():
        return _skip_payload(
            family=family_key,
            case_id=case_key,
            output_dir=output,
            reason="optional Qiskit HEA dependency is not importable",
        )

    spec = _spec_by_case_id(family_key, case_key)
    context = _resolve_context_from_spec(spec)
    psi_ref = np.asarray(context.reference_state.build_state(), dtype=complex).reshape(-1)
    try:
        ansatz = build_qiskit_hea_ansatz(num_qubits=int(context.layout.total_qubits), reps=int(reps))
    except QiskitHeaUnavailable as exc:
        return _skip_payload(family=family_key, case_id=case_key, output_dir=output, reason=str(exc))
    stats = ansatz.circuit_stats()

    def _objective_value_transform(event: Mapping[str, Any]) -> float:
        return float(
            decision_noise_recorder.apply(
                float(event["energy_ideal"]),
                surface="vqe_objective",
                value_kind="energy",
                phase="optimizer",
                extra_scope={
                    "restart_index": int(event.get("restart_index", 0) or 0),
                    "nfev_restart": int(event.get("nfev_restart", 0) or 0),
                    "nfev_total_estimate": int(event.get("nfev_total_estimate", 0) or 0),
                    "progress_label": str(event.get("progress_label") or ""),
                },
            )
        )

    # Exact references are intentionally not resolved until after this optimizer call.
    if optimizer_settings["optimizer_kind"] == "spsa":
        opt = _qiskit_algorithms_spsa_vqe_minimize(
            context.hamiltonian,
            ansatz,
            psi_ref,
            restarts=int(restarts),
            seed=int(optimizer_settings["seed"]),
            maxiter=int(optimizer_settings["maxiter"]),
            learning_rate=optimizer_settings["hea_spsa_learning_rate"],
            perturbation=optimizer_settings["hea_spsa_perturbation"],
            objective_value_transform=_objective_value_transform if bool(decision_noise_config.enabled) else None,
        )
    else:
        opt = vqe_minimize(
            context.hamiltonian,
            ansatz,
            psi_ref,
            restarts=int(restarts),
            seed=int(optimizer_settings["seed"]),
            maxiter=int(optimizer_settings["maxiter"]),
            method=str(optimizer_settings["optimizer"]),
            energy_backend="legacy",
            track_history=False,
            objective_value_transform=_objective_value_transform if bool(decision_noise_config.enabled) else None,
        )
    theta = np.asarray(opt.theta, dtype=float).reshape(-1)
    psi_final = np.asarray(ansatz.prepare_state(theta, psi_ref), dtype=complex).reshape(-1)
    exact_energy = _safe_exact_energy(context)
    optimizer_decision_energy = float(opt.energy)
    if bool(decision_noise_config.enabled):
        energy = float(expval_pauli_polynomial(psi_final, context.hamiltonian))
    else:
        energy = optimizer_decision_energy
    abs_delta = None if exact_energy is None else abs(float(energy) - float(exact_energy))
    decision_noise_metadata = None
    if bool(decision_noise_config.enabled):
        decision_noise_metadata = copy_decision_noise_metadata(
            decision_noise_recorder.summary(
                status="ok",
                supported=True,
                extra={"runner": "generic_static_hea_qiskit_vqe"},
            )
        )
    sector = sector_probability(context, psi_final)
    walltime = float(time.perf_counter() - t0)
    finished_utc = _utc_now()
    hamiltonian_pauli_term_count = _hamiltonian_pauli_term_count(context.hamiltonian)
    shot_proxy = _hea_shot_proxy_fields(
        hamiltonian_pauli_term_count=hamiltonian_pauli_term_count,
        energy_eval_count=int(opt.nfev),
    )
    compiled_stats = _compiled_circuit_stats(getattr(ansatz, "circuit", None))

    row = {
        "run_id": f"{case_key}::{_METHOD_ID}",
        "schema": SCHEMA_VERSION,
        "family": family_key,
        "problem": family_key,
        "L": int(context.request.num_sites),
        "hamiltonian_id": case_key,
        "case_id": case_key,
        "method_id": _METHOD_ID,
        "method_kind": "fixed_ansatz_vqe",
        "ansatz_name": ansatz.ansatz_name,
        "algorithm_origin": "external_fixed_ansatz_qiskit_hea",
        "status": "ok",
        "qiskit_available": True,
        "energy": energy,
        "exact_energy": exact_energy,
        "exact_gs_energy": exact_energy,
        "delta_E_abs": abs_delta,
        "abs_delta_e": abs_delta,
        "infidelity_exact": None,
        "infidelity_status": "not_available_exact_state_not_exposed_by_problem_context",
        "observable_error_status": "not_implemented_static_train_suite",
        **shot_proxy,
        "num_qubits": int(context.layout.total_qubits),
        "num_parameters": int(ansatz.num_parameters),
        "vqe_reps": int(reps),
        "vqe_restarts": int(restarts),
        "vqe_maxiter": int(optimizer_settings["maxiter"]),
        "optimizer": str(optimizer_settings["optimizer"]),
        "optimizer_kind": str(optimizer_settings["optimizer_kind"]),
        "optimizer_requested": str(optimizer_settings["optimizer_requested"]),
        "optimizer_source": str(optimizer_settings["optimizer_source"]),
        "optimizer_profile": optimizer_settings["optimizer_profile"],
        "optimizer_profile_source": optimizer_settings["optimizer_profile_source"],
        "optimizer_overlay_source": optimizer_settings["optimizer_overlay_source"],
        "hea_spsa_maxiter": optimizer_settings["hea_spsa_maxiter"],
        "hea_spsa_seed": optimizer_settings["hea_spsa_seed"],
        "hea_spsa_learning_rate": optimizer_settings["hea_spsa_learning_rate"],
        "hea_spsa_perturbation": optimizer_settings["hea_spsa_perturbation"],
        "spsa_maxiter": optimizer_settings["hea_spsa_maxiter"],
        "spsa_seed": optimizer_settings["hea_spsa_seed"],
        "spsa_learning_rate": optimizer_settings["hea_spsa_learning_rate"],
        "spsa_perturbation": optimizer_settings["hea_spsa_perturbation"],
        "optimizer_success": bool(opt.success),
        "optimizer_message": str(opt.message),
        "optimizer_decision_energy": optimizer_decision_energy,
        "optimizer_reported_energy": optimizer_decision_energy,
        "nfev": int(opt.nfev),
        "nit": int(opt.nit),
        "best_restart": int(opt.best_restart),
        "runtime_s": walltime,
        "count_2q": stats.count_2q,
        "depth_proxy": stats.depth,
        "circuit_depth": stats.depth,
        "qiskit_op_counts": stats.op_counts,
        **compiled_stats,
        "sector_probability": sector["sector_probability"],
        "sector_leak_probability": sector["sector_leak_probability"],
        "sector_leak_flag": sector["sector_leak_flag"],
        "sector_leak_threshold": sector["sector_leak_threshold"],
        "boson_legal_probability_min": sector.get("boson_legal_probability_min"),
        "boson_illegal_probability_max": sector.get("boson_illegal_probability_max"),
        "boson_truncation_leak_flag": sector.get("boson_truncation_leak_flag"),
        "boson_subspace_diagnostics": sector.get("boson_subspace_diagnostics"),
        "truncation_diagnostics": sector.get("truncation_constraints_evaluated"),
        "sector_diagnostics": sector,
        "uses_exact_for_decision": False,
        "exact_reference_usage": "reporting_only_after_optimization",
        "phase3_controller_called": False,
        "pauli_ordering": "left-to-right q_(n-1)...q_0; qubit 0 rightmost",
        "internal_pauli_alphabet": "e/x/y/z",
        "qiskit_boundary": "pipelines.exact_bench_only",
        **_source_fields(),
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "theta": theta.tolist(),
    }
    if decision_noise_metadata is not None:
        row.update(
            {
                "benchmark_decision_noise_status": "ok",
                "benchmark_decision_noise": copy_decision_noise_metadata(decision_noise_metadata),
            }
        )
    payload = {
        "schema": SCHEMA_VERSION,
        "family": family_key,
        "case_id": case_key,
        "algorithm_id": _METHOD_ID,
        "method_id": _METHOD_ID,
        "status": "completed",
        "qiskit_available": True,
        "runner": "pipelines.exact_bench.generic_static_hea_qiskit_vqe.run_static_hea_qiskit_vqe_single",
        "table_i": {
            "tex_label": "tab:static_claims",
            "static_train_suite": True,
            "sweep_complete": False,
        },
        "guardrails": {
            "uses_exact_for_decision": False,
            "exact_reference_usage": "reporting_only_after_optimization",
            "phase3_controller_called": False,
            "qiskit_boundary": "pipelines.exact_bench_only",
        },
        "comparator_source": _source_fields(),
        "spec": {
            "benchmark_id": spec.benchmark_id,
            "family": spec.family,
            "base_pipeline_args": list(spec.base_pipeline_args),
            "split": spec.split,
            "tags": list(spec.tags),
        },
        "result": row,
        "rows": [row],
    }
    if decision_noise_metadata is not None:
        payload.update(
            {
                "benchmark_decision_noise_status": "ok",
                "benchmark_decision_noise": copy_decision_noise_metadata(decision_noise_metadata),
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    rows_payload = {"schema": f"{SCHEMA_VERSION}_rows", "rows": [row]}
    if decision_noise_metadata is not None:
        rows_payload.update(
            {
                "benchmark_decision_noise_status": "ok",
                "benchmark_decision_noise": copy_decision_noise_metadata(decision_noise_metadata),
            }
        )
    _write_json(output / "result.json", payload)
    _write_json(output / "rows.json", rows_payload)
    _write_json(
        output / "manifest.json",
        {"schema": f"{SCHEMA_VERSION}_manifest", **{k: v for k, v in payload.items() if k != "schema"}},
    )
    _write_json(output / "generic_static_single.json", payload)
    write_proxy_sidecars([row], output, summary_extras={"schema_source": SCHEMA_VERSION})
    return payload


def run_static_hea_qiskit_vqe_single(
    *,
    family: str,
    case_id: str,
    output_dir: Path,
    reps: int = 2,
    restarts: int = 3,
    maxiter: int = 800,
    optimizer: str = "COBYLA",
    seed: int = 42,
    benchmark_decision_noise_config: BenchmarkDecisionNoiseConfig | Mapping[str, Any] | None = None,
    optimizer_profile: str | None = None,
    optimizer_profile_source: str | None = None,
    hea_optimizer: str | None = None,
    hea_spsa_maxiter: int | None = None,
    hea_spsa_seed: int | None = None,
    hea_spsa_learning_rate: float | str | None = None,
    hea_spsa_perturbation: float | str | None = None,
    optimizer_overlay_source: str | None = None,
) -> dict[str, Any]:
    """Run one generic Qiskit HEA VQE benchmark case and always emit artifacts."""
    try:
        return _run_static_hea_qiskit_vqe_single_impl(
            family=family,
            case_id=case_id,
            output_dir=output_dir,
            reps=reps,
            restarts=restarts,
            maxiter=maxiter,
            optimizer=optimizer,
            seed=seed,
            benchmark_decision_noise_config=benchmark_decision_noise_config,
            optimizer_profile=optimizer_profile,
            optimizer_profile_source=optimizer_profile_source,
            hea_optimizer=hea_optimizer,
            hea_spsa_maxiter=hea_spsa_maxiter,
            hea_spsa_seed=hea_spsa_seed,
            hea_spsa_learning_rate=hea_spsa_learning_rate,
            hea_spsa_perturbation=hea_spsa_perturbation,
            optimizer_overlay_source=optimizer_overlay_source,
        )
    except QiskitHeaUnavailable as exc:
        return _skip_payload(
            family=str(family).strip(),
            case_id=str(case_id).strip(),
            output_dir=Path(output_dir),
            reason=str(exc),
        )
    except Exception as exc:
        return _failure_payload(
            family=str(family).strip(),
            case_id=str(case_id).strip(),
            output_dir=Path(output_dir),
            reason=str(exc),
            exception_type=type(exc).__name__,
            qiskit_available=has_qiskit_hea_support(),
        )


__all__ = [
    "SCHEMA_VERSION",
    "default_static_hea_case_ids",
    "run_static_hea_qiskit_vqe_single",
    "sector_probability",
]
