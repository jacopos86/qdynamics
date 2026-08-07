#!/usr/bin/env python3
"""Static/no-drive HH ground-state benchmark matrix runner.

This runner intentionally stays under ``pipelines/exact_bench`` and reuses the
canonical static ADAPT problem setup rather than refactoring ``cross_check_suite``.
It emits one normalized row per ``(algorithm, Hamiltonian)`` pair with an explicit
``hamiltonian_id`` for downstream benchmark tables.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from pipelines.exact_bench.benchmark_metrics_proxy import (
    SCHEMA_VERSION as PROXY_SCHEMA_VERSION,
    write_proxy_sidecars,
)
from pipelines.exact_bench.benchmark_decision_noise import (
    BenchmarkDecisionNoiseConfig,
    coerce_config as coerce_benchmark_decision_noise_config,
    copy_decision_noise_metadata,
)
from pipelines.exact_bench.hh_conventional_vqe import (
    default_hh_conventional_vqe_config,
    has_qiskit_hea_support,
    run_compiled_operator_avqite_trial,
    run_compiled_operator_qsci_trial,
    run_compiled_operator_sqd_trial,
    run_compiled_operator_vqe_trial,
    run_hh_conventional_vqe_trial,
)
from pipelines.static_adapt.adapt_pipeline import _run_hardcoded_adapt_vqe
from src.quantum.compiled_polynomial import compile_polynomial_action, energy_via_one_apply
from pipelines.static_adapt.builders.pool_resolution import resolve_pool_plan
from pipelines.static_adapt.builders.primitive_pools import (
    _UCCSD_DOUBLE_LABEL_RE,
    _UCCSD_SINGLE_LABEL_RE,
    _build_hh_uccsd_fermion_lifted_pool,
    _fermion_mode_to_site,
)
from pipelines.static_adapt.builders.problem_registry import (
    ProblemRequest,
    resolve_problem_context,
)
from src.quantum.ed_hubbard_holstein import build_hh_sector_hamiltonian_ed
from src.quantum.vqe_latex_python_pairs import half_filled_num_particles


RUNNER_SCHEMA_VERSION = "hh_static_ground_state_benchmark_v1"
_DECISION_NOISE_SUPPORTED_HH_ALGORITHM_IDS = frozenset(
    {
        "hh_hva_termwise_vqe",
        "hh_hva_layerwise_vqe",
        "hh_hea_qiskit_vqe",
        "hh_uccsd_lifted_vqe",
        "hh_lang_firsov_sq_lf_vqe",
    }
)


@dataclass(frozen=True)
class HHBenchmarkCase:
    case_id: str
    num_sites: int
    t: float
    u: float
    dv: float
    omega0: float
    g_ep: float
    n_ph_max: int
    boson_encoding: str
    ordering: str
    boundary: str
    include_zero_point: bool

    def to_problem_request(self) -> ProblemRequest:
        return ProblemRequest(
            problem_key="hh",
            num_sites=int(self.num_sites),
            t=float(self.t),
            u=float(self.u),
            dv=float(self.dv),
            omega0=float(self.omega0),
            g_ep=float(self.g_ep),
            n_ph_max=int(self.n_ph_max),
            boson_encoding=str(self.boson_encoding),
            ordering=str(self.ordering),
            boundary=str(self.boundary),
            include_zero_point=bool(self.include_zero_point),
        )


@dataclass(frozen=True)
class HHBenchmarkAlgorithmSpec:
    algorithm_id: str
    runner_kind: Literal[
        "adapt_vqe",
        "conventional_vqe",
        "compiled_operator_vqe",
        "compiled_operator_avqite",
        "compiled_operator_qsci",
        "compiled_operator_sqd",
    ] = "adapt_vqe"
    display_name: str = ""

    # Conventional fixed-ansatz VQE fields. None means use the legacy HH
    # per-(L, n_ph_max) defaults shared with cross_check_suite.
    ansatz_kind: str = ""
    ansatz_name: str = ""
    vqe_reps: int | None = None
    vqe_restarts: int | None = None
    vqe_maxiter: int | None = None
    optimizer: str = "COBYLA"

    # Benchmark-only compiled operator-list VQE/AVQITE fields.
    operator_source: str = ""
    parameterization_mode: str = "logical_shared"
    avqite_step_size: float | None = None
    avqite_max_steps: int | None = None
    avqite_energy_tol: float | None = None
    avqite_residual_tol: float | None = None
    basis_probe_angle: float | None = None
    basis_amp_cutoff: float | None = None
    qsci_max_basis_states: int | None = None
    sqd_shots_per_probe: int | None = None
    sqd_max_basis_states: int | None = None
    sqd_seed: int | None = None

    # ADAPT fields.
    adapt_pool: str = ""
    continuation_mode: str = "legacy"
    max_depth: int = 0
    eps_grad: float = 1.0e-5
    eps_energy: float = 1.0e-8
    maxiter: int = 0
    seed: int = 7
    allow_repeats: bool = True
    finite_angle_fallback: bool = True
    finite_angle: float = 0.1
    finite_angle_min_improvement: float = 1.0e-12
    adapt_reopt_policy: str = "full"
    paop_r: int = 1
    paop_split_paulis: bool = False
    paop_prune_eps: float = 0.0
    paop_normalization: str = "none"
    phase2_batch_selection_mode: str = ""
    phase3_backend_cost_mode: str = "proxy"


def canonical_hh_benchmark_cases() -> tuple[HHBenchmarkCase, ...]:
    """Return the static/no-drive HH benchmark Hamiltonian matrix."""
    common = {
        "t": 1.0,
        "dv": 0.0,
        "omega0": 1.0,
        "n_ph_max": 1,
        "boson_encoding": "binary",
        "ordering": "blocked",
        "boundary": "open",
        "include_zero_point": True,
    }
    return (
        HHBenchmarkCase(
            case_id="hh_L2_strong_canonical",
            num_sites=2,
            u=4.0,
            g_ep=0.5,
            **common,
        ),
        HHBenchmarkCase(
            case_id="hh_L2_weak_diagnostic",
            num_sites=2,
            u=0.5,
            g_ep=0.2,
            **common,
        ),
        HHBenchmarkCase(
            case_id="hh_L3_weak_current_success",
            num_sites=3,
            u=0.5,
            g_ep=0.2,
            **common,
        ),
        HHBenchmarkCase(
            case_id="hh_L3_strong_historical_anchor",
            num_sites=3,
            u=4.0,
            g_ep=0.5,
            **common,
        ),
    )


def default_hh_benchmark_algorithms() -> tuple[HHBenchmarkAlgorithmSpec, ...]:
    """Return the mixed existing HH ADAPT + conventional algorithm inventory."""
    common = {
        "continuation_mode": "legacy",
        "eps_grad": 1.0e-5,
        "eps_energy": 1.0e-8,
        "seed": 7,
        "allow_repeats": True,
        "finite_angle_fallback": True,
        "finite_angle": 0.1,
        "finite_angle_min_improvement": 1.0e-12,
        "adapt_reopt_policy": "full",
        "paop_r": 1,
        "paop_split_paulis": False,
        "paop_prune_eps": 0.0,
        "paop_normalization": "none",
    }
    algorithms = [
        HHBenchmarkAlgorithmSpec(
            algorithm_id="hh_adapt_full_hamiltonian_legacy",
            adapt_pool="full_hamiltonian",
            max_depth=8,
            maxiter=300,
            **common,
        ),
        HHBenchmarkAlgorithmSpec(
            algorithm_id="hh_adapt_hva_legacy",
            adapt_pool="hva",
            max_depth=12,
            maxiter=400,
            **common,
        ),
        HHBenchmarkAlgorithmSpec(
            algorithm_id="hh_adapt_paop_lf_std_legacy",
            adapt_pool="paop_lf_std",
            max_depth=10,
            maxiter=300,
            **common,
        ),
        HHBenchmarkAlgorithmSpec(
            algorithm_id="hh_adapt_qeb_sq_lf_std_legacy",
            adapt_pool="sq_lf_std",
            max_depth=10,
            maxiter=300,
            **common,
        ),
        HHBenchmarkAlgorithmSpec(
            algorithm_id="hh_adapt_tetris_paop_lf_std_phase3",
            adapt_pool="paop_lf_std",
            continuation_mode="phase3_v1",
            max_depth=10,
            maxiter=300,
            phase2_batch_selection_mode="tetris_disjoint_benchmark",
            **{k: v for k, v in common.items() if k != "continuation_mode"},
        ),
        HHBenchmarkAlgorithmSpec(
            algorithm_id="hh_adapt_overlap_paop_lf_std_phase3",
            adapt_pool="paop_lf_std",
            continuation_mode="phase3_v1",
            max_depth=10,
            maxiter=300,
            phase2_batch_selection_mode="overlap_orthogonal_benchmark",
            **{k: v for k, v in common.items() if k != "continuation_mode"},
        ),
        HHBenchmarkAlgorithmSpec(
            algorithm_id="hh_adapt_ceo_paop_lf_std_phase3",
            adapt_pool="paop_lf_std",
            continuation_mode="phase3_v1",
            max_depth=10,
            maxiter=300,
            phase2_batch_selection_mode="ceo_commuting_benchmark",
            **{k: v for k, v in common.items() if k != "continuation_mode"},
        ),
        HHBenchmarkAlgorithmSpec(
            algorithm_id="hh_adapt_uccsd_otimes_paop_lf_std_legacy",
            adapt_pool="uccsd_otimes_paop_lf_std",
            max_depth=12,
            maxiter=400,
            **common,
        ),
        HHBenchmarkAlgorithmSpec(
            algorithm_id="hh_adapt_full_meta_legacy",
            adapt_pool="full_meta",
            max_depth=6,
            maxiter=240,
            **common,
        ),
        HHBenchmarkAlgorithmSpec(
            algorithm_id="hh_adapt_pareto_lean_legacy",
            adapt_pool="pareto_lean",
            max_depth=6,
            maxiter=240,
            **common,
        ),
        HHBenchmarkAlgorithmSpec(
            algorithm_id="hh_hva_termwise_vqe",
            runner_kind="conventional_vqe",
            display_name="HH-Termwise",
            ansatz_kind="termwise",
            ansatz_name="hh_hva_termwise",
            optimizer="COBYLA",
            seed=42,
        ),
        HHBenchmarkAlgorithmSpec(
            algorithm_id="hh_hva_layerwise_vqe",
            runner_kind="conventional_vqe",
            display_name="HH-Layerwise",
            ansatz_kind="layerwise",
            ansatz_name="hh_hva_layerwise",
            optimizer="COBYLA",
            seed=42,
        ),
        HHBenchmarkAlgorithmSpec(
            algorithm_id="hh_uccsd_lifted_vqe",
            runner_kind="compiled_operator_vqe",
            display_name="HH-UCCSD-Lifted",
            ansatz_name="hh_uccsd_lifted",
            operator_source="hh_uccsd_lifted",
            parameterization_mode="logical_shared",
            optimizer="COBYLA",
            seed=42,
        ),
        HHBenchmarkAlgorithmSpec(
            algorithm_id="hh_avqite_uccsd_lifted",
            runner_kind="compiled_operator_avqite",
            display_name="HH-AVQITE-UCCSD-Lifted",
            ansatz_name="hh_uccsd_lifted",
            operator_source="hh_uccsd_lifted",
            parameterization_mode="logical_shared",
            avqite_step_size=0.1,
            avqite_max_steps=80,
            avqite_energy_tol=1e-8,
            avqite_residual_tol=1e-6,
        ),
        HHBenchmarkAlgorithmSpec(
            algorithm_id="hh_qsci_sq_lf_std",
            runner_kind="compiled_operator_qsci",
            display_name="HH-QSCI-SQ-LF-Std",
            ansatz_name="hh_qsci_sq_lf_std",
            operator_source="hh_sq_lf_std_pool",
            basis_probe_angle=float(np.pi / 2),
            basis_amp_cutoff=1e-9,
            qsci_max_basis_states=32,
        ),
        HHBenchmarkAlgorithmSpec(
            algorithm_id="hh_sqd_sq_lf_std",
            runner_kind="compiled_operator_sqd",
            display_name="HH-SQD-SQ-LF-Std",
            ansatz_name="hh_sqd_sq_lf_std",
            operator_source="hh_sq_lf_std_pool",
            basis_probe_angle=float(np.pi / 2),
            sqd_shots_per_probe=256,
            sqd_max_basis_states=32,
            sqd_seed=7,
        ),
        HHBenchmarkAlgorithmSpec(
            algorithm_id="hh_puccd_lifted_vqe",
            runner_kind="compiled_operator_vqe",
            display_name="HH-pUCCD-Lifted",
            ansatz_name="hh_puccd_lifted",
            operator_source="hh_puccd_lifted",
            parameterization_mode="logical_shared",
            optimizer="COBYLA",
            seed=42,
        ),
        HHBenchmarkAlgorithmSpec(
            algorithm_id="hh_lang_firsov_sq_lf_vqe",
            runner_kind="compiled_operator_vqe",
            display_name="HH-LangFirsov-SQ-LF-VQE",
            ansatz_name="hh_lang_firsov_sq_lf",
            operator_source="hh_sq_lf_std_lf_only",
            parameterization_mode="logical_shared",
            optimizer="COBYLA",
            seed=42,
        ),
    ]
    if has_qiskit_hea_support():
        algorithms.append(
            HHBenchmarkAlgorithmSpec(
                algorithm_id="hh_hea_qiskit_vqe",
                runner_kind="conventional_vqe",
                display_name="HH-HEA-Qiskit",
                ansatz_kind="qiskit_hea",
                ansatz_name="hh_hea_qiskit",
                vqe_reps=2,
                optimizer="COBYLA",
                seed=42,
            )
        )
    return tuple(algorithms)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return _json_ready(value.tolist())
        except Exception:
            pass
    return value


def _json_default(value: Any) -> Any:
    ready = _json_ready(value)
    if ready is value:
        return str(value)
    return ready


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _select_cases(
    all_cases: Sequence[HHBenchmarkCase],
    case_ids: Sequence[str] | None,
) -> tuple[HHBenchmarkCase, ...]:
    if not case_ids:
        return tuple(all_cases)
    lookup = {case.case_id: case for case in all_cases}
    selected: list[HHBenchmarkCase] = []
    for raw_case_id in case_ids:
        case_id = str(raw_case_id)
        if case_id not in lookup:
            known = ", ".join(sorted(lookup))
            raise ValueError(f"Unknown HH benchmark case_id={case_id!r}; known: {known}")
        selected.append(lookup[case_id])
    return tuple(selected)


def _payload_path(output_dir: Path, case: HHBenchmarkCase, algorithm: HHBenchmarkAlgorithmSpec) -> Path:
    return output_dir / "runs" / case.case_id / f"{algorithm.algorithm_id}.json"


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _case_reference_energy_audit(
    *,
    h_poly: Any,
    resolved_problem: Any,
    exact_gs: float,
) -> dict[str, Any]:
    """Return reference-state energy diagnostics without failing the benchmark matrix."""

    try:
        psi_ref = _reference_state_vector(resolved_problem)
        if psi_ref is None:
            raise ValueError("resolved problem has no reference_state.build_state()")
        psi_arr = np.asarray(psi_ref, dtype=complex).reshape(-1)
        compiled_h = compile_polynomial_action(h_poly, tol=1.0e-12)
        reference_energy, _ = energy_via_one_apply(psi_arr, compiled_h)
        reference_energy = float(reference_energy)
        exact_energy = float(exact_gs)
        return {
            "reference_energy_status": "ok",
            "reference_state_energy": reference_energy,
            "reference_abs_delta_e": abs(reference_energy - exact_energy),
            "reference_state_source": "resolved_problem.reference_state",
        }
    except Exception as exc:  # pragma: no cover - defensive path for unusual problem contexts.
        return {
            "reference_energy_status": "unavailable",
            "reference_state_energy": None,
            "reference_abs_delta_e": None,
            "reference_state_source": "resolved_problem.reference_state",
            "reference_energy_error": f"{type(exc).__name__}: {exc}",
        }


def _build_adapt_kwargs(
    *,
    case: HHBenchmarkCase,
    algorithm: HHBenchmarkAlgorithmSpec,
    h_poly: Any,
    resolved_problem: Any,
    exact_gs: float,
) -> dict[str, Any]:
    kwargs = {
        "h_poly": h_poly,
        "resolved_problem_context": resolved_problem,
        "num_sites": int(case.num_sites),
        "ordering": str(case.ordering),
        "problem": "hh",
        "adapt_pool": str(algorithm.adapt_pool),
        "t": float(case.t),
        "u": float(case.u),
        "dv": float(case.dv),
        "boundary": str(case.boundary),
        "omega0": float(case.omega0),
        "g_ep": float(case.g_ep),
        "n_ph_max": int(case.n_ph_max),
        "boson_encoding": str(case.boson_encoding),
        "include_zero_point": bool(case.include_zero_point),
        "max_depth": int(algorithm.max_depth),
        "eps_grad": float(algorithm.eps_grad),
        "eps_energy": float(algorithm.eps_energy),
        "maxiter": int(algorithm.maxiter),
        "seed": int(algorithm.seed),
        "adapt_inner_optimizer": "POWELL",
        "allow_repeats": bool(algorithm.allow_repeats),
        "finite_angle_fallback": bool(algorithm.finite_angle_fallback),
        "finite_angle": float(algorithm.finite_angle),
        "finite_angle_min_improvement": float(algorithm.finite_angle_min_improvement),
        "adapt_reopt_policy": str(algorithm.adapt_reopt_policy),
        "adapt_continuation_mode": str(algorithm.continuation_mode),
        "paop_r": int(algorithm.paop_r),
        "paop_split_paulis": bool(algorithm.paop_split_paulis),
        "paop_prune_eps": float(algorithm.paop_prune_eps),
        "paop_normalization": str(algorithm.paop_normalization),
        "exact_gs_override": float(exact_gs),
        "phase3_backend_cost_mode": str(algorithm.phase3_backend_cost_mode or "proxy"),
    }
    if str(algorithm.phase2_batch_selection_mode).strip():
        kwargs["phase2_batch_selection_mode"] = str(algorithm.phase2_batch_selection_mode).strip()
    return kwargs


def _resolved_conventional_vqe_config(
    *,
    case: HHBenchmarkCase,
    algorithm: HHBenchmarkAlgorithmSpec,
) -> dict[str, Any]:
    config = default_hh_conventional_vqe_config(
        int(case.num_sites),
        int(case.n_ph_max),
    )
    if algorithm.vqe_reps is not None:
        config["reps"] = int(algorithm.vqe_reps)
    if algorithm.vqe_restarts is not None:
        config["restarts"] = int(algorithm.vqe_restarts)
    if algorithm.vqe_maxiter is not None:
        config["maxiter"] = int(algorithm.vqe_maxiter)
    if str(algorithm.optimizer).strip():
        config["method"] = str(algorithm.optimizer)
    return {
        "reps": int(config["reps"]),
        "restarts": int(config["restarts"]),
        "maxiter": int(config["maxiter"]),
        "optimizer": str(config["method"]),
    }


def _reference_state_vector(resolved_problem: Any) -> Any | None:
    reference_state = getattr(resolved_problem, "reference_state", None)
    build_state = getattr(reference_state, "build_state", None)
    if callable(build_state):
        return build_state()
    return None


def _build_conventional_kwargs(
    *,
    case: HHBenchmarkCase,
    algorithm: HHBenchmarkAlgorithmSpec,
    h_poly: Any,
    resolved_problem: Any,
    exact_gs: float,
) -> dict[str, Any]:
    config = _resolved_conventional_vqe_config(case=case, algorithm=algorithm)
    return {
        "ansatz_kind": str(algorithm.ansatz_kind or algorithm.algorithm_id),
        "h_poly": h_poly,
        "exact_gs": float(exact_gs),
        "num_sites": int(case.num_sites),
        "t": float(case.t),
        "u": float(case.u),
        "dv": float(case.dv),
        "omega0": float(case.omega0),
        "g_ep": float(case.g_ep),
        "n_ph_max": int(case.n_ph_max),
        "boson_encoding": str(case.boson_encoding),
        "boundary": str(case.boundary),
        "ordering": str(case.ordering),
        "include_zero_point": bool(case.include_zero_point),
        "reps": int(config["reps"]),
        "optimizer": str(config["optimizer"]),
        "maxiter": int(config["maxiter"]),
        "restarts": int(config["restarts"]),
        "seed": int(algorithm.seed),
        "psi_ref": _reference_state_vector(resolved_problem),
    }


def _build_hh_uccsd_lifted_terms(*, case: HHBenchmarkCase) -> list[Any]:
    return _build_hh_uccsd_fermion_lifted_pool(
        int(case.num_sites),
        int(case.n_ph_max),
        str(case.boson_encoding),
        str(case.ordering),
        str(case.boundary),
        num_particles=tuple(half_filled_num_particles(int(case.num_sites))),
    )


def _lifted_uccsd_excitation_metadata(term: Any) -> tuple[str, str, tuple[int, ...], tuple[int, ...]]:
    """Return structured excitation metadata from the existing lifted-UCCSD label contract."""
    if not hasattr(term, "label") or not hasattr(term, "polynomial"):
        raise ValueError(
            "Lifted HH UCCSD pool entries must expose AnsatzTerm-like 'label' and 'polynomial' fields."
        )
    label = str(getattr(term, "label")).strip()
    prefix = "uccsd_ferm_lifted::"
    if not label.startswith(prefix):
        raise ValueError(f"Unsupported lifted HH UCCSD entry label {label!r}; expected prefix {prefix!r}.")
    body = label[len(prefix):]

    single_match = _UCCSD_SINGLE_LABEL_RE.match(body)
    if single_match is not None:
        return (
            "single",
            str(single_match.group(1)),
            (int(single_match.group(2)),),
            (int(single_match.group(3)),),
        )

    double_match = _UCCSD_DOUBLE_LABEL_RE.match(body)
    if double_match is None:
        raise ValueError(
            f"Could not parse lifted HH UCCSD excitation label {label!r}; refusing pUCCD filtering."
        )
    return (
        "double",
        str(double_match.group(1)),
        (int(double_match.group(2)), int(double_match.group(3))),
        (int(double_match.group(4)), int(double_match.group(5))),
    )


def _is_hh_puccd_lifted_paired_double(
    term: Any,
    *,
    num_sites: int,
    ordering: str,
) -> bool:
    rank, spin_block, occupied_modes, virtual_modes = _lifted_uccsd_excitation_metadata(term)
    if rank != "double":
        return False
    if spin_block != "ab":
        return False
    if len(occupied_modes) != 2 or len(virtual_modes) != 2:
        return False
    if len(set(occupied_modes)) != 2 or len(set(virtual_modes)) != 2:
        return False

    occupied_sites = tuple(
        _fermion_mode_to_site(mode, num_sites=int(num_sites), ordering=str(ordering))
        for mode in occupied_modes
    )
    virtual_sites = tuple(
        _fermion_mode_to_site(mode, num_sites=int(num_sites), ordering=str(ordering))
        for mode in virtual_modes
    )
    return occupied_sites[0] == occupied_sites[1] and virtual_sites[0] == virtual_sites[1]


def _build_hh_puccd_lifted_terms(*, case: HHBenchmarkCase) -> list[Any]:
    lifted_terms = _build_hh_uccsd_lifted_terms(case=case)
    paired_doubles = [
        term
        for term in lifted_terms
        if _is_hh_puccd_lifted_paired_double(
            term,
            num_sites=int(case.num_sites),
            ordering=str(case.ordering),
        )
    ]
    if not paired_doubles:
        raise ValueError(
            "operator_source='hh_puccd_lifted' selected zero paired-double operators from "
            f"hh_uccsd_lifted for case={case.case_id!r}; lifted UCCSD entries must expose "
            "parseable excitation labels or structured metadata."
        )
    return paired_doubles


def _num_particles_for_problem(case: HHBenchmarkCase, resolved_problem: Any | None) -> tuple[int, int]:
    sector = getattr(resolved_problem, "sector", None)
    sector_particles = getattr(sector, "num_particles", None)
    if sector_particles is not None:
        particles = tuple(int(x) for x in tuple(sector_particles))
        if len(particles) == 2:
            return particles  # type: ignore[return-value]
    return tuple(int(x) for x in half_filled_num_particles(int(case.num_sites)))  # type: ignore[return-value]


def _build_hh_sq_lf_std_pool_terms(
    *,
    case: HHBenchmarkCase,
    algorithm: HHBenchmarkAlgorithmSpec,
    resolved_problem: Any | None,
) -> list[Any]:
    resolved = resolved_problem if resolved_problem is not None else resolve_problem_context(case.to_problem_request())
    pool_plan = resolve_pool_plan(
        resolved_problem=resolved,
        continuation_mode="legacy",
        adapt_pool="sq_lf_std",
        paop_r=int(algorithm.paop_r),
        paop_split_paulis=bool(algorithm.paop_split_paulis),
        paop_prune_eps=float(algorithm.paop_prune_eps),
        paop_normalization=str(algorithm.paop_normalization),
        phase3_symmetry_mitigation_mode="off",
        filter_resolution=None,
        ai_log=None,
    )
    terms = list(pool_plan.pool)
    if not terms:
        raise ValueError("operator_source='hh_sq_lf_std_pool' resolved zero sq_lf_std operators.")
    return terms


def _is_hh_sq_lf_std_lf_displacement_term(term: Any) -> bool:
    """Return True for stable sq_lf_std LF-displacement labels only."""
    label = str(getattr(term, "label", ""))
    return label.startswith("sq_lf_std:lf_disp(")


def _build_hh_sq_lf_std_lf_only_terms(
    *,
    case: HHBenchmarkCase,
    algorithm: HHBenchmarkAlgorithmSpec,
    resolved_problem: Any | None,
) -> list[Any]:
    terms = _build_hh_sq_lf_std_pool_terms(
        case=case,
        algorithm=algorithm,
        resolved_problem=resolved_problem,
    )
    lf_terms = [term for term in terms if _is_hh_sq_lf_std_lf_displacement_term(term)]
    if not lf_terms:
        raise ValueError(
            "operator_source='hh_sq_lf_std_lf_only' resolved zero LF displacement operators "
            "from sq_lf_std labels."
        )
    return lf_terms


def _resolve_compiled_operator_terms(
    *,
    case: HHBenchmarkCase,
    algorithm: HHBenchmarkAlgorithmSpec,
    resolved_problem: Any | None = None,
) -> list[Any]:
    source = str(algorithm.operator_source).strip()
    if source == "hh_uccsd_lifted":
        return _build_hh_uccsd_lifted_terms(case=case)
    if source == "hh_puccd_lifted":
        return _build_hh_puccd_lifted_terms(case=case)
    if source == "hh_sq_lf_std_pool":
        return _build_hh_sq_lf_std_pool_terms(
            case=case,
            algorithm=algorithm,
            resolved_problem=resolved_problem,
        )
    if source == "hh_sq_lf_std_lf_only":
        return _build_hh_sq_lf_std_lf_only_terms(
            case=case,
            algorithm=algorithm,
            resolved_problem=resolved_problem,
        )
    raise ValueError(f"Unknown compiled operator_source={algorithm.operator_source!r}")


def _build_hh_sector_projection_data(
    *,
    case: HHBenchmarkCase,
    resolved_problem: Any | None,
) -> tuple[Any, list[int]]:
    h_sector, basis = build_hh_sector_hamiltonian_ed(
        dims=int(case.num_sites),
        J=float(case.t),
        U=float(case.u),
        omega0=float(case.omega0),
        g=float(case.g_ep),
        n_ph_max=int(case.n_ph_max),
        num_particles=_num_particles_for_problem(case, resolved_problem),
        indexing=str(case.ordering),
        boson_encoding=str(case.boson_encoding),
        pbc=(str(case.boundary).strip().lower() == "periodic"),
        delta_v=float(case.dv),
        include_zero_point=bool(case.include_zero_point),
        sparse=True,
        return_basis=True,
    )
    return h_sector, [int(idx) for idx in basis.basis_indices]


def _build_compiled_operator_kwargs(
    *,
    case: HHBenchmarkCase,
    algorithm: HHBenchmarkAlgorithmSpec,
    h_poly: Any,
    resolved_problem: Any,
    exact_gs: float,
) -> dict[str, Any]:
    config = _resolved_conventional_vqe_config(case=case, algorithm=algorithm)
    psi_ref = _reference_state_vector(resolved_problem)
    if psi_ref is None:
        raise ValueError("Resolved HH problem lacks reference_state.build_state() for compiled-operator VQE.")
    display_name = str(algorithm.display_name or algorithm.algorithm_id)
    ansatz_name = str(algorithm.ansatz_name or algorithm.algorithm_id)
    return {
        "operator_terms": _resolve_compiled_operator_terms(
            case=case,
            algorithm=algorithm,
            resolved_problem=resolved_problem,
        ),
        "ansatz_name": ansatz_name,
        "display_name": display_name,
        "h_poly": h_poly,
        "exact_gs": float(exact_gs),
        "psi_ref": psi_ref,
        "optimizer": str(config["optimizer"]),
        "maxiter": int(config["maxiter"]),
        "restarts": int(config["restarts"]),
        "seed": int(algorithm.seed),
        "parameterization_mode": str(algorithm.parameterization_mode),
    }


def _build_compiled_operator_avqite_kwargs(
    *,
    case: HHBenchmarkCase,
    algorithm: HHBenchmarkAlgorithmSpec,
    h_poly: Any,
    resolved_problem: Any,
    exact_gs: float,
) -> dict[str, Any]:
    psi_ref = _reference_state_vector(resolved_problem)
    if psi_ref is None:
        raise ValueError("Resolved HH problem lacks reference_state.build_state() for compiled-operator AVQITE.")
    display_name = str(algorithm.display_name or algorithm.algorithm_id)
    ansatz_name = str(algorithm.ansatz_name or algorithm.algorithm_id)
    kwargs: dict[str, Any] = {
        "operator_terms": _resolve_compiled_operator_terms(
            case=case,
            algorithm=algorithm,
            resolved_problem=resolved_problem,
        ),
        "ansatz_name": ansatz_name,
        "display_name": display_name,
        "h_poly": h_poly,
        "exact_gs": float(exact_gs),
        "psi_ref": psi_ref,
        "parameterization_mode": str(algorithm.parameterization_mode),
    }
    if algorithm.avqite_step_size is not None:
        kwargs["avqite_step_size"] = float(algorithm.avqite_step_size)
    if algorithm.avqite_max_steps is not None:
        kwargs["avqite_max_steps"] = int(algorithm.avqite_max_steps)
    if algorithm.avqite_energy_tol is not None:
        kwargs["avqite_energy_tol"] = float(algorithm.avqite_energy_tol)
    if algorithm.avqite_residual_tol is not None:
        kwargs["avqite_residual_tol"] = float(algorithm.avqite_residual_tol)
    return kwargs


def _build_compiled_operator_qsci_kwargs(
    *,
    case: HHBenchmarkCase,
    algorithm: HHBenchmarkAlgorithmSpec,
    resolved_problem: Any,
    exact_gs: float,
) -> dict[str, Any]:
    psi_ref = _reference_state_vector(resolved_problem)
    if psi_ref is None:
        raise ValueError("Resolved HH problem lacks reference_state.build_state() for compiled-operator QSCI.")
    display_name = str(algorithm.display_name or algorithm.algorithm_id)
    ansatz_name = str(algorithm.ansatz_name or algorithm.algorithm_id)
    operator_terms = _resolve_compiled_operator_terms(
        case=case,
        algorithm=algorithm,
        resolved_problem=resolved_problem,
    )
    h_sector, sector_basis_full_indices = _build_hh_sector_projection_data(
        case=case,
        resolved_problem=resolved_problem,
    )
    kwargs: dict[str, Any] = {
        "operator_terms": operator_terms,
        "operator_labels": [
            str(getattr(term, "label", f"operator_{idx}"))
            for idx, term in enumerate(operator_terms)
        ],
        "ansatz_name": ansatz_name,
        "display_name": display_name,
        "sector_hamiltonian": h_sector,
        "sector_basis_full_indices": sector_basis_full_indices,
        "exact_gs": float(exact_gs),
        "psi_ref": psi_ref,
    }
    if algorithm.basis_probe_angle is not None:
        kwargs["basis_probe_angle"] = float(algorithm.basis_probe_angle)
    if algorithm.basis_amp_cutoff is not None:
        kwargs["basis_amp_cutoff"] = float(algorithm.basis_amp_cutoff)
    if algorithm.qsci_max_basis_states is not None:
        kwargs["qsci_max_basis_states"] = int(algorithm.qsci_max_basis_states)
    return kwargs


def _build_compiled_operator_sqd_kwargs(
    *,
    case: HHBenchmarkCase,
    algorithm: HHBenchmarkAlgorithmSpec,
    resolved_problem: Any,
    exact_gs: float,
) -> dict[str, Any]:
    psi_ref = _reference_state_vector(resolved_problem)
    if psi_ref is None:
        raise ValueError("Resolved HH problem lacks reference_state.build_state() for compiled-operator SQD.")
    display_name = str(algorithm.display_name or algorithm.algorithm_id)
    ansatz_name = str(algorithm.ansatz_name or algorithm.algorithm_id)
    operator_terms = _resolve_compiled_operator_terms(
        case=case,
        algorithm=algorithm,
        resolved_problem=resolved_problem,
    )
    h_sector, sector_basis_full_indices = _build_hh_sector_projection_data(
        case=case,
        resolved_problem=resolved_problem,
    )
    kwargs: dict[str, Any] = {
        "operator_terms": operator_terms,
        "operator_labels": [
            str(getattr(term, "label", f"operator_{idx}"))
            for idx, term in enumerate(operator_terms)
        ],
        "ansatz_name": ansatz_name,
        "display_name": display_name,
        "sector_hamiltonian": h_sector,
        "sector_basis_full_indices": sector_basis_full_indices,
        "exact_gs": float(exact_gs),
        "psi_ref": psi_ref,
    }
    if algorithm.basis_probe_angle is not None:
        kwargs["basis_probe_angle"] = float(algorithm.basis_probe_angle)
    if algorithm.sqd_shots_per_probe is not None:
        kwargs["sqd_shots_per_probe"] = int(algorithm.sqd_shots_per_probe)
    if algorithm.sqd_max_basis_states is not None:
        kwargs["sqd_max_basis_states"] = int(algorithm.sqd_max_basis_states)
    if algorithm.sqd_seed is not None:
        kwargs["sqd_seed"] = int(algorithm.sqd_seed)
    return kwargs


def _run_one_adapt_algorithm(
    *,
    case: HHBenchmarkCase,
    algorithm: HHBenchmarkAlgorithmSpec,
    resolved_problem: Any,
    exact_gs: float,
) -> dict[str, Any]:
    run_result = _run_hardcoded_adapt_vqe(
        **_build_adapt_kwargs(
            case=case,
            algorithm=algorithm,
            h_poly=getattr(resolved_problem, "hamiltonian"),
            resolved_problem=resolved_problem,
            exact_gs=float(exact_gs),
        )
    )
    payload_raw = run_result[0] if isinstance(run_result, tuple) else run_result
    return dict(payload_raw) if isinstance(payload_raw, Mapping) else {"payload": payload_raw}


def _run_one_conventional_algorithm(
    *,
    case: HHBenchmarkCase,
    algorithm: HHBenchmarkAlgorithmSpec,
    resolved_problem: Any,
    exact_gs: float,
    benchmark_decision_noise_config: BenchmarkDecisionNoiseConfig | None = None,
) -> dict[str, Any]:
    kwargs = _build_conventional_kwargs(
        case=case,
        algorithm=algorithm,
        h_poly=getattr(resolved_problem, "hamiltonian"),
        resolved_problem=resolved_problem,
        exact_gs=float(exact_gs),
    )
    if benchmark_decision_noise_config is not None and bool(benchmark_decision_noise_config.enabled):
        kwargs.update(
            {
                "benchmark_decision_noise_config": benchmark_decision_noise_config,
                "benchmark_decision_noise_scope": {
                    "family": "hh",
                    "case_id": str(case.case_id),
                    "algorithm_id": str(algorithm.algorithm_id),
                    "hh_runner_kind": str(algorithm.runner_kind),
                },
            }
        )
    return dict(
        run_hh_conventional_vqe_trial(
            **kwargs,
        )
    )


def _run_one_compiled_operator_algorithm(
    *,
    case: HHBenchmarkCase,
    algorithm: HHBenchmarkAlgorithmSpec,
    resolved_problem: Any,
    exact_gs: float,
    benchmark_decision_noise_config: BenchmarkDecisionNoiseConfig | None = None,
) -> dict[str, Any]:
    kwargs = _build_compiled_operator_kwargs(
        case=case,
        algorithm=algorithm,
        h_poly=getattr(resolved_problem, "hamiltonian"),
        resolved_problem=resolved_problem,
        exact_gs=float(exact_gs),
    )
    if benchmark_decision_noise_config is not None and bool(benchmark_decision_noise_config.enabled):
        kwargs.update(
            {
                "benchmark_decision_noise_config": benchmark_decision_noise_config,
                "benchmark_decision_noise_scope": {
                    "family": "hh",
                    "case_id": str(case.case_id),
                    "algorithm_id": str(algorithm.algorithm_id),
                    "hh_runner_kind": str(algorithm.runner_kind),
                },
            }
        )
    return dict(
        run_compiled_operator_vqe_trial(
            **kwargs,
        )
    )


def _run_one_compiled_operator_avqite_algorithm(
    *,
    case: HHBenchmarkCase,
    algorithm: HHBenchmarkAlgorithmSpec,
    resolved_problem: Any,
    exact_gs: float,
) -> dict[str, Any]:
    return dict(
        run_compiled_operator_avqite_trial(
            **_build_compiled_operator_avqite_kwargs(
                case=case,
                algorithm=algorithm,
                h_poly=getattr(resolved_problem, "hamiltonian"),
                resolved_problem=resolved_problem,
                exact_gs=float(exact_gs),
            )
        )
    )


def _run_one_compiled_operator_qsci_algorithm(
    *,
    case: HHBenchmarkCase,
    algorithm: HHBenchmarkAlgorithmSpec,
    resolved_problem: Any,
    exact_gs: float,
) -> dict[str, Any]:
    return dict(
        run_compiled_operator_qsci_trial(
            **_build_compiled_operator_qsci_kwargs(
                case=case,
                algorithm=algorithm,
                resolved_problem=resolved_problem,
                exact_gs=float(exact_gs),
            )
        )
    )


def _run_one_compiled_operator_sqd_algorithm(
    *,
    case: HHBenchmarkCase,
    algorithm: HHBenchmarkAlgorithmSpec,
    resolved_problem: Any,
    exact_gs: float,
) -> dict[str, Any]:
    return dict(
        run_compiled_operator_sqd_trial(
            **_build_compiled_operator_sqd_kwargs(
                case=case,
                algorithm=algorithm,
                resolved_problem=resolved_problem,
                exact_gs=float(exact_gs),
            )
        )
    )


def _status_from_payload(payload: Mapping[str, Any]) -> str:
    if bool(payload.get("success", True)):
        return "ok"
    return "failed"


def _quality_status_from_flags(*, status: str, flags: Sequence[str]) -> str:
    if str(status).lower() != "ok":
        return "failed"
    flag_set = {str(flag) for flag in flags}
    if "zero_operator_scaffold" in flag_set or "does_not_improve_reference_state" in flag_set:
        return "ok_reference_not_improved"
    if "optimizer_not_converged" in flag_set or "not_converged" in flag_set:
        return "ok_optimizer_suspect"
    if "large_energy_error_gt_0p1" in flag_set:
        return "ok_large_error"
    return "ok_paper_candidate"


def _benchmark_audit_flags(
    *,
    payload: Mapping[str, Any],
    runner_kind: str,
    abs_delta_e: float | None,
    exact_energy: float | None,
    reference_abs_delta_e: float | None,
) -> list[str]:
    flags: list[str] = []
    status = _status_from_payload(payload)
    if status != "ok":
        flags.append(f"status_{status}")
    optimizer_success = payload.get("optimizer_success")
    if optimizer_success is False:
        flags.append("optimizer_not_converged")
    converged = payload.get("converged")
    if converged is False and str(runner_kind) in {
        "conventional_vqe",
        "compiled_operator_vqe",
        "compiled_operator_avqite",
    }:
        flags.append("not_converged")
    if abs_delta_e is not None and reference_abs_delta_e is not None:
        if abs_delta_e >= reference_abs_delta_e - 1.0e-12:
            flags.append("does_not_improve_reference_state")
    energy = _finite_float_or_none(payload.get("energy"))
    if energy is not None and exact_energy is not None and energy < exact_energy - 1.0e-8:
        flags.append("energy_below_exact_check")
    if str(runner_kind) == "adapt_vqe" and int(payload.get("ansatz_depth", -1) or 0) == 0:
        flags.append("zero_operator_scaffold")
    if abs_delta_e is not None and abs_delta_e > 1.0e-1:
        flags.append("large_energy_error_gt_0p1")
    return sorted(dict.fromkeys(flags))


def _raw_row_from_payload(
    *,
    case: HHBenchmarkCase,
    algorithm: HHBenchmarkAlgorithmSpec,
    payload: Mapping[str, Any],
    artifact_path: Path,
    started_utc: str,
    finished_utc: str,
    runtime_s: float,
    exact_target_kind: str,
    reference_energy_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    runner_kind = str(algorithm.runner_kind)
    fallback_method_kind = (
        "conventional_vqe"
        if runner_kind == "compiled_operator_vqe"
        else "avqite"
        if runner_kind == "compiled_operator_avqite"
        else "qsci"
        if runner_kind == "compiled_operator_qsci"
        else "sqd"
        if runner_kind == "compiled_operator_sqd"
        else runner_kind
    )
    method_kind = str(payload.get("method_kind", fallback_method_kind))
    is_avqite = method_kind == "avqite" or runner_kind == "compiled_operator_avqite"
    is_qsci = method_kind == "qsci" or runner_kind == "compiled_operator_qsci"
    is_sqd = method_kind == "sqd" or runner_kind == "compiled_operator_sqd"
    is_compiled_operator = runner_kind in {
        "compiled_operator_vqe",
        "compiled_operator_avqite",
        "compiled_operator_qsci",
        "compiled_operator_sqd",
    }
    is_vqe_like = (
        method_kind == "conventional_vqe"
        or runner_kind in {"conventional_vqe", "compiled_operator_vqe"}
    )
    is_fixed_ansatz = is_vqe_like or is_avqite or is_qsci or is_sqd
    conventional_config = (
        _resolved_conventional_vqe_config(case=case, algorithm=algorithm)
        if is_vqe_like
        else {}
    )
    abs_delta_e = payload.get("abs_delta_e", payload.get("delta_E_abs"))
    vqe_reps = (
        None
        if is_compiled_operator
        else payload.get(
            "vqe_reps",
            payload.get("vqe_reps_used", conventional_config.get("reps", algorithm.vqe_reps)),
        )
    )
    vqe_restarts = (
        None
        if is_avqite or is_qsci or is_sqd
        else payload.get(
            "vqe_restarts",
            conventional_config.get("restarts", algorithm.vqe_restarts),
        )
    )
    vqe_maxiter = (
        None
        if is_avqite or is_qsci or is_sqd
        else payload.get(
            "vqe_maxiter",
            payload.get("vqe_maxiter_used", conventional_config.get("maxiter", algorithm.vqe_maxiter)),
        )
    )
    ansatz_name = (
        str(algorithm.ansatz_name or payload.get("ansatz_name", algorithm.algorithm_id))
        if is_fixed_ansatz
        else str(payload.get("ansatz_name", algorithm.algorithm_id))
    )
    source = str(algorithm.operator_source).strip()
    pool_name = (
        "sq_lf_std"
        if (
            ((is_qsci or is_sqd) and source == "hh_sq_lf_std_pool")
            or (runner_kind == "compiled_operator_vqe" and source == "hh_sq_lf_std_lf_only")
        )
        else "" if is_fixed_ansatz else str(payload.get("pool_type", algorithm.adapt_pool))
    )
    reference_audit = dict(reference_energy_audit or {})
    exact_energy = _finite_float_or_none(payload.get("exact_gs_energy", payload.get("exact_energy")))
    abs_delta_e_float = _finite_float_or_none(abs_delta_e)
    reference_energy = _finite_float_or_none(reference_audit.get("reference_state_energy"))
    reference_abs_delta_e = _finite_float_or_none(reference_audit.get("reference_abs_delta_e"))
    improvement_over_reference = (
        None
        if abs_delta_e_float is None or reference_abs_delta_e is None
        else float(reference_abs_delta_e - abs_delta_e_float)
    )
    beats_reference = None if improvement_over_reference is None else bool(improvement_over_reference > 0.0)
    benchmark_audit_flags = _benchmark_audit_flags(
        payload=payload,
        runner_kind=runner_kind,
        abs_delta_e=abs_delta_e_float,
        exact_energy=exact_energy,
        reference_abs_delta_e=reference_abs_delta_e,
    )
    status = _status_from_payload(payload)
    quality_status = _quality_status_from_flags(status=status, flags=benchmark_audit_flags)
    actual_optimizer = (
        "POWELL"
        if runner_kind == "adapt_vqe"
        else str(payload.get("optimizer", conventional_config.get("optimizer", "")) or "")
    )
    return {
        "schema": RUNNER_SCHEMA_VERSION,
        "run_id": f"{case.case_id}__{algorithm.algorithm_id}",
        "hamiltonian_id": str(case.case_id),
        "case_id": str(case.case_id),
        "method_id": str(algorithm.algorithm_id),
        "method_kind": method_kind,
        "display_name": str(algorithm.display_name or payload.get("display_name", algorithm.algorithm_id)),
        "ansatz_name": ansatz_name,
        "pool_name": pool_name,
        "problem": "hh",
        "L": int(case.num_sites),
        "status": status,
        "quality_status": quality_status,
        "started_utc": str(started_utc),
        "finished_utc": str(finished_utc),
        "runtime_s": float(runtime_s),
        "energy": payload.get("energy"),
        "exact_energy": payload.get("exact_gs_energy", payload.get("exact_energy")),
        "delta_E_abs": abs_delta_e,
        "abs_delta_e": abs_delta_e,
        "reference_energy_status": reference_audit.get("reference_energy_status", "unavailable"),
        "reference_state_energy": reference_energy,
        "reference_abs_delta_e": reference_abs_delta_e,
        "improvement_over_reference_abs_delta_e": improvement_over_reference,
        "beats_reference_state": beats_reference,
        "benchmark_audit_flags": benchmark_audit_flags,
        "algorithm_spec_optimizer": str(algorithm.optimizer or ""),
        "configured_optimizer": actual_optimizer,
        "actual_optimizer": actual_optimizer,
        "nfev": payload.get("nfev_total", payload.get("nfev")),
        "shots_total": payload.get("shots_total"),
        "shots_per_pauli_term_proxy": payload.get("shots_per_pauli_term_proxy"),
        "shot_proxy_formula": payload.get("shot_proxy_formula"),
        "static_shot_estimate_status": payload.get("static_shot_estimate_status"),
        "hamiltonian_pauli_term_count": payload.get("hamiltonian_pauli_term_count"),
        "energy_eval_count_proxy": payload.get("energy_eval_count_proxy"),
        "compiled_depth_total": payload.get("compiled_depth_total"),
        "compiled_count_2q_total": payload.get("compiled_count_2q_total"),
        "compiled_op_counts": payload.get("compiled_op_counts"),
        "compiled_circuit_stats_status": payload.get("compiled_circuit_stats_status"),
        "circuit_depth": payload.get("circuit_depth"),
        "count_2q": payload.get("count_2q"),
        "depth_proxy": payload.get("depth_proxy"),
        "phase3_controller_called": payload.get("phase3_controller_called", False),
        "phase3_emulation": payload.get("phase3_emulation", False),
        "uses_exact_for_decision": payload.get("uses_exact_for_decision", False),
        "algorithm_origin": payload.get("algorithm_origin"),
        "nit": payload.get("nit"),
        "num_parameters": payload.get("num_parameters"),
        "selected_operator_count": payload.get("selected_operator_count"),
        "subspace_dimension": payload.get("subspace_dimension"),
        "avqite_steps_completed": payload.get("avqite_steps_completed"),
        "avqite_stop_reason": payload.get("avqite_stop_reason", ""),
        "imaginary_time_total": payload.get("imaginary_time_total"),
        "vqe_reps": vqe_reps,
        "vqe_restarts": vqe_restarts,
        "vqe_maxiter": vqe_maxiter,
        "optimizer": payload.get("optimizer", conventional_config.get("optimizer", "")),
        "optimizer_success": payload.get("optimizer_success"),
        "optimizer_message": payload.get("optimizer_message", ""),
        "optimizer_decision_energy": payload.get("optimizer_decision_energy"),
        "optimizer_reported_energy": payload.get("optimizer_reported_energy"),
        "converged": payload.get("converged"),
        "adapt_depth_reached": None if is_fixed_ansatz else payload.get("ansatz_depth"),
        "adapt_stop_reason": "" if is_fixed_ansatz else str(payload.get("stop_reason", "")),
        "continuation_mode": "" if is_fixed_ansatz else str(algorithm.continuation_mode),
        "adapt_pool": "" if is_fixed_ansatz else str(algorithm.adapt_pool),
        "adapt_reopt_policy": "" if is_fixed_ansatz else str(algorithm.adapt_reopt_policy),
        "boundary": str(case.boundary),
        "ordering": str(case.ordering),
        "boson_encoding": str(case.boson_encoding),
        "include_zero_point": bool(case.include_zero_point),
        "n_ph_max": int(case.n_ph_max),
        "t": float(case.t),
        "u": float(case.u),
        "dv": float(case.dv),
        "omega0": float(case.omega0),
        "g_ep": float(case.g_ep),
        "fermion_sector": "half_filled",
        "exact_target_kind": str(exact_target_kind),
        "artifact_json": str(artifact_path),
        "benchmark_decision_noise_status": payload.get("benchmark_decision_noise_status"),
        "benchmark_decision_noise": payload.get("benchmark_decision_noise"),
    }


def _failure_payload(
    *,
    case: HHBenchmarkCase,
    algorithm: HHBenchmarkAlgorithmSpec,
    started_utc: str,
    finished_utc: str,
    runtime_s: float,
    error: BaseException,
    stage: str,
) -> dict[str, Any]:
    return {
        "success": False,
        "benchmark_status": "failed",
        "benchmark_stage": str(stage),
        "benchmark_case": asdict(case),
        "benchmark_algorithm": asdict(algorithm),
        "started_utc": str(started_utc),
        "finished_utc": str(finished_utc),
        "runtime_s": float(runtime_s),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
    }


def _run_one_algorithm(
    *,
    output_dir: Path,
    case: HHBenchmarkCase,
    algorithm: HHBenchmarkAlgorithmSpec,
    resolved_problem: Any,
    exact_gs: float,
    reference_energy_audit: Mapping[str, Any] | None = None,
    benchmark_decision_noise_config: BenchmarkDecisionNoiseConfig | None = None,
) -> dict[str, Any]:
    artifact_path = _payload_path(output_dir, case, algorithm)
    started_utc = _utc_now()
    t0 = time.perf_counter()
    exact_target_kind = str(getattr(getattr(resolved_problem, "exact_target", None), "kind", ""))
    stage_by_kind = {
        "adapt_vqe": "adapt_run",
        "conventional_vqe": "conventional_run",
        "compiled_operator_vqe": "compiled_operator_run",
        "compiled_operator_avqite": "compiled_operator_avqite_run",
        "compiled_operator_qsci": "compiled_operator_qsci_run",
        "compiled_operator_sqd": "compiled_operator_sqd_run",
    }
    stage = stage_by_kind.get(str(algorithm.runner_kind), "benchmark_run")
    try:
        if algorithm.runner_kind == "adapt_vqe":
            payload = _run_one_adapt_algorithm(
                case=case,
                algorithm=algorithm,
                resolved_problem=resolved_problem,
                exact_gs=float(exact_gs),
            )
        elif algorithm.runner_kind == "conventional_vqe":
            payload = _run_one_conventional_algorithm(
                case=case,
                algorithm=algorithm,
                resolved_problem=resolved_problem,
                exact_gs=float(exact_gs),
                benchmark_decision_noise_config=benchmark_decision_noise_config,
            )
        elif algorithm.runner_kind == "compiled_operator_vqe":
            payload = _run_one_compiled_operator_algorithm(
                case=case,
                algorithm=algorithm,
                resolved_problem=resolved_problem,
                exact_gs=float(exact_gs),
                benchmark_decision_noise_config=benchmark_decision_noise_config,
            )
        elif algorithm.runner_kind == "compiled_operator_avqite":
            payload = _run_one_compiled_operator_avqite_algorithm(
                case=case,
                algorithm=algorithm,
                resolved_problem=resolved_problem,
                exact_gs=float(exact_gs),
            )
        elif algorithm.runner_kind == "compiled_operator_qsci":
            payload = _run_one_compiled_operator_qsci_algorithm(
                case=case,
                algorithm=algorithm,
                resolved_problem=resolved_problem,
                exact_gs=float(exact_gs),
            )
        elif algorithm.runner_kind == "compiled_operator_sqd":
            payload = _run_one_compiled_operator_sqd_algorithm(
                case=case,
                algorithm=algorithm,
                resolved_problem=resolved_problem,
                exact_gs=float(exact_gs),
            )
        else:
            raise ValueError(f"Unknown HH benchmark runner_kind={algorithm.runner_kind!r}")
        finished_utc = _utc_now()
        runtime_s = float(time.perf_counter() - t0)
        artifact_payload = {k: v for k, v in payload.items() if not str(k).startswith("_")}
        artifact_payload.update(
            {
                "benchmark_status": _status_from_payload(payload),
                "benchmark_schema": RUNNER_SCHEMA_VERSION,
                "benchmark_case": asdict(case),
                "benchmark_algorithm": asdict(algorithm),
                "started_utc": started_utc,
                "finished_utc": finished_utc,
                "runtime_s": runtime_s,
                "hamiltonian_id": case.case_id,
                "benchmark_reference_energy_audit": dict(reference_energy_audit or {}),
            }
        )
        _write_json(artifact_path, artifact_payload)
        return _raw_row_from_payload(
            case=case,
            algorithm=algorithm,
            payload=payload,
            artifact_path=artifact_path,
            started_utc=started_utc,
            finished_utc=finished_utc,
            runtime_s=runtime_s,
            exact_target_kind=exact_target_kind,
            reference_energy_audit=reference_energy_audit,
        )
    except Exception as exc:
        finished_utc = _utc_now()
        runtime_s = float(time.perf_counter() - t0)
        payload = _failure_payload(
            case=case,
            algorithm=algorithm,
            started_utc=started_utc,
            finished_utc=finished_utc,
            runtime_s=runtime_s,
            error=exc,
            stage=stage,
        )
        _write_json(artifact_path, payload)
        return _raw_row_from_payload(
            case=case,
            algorithm=algorithm,
            payload=payload,
            artifact_path=artifact_path,
            started_utc=started_utc,
            finished_utc=finished_utc,
            runtime_s=runtime_s,
            exact_target_kind=exact_target_kind,
            reference_energy_audit=reference_energy_audit,
        )


def _decision_noise_rows_payload(
    rows: Sequence[Mapping[str, Any]],
    *,
    decision_noise_metadata: Mapping[str, Any] | None,
) -> Any:
    if not isinstance(decision_noise_metadata, Mapping):
        return list(rows)
    return {
        "schema": f"{RUNNER_SCHEMA_VERSION}_rows",
        "benchmark_decision_noise_status": decision_noise_metadata.get("status"),
        "benchmark_decision_noise": copy_decision_noise_metadata(decision_noise_metadata),
        "rows": list(rows),
    }


def _aggregate_decision_noise_metadata(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    row_metadata = [
        row.get("benchmark_decision_noise")
        for row in rows
        if isinstance(row.get("benchmark_decision_noise"), Mapping)
    ]
    if not row_metadata:
        return None
    first = copy_decision_noise_metadata(row_metadata[0])
    status_counts: dict[str, int] = {}
    draw_count_by_surface: dict[str, int] = {}
    draw_count_total = 0
    trace_preview: list[Any] = []
    trace_truncated_count = 0
    for meta_raw in row_metadata:
        meta = meta_raw if isinstance(meta_raw, Mapping) else {}
        status = str(meta.get("status") or "missing")
        status_counts[status] = int(status_counts.get(status, 0) + 1)
        try:
            draw_count_total += int(meta.get("draw_count_total") or 0)
        except Exception:
            pass
        by_surface = meta.get("draw_count_by_surface")
        if isinstance(by_surface, Mapping):
            for surface, count in by_surface.items():
                try:
                    draw_count_by_surface[str(surface)] = int(draw_count_by_surface.get(str(surface), 0) + int(count))
                except Exception:
                    continue
        trace = meta.get("trace_preview")
        if isinstance(trace, list) and len(trace_preview) < 32:
            remaining = 32 - len(trace_preview)
            trace_preview.extend(trace[:remaining])
        try:
            trace_truncated_count += int(meta.get("trace_truncated_count") or 0)
        except Exception:
            pass
    aggregate = {
        **first,
        "status": "ok" if set(status_counts) == {"ok"} else "mixed",
        "supported": all(bool(meta.get("supported", False)) for meta in row_metadata if isinstance(meta, Mapping)),
        "applied": all(bool(meta.get("applied", False)) for meta in row_metadata if isinstance(meta, Mapping)),
        "draw_count_total": int(draw_count_total),
        "draw_count_by_surface": dict(sorted(draw_count_by_surface.items())),
        "surfaces_affected": sorted(draw_count_by_surface),
        "trace_preview": trace_preview,
        "trace_truncated_count": int(trace_truncated_count),
        "status_counts": status_counts,
        "row_target_count": int(len(row_metadata)),
        "handled_row_count": int(len(row_metadata)),
        "applied_row_count": int(sum(1 for meta in row_metadata if isinstance(meta, Mapping) and bool(meta.get("applied", False)))),
        "scope": {"family": "hh", "runner": "hh_static_ground_state_benchmark"},
    }
    return copy_decision_noise_metadata(aggregate)


def run_hh_static_ground_state_benchmark(
    *,
    output_dir: str | Path,
    case_ids: Sequence[str] | None = None,
    cases: Sequence[HHBenchmarkCase] | None = None,
    algorithms: Sequence[HHBenchmarkAlgorithmSpec] | None = None,
    benchmark_decision_noise_config: BenchmarkDecisionNoiseConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the selected HH benchmark matrix and write raw/proxy sidecars."""
    output_path = Path(output_dir)
    selected_cases = _select_cases(tuple(cases or canonical_hh_benchmark_cases()), case_ids)
    selected_algorithms = tuple(algorithms or default_hh_benchmark_algorithms())
    decision_noise_config = coerce_benchmark_decision_noise_config(
        benchmark_decision_noise_config,
        family="hh",
        case_id="hh_static_ground_state_benchmark",
        algorithm_id="hh_static_ground_state_benchmark",
    )
    if not selected_cases:
        raise ValueError("At least one HH benchmark case is required.")
    if not selected_algorithms:
        raise ValueError("At least one HH benchmark algorithm is required.")
    if bool(decision_noise_config.enabled):
        unsupported_decision_noise_algorithms = sorted(
            {
                str(algorithm.algorithm_id)
                for algorithm in selected_algorithms
                if str(algorithm.algorithm_id) not in _DECISION_NOISE_SUPPORTED_HH_ALGORITHM_IDS
            }
        )
        if unsupported_decision_noise_algorithms:
            supported = ", ".join(sorted(_DECISION_NOISE_SUPPORTED_HH_ALGORITHM_IDS))
            raise ValueError(
                "benchmark_decision_noise is supported only for HH VQE-style algorithm_ids "
                f"[{supported}] in this slice; unsupported HH algorithm_ids: "
                f"{unsupported_decision_noise_algorithms}"
            )

    manifest_path = output_path / "hh_static_benchmark_manifest.json"
    rows_path = output_path / "hh_static_benchmark_rows.json"
    summary_dir = output_path / "summary"
    manifest = {
        "schema": RUNNER_SCHEMA_VERSION,
        "proxy_schema": PROXY_SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "output_dir": str(output_path),
        "cases": [asdict(case) for case in selected_cases],
        "algorithms": [asdict(algorithm) for algorithm in selected_algorithms],
        "conventions": {
            "problem": "hh",
            "drive": "static/no-drive",
            "boson_encoding": "binary",
            "ordering": "blocked",
            "boundary": "open",
            "fermion_sector": "half_filled",
            "exact_target": "sector_filtered_hh_ed_lanczos_contract",
        },
    }
    _write_json(manifest_path, manifest)

    rows: list[dict[str, Any]] = []
    for case in selected_cases:
        try:
            resolved = resolve_problem_context(case.to_problem_request())
            exact_gs = float(resolved.exact_target.resolve_energy(ai_log=None))
            reference_energy_audit = _case_reference_energy_audit(
                h_poly=getattr(resolved, "hamiltonian"),
                resolved_problem=resolved,
                exact_gs=exact_gs,
            )
        except Exception as exc:
            for algorithm in selected_algorithms:
                artifact_path = _payload_path(output_path, case, algorithm)
                started_utc = _utc_now()
                finished_utc = started_utc
                payload = _failure_payload(
                    case=case,
                    algorithm=algorithm,
                    started_utc=started_utc,
                    finished_utc=finished_utc,
                    runtime_s=0.0,
                    error=exc,
                    stage="case_setup",
                )
                _write_json(artifact_path, payload)
                rows.append(
                    _raw_row_from_payload(
                        case=case,
                        algorithm=algorithm,
                        payload=payload,
                        artifact_path=artifact_path,
                        started_utc=started_utc,
                        finished_utc=finished_utc,
                        runtime_s=0.0,
                        exact_target_kind="",
                    )
                )
            decision_noise_metadata = (
                _aggregate_decision_noise_metadata(rows)
                if bool(decision_noise_config.enabled)
                else None
            )
            _write_json(rows_path, _decision_noise_rows_payload(rows, decision_noise_metadata=decision_noise_metadata))
            continue

        for algorithm in selected_algorithms:
            print(
                f"[hh-static-benchmark] case={case.case_id} algorithm={algorithm.algorithm_id}",
                flush=True,
            )
            rows.append(
                _run_one_algorithm(
                    output_dir=output_path,
                    case=case,
                    algorithm=algorithm,
                    resolved_problem=resolved,
                    exact_gs=exact_gs,
                    reference_energy_audit=reference_energy_audit,
                    benchmark_decision_noise_config=(
                        decision_noise_config
                        if bool(decision_noise_config.enabled)
                        and str(algorithm.algorithm_id) in _DECISION_NOISE_SUPPORTED_HH_ALGORITHM_IDS
                        else None
                    ),
                )
            )
            decision_noise_metadata = (
                _aggregate_decision_noise_metadata(rows)
                if bool(decision_noise_config.enabled)
                else None
            )
            _write_json(rows_path, _decision_noise_rows_payload(rows, decision_noise_metadata=decision_noise_metadata))

    sidecars = write_proxy_sidecars(
        rows,
        summary_dir,
        summary_extras={
            "benchmark_schema": RUNNER_SCHEMA_VERSION,
            "case_ids": [case.case_id for case in selected_cases],
            "algorithm_ids": [algorithm.algorithm_id for algorithm in selected_algorithms],
            "audit_fields": [
                "reference_state_energy",
                "reference_abs_delta_e",
                "improvement_over_reference_abs_delta_e",
                "beats_reference_state",
                "benchmark_audit_flags",
                "quality_status",
                "algorithm_spec_optimizer",
                "configured_optimizer",
                "actual_optimizer",
            ],
        },
    )
    failed_rows = [row for row in rows if str(row.get("status", "")).lower() != "ok"]
    decision_noise_metadata = (
        _aggregate_decision_noise_metadata(rows)
        if bool(decision_noise_config.enabled)
        else None
    )
    result = {
        "schema": RUNNER_SCHEMA_VERSION,
        "proxy_schema": PROXY_SCHEMA_VERSION,
        "output_dir": str(output_path),
        "manifest_json": str(manifest_path),
        "rows_json": str(rows_path),
        "sidecars": {key: str(path) for key, path in sidecars.items()},
        "row_count": len(rows),
        "failed_row_count": len(failed_rows),
        "ok_row_count": len(rows) - len(failed_rows),
        "rows": rows,
    }
    if decision_noise_metadata is not None:
        result.update(
            {
                "benchmark_decision_noise_status": decision_noise_metadata.get("status"),
                "benchmark_decision_noise": copy_decision_noise_metadata(decision_noise_metadata),
            }
        )
    _write_json(output_path / "hh_static_benchmark_result.json", result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the static/no-drive HH ground-state benchmark matrix."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/agent_runs/hh_static_ground_state_benchmark"),
        help="Directory for manifest, rows, per-run JSON payloads, and proxy sidecars.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        default=None,
        help="Canonical HH benchmark case id to include; repeat to run multiple cases.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        result = run_hh_static_ground_state_benchmark(
            output_dir=args.output_dir,
            case_ids=args.case_ids,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2, sort_keys=True))
    if int(result.get("failed_row_count", 0)) > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
