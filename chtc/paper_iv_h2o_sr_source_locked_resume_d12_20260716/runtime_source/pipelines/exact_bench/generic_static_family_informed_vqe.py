#!/usr/bin/env python3
"""Generic family-informed fixed-ansatz VQE benchmark runner.

This is an exact-bench-local fixed-ansatz row.  It builds a non-adaptive,
family-native ansatz from the problem-local ``full_meta`` pool, optimizes all
parameters at once, and resolves exact references only after optimization.
It does not call the Phase3/SNAKE/static ADAPT controller.
"""

from __future__ import annotations

import json
import os
import time
from argparse import Namespace
from dataclasses import asdict, dataclass, is_dataclass
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
from pipelines.exact_bench.generic_static_adapt_variants import (
    _POOL_TERM_CAP as _FULL_META_POOL_TERM_CAP,
    _compiled_proxy_stats,
    _hamiltonian_pauli_term_count,
    _import_scipy_minimize,
    _sector_or_unavailable,
    build_full_meta_candidate_pool,
    has_scipy_minimize_support,
)
from pipelines.static_adapt.builders.hh_pool_presets import _classify_hh_full_meta_label
from pipelines.static_adapt.builders.problem_registry import (
    ProblemRequest,
    ResolvedProblemContext,
    resolve_problem_context,
)
from pipelines.static_adapt.optimization.phase3_policy_optuna import (
    HamiltonianBenchmarkSpec,
)
from pipelines.exact_bench.table_i_canonical_cases import (
    table_i_canonical_case_ids,
    table_i_canonical_spec_by_case_id,
)
from pipelines.exact_bench.table_i_qiskit_resource_compile import (
    TableICompileUnavailable,
    compile_table_i_ansatz_terms,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.compiled_polynomial import compile_polynomial_action, energy_via_one_apply
from src.quantum.spsa_optimizer import spsa_minimize
from src.quantum.vqe_latex_python_pairs import AnsatzTerm

SCHEMA_VERSION = "generic_static_family_informed_vqe_v1"
_METHOD_ID = "static_family_informed_vqe"
_RUNNER_MODULE = "pipelines.exact_bench.generic_static_family_informed_vqe"
_QUBIT_CAP = 10
_POOL_TERM_CAP = int(_FULL_META_POOL_TERM_CAP)
_RESOURCE_QUBIT_CAP_ENV = "GENERIC_STATIC_TABLE_RESOURCE_QUBIT_CAP"
_RESOURCE_POOL_TERM_CAP_ENV = "GENERIC_STATIC_TABLE_RESOURCE_POOL_TERM_CAP"
_DEFAULT_MAX_ANSATZ_TERMS = 12
_DEFAULT_OPTIMIZER_MAXITER = 250
_DEFAULT_SHOTS_PER_PAULI_TERM_PROXY = 1024
_SHOT_PROXY_FORMULA = "shots_total = shots_per_pauli_term_proxy * hamiltonian_pauli_term_count * energy_eval_count_proxy"
_POLICY_MATCH_VERSION = "family_informed_explicit_policy_v1"
_OPTIMIZER_BFGS = "scipy.optimize.minimize:BFGS"
_OPTIMIZER_SPSA = "repo_native_spsa:spsa_minimize"
_NATIVE_SPSA_DEFAULTS = {
    "family_informed_spsa_a": 0.2,
    "family_informed_spsa_c": 0.1,
    "family_informed_spsa_alpha": 0.602,
    "family_informed_spsa_gamma": 0.101,
    "family_informed_spsa_big_a": 10.0,
    "family_informed_spsa_eval_repeats": 1,
    "family_informed_spsa_avg_last": 0,
}


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
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{field} must be a positive finite float; got {value!r}.")
    return float(out)


def _nonnegative_int(value: int | str | None, *, field: str) -> int:
    try:
        out = int(value)  # type: ignore[arg-type]
    except Exception as exc:
        raise ValueError(f"{field} must be a nonnegative integer; got {value!r}.") from exc
    if out < 0:
        raise ValueError(f"{field} must be a nonnegative integer; got {value!r}.")
    return int(out)


def _native_spsa_kwargs_from_schedule(schedule: Mapping[str, Any]) -> dict[str, Any]:
    mapping = {
        "family_informed_spsa_a": "a",
        "family_informed_spsa_c": "c",
        "family_informed_spsa_alpha": "alpha",
        "family_informed_spsa_gamma": "gamma",
        "family_informed_spsa_big_a": "A",
        "family_informed_spsa_eval_repeats": "eval_repeats",
        "family_informed_spsa_avg_last": "avg_last",
    }
    out: dict[str, Any] = {}
    for source, target in mapping.items():
        value = schedule.get(source)
        if value is not None:
            out[target] = value
    return out


def _normalize_optimizer_settings(
    *,
    optimizer_maxiter: int,
    seed: int,
    optimizer_kind: str | None = None,
    family_informed_optimizer: str | None = None,
    family_informed_spsa_maxiter: int | None = None,
    family_informed_spsa_seed: int | None = None,
    family_informed_spsa_a: float | str | None = None,
    family_informed_spsa_c: float | str | None = None,
    family_informed_spsa_alpha: float | str | None = None,
    family_informed_spsa_gamma: float | str | None = None,
    family_informed_spsa_big_a: float | str | None = None,
    family_informed_spsa_eval_repeats: int | str | None = None,
    family_informed_spsa_avg_last: int | str | None = None,
    optimizer_profile: str | None = None,
    optimizer_profile_source: str | None = None,
    optimizer_overlay_source: str | None = None,
) -> dict[str, Any]:
    raw_schedule = {
        "family_informed_spsa_a": _blank_to_none(family_informed_spsa_a),
        "family_informed_spsa_c": _blank_to_none(family_informed_spsa_c),
        "family_informed_spsa_alpha": _blank_to_none(family_informed_spsa_alpha),
        "family_informed_spsa_gamma": _blank_to_none(family_informed_spsa_gamma),
        "family_informed_spsa_big_a": _blank_to_none(family_informed_spsa_big_a),
        "family_informed_spsa_eval_repeats": _blank_to_none(family_informed_spsa_eval_repeats),
        "family_informed_spsa_avg_last": _blank_to_none(family_informed_spsa_avg_last),
    }
    schedule_requested = any(value is not None for value in raw_schedule.values())
    profile = normalize_paper_i_main_tables_spsa_profile(optimizer_profile)
    if profile == PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID and family_informed_optimizer in {None, ""} and optimizer_kind in {None, ""}:
        requested = "spsa"
        optimizer_source = "optimizer_profile_default"
    else:
        requested = str(
            family_informed_optimizer
            if family_informed_optimizer not in {None, ""}
            else optimizer_kind
            if optimizer_kind not in {None, ""}
            else "bfgs"
        ).strip().lower()
        optimizer_source = (
            "family_informed_optimizer"
            if family_informed_optimizer not in {None, ""}
            else "optimizer_kind"
            if optimizer_kind not in {None, ""}
            else "default"
        )
    if requested not in {"bfgs", "spsa"}:
        raise ValueError(f"family-informed optimizer must be one of {{bfgs, spsa}}; got {requested!r}.")
    if profile == PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID and requested != "spsa":
        raise ValueError(f"optimizer_profile={profile} requires family-informed optimizer=spsa; got {requested!r}.")
    if schedule_requested and requested != "spsa":
        names = ", ".join(sorted(field for field, value in raw_schedule.items() if value is not None))
        raise ValueError(f"family-informed SPSA schedule fields {{{names}}} require optimizer=spsa; got {requested!r}.")
    schedule: dict[str, Any] = {field: None for field in raw_schedule}
    schedule_sources: dict[str, str] | None = None
    schedule_kwargs: dict[str, Any] = {}
    if requested == "spsa":
        schedule = dict(_NATIVE_SPSA_DEFAULTS)
        schedule_sources = {field: "native_default" for field in raw_schedule}
        effective_maxiter = _positive_int(
            family_informed_spsa_maxiter if family_informed_spsa_maxiter is not None else optimizer_maxiter,
            field="family_informed_spsa_maxiter" if family_informed_spsa_maxiter is not None else "optimizer_maxiter",
        )
        effective_seed = _positive_int(
            family_informed_spsa_seed if family_informed_spsa_seed is not None else seed,
            field="family_informed_spsa_seed" if family_informed_spsa_seed is not None else "seed",
        )
        optimizer_name = _OPTIMIZER_SPSA
        for field in (
            "family_informed_spsa_a",
            "family_informed_spsa_c",
            "family_informed_spsa_alpha",
            "family_informed_spsa_gamma",
            "family_informed_spsa_big_a",
        ):
            if raw_schedule[field] is not None:
                schedule[field] = _positive_float(raw_schedule[field], field=field)
                schedule_sources[field] = "explicit"
        if raw_schedule["family_informed_spsa_eval_repeats"] is not None:
            schedule["family_informed_spsa_eval_repeats"] = _positive_int(
                raw_schedule["family_informed_spsa_eval_repeats"],
                field="family_informed_spsa_eval_repeats",
            )
            schedule_sources["family_informed_spsa_eval_repeats"] = "explicit"
        if raw_schedule["family_informed_spsa_avg_last"] is not None:
            schedule["family_informed_spsa_avg_last"] = _nonnegative_int(
                raw_schedule["family_informed_spsa_avg_last"],
                field="family_informed_spsa_avg_last",
            )
            schedule_sources["family_informed_spsa_avg_last"] = "explicit"
        schedule_kwargs = _native_spsa_kwargs_from_schedule(
            {field: value for field, value in schedule.items() if raw_schedule[field] is not None}
        )
    else:
        effective_maxiter = _positive_int(optimizer_maxiter, field="optimizer_maxiter")
        effective_seed = _positive_int(seed, field="seed")
        optimizer_name = _OPTIMIZER_BFGS
    return {
        "optimizer_kind": requested,
        "optimizer": optimizer_name,
        "optimizer_source": optimizer_source,
        "optimizer_profile": profile,
        "optimizer_profile_source": optimizer_profile_source if profile is not None else None,
        "optimizer_overlay_source": optimizer_overlay_source,
        "optimizer_maxiter": int(effective_maxiter),
        "optimizer_seed": int(effective_seed),
        "family_informed_spsa_maxiter": int(effective_maxiter) if requested == "spsa" else None,
        "family_informed_spsa_seed": int(effective_seed) if requested == "spsa" else None,
        "family_informed_spsa_schedule_sources": schedule_sources,
        "family_informed_spsa_schedule_kwargs": schedule_kwargs,
        **schedule,
    }


def _resource_cap_from_env(name: str, default: int | None) -> int | None:
    raw = os.environ.get(str(name), "")
    if raw is None or str(raw).strip() == "":
        return default
    key = str(raw).strip().lower()
    if key in {"0", "none", "off", "false", "unbounded", "unlimited"}:
        return None
    value = int(key)
    if value < 1:
        return None
    return int(value)


def _native_spsa_kwargs_from_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    return dict(settings.get("family_informed_spsa_schedule_kwargs") or {})


GENERIC_STATIC_FAMILY_INFORMED_VQE_FAMILIES: tuple[str, ...] = (
    "hh",
    "hubbard",
    "ionic_hubbard",
    "extended_hubbard",
    "ttprime_hubbard",
    "spinless_tv",
    "spin_boson",
    "bose_hubbard",
    "harmonic_kerr_chain",
    "molecular_vibronic_h2",
)

FAMILY_INFORMED_FIXED_ANSATZ_POLICY: dict[str, dict[str, str]] = {
    "hubbard": {"ansatz_family": "uccsd_style", "rationale": "spinful fermionic fixed baseline"},
    "ionic_hubbard": {"ansatz_family": "uccsd_style", "rationale": "spinful fermionic fixed baseline"},
    "extended_hubbard": {"ansatz_family": "uccsd_style", "rationale": "spinful fermionic fixed baseline"},
    "ttprime_hubbard": {"ansatz_family": "uccsd_style", "rationale": "spinful fermionic fixed baseline"},
    "spinless_tv": {"ansatz_family": "number_conserving_hopping_style", "rationale": "spinless fermion fixed baseline"},
    "molecular_vibronic_h2": {"ansatz_family": "lang_firsov_uccsd_quadrature_hybrid", "rationale": "molecular vibronic mixed fixed baseline"},
    "bose_hubbard": {"ansatz_family": "quadrature_or_hva_style", "rationale": "bosonic fixed baseline"},
    "harmonic_kerr_chain": {"ansatz_family": "quadrature_or_hva_style", "rationale": "bosonic oscillator fixed baseline"},
    "hh": {"ansatz_family": "lang_firsov_uccsd_quadrature_hybrid", "rationale": "mixed electron-phonon fixed baseline"},
    "spin_boson": {"ansatz_family": "spin_boson_displacement_hybrid", "rationale": "mixed spin-boson fixed baseline"},
}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
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


def default_static_family_informed_vqe_case_ids(family: str) -> tuple[str, ...]:
    """Return canonical Paper-I Table-I cases for this fixed baseline."""
    family_key = str(family).strip()
    if family_key not in GENERIC_STATIC_FAMILY_INFORMED_VQE_FAMILIES:
        return ()
    return table_i_canonical_case_ids(family_key)


def _spec_by_case_id(family: str, case_id: str) -> HamiltonianBenchmarkSpec:
    family_key = str(family).strip()
    case_key = str(case_id).strip()
    if family_key not in GENERIC_STATIC_FAMILY_INFORMED_VQE_FAMILIES:
        raise ValueError(f"{_METHOD_ID} is not implemented for family={family_key!r}")
    if case_key not in default_static_family_informed_vqe_case_ids(family_key):
        raise ValueError(f"{_METHOD_ID} is not implemented for {family_key}/{case_key}")
    return with_molecular_vibronic_h2_fixture_override(
        table_i_canonical_spec_by_case_id(family_key, case_key),
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
    return resolve_problem_context(ProblemRequest.from_namespace(_namespace_from_base_args(spec.base_pipeline_args)))


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


def _spec_metadata(spec: HamiltonianBenchmarkSpec) -> dict[str, Any]:
    features = getattr(spec, "features", None)
    return {
        "benchmark_id": str(spec.benchmark_id),
        "family": str(spec.family),
        "base_pipeline_args": list(spec.base_pipeline_args),
        "split": str(spec.split),
        "tags": list(getattr(spec, "tags", ())),
        "features": asdict(features) if is_dataclass(features) else _json_default(features),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _source_fields(**overrides: Any) -> dict[str, Any]:
    return comparator_source_fields(_METHOD_ID, runner_module=_RUNNER_MODULE, **overrides)


def _write_artifacts(output_dir: Path, payload: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_with_source = dict(payload)
    payload_with_source.setdefault("comparator_source", _source_fields())
    rows_payload = {"schema": f"{SCHEMA_VERSION}_rows", "rows": list(rows)}
    if isinstance(payload_with_source.get("benchmark_decision_noise"), Mapping):
        rows_payload.update(
            {
                "benchmark_decision_noise_status": payload_with_source.get("benchmark_decision_noise_status"),
                "benchmark_decision_noise": copy_decision_noise_metadata(payload_with_source["benchmark_decision_noise"]),
            }
        )
    _write_json(output_dir / "result.json", payload_with_source)
    _write_json(output_dir / "rows.json", rows_payload)
    _write_json(output_dir / "manifest.json", {"schema": f"{SCHEMA_VERSION}_manifest", **{k: v for k, v in payload_with_source.items() if k != "schema"}})
    _write_json(output_dir / "generic_static_single.json", payload_with_source)
    write_proxy_sidecars(rows, output_dir, summary_extras={"schema_source": SCHEMA_VERSION})
    return dict(payload_with_source)


def _guardrails(*, exact_reference_usage: str) -> dict[str, Any]:
    return {
        "uses_exact_for_decision": False,
        "uses_reference_for_decision": False,
        "exact_reference_usage": str(exact_reference_usage),
        "phase3_controller_called": False,
        "static_adapt_controller_boundary": "not_called",
        "phase3_emulation": False,
        "runner_boundary": "pipelines.exact_bench.generic_static_family_informed_vqe_only",
        "pool_source": "problem_local_full_meta_pool_filtered_family_informed",
        "pool_name": "family_informed_full_meta_subset",
        "taxonomy_role": "fixed_ansatz_comparator",
        "adapt_append_only": False,
    }


def _base_row(*, family: str, case_id: str, status: str, started_utc: str, finished_utc: str) -> dict[str, Any]:
    return {
        "run_id": f"{case_id}::{_METHOD_ID}",
        "schema": SCHEMA_VERSION,
        "family": family,
        "problem": family,
        "hamiltonian_id": case_id,
        "case_id": case_id,
        "algorithm_id": _METHOD_ID,
        "method_id": _METHOD_ID,
        "method_label": "family-informed VQE",
        "method_kind": "family_informed_fixed_ansatz_vqe",
        "ansatz_name": "benchmark_local_family_informed_full_meta_subset_fixed_vqe",
        "algorithm_origin": "benchmark_local_fixed_ansatz_statevector_vqe",
        "status": status,
        "uses_exact_for_decision": False,
        "uses_reference_for_decision": False,
        "exact_reference_usage": "reporting_only_after_optimization",
        "phase3_controller_called": False,
        "static_adapt_controller_boundary": "not_called",
        "phase3_emulation": False,
        "adapt_append_only": False,
        "pool_source": "problem_local_full_meta_pool_filtered_family_informed",
        "pool_name": "family_informed_full_meta_subset",
        "taxonomy_role": "fixed_ansatz_comparator",
        "pauli_ordering": "left-to-right q_(n-1)...q_0; qubit 0 rightmost",
        "internal_pauli_alphabet": "e/x/y/z",
        "shots_total": 0,
        "static_shot_estimate_status": "not_applicable_not_completed",
        "shot_proxy_formula": _SHOT_PROXY_FORMULA,
        "shots_per_pauli_term_proxy": _DEFAULT_SHOTS_PER_PAULI_TERM_PROXY,
        "hamiltonian_pauli_term_count": 0,
        "energy_eval_count_proxy": 0,
        "compiled_depth_total": 0,
        "compiled_count_2q_total": 0,
        "compiled_op_counts": {},
        "compiled_circuit_stats_status": "not_applicable_not_completed",
        **_source_fields(),
        "started_utc": started_utc,
        "finished_utc": finished_utc,
    }


def _failure_payload(*, family: str, case_id: str, output_dir: Path, reason: str, exception_type: str, started_utc: str) -> dict[str, Any]:
    finished = _utc_now()
    row = _base_row(family=family, case_id=case_id, status="failed", started_utc=started_utc, finished_utc=finished)
    row.update({"reason": reason, "exception_type": exception_type, "energy": None, "exact_energy": None, "delta_E_abs": None})
    payload = {
        "schema": SCHEMA_VERSION,
        "family": family,
        "case_id": case_id,
        "algorithm_id": _METHOD_ID,
        "method_id": _METHOD_ID,
        "status": "failed",
        "reason": reason,
        "exception_type": exception_type,
        "runner": "pipelines.exact_bench.generic_static_family_informed_vqe.run_static_family_informed_vqe_single",
        "table_i": {"tex_label": "tab:benchmark_suite", "table_i_canonical_suite": True, "first_slice": False, "sweep_complete": False},
        "guardrails": _guardrails(exact_reference_usage="reporting_only_after_optimization_or_not_reached"),
        "result": row,
        "rows": [row],
        "started_utc": started_utc,
        "finished_utc": finished,
    }
    return _write_artifacts(output_dir, payload, [row])


def _skip_payload(*, family: str, case_id: str, output_dir: Path, reason: str, started_utc: str) -> dict[str, Any]:
    finished = _utc_now()
    row = _base_row(family=family, case_id=case_id, status="skipped_optional_dependency", started_utc=started_utc, finished_utc=finished)
    row.update({"reason": reason, "exact_reference_usage": "not_resolved_for_dependency_skip"})
    payload = {
        "schema": SCHEMA_VERSION,
        "family": family,
        "case_id": case_id,
        "algorithm_id": _METHOD_ID,
        "method_id": _METHOD_ID,
        "status": "skipped_optional_dependency",
        "reason": reason,
        "runner": "pipelines.exact_bench.generic_static_family_informed_vqe.run_static_family_informed_vqe_single",
        "table_i": {"tex_label": "tab:benchmark_suite", "table_i_canonical_suite": True, "first_slice": False, "sweep_complete": False},
        "guardrails": _guardrails(exact_reference_usage="not_resolved_for_dependency_skip"),
        "result": row,
        "rows": [row],
        "started_utc": started_utc,
        "finished_utc": finished,
    }
    return _write_artifacts(output_dir, payload, [row])


def _resource_guard_payload(*, family: str, case_id: str, output_dir: Path, spec: HamiltonianBenchmarkSpec, started_utc: str, reason: str, guard: Mapping[str, Any]) -> dict[str, Any]:
    finished = _utc_now()
    row = _base_row(family=family, case_id=case_id, status="skipped_resource_guard", started_utc=started_utc, finished_utc=finished)
    row.update({
        "reason": reason,
        "num_qubits": guard.get("num_qubits"),
        "pool_term_count": guard.get("pool_term_count"),
        "resource_guard": True,
        "resource_guard_kind": guard.get("resource_guard_kind"),
        "family_informed_policy_match_version": _POLICY_MATCH_VERSION,
        "family_informed_policy_match_status": guard.get("family_informed_policy_match_status", "not_completed_resource_guard"),
        "family_informed_approved_pool_count": guard.get("family_informed_approved_pool_count"),
        "family_informed_rejected_pool_count": guard.get("family_informed_rejected_pool_count"),
        "family_informed_dropped_ham_full_count": guard.get("family_informed_dropped_ham_full_count"),
    })
    payload = {
        "schema": SCHEMA_VERSION,
        "family": family,
        "case_id": case_id,
        "algorithm_id": _METHOD_ID,
        "method_id": _METHOD_ID,
        "status": "skipped_resource_guard",
        "reason": reason,
        "runner": "pipelines.exact_bench.generic_static_family_informed_vqe.run_static_family_informed_vqe_single",
        "table_i": {"tex_label": "tab:benchmark_suite", "table_i_canonical_suite": True, "first_slice": False, "sweep_complete": False},
        "guardrails": _guardrails(exact_reference_usage="not_resolved_resource_guard"),
        "resource_guard": dict(guard),
        "spec": _spec_metadata(spec),
        "result": row,
        "rows": [row],
        "started_utc": started_utc,
        "finished_utc": finished,
    }
    return _write_artifacts(output_dir, payload, [row])



@dataclass(frozen=True)
class _FamilyInformedPolicyMatch:
    policy_class: str
    priority: int
    reason: str


@dataclass(frozen=True)
class _FamilyInformedSelection:
    selected: tuple[Any, ...]
    selected_matches: tuple[_FamilyInformedPolicyMatch, ...]
    approved_pool_count: int
    rejected_pool_count: int
    dropped_ham_full_count: int


def _reject_common_family_informed_label(label: str) -> bool:
    lab = str(label)
    return (
        lab == "ham_full"
        or lab.startswith("ham_term(")
        or lab.startswith("ham_unit_term(")
        or lab.startswith("hh_termwise_ham_unit_term(")
    )


def _family_informed_policy_match(family: str, label: str) -> _FamilyInformedPolicyMatch | None:
    """Return the explicit family-informed policy class for a full_meta label.

    This is deliberately fail-closed.  It approves only declared family-native
    operator classes, not generic Pauli words that happen to contain useful
    letters such as X/Y or broad labels such as raw Hamiltonian terms.
    """
    family_key = str(family).strip()
    lab = str(label)
    if _reject_common_family_informed_label(lab):
        return None

    if family_key in {
        "hubbard",
        "ionic_hubbard",
        "extended_hubbard",
        "ttprime_hubbard",
    }:
        if lab.startswith("uccsd_dbl("):
            return _FamilyInformedPolicyMatch("uccsd_dbl", 0, "spinful fermionic UCCSD doubles")
        if lab.startswith("uccsd_sing("):
            return _FamilyInformedPolicyMatch("uccsd_sing", 1, "spinful fermionic UCCSD singles")
        return None

    if family_key == "spinless_tv":
        if lab.startswith("ham_quad::"):
            return _FamilyInformedPolicyMatch("spinless_hamiltonian_quadrature", 0, "spinless number-conserving Hamiltonian quadrature")
        if lab.startswith("hva_term::"):
            return _FamilyInformedPolicyMatch("spinless_hva_term", 1, "spinless Hamiltonian variational term")
        if lab.startswith("ham_block::"):
            return _FamilyInformedPolicyMatch("spinless_hamiltonian_block", 2, "spinless Hamiltonian block")
        return None

    if family_key in {"bose_hubbard", "harmonic_kerr_chain"}:
        if lab.startswith("full_meta::"):
            inner = lab.split("::", 1)[1]
            allowed = (
                "x_",
                "p_",
                "n_",
                "x_sq_",
                "p_sq_",
                "n_sq_",
                "squeeze_x_",
                "squeeze_p_",
                "hop_",
                "current_",
                "xx_",
                "pp_",
                "n_x_",
                "n_p_",
                "n_x_sq_",
                "n_p_sq_",
                "x_p_sym_",
                "kerr_",
            )
            if not inner.startswith(allowed):
                return None
            return _FamilyInformedPolicyMatch("bosonic_full_meta_generator", 0, "boson-native full_meta generator")
        if lab.startswith("ham_quad::"):
            return _FamilyInformedPolicyMatch("bosonic_hamiltonian_quadrature", 1, "bosonic Hamiltonian quadrature")
        if lab.startswith("hva_term::"):
            return _FamilyInformedPolicyMatch("bosonic_hva_term", 2, "bosonic Hamiltonian variational term")
        if lab.startswith("ham_block::"):
            return _FamilyInformedPolicyMatch("bosonic_hamiltonian_block", 3, "bosonic Hamiltonian block")
        return None

    if family_key == "spin_boson":
        if lab.startswith("full_meta::"):
            inner = lab.split("::", 1)[1]
            allowed = (
                "emitter_",
                "boson_",
                "longitudinal_",
                "transverse_",
                "number_weighted_",
                "x_sq_",
                "p_sq_",
                "n_sq_",
            )
            if not inner.startswith(allowed):
                return None
            return _FamilyInformedPolicyMatch("spin_boson_full_meta_generator", 0, "spin-boson native full_meta generator")
        if lab.startswith("ham_quad::"):
            return _FamilyInformedPolicyMatch("spin_boson_hamiltonian_quadrature", 1, "spin-boson Hamiltonian quadrature")
        if lab.startswith("hva_term::"):
            return _FamilyInformedPolicyMatch("spin_boson_hva_term", 2, "spin-boson Hamiltonian variational term")
        if lab.startswith("ham_block::"):
            return _FamilyInformedPolicyMatch("spin_boson_hamiltonian_block", 3, "spin-boson Hamiltonian block")
        return None

    if family_key == "molecular_vibronic_h2":
        if lab.startswith("el::uccsd_dbl("):
            return _FamilyInformedPolicyMatch("uccsd_dbl", 0, "molecular vibronic electronic UCCSD doubles")
        if lab.startswith("el::uccsd_sing("):
            return _FamilyInformedPolicyMatch("uccsd_sing", 1, "molecular vibronic electronic UCCSD singles")
        if lab.startswith("boson::"):
            return _FamilyInformedPolicyMatch("vibronic_boson_generator", 2, "molecular vibronic boson generator")
        if lab.startswith("coupled::"):
            return _FamilyInformedPolicyMatch("vibronic_coupled_generator", 3, "molecular vibronic electron-phonon generator")
        return None

    if family_key == "hh":
        hh_class = _classify_hh_full_meta_label(lab)
        priorities = {
            "uccsd_dbl": 0,
            "uccsd_sing": 1,
            "hh_vlf_sq": 2,
            "uccsd_paop_product": 3,
            "uccsd_paop_product_seq_ferm": 4,
            "uccsd_paop_product_seq_motif": 5,
            "paop_cloud_p": 6,
            "paop_cloud_x": 6,
            "paop_disp": 6,
            "paop_dbl": 6,
            "paop_hopdrag": 6,
            "paop_dbl_p": 6,
            "paop_dbl_x": 6,
            "paop_curdrag": 6,
            "paop_hop2": 6,
            "hva_layer": 7,
            "hh_termwise_quadrature": 8,
            "hh_hamiltonian_block": 9,
            "hh_fermionic_reusable": 10,
        }
        if hh_class not in priorities:
            return None
        return _FamilyInformedPolicyMatch(
            str(hh_class),
            int(priorities[hh_class]),
            f"HH full_meta class {hh_class}",
        )

    return None


def _select_family_informed_candidates(
    family: str,
    pool: Sequence[Any],
    *,
    max_terms: int,
) -> _FamilyInformedSelection:
    grouped: dict[str, list[tuple[Any, _FamilyInformedPolicyMatch]]] = {}
    dropped_ham_full = 0
    rejected = 0
    for candidate in pool:
        label = str(candidate.label)
        if label == "ham_full":
            dropped_ham_full += 1
            continue
        match = _family_informed_policy_match(family, label)
        if match is None:
            rejected += 1
            continue
        grouped.setdefault(match.policy_class, []).append((candidate, match))

    for items in grouped.values():
        items.sort(key=lambda item: str(item[0].label))

    class_order = sorted(
        grouped,
        key=lambda cls: (grouped[cls][0][1].priority, cls),
    )
    selected_pairs: list[tuple[Any, _FamilyInformedPolicyMatch]] = []
    cap = max(1, int(max_terms))
    while len(selected_pairs) < cap and class_order:
        next_order: list[str] = []
        for cls in class_order:
            if len(selected_pairs) >= cap:
                break
            items = grouped.get(cls, [])
            if not items:
                continue
            selected_pairs.append(items.pop(0))
            if items:
                next_order.append(cls)
        class_order = next_order

    return _FamilyInformedSelection(
        selected=tuple(pair[0] for pair in selected_pairs),
        selected_matches=tuple(pair[1] for pair in selected_pairs),
        approved_pool_count=sum(len(items) for items in grouped.values()) + len(selected_pairs),
        rejected_pool_count=int(rejected),
        dropped_ham_full_count=int(dropped_ham_full),
    )


def _select_family_informed_ansatz(family: str, pool: Sequence[Any], *, max_terms: int) -> tuple[Any, ...]:
    return _select_family_informed_candidates(family, pool, max_terms=max_terms).selected

def _optimize_fixed_ansatz(
    *,
    minimize_fn: Any | None,
    selected: Sequence[Any],
    psi_ref: np.ndarray,
    h_compiled: Any,
    pauli_action_cache: dict[str, Any],
    optimizer_maxiter: int,
    optimizer_kind: str = "bfgs",
    spsa_seed: int | None = None,
    spsa_kwargs: Mapping[str, Any] | None = None,
    decision_noise_recorder: BenchmarkDecisionNoiseRecorder | None = None,
) -> tuple[np.ndarray, float, np.ndarray, dict[str, Any]]:
    optimizer_kind_key = str(optimizer_kind).strip().lower()
    if optimizer_kind_key not in {"bfgs", "spsa"}:
        raise ValueError(f"family-informed optimizer_kind must be one of {{bfgs, spsa}}; got {optimizer_kind!r}.")
    if not selected:
        energy, _ = energy_via_one_apply(psi_ref, h_compiled)
        exact_energy = float(energy)
        return (
            np.zeros(0, dtype=float),
            exact_energy,
            psi_ref.copy(),
            {
                "nfev": 1,
                "nit": 0,
                "success": True,
                "message": "empty_ansatz",
                "optimizer": _OPTIMIZER_SPSA if optimizer_kind_key == "spsa" else _OPTIMIZER_BFGS,
                "optimizer_decision_energy": exact_energy,
                "optimizer_reported_energy": exact_energy,
                "spsa_seed": None if optimizer_kind_key != "spsa" else int(spsa_seed if spsa_seed is not None else 0),
            },
        )
    executor = CompiledAnsatzExecutor(
        list(selected),
        coefficient_tolerance=1e-12,
        ignore_identity=True,
        sort_terms=True,
        pauli_action_cache=pauli_action_cache,
    )
    eval_count = 0

    def objective(theta_vec: np.ndarray) -> float:
        nonlocal eval_count
        eval_count += 1
        psi = np.asarray(executor.prepare_state(np.asarray(theta_vec, dtype=float).reshape(-1), psi_ref), dtype=complex).reshape(-1)
        energy, _ = energy_via_one_apply(psi, h_compiled)
        energy_exact = float(energy)
        if decision_noise_recorder is not None:
            return float(
                decision_noise_recorder.apply(
                    energy_exact,
                    surface="family_informed_objective",
                    value_kind="energy",
                    phase="optimizer",
                    extra_scope={"eval_index": int(eval_count), "selected_operator_count": int(len(selected))},
                )
            )
        return energy_exact

    x0 = np.zeros(len(selected), dtype=float)
    if optimizer_kind_key == "spsa":
        schedule_kwargs = dict(spsa_kwargs or {})
        result = spsa_minimize(
            objective,
            x0,
            maxiter=int(optimizer_maxiter),
            seed=int(spsa_seed if spsa_seed is not None else 0),
            bounds=[(-np.pi, np.pi) for _ in range(len(selected))],
            project="clip",
            **schedule_kwargs,
        )
        optimizer_name = _OPTIMIZER_SPSA
    else:
        if minimize_fn is None:
            raise RuntimeError("family-informed BFGS optimizer requires scipy.optimize.minimize")
        result = minimize_fn(objective, x0, method="BFGS", options={"maxiter": int(optimizer_maxiter), "gtol": 1e-8})
        optimizer_name = _OPTIMIZER_BFGS
    theta = np.asarray(getattr(result, "x", x0), dtype=float).reshape(-1)
    psi_final = np.asarray(executor.prepare_state(theta, psi_ref), dtype=complex).reshape(-1)
    fun = getattr(result, "fun", None)
    optimizer_decision_energy = None if fun is None else float(fun)
    energy_final, _ = energy_via_one_apply(psi_final, h_compiled)
    energy = float(energy_final)
    return theta, energy, psi_final, {
        "nfev": int(getattr(result, "nfev", eval_count) or eval_count),
        "nit": int(getattr(result, "nit", 0) or 0),
        "success": bool(getattr(result, "success", False)),
        "message": str(getattr(result, "message", "")),
        "optimizer": optimizer_name,
        "optimizer_decision_energy": optimizer_decision_energy,
        "optimizer_reported_energy": optimizer_decision_energy,
        "spsa_seed": None if optimizer_kind_key != "spsa" else int(spsa_seed if spsa_seed is not None else 0),
    }


def _ansatz_terms_from_selected(selected: Sequence[Any]) -> tuple[AnsatzTerm, ...]:
    return tuple(
        AnsatzTerm(label=str(getattr(candidate, "label")), polynomial=getattr(candidate, "polynomial"))
        for candidate in selected
    )


def _terminal_qiskit_compile_stats(
    *,
    selected: Sequence[Any],
    num_qubits: int,
    reference_state: np.ndarray | Sequence[complex],
) -> dict[str, Any]:
    diagnostic = {"diagnostic_pauli_rotation_proxy_stats": _compiled_proxy_stats(selected)}
    try:
        compiled = compile_table_i_ansatz_terms(
            ops=_ansatz_terms_from_selected(selected),
            num_qubits=int(num_qubits),
            reference_state=reference_state,
            source_kind="qiskit_compiled_terminal_only_fixed_ansatz",
        )
    except TableICompileUnavailable as exc:
        return {
            "compiled_circuit_stats_status": "qiskit_terminal_compile_unavailable",
            "compiled_circuit_stats_error": f"{exc.status}: {exc.reason}",
            "first_hit_cost_source_kind": "qiskit_compiled_terminal_only_fixed_ansatz",
            "compiled_resource_source_kind": "qiskit_compiled_terminal_only_fixed_ansatz",
            "compiled_resource_qiskit_validated": False,
            "qiskit_first_hit_cost_validated": False,
            **diagnostic,
        }
    return {**compiled, **diagnostic}


def _shot_proxy_fields(*, hamiltonian_pauli_term_count: int, energy_eval_count: int | None, shots_per_pauli_term_proxy: int) -> dict[str, Any]:
    h_count = max(0, int(hamiltonian_pauli_term_count))
    energy_count = max(1, int(energy_eval_count or 0))
    shots_per_term = max(0, int(shots_per_pauli_term_proxy))
    return {
        "shots_total": int(shots_per_term * h_count * energy_count),
        "static_shot_estimate_status": "deterministic_proxy_not_physical_shots",
        "shot_proxy_formula": _SHOT_PROXY_FORMULA,
        "shot_proxy_note": "Benchmark-table deterministic proxy only; not a hardware shot allocation.",
        "shots_per_pauli_term_proxy": shots_per_term,
        "hamiltonian_pauli_term_count": h_count,
        "energy_eval_count_proxy": energy_count,
    }


def _run_impl(
    *,
    family: str,
    case_id: str,
    output_dir: Path,
    optimizer_maxiter: int = _DEFAULT_OPTIMIZER_MAXITER,
    max_ansatz_terms: int = _DEFAULT_MAX_ANSATZ_TERMS,
    seed: int = 42,
    shots_per_pauli_term_proxy: int = _DEFAULT_SHOTS_PER_PAULI_TERM_PROXY,
    benchmark_decision_noise_config: BenchmarkDecisionNoiseConfig | Mapping[str, Any] | None = None,
    optimizer_kind: str | None = None,
    optimizer_profile: str | None = None,
    optimizer_profile_source: str | None = None,
    family_informed_optimizer: str | None = None,
    family_informed_spsa_maxiter: int | None = None,
    family_informed_spsa_seed: int | None = None,
    family_informed_spsa_a: float | str | None = None,
    family_informed_spsa_c: float | str | None = None,
    family_informed_spsa_alpha: float | str | None = None,
    family_informed_spsa_gamma: float | str | None = None,
    family_informed_spsa_big_a: float | str | None = None,
    family_informed_spsa_eval_repeats: int | str | None = None,
    family_informed_spsa_avg_last: int | str | None = None,
    optimizer_overlay_source: str | None = None,
) -> dict[str, Any]:
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
    optimizer_settings = _normalize_optimizer_settings(
        optimizer_maxiter=int(optimizer_maxiter),
        seed=int(seed),
        optimizer_kind=optimizer_kind,
        family_informed_optimizer=family_informed_optimizer,
        family_informed_spsa_maxiter=family_informed_spsa_maxiter,
        family_informed_spsa_seed=family_informed_spsa_seed,
        family_informed_spsa_a=family_informed_spsa_a,
        family_informed_spsa_c=family_informed_spsa_c,
        family_informed_spsa_alpha=family_informed_spsa_alpha,
        family_informed_spsa_gamma=family_informed_spsa_gamma,
        family_informed_spsa_big_a=family_informed_spsa_big_a,
        family_informed_spsa_eval_repeats=family_informed_spsa_eval_repeats,
        family_informed_spsa_avg_last=family_informed_spsa_avg_last,
        optimizer_profile=optimizer_profile,
        optimizer_profile_source=optimizer_profile_source,
        optimizer_overlay_source=optimizer_overlay_source,
    )
    output = Path(output_dir)
    started_utc = _utc_now()
    t0 = time.perf_counter()
    spec = _spec_by_case_id(family_key, case_key)
    minimize_fn = None
    if optimizer_settings["optimizer_kind"] == "bfgs":
        if not has_scipy_minimize_support():
            return _skip_payload(family=family_key, case_id=case_key, output_dir=output, reason="optional scipy.optimize.minimize dependency is not importable", started_utc=started_utc)
        minimize_fn = _import_scipy_minimize()
    try:
        np.random.seed(int(seed))
    except Exception:
        pass

    context = _resolve_context_from_spec(spec)
    num_qubits = int(context.layout.total_qubits)
    qubit_cap = _resource_cap_from_env(_RESOURCE_QUBIT_CAP_ENV, _QUBIT_CAP)
    pool_term_cap = _resource_cap_from_env(_RESOURCE_POOL_TERM_CAP_ENV, _POOL_TERM_CAP)
    if qubit_cap is not None and num_qubits > int(qubit_cap):
        guard = {
            "resource_guard": True,
            "resource_guard_kind": "family_informed_vqe_qubit_cap",
            "reason": "family-informed fixed VQE canonical case qubit count exceeds cap",
            "num_qubits": num_qubits,
            "qubit_cap": int(qubit_cap),
            "pool_term_count": 0,
            "pool_term_cap": None if pool_term_cap is None else int(pool_term_cap),
        }
        return _resource_guard_payload(family=family_key, case_id=case_key, output_dir=output, spec=spec, started_utc=started_utc, reason=str(guard["reason"]), guard=guard)
    try:
        pool = build_full_meta_candidate_pool(context, max_terms=pool_term_cap)
    except ValueError as exc:
        if "full_meta pool exceeds cap" not in str(exc):
            raise
        guard = {
            "resource_guard": True,
            "resource_guard_kind": "family_informed_vqe_full_meta_pool_term_cap",
            "reason": "family-informed VQE full_meta pool exceeds cap",
            "num_qubits": num_qubits,
            "qubit_cap": None if qubit_cap is None else int(qubit_cap),
            "pool_term_count": None if pool_term_cap is None else int(pool_term_cap + 1),
            "pool_term_cap": None if pool_term_cap is None else int(pool_term_cap),
        }
        return _resource_guard_payload(family=family_key, case_id=case_key, output_dir=output, spec=spec, started_utc=started_utc, reason=str(guard["reason"]), guard=guard)
    selection = _select_family_informed_candidates(family_key, pool, max_terms=int(max_ansatz_terms))
    selected = selection.selected
    if not selected:
        guard = {
            "resource_guard": True,
            "resource_guard_kind": "family_informed_vqe_no_policy_match",
            "reason": "family-informed policy did not match any full_meta candidates",
            "num_qubits": num_qubits,
            "qubit_cap": None if qubit_cap is None else int(qubit_cap),
            "pool_term_count": int(len(pool)),
            "pool_term_cap": None if pool_term_cap is None else int(pool_term_cap),
        }
        guard.update(
            {
                "family_informed_policy_match_status": "no_policy_match",
                "family_informed_approved_pool_count": selection.approved_pool_count,
                "family_informed_rejected_pool_count": selection.rejected_pool_count,
                "family_informed_dropped_ham_full_count": selection.dropped_ham_full_count,
            }
        )
        return _resource_guard_payload(family=family_key, case_id=case_key, output_dir=output, spec=spec, started_utc=started_utc, reason=str(guard["reason"]), guard=guard)

    psi_ref = np.asarray(context.reference_state.build_state(), dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(psi_ref))
    if norm <= 0.0:
        raise ValueError("reference state has zero norm")
    psi_ref = psi_ref / norm
    pauli_action_cache: dict[str, Any] = {}
    h_compiled = compile_polynomial_action(context.hamiltonian, tol=1e-12, pauli_action_cache=pauli_action_cache)
    theta, energy, psi_final, opt = _optimize_fixed_ansatz(
        minimize_fn=minimize_fn,
        selected=selected,
        psi_ref=psi_ref,
        h_compiled=h_compiled,
        pauli_action_cache=pauli_action_cache,
        optimizer_maxiter=int(optimizer_settings["optimizer_maxiter"]),
        optimizer_kind=str(optimizer_settings["optimizer_kind"]),
        spsa_seed=int(optimizer_settings["optimizer_seed"]),
        spsa_kwargs=_native_spsa_kwargs_from_settings(optimizer_settings),
        decision_noise_recorder=decision_noise_recorder,
    )
    decision_noise_metadata = None
    if bool(decision_noise_config.enabled):
        decision_noise_metadata = copy_decision_noise_metadata(
            decision_noise_recorder.summary(
                status="ok",
                supported=True,
                extra={"runner": "generic_static_family_informed_vqe"},
            )
        )
    exact_energy = _safe_exact_energy(context)
    abs_delta = None if exact_energy is None else abs(float(energy) - float(exact_energy))
    sector = _sector_or_unavailable(context, psi_final)
    h_count = _hamiltonian_pauli_term_count(context.hamiltonian)
    compiled_stats = _terminal_qiskit_compile_stats(
        selected=selected,
        num_qubits=int(num_qubits),
        reference_state=psi_ref,
    )
    shot_proxy = _shot_proxy_fields(hamiltonian_pauli_term_count=h_count, energy_eval_count=opt["nfev"], shots_per_pauli_term_proxy=int(shots_per_pauli_term_proxy))
    finished = _utc_now()
    row = _base_row(family=family_key, case_id=case_key, status="ok", started_utc=started_utc, finished_utc=finished)
    policy = FAMILY_INFORMED_FIXED_ANSATZ_POLICY.get(family_key, {})
    row.update(
        {
            "L": int(context.request.num_sites),
            "energy": float(energy),
            "exact_energy": exact_energy,
            "exact_gs_energy": exact_energy,
            "delta_E_abs": abs_delta,
            "abs_delta_e": abs_delta,
            "infidelity_exact": None,
            "infidelity_status": "not_available_exact_state_not_exposed_by_problem_context",
            "observable_error_status": "not_implemented_static_train_suite",
            "num_qubits": num_qubits,
            "num_parameters": int(theta.size),
            "selected_operator_count": int(len(selected)),
            "pool_term_count": int(len(selected)),
            "full_meta_pool_term_count": int(len(pool)),
            "pool_labels": [str(candidate.label) for candidate in selected],
            "pool_pauli_labels_exyz": {str(candidate.label): list(candidate.pauli_labels_exyz) for candidate in selected},
            "pool_qubit_supports": {str(candidate.label): list(candidate.support) for candidate in selected},
            "family_informed_policy": policy,
            "fixed_ansatz_family": policy.get("ansatz_family"),
            "family_informed_policy_match_version": _POLICY_MATCH_VERSION,
            "family_informed_policy_match_status": "approved_policy_labels",
            "family_informed_policy_classes": sorted({match.policy_class for match in selection.selected_matches}),
            "selected_operator_policy_classes": [match.policy_class for match in selection.selected_matches],
            "family_informed_approved_pool_count": selection.approved_pool_count,
            "family_informed_rejected_pool_count": selection.rejected_pool_count,
            "family_informed_dropped_ham_full_count": selection.dropped_ham_full_count,
            "fixed_ansatz_selection_rule": "family-prioritized fixed subset of the problem-local full_meta pool; no adaptive selection",
            "optimizer": str(opt.get("optimizer") or optimizer_settings["optimizer"]),
            "optimizer_kind": str(optimizer_settings["optimizer_kind"]),
            "optimizer_source": str(optimizer_settings["optimizer_source"]),
            "optimizer_profile": optimizer_settings["optimizer_profile"],
            "optimizer_profile_source": optimizer_settings["optimizer_profile_source"],
            "optimizer_overlay_source": optimizer_settings["optimizer_overlay_source"],
            "optimizer_maxiter": int(optimizer_settings["optimizer_maxiter"]),
            "family_informed_spsa_maxiter": optimizer_settings["family_informed_spsa_maxiter"],
            "family_informed_spsa_seed": optimizer_settings["family_informed_spsa_seed"],
            "family_informed_spsa_a": optimizer_settings["family_informed_spsa_a"],
            "family_informed_spsa_c": optimizer_settings["family_informed_spsa_c"],
            "family_informed_spsa_alpha": optimizer_settings["family_informed_spsa_alpha"],
            "family_informed_spsa_gamma": optimizer_settings["family_informed_spsa_gamma"],
            "family_informed_spsa_big_a": optimizer_settings["family_informed_spsa_big_a"],
            "family_informed_spsa_eval_repeats": optimizer_settings["family_informed_spsa_eval_repeats"],
            "family_informed_spsa_avg_last": optimizer_settings["family_informed_spsa_avg_last"],
            "family_informed_spsa_schedule_sources": optimizer_settings["family_informed_spsa_schedule_sources"],
            "spsa_maxiter": optimizer_settings["family_informed_spsa_maxiter"],
            "spsa_seed": opt.get("spsa_seed"),
            "spsa_a": optimizer_settings["family_informed_spsa_a"],
            "spsa_c": optimizer_settings["family_informed_spsa_c"],
            "spsa_alpha": optimizer_settings["family_informed_spsa_alpha"],
            "spsa_gamma": optimizer_settings["family_informed_spsa_gamma"],
            "spsa_A": optimizer_settings["family_informed_spsa_big_a"],
            "spsa_big_a": optimizer_settings["family_informed_spsa_big_a"],
            "spsa_eval_repeats": optimizer_settings["family_informed_spsa_eval_repeats"],
            "spsa_avg_last": optimizer_settings["family_informed_spsa_avg_last"],
            "optimizer_success": bool(opt["success"]),
            "optimizer_message": str(opt["message"]),
            "optimizer_decision_energy": opt.get("optimizer_decision_energy"),
            "optimizer_reported_energy": opt.get("optimizer_reported_energy"),
            "nfev": int(opt["nfev"]),
            "nit": int(opt["nit"]),
            "runtime_s": float(time.perf_counter() - t0),
            **compiled_stats,
            **shot_proxy,
            "sector_probability": sector.get("sector_probability"),
            "sector_leak_probability": sector.get("sector_leak_probability"),
            "sector_leak_flag": sector.get("sector_leak_flag"),
            "sector_leak_threshold": sector.get("sector_leak_threshold"),
            "boson_legal_probability_min": sector.get("boson_legal_probability_min"),
            "boson_illegal_probability_max": sector.get("boson_illegal_probability_max"),
            "boson_truncation_leak_flag": sector.get("boson_truncation_leak_flag"),
            "boson_subspace_diagnostics": sector.get("boson_subspace_diagnostics"),
            "truncation_diagnostics": sector.get("truncation_constraints_evaluated"),
            "sector_diagnostics": sector,
            "theta": theta.tolist(),
        }
    )
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
        "runner": "pipelines.exact_bench.generic_static_family_informed_vqe.run_static_family_informed_vqe_single",
        "table_i": {"tex_label": "tab:benchmark_suite", "table_i_canonical_suite": True, "first_slice": False, "sweep_complete": False},
        "guardrails": _guardrails(exact_reference_usage="reporting_only_after_optimization"),
        "spec": _spec_metadata(spec),
        "result": row,
        "rows": [row],
        "started_utc": started_utc,
        "finished_utc": finished,
    }
    if decision_noise_metadata is not None:
        payload.update(
            {
                "benchmark_decision_noise_status": "ok",
                "benchmark_decision_noise": copy_decision_noise_metadata(decision_noise_metadata),
            }
        )
    return _write_artifacts(output, payload, [row])


def run_static_family_informed_vqe_single(
    *,
    family: str,
    case_id: str,
    output_dir: Path,
    optimizer_maxiter: int = _DEFAULT_OPTIMIZER_MAXITER,
    max_ansatz_terms: int = _DEFAULT_MAX_ANSATZ_TERMS,
    seed: int = 42,
    shots_per_pauli_term_proxy: int = _DEFAULT_SHOTS_PER_PAULI_TERM_PROXY,
    benchmark_decision_noise_config: BenchmarkDecisionNoiseConfig | Mapping[str, Any] | None = None,
    optimizer_kind: str | None = None,
    optimizer_profile: str | None = None,
    optimizer_profile_source: str | None = None,
    family_informed_optimizer: str | None = None,
    family_informed_spsa_maxiter: int | None = None,
    family_informed_spsa_seed: int | None = None,
    family_informed_spsa_a: float | str | None = None,
    family_informed_spsa_c: float | str | None = None,
    family_informed_spsa_alpha: float | str | None = None,
    family_informed_spsa_gamma: float | str | None = None,
    family_informed_spsa_big_a: float | str | None = None,
    family_informed_spsa_eval_repeats: int | str | None = None,
    family_informed_spsa_avg_last: int | str | None = None,
    optimizer_overlay_source: str | None = None,
) -> dict[str, Any]:
    """Run one family-informed fixed VQE benchmark case and emit artifacts."""
    started_utc = _utc_now()
    try:
        return _run_impl(
            family=family,
            case_id=case_id,
            output_dir=Path(output_dir),
            optimizer_maxiter=optimizer_maxiter,
            max_ansatz_terms=max_ansatz_terms,
            seed=seed,
            shots_per_pauli_term_proxy=shots_per_pauli_term_proxy,
            benchmark_decision_noise_config=benchmark_decision_noise_config,
            optimizer_kind=optimizer_kind,
            optimizer_profile=optimizer_profile,
            optimizer_profile_source=optimizer_profile_source,
            family_informed_optimizer=family_informed_optimizer,
            family_informed_spsa_maxiter=family_informed_spsa_maxiter,
            family_informed_spsa_seed=family_informed_spsa_seed,
            family_informed_spsa_a=family_informed_spsa_a,
            family_informed_spsa_c=family_informed_spsa_c,
            family_informed_spsa_alpha=family_informed_spsa_alpha,
            family_informed_spsa_gamma=family_informed_spsa_gamma,
            family_informed_spsa_big_a=family_informed_spsa_big_a,
            family_informed_spsa_eval_repeats=family_informed_spsa_eval_repeats,
            family_informed_spsa_avg_last=family_informed_spsa_avg_last,
            optimizer_overlay_source=optimizer_overlay_source,
        )
    except Exception as exc:
        return _failure_payload(
            family=str(family).strip(),
            case_id=str(case_id).strip(),
            output_dir=Path(output_dir),
            reason=str(exc),
            exception_type=type(exc).__name__,
            started_utc=started_utc,
        )


__all__ = [
    "FAMILY_INFORMED_FIXED_ANSATZ_POLICY",
    "GENERIC_STATIC_FAMILY_INFORMED_VQE_FAMILIES",
    "SCHEMA_VERSION",
    "default_static_family_informed_vqe_case_ids",
    "run_static_family_informed_vqe_single",
]
