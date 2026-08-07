#!/usr/bin/env python3
"""Post-hoc metric enrichment for generic static Table-I benchmark artifacts.

This module is deliberately downstream of benchmark execution. It reconstructs
reporting-only metrics from immutable benchmark outputs and writes sidecars; it
never mutates raw payloads and never feeds exact references back into algorithm
selection, stopping, or optimization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipelines.exact_bench.generic_static_adapt_variants import (
    _blocked_labels_for_config,
    _get_config,
    _is_geo_config,
    _pool_name_for_config,
    _prepare_selected_state,
    build_full_meta_candidate_pool,
    build_pairwise_qubit_excitation_pool,
)
from pipelines.exact_bench.generic_static_hea_qiskit_vqe import _resolve_context_from_spec as _hea_resolve_context
from pipelines.exact_bench.qiskit_hea_adapter import QiskitHeaUnavailable, build_qiskit_hea_ansatz
from pipelines.exact_bench.qiskit_adaptvqe_adapter import (
    build_reference_state_circuit,
    import_qiskit_adaptvqe_components,
)
from pipelines.exact_bench.table_i_qiskit_resource_compile import (
    TableICompileUnavailable,
    compile_table_i_ansatz_terms,
)
from pipelines.exact_bench.table_i_canonical_cases import table_i_canonical_spec_by_case_id
from pipelines.static_adapt.builders.problem_setup import resolve_exact_reference_state_for_problem
from pipelines.static_adapt.optimization.phase3_policy_optuna import (
    HamiltonianBenchmarkSpec,
    _reference_cutoff_energy_for_spec,
)
from src.quantum.compiled_polynomial import apply_compiled_polynomial, compile_polynomial_action
from src.quantum.hubbard_latex_python_pairs import boson_qubits_per_site
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm
from src.quantum.vqe_latex_python_pairs import expval_pauli_polynomial_one_apply

SCHEMA_VERSION = "generic_static_metric_enrichment_v1"
ENRICHMENT_FILENAME = "generic_static_metric_enrichment.json"
NORMALIZED_MEASUREMENT_WORK_SCHEMA = "normalized_measurement_work_v1"
GROUPED_MEASUREMENT_PROXY_SCHEMA = "grouped_measurement_proxy_v1"
ALGORITHMIC_MEASUREMENT_WORK_SCHEMA = "algorithmic_measurement_work_v1"
PHYSICAL_MEASUREMENT_WORK_SCHEMA = "physical_measurement_work_v1"
STATEVECTOR_VARIANCE_METRIC_SCHEMA = "statevector_grouped_variance_metric_v1"
TABLE_I_EVENT_LEDGER_SCHEMA = "table_i_measurement_event_ledger_v1"
TABLE_I_THRESHOLD_COST_SCHEMA = "table_i_threshold_cost_v1"
DEFAULT_S_NORM_WEIGHTS = {
    "s_H_outer": 1.0,
    "s_g": 1.0,
    "s_F": 1.0,
    "s_H_refit": 1.0,
}
DEFAULT_S_ALG_WEIGHTS = {
    "s_H_outer": 1.0,
    "s_grad": 1.0,
    "s_metric": 1.0,
    "s_H_refit": 1.0,
}

MEASUREMENT_WORK_COMPONENT_ALIASES = {
    "N_H_outer_eval": (
        "S_norm_N_H_outer_eval",
        "measurement_work_N_H_outer_eval",
        "N_H_outer_eval",
        "S_norm_N_H_eval",
        "measurement_work_N_H_eval",
        "N_H_eval",
    ),
    "N_grad": ("N_grad", "measurement_work_N_grad", "S_norm_N_grad"),
    "N_metric": ("N_metric", "measurement_work_N_metric", "S_norm_N_metric"),
    "N_H_refit_eval": (
        "S_norm_N_H_refit_eval",
        "measurement_work_N_H_refit_eval",
        "N_H_refit_eval",
        "S_norm_N_refit_eval",
        "measurement_work_N_refit_eval",
        "N_refit_eval",
    ),
}
GROUPED_MEASUREMENT_COMPONENT_ALIASES = {
    "S_grp_H_outer": ("S_grp_H_outer", "S_grp_H_outer_eval", "grouped_measurement_S_H_outer"),
    "S_grp_grad": ("S_grp_grad", "grouped_measurement_S_grad"),
    "S_grp_metric": ("S_grp_metric", "grouped_measurement_S_metric"),
    "S_grp_H_refit": ("S_grp_H_refit", "S_grp_H_refit_eval", "grouped_measurement_S_H_refit"),
}
MEASUREMENT_WORK_OTHER_QUANTUM_ALIASES = (
    "S_norm_N_other_quantum",
    "measurement_work_N_other_quantum",
    "N_other_quantum",
)
S_ALG_COMPONENT_ALIASES = {
    "N_H_outer_eval": (
        "S_alg_N_H_outer_eval",
        "algorithmic_measurement_work_N_H_outer_eval",
    ),
    "N_grad_probe": (
        "S_alg_N_grad_probe",
        "algorithmic_measurement_work_N_grad_probe",
    ),
    "N_metric_probe": (
        "S_alg_N_metric_probe",
        "algorithmic_measurement_work_N_metric_probe",
    ),
    "N_H_refit_eval": (
        "S_alg_N_H_refit_eval",
        "algorithmic_measurement_work_N_H_refit_eval",
    ),
}
S_ALG_OTHER_QUANTUM_ALIASES = (
    "S_alg_N_other_quantum",
    "algorithmic_measurement_work_N_other_quantum",
    "N_other_algorithmic_quantum",
)
S_PHYS_COMPONENT_ALIASES = {
    "S_phys_H_outer": ("S_phys_H_outer", "physical_measurement_work_S_H_outer"),
    "S_phys_grad": ("S_phys_grad", "physical_measurement_work_S_grad"),
    "S_phys_metric": ("S_phys_metric", "physical_measurement_work_S_metric"),
    "S_phys_H_refit": ("S_phys_H_refit", "physical_measurement_work_S_H_refit"),
}
S_L2_COMPONENT_ALIASES = {
    "S_l2_H_outer": ("S_l2_H_outer", "grouped_l2_measurement_work_S_H_outer"),
    "S_l2_grad": ("S_l2_grad", "grouped_l2_measurement_work_S_grad"),
    "S_l2_metric": ("S_l2_metric", "grouped_l2_measurement_work_S_metric"),
    "S_l2_H_refit": ("S_l2_H_refit", "grouped_l2_measurement_work_S_H_refit"),
}
S_VAR_COMPONENT_ALIASES = {
    "S_var_H_outer": (
        "S_var_H_outer",
        "S_phys_var_H_outer",
        "statevector_variance_measurement_work_S_H_outer",
    ),
    "S_var_grad": (
        "S_var_grad",
        "S_phys_var_grad",
        "statevector_variance_measurement_work_S_grad",
    ),
    "S_var_metric": (
        "S_var_metric",
        "S_phys_var_metric",
        "statevector_variance_measurement_work_S_metric",
    ),
    "S_var_H_refit": (
        "S_var_H_refit",
        "S_phys_var_H_refit",
        "statevector_variance_measurement_work_S_H_refit",
    ),
}
MEASUREMENT_WORK_SPLIT_SUFFIXES = ("request", "fresh", "cache")


_ADAPT_VARIANT_IDS = {
    "static_full_meta_append_adapt_vqe",
    "static_qubit_qeb_adapt_vqe",
    "static_tetris_qubit_adapt_vqe",
    "static_geo_qubit_adapt_vqe",
    "static_geo_qeb_adapt_vqe",
    "static_geo_adapt_vqe",
    "static_pos_geo_adapt_vqe",
}
_FIXED_TERMINAL_TABLE_I_METHOD_IDS = {
    "static_hea_qiskit_vqe",
    "static_family_informed_vqe",
}
SNAKE_TABLE_I_ALGORITHM_ID = "static_family_native_adapt_phase3"
SNAKE_FIRST_CROSSING_COST_SCHEMA = "snake_first_crossing_compiled_cost_v1"
SNAKE_FIRST_CROSSING_COST_KEYS = (
    "paper_i_first_crossing_compiled_cost",
    "snake_first_crossing_compiled_cost",
)
_ADAPTIVE_TABLE_I_METHOD_IDS = set(_ADAPT_VARIANT_IDS) | {"static_qiskit_adapt_vqe", SNAKE_TABLE_I_ALGORITHM_ID}
QISKIT_FIRST_HIT_COST_SOURCE_KINDS = {
    "qiskit_compiled_first_hit_ansatz_circuit",
    "qiskit_compiled_terminal_only_fixed_ansatz",
    "snake_qiskit_compiled_first_hit_ansatz_circuit",
}
QISKIT_FINAL_ANSATZ_COST_SOURCE_KINDS = {
    "qiskit_compiled_final_ansatz_circuit",
}
QISKIT_REPORTABLE_COST_SOURCE_KINDS = (
    QISKIT_FIRST_HIT_COST_SOURCE_KINDS | QISKIT_FINAL_ANSATZ_COST_SOURCE_KINDS
)
FORBIDDEN_TABLE_I_RESOURCE_SOURCE_TOKENS = (
    "proxy",
    "deterministic",
    "terminal",
    "final",
    "tie",
    "objective_score",
    "live_overlay",
    "live_snake_overlay",
    "supplemental",
    "synthetic",
    "fabricated",
    "current_best",
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


class NotReconstructable(RuntimeError):
    """Expected absence of post-hoc ansatz/circuit reconstruction artifacts."""

    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _load_records(path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines(), delimiter="\t"))
    required = {"record_id", "family", "case_id", "algorithm_id"}
    missing = required - set(rows[0].keys() if rows else ())
    if missing:
        raise ValueError(f"records file {path} missing columns: {sorted(missing)}")
    return rows


def _read_payload(root: Path, record_id: str) -> tuple[Path, Mapping[str, Any] | None]:
    result_dir = root / record_id / "result"
    for name in (
        "generic_static_single.json",
        "result.json",
        "manifest.json",
        "skip.json",
        "hh_static_benchmark_result.json",
        "hh_static_benchmark_rows.json",
    ):
        path = result_dir / name
        if path.exists():
            return path, json.loads(path.read_text(encoding="utf-8"))
    return result_dir / "generic_static_single.json", None


def _result(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    result = payload.get("result")
    if isinstance(result, Mapping):
        return result
    rows = payload.get("rows")
    if isinstance(rows, list) and rows and isinstance(rows[0], Mapping):
        return rows[0]
    return payload


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _first_num(row: Mapping[str, Any], keys: Sequence[str]) -> tuple[float | None, str | None]:
    for key in keys:
        value = _num(row.get(key))
        if value is not None:
            return value, str(key)
    return None, None


def _measurement_work_other_quantum(row: Mapping[str, Any]) -> tuple[float, str]:
    value, source = _first_num(row, MEASUREMENT_WORK_OTHER_QUANTUM_ALIASES)
    if value is None:
        return 0.0, "default_zero"
    return float(value), str(source)


def _measurement_work_split_aliases(component: str, split: str) -> tuple[str, ...]:
    canonical = {
        "N_H_outer_eval": "H_outer",
        "N_grad": "grad",
        "N_metric": "metric",
        "N_H_refit_eval": "H_refit",
    }[component]
    legacy = {
        "N_H_outer_eval": "H_eval",
        "N_grad": "grad",
        "N_metric": "metric",
        "N_H_refit_eval": "refit_eval",
    }[component]
    return (
        f"S_norm_N_{canonical}_{split}",
        f"measurement_work_N_{canonical}_{split}",
        f"N_{canonical}_{split}",
        f"S_norm_N_{legacy}_{split}",
        f"measurement_work_N_{legacy}_{split}",
        f"N_{legacy}_{split}",
    )


def _measurement_work_splits(row: Mapping[str, Any], components: Mapping[str, float]) -> dict[str, dict[str, Any]]:
    """Return request/fresh/cache telemetry for the four disjoint S_norm bins.

    Existing comparator rows do not yet emit cache telemetry. For those rows the
    component count is treated as fresh measurement-bearing work, and the status
    makes that convention explicit rather than silently changing the table
    denominator.
    """

    splits: dict[str, dict[str, Any]] = {}
    for component, raw_value in components.items():
        value = float(raw_value)
        entry: dict[str, Any] = {
            "request": value,
            "fresh": value,
            "cache": 0.0,
            "status": "assumed_fresh_no_cache_telemetry",
            "sources": {
                "request": "component_value",
                "fresh": "component_value",
                "cache": "default_zero",
            },
        }
        any_explicit = False
        for split in MEASUREMENT_WORK_SPLIT_SUFFIXES:
            split_value, split_source = _first_num(row, _measurement_work_split_aliases(component, split))
            if split_value is not None:
                entry[split] = float(split_value)
                entry["sources"][split] = str(split_source)
                any_explicit = True
        if any_explicit:
            request = float(entry.get("request") or 0.0)
            fresh = float(entry.get("fresh") or 0.0)
            cache = float(entry.get("cache") or 0.0)
            if fresh < 0.0 or cache < 0.0 or request < 0.0:
                entry["status"] = "invalid_negative_split"
            elif abs((fresh + cache) - request) > 1e-9:
                entry["status"] = "inconsistent_request_fresh_cache_split"
            else:
                entry["status"] = "explicit"
        splits[component] = entry
    return splits


def _selected_operator_count(row: Mapping[str, Any]) -> tuple[int | None, str | None]:
    value = _num(row.get("selected_operator_count"))
    if value is not None:
        return max(0, int(round(value))), "selected_operator_count"
    selected = row.get("selected_operators")
    if isinstance(selected, list):
        return len(selected), "len(selected_operators)"
    theta = row.get("theta")
    if isinstance(theta, list):
        return len(theta), "len(theta)"
    return None, None


def _explicit_measurement_work_components(row: Mapping[str, Any]) -> tuple[dict[str, float] | None, dict[str, str]]:
    components: dict[str, float] = {}
    sources: dict[str, str] = {}
    missing: list[str] = []
    for name, keys in MEASUREMENT_WORK_COMPONENT_ALIASES.items():
        value, source = _first_num(row, keys)
        if value is None:
            missing.append(name)
        elif float(value) < 0.0:
            return None, {"invalid": str(source)}
        else:
            components[name] = float(value)
            sources[name] = str(source)
    if not missing:
        return components, sources
    return None, {"missing": ",".join(missing)}


def normalized_measurement_work_from_explicit_row(
    *,
    row: Mapping[str, Any],
    raw_proxy: Mapping[str, Any] | None = None,
    missing_reason: str = "missing_component_breakdown",
) -> tuple[dict[str, Any], dict[str, float], dict[str, str]]:
    """Build ``S_norm`` only from an explicit four-component decomposition.

    This helper is for reporting bridges, especially SNAKE/Table-I support
    artifacts. It deliberately does not infer missing components from raw shot
    totals, controller totals, optimizer counts, or any algorithm-specific
    fallback.
    """

    weights = dict(DEFAULT_S_NORM_WEIGHTS)
    raw = {str(key): _num(value) for key, value in dict(raw_proxy or {}).items()}
    other_quantum, other_quantum_source = _measurement_work_other_quantum(row)
    if other_quantum < 0.0:
        return (
            {
                "schema": NORMALIZED_MEASUREMENT_WORK_SCHEMA,
                "status": "invalid_component_value",
                "reason": f"negative_{other_quantum_source}",
                "S_norm": None,
                "weights": weights,
                "components": None,
                "component_sources": None,
                "N_other_quantum": other_quantum,
                "N_other_quantum_source": other_quantum_source,
                "legacy_raw_proxy": raw,
                "unit": "normalized_estimator_or_probe_count_not_physical_shots",
            },
            {},
            {"S_norm": "invalid_component_value"},
        )
    if other_quantum > 0.0:
        return (
            {
                "schema": NORMALIZED_MEASUREMENT_WORK_SCHEMA,
                "status": "unassigned_other_quantum_work",
                "reason": "nonzero_N_other_quantum_requires_assignment_to_disjoint_bins",
                "S_norm": None,
                "weights": weights,
                "components": None,
                "component_sources": None,
                "N_other_quantum": other_quantum,
                "N_other_quantum_source": other_quantum_source,
                "legacy_raw_proxy": raw,
                "unit": "normalized_estimator_or_probe_count_not_physical_shots",
            },
            {},
            {"S_norm": "unassigned_other_quantum_work"},
        )
    components: dict[str, float] = {}
    sources: dict[str, str] = {}
    missing: list[str] = []
    for name, keys in MEASUREMENT_WORK_COMPONENT_ALIASES.items():
        value, source = _first_num(row, keys)
        if value is None:
            missing.append(name)
            continue
        if float(value) < 0.0:
            return (
                {
                    "schema": NORMALIZED_MEASUREMENT_WORK_SCHEMA,
                    "status": "invalid_component_value",
                    "reason": f"negative_{source}",
                    "S_norm": None,
                    "weights": weights,
                    "components": None,
                    "component_sources": None,
                    "legacy_raw_proxy": raw,
                    "unit": "normalized_estimator_or_probe_count_not_physical_shots",
                },
                {},
                {"S_norm": "invalid_component_value"},
            )
        components[name] = float(value)
        sources[name] = str(source)
    if missing:
        return (
            {
                "schema": NORMALIZED_MEASUREMENT_WORK_SCHEMA,
                "status": "missing_component_breakdown",
                "reason": missing_reason,
                "missing_components": missing,
                "S_norm": None,
                "weights": weights,
                "components": None,
                "component_sources": None,
                "N_other_quantum": other_quantum,
                "N_other_quantum_source": other_quantum_source,
                "legacy_raw_proxy": raw,
                "unit": "normalized_estimator_or_probe_count_not_physical_shots",
            },
            {},
            {"S_norm": "missing_component_breakdown"},
        )
    s_norm = (
        weights["s_H_outer"] * components["N_H_outer_eval"]
        + weights["s_g"] * components["N_grad"]
        + weights["s_F"] * components["N_metric"]
        + weights["s_H_refit"] * components["N_H_refit_eval"]
    )
    row_updates = {
        "S_norm": float(s_norm),
        "S_norm_N_H_outer_eval": float(components["N_H_outer_eval"]),
        "S_norm_N_grad": float(components["N_grad"]),
        "S_norm_N_metric": float(components["N_metric"]),
        "S_norm_N_H_refit_eval": float(components["N_H_refit_eval"]),
        "S_norm_N_H_eval": float(components["N_H_outer_eval"]),
        "S_norm_N_refit_eval": float(components["N_H_refit_eval"]),
        "S_norm_N_other_quantum": 0.0,
    }
    return (
        {
            "schema": NORMALIZED_MEASUREMENT_WORK_SCHEMA,
            "status": "ok",
            "S_norm": float(s_norm),
            "weights": weights,
            "components": {key: float(value) for key, value in components.items()},
            "component_sources": sources,
            "N_other_quantum": 0.0,
            "N_other_quantum_source": other_quantum_source,
            "component_splits": _measurement_work_splits(row, components),
            "event_count_convention": "fresh_measurement_bearing_calls_when_split_telemetry_absent",
            "legacy_raw_proxy": raw,
            "unit": "normalized_estimator_or_probe_count_not_physical_shots",
        },
        row_updates,
        {"S_norm": "ok"},
    )


def grouped_measurement_proxy_from_explicit_row(
    *,
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, float], dict[str, str]]:
    """Build a grouped-Pauli physical proxy only from explicit components."""

    components: dict[str, float] = {}
    sources: dict[str, str] = {}
    missing: list[str] = []
    for name, keys in GROUPED_MEASUREMENT_COMPONENT_ALIASES.items():
        value, source = _first_num(row, keys)
        if value is None:
            missing.append(name)
            continue
        if float(value) < 0.0:
            return (
                {
                    "schema": GROUPED_MEASUREMENT_PROXY_SCHEMA,
                    "status": "invalid_grouped_measurement_value",
                    "reason": f"negative_{source}",
                    "S_grp_total": None,
                    "components": None,
                    "component_sources": None,
                    "unit": "grouped_pauli_shot_proxy_under_common_measurement_model",
                },
                {},
                {"S_grp": "invalid_grouped_measurement_value"},
            )
        components[name] = float(value)
        sources[name] = str(source)
    if missing:
        return (
            {
                "schema": GROUPED_MEASUREMENT_PROXY_SCHEMA,
                "status": "missing_grouped_measurement_breakdown",
                "missing_components": missing,
                "S_grp_total": None,
                "components": None,
                "component_sources": None,
                "unit": "grouped_pauli_shot_proxy_under_common_measurement_model",
            },
            {},
            {"S_grp": "missing_grouped_measurement_breakdown"},
        )
    total = sum(float(value) for value in components.values())
    row_updates = {
        "S_grp_total": float(total),
        "S_grp_H_outer": float(components["S_grp_H_outer"]),
        "S_grp_grad": float(components["S_grp_grad"]),
        "S_grp_metric": float(components["S_grp_metric"]),
        "S_grp_H_refit": float(components["S_grp_H_refit"]),
    }
    return (
        {
            "schema": GROUPED_MEASUREMENT_PROXY_SCHEMA,
            "status": "ok",
            "S_grp_total": float(total),
            "components": {key: float(value) for key, value in components.items()},
            "component_sources": sources,
            "unit": "grouped_pauli_shot_proxy_under_common_measurement_model",
        },
        row_updates,
        {"S_grp": "ok"},
    )


def _strict_first_num(row: Mapping[str, Any], keys: Sequence[str]) -> tuple[float | None, str | None, str | None]:
    """Return first finite numeric value, preserving explicit invalid fields.

    ``_first_num`` deliberately conflates missing and invalid values.  The strict
    S_alg/S_phys gates need to distinguish them so a nonfinite/negative emitted
    event component cannot silently become "missing" and fall through to a raw
    proxy.
    """

    for key in keys:
        if key not in row:
            continue
        raw = row.get(key)
        if raw is None or raw == "":
            continue
        value = _num(raw)
        if value is None:
            return None, str(key), "invalid_nonfinite"
        return float(value), str(key), None
    return None, None, None


def _legacy_algorithmic_proxy_present(row: Mapping[str, Any], raw_proxy: Mapping[str, Any] | None = None) -> bool:
    legacy_keys: set[str] = {
        "S_norm",
        "energy_eval_count_proxy",
        "gradient_operator_probe_count_proxy",
        "metric_operator_probe_count_proxy",
        "gradient_scan_count_proxy",
        "selected_operator_count",
        "nfev",
        "nfev_total",
        "shots_total",
        "shot_cost_proxy",
        "measurement_shots_proxy",
        "shot_proxy",
        "measurement_work_proxy",
    }
    for aliases in MEASUREMENT_WORK_COMPONENT_ALIASES.values():
        legacy_keys.update(map(str, aliases))
    legacy_keys.update(MEASUREMENT_WORK_OTHER_QUANTUM_ALIASES)
    if any(key in row for key in legacy_keys):
        return True
    if raw_proxy:
        return any(_num(value) is not None for value in raw_proxy.values())
    return False


def _legacy_grouped_proxy_present(row: Mapping[str, Any]) -> bool:
    legacy_keys = {"S_grp_total", "grouped_measurement_work", "grouped_measurement_proxy"}
    for aliases in GROUPED_MEASUREMENT_COMPONENT_ALIASES.values():
        legacy_keys.update(map(str, aliases))
    return any(key in row for key in legacy_keys)


def _parse_s_alg_other_quantum(row: Mapping[str, Any]) -> tuple[float | None, str, str | None]:
    value, source, invalid = _strict_first_num(row, S_ALG_OTHER_QUANTUM_ALIASES)
    if invalid is not None:
        return None, str(source), invalid
    if value is None:
        return 0.0, "default_zero", None
    return float(value), str(source), None


def _explicit_s_alg_components(row: Mapping[str, Any]) -> tuple[dict[str, float] | None, dict[str, str], str | None, list[str]]:
    components: dict[str, float] = {}
    sources: dict[str, str] = {}
    missing: list[str] = []
    any_present = False
    for name, aliases in S_ALG_COMPONENT_ALIASES.items():
        value, source, invalid = _strict_first_num(row, aliases)
        if source is not None:
            any_present = True
        if invalid is not None:
            return None, {"invalid": str(source)}, f"invalid_{source}", []
        if value is None:
            missing.append(name)
            continue
        if float(value) < 0.0:
            return None, {"invalid": str(source)}, f"negative_{source}", []
        components[name] = float(value)
        sources[name] = str(source)
    if missing:
        return None, sources, "partial" if any_present else None, missing
    return components, sources, None, []


def _s_alg_splits(row: Mapping[str, Any], components: Mapping[str, float]) -> dict[str, dict[str, Any]]:
    splits: dict[str, dict[str, Any]] = {}
    aliases = {
        "N_H_outer_eval": "H_outer",
        "N_grad_probe": "grad",
        "N_metric_probe": "metric",
        "N_H_refit_eval": "H_refit",
    }
    for component, raw_value in components.items():
        base = aliases[component]
        value = float(raw_value)
        entry: dict[str, Any] = {
            "request": value,
            "fresh": value,
            "cache": 0.0,
            "status": "assumed_fresh_no_cache_telemetry",
            "sources": {"request": "component_value", "fresh": "component_value", "cache": "default_zero"},
        }
        explicit = False
        for split in MEASUREMENT_WORK_SPLIT_SUFFIXES:
            keys = (
                f"S_alg_N_{base}_{split}",
                f"algorithmic_measurement_work_N_{base}_{split}",
            )
            split_value, split_source, invalid = _strict_first_num(row, keys)
            if invalid is not None:
                entry["status"] = f"invalid_{split_source}"
                explicit = True
                continue
            if split_value is not None:
                entry[split] = float(split_value)
                entry["sources"][split] = str(split_source)
                explicit = True
        if explicit and not str(entry.get("status", "")).startswith("invalid_"):
            request = float(entry.get("request") or 0.0)
            fresh = float(entry.get("fresh") or 0.0)
            cache = float(entry.get("cache") or 0.0)
            if request < 0.0 or fresh < 0.0 or cache < 0.0:
                entry["status"] = "invalid_negative_split"
            elif abs((fresh + cache) - request) > 1e-9:
                entry["status"] = "inconsistent_request_fresh_cache_split"
            else:
                entry["status"] = "explicit"
        splits[component] = entry
    return splits


def _ledger_component_from_event(event: Mapping[str, Any]) -> str:
    raw = str(event.get("bin_id") or event.get("event_type") or event.get("method_stage") or "").strip().lower()
    raw = raw.replace("-", "_").replace(" ", "_")
    if raw in {"h_outer", "h_outer_eval", "outer_energy", "outer_objective", "h", "energy_outer"}:
        return "N_H_outer_eval"
    if raw in {"grad", "gradient", "candidate_probe", "position_probe", "operator_probe", "measured_gradient_probe", "probe"}:
        return "N_grad_probe"
    if raw in {"metric", "f", "qgt", "qfi", "fubini_study", "overlap", "natural_gradient"}:
        return "N_metric_probe"
    if raw in {"h_refit", "h_refit_eval", "refit", "inner_refit", "post_selection_refit", "refit_energy"}:
        return "N_H_refit_eval"
    return "N_other_quantum"


def _algorithmic_work_from_event_ledger(row: Mapping[str, Any]) -> tuple[dict[str, float] | None, dict[str, Any] | None, str | None]:
    ledger = None
    for key in ("table_i_measurement_event_ledger", "measurement_event_ledger", "measurement_events"):
        if key in row:
            ledger = row.get(key)
            break
    if ledger is None:
        return None, None, None

    components = {"N_H_outer_eval": 0.0, "N_grad_probe": 0.0, "N_metric_probe": 0.0, "N_H_refit_eval": 0.0}
    cache_components = {key: 0.0 for key in components}
    other_fresh = 0.0
    other_cache = 0.0
    meta: dict[str, Any] = {
        "schema": TABLE_I_EVENT_LEDGER_SCHEMA,
        "source": "table_i_measurement_event_ledger",
        "event_count_convention": "fresh_measurement_bearing_estimator_or_probe_events",
    }

    if isinstance(ledger, Mapping):
        if str(ledger.get("schema") or "") != TABLE_I_EVENT_LEDGER_SCHEMA:
            return None, {"status": "invalid_event_ledger_schema", "schema": ledger.get("schema")}, "invalid_event_ledger"
        if "status" in ledger and str(ledger.get("status") or "") != "ok":
            return None, {"status": "invalid_event_ledger_status", "ledger_status": ledger.get("status")}, "invalid_event_ledger"
        totals = ledger.get("component_totals")
        if isinstance(totals, Mapping):
            source_map: dict[str, str] = {}
            aliases = {
                "N_H_outer_eval": ("N_H_outer_eval", "S_alg_N_H_outer_eval"),
                "N_grad_probe": ("N_grad_probe", "N_grad", "S_alg_N_grad_probe"),
                "N_metric_probe": ("N_metric_probe", "N_metric", "S_alg_N_metric_probe"),
                "N_H_refit_eval": ("N_H_refit_eval", "N_refit_eval", "S_alg_N_H_refit_eval"),
            }
            for component, keys in aliases.items():
                value, source, invalid = _strict_first_num(totals, keys)
                if invalid is not None or value is None or float(value) < 0.0:
                    return None, {"status": "invalid_event_ledger_component", "component": component, "source": source}, "invalid_event_ledger"
                components[component] = float(value)
                source_map[component] = str(source)
            other_value, other_source, invalid = _strict_first_num(totals, ("N_other_quantum", "N_other_algorithmic_quantum", "S_alg_N_other_quantum"))
            if invalid is not None or (other_value is not None and float(other_value) < 0.0):
                return None, {"status": "invalid_event_ledger_other_quantum", "source": other_source}, "invalid_event_ledger"
            if other_value and float(other_value) > 0.0:
                return None, {"status": "unassigned_other_algorithmic_work", "N_other_quantum": float(other_value)}, "unassigned_other_algorithmic_work"
            meta.update(status="ok", source="component_totals", component_sources=source_map, N_other_quantum=0.0)
            return components, meta, None
        events = ledger.get("events")
    elif isinstance(ledger, list):
        events = ledger
        meta["source"] = "measurement_events"
    else:
        return None, {"status": "invalid_event_ledger_type"}, "invalid_event_ledger"

    if not isinstance(events, list):
        return None, {"status": "missing_event_list"}, "invalid_event_ledger"
    for idx, event in enumerate(events):
        if not isinstance(event, Mapping):
            return None, {"status": "invalid_event", "event_index": int(idx)}, "invalid_event_ledger"
        component = _ledger_component_from_event(event)
        count = _num(event.get("event_count", event.get("count", 1.0)))
        if count is None or float(count) < 0.0:
            return None, {"status": "invalid_event_count", "event_index": int(idx)}, "invalid_event_ledger"
        status = str(event.get("fresh_or_cache") or event.get("request_or_fresh_or_cache") or event.get("measurement_status") or "fresh").lower()
        cache_hit = bool(event.get("cache_hit") is True) or status in {"cache", "cached", "cache_hit"}
        if component == "N_other_quantum":
            if cache_hit:
                other_cache += float(count)
            else:
                other_fresh += float(count)
            continue
        if cache_hit:
            cache_components[component] += float(count)
        else:
            components[component] += float(count)
    if other_fresh > 0.0:
        return None, {"status": "unassigned_other_algorithmic_work", "N_other_quantum": float(other_fresh)}, "unassigned_other_algorithmic_work"
    meta.update(
        status="ok",
        source="events",
        fresh_components=components,
        cache_components=cache_components,
        N_other_quantum=0.0,
        N_other_quantum_cache=float(other_cache),
        event_count=int(len(events)),
    )
    return components, meta, None


def algorithmic_measurement_work_from_row(
    *,
    row: Mapping[str, Any],
    raw_proxy: Mapping[str, Any] | None = None,
    missing_reason: str = "missing_event_ledger_component_breakdown",
) -> tuple[dict[str, Any], dict[str, float], dict[str, str]]:
    """Build paper-facing ``S_alg`` only from event-granular telemetry.

    Legacy S_norm/proxy fields are retained as provenance elsewhere; they are
    intentionally not accepted here because they may compress pool scans,
    metric probes, optimizer calls, cache reuse, and raw shot surrogates into
    incompatible currencies.
    """

    raw = {str(key): _num(value) for key, value in dict(raw_proxy or {}).items()}
    weights = dict(DEFAULT_S_ALG_WEIGHTS)
    components, sources, invalid, missing = _explicit_s_alg_components(row)
    source_kind = "explicit_components"
    ledger_meta: dict[str, Any] | None = None
    if invalid is not None and invalid != "partial":
        status = "invalid_event_component_value"
        return (
            {
                "schema": ALGORITHMIC_MEASUREMENT_WORK_SCHEMA,
                "status": status,
                "reason": invalid,
                "S_alg": None,
                "weights": weights,
                "components": None,
                "component_sources": sources or None,
                "legacy_raw_proxy": raw,
                "unit": "algorithmic_estimator_or_probe_event_count_not_physical_shots",
            },
            {},
            {"S_alg": status},
        )
    if components is None:
        ledger_components, ledger_meta, ledger_status = _algorithmic_work_from_event_ledger(row)
        if ledger_status is not None:
            status = str(ledger_status)
            return (
                {
                    "schema": ALGORITHMIC_MEASUREMENT_WORK_SCHEMA,
                    "status": status,
                    "reason": status,
                    "S_alg": None,
                    "weights": weights,
                    "components": None,
                    "event_ledger": ledger_meta,
                    "legacy_raw_proxy": raw,
                    "unit": "algorithmic_estimator_or_probe_event_count_not_physical_shots",
                },
                {},
                {"S_alg": status},
            )
        if ledger_components is not None:
            components = ledger_components
            sources = {key: "table_i_measurement_event_ledger" for key in components}
            source_kind = "event_ledger"
        else:
            status = "legacy_proxy_not_event_ledger" if _legacy_algorithmic_proxy_present(row, raw) else missing_reason
            return (
                {
                    "schema": ALGORITHMIC_MEASUREMENT_WORK_SCHEMA,
                    "status": status,
                    "reason": status if status != missing_reason else missing_reason,
                    "missing_components": missing or list(S_ALG_COMPONENT_ALIASES.keys()),
                    "S_alg": None,
                    "weights": weights,
                    "components": None,
                    "component_sources": None,
                    "legacy_raw_proxy": raw,
                    "unit": "algorithmic_estimator_or_probe_event_count_not_physical_shots",
                },
                {},
                {"S_alg": status},
            )
    other_quantum, other_source, other_invalid = _parse_s_alg_other_quantum(row)
    if other_invalid is not None or other_quantum is None or float(other_quantum) < 0.0:
        status = "invalid_event_component_value"
        return (
            {
                "schema": ALGORITHMIC_MEASUREMENT_WORK_SCHEMA,
                "status": status,
                "reason": f"invalid_{other_source}",
                "S_alg": None,
                "weights": weights,
                "components": None,
                "component_sources": None,
                "legacy_raw_proxy": raw,
                "unit": "algorithmic_estimator_or_probe_event_count_not_physical_shots",
            },
            {},
            {"S_alg": status},
        )
    if float(other_quantum) > 0.0:
        status = "unassigned_other_algorithmic_work"
        return (
            {
                "schema": ALGORITHMIC_MEASUREMENT_WORK_SCHEMA,
                "status": status,
                "reason": "nonzero_S_alg_N_other_quantum_requires_assignment_to_disjoint_bins",
                "S_alg": None,
                "weights": weights,
                "components": None,
                "component_sources": None,
                "N_other_quantum": float(other_quantum),
                "N_other_quantum_source": other_source,
                "legacy_raw_proxy": raw,
                "unit": "algorithmic_estimator_or_probe_event_count_not_physical_shots",
            },
            {},
            {"S_alg": status},
        )
    for key, value in components.items():
        if not math.isfinite(float(value)) or float(value) < 0.0:
            status = "invalid_event_component_value"
            return (
                {
                    "schema": ALGORITHMIC_MEASUREMENT_WORK_SCHEMA,
                    "status": status,
                    "reason": f"invalid_{key}",
                    "S_alg": None,
                    "weights": weights,
                    "components": None,
                    "component_sources": sources or None,
                    "legacy_raw_proxy": raw,
                    "unit": "algorithmic_estimator_or_probe_event_count_not_physical_shots",
                },
                {},
                {"S_alg": status},
            )
    s_alg = (
        weights["s_H_outer"] * components["N_H_outer_eval"]
        + weights["s_grad"] * components["N_grad_probe"]
        + weights["s_metric"] * components["N_metric_probe"]
        + weights["s_H_refit"] * components["N_H_refit_eval"]
    )
    row_updates = {
        "S_alg": float(s_alg),
        "S_alg_N_H_outer_eval": float(components["N_H_outer_eval"]),
        "S_alg_N_grad_probe": float(components["N_grad_probe"]),
        "S_alg_N_metric_probe": float(components["N_metric_probe"]),
        "S_alg_N_H_refit_eval": float(components["N_H_refit_eval"]),
        "S_alg_N_other_quantum": 0.0,
    }
    return (
        {
            "schema": ALGORITHMIC_MEASUREMENT_WORK_SCHEMA,
            "status": "ok",
            "S_alg": float(s_alg),
            "weights": weights,
            "components": {key: float(value) for key, value in components.items()},
            "component_sources": dict(sources),
            "component_splits": _s_alg_splits(row, components) if source_kind == "explicit_components" else None,
            "event_ledger": ledger_meta,
            "N_other_quantum": 0.0,
            "N_other_quantum_source": other_source,
            "event_count_convention": "fresh_measurement_bearing_estimator_or_probe_events",
            "source_kind": source_kind,
            "legacy_raw_proxy": raw,
            "unit": "algorithmic_estimator_or_probe_event_count_not_physical_shots",
        },
        row_updates,
        {"S_alg": "ok"},
    )


def _sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.exists() or not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strip_snake_sidecars_for_hash(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_snake_sidecars_for_hash(item)
            for key, item in value.items()
            if key not in set(SNAKE_FIRST_CROSSING_COST_KEYS) | {"source_result_sha256"}
        }
    if isinstance(value, list):
        return [_strip_snake_sidecars_for_hash(item) for item in value]
    return value


def _sha256_json_without_snake_sidecars(path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.exists() or not candidate.is_file():
        return None
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return None
    stripped = _strip_snake_sidecars_for_hash(payload)
    body = json.dumps(stripped, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _mapping_at(row: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    value = row.get(key)
    return value if isinstance(value, Mapping) else None


def _snake_first_crossing_from_row(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    direct = _mapping_at(row, "paper_i_first_crossing")
    if direct is not None:
        return direct
    for parent_key in ("result", "adapt_vqe", "best_user_attrs"):
        parent = _mapping_at(row, parent_key)
        if parent is None:
            continue
        nested = _mapping_at(parent, "paper_i_first_crossing")
        if nested is not None:
            return nested
    return None


def _snake_compiled_cost_sidecar(row: Mapping[str, Any]) -> tuple[str | None, Mapping[str, Any] | None]:
    for key in SNAKE_FIRST_CROSSING_COST_KEYS:
        sidecar = _mapping_at(row, key)
        if sidecar is not None:
            return key, sidecar
    return None, None


def _first_numeric_from_mapping(mapping: Mapping[str, Any] | None, *keys: str) -> float | None:
    if mapping is None:
        return None
    for key in keys:
        value = _num(mapping.get(key))
        if value is not None:
            return float(value)
    return None


def _integer_position(value: Any) -> int | None:
    parsed = _num(value)
    if parsed is None or not math.isfinite(float(parsed)):
        return None
    if not float(parsed).is_integer():
        return None
    return int(parsed)


def _snake_crossing_primary_error(
    crossing: Mapping[str, Any] | None,
    row: Mapping[str, Any],
    *,
    allow_row_fallback: bool = True,
) -> float | None:
    value = _first_numeric_from_mapping(
        crossing,
        "primary_error_at_crossing",
        "abs_delta_e_at_crossing",
        "abs_error_at_crossing",
        "primary_error",
    )
    if value is not None:
        return value
    if not allow_row_fallback:
        return None
    return _first_numeric_from_mapping(row, "abs_delta_e", "primary_error", "delta_E_abs")


def _method_cost_semantics(algorithm_id: str) -> str:
    alg = str(algorithm_id)
    if alg == SNAKE_TABLE_I_ALGORITHM_ID:
        return "snake_first_hit_sidecar_required"
    if alg in _FIXED_TERMINAL_TABLE_I_METHOD_IDS:
        return "terminal_only_fixed_ansatz"
    if alg in _ADAPTIVE_TABLE_I_METHOD_IDS:
        return "adaptive_qiskit_compiled_first_hit_or_final_ansatz"
    return "unknown_table_i_cost_semantics"


def _terminal_final_scrubbed_source_text(value: Any) -> str:
    text = str(value).lower()
    # ``terminal``/``final`` are forbidden as generic resource provenance, but
    # they are expected descriptors for the explicitly validated final-ansatz
    # non-hit cost path.  Preserve other forbidden words such as proxy/synthetic.
    return text.replace("terminal", "").replace("final", "")


def _source_text_for_resource_validation(*items: Any) -> str:
    parts: list[str] = []
    for item in items:
        if isinstance(item, Mapping):
            mapping_source_kind = str(
                item.get("first_hit_cost_source_kind")
                or item.get("compiled_resource_source_kind")
                or item.get("source_kind")
                or ""
            )
            mapping_is_reportable_qiskit = mapping_source_kind in QISKIT_REPORTABLE_COST_SOURCE_KINDS
            for key in (
                "first_hit_cost_source_kind",
                "compiled_resource_source_kind",
                "source_kind",
                "source",
                "cost_source",
                "first_hit_semantics",
                "compiled_circuit_stats_status",
                "compiled_depth_2q_semantics",
                "depth_2q_semantics",
                "compiled_resource_provenance",
                "compiled_resource_exactness",
            ):
                value = item.get(key)
                if key in {"first_hit_cost_source_kind", "compiled_resource_source_kind", "source_kind"} and value in QISKIT_REPORTABLE_COST_SOURCE_KINDS:
                    continue
                if value is not None:
                    if mapping_is_reportable_qiskit and key in {"source", "first_hit_semantics"}:
                        parts.append(_terminal_final_scrubbed_source_text(value))
                    else:
                        parts.append(str(value))
        elif item is not None and item not in QISKIT_REPORTABLE_COST_SOURCE_KINDS:
            parts.append(str(item))
    return " ".join(parts).lower()


def _forbidden_resource_source_reason(*items: Any) -> str | None:
    text = _source_text_for_resource_validation(*items)
    for token in FORBIDDEN_TABLE_I_RESOURCE_SOURCE_TOKENS:
        if token in text:
            return f"forbidden_resource_source_{token}"
    return None


def _compiled_depth_ordering_check(
    *,
    depth_2q: float | None,
    circuit_depth: float | None,
    source_text: str,
) -> tuple[bool, str | None]:
    if depth_2q is None or circuit_depth is None:
        return True, None
    if "count" in source_text and "depth" not in source_text:
        return False, "compiled_depth_2q_semantic_mismatch_count_not_depth"
    if float(circuit_depth) < float(depth_2q):
        return False, "compiled_depth_total_less_than_two_qubit_depth"
    return True, None


def _compiled_resource_validation(
    *,
    algorithm_id: str,
    row: Mapping[str, Any],
    threshold_status: str,
) -> dict[str, Any]:
    """Validate non-SNAKE Table-I resource columns before promotion.

    ``N_2q``, ``D_2q`` and ``D_circ`` are reportable only when the row carries
    explicit Qiskit-compiled ansatz-circuit provenance for the displayed ansatz:
    the native first-hit ansatz for hits, and the final terminal ansatz for
    non-hits.  Deterministic Pauli-rotation estimates, selector costs, and other
    proxies are diagnostics only.
    """

    semantics = _method_cost_semantics(str(algorithm_id))
    count_2q = _first_numeric_from_mapping(row, "compiled_count_2q_total", "count_2q", "N_2q")
    circuit_depth = _first_numeric_from_mapping(row, "compiled_depth_total", "circuit_depth", "D_circ")
    depth_2q = _first_numeric_from_mapping(row, "compiled_depth_2q_total", "depth_2q", "D_2q")
    source_kind = str(
        row.get("first_hit_cost_source_kind")
        or row.get("compiled_resource_source_kind")
        or row.get("source_kind")
        or ""
    )
    source_text = _source_text_for_resource_validation(row, source_kind, threshold_status, semantics)
    forbidden = _forbidden_resource_source_reason(row, source_kind)
    status = str(row.get("compiled_circuit_stats_status") or row.get("qiskit_compile_status") or "").lower()
    qiskit_validated = (
        row.get("qiskit_first_hit_cost_validated") is True
        or row.get("compiled_resource_qiskit_validated") is True
        or source_kind in QISKIT_REPORTABLE_COST_SOURCE_KINDS
    )
    qiskit_text_ok = "qiskit" in source_text
    if status and status not in {"ok", "compiled_ok", "qiskit_ok"}:
        return {
            "resource_display_allowed": False,
            "compiled_resource_validation_status": "invalid",
            "compiled_resource_validation_reason": f"compiled_circuit_stats_status={status}",
            "first_hit_cost_source_kind": source_kind or None,
            "method_cost_semantics": semantics,
            "source_resource_fields_present": {
                "count_2q": count_2q is not None,
                "depth_2q": depth_2q is not None,
                "circuit_depth": circuit_depth is not None,
            },
        }
    if count_2q is None or depth_2q is None or circuit_depth is None:
        return {
            "resource_display_allowed": False,
            "compiled_resource_validation_status": "missing",
            "compiled_resource_validation_reason": "compiled_count_2q_or_depth_2q_or_depth_total_missing",
            "first_hit_cost_source_kind": source_kind or None,
            "method_cost_semantics": semantics,
            "source_resource_fields_present": {
                "count_2q": count_2q is not None,
                "depth_2q": depth_2q is not None,
                "circuit_depth": circuit_depth is not None,
            },
        }
    for name, value in (("count_2q", count_2q), ("depth_2q", depth_2q), ("circuit_depth", circuit_depth)):
        if value is not None and float(value) < 0.0:
            return {
                "resource_display_allowed": False,
                "compiled_resource_validation_status": "invalid",
                "compiled_resource_validation_reason": f"invalid_negative_compiled_resource_value:{name}",
                "first_hit_cost_source_kind": source_kind or None,
                "method_cost_semantics": semantics,
                "source_resource_fields_present": {
                    "count_2q": True,
                    "depth_2q": depth_2q is not None,
                    "circuit_depth": True,
                },
            }
    if forbidden is not None:
        return {
            "resource_display_allowed": False,
            "compiled_resource_validation_status": "invalid",
            "compiled_resource_validation_reason": forbidden,
            "first_hit_cost_source_kind": source_kind or None,
            "method_cost_semantics": semantics,
            "source_resource_fields_present": {
                "count_2q": True,
                "depth_2q": depth_2q is not None,
                "circuit_depth": True,
            },
        }
    if not (qiskit_validated or qiskit_text_ok):
        return {
            "resource_display_allowed": False,
            "compiled_resource_validation_status": "invalid",
            "compiled_resource_validation_reason": "qiskit_compiled_ansatz_provenance_missing",
            "first_hit_cost_source_kind": source_kind or None,
            "method_cost_semantics": semantics,
            "source_resource_fields_present": {
                "count_2q": True,
                "depth_2q": depth_2q is not None,
                "circuit_depth": True,
            },
        }
    if semantics == "adaptive_qiskit_compiled_first_hit_or_final_ansatz":
        if source_kind == "qiskit_compiled_first_hit_ansatz_circuit":
            first_hit_semantics = str(row.get("first_hit_semantics") or "")
            source = str(row.get("source") or row.get("first_hit_source") or "")
            if "native_first_crossing" not in first_hit_semantics and source not in {
                "native_adaptive_iteration",
                "native_first_hit",
                "native_adapt_iteration",
            }:
                return {
                    "resource_display_allowed": False,
                    "compiled_resource_validation_status": "invalid",
                    "compiled_resource_validation_reason": "adaptive_first_hit_semantics_missing",
                    "first_hit_cost_source_kind": source_kind or None,
                    "method_cost_semantics": semantics,
                    "source_resource_fields_present": {
                        "count_2q": True,
                        "depth_2q": depth_2q is not None,
                        "circuit_depth": True,
                    },
                }
        elif source_kind == "qiskit_compiled_final_ansatz_circuit":
            if "not_reached" not in str(threshold_status or "").lower():
                return {
                    "resource_display_allowed": False,
                    "compiled_resource_validation_status": "invalid",
                    "compiled_resource_validation_reason": "adaptive_final_ansatz_cost_requires_not_reached_status",
                    "first_hit_cost_source_kind": source_kind or None,
                    "method_cost_semantics": semantics,
                    "source_resource_fields_present": {
                        "count_2q": True,
                        "depth_2q": depth_2q is not None,
                        "circuit_depth": True,
                    },
                }
        else:
            return {
                "resource_display_allowed": False,
                "compiled_resource_validation_status": "invalid",
                "compiled_resource_validation_reason": "adaptive_qiskit_source_kind_mismatch",
                "first_hit_cost_source_kind": source_kind or None,
                "method_cost_semantics": semantics,
                "source_resource_fields_present": {
                    "count_2q": True,
                    "depth_2q": depth_2q is not None,
                    "circuit_depth": True,
                },
            }
    if semantics == "terminal_only_fixed_ansatz" and source_kind != "qiskit_compiled_terminal_only_fixed_ansatz":
        return {
            "resource_display_allowed": False,
            "compiled_resource_validation_status": "invalid",
            "compiled_resource_validation_reason": "fixed_terminal_qiskit_source_kind_mismatch",
            "first_hit_cost_source_kind": source_kind or None,
            "method_cost_semantics": semantics,
            "source_resource_fields_present": {
                "count_2q": True,
                "depth_2q": depth_2q is not None,
                "circuit_depth": True,
            },
        }
    depth_ok, depth_reason = _compiled_depth_ordering_check(
        depth_2q=depth_2q,
        circuit_depth=circuit_depth,
        source_text=source_text,
    )
    if not depth_ok:
        return {
            "resource_display_allowed": False,
            "compiled_resource_validation_status": "invalid",
            "compiled_resource_validation_reason": depth_reason,
            "first_hit_cost_source_kind": source_kind or None,
            "method_cost_semantics": semantics,
            "source_resource_fields_present": {
                "count_2q": True,
                "depth_2q": True,
                "circuit_depth": True,
            },
        }
    if not source_kind:
        source_kind = (
            "qiskit_compiled_terminal_only_fixed_ansatz"
            if semantics == "terminal_only_fixed_ansatz"
            else "qiskit_compiled_first_hit_ansatz_circuit"
        )
    return {
        "resource_display_allowed": True,
        "compiled_resource_validation_status": "ok",
        "compiled_resource_validation_reason": None,
        "first_hit_cost_source_kind": source_kind,
        "method_cost_semantics": semantics,
        "source_resource_fields_present": {
            "count_2q": True,
            "depth_2q": depth_2q is not None,
            "circuit_depth": True,
        },
        "count_2q": float(count_2q),
        "depth_2q": None if depth_2q is None else float(depth_2q),
        "circuit_depth": float(circuit_depth),
        "compiled_count_2q_total": float(count_2q),
        "compiled_depth_2q_total": None if depth_2q is None else float(depth_2q),
        "compiled_depth_total": float(circuit_depth),
    }


def _snake_status_result(
    base: Mapping[str, Any],
    *,
    threshold_status: str,
    reason: str | None = None,
    crossing: Mapping[str, Any] | None = None,
    sidecar_key: str | None = None,
    abs_delta_e: float | None = None,
    sidecar_validation_status: str | None = None,
    sidecar_validation_reason: str | None = None,
    sidecar_hash_verified: bool = False,
    sidecar_source_kind: str | None = None,
    source_result_sha256: str | None = None,
    source_result_path: str | None = None,
    reconstructability_status: str | None = None,
    s_alg_missing_reason: str | None = None,
) -> dict[str, Any]:
    out = {
        **base,
        "threshold_status": threshold_status,
        "reason": reason,
        "paper_i_first_crossing": dict(crossing) if isinstance(crossing, Mapping) else None,
        "snake_first_crossing_cost_sidecar_key": sidecar_key,
        "sidecar_validation_status": sidecar_validation_status or ("missing" if sidecar_key is None else "invalid"),
        "sidecar_validation_reason": sidecar_validation_reason or reason,
        "sidecar_hash_verified": bool(sidecar_hash_verified),
        "sidecar_source_kind": sidecar_source_kind,
        "source_result_sha256": source_result_sha256,
        "source_result_path": source_result_path,
        "reconstructability_status": reconstructability_status,
        "S_alg_missing_reason": s_alg_missing_reason,
        "first_hit_cost_source_kind": sidecar_source_kind,
        "resource_display_allowed": False,
        "compiled_resource_validation_status": sidecar_validation_status or ("missing" if sidecar_key is None else "invalid"),
        "compiled_resource_validation_reason": sidecar_validation_reason or reason,
        "method_cost_semantics": "snake_first_hit_sidecar_required",
        "S_var_status": "missing_threshold_state" if threshold_status.startswith("ok_") else None,
    }
    if abs_delta_e is not None:
        out["abs_delta_e"] = float(abs_delta_e)
    return out


def _table_i_snake_threshold_cost_from_row(
    *,
    row: Mapping[str, Any],
    threshold_value: float,
    base: Mapping[str, Any],
    record: Mapping[str, Any] | None,
    result_path: str | Path | None,
) -> dict[str, Any]:
    crossing = _snake_first_crossing_from_row(row)
    row_status = str(row.get("threshold_status") or row.get("status") or "").lower()
    if crossing is not None and not row_status:
        row_status = str(crossing.get("status") or "").lower()
    primary_error = _snake_crossing_primary_error(crossing, row, allow_row_fallback=True)
    if "running" in row_status:
        status = "running_current_best_reached" if primary_error is not None and primary_error <= threshold_value else "running_current_best_not_reached"
        return _snake_status_result(
            base,
            threshold_status=status,
            reason="running_current_best_cost_ineligible",
            crossing=crossing,
            abs_delta_e=primary_error,
        )
    if crossing is None:
        if primary_error is None:
            return _snake_status_result(base, threshold_status="missing_delta_e", reason="paper_i_first_crossing_missing")
        if primary_error > threshold_value:
            return _snake_status_result(
                base,
                threshold_status="not_reached",
                reason="paper_i_first_crossing_missing_terminal_error_above_threshold",
                abs_delta_e=primary_error,
            )
        return _snake_status_result(
            base,
            threshold_status="snake_audited_first_crossing_cost_missing",
            reason="paper_i_first_crossing_missing",
            abs_delta_e=primary_error,
        )

    tau = _first_numeric_from_mapping(crossing, "tau_phys", "threshold")
    if tau is not None and not math.isclose(float(tau), threshold_value, rel_tol=0.0, abs_tol=1e-12):
        return _snake_status_result(
            base,
            threshold_status="snake_first_crossing_threshold_mismatch",
            reason=f"paper_i_first_crossing_tau_phys={tau}",
            crossing=crossing,
            abs_delta_e=primary_error,
        )
    reached = crossing.get("reached") is True or str(crossing.get("status") or "").lower() == "reached"
    if not reached:
        return _snake_status_result(
            base,
            threshold_status="not_reached",
            reason="paper_i_first_crossing_not_reached",
            crossing=crossing,
            abs_delta_e=primary_error,
        )
    history_position = _integer_position(crossing.get("history_position_tau"))
    if history_position is None:
        return _snake_status_result(
            base,
            threshold_status="snake_audited_first_crossing_cost_invalid",
            reason="paper_i_first_crossing_history_position_tau_missing",
            crossing=crossing,
            abs_delta_e=primary_error,
        )
    crossing_primary_error = _snake_crossing_primary_error(crossing, row, allow_row_fallback=False)
    if crossing_primary_error is None:
        return _snake_status_result(
            base,
            threshold_status="snake_audited_first_crossing_cost_invalid",
            reason="paper_i_first_crossing_primary_error_missing",
            crossing=crossing,
            abs_delta_e=primary_error,
            sidecar_validation_status="invalid",
            sidecar_validation_reason="paper_i_first_crossing_primary_error_missing",
        )
    if crossing_primary_error > threshold_value:
        return _snake_status_result(
            base,
            threshold_status="snake_audited_first_crossing_cost_invalid",
            reason="paper_i_first_crossing_primary_error_above_threshold",
            crossing=crossing,
            abs_delta_e=crossing_primary_error,
            sidecar_validation_status="invalid",
            sidecar_validation_reason="paper_i_first_crossing_primary_error_above_threshold",
        )
    sidecar_key, sidecar = _snake_compiled_cost_sidecar(row)
    if sidecar is None:
        return _snake_status_result(
            base,
            threshold_status="snake_audited_first_crossing_cost_missing",
            reason="compiled_cost_sidecar_missing",
            crossing=crossing,
            abs_delta_e=primary_error,
            sidecar_validation_status="missing",
            sidecar_validation_reason="compiled_cost_sidecar_missing",
        )
    if str(sidecar.get("schema") or "") != SNAKE_FIRST_CROSSING_COST_SCHEMA:
        return _snake_status_result(
            base,
            threshold_status="snake_audited_first_crossing_cost_invalid",
            reason="compiled_cost_sidecar_schema_mismatch",
            crossing=crossing,
            sidecar_key=sidecar_key,
            abs_delta_e=primary_error,
            sidecar_validation_status="invalid",
            sidecar_validation_reason="compiled_cost_sidecar_schema_mismatch",
        )
    sidecar_source_kind = str(
        sidecar.get("first_hit_cost_source_kind")
        or sidecar.get("compiled_resource_source_kind")
        or sidecar.get("source_kind")
        or sidecar.get("source")
        or ""
    )
    sidecar_source_text = _source_text_for_resource_validation(
        sidecar,
        sidecar_source_kind,
        "snake_first_hit_sidecar_required",
    )
    sidecar_source_result_path = str(sidecar.get("source_result_path") or row.get("source_result_path") or "").strip()
    sidecar_reconstructability_status = str(
        sidecar.get("reconstructability_status") or row.get("reconstructability_status") or ""
    ).strip()
    sidecar_s_alg_missing_reason = str(sidecar.get("S_alg_missing_reason") or row.get("S_alg_missing_reason") or "").strip()

    def _sidecar_status_result(*args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("source_result_path", sidecar_source_result_path or None)
        kwargs.setdefault("reconstructability_status", sidecar_reconstructability_status or None)
        kwargs.setdefault("s_alg_missing_reason", sidecar_s_alg_missing_reason or None)
        return _snake_status_result(*args, **kwargs)

    forbidden_source_reason = _forbidden_resource_source_reason(sidecar, sidecar_source_kind)
    if forbidden_source_reason is not None:
        return _sidecar_status_result(
            base,
            threshold_status="snake_audited_first_crossing_cost_invalid",
            reason=forbidden_source_reason,
            crossing=crossing,
            sidecar_key=sidecar_key,
            abs_delta_e=primary_error,
            sidecar_validation_status="invalid",
            sidecar_validation_reason=forbidden_source_reason,
            sidecar_source_kind=sidecar_source_kind or None,
        )
    if sidecar_source_kind != "snake_qiskit_compiled_first_hit_ansatz_circuit":
        return _sidecar_status_result(
            base,
            threshold_status="snake_audited_first_crossing_cost_invalid",
            reason="compiled_cost_source_kind_mismatch",
            crossing=crossing,
            sidecar_key=sidecar_key,
            abs_delta_e=primary_error,
            sidecar_validation_status="invalid",
            sidecar_validation_reason="compiled_cost_source_kind_mismatch",
            sidecar_source_kind=sidecar_source_kind or None,
        )
    compile_status = str(sidecar.get("compiled_circuit_stats_status") or sidecar.get("qiskit_compile_status") or "").lower()
    if compile_status and compile_status not in {"ok", "compiled_ok", "qiskit_ok"}:
        return _snake_status_result(
            base,
            threshold_status="snake_audited_first_crossing_cost_invalid",
            reason=f"compiled_circuit_stats_status={compile_status}",
            crossing=crossing,
            sidecar_key=sidecar_key,
            abs_delta_e=primary_error,
            sidecar_validation_status="invalid",
            sidecar_validation_reason=f"compiled_circuit_stats_status={compile_status}",
            sidecar_source_kind=sidecar_source_kind or None,
        )
    qiskit_validated = (
        sidecar.get("qiskit_first_hit_cost_validated") is True
        or sidecar.get("compiled_resource_qiskit_validated") is True
        or sidecar_source_kind in QISKIT_FIRST_HIT_COST_SOURCE_KINDS
        or "qiskit" in sidecar_source_text
    )
    if not qiskit_validated:
        return _snake_status_result(
            base,
            threshold_status="snake_audited_first_crossing_cost_invalid",
            reason="compiled_cost_qiskit_first_hit_provenance_missing",
            crossing=crossing,
            sidecar_key=sidecar_key,
            abs_delta_e=primary_error,
            sidecar_validation_status="invalid",
            sidecar_validation_reason="compiled_cost_qiskit_first_hit_provenance_missing",
            sidecar_source_kind=sidecar_source_kind or None,
        )
    sidecar_tau = _first_numeric_from_mapping(
        sidecar,
        "tau_phys",
        "threshold",
        "current_target_threshold",
    )
    if sidecar_tau is None or not math.isclose(float(sidecar_tau), threshold_value, rel_tol=0.0, abs_tol=1e-12):
        return _snake_status_result(
            base,
            threshold_status="snake_audited_first_crossing_cost_invalid",
            reason="compiled_cost_tau_phys_mismatch_or_missing",
            crossing=crossing,
            sidecar_key=sidecar_key,
            abs_delta_e=primary_error,
            sidecar_validation_status="invalid",
            sidecar_validation_reason="compiled_cost_tau_phys_mismatch_or_missing",
            sidecar_source_kind=sidecar_source_kind or None,
        )
    sidecar_history_position = _integer_position(sidecar.get("history_position_tau"))
    if sidecar_history_position is None or sidecar_history_position != history_position:
        return _snake_status_result(
            base,
            threshold_status="snake_audited_first_crossing_cost_invalid",
            reason="compiled_cost_history_position_tau_mismatch",
            crossing=crossing,
            sidecar_key=sidecar_key,
            abs_delta_e=primary_error,
            sidecar_validation_status="invalid",
            sidecar_validation_reason="compiled_cost_history_position_tau_mismatch",
            sidecar_source_kind=sidecar_source_kind or None,
        )
    expected_benchmark = str(
        crossing.get("benchmark_id")
        or row.get("benchmark_id")
        or (record or {}).get("case_id")
        or (record or {}).get("record_id")
        or ""
    )
    sidecar_benchmark = str(sidecar.get("benchmark_id") or "")
    if sidecar_benchmark and expected_benchmark and sidecar_benchmark != expected_benchmark:
        return _snake_status_result(
            base,
            threshold_status="snake_audited_first_crossing_cost_invalid",
            reason="compiled_cost_benchmark_id_mismatch",
            crossing=crossing,
            sidecar_key=sidecar_key,
            abs_delta_e=primary_error,
            sidecar_validation_status="invalid",
            sidecar_validation_reason="compiled_cost_benchmark_id_mismatch",
            sidecar_source_kind=sidecar_source_kind or None,
        )
    expected_hash = str(sidecar.get("source_result_sha256") or "")
    if not expected_hash:
        return _snake_status_result(
            base,
            threshold_status="snake_audited_first_crossing_cost_invalid",
            reason="compiled_cost_source_result_sha256_missing",
            crossing=crossing,
            sidecar_key=sidecar_key,
            abs_delta_e=primary_error,
            sidecar_validation_status="invalid",
            sidecar_validation_reason="compiled_cost_source_result_sha256_missing",
            sidecar_source_kind=sidecar_source_kind or None,
        )
    hash_path: str | Path | None = sidecar_source_result_path or result_path
    actual_file_hash = _sha256_file(hash_path)
    actual_canonical_hash = _sha256_json_without_snake_sidecars(hash_path)
    row_hash = str(row.get("source_result_sha256") or "")
    if actual_file_hash is not None or actual_canonical_hash is not None:
        hash_verified = bool(
            (actual_file_hash and actual_file_hash == expected_hash)
            or (actual_canonical_hash and actual_canonical_hash == expected_hash)
        )
    else:
        hash_verified = bool(row_hash and row_hash == expected_hash)
    if not hash_verified:
        return _snake_status_result(
            base,
            threshold_status="snake_audited_first_crossing_cost_invalid",
            reason="compiled_cost_source_result_sha256_mismatch",
            crossing=crossing,
            sidecar_key=sidecar_key,
            abs_delta_e=primary_error,
            sidecar_validation_status="invalid",
            sidecar_validation_reason="compiled_cost_source_result_sha256_mismatch",
            sidecar_hash_verified=False,
            sidecar_source_kind=sidecar_source_kind or None,
            source_result_sha256=expected_hash,
        )
    sidecar_error = _first_numeric_from_mapping(sidecar, "primary_error_at_crossing", "abs_delta_e_at_crossing")
    if sidecar_error is None:
        return _snake_status_result(
            base,
            threshold_status="snake_audited_first_crossing_cost_invalid",
            reason="compiled_cost_primary_error_missing",
            crossing=crossing,
            sidecar_key=sidecar_key,
            abs_delta_e=primary_error,
            sidecar_validation_status="invalid",
            sidecar_validation_reason="compiled_cost_primary_error_missing",
            sidecar_hash_verified=hash_verified,
            sidecar_source_kind=sidecar_source_kind or None,
            source_result_sha256=expected_hash,
        )
    if sidecar_error > threshold_value:
        return _sidecar_status_result(
            base,
            threshold_status="snake_audited_first_crossing_cost_invalid",
            reason="compiled_cost_primary_error_above_threshold",
            crossing=crossing,
            sidecar_key=sidecar_key,
            abs_delta_e=sidecar_error,
            sidecar_validation_status="invalid",
            sidecar_validation_reason="compiled_cost_primary_error_above_threshold",
            sidecar_hash_verified=hash_verified,
            sidecar_source_kind=sidecar_source_kind or None,
            source_result_sha256=expected_hash,
        )
    if not math.isclose(float(sidecar_error), float(crossing_primary_error), rel_tol=0.0, abs_tol=1e-12):
        return _sidecar_status_result(
            base,
            threshold_status="snake_audited_first_crossing_cost_invalid",
            reason="compiled_cost_primary_error_mismatch",
            crossing=crossing,
            sidecar_key=sidecar_key,
            abs_delta_e=sidecar_error,
            sidecar_validation_status="invalid",
            sidecar_validation_reason="compiled_cost_primary_error_mismatch",
            sidecar_hash_verified=hash_verified,
            sidecar_source_kind=sidecar_source_kind or None,
            source_result_sha256=expected_hash,
        )
    count_2q = _first_numeric_from_mapping(sidecar, "compiled_count_2q_total", "count_2q", "N_2q")
    circuit_depth = _first_numeric_from_mapping(sidecar, "compiled_depth_total", "circuit_depth", "D_circ")
    depth_2q = _first_numeric_from_mapping(sidecar, "compiled_depth_2q_total", "depth_2q", "D_2q")
    if count_2q is None or depth_2q is None or circuit_depth is None:
        return _sidecar_status_result(
            base,
            threshold_status="snake_audited_first_crossing_cost_invalid",
            reason="compiled_cost_required_resource_missing",
            crossing=crossing,
            sidecar_key=sidecar_key,
            abs_delta_e=primary_error,
            sidecar_validation_status="invalid",
            sidecar_validation_reason="compiled_cost_required_resource_missing",
            sidecar_hash_verified=hash_verified,
            sidecar_source_kind=sidecar_source_kind or None,
            source_result_sha256=expected_hash,
        )
    for name, value in (("count_2q", count_2q), ("depth_2q", depth_2q), ("circuit_depth", circuit_depth)):
        if value is not None and float(value) < 0.0:
            return _snake_status_result(
                base,
                threshold_status="snake_audited_first_crossing_cost_invalid",
                reason=f"invalid_negative_compiled_resource_value:{name}",
                crossing=crossing,
                sidecar_key=sidecar_key,
                abs_delta_e=primary_error,
                sidecar_validation_status="invalid",
                sidecar_validation_reason=f"invalid_negative_compiled_resource_value:{name}",
                sidecar_hash_verified=hash_verified,
                sidecar_source_kind=sidecar_source_kind or None,
                source_result_sha256=expected_hash,
            )
    depth_ok, depth_reason = _compiled_depth_ordering_check(
        depth_2q=depth_2q,
        circuit_depth=circuit_depth,
        source_text=sidecar_source_text,
    )
    if not depth_ok:
        return _snake_status_result(
            base,
            threshold_status="snake_audited_first_crossing_cost_invalid",
            reason=depth_reason,
            crossing=crossing,
            sidecar_key=sidecar_key,
            abs_delta_e=primary_error,
            sidecar_validation_status="invalid",
            sidecar_validation_reason=depth_reason,
            sidecar_hash_verified=hash_verified,
            sidecar_source_kind=sidecar_source_kind or None,
            source_result_sha256=expected_hash,
        )
    s_alg = _first_numeric_from_mapping(sidecar, "S_alg")
    s_alg_missing_reason = sidecar.get("S_alg_missing_reason")
    if s_alg is None and not s_alg_missing_reason:
        return _snake_status_result(
            base,
            threshold_status="snake_audited_first_crossing_cost_invalid",
            reason="compiled_cost_S_alg_missing_without_reason",
            crossing=crossing,
            sidecar_key=sidecar_key,
            abs_delta_e=primary_error,
            sidecar_validation_status="invalid",
            sidecar_validation_reason="compiled_cost_S_alg_missing_without_reason",
            sidecar_hash_verified=hash_verified,
            sidecar_source_kind=sidecar_source_kind or None,
            source_result_sha256=expected_hash,
        )
    if s_alg is not None and float(s_alg) < 0.0:
        return _snake_status_result(
            base,
            threshold_status="snake_audited_first_crossing_cost_invalid",
            reason="invalid_negative_compiled_resource_value:S_alg",
            crossing=crossing,
            sidecar_key=sidecar_key,
            abs_delta_e=primary_error,
            sidecar_validation_status="invalid",
            sidecar_validation_reason="invalid_negative_compiled_resource_value:S_alg",
            sidecar_hash_verified=hash_verified,
            sidecar_source_kind=sidecar_source_kind or None,
            source_result_sha256=expected_hash,
        )
    metric_count = _first_numeric_from_mapping(sidecar, "N_metric", "metric_operator_probe_count_proxy")
    return {
        **base,
        "threshold_status": "ok_native_first_hit",
        "abs_delta_e": float(sidecar_error if sidecar_error is not None else primary_error if primary_error is not None else 0.0),
        "source": "snake_audited_first_crossing_compiled_cost",
        "first_hit_semantics": "snake_audited_history_position_tau",
        "cost_source": "snake_audited_first_crossing_compiled_cost",
        "S_alg": None if s_alg is None else float(s_alg),
        "S_norm": None if s_alg is None else float(s_alg),
        "S_alg_status": "missing_from_sidecar" if s_alg is None else "ok",
        "S_alg_missing_reason": str(s_alg_missing_reason) if s_alg_missing_reason else None,
        "components": sidecar.get("table_i_measurement_event_ledger") or sidecar.get("components"),
        "component_sources": {"sidecar": sidecar_key},
        "count_2q": float(count_2q),
        "depth_2q": None if depth_2q is None else float(depth_2q),
        "circuit_depth": float(circuit_depth),
        "compiled_count_2q_total": float(count_2q),
        "compiled_depth_2q_total": None if depth_2q is None else float(depth_2q),
        "compiled_depth_total": float(circuit_depth),
        "N_metric": None if metric_count is None else float(metric_count),
        "metric_fraction": None if metric_count is None or s_alg is None or s_alg <= 0.0 else float(metric_count / s_alg),
        "paper_i_first_crossing": dict(crossing),
        "snake_first_crossing_cost_sidecar_key": sidecar_key,
        "snake_first_crossing_history_position_tau": history_position,
        "sidecar_validation_status": "ok",
        "sidecar_validation_reason": None,
        "sidecar_hash_verified": True,
        "sidecar_source_kind": sidecar_source_kind or "snake_qiskit_compiled_first_hit_ansatz_circuit",
        "source_result_sha256": expected_hash,
        "source_result_path": sidecar_source_result_path or None,
        "reconstructability_status": sidecar_reconstructability_status or None,
        "first_hit_cost_source_kind": sidecar_source_kind or "snake_qiskit_compiled_first_hit_ansatz_circuit",
        "resource_display_allowed": True,
        "compiled_resource_validation_status": "ok",
        "compiled_resource_validation_reason": None,
        "method_cost_semantics": "snake_first_hit_sidecar_required",
        "source_resource_fields_present": {
            "count_2q": True,
            "depth_2q": depth_2q is not None,
            "circuit_depth": True,
        },
        "S_var_status": "missing_threshold_state",
    }


def table_i_threshold_cost_from_row(
    *,
    algorithm_id: str,
    row: Mapping[str, Any],
    threshold: float,
    record: Mapping[str, Any] | None = None,
    result_path: str | Path | None = None,
    enrichment_path: str | Path | None = None,
) -> dict[str, Any]:
    """Classify and normalize one Table-I fixed-accuracy threshold row.

    This is the reporting gate for calibrated fixed-accuracy tables.  It
    separates target hits from target misses and refuses to promote raw
    shot/proxy totals into ``S_alg``.  For adaptive methods, hits require the
    native first-hit ansatz cost; misses may report the Qiskit-compiled final
    ansatz cost when that explicit provenance is present.
    """

    alg = str(algorithm_id)
    threshold_value = float(threshold)
    delta = _num(row.get("delta_E_abs", row.get("abs_delta_e")))
    source = str(row.get("source") or row.get("first_hit_source") or "").strip()
    first_hit_semantics = str(row.get("first_hit_semantics") or "").strip()
    base: dict[str, Any] = {
        "schema": TABLE_I_THRESHOLD_COST_SCHEMA,
        "algorithm_id": alg,
        "threshold": threshold_value,
        "record_id": None if record is None else record.get("record_id"),
        "case_id": None if record is None else record.get("case_id"),
        "result_path": None if result_path is None else str(result_path),
        "enrichment_path": None if enrichment_path is None else str(enrichment_path),
        "abs_delta_e": None if delta is None else float(delta),
        "source": source or None,
        "first_hit_semantics": first_hit_semantics or None,
        "S_alg": None,
        "components": None,
        "cost_source": None,
        "S_var": None,
        "S_phys_var": None,
        "S_var_status": None,
        "S_var_cost_source": None,
        "S_var_components": None,
        "method_cost_semantics": _method_cost_semantics(alg),
        "resource_display_allowed": False,
        "compiled_resource_validation_status": None,
        "compiled_resource_validation_reason": None,
        "first_hit_cost_source_kind": None,
        "source_resource_fields_present": None,
    }
    if alg == SNAKE_TABLE_I_ALGORITHM_ID:
        return _table_i_snake_threshold_cost_from_row(
            row=row,
            threshold_value=threshold_value,
            base=base,
            record=record,
            result_path=result_path,
        )
    if delta is None:
        return {**base, "threshold_status": "missing_delta_e"}

    native_sources = {
        "native_adaptive_iteration",
        "native_first_hit",
        "native_adapt_iteration",
    }
    terminal_fallback_sources = {
        "final_row_fallback",
        "terminal_row_fallback",
    }
    source_kind = str(
        row.get("first_hit_cost_source_kind")
        or row.get("compiled_resource_source_kind")
        or row.get("source_kind")
        or ""
    )
    if float(delta) > threshold_value:
        has_final_cost_source = (
            source_kind in QISKIT_FINAL_ANSATZ_COST_SOURCE_KINDS
            or (
                alg in _FIXED_TERMINAL_TABLE_I_METHOD_IDS
                and source_kind == "qiskit_compiled_terminal_only_fixed_ansatz"
            )
        )
        threshold_status = "not_reached_final_ansatz" if has_final_cost_source else "not_reached"
    elif alg in _FIXED_TERMINAL_TABLE_I_METHOD_IDS:
        threshold_status = "ok_terminal_only_method"
    elif source in native_sources or first_hit_semantics.startswith("native_first_crossing"):
        threshold_status = "ok_native_first_hit"
    elif source in terminal_fallback_sources or alg in _ADAPTIVE_TABLE_I_METHOD_IDS:
        return {
            **base,
            "threshold_status": "terminal_upper_bound_missing_native_first_hit",
            "S_var_status": "missing_threshold_state",
        }
    else:
        return {**base, "threshold_status": "missing_required_components", "reason": "unknown_threshold_row_source"}

    raw_proxy = {
        "shots_total": _num(row.get("shots_total")),
        "shot_cost_proxy": _num(row.get("shot_cost_proxy")),
        "measurement_shots_proxy": _num(row.get("measurement_shots_proxy")),
        "shot_proxy": _num(row.get("shot_proxy")),
    }
    try:
        replay_ledger, replay_status = _table_i_event_ledger_from_comparator_row(algorithm_id=alg, row=row)
        alg_row = dict(row)
        if replay_ledger is not None and "table_i_measurement_event_ledger" not in alg_row:
            alg_row["table_i_measurement_event_ledger"] = replay_ledger
        metric, updates, statuses = algorithmic_measurement_work_from_row(row=alg_row, raw_proxy=raw_proxy)
    except Exception as exc:  # pragma: no cover - defensive reporting guard
        return {
            **base,
            "threshold_status": "missing_required_components",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    if str(statuses.get("S_alg") or "") != "ok" or _num(updates.get("S_alg")) is None:
        status = str(metric.get("status") or statuses.get("S_alg") or "missing_required_components")
        threshold_status = "raw_proxy_rejected" if status == "legacy_proxy_not_event_ledger" else "missing_required_components"
        return {
            **base,
            "threshold_status": threshold_status,
            "reason": status,
            "legacy_raw_proxy": metric.get("legacy_raw_proxy"),
        }
    components = metric.get("components") if isinstance(metric, Mapping) else None
    metric_count = None
    if isinstance(components, Mapping):
        metric_count = _num(components.get("N_metric_probe", components.get("N_metric")))
    s_alg = float(updates["S_alg"])
    row_variance_metric = row.get("statevector_variance_metric")
    if not isinstance(row_variance_metric, Mapping):
        row_variance_metric = row.get("grouped_statevector_variance_metric")
    var_metric, var_updates, var_status = _statevector_variance_metric_from_components(
        row_variance_metric if isinstance(row_variance_metric, Mapping) else None
    )
    s_var = _num(var_updates.get("S_var")) if var_status == "ok" else None
    if threshold_status == "ok_native_first_hit":
        variance_scope = ""
        variance_provenance = ""
        if isinstance(row_variance_metric, Mapping):
            variance_scope = str(row_variance_metric.get("state_scope") or "")
            variance_provenance = str(
                row_variance_metric.get("provenance")
                or row_variance_metric.get("source_kind")
                or ""
            ).lower()
        has_threshold_state = (
            variance_scope in {"threshold_first_hit_state", "event_local", "event_local_threshold_state"}
            or "threshold" in variance_provenance
            or "first_hit" in variance_provenance
            or "event_local" in variance_provenance
        )
        if var_status != "ok" or not has_threshold_state:
            # Native first-hit rows in current exact-bench artifacts have threshold
            # costs but not the first-hit state/theta/schedule needed for a true
            # threshold-local grouped-variance replay.  Keep this explicit so the
            # fixed-accuracy surface cannot silently promote terminal variance.
            var_status = "missing_threshold_state"
            s_var = None
    resource_validation = _compiled_resource_validation(
        algorithm_id=alg,
        row=row,
        threshold_status=threshold_status,
    )
    return {
        **base,
        **resource_validation,
        "threshold_status": threshold_status,
        "S_alg": s_alg,
        "S_norm": s_alg,
        "components": components,
        "component_sources": metric.get("component_sources") if isinstance(metric, Mapping) else None,
        "cost_source": "algorithmic_measurement_work",
        "S_alg_status": "ok",
        "N_metric": None if metric_count is None else float(metric_count),
        "metric_fraction": None if metric_count is None or s_alg <= 0.0 else float(metric_count / s_alg),
        "S_var": None if s_var is None else float(s_var),
        "S_phys_var": None if s_var is None else float(s_var),
        "S_var_status": var_status,
        "S_var_cost_source": "threshold_statevector_variance_metric" if s_var is not None else None,
        "S_var_components": var_metric.get("components") if isinstance(var_metric, Mapping) and var_status == "ok" else None,
    }


def _explicit_physical_components(
    row: Mapping[str, Any],
    aliases: Mapping[str, Sequence[str]],
) -> tuple[dict[str, float] | None, dict[str, str], str | None, list[str]]:
    components: dict[str, float] = {}
    sources: dict[str, str] = {}
    missing: list[str] = []
    any_present = False
    for name, keys in aliases.items():
        value, source, invalid = _strict_first_num(row, keys)
        if source is not None:
            any_present = True
        if invalid is not None:
            return None, {"invalid": str(source)}, f"invalid_{source}", []
        if value is None:
            missing.append(name)
            continue
        if float(value) < 0.0:
            return None, {"invalid": str(source)}, f"negative_{source}", []
        components[name] = float(value)
        sources[name] = str(source)
    if missing:
        return None, sources, "partial" if any_present else None, missing
    return components, sources, None, []


def _physical_metric_from_components(
    *,
    kind: str,
    row_key: str,
    aliases: Mapping[str, Sequence[str]],
    row: Mapping[str, Any],
    missing_status: str = "missing_fresh_grouped_event_components",
    legacy_missing_status: str | None = None,
) -> tuple[dict[str, Any], dict[str, float], str]:
    components, sources, invalid, missing = _explicit_physical_components(row, aliases)
    if invalid is not None and invalid != "partial":
        status = "invalid_physical_component_value"
        return (
            {
                "status": status,
                "reason": invalid,
                row_key: None,
                "components": None,
                "component_sources": sources or None,
            },
            {},
            status,
        )
    if components is None:
        legacy_status = legacy_missing_status or "legacy_grouped_proxy_not_fresh_event_ledger"
        status = legacy_status if _legacy_grouped_proxy_present(row) else str(missing_status)
        return (
            {
                "status": status,
                "reason": status,
                "missing_components": missing or list(aliases.keys()),
                row_key: None,
                "components": None,
                "component_sources": None,
            },
            {},
            status,
        )
    total = sum(float(value) for value in components.values())
    updates = {row_key: float(total)}
    updates.update({key: float(value) for key, value in components.items()})
    return (
        {
            "status": "ok",
            row_key: float(total),
            "components": {key: float(value) for key, value in components.items()},
            "component_sources": sources,
            "measurement_model": kind,
        },
        updates,
        "ok",
    )


def physical_measurement_work_from_row(
    *,
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, float], dict[str, str]]:
    """Build optional physical proxies only from fresh event-level components.

    Legacy ``S_grp`` remains available as provenance, but this helper refuses to
    promote it into ``S_phys``/``S_l2`` because previous grouped fields may have
    been reconstructed from coarse algorithm-level summaries rather than an
    event ledger with a common variance/precision convention.
    """

    s_phys, phys_updates, phys_status = _physical_metric_from_components(
        kind="variance_grouped_observable_event_sum",
        row_key="S_phys",
        aliases=S_PHYS_COMPONENT_ALIASES,
        row=row,
    )
    s_l2, l2_updates, l2_status = _physical_metric_from_components(
        kind="grouped_l2_coefficient_event_sum",
        row_key="S_l2",
        aliases=S_L2_COMPONENT_ALIASES,
        row=row,
    )
    s_var, var_updates, var_status = _physical_metric_from_components(
        kind="statevector_grouped_variance_event_sum",
        row_key="S_var",
        aliases=S_VAR_COMPONENT_ALIASES,
        row=row,
        missing_status="missing_statevector_variance_event_components",
        legacy_missing_status="missing_statevector_variance_event_components",
    )
    updates: dict[str, float] = {}
    if phys_status == "ok":
        updates.update(phys_updates)
    if l2_status == "ok":
        updates.update(l2_updates)
    if var_status == "ok":
        updates.update(var_updates)
    metric = {
        "schema": PHYSICAL_MEASUREMENT_WORK_SCHEMA,
        "status": "ok" if phys_status == "ok" or l2_status == "ok" or var_status == "ok" else phys_status,
        "S_phys": s_phys,
        "S_l2": s_l2,
        "S_var": s_var,
        "S_phys_var": s_var,
        "event_count_convention": "fresh_measurement_bearing_events_only",
        "unit": "event_summed_physical_measurement_proxy_under_declared_model",
    }
    return metric, updates, {
        "S_phys": phys_status,
        "S_l2": l2_status,
        "S_var": var_status,
        "S_phys_var": var_status,
    }


def _table_i_event_ledger_from_comparator_row(
    *,
    algorithm_id: str,
    row: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    """Replay exact-bench comparator telemetry into the strict Table-I ledger.

    This is the noiseless measurement schedule currency.  It uses only runtime
    event-count fields emitted by the benchmark runners, not aggregate raw shot
    scalars.  The ledger is reporting-only and does not change the optimizer or
    algorithm path.
    """

    alg = str(algorithm_id)
    fixed_ids = {"static_hea_qiskit_vqe", "static_family_informed_vqe"}
    adaptive_ids = set(_ADAPT_VARIANT_IDS) | {"static_qiskit_adapt_vqe"}
    if alg not in fixed_ids and alg not in adaptive_ids:
        return None, "unsupported_algorithm_without_replay_rule"

    def nonnegative_value(keys: Sequence[str], *, required: bool = True, default: float | None = None) -> tuple[float | None, str | None, str | None]:
        value, source = _first_num(row, keys)
        if value is None:
            if required:
                return None, None, f"missing_{keys[0]}"
            return default, None, None
        if float(value) < 0.0:
            return None, source, f"negative_{source}"
        return float(value), source, None

    if alg in fixed_ids:
        energy_eval, energy_source, error = nonnegative_value(("energy_eval_count_proxy", "nfev"))
        if error is not None or energy_eval is None:
            return None, str(error or "missing_energy_eval_count_proxy")
        totals = {
            "N_H_outer_eval": float(energy_eval),
            "N_grad_probe": 0.0,
            "N_metric_probe": 0.0,
            "N_H_refit_eval": 0.0,
            "N_other_quantum": 0.0,
        }
        sources = {
            "N_H_outer_eval": str(energy_source),
            "N_grad_probe": "method_zero",
            "N_metric_probe": "method_zero",
            "N_H_refit_eval": "method_zero",
            "N_other_quantum": "method_zero",
        }
    else:
        selected_count, selected_source = _selected_operator_count(row)
        if selected_count is None:
            return None, "missing_selected_operator_count"
        selected_raw = _num(row.get("selected_operator_count"))
        if selected_raw is not None and float(selected_raw) < 0.0:
            return None, "negative_selected_operator_count"
        energy_eval, energy_source, error = nonnegative_value(("energy_eval_count_proxy", "nfev"))
        if error is not None or energy_eval is None:
            return None, str(error or "missing_energy_eval_count_proxy")
        grad, grad_source, error = nonnegative_value(("gradient_operator_probe_count_proxy",))
        if error is not None or grad is None:
            return None, str(error or "missing_gradient_operator_probe_count_proxy")
        metric = 0.0
        metric_source = "method_zero"
        metric_value, metric_source_found, metric_error = nonnegative_value(
            ("metric_operator_probe_count_proxy",),
            required=(alg != "static_qiskit_adapt_vqe"),
            default=0.0,
        )
        if metric_error is not None:
            return None, str(metric_error)
        if metric_value is not None and float(metric_value) > 0.0:
            metric = float(metric_value)
            metric_source = str(metric_source_found)
        if int(selected_count) > 0:
            n_h_outer = 0.0
            n_h_refit = float(energy_eval)
            h_outer_source = "adaptive_refit_partition"
            h_refit_source = str(energy_source)
        else:
            n_h_outer = float(energy_eval)
            n_h_refit = 0.0
            h_outer_source = str(energy_source)
            h_refit_source = "no_selected_operators"
        totals = {
            "N_H_outer_eval": float(n_h_outer),
            "N_grad_probe": float(grad),
            "N_metric_probe": float(metric),
            "N_H_refit_eval": float(n_h_refit),
            "N_other_quantum": 0.0,
        }
        sources = {
            "N_H_outer_eval": h_outer_source,
            "N_grad_probe": str(grad_source),
            "N_metric_probe": metric_source,
            "N_H_refit_eval": h_refit_source,
            "N_other_quantum": "method_zero",
            "selected_operator_count": str(selected_source),
        }

    return (
        {
            "schema": TABLE_I_EVENT_LEDGER_SCHEMA,
            "status": "ok",
            "source_kind": "exact_bench_noiseless_measurement_schedule_replay_v1",
            "algorithm_id": alg,
            "component_totals": totals,
            "component_sources": sources,
            "event_count_convention": "fresh_measurement_bearing_estimator_or_probe_events",
            "cache_policy": "no_cache_reuse_in_current_exact_bench_replay",
            "measurement_model_id": "noiseless_estimator_schedule_count_v1",
            "N_other_quantum": 0.0,
        },
        "ok",
    )


def _statevector_variance_metric_from_components(
    metric: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, float], str]:
    """Promote event-summed grouped statevector-variance components into ``S_var``."""

    missing_status = "missing_statevector_variance_event_components"
    if not isinstance(metric, Mapping) or str(metric.get("status") or "") != "ok":
        return (
            {
                "schema": STATEVECTOR_VARIANCE_METRIC_SCHEMA,
                "status": missing_status,
                "reason": str(metric.get("status") if isinstance(metric, Mapping) else "missing_statevector_variance_metric"),
                "S_var": None,
                "components": None,
                "component_sources": None,
                "measurement_model": "deterministic_greedy_qwc_grouped_statevector_variance_proxy_v1",
            },
            {},
            missing_status,
        )
    raw_components = metric.get("components")
    if not isinstance(raw_components, Mapping):
        return (
            {
                "schema": STATEVECTOR_VARIANCE_METRIC_SCHEMA,
                "status": missing_status,
                "reason": "missing_statevector_variance_components",
                "S_var": None,
                "components": None,
                "component_sources": None,
                "measurement_model": "deterministic_greedy_qwc_grouped_statevector_variance_proxy_v1",
            },
            {},
            missing_status,
        )
    components: dict[str, float] = {}
    component_sources: dict[str, str] = {}
    for key, aliases in S_VAR_COMPONENT_ALIASES.items():
        value: float | None = None
        source: str | None = None
        for alias in aliases:
            parsed = _num(raw_components.get(alias))
            if parsed is not None:
                value = float(parsed)
                source = str(alias)
                break
        if value is None:
            return (
                {
                    "schema": STATEVECTOR_VARIANCE_METRIC_SCHEMA,
                    "status": missing_status,
                    "reason": f"missing_{key}",
                    "S_var": None,
                    "components": None,
                    "component_sources": metric.get("component_sources"),
                    "measurement_model": "deterministic_greedy_qwc_grouped_statevector_variance_proxy_v1",
                },
                {},
                missing_status,
            )
        if float(value) < 0.0:
            return (
                {
                    "schema": STATEVECTOR_VARIANCE_METRIC_SCHEMA,
                    "status": "invalid_physical_component_value",
                    "reason": f"invalid_{source or key}",
                    "S_var": None,
                    "components": None,
                    "component_sources": metric.get("component_sources"),
                    "measurement_model": "deterministic_greedy_qwc_grouped_statevector_variance_proxy_v1",
                },
                {},
                "invalid_physical_component_value",
            )
        components[key] = float(value)
        component_sources[key] = str(source or key)
    total = float(sum(components.values()))
    updates = {"S_var": total, **components}
    return (
        {
            "schema": STATEVECTOR_VARIANCE_METRIC_SCHEMA,
            "status": "ok",
            "S_var": total,
            "components": components,
            "component_sources": metric.get("component_sources") or component_sources,
            "measurement_model": str(
                metric.get("measurement_model")
                or metric.get("model")
                or "deterministic_greedy_qwc_grouped_statevector_variance_proxy_v1"
            ),
            "state_scope": str(metric.get("state_scope") or "event_local_or_threshold_local_from_source_metric"),
            "provenance": str(metric.get("provenance") or metric.get("source_kind") or "statevector_variance_metric_components"),
            "source_kind": str(
                metric.get("source_kind")
                or "exact_bench_noiseless_grouped_statevector_variance_replay_from_event_statevectors"
            ),
        },
        updates,
        "ok",
    )


def physical_measurement_work_from_grouped_replay(
    *,
    grouped_metric: Mapping[str, Any],
    statevector_variance_metric: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, float], dict[str, str]]:
    """Promote a freshly replayed grouped-L2 schedule into ``S_l2``.

    This is not the legacy raw shot scalar.  It is the deterministic grouped
    coefficient proxy reconstructed from the resolved Hamiltonian, candidate
    gradient schedule, metric probe schedule, and refit/objective event counts.
    """

    if not isinstance(grouped_metric, Mapping) or str(grouped_metric.get("status") or "") != "ok":
        return (
            {
                "schema": PHYSICAL_MEASUREMENT_WORK_SCHEMA,
                "status": "missing_fresh_grouped_event_components",
                "reason": str(grouped_metric.get("status") if isinstance(grouped_metric, Mapping) else "missing_grouped_metric"),
                "S_phys": {"status": "missing_fresh_grouped_event_components", "S_phys": None},
                "S_l2": {"status": "missing_fresh_grouped_event_components", "S_l2": None},
                "S_var": {"status": "missing_statevector_variance_event_components", "S_var": None},
                "S_phys_var": {"status": "missing_statevector_variance_event_components", "S_var": None},
                "unit": "event_summed_physical_measurement_proxy_under_declared_model",
            },
            {},
            {
                "S_phys": "missing_fresh_grouped_event_components",
                "S_l2": "missing_fresh_grouped_event_components",
                "S_var": "missing_statevector_variance_event_components",
                "S_phys_var": "missing_statevector_variance_event_components",
            },
        )
    components = grouped_metric.get("components")
    if not isinstance(components, Mapping):
        return (
            {
                "schema": PHYSICAL_MEASUREMENT_WORK_SCHEMA,
                "status": "missing_fresh_grouped_event_components",
                "reason": "missing_grouped_components",
                "S_phys": {"status": "missing_fresh_grouped_event_components", "S_phys": None},
                "S_l2": {"status": "missing_fresh_grouped_event_components", "S_l2": None},
                "S_var": {"status": "missing_statevector_variance_event_components", "S_var": None},
                "S_phys_var": {"status": "missing_statevector_variance_event_components", "S_var": None},
                "unit": "event_summed_physical_measurement_proxy_under_declared_model",
            },
            {},
            {
                "S_phys": "missing_fresh_grouped_event_components",
                "S_l2": "missing_fresh_grouped_event_components",
                "S_var": "missing_statevector_variance_event_components",
                "S_phys_var": "missing_statevector_variance_event_components",
            },
        )
    mapping = {
        "S_l2_H_outer": "S_grp_H_outer",
        "S_l2_grad": "S_grp_grad",
        "S_l2_metric": "S_grp_metric",
        "S_l2_H_refit": "S_grp_H_refit",
    }
    l2_components: dict[str, float] = {}
    for dst, src in mapping.items():
        value = _num(components.get(src))
        if value is None or float(value) < 0.0:
            return (
                {
                    "schema": PHYSICAL_MEASUREMENT_WORK_SCHEMA,
                    "status": "invalid_physical_component_value",
                    "reason": f"invalid_{src}",
                    "S_phys": {"status": "missing_fresh_grouped_event_components", "S_phys": None},
                    "S_l2": {"status": "invalid_physical_component_value", "S_l2": None},
                    "S_var": {"status": "missing_statevector_variance_event_components", "S_var": None},
                    "S_phys_var": {"status": "missing_statevector_variance_event_components", "S_var": None},
                    "unit": "event_summed_physical_measurement_proxy_under_declared_model",
                },
                {},
                {
                    "S_phys": "missing_fresh_grouped_event_components",
                    "S_l2": "invalid_physical_component_value",
                    "S_var": "missing_statevector_variance_event_components",
                    "S_phys_var": "missing_statevector_variance_event_components",
                },
            )
        l2_components[dst] = float(value)
    total = float(sum(l2_components.values()))
    updates = {"S_l2": total, **l2_components}
    variance_source = statevector_variance_metric
    if variance_source is None:
        embedded_variance = grouped_metric.get("statevector_variance_metric")
        if isinstance(embedded_variance, Mapping):
            variance_source = embedded_variance
    var_metric, var_updates, var_status = _statevector_variance_metric_from_components(variance_source)
    if var_status == "ok":
        updates.update(var_updates)
    return (
        {
            "schema": PHYSICAL_MEASUREMENT_WORK_SCHEMA,
            "status": "ok",
            "S_phys": {"status": "missing_fresh_variance_event_components", "S_phys": None},
            "S_l2": {
                "status": "ok",
                "S_l2": total,
                "components": l2_components,
                "component_sources": grouped_metric.get("component_sources"),
                "measurement_model": "deterministic_greedy_qwc_grouped_l2_coeff_proxy_v1",
                "source_kind": "exact_bench_noiseless_grouped_l2_replay_from_context",
            },
            "S_var": var_metric,
            "S_phys_var": var_metric,
            "event_count_convention": "fresh_measurement_bearing_events_only",
            "unit": "event_summed_physical_measurement_proxy_under_declared_model",
        },
        updates,
        {
            "S_phys": "missing_fresh_variance_event_components",
            "S_l2": "ok",
            "S_var": var_status,
            "S_phys_var": var_status,
        },
    )


def _qwc_weight_exyz(label_exyz: str) -> int:
    return int(sum(1 for ch in str(label_exyz).lower() if ch in {"x", "y", "z"}))


def _qwc_merge(lhs_key: str, rhs_key: str) -> str | None:
    lhs = str(lhs_key).lower()
    rhs = str(rhs_key).lower()
    if len(lhs) != len(rhs):
        return None
    merged: list[str] = []
    for lhs_ch, rhs_ch in zip(lhs, rhs):
        if lhs_ch == "e":
            merged.append(rhs_ch)
        elif rhs_ch in {"e", lhs_ch}:
            merged.append(lhs_ch)
        else:
            return None
    return "".join(merged)


def _pauli_polynomial_qwc_groups(
    polynomial: Any,
    *,
    coeff_tol: float = 1e-12,
) -> tuple[list[dict[str, Any]], int | None, list[tuple[str, complex]]]:
    """Return deterministic greedy QWC groups in repo e/x/y/z ordering."""

    terms_by_label: dict[str, complex] = {}
    width: int | None = None
    for term in polynomial.return_polynomial():
        label = str(term.pw2strng()).lower()
        coeff = complex(term.p_coeff)
        if any(ch not in {"e", "x", "y", "z"} for ch in label):
            raise ValueError(f"unsupported Pauli label {label!r}")
        if width is None:
            width = len(label)
        elif len(label) != width:
            raise ValueError("Hamiltonian Pauli labels have inconsistent widths")
        if abs(coeff) <= coeff_tol or _qwc_weight_exyz(label) == 0:
            continue
        terms_by_label[label] = terms_by_label.get(label, 0.0j) + coeff

    active_terms = [
        (label, coeff)
        for label, coeff in terms_by_label.items()
        if abs(coeff) > coeff_tol and _qwc_weight_exyz(label) > 0
    ]
    active_terms.sort(key=lambda item: (-_qwc_weight_exyz(item[0]), item[0]))

    groups: list[dict[str, Any]] = []
    for label, coeff in active_terms:
        best_idx: int | None = None
        best_key: str | None = None
        best_delta: tuple[int, int] | None = None
        for idx, group in enumerate(groups):
            merged = _qwc_merge(str(group["basis_key"]), label)
            if merged is None:
                continue
            delta = (_qwc_weight_exyz(merged) - _qwc_weight_exyz(str(group["basis_key"])), idx)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_idx = int(idx)
                best_key = str(merged)
        if best_idx is None or best_key is None:
            groups.append({"basis_key": label, "terms": [(label, coeff)]})
        else:
            groups[best_idx]["basis_key"] = best_key
            groups[best_idx]["terms"].append((label, coeff))
    return groups, width, active_terms


def _pauli_polynomial_grouped_measurement_proxy(
    polynomial: Any,
    *,
    observable_kind: str,
    target_sigma: float = 1.0,
    coeff_tol: float = 1e-12,
) -> dict[str, Any]:
    """Compute the event-level grouped-Pauli proxy for one Pauli observable.

    The cost model is the unit-target-variance grouped observable proxy

        C_grp(O) = sigma^-2 (sum_b sqrt(sum_{nu in b} |c_nu|^2))^2,

    using a deterministic greedy qubit-wise-commuting basis cover in the repo's
    internal e/x/y/z Pauli convention. Identity offsets are classical constants
    and do not contribute measurement groups.
    """

    if target_sigma <= 0.0 or not math.isfinite(float(target_sigma)):
        raise ValueError("target_sigma must be positive and finite")
    groups, _width, active_terms = _pauli_polynomial_qwc_groups(polynomial, coeff_tol=coeff_tol)

    group_l2_coeff_sums: list[float] = []
    for group in groups:
        l2 = math.sqrt(sum(abs(complex(coeff)) ** 2 for _label, coeff in group["terms"]))
        group_l2_coeff_sums.append(float(l2))
    c_grp = (sum(group_l2_coeff_sums) ** 2) / (float(target_sigma) ** 2)
    return {
        "model": "deterministic_greedy_qwc_grouped_l2_coeff_proxy_v1",
        "observable_kind": str(observable_kind),
        "target_sigma": float(target_sigma),
        "term_count": int(len(active_terms)),
        "group_count": int(len(groups)),
        "group_basis_keys": [str(group["basis_key"]) for group in groups],
        "group_l2_coeff_sums": group_l2_coeff_sums,
        "C_grp": float(c_grp),
    }


def _pauli_group_polynomial(
    group: Mapping[str, Any],
    *,
    n_qubits: int,
    coeff_tol: float,
) -> PauliPolynomial:
    terms: list[PauliTerm] = []
    for label, coeff in group.get("terms", []):
        coeff_c = complex(coeff)
        if abs(coeff_c) <= coeff_tol:
            continue
        if abs(coeff_c.imag) > max(1e-10, 100.0 * float(coeff_tol)):
            raise ValueError(
                "statevector grouped variance requires Hermitian Pauli observables with real coefficients"
            )
        terms.append(PauliTerm(int(n_qubits), ps=str(label), pc=float(coeff_c.real)))
    if not terms:
        terms.append(PauliTerm(int(n_qubits), ps="e" * int(n_qubits), pc=0.0))
    return PauliPolynomial("JW", terms)


def _pauli_polynomial_grouped_statevector_variance_proxy(
    polynomial: Any,
    statevector: Any,
    *,
    observable_kind: str,
    target_sigma: float = 1.0,
    coeff_tol: float = 1e-12,
) -> dict[str, Any]:
    """Compute grouped noiseless statevector-variance cost for one observable.

    For each deterministic QWC group ``O_b``, this reports
    ``Var_psi(O_b)=<O_b^2>_psi-<O_b>_psi^2`` and
    ``C_var=sigma^-2*(sum_b sqrt(max(Var_psi(O_b), 0)))^2``.
    Identity offsets are classical constants and do not create measurement
    groups or variance.
    """

    if target_sigma <= 0.0 or not math.isfinite(float(target_sigma)):
        raise ValueError("target_sigma must be positive and finite")
    psi = _normalize_state(np.asarray(statevector, dtype=complex).reshape(-1))
    groups, width, active_terms = _pauli_polynomial_qwc_groups(polynomial, coeff_tol=coeff_tol)
    if width is not None and psi.size != (1 << int(width)):
        raise ValueError(
            f"Statevector length mismatch: got {psi.size}, expected {1 << int(width)} for nq={width}."
        )
    if not groups:
        return {
            "model": "deterministic_greedy_qwc_grouped_statevector_variance_proxy_v1",
            "observable_kind": str(observable_kind),
            "target_sigma": float(target_sigma),
            "term_count": int(len(active_terms)),
            "group_count": 0,
            "group_basis_keys": [],
            "group_variances": [],
            "group_sqrt_variances": [],
            "group_expectations": [],
            "group_second_moments": [],
            "C_var": 0.0,
            "statevector_dimension": int(psi.size),
            "variance_formula": "<O_b^2>_psi_minus_<O_b>_psi_squared",
        }

    if width is None:
        raise ValueError("Cannot infer observable width for non-empty grouped variance proxy")
    cache: dict[str, Any] = {}
    group_variances: list[float] = []
    group_sqrt_variances: list[float] = []
    group_expectations: list[float] = []
    group_second_moments: list[float] = []
    for group in groups:
        group_poly = _pauli_group_polynomial(group, n_qubits=int(width), coeff_tol=coeff_tol)
        compiled = compile_polynomial_action(group_poly, tol=float(coeff_tol), pauli_action_cache=cache)
        o_psi = apply_compiled_polynomial(psi, compiled)
        expectation = complex(np.vdot(psi, o_psi))
        if abs(expectation.imag) > max(1e-10, 100.0 * float(coeff_tol)):
            raise ValueError("statevector grouped variance expectation has non-negligible imaginary part")
        second_moment = float(np.real(np.vdot(o_psi, o_psi)))
        variance_raw = float(second_moment - float(expectation.real) ** 2)
        variance = float(max(variance_raw, 0.0))
        group_variances.append(variance)
        group_sqrt_variances.append(float(math.sqrt(variance)))
        group_expectations.append(float(expectation.real))
        group_second_moments.append(second_moment)
    c_var = (sum(group_sqrt_variances) ** 2) / (float(target_sigma) ** 2)
    return {
        "model": "deterministic_greedy_qwc_grouped_statevector_variance_proxy_v1",
        "observable_kind": str(observable_kind),
        "target_sigma": float(target_sigma),
        "term_count": int(len(active_terms)),
        "group_count": int(len(groups)),
        "group_basis_keys": [str(group["basis_key"]) for group in groups],
        "group_variances": group_variances,
        "group_sqrt_variances": group_sqrt_variances,
        "group_expectations": group_expectations,
        "group_second_moments": group_second_moments,
        "C_var": float(c_var),
        "statevector_dimension": int(psi.size),
        "variance_formula": "<O_b^2>_psi_minus_<O_b>_psi_squared",
    }


def _hamiltonian_grouped_measurement_proxy(
    hamiltonian: Any,
    *,
    target_sigma: float = 1.0,
    coeff_tol: float = 1e-12,
) -> dict[str, Any]:
    return _pauli_polynomial_grouped_measurement_proxy(
        hamiltonian,
        observable_kind="hamiltonian_energy",
        target_sigma=target_sigma,
        coeff_tol=coeff_tol,
    )


def _gradient_observable_polynomial(hamiltonian: Any, generator: Any) -> PauliPolynomial:
    """Return the Hermitian ADAPT gradient observable i[H, A]."""

    return 1j * ((hamiltonian * generator) - (generator * hamiltonian))


def _explicit_grouped_components(row: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, str], list[str], str | None]:
    components: dict[str, float] = {}
    sources: dict[str, str] = {}
    missing: list[str] = []
    for name, keys in GROUPED_MEASUREMENT_COMPONENT_ALIASES.items():
        value, source = _first_num(row, keys)
        if value is None:
            missing.append(name)
            continue
        if float(value) < 0.0:
            return {}, {}, [], f"negative_{source}"
        components[name] = float(value)
        sources[name] = str(source)
    return components, sources, missing, None


def _adaptive_pool_for_grouped_measurement(algorithm_id: str, context: Any) -> tuple[Any, ...]:
    alg = str(algorithm_id)
    if alg == "static_qiskit_adapt_vqe":
        return tuple(build_full_meta_candidate_pool(context))
    config = _get_config(alg)
    if _pool_name_for_config(config) == "qubit_excitation_singles_doubles_pool":
        return tuple(build_pairwise_qubit_excitation_pool(int(context.layout.total_qubits)))
    return tuple(build_full_meta_candidate_pool(context))


def _candidate_gradient_grouped_costs(context: Any, pool: Sequence[Any]) -> dict[str, dict[str, Any]]:
    costs: dict[str, dict[str, Any]] = {}
    for candidate in pool:
        observable = _gradient_observable_polynomial(context.hamiltonian, candidate.polynomial)
        proxy = _pauli_polynomial_grouped_measurement_proxy(
            observable,
            observable_kind="adapt_gradient_commutator_i_H_A",
        )
        costs[str(candidate.label)] = proxy
    return costs


def _adaptive_gradient_scan_labels(
    *,
    algorithm_id: str,
    row: Mapping[str, Any],
    pool: Sequence[Any],
) -> tuple[list[list[str]] | None, dict[str, Any]]:
    """Reconstruct gradient-probe event labels from benchmark scheduling telemetry."""

    alg = str(algorithm_id)
    pool_labels = [str(candidate.label) for candidate in pool]
    scan_count_raw, scan_source = _first_num(row, ("gradient_scan_count_proxy",))
    probe_count_raw, probe_source = _first_num(row, ("gradient_operator_probe_count_proxy",))
    if scan_count_raw is None or probe_count_raw is None:
        return None, {
            "status": "missing_gradient_probe_counts",
            "scan_source": scan_source,
            "probe_source": probe_source,
        }
    if float(scan_count_raw) < 0.0 or float(probe_count_raw) < 0.0:
        return None, {"status": "invalid_negative_gradient_probe_counts"}
    scan_count = int(round(float(scan_count_raw)))
    expected_probe_count = int(round(float(probe_count_raw)))

    if alg == "static_qiskit_adapt_vqe":
        scans = [list(pool_labels) for _ in range(scan_count)]
        event_count = sum(len(scan) for scan in scans)
        if event_count != expected_probe_count:
            return None, {
                "status": "gradient_schedule_probe_count_mismatch",
                "event_count": int(event_count),
                "expected_probe_count": int(expected_probe_count),
                "event_model": "qiskit_adapt_full_pool_each_scan",
            }
        return scans, {
            "status": "ok",
            "event_model": "qiskit_adapt_full_pool_each_scan",
            "scan_count": int(scan_count),
            "event_count": int(event_count),
        }

    config = _get_config(alg)
    history = row.get("adapt_history")
    if not isinstance(history, list):
        return None, {"status": "missing_adapt_history_for_gradient_schedule"}
    by_iteration = sorted(
        [entry for entry in history if isinstance(entry, Mapping)],
        key=lambda entry: int(entry.get("iteration") or 0),
    )
    scans: list[list[str]] = []
    selected_labels: set[str] = set()
    previous_selected_label: str | None = None
    for scan_idx in range(scan_count):
        if config.repeat_policy == "exclude_selected_labels":
            blocked = set(selected_labels)
        elif config.repeat_policy == "with_replacement_except_immediate_repeat" and previous_selected_label:
            blocked = {str(previous_selected_label)}
        else:
            blocked = set()
        labels = [label for label in pool_labels if label not in blocked]
        scans.append(labels)
        if scan_idx < len(by_iteration):
            entry = by_iteration[scan_idx]
            scored_count = _num(entry.get("candidate_count_scored"))
            if scored_count is not None and int(round(float(scored_count))) != len(labels):
                return None, {
                    "status": "gradient_schedule_candidate_count_mismatch",
                    "scan_index": int(scan_idx),
                    "event_count": int(len(labels)),
                    "reported_candidate_count_scored": int(round(float(scored_count))),
                }
            selected_batch = entry.get("selected_batch_labels")
            batch_labels = [str(label) for label in selected_batch] if isinstance(selected_batch, list) else []
            for label in batch_labels:
                selected_labels.add(str(label))
            if batch_labels:
                previous_selected_label = str(batch_labels[-1])
    event_count = sum(len(scan) for scan in scans)
    if event_count != expected_probe_count:
        return None, {
            "status": "gradient_schedule_probe_count_mismatch",
            "event_count": int(event_count),
            "expected_probe_count": int(expected_probe_count),
            "event_model": f"{config.repeat_policy}_reconstructed_from_adapt_history",
        }
    return scans, {
        "status": "ok",
        "event_model": f"{config.repeat_policy}_reconstructed_from_adapt_history",
        "scan_count": int(scan_count),
        "event_count": int(event_count),
        "history_length": int(len(by_iteration)),
    }


def _adaptive_gradient_grouped_component(
    *,
    algorithm_id: str,
    row: Mapping[str, Any],
    context: Any,
) -> tuple[float | None, dict[str, Any]]:
    pool = _adaptive_pool_for_grouped_measurement(str(algorithm_id), context)
    if not pool:
        return None, {"status": "empty_pool"}
    scans, schedule = _adaptive_gradient_scan_labels(algorithm_id=str(algorithm_id), row=row, pool=pool)
    if scans is None:
        return None, schedule
    costs = _candidate_gradient_grouped_costs(context, pool)
    missing_labels = sorted({label for scan in scans for label in scan if label not in costs})
    if missing_labels:
        return None, {"status": "missing_candidate_gradient_costs", "missing_labels": missing_labels}
    total = 0.0
    scan_costs: list[float] = []
    for scan in scans:
        cost = sum(float(costs[label]["C_grp"]) for label in scan)
        scan_costs.append(float(cost))
        total += float(cost)
    candidate_cost_values = [float(proxy["C_grp"]) for proxy in costs.values()]
    return float(total), {
        "status": "ok",
        **schedule,
        "pool_size": int(len(pool)),
        "candidate_cost_min": min(candidate_cost_values) if candidate_cost_values else None,
        "candidate_cost_max": max(candidate_cost_values) if candidate_cost_values else None,
        "candidate_cost_sum_per_full_scan": float(sum(candidate_cost_values)),
        "scan_costs": scan_costs,
        "observable_kind": "adapt_gradient_commutator_i_H_A",
        "cost_model": "event_sum_of_candidate_C_grp_i_H_A",
    }


def _adaptive_metric_component_is_semantic_zero(algorithm_id: str, row: Mapping[str, Any]) -> tuple[bool, str | None]:
    metric_count, metric_source = _first_num(row, ("metric_operator_probe_count_proxy",))
    if str(algorithm_id) == "static_qiskit_adapt_vqe":
        if metric_count is None or float(metric_count) == 0.0:
            return True, "append_only_adapt_has_no_metric_stage"
        if float(metric_count) < 0.0:
            return False, f"negative_{metric_source}"
        return False, "unexpected_metric_probe_count_for_metricless_method"
    config = _get_config(str(algorithm_id))
    if _is_geo_config(config):
        return False, None
    if metric_count is None or float(metric_count) == 0.0:
        return True, "non_geo_adapt_has_no_metric_stage"
    if float(metric_count) < 0.0:
        return False, f"negative_{metric_source}"
    return False, None


def _symmetrized_generator_product_polynomial(lhs: Any, rhs: Any) -> PauliPolynomial:
    return 0.5 * ((lhs * rhs) + (rhs * lhs))


def _metric_pair_labels(labels: Sequence[str]) -> list[tuple[str, str]]:
    ordered = [str(label) for label in labels]
    return [(ordered[i], ordered[j]) for i in range(len(ordered)) for j in range(i, len(ordered))]


def _metric_pair_grouped_cost(
    *,
    lhs_label: str,
    rhs_label: str,
    by_label: Mapping[str, Any],
    cache: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    key = (str(lhs_label), str(rhs_label))
    symmetric_key = (str(rhs_label), str(lhs_label))
    if key in cache:
        return cache[key]
    if symmetric_key in cache:
        return cache[symmetric_key]
    lhs = by_label.get(str(lhs_label))
    rhs = by_label.get(str(rhs_label))
    if lhs is None or rhs is None:
        missing = lhs_label if lhs is None else rhs_label
        raise KeyError(f"metric pair candidate {missing!r} not found in rebuilt pool")
    observable = _symmetrized_generator_product_polynomial(lhs.polynomial, rhs.polynomial)
    proxy = _pauli_polynomial_grouped_measurement_proxy(
        observable,
        observable_kind="symmetrized_generator_product_metric_proxy",
    )
    cache[key] = proxy
    return proxy


def _metric_block_grouped_cost(
    *,
    labels: Sequence[str],
    metric_eval_count: int,
    metric_operator_probe_count: int,
    by_label: Mapping[str, Any],
    pair_cost_cache: dict[tuple[str, str], dict[str, Any]],
) -> tuple[float, int]:
    pairs = _metric_pair_labels(labels)
    if int(metric_operator_probe_count) != int(metric_eval_count) * len(pairs):
        raise ValueError(
            "metric block probe count mismatch: "
            f"{metric_operator_probe_count} != {metric_eval_count} * {len(pairs)}"
        )
    cost_per_eval = 0.0
    for lhs, rhs in pairs:
        cost_per_eval += float(
            _metric_pair_grouped_cost(
                lhs_label=lhs,
                rhs_label=rhs,
                by_label=by_label,
                cache=pair_cost_cache,
            )["C_grp"]
        )
    return float(metric_eval_count) * float(cost_per_eval), len(pairs)


def _geo_metric_grouped_component(
    *,
    algorithm_id: str,
    row: Mapping[str, Any],
    context: Any,
) -> tuple[float | None, dict[str, Any]]:
    config = _get_config(str(algorithm_id))
    if not _is_geo_config(config):
        return None, {"status": "not_geo_algorithm"}
    metric_count, metric_source = _first_num(row, ("metric_operator_probe_count_proxy",))
    if metric_count is None:
        return None, {"status": "missing_metric_operator_probe_count_proxy"}
    if float(metric_count) < 0.0:
        return None, {"status": f"negative_{metric_source}"}
    pool = _adaptive_pool_for_grouped_measurement(str(algorithm_id), context)
    by_label = {str(candidate.label): candidate for candidate in pool}
    pair_cost_cache: dict[tuple[str, str], dict[str, Any]] = {}
    history = row.get("adapt_history")
    if not isinstance(history, list):
        return None, {"status": "missing_adapt_history_for_geo_metric_schedule"}

    total = 0.0
    reported_probe_count = 0
    blocks: list[dict[str, Any]] = []
    for entry in sorted(
        [item for item in history if isinstance(item, Mapping)],
        key=lambda item: int(item.get("iteration") or 0),
    ):
        iteration = int(entry.get("iteration") or 0)
        selector_labels = entry.get("selector_metric_candidate_labels")
        selector_probe_count = _num(entry.get("selector_metric_probe_count"))
        if not isinstance(selector_labels, list) or selector_probe_count is None:
            return None, {
                "status": "missing_selector_metric_event_telemetry",
                "iteration": iteration,
            }
        selector_labels_s = [str(label) for label in selector_labels]
        selector_cost, selector_pair_count = _metric_block_grouped_cost(
            labels=selector_labels_s,
            metric_eval_count=1,
            metric_operator_probe_count=int(round(float(selector_probe_count))),
            by_label=by_label,
            pair_cost_cache=pair_cost_cache,
        )
        total += selector_cost
        reported_probe_count += int(round(float(selector_probe_count)))
        blocks.append(
            {
                "iteration": iteration,
                "block_kind": "geo_selector_metric",
                "label_count": int(len(selector_labels_s)),
                "metric_eval_count": 1,
                "metric_pair_count_per_eval": int(selector_pair_count),
                "metric_operator_probe_count": int(round(float(selector_probe_count))),
                "C_grp": float(selector_cost),
            }
        )

        qngd_blocks = entry.get("qngd_metric_event_blocks")
        if not isinstance(qngd_blocks, list):
            return None, {
                "status": "missing_qngd_metric_event_blocks",
                "iteration": iteration,
            }
        for block in qngd_blocks:
            if not isinstance(block, Mapping):
                return None, {"status": "invalid_qngd_metric_event_block", "iteration": iteration}
            labels = block.get("selected_labels")
            eval_count = _num(block.get("metric_eval_count"))
            probe_count = _num(block.get("metric_operator_probe_count"))
            if not isinstance(labels, list) or eval_count is None or probe_count is None:
                return None, {"status": "incomplete_qngd_metric_event_block", "iteration": iteration}
            labels_s = [str(label) for label in labels]
            block_cost, pair_count = _metric_block_grouped_cost(
                labels=labels_s,
                metric_eval_count=int(round(float(eval_count))),
                metric_operator_probe_count=int(round(float(probe_count))),
                by_label=by_label,
                pair_cost_cache=pair_cost_cache,
            )
            total += block_cost
            reported_probe_count += int(round(float(probe_count)))
            blocks.append(
                {
                    "iteration": iteration,
                    "block_kind": str(block.get("block_kind") or "geo_qngd_metric"),
                    "label_count": int(len(labels_s)),
                    "metric_eval_count": int(round(float(eval_count))),
                    "metric_pair_count_per_eval": int(pair_count),
                    "metric_operator_probe_count": int(round(float(probe_count))),
                    "C_grp": float(block_cost),
                }
            )

    expected = int(round(float(metric_count)))
    if reported_probe_count != expected:
        # The ADAPT runner counts the final Geo natural-gradient stop check in
        # metric_operator_probe_count_proxy.  That stop-check scan does not
        # append an operator, so older/current artifacts have no adapt_history
        # entry for it.  Reconstruct the scan labels from the pool and the
        # immediate-repeat blocking rule; do this only when the count mismatch
        # is exactly explained by one final selector metric block.
        selected_labels_seen = {
            str(label)
            for label in (row.get("selected_operators") or [])
            if str(label)
        }
        previous_label = None
        if isinstance(row.get("selected_operators"), list) and row.get("selected_operators"):
            previous_label = str(row.get("selected_operators")[-1])
        elif history:
            last_entry = sorted(
                [item for item in history if isinstance(item, Mapping)],
                key=lambda item: int(item.get("iteration") or 0),
            )[-1]
            batch_labels = last_entry.get("selected_batch_labels")
            if isinstance(batch_labels, list) and batch_labels:
                previous_label = str(batch_labels[-1])

        blocked = _blocked_labels_for_config(
            config,
            selected_labels=selected_labels_seen,
            previous_selected_label=previous_label,
        )
        final_labels = [str(candidate.label) for candidate in pool if str(candidate.label) not in blocked]
        remaining = int(expected) - int(reported_probe_count)
        final_pair_count = len(final_labels) * (len(final_labels) + 1) // 2
        if remaining == final_pair_count and remaining > 0:
            stop_cost, pair_count = _metric_block_grouped_cost(
                labels=final_labels,
                metric_eval_count=1,
                metric_operator_probe_count=int(remaining),
                by_label=by_label,
                pair_cost_cache=pair_cost_cache,
            )
            total += float(stop_cost)
            reported_probe_count += int(remaining)
            blocks.append(
                {
                    "iteration": int(len(history)),
                    "block_kind": "geo_final_stop_selector_metric",
                    "label_count": int(len(final_labels)),
                    "metric_eval_count": 1,
                    "metric_pair_count_per_eval": int(pair_count),
                    "metric_operator_probe_count": int(remaining),
                    "C_grp": float(stop_cost),
                    "source": "reconstructed_from_metric_probe_count_proxy_and_repeat_policy",
                }
            )

    if reported_probe_count != expected:
        return None, {
            "status": "geo_metric_probe_count_mismatch",
            "reported_probe_count": int(reported_probe_count),
            "metric_operator_probe_count_proxy": int(expected),
        }
    return float(total), {
        "status": "ok",
        "model": "symmetrized_generator_product_grouped_proxy_v1",
        "exactness": "proxy_not_dressed_circuit_metric_measurement_product_term_only",
        "includes_generator_means": False,
        "source": "reconstructed_geo_metric_pair_schedule",
        "metric_operator_probe_count": int(reported_probe_count),
        "block_count": int(len(blocks)),
        "blocks": blocks,
        "unique_metric_pair_cost_count": int(len(pair_cost_cache)),
        "S_grp_metric": float(total),
    }


def grouped_measurement_proxy_from_row_and_context(
    *,
    algorithm_id: str,
    row: Mapping[str, Any],
    context: Any,
) -> tuple[dict[str, Any], dict[str, float], dict[str, str]]:
    """Build grouped physical measurement proxy from explicit fields plus H structure.

    Hamiltonian energy/refit components may be derived from the resolved
    Hamiltonian and disjoint energy-evaluation bins. Gradient and metric grouped
    components are promoted only when explicit grouped component fields are
    present or when the corresponding method/count is semantically zero.
    """

    components, sources, missing, invalid = _explicit_grouped_components(row)
    if invalid is not None:
        return (
            {
                "schema": GROUPED_MEASUREMENT_PROXY_SCHEMA,
                "status": "invalid_grouped_measurement_value",
                "reason": invalid,
                "S_grp_total": None,
                "components": None,
                "component_sources": None,
                "unit": "grouped_pauli_shot_proxy_under_common_measurement_model",
            },
            {},
            {"S_grp": "invalid_grouped_measurement_value"},
        )

    h_proxy = _hamiltonian_grouped_measurement_proxy(context.hamiltonian)
    h_cost = float(h_proxy["C_grp"])
    gradient_proxy: dict[str, Any] | None = None
    metric_proxy: dict[str, Any] | None = None
    alg = str(algorithm_id)
    fixed_ids = {"static_hea_qiskit_vqe", "static_family_informed_vqe"}
    adaptive_ids = set(_ADAPT_VARIANT_IDS) | {"static_qiskit_adapt_vqe"}

    row_updates: dict[str, float] = {}

    if alg in fixed_ids:
        value, source = _first_num(row, ("energy_eval_count_proxy", "nfev"))
        if value is None:
            status = "missing_grouped_measurement_breakdown"
            missing = sorted(set(missing + ["S_grp_H_outer"]))
        elif float(value) < 0.0:
            return (
                {
                    "schema": GROUPED_MEASUREMENT_PROXY_SCHEMA,
                    "status": "invalid_grouped_measurement_value",
                    "reason": f"negative_{source}",
                    "S_grp_total": None,
                    "components": None,
                    "component_sources": None,
                    "hamiltonian_observable_proxy": h_proxy,
                    "unit": "grouped_pauli_shot_proxy_under_common_measurement_model",
                },
                {},
                {"S_grp": "invalid_grouped_measurement_value"},
            )
        else:
            components.setdefault("S_grp_H_outer", h_cost * float(value))
            sources.setdefault("S_grp_H_outer", f"hamiltonian_C_grp_unit_sigma*{source}")
        for key in ("S_grp_grad", "S_grp_metric", "S_grp_H_refit"):
            components.setdefault(key, 0.0)
            sources.setdefault(key, "method_zero")
    elif alg in adaptive_ids:
        selected_raw = _num(row.get("selected_operator_count"))
        if selected_raw is not None and float(selected_raw) < 0.0:
            return (
                {
                    "schema": GROUPED_MEASUREMENT_PROXY_SCHEMA,
                    "status": "invalid_grouped_measurement_value",
                    "reason": "negative_selected_operator_count",
                    "S_grp_total": None,
                    "components": None,
                    "component_sources": None,
                    "hamiltonian_observable_proxy": h_proxy,
                    "unit": "grouped_pauli_shot_proxy_under_common_measurement_model",
                },
                {},
                {"S_grp": "invalid_grouped_measurement_value"},
            )
        selected_count, selected_source = _selected_operator_count(row)
        energy_eval, energy_source = _first_num(row, ("energy_eval_count_proxy", "nfev"))
        if selected_count is not None and energy_eval is not None and float(energy_eval) >= 0.0:
            if int(selected_count) > 0:
                components.setdefault("S_grp_H_outer", 0.0)
                sources.setdefault("S_grp_H_outer", "adaptive_refit_partition")
                components.setdefault("S_grp_H_refit", h_cost * float(energy_eval))
                sources.setdefault("S_grp_H_refit", f"hamiltonian_C_grp_unit_sigma*{energy_source}")
            else:
                components.setdefault("S_grp_H_outer", h_cost * float(energy_eval))
                sources.setdefault("S_grp_H_outer", f"hamiltonian_C_grp_unit_sigma*{energy_source}")
                components.setdefault("S_grp_H_refit", 0.0)
                sources.setdefault("S_grp_H_refit", "no_selected_operators")
            sources.setdefault("selected_operator_count", str(selected_source))
        elif energy_eval is not None and float(energy_eval) < 0.0:
            return (
                {
                    "schema": GROUPED_MEASUREMENT_PROXY_SCHEMA,
                    "status": "invalid_grouped_measurement_value",
                    "reason": f"negative_{energy_source}",
                    "S_grp_total": None,
                    "components": None,
                    "component_sources": None,
                    "hamiltonian_observable_proxy": h_proxy,
                    "unit": "grouped_pauli_shot_proxy_under_common_measurement_model",
                },
                {},
                {"S_grp": "invalid_grouped_measurement_value"},
            )

        if "S_grp_grad" not in components:
            grad_cost, gradient_proxy = _adaptive_gradient_grouped_component(
                algorithm_id=alg,
                row=row,
                context=context,
            )
            if grad_cost is not None:
                components["S_grp_grad"] = float(grad_cost)
                sources["S_grp_grad"] = "event_sum_C_grp_i_H_A_from_reconstructed_gradient_schedule"
        else:
            gradient_proxy = {"status": "explicit_component_used"}

        metric_count, metric_source = _first_num(row, ("metric_operator_probe_count_proxy",))
        if metric_count is not None and float(metric_count) < 0.0:
            return (
                {
                    "schema": GROUPED_MEASUREMENT_PROXY_SCHEMA,
                    "status": "invalid_grouped_measurement_value",
                    "reason": f"negative_{metric_source}",
                    "S_grp_total": None,
                    "components": None,
                    "component_sources": None,
                    "hamiltonian_observable_proxy": h_proxy,
                    "unit": "grouped_pauli_shot_proxy_under_common_measurement_model",
                },
                {},
                {"S_grp": "invalid_grouped_measurement_value"},
            )
        if "S_grp_metric" not in components:
            metric_zero, metric_zero_source = _adaptive_metric_component_is_semantic_zero(alg, row)
            if metric_zero:
                components["S_grp_metric"] = 0.0
                sources["S_grp_metric"] = str(metric_zero_source)
                metric_proxy = {"status": "semantic_zero", "source": str(metric_zero_source)}
            elif metric_zero_source is not None and str(metric_zero_source).startswith("negative_"):
                return (
                    {
                        "schema": GROUPED_MEASUREMENT_PROXY_SCHEMA,
                        "status": "invalid_grouped_measurement_value",
                        "reason": metric_zero_source,
                        "S_grp_total": None,
                        "components": None,
                        "component_sources": None,
                        "hamiltonian_observable_proxy": h_proxy,
                        "gradient_observable_proxy": gradient_proxy,
                        "unit": "grouped_pauli_shot_proxy_under_common_measurement_model",
                    },
                    {},
                    {"S_grp": "invalid_grouped_measurement_value"},
                )
            elif metric_zero_source is not None:
                metric_proxy = {
                    "status": str(metric_zero_source),
                    "reason": "metricless adaptive method reported metric probes without explicit grouped metric component",
                }
            else:
                try:
                    metric_cost, metric_proxy = _geo_metric_grouped_component(
                        algorithm_id=alg,
                        row=row,
                        context=context,
                    )
                except Exception as exc:
                    metric_cost, metric_proxy = None, {
                        "status": "geo_metric_grouped_component_failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                if metric_cost is not None:
                    components["S_grp_metric"] = float(metric_cost)
                    sources["S_grp_metric"] = "symmetrized_generator_product_grouped_proxy"
                elif metric_proxy is None:
                    metric_proxy = {
                        "status": "missing_metric_observable_model",
                        "reason": "geo_metric_grouped_observable_cost_not_reconstructed_without_explicit_metric_components",
                    }
        else:
            metric_proxy = {"status": "explicit_component_used"}

    required = ("S_grp_H_outer", "S_grp_grad", "S_grp_metric", "S_grp_H_refit")
    missing_now = [key for key in required if key not in components]
    for key, value in components.items():
        if key in GROUPED_MEASUREMENT_COMPONENT_ALIASES:
            if not math.isfinite(float(value)) or float(value) < 0.0:
                return (
                    {
                        "schema": GROUPED_MEASUREMENT_PROXY_SCHEMA,
                        "status": "invalid_grouped_measurement_value",
                        "reason": f"invalid_{key}",
                        "S_grp_total": None,
                        "components": None,
                        "component_sources": None,
                        "hamiltonian_observable_proxy": h_proxy,
                        "gradient_observable_proxy": gradient_proxy,
                        "metric_observable_proxy": metric_proxy,
                        "unit": "grouped_pauli_shot_proxy_under_common_measurement_model",
                    },
                    {},
                    {"S_grp": "invalid_grouped_measurement_value"},
                )
            row_updates[key] = float(value)

    if missing_now:
        status = "partial_grouped_measurement_breakdown" if row_updates else "missing_grouped_measurement_breakdown"
        return (
            {
                "schema": GROUPED_MEASUREMENT_PROXY_SCHEMA,
                "status": status,
                "missing_components": missing_now,
                "S_grp_total": None,
                "components": {key: float(value) for key, value in components.items()} if components else None,
                "component_sources": sources if sources else None,
                "hamiltonian_observable_proxy": h_proxy,
                "gradient_observable_proxy": gradient_proxy,
                "metric_observable_proxy": metric_proxy,
                "unit": "grouped_pauli_shot_proxy_under_common_measurement_model",
            },
            row_updates,
            {"S_grp": status},
        )

    total = sum(float(components[key]) for key in required)
    row_updates["S_grp_total"] = float(total)
    return (
        {
            "schema": GROUPED_MEASUREMENT_PROXY_SCHEMA,
            "status": "ok",
            "S_grp_total": float(total),
            "components": {key: float(components[key]) for key in required},
            "component_sources": sources,
            "hamiltonian_observable_proxy": h_proxy,
            "gradient_observable_proxy": gradient_proxy,
            "metric_observable_proxy": metric_proxy,
            "unit": "grouped_pauli_shot_proxy_under_common_measurement_model",
        },
        row_updates,
        {"S_grp": "ok"},
    )


def _normalized_measurement_work(
    *,
    algorithm_id: str,
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, float], dict[str, str]]:
    """Build a cross-method normalized measurement-work proxy from raw telemetry.

    The proxy is reporting-only. It preserves raw shot fields and never derives
    components from legacy aggregate `shots_total` values.
    """
    raw_proxy = {
        "shots_total": _num(row.get("shots_total")),
        "shot_cost_proxy": _num(row.get("shot_cost_proxy")),
        "measurement_shots_proxy": _num(row.get("measurement_shots_proxy")),
    }
    weights = dict(DEFAULT_S_NORM_WEIGHTS)
    other_quantum, other_quantum_source = _measurement_work_other_quantum(row)
    if other_quantum < 0.0:
        return (
            {
                "schema": NORMALIZED_MEASUREMENT_WORK_SCHEMA,
                "status": "invalid_component_value",
                "reason": f"negative_{other_quantum_source}",
                "S_norm": None,
                "weights": weights,
                "components": None,
                "legacy_raw_proxy": raw_proxy,
                "N_other_quantum": other_quantum,
                "N_other_quantum_source": other_quantum_source,
            },
            {},
            {"S_norm": "invalid_component_value"},
        )
    if other_quantum > 0.0:
        return (
            {
                "schema": NORMALIZED_MEASUREMENT_WORK_SCHEMA,
                "status": "unassigned_other_quantum_work",
                "reason": "nonzero_N_other_quantum_requires_assignment_to_disjoint_bins",
                "S_norm": None,
                "weights": weights,
                "components": None,
                "legacy_raw_proxy": raw_proxy,
                "N_other_quantum": other_quantum,
                "N_other_quantum_source": other_quantum_source,
            },
            {},
            {"S_norm": "unassigned_other_quantum_work"},
        )
    components, sources = _explicit_measurement_work_components(row)
    if components is None and "invalid" in sources:
        return (
            {
                "schema": NORMALIZED_MEASUREMENT_WORK_SCHEMA,
                "status": "invalid_component_value",
                "reason": f"negative_{sources['invalid']}",
                "S_norm": None,
                "weights": weights,
                "components": None,
                "legacy_raw_proxy": raw_proxy,
            },
            {},
            {"S_norm": "invalid_component_value"},
        )
    if components is None:
        alg = str(algorithm_id)
        sources = {}
        components = {"N_H_outer_eval": 0.0, "N_grad": 0.0, "N_metric": 0.0, "N_H_refit_eval": 0.0}
        fixed_ids = {"static_hea_qiskit_vqe", "static_family_informed_vqe"}
        adaptive_ids = set(_ADAPT_VARIANT_IDS) | {"static_qiskit_adapt_vqe"}
        if alg in fixed_ids:
            value, source = _first_num(row, ("energy_eval_count_proxy", "nfev"))
            if value is None:
                return (
                    {
                        "schema": NORMALIZED_MEASUREMENT_WORK_SCHEMA,
                        "status": "missing_component_breakdown",
                        "reason": "missing_energy_eval_count_proxy",
                        "S_norm": None,
                        "weights": weights,
                        "components": None,
                        "legacy_raw_proxy": raw_proxy,
                    },
                    {},
                    {"S_norm": "missing_component_breakdown"},
                )
            if float(value) < 0.0:
                return (
                    {
                        "schema": NORMALIZED_MEASUREMENT_WORK_SCHEMA,
                        "status": "invalid_component_value",
                        "reason": f"negative_{source}",
                        "S_norm": None,
                        "weights": weights,
                        "components": None,
                        "legacy_raw_proxy": raw_proxy,
                    },
                    {},
                    {"S_norm": "invalid_component_value"},
                )
            components["N_H_outer_eval"] = max(0.0, float(value))
            sources["N_H_outer_eval"] = str(source)
            for key in ("N_grad", "N_metric", "N_H_refit_eval"):
                sources[key] = "method_zero"
        elif alg in adaptive_ids:
            selected_count, selected_source = _selected_operator_count(row)
            if selected_count is None:
                return (
                    {
                        "schema": NORMALIZED_MEASUREMENT_WORK_SCHEMA,
                        "status": "missing_component_breakdown",
                        "reason": "missing_selected_operator_count",
                        "S_norm": None,
                        "weights": weights,
                        "components": None,
                        "legacy_raw_proxy": raw_proxy,
                    },
                    {},
                    {"S_norm": "missing_component_breakdown"},
                )
            selected_raw = _num(row.get("selected_operator_count"))
            if selected_raw is not None and float(selected_raw) < 0.0:
                return (
                    {
                        "schema": NORMALIZED_MEASUREMENT_WORK_SCHEMA,
                        "status": "invalid_component_value",
                        "reason": "negative_selected_operator_count",
                        "S_norm": None,
                        "weights": weights,
                        "components": None,
                        "legacy_raw_proxy": raw_proxy,
                    },
                    {},
                    {"S_norm": "invalid_component_value"},
                )
            grad, grad_source = _first_num(row, ("gradient_operator_probe_count_proxy",))
            if grad is None:
                return (
                    {
                        "schema": NORMALIZED_MEASUREMENT_WORK_SCHEMA,
                        "status": "missing_component_breakdown",
                        "reason": "missing_gradient_operator_probe_count_proxy",
                        "S_norm": None,
                        "weights": weights,
                        "components": None,
                        "legacy_raw_proxy": raw_proxy,
                    },
                    {},
                    {"S_norm": "missing_component_breakdown"},
                )
            if float(grad) < 0.0:
                return (
                    {
                        "schema": NORMALIZED_MEASUREMENT_WORK_SCHEMA,
                        "status": "invalid_component_value",
                        "reason": f"negative_{grad_source}",
                        "S_norm": None,
                        "weights": weights,
                        "components": None,
                        "legacy_raw_proxy": raw_proxy,
                    },
                    {},
                    {"S_norm": "invalid_component_value"},
                )
            energy_eval, energy_source = _first_num(row, ("energy_eval_count_proxy", "nfev"))
            if energy_eval is None:
                return (
                    {
                        "schema": NORMALIZED_MEASUREMENT_WORK_SCHEMA,
                        "status": "missing_component_breakdown",
                        "reason": "missing_energy_eval_count_proxy",
                        "S_norm": None,
                        "weights": weights,
                        "components": None,
                        "legacy_raw_proxy": raw_proxy,
                    },
                    {},
                    {"S_norm": "missing_component_breakdown"},
                )
            if float(energy_eval) < 0.0:
                return (
                    {
                        "schema": NORMALIZED_MEASUREMENT_WORK_SCHEMA,
                        "status": "invalid_component_value",
                        "reason": f"negative_{energy_source}",
                        "S_norm": None,
                        "weights": weights,
                        "components": None,
                        "legacy_raw_proxy": raw_proxy,
                    },
                    {},
                    {"S_norm": "invalid_component_value"},
                )
            components["N_grad"] = max(0.0, float(grad))
            sources["N_grad"] = str(grad_source)
            metric = 0.0
            metric_source = "method_zero"
            metric_value, metric_source_found = _first_num(row, ("metric_operator_probe_count_proxy",))
            if alg == "static_qiskit_adapt_vqe":
                if metric_value is not None:
                    if float(metric_value) < 0.0:
                        return (
                            {
                                "schema": NORMALIZED_MEASUREMENT_WORK_SCHEMA,
                                "status": "invalid_component_value",
                                "reason": f"negative_{metric_source_found}",
                                "S_norm": None,
                                "weights": weights,
                                "components": None,
                                "legacy_raw_proxy": raw_proxy,
                            },
                            {},
                            {"S_norm": "invalid_component_value"},
                        )
                    metric = max(0.0, float(metric_value))
                    metric_source = str(metric_source_found) if metric > 0.0 else "method_zero"
            else:
                if metric_value is None:
                    return (
                        {
                            "schema": NORMALIZED_MEASUREMENT_WORK_SCHEMA,
                            "status": "missing_component_breakdown",
                            "reason": "missing_metric_operator_probe_count_proxy",
                            "S_norm": None,
                            "weights": weights,
                            "components": None,
                            "legacy_raw_proxy": raw_proxy,
                        },
                        {},
                        {"S_norm": "missing_component_breakdown"},
                    )
                if float(metric_value) < 0.0:
                    return (
                        {
                            "schema": NORMALIZED_MEASUREMENT_WORK_SCHEMA,
                            "status": "invalid_component_value",
                            "reason": f"negative_{metric_source_found}",
                            "S_norm": None,
                            "weights": weights,
                            "components": None,
                            "legacy_raw_proxy": raw_proxy,
                        },
                        {},
                        {"S_norm": "invalid_component_value"},
                    )
                metric = max(0.0, float(metric_value))
                metric_source = str(metric_source_found)
            components["N_metric"] = metric
            sources["N_metric"] = metric_source
            if int(selected_count) > 0:
                components["N_H_refit_eval"] = max(0.0, float(energy_eval))
                sources["N_H_refit_eval"] = str(energy_source)
                components["N_H_outer_eval"] = 0.0
                sources["N_H_outer_eval"] = "adaptive_refit_partition"
            else:
                components["N_H_outer_eval"] = max(0.0, float(energy_eval))
                sources["N_H_outer_eval"] = str(energy_source)
                components["N_H_refit_eval"] = 0.0
                sources["N_H_refit_eval"] = "no_selected_operators"
            sources["selected_operator_count"] = str(selected_source)
        else:
            return (
                {
                    "schema": NORMALIZED_MEASUREMENT_WORK_SCHEMA,
                    "status": "missing_component_breakdown",
                    "reason": "unsupported_algorithm_without_explicit_components",
                    "S_norm": None,
                    "weights": weights,
                    "components": None,
                    "legacy_raw_proxy": raw_proxy,
                },
                {},
                {"S_norm": "missing_component_breakdown"},
            )
    s_norm = (
        weights["s_H_outer"] * components["N_H_outer_eval"]
        + weights["s_g"] * components["N_grad"]
        + weights["s_F"] * components["N_metric"]
        + weights["s_H_refit"] * components["N_H_refit_eval"]
    )
    row_updates = {
        "S_norm": float(s_norm),
        "S_norm_N_H_outer_eval": float(components["N_H_outer_eval"]),
        "S_norm_N_grad": float(components["N_grad"]),
        "S_norm_N_metric": float(components["N_metric"]),
        "S_norm_N_H_refit_eval": float(components["N_H_refit_eval"]),
        "S_norm_N_H_eval": float(components["N_H_outer_eval"]),
        "S_norm_N_refit_eval": float(components["N_H_refit_eval"]),
        "S_norm_N_other_quantum": 0.0,
    }
    return (
        {
            "schema": NORMALIZED_MEASUREMENT_WORK_SCHEMA,
            "status": "ok",
            "S_norm": float(s_norm),
            "weights": weights,
            "components": {key: float(value) for key, value in components.items()},
            "component_sources": sources,
            "N_other_quantum": 0.0,
            "N_other_quantum_source": other_quantum_source,
            "component_splits": _measurement_work_splits(row, components),
            "event_count_convention": "fresh_measurement_bearing_calls_when_split_telemetry_absent",
            "legacy_raw_proxy": raw_proxy,
            "unit": "normalized_estimator_or_probe_count_not_physical_shots",
        },
        row_updates,
        {"S_norm": "ok"},
    )


def _spec_by_case_id(family: str, case_id: str, profile: str | None = None) -> HamiltonianBenchmarkSpec:
    family_key = str(family).strip()
    case_key = str(case_id).strip()
    return table_i_canonical_spec_by_case_id(family_key, case_key, profile=profile)


def _resolve_context(spec: HamiltonianBenchmarkSpec):  # noqa: ANN001
    # The HEA runner's resolver is a benchmark-local thin wrapper around
    # ProblemRequest.from_namespace(resolve_problem_context(...)). Reusing it
    # avoids duplicating CLI default handling here.
    return _hea_resolve_context(spec)


def _normalize_state(psi: np.ndarray) -> np.ndarray:
    arr = np.asarray(psi, dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        raise ValueError("state has zero norm")
    return arr / norm


def _statevector_from_re_im(raw: Any) -> np.ndarray | None:
    if not isinstance(raw, list) or not raw:
        return None
    out: list[complex] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None
        try:
            out.append(complex(float(item[0]), float(item[1])))
        except (TypeError, ValueError):
            return None
    return _normalize_state(np.asarray(out, dtype=complex).reshape(-1))


def _to_ixyz(label_exyz: str) -> str:
    return (
        str(label_exyz)
        .lower()
        .replace("e", "I")
        .replace("x", "X")
        .replace("y", "Y")
        .replace("z", "Z")
    )


def _append_pauli_rotation(circuit: Any, label_exyz: str, angle: float) -> None:
    """Append a benchmark-local Pauli-rotation synthesis to a Qiskit circuit.

    Repo Pauli labels are ordered left-to-right as q_(n-1) ... q_0.  Qiskit
    qubit 0 is therefore the rightmost label character.  The angle is used only
    to keep a nontrivial RZ in the compiled circuit; resource metrics do not
    depend on its value.
    """
    label = str(label_exyz).lower()
    nq = int(circuit.num_qubits)
    if len(label) != nq:
        raise NotReconstructable(
            "not_reconstructable_qiskit_compile_label_width",
            f"Pauli label {label!r} has width {len(label)}, expected {nq}",
        )
    active: list[tuple[int, str]] = []
    for pos, ch in enumerate(label):
        if ch == "e":
            continue
        if ch not in {"x", "y", "z"}:
            raise NotReconstructable(
                "not_reconstructable_qiskit_compile_bad_label",
                f"Unsupported Pauli character {ch!r} in {label!r}",
            )
        active.append((nq - 1 - pos, ch))
    if not active:
        return

    for qubit, ch in active:
        if ch == "x":
            circuit.h(qubit)
        elif ch == "y":
            circuit.sdg(qubit)
            circuit.h(qubit)

    target = active[-1][0]
    controls = [qubit for qubit, _ in active[:-1]]
    for qubit in controls:
        circuit.cx(qubit, target)
    circuit.rz(float(angle) if abs(float(angle)) > 1e-12 else 1.0, target)
    for qubit in reversed(controls):
        circuit.cx(qubit, target)

    for qubit, ch in reversed(active):
        if ch == "x":
            circuit.h(qubit)
        elif ch == "y":
            circuit.h(qubit)
            circuit.s(qubit)


def _qiskit_circuit_stats(circuit: Any) -> dict[str, Any]:
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
    return {
        "compiled_depth_total": depth,
        "compiled_depth_2q_total": depth_2q,
        "compiled_count_2q_total": int(count_2q),
        "compiled_op_counts": op_counts,
    }


def _selected_pauli_label_groups(
    row: Mapping[str, Any],
    *,
    selected: Sequence[Any] | None = None,
) -> list[list[str]]:
    raw_groups = row.get("selected_operator_pauli_labels_exyz")
    if isinstance(raw_groups, list):
        groups: list[list[str]] = []
        for group in raw_groups:
            if isinstance(group, str):
                groups.append([group])
            elif isinstance(group, list):
                groups.append([str(x) for x in group])
            else:
                raise NotReconstructable(
                    "not_reconstructable_qiskit_compile_bad_pauli_groups",
                    "selected_operator_pauli_labels_exyz contains a non-list group",
                )
        return groups

    if selected is not None:
        return [[str(label) for label in tuple(getattr(candidate, "pauli_labels_exyz", ()) or ())] for candidate in selected]

    pool_labels = row.get("pool_labels")
    pool_map = row.get("pool_pauli_labels_exyz")
    if isinstance(pool_labels, list) and isinstance(pool_map, Mapping):
        groups = []
        for label in pool_labels:
            labels = pool_map.get(str(label))
            if isinstance(labels, list):
                groups.append([str(x) for x in labels])
        if len(groups) == len(pool_labels):
            return groups

    raise NotReconstructable(
        "not_reconstructable_qiskit_compile_missing_pauli_groups",
        "missing selected_operator_pauli_labels_exyz or pool label map",
    )


def _qiskit_compile_selected_pauli_groups(
    context: Any,
    row: Mapping[str, Any],
    *,
    selected: Sequence[Any] | None = None,
    source_kind: str = "qiskit_compiled_final_ansatz_circuit",
) -> tuple[dict[str, Any], dict[str, float], dict[str, str]]:
    row_updates: dict[str, float] = {}
    statuses: dict[str, str] = {}
    groups = _selected_pauli_label_groups(row, selected=selected)
    theta = np.asarray(row.get("theta") or (), dtype=float).reshape(-1)
    psi_ref = _normalize_state(np.asarray(context.reference_state.build_state(), dtype=complex).reshape(-1))

    if selected is not None:
        selected_modes = [
            str(getattr(candidate, "execution_mode", "termwise_product") or "termwise_product")
            for candidate in selected
        ]
        try:
            compiled = compile_table_i_ansatz_terms(
                ops=tuple(
                    AnsatzTerm(
                        label=str(getattr(candidate, "label")),
                        polynomial=getattr(candidate, "polynomial"),
                        execution_mode=mode,
                    )
                    for candidate, mode in zip(selected, selected_modes, strict=True)
                ),
                num_qubits=int(context.layout.total_qubits),
                reference_state=psi_ref,
                source_kind=str(source_kind),
            )
            for key in ("compiled_depth_total", "compiled_depth_2q_total", "compiled_count_2q_total"):
                if compiled.get(key) is not None:
                    row_updates[key] = float(compiled[key])
                    statuses[key] = "ok"
            row_updates["compiled_circuit_stats_status"] = "ok"  # type: ignore[assignment]
            row_updates["compiled_resource_source_kind"] = str(source_kind)  # type: ignore[assignment]
            row_updates["first_hit_cost_source_kind"] = str(source_kind)  # type: ignore[assignment]
            row_updates["compiled_resource_qiskit_validated"] = True  # type: ignore[assignment]
            return (
                {
                    "status": "ok",
                    "compiled_depth_total": compiled.get("compiled_depth_total"),
                    "compiled_depth_2q_total": compiled.get("compiled_depth_2q_total"),
                    "compiled_count_2q_total": compiled.get("compiled_count_2q_total"),
                    "compiled_op_counts": compiled.get("compiled_op_counts"),
                    "compiled_basis_gates": compiled.get("compiled_basis_gates"),
                    "depth_2q_semantics": compiled.get("compiled_depth_2q_semantics") or compiled.get("depth_2q_semantics"),
                    "grouped_exact_synthesis_id": compiled.get("grouped_exact_synthesis_id"),
                    "generator_coefficients_sha256": compiled.get("generator_coefficients_sha256"),
                    "operator_synthesis": compiled.get("operator_synthesis"),
                    "synthesis": "table_i_structural_ansatz_compiler",
                    "source_kind": str(source_kind),
                },
                row_updates,
                statuses,
            )
        except TableICompileUnavailable as exc:
            statuses["compiled_depth_2q_total"] = exc.status
            if "grouped_exact" in selected_modes:
                raise NotReconstructable(
                    "not_reconstructable_grouped_exact_qiskit_compile_failed",
                    f"{exc.status}: {exc.reason}",
                ) from exc
        except Exception as exc:
            statuses["compiled_depth_2q_total"] = f"table_i_structural_compile_failed:{type(exc).__name__}"
            if "grouped_exact" in selected_modes:
                raise NotReconstructable(
                    "not_reconstructable_grouped_exact_qiskit_compile_failed",
                    f"{type(exc).__name__}: {exc}",
                ) from exc

    raw_modes = row.get("selected_operator_execution_modes")
    if selected is None and isinstance(raw_modes, list) and "grouped_exact" in {
        str(mode).strip().lower() for mode in raw_modes
    }:
        raise NotReconstructable(
            "not_reconstructable_grouped_exact_coefficients_missing",
            "grouped_exact Qiskit reconstruction requires coefficient-bearing selected generators",
        )

    components = import_qiskit_adaptvqe_components()
    circuit = build_reference_state_circuit(
        psi_ref,
        num_qubits=int(context.layout.total_qubits),
        quantum_circuit_cls=components.QuantumCircuit,
    )
    for idx, labels in enumerate(groups):
        angle = float(theta[idx]) if idx < int(theta.size) else 1.0
        for label in labels:
            if str(label).lower() == "e" * int(context.layout.total_qubits):
                continue
            _append_pauli_rotation(circuit, str(label), angle)
    try:
        from qiskit import transpile

        compiled = transpile(
            circuit.decompose(reps=10),
            basis_gates=list(_COMPILED_BASIS_GATES),
            optimization_level=0,
        )
    except Exception as exc:
        raise NotReconstructable("not_reconstructable_qiskit_compile_failed", str(exc)) from exc

    stats = _qiskit_circuit_stats(compiled)
    for key in ("compiled_depth_total", "compiled_depth_2q_total", "compiled_count_2q_total"):
        if stats.get(key) is not None:
            row_updates[key] = float(stats[key])
            statuses[key] = "ok"
    return (
        {
            "status": "ok",
            "compiled_depth_total": stats.get("compiled_depth_total"),
            "compiled_depth_2q_total": stats.get("compiled_depth_2q_total"),
            "compiled_count_2q_total": stats.get("compiled_count_2q_total"),
            "compiled_op_counts": stats.get("compiled_op_counts"),
            "compiled_basis_gates": list(_COMPILED_BASIS_GATES),
            "depth_2q_semantics": "qiskit_compiled_two_qubit_layer_depth_termwise_pauli_rotation_synthesis",
            "synthesis": "benchmark_local_termwise_pauli_rotation_circuit",
        },
        row_updates,
        statuses,
    )


def _exact_energy_for_context(context: Any) -> float | None:
    try:
        return float(context.exact_target.resolve_energy(ai_log=None))
    except TypeError:
        try:
            return float(context.exact_target.resolve_energy())
        except Exception:
            return None
    except Exception:
        return None


def _same_cutoff_unreconstructable_metrics(
    context: Any,
    energy: float | None,
    infidelity_status: str,
) -> tuple[dict[str, Any], dict[str, float], dict[str, str]]:
    row_updates: dict[str, float] = {}
    statuses: dict[str, str] = {"infidelity_same": infidelity_status}
    exact_energy = _exact_energy_for_context(context)
    if energy is not None and exact_energy is not None:
        row_updates["abs_delta_e_same_cutoff"] = abs(float(energy) - float(exact_energy))
        statuses["abs_delta_e_same_cutoff"] = "ok"
    return (
        {
            "status": infidelity_status,
            "energy": energy,
            "exact_energy": exact_energy,
            "abs_delta_e_same_cutoff": row_updates.get("abs_delta_e_same_cutoff"),
            "infidelity_same": None,
        },
        row_updates,
        statuses,
    )


def _same_cutoff_metrics(context: Any, psi_final: np.ndarray, energy: float | None) -> tuple[dict[str, Any], dict[str, float], dict[str, str]]:
    row_updates: dict[str, float] = {}
    statuses: dict[str, str] = {}
    exact_energy = _exact_energy_for_context(context)

    if energy is not None and exact_energy is not None:
        row_updates["abs_delta_e_same_cutoff"] = abs(float(energy) - float(exact_energy))
        statuses["abs_delta_e_same_cutoff"] = "ok"

    resolution = resolve_exact_reference_state_for_problem(
        context.hamiltonian,
        resolved_problem=context,
        ai_log=None,
        max_dense_dim=8192,
    )
    if not bool(resolution.available) or resolution.state is None:
        statuses["infidelity_same"] = str(resolution.skip_reason or resolution.source or "exact_state_unavailable")
        return (
            {
                "status": statuses["infidelity_same"],
                "energy": energy,
                "exact_energy": exact_energy,
                "abs_delta_e_same_cutoff": row_updates.get("abs_delta_e_same_cutoff"),
                "infidelity_same": None,
                "exact_state_source": resolution.source,
                "state_dimension": resolution.state_dimension,
            },
            row_updates,
            statuses,
        )

    psi_exact = _normalize_state(np.asarray(resolution.state, dtype=complex).reshape(-1))
    psi = _normalize_state(np.asarray(psi_final, dtype=complex).reshape(-1))
    if psi_exact.size != psi.size:
        statuses["infidelity_same"] = "dimension_mismatch"
        infidelity = None
    else:
        fidelity = float(abs(np.vdot(psi_exact, psi)) ** 2)
        infidelity = float(max(0.0, min(1.0, 1.0 - fidelity)))
        row_updates["infidelity_same"] = infidelity
        statuses["infidelity_same"] = "ok"
    return (
        {
            "status": statuses.get("infidelity_same", "ok"),
            "energy": energy,
            "exact_energy": exact_energy,
            "abs_delta_e_same_cutoff": row_updates.get("abs_delta_e_same_cutoff"),
            "infidelity_same": infidelity,
            "exact_state_source": resolution.source,
            "state_dimension": int(psi.size),
        },
        row_updates,
        statuses,
    )


def _replace_n_ph_arg(args: Sequence[str], ref_nph: int) -> tuple[str, ...]:
    out = [str(x) for x in args]
    for idx, token in enumerate(out):
        if token == "--n-ph-max" and idx + 1 < len(out):
            out[idx + 1] = str(int(ref_nph))
            return tuple(out)
    return tuple(out + ["--n-ph-max", str(int(ref_nph))])


def _block_by_name_or_kind(layout: Any, block: Any) -> Any | None:
    blocks = tuple(getattr(layout, "blocks", ()) or ())
    name = str(getattr(block, "name", ""))
    kind = str(getattr(block, "kind", ""))
    for candidate in blocks:
        if str(getattr(candidate, "name", "")) == name:
            return candidate
    same_kind = [candidate for candidate in blocks if str(getattr(candidate, "kind", "")) == kind]
    if len(same_kind) == 1:
        return same_kind[0]
    return None


def _single_boson_block(layout: Any) -> Any | None:
    blocks = [block for block in tuple(getattr(layout, "blocks", ()) or ()) if str(getattr(block, "kind", "")) == "boson"]
    return blocks[0] if len(blocks) == 1 else None


def _local_n_ph_from_context(context: Any) -> int | None:
    raw = getattr(getattr(context, "request", None), "n_ph_max", None)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _embed_state_in_reference_cutoff(psi: np.ndarray, context: Any, ref_context: Any) -> np.ndarray:
    psi_local = _normalize_state(np.asarray(psi, dtype=complex).reshape(-1))
    local_qubits = int(context.layout.total_qubits)
    ref_qubits = int(ref_context.layout.total_qubits)
    if psi_local.size != (1 << local_qubits):
        raise NotReconstructable(
            "not_reconstructable_dimension_mismatch",
            f"state dimension {psi_local.size} does not match local qubit count {local_qubits}",
        )
    if local_qubits == ref_qubits:
        return psi_local

    local_boson = _single_boson_block(context.layout)
    ref_boson = _single_boson_block(ref_context.layout)
    local_nph = _local_n_ph_from_context(context)
    ref_nph = _local_n_ph_from_context(ref_context)
    encoding = str(getattr(context.layout, "boson_encoding", None) or getattr(context.request, "boson_encoding", "binary"))
    ref_encoding = str(getattr(ref_context.layout, "boson_encoding", None) or getattr(ref_context.request, "boson_encoding", "binary"))
    if local_boson is None or ref_boson is None or local_nph is None or ref_nph is None or encoding != ref_encoding:
        raise NotReconstructable(
            "not_reconstructable_cutoff_embedding_layout",
            "cannot identify compatible single boson blocks for cutoff embedding",
        )
    if encoding != "binary":
        raise NotReconstructable(
            "not_reconstructable_cutoff_embedding_encoding",
            f"cutoff embedding is implemented for binary boson encoding, not {encoding!r}",
        )
    local_bps = int(boson_qubits_per_site(local_nph, encoding))
    ref_bps = int(boson_qubits_per_site(ref_nph, ref_encoding))
    local_boson_width = int(local_boson.stop_qubit) - int(local_boson.start_qubit)
    ref_boson_width = int(ref_boson.stop_qubit) - int(ref_boson.start_qubit)
    if local_bps <= 0 or ref_bps <= 0 or local_boson_width % local_bps or ref_boson_width % ref_bps:
        raise NotReconstructable("not_reconstructable_cutoff_embedding_layout", "boson block widths do not divide into site registers")
    local_sites = local_boson_width // local_bps
    ref_sites = ref_boson_width // ref_bps
    if local_sites != ref_sites:
        raise NotReconstructable("not_reconstructable_cutoff_embedding_layout", "local/reference boson site counts differ")

    non_boson_maps: list[tuple[Any, Any]] = []
    for block in tuple(getattr(context.layout, "blocks", ()) or ()):
        if str(getattr(block, "kind", "")) == "boson":
            continue
        ref_block = _block_by_name_or_kind(ref_context.layout, block)
        if ref_block is None:
            raise NotReconstructable("not_reconstructable_cutoff_embedding_layout", f"missing reference block for {getattr(block, 'name', '?')}")
        if (int(block.stop_qubit) - int(block.start_qubit)) != (int(ref_block.stop_qubit) - int(ref_block.start_qubit)):
            raise NotReconstructable("not_reconstructable_cutoff_embedding_layout", "non-boson block widths differ")
        non_boson_maps.append((block, ref_block))

    psi_ref = np.zeros(1 << ref_qubits, dtype=complex)
    for local_index, amp in enumerate(psi_local):
        if abs(amp) <= 0.0:
            continue
        ref_index = 0
        for block, ref_block in non_boson_maps:
            for offset in range(int(block.stop_qubit) - int(block.start_qubit)):
                if (local_index >> (int(block.start_qubit) + offset)) & 1:
                    ref_index |= 1 << (int(ref_block.start_qubit) + offset)
        for site in range(local_sites):
            local_value = 0
            for bit in range(local_bps):
                if (local_index >> (int(local_boson.start_qubit) + site * local_bps + bit)) & 1:
                    local_value |= 1 << bit
            if local_value > ref_nph:
                raise NotReconstructable("not_reconstructable_cutoff_embedding_truncation", "local boson occupation exceeds reference cutoff")
            for bit in range(ref_bps):
                if (local_value >> bit) & 1:
                    ref_index |= 1 << (int(ref_boson.start_qubit) + site * ref_bps + bit)
        psi_ref[ref_index] += amp
    return _normalize_state(psi_ref)


def _reference_context_for_spec(spec: HamiltonianBenchmarkSpec, ref_nph: int) -> Any:
    return _hea_resolve_context(replace(spec, base_pipeline_args=_replace_n_ph_arg(spec.base_pipeline_args, int(ref_nph))))


def _reference_cutoff_metrics(
    spec: HamiltonianBenchmarkSpec,
    context: Any,
    psi_final: np.ndarray | None,
) -> tuple[dict[str, Any], dict[str, float], dict[str, str]]:
    row_updates: dict[str, float] = {}
    statuses: dict[str, str] = {}
    exact_ref, ref_nph, error = _reference_cutoff_energy_for_spec(spec)
    if ref_nph is None:
        statuses["abs_delta_e_reference"] = "not_applicable"
        statuses["infidelity_4"] = "not_applicable"
        return (
            {
                "status": "not_applicable",
                "exact_reference_n_ph_max": None,
                "exact_reference_energy": None,
                "abs_delta_e_reference": None,
                "cutoff_abs_delta_e": None,
                "infidelity_reference": None,
                "infidelity_status": "not_applicable",
            },
            row_updates,
            statuses,
        )
    if error is not None or exact_ref is None:
        status = error or "missing_reference_energy"
        statuses["abs_delta_e_reference"] = status
        statuses["infidelity_4"] = "not_reconstructable_cutoff_embedding_not_implemented_v1"
        return (
            {
                "status": status,
                "exact_reference_n_ph_max": ref_nph,
                "exact_reference_energy": exact_ref,
                "abs_delta_e_reference": None,
                "cutoff_abs_delta_e": None,
                "infidelity_reference": None,
                "infidelity_status": statuses["infidelity_4"],
            },
            row_updates,
            statuses,
        )
    if psi_final is None:
        status = "not_reconstructable_missing_final_state"
        statuses["abs_delta_e_reference"] = status
        statuses["infidelity_4"] = "not_reconstructable_missing_final_state"
        return (
            {
                "status": status,
                "exact_reference_n_ph_max": ref_nph,
                "exact_reference_energy": exact_ref,
                "abs_delta_e_reference": None,
                "cutoff_abs_delta_e": None,
                "infidelity_reference": None,
                "infidelity_status": statuses["infidelity_4"],
            },
            row_updates,
            statuses,
        )
    try:
        ref_context = _reference_context_for_spec(spec, int(ref_nph))
        psi_embedded = _embed_state_in_reference_cutoff(psi_final, context, ref_context)
        embedded_energy = float(expval_pauli_polynomial_one_apply(psi_embedded, ref_context.hamiltonian))
    except NotReconstructable as exc:
        statuses["abs_delta_e_reference"] = exc.status
        statuses["infidelity_4"] = exc.status
        return (
            {
                "status": exc.status,
                "exact_reference_n_ph_max": ref_nph,
                "exact_reference_energy": exact_ref,
                "embedded_energy_reference": None,
                "abs_delta_e_reference": None,
                "cutoff_abs_delta_e": None,
                "infidelity_reference": None,
                "infidelity_status": exc.status,
                "reason": exc.reason,
            },
            row_updates,
            statuses,
        )
    delta = abs(float(embedded_energy) - float(exact_ref))
    row_updates["abs_delta_e_reference"] = float(delta)
    statuses["abs_delta_e_reference"] = "ok"
    infidelity_reference: float | None = None
    try:
        resolution = resolve_exact_reference_state_for_problem(
            ref_context.hamiltonian,
            resolved_problem=ref_context,
            ai_log=None,
            max_dense_dim=8192,
        )
        if bool(resolution.available) and resolution.state is not None:
            psi_exact_ref = _normalize_state(np.asarray(resolution.state, dtype=complex).reshape(-1))
            if psi_exact_ref.size == psi_embedded.size:
                fidelity = float(abs(np.vdot(psi_exact_ref, _normalize_state(psi_embedded))) ** 2)
                infidelity_reference = float(max(0.0, min(1.0, 1.0 - fidelity)))
                row_updates["infidelity_4"] = infidelity_reference
                statuses["infidelity_4"] = "ok"
            else:
                statuses["infidelity_4"] = "reference_state_dimension_mismatch"
        else:
            statuses["infidelity_4"] = str(resolution.skip_reason or resolution.source or "exact_state_unavailable")
    except Exception as exc:
        statuses["infidelity_4"] = f"reference_state_failed:{type(exc).__name__}"
    return (
        {
            "status": "ok",
            "exact_reference_n_ph_max": int(ref_nph),
            "exact_reference_energy": float(exact_ref),
            "embedded_energy_reference": float(embedded_energy),
            "abs_delta_e_reference": float(delta),
            "cutoff_abs_delta_e": float(delta),
            "infidelity_reference": infidelity_reference,
            "infidelity_status": statuses["infidelity_4"],
        },
        row_updates,
        statuses,
    )


def _build_adapt_variant_state(algorithm_id: str, context: Any, row: Mapping[str, Any]) -> np.ndarray:
    config = _get_config(algorithm_id)
    pool = (
        build_pairwise_qubit_excitation_pool(int(context.layout.total_qubits))
        if _pool_name_for_config(config) == "qubit_excitation_singles_doubles_pool"
        else build_full_meta_candidate_pool(context)
    )
    by_label = {str(candidate.label): candidate for candidate in pool}
    if "selected_operators" not in row or row.get("selected_operators") is None or "theta" not in row or row.get("theta") is None:
        raise NotReconstructable(
            "not_reconstructable_missing_ansatz_artifacts",
            "missing selected_operators or theta fields",
        )
    selected_labels = [str(label) for label in (row.get("selected_operators") or ())]
    selected = []
    for label in selected_labels:
        if label not in by_label:
            raise NotReconstructable(
                "not_reconstructable_selected_operator_not_found",
                f"selected operator {label!r} not found in rebuilt pool",
            )
        selected.append(by_label[label])
    theta = np.asarray(row.get("theta") or (), dtype=float).reshape(-1)
    if theta.size != len(selected):
        raise NotReconstructable(
            "not_reconstructable_parameter_count_mismatch",
            f"theta length {theta.size} does not match selected operator count {len(selected)}",
        )
    psi_ref = _normalize_state(np.asarray(context.reference_state.build_state(), dtype=complex).reshape(-1))
    return _prepare_selected_state(selected=selected, theta=theta, psi_ref=psi_ref, pauli_action_cache={})


def _build_family_informed_state(context: Any, row: Mapping[str, Any]) -> np.ndarray:
    pool = build_full_meta_candidate_pool(context)
    by_label = {str(candidate.label): candidate for candidate in pool}
    if "pool_labels" not in row or row.get("pool_labels") is None or "theta" not in row or row.get("theta") is None:
        raise NotReconstructable(
            "not_reconstructable_missing_ansatz_artifacts",
            "missing pool_labels or theta fields",
        )
    selected_labels = [str(label) for label in (row.get("pool_labels") or ())]
    selected = []
    for label in selected_labels:
        if label not in by_label:
            raise NotReconstructable(
                "not_reconstructable_selected_operator_not_found",
                f"selected operator {label!r} not found in rebuilt full_meta pool",
            )
        selected.append(by_label[label])
    theta = np.asarray(row.get("theta") or (), dtype=float).reshape(-1)
    if theta.size != len(selected):
        raise NotReconstructable(
            "not_reconstructable_parameter_count_mismatch",
            f"theta length {theta.size} does not match selected operator count {len(selected)}",
        )
    psi_ref = _normalize_state(np.asarray(context.reference_state.build_state(), dtype=complex).reshape(-1))
    return _prepare_selected_state(selected=selected, theta=theta, psi_ref=psi_ref, pauli_action_cache={})


def _build_hea_state_and_depth(context: Any, row: Mapping[str, Any]) -> tuple[np.ndarray, int | None, str]:
    if "theta" not in row or row.get("theta") is None:
        raise NotReconstructable("not_reconstructable_missing_ansatz_artifacts", "missing theta field")
    theta = np.asarray(row.get("theta") or (), dtype=float).reshape(-1)
    reps = int(row.get("vqe_reps") or 2)
    psi_ref = _normalize_state(np.asarray(context.reference_state.build_state(), dtype=complex).reshape(-1))
    ansatz = build_qiskit_hea_ansatz(num_qubits=int(context.layout.total_qubits), reps=reps)
    if theta.size != len(ansatz.parameters):
        raise NotReconstructable(
            "not_reconstructable_parameter_count_mismatch",
            f"theta length {theta.size} does not match HEA parameter count {len(ansatz.parameters)}",
        )
    psi = ansatz.prepare_state(theta, psi_ref)
    try:
        from qiskit import transpile
        from pipelines.qiskit_backend_tools import safe_two_qubit_depth

        assignments = {param: float(theta[idx]) for idx, param in enumerate(ansatz.parameters)}
        bound = ansatz.circuit.assign_parameters(assignments, inplace=False)
        compiled = transpile(bound.decompose(reps=10), basis_gates=["id", "x", "sx", "rx", "ry", "rz", "h", "s", "sdg", "cx", "cz"], optimization_level=0)
        return psi, int(safe_two_qubit_depth(compiled)), "qiskit_compiled_two_qubit_layer_depth"
    except Exception:
        return psi, None, "qiskit_two_qubit_depth_unavailable"


def _reconstruct_state_and_depth(
    algorithm_id: str,
    context: Any,
    row: Mapping[str, Any],
) -> tuple[np.ndarray | None, dict[str, Any], dict[str, float], dict[str, str]]:
    row_updates: dict[str, float] = {}
    statuses: dict[str, str] = {}
    if algorithm_id in _ADAPT_VARIANT_IDS:
        try:
            psi = _build_adapt_variant_state(algorithm_id, context, row)
        except NotReconstructable as exc:
            statuses["infidelity_same"] = exc.status
            statuses["compiled_depth_2q_total"] = "not_reconstructable"
            return None, {"status": "not_reconstructable", "compiled_depth_2q_total": None, "depth_2q_semantics": "none", "reason": exc.reason}, row_updates, statuses
        try:
            config = _get_config(algorithm_id)
            pool = (
                build_pairwise_qubit_excitation_pool(int(context.layout.total_qubits))
                if _pool_name_for_config(config) == "qubit_excitation_singles_doubles_pool"
                else build_full_meta_candidate_pool(context)
            )
            by_label = {str(candidate.label): candidate for candidate in pool}
            selected = [by_label[str(label)] for label in (row.get("selected_operators") or ())]
            depth_metric, depth_updates, depth_statuses = _qiskit_compile_selected_pauli_groups(
                context,
                row,
                selected=selected,
                source_kind="qiskit_compiled_final_ansatz_circuit",
            )
            row_updates.update(depth_updates)
            statuses.update(depth_statuses)
            return psi, depth_metric, row_updates, statuses
        except Exception as exc:
            statuses["compiled_depth_2q_total"] = (
                exc.status if isinstance(exc, NotReconstructable) else f"qiskit_compile_failed:{type(exc).__name__}"
            )
            return psi, {
                "status": statuses["compiled_depth_2q_total"],
                "compiled_depth_2q_total": None,
                "depth_2q_semantics": "none",
                "reason": str(exc),
            }, row_updates, statuses
    if algorithm_id == "static_family_informed_vqe":
        try:
            psi = _build_family_informed_state(context, row)
        except NotReconstructable as exc:
            statuses["infidelity_same"] = exc.status
            statuses["compiled_depth_2q_total"] = "not_reconstructable"
            return None, {"status": "not_reconstructable", "compiled_depth_2q_total": None, "depth_2q_semantics": "none", "reason": exc.reason}, row_updates, statuses
        try:
            pool = build_full_meta_candidate_pool(context)
            by_label = {str(candidate.label): candidate for candidate in pool}
            selected = [by_label[str(label)] for label in (row.get("pool_labels") or ())]
            depth_metric, depth_updates, depth_statuses = _qiskit_compile_selected_pauli_groups(
                context,
                row,
                selected=selected,
                source_kind="qiskit_compiled_terminal_only_fixed_ansatz",
            )
            row_updates.update(depth_updates)
            statuses.update(depth_statuses)
            return psi, depth_metric, row_updates, statuses
        except Exception as exc:
            statuses["compiled_depth_2q_total"] = (
                exc.status if isinstance(exc, NotReconstructable) else f"qiskit_compile_failed:{type(exc).__name__}"
            )
            return psi, {
                "status": statuses["compiled_depth_2q_total"],
                "compiled_depth_2q_total": None,
                "depth_2q_semantics": "none",
                "reason": str(exc),
            }, row_updates, statuses
    if algorithm_id == "static_qiskit_adapt_vqe":
        raw_depth = _num(row.get("compiled_depth_2q_total"))
        raw_status = str(row.get("compiled_circuit_stats_status") or "")
        semantics = str(row.get("compiled_depth_2q_semantics") or "qiskit_compiled_two_qubit_layer_depth_final_adapt_circuit")
        psi = _statevector_from_re_im(row.get("final_statevector_re_im"))
        if raw_status == "ok" and raw_depth is not None:
            row_updates["compiled_depth_2q_total"] = float(raw_depth)
            raw_count = _num(row.get("compiled_count_2q_total"))
            raw_depth_total = _num(row.get("compiled_depth_total"))
            if raw_count is not None:
                row_updates["compiled_count_2q_total"] = float(raw_count)
                statuses["compiled_count_2q_total"] = "ok"
            if raw_depth_total is not None:
                row_updates["compiled_depth_total"] = float(raw_depth_total)
                statuses["compiled_depth_total"] = "ok"
            statuses["compiled_depth_2q_total"] = "ok"
            return psi, {
                "status": "ok",
                "compiled_depth_2q_total": raw_depth,
                "depth_2q_semantics": semantics,
                "source": "raw_qiskit_adaptvqe_compiled_circuit_telemetry",
            }, row_updates, statuses
        statuses["compiled_depth_2q_total"] = "not_reconstructable_missing_qiskit_adaptvqe_depth_telemetry"
        statuses["infidelity_same"] = "not_reconstructable_missing_final_state"
        return None, {
            "status": statuses["compiled_depth_2q_total"],
            "compiled_depth_2q_total": None,
            "depth_2q_semantics": "none",
            "compiled_circuit_stats_status": raw_status,
        }, row_updates, statuses
    if algorithm_id == "static_hea_qiskit_vqe":
        try:
            psi, depth_2q, semantics = _build_hea_state_and_depth(context, row)
        except QiskitHeaUnavailable:
            statuses["compiled_depth_2q_total"] = "qiskit_unavailable"
            return None, {"status": "qiskit_unavailable", "compiled_depth_2q_total": None, "depth_2q_semantics": "none"}, row_updates, statuses
        except NotReconstructable as exc:
            statuses["infidelity_same"] = exc.status
            statuses["compiled_depth_2q_total"] = "not_reconstructable"
            return None, {"status": "not_reconstructable", "compiled_depth_2q_total": None, "depth_2q_semantics": "none", "reason": exc.reason}, row_updates, statuses
        if depth_2q is not None:
            row_updates["compiled_depth_2q_total"] = float(depth_2q)
            statuses["compiled_depth_2q_total"] = "ok"
            status = "ok"
        else:
            statuses["compiled_depth_2q_total"] = semantics
            status = semantics
        return psi, {"status": status, "compiled_depth_2q_total": depth_2q, "depth_2q_semantics": semantics}, row_updates, statuses
    statuses["infidelity_same"] = "not_reconstructable_missing_final_circuit_or_selected_pool_labels"
    statuses["compiled_depth_2q_total"] = "not_reconstructable"
    return None, {"status": "not_reconstructable", "compiled_depth_2q_total": None, "depth_2q_semantics": "none"}, row_updates, statuses


def _base_payload(record: Mapping[str, str], payload_path: Path | None = None) -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "record_id": record["record_id"],
        "family": record["family"],
        "case_id": record["case_id"],
        "algorithm_id": record["algorithm_id"],
        "suite_profile": record.get("suite_profile") or None,
        "status": "completed",
        "source_payload_path": None if payload_path is None else str(payload_path),
        "guardrails": {
            "uses_exact_for_decision": False,
            "exact_reference_usage": "post_hoc_reporting_only",
            "phase3_controller_called": False,
            "raw_payload_mutated": False,
        },
        "metrics": {},
        "row_updates": {},
        "metric_statuses": {},
    }


def enrich_record(
    *,
    record: Mapping[str, str],
    input_root: Path,
    output_dir: Path,
    suite_profile: str | None = None,
) -> dict[str, Any]:
    payload_path, payload = _read_payload(Path(input_root), record["record_id"])
    out = _base_payload(record, payload_path)
    if payload is None:
        out["status"] = "payload_missing"
        out["metrics"]["same_cutoff"] = {"status": "payload_missing"}
        out["metric_statuses"]["payload"] = "payload_missing"
    else:
        row = _result(payload)
        energy = _num(row.get("energy"))
        try:
            work_metric, work_updates, work_statuses = _normalized_measurement_work(
                algorithm_id=str(record["algorithm_id"]),
                row=row,
            )
        except Exception as exc:  # pragma: no cover - defensive reporting guard
            work_metric, work_updates, work_statuses = (
                {
                    "schema": NORMALIZED_MEASUREMENT_WORK_SCHEMA,
                    "status": "failed",
                    "reason": str(exc),
                    "S_norm": None,
                },
                {},
                {"S_norm": f"failed:{type(exc).__name__}"},
            )
        out["metrics"]["measurement_work"] = work_metric
        out["row_updates"].update(work_updates)
        out["metric_statuses"].update(work_statuses)
        raw_proxy = {
            "shots_total": _num(row.get("shots_total")),
            "shot_cost_proxy": _num(row.get("shot_cost_proxy")),
            "measurement_shots_proxy": _num(row.get("measurement_shots_proxy")),
            "shot_proxy": _num(row.get("shot_proxy")),
        }
        try:
            replay_ledger, replay_status = _table_i_event_ledger_from_comparator_row(
                algorithm_id=str(record["algorithm_id"]),
                row=row,
            )
            alg_row = dict(row)
            if replay_ledger is not None:
                alg_row["table_i_measurement_event_ledger"] = replay_ledger
            alg_metric, alg_updates, alg_statuses = algorithmic_measurement_work_from_row(
                row=alg_row,
                raw_proxy=raw_proxy,
            )
            alg_metric = dict(alg_metric)
            alg_metric["replay_status"] = replay_status
        except Exception as exc:  # pragma: no cover - defensive reporting guard
            alg_metric, alg_updates, alg_statuses = (
                {
                    "schema": ALGORITHMIC_MEASUREMENT_WORK_SCHEMA,
                    "status": "failed",
                    "reason": str(exc),
                    "S_alg": None,
                },
                {},
                {"S_alg": f"failed:{type(exc).__name__}"},
            )
        out["metrics"]["algorithmic_measurement_work"] = alg_metric
        out["row_updates"].update(alg_updates)
        out["metric_statuses"].update(alg_statuses)
        try:
            physical_metric, physical_updates, physical_statuses = physical_measurement_work_from_row(row=row)
        except Exception as exc:  # pragma: no cover - defensive reporting guard
            physical_metric, physical_updates, physical_statuses = (
                {
                    "schema": PHYSICAL_MEASUREMENT_WORK_SCHEMA,
                    "status": "failed",
                    "reason": str(exc),
                    "S_phys": None,
                    "S_l2": None,
                    "S_var": None,
                    "S_phys_var": None,
                },
                {},
                {
                    "S_phys": f"failed:{type(exc).__name__}",
                    "S_l2": f"failed:{type(exc).__name__}",
                    "S_var": f"failed:{type(exc).__name__}",
                    "S_phys_var": f"failed:{type(exc).__name__}",
                },
            )
        out["metrics"]["physical_measurement_work"] = physical_metric
        out["row_updates"].update(physical_updates)
        out["metric_statuses"].update(physical_statuses)
        try:
            grouped_metric, grouped_updates, grouped_statuses = grouped_measurement_proxy_from_explicit_row(row=row)
        except Exception as exc:  # pragma: no cover - defensive reporting guard
            grouped_metric, grouped_updates, grouped_statuses = (
                {
                    "schema": GROUPED_MEASUREMENT_PROXY_SCHEMA,
                    "status": "failed",
                    "reason": str(exc),
                    "S_grp_total": None,
                },
                {},
                {"S_grp": f"failed:{type(exc).__name__}"},
            )
        out["metrics"]["grouped_measurement_work"] = grouped_metric
        out["row_updates"].update(grouped_updates)
        out["metric_statuses"].update(grouped_statuses)
        try:
            profile = record.get("suite_profile") or suite_profile
            if profile:
                out["suite_profile"] = str(profile)
            spec = _spec_by_case_id(record["family"], record["case_id"], profile=profile)
            context = _resolve_context(spec)
            try:
                grouped_metric, grouped_updates, grouped_statuses = grouped_measurement_proxy_from_row_and_context(
                    algorithm_id=str(record["algorithm_id"]),
                    row=row,
                    context=context,
                )
            except Exception as exc:  # pragma: no cover - defensive reporting guard
                grouped_metric, grouped_updates, grouped_statuses = (
                    {
                        "schema": GROUPED_MEASUREMENT_PROXY_SCHEMA,
                        "status": "failed",
                        "reason": str(exc),
                        "S_grp_total": None,
                    },
                    {},
                    {"S_grp": f"failed:{type(exc).__name__}"},
                )
            out["metrics"]["grouped_measurement_work"] = grouped_metric
            out["row_updates"].update(grouped_updates)
            out["metric_statuses"].update(grouped_statuses)
            if str(grouped_statuses.get("S_grp") or "") == "ok":
                row_variance_metric = row.get("statevector_variance_metric")
                if not isinstance(row_variance_metric, Mapping):
                    row_variance_metric = row.get("grouped_statevector_variance_metric")
                prior_physical_metric = out["metrics"].get("physical_measurement_work")
                prior_s_var_ok = str(out["metric_statuses"].get("S_var") or "") == "ok" and _num(out["row_updates"].get("S_var")) is not None
                prior_s_var_metric = prior_physical_metric.get("S_var") if isinstance(prior_physical_metric, Mapping) else None
                physical_metric, physical_updates, physical_statuses = physical_measurement_work_from_grouped_replay(
                    grouped_metric=grouped_metric,
                    statevector_variance_metric=row_variance_metric if isinstance(row_variance_metric, Mapping) else None,
                )
                if prior_s_var_ok and str(physical_statuses.get("S_var") or "") != "ok":
                    # Grouped-L2 replay is allowed to replace S_l2, but it must
                    # not demote explicit fresh event-summed S_var components
                    # already promoted from the raw row.
                    physical_metric = dict(physical_metric)
                    if isinstance(prior_s_var_metric, Mapping):
                        physical_metric["S_var"] = prior_s_var_metric
                        physical_metric["S_phys_var"] = prior_s_var_metric
                    for key in ("S_var", "S_var_H_outer", "S_var_grad", "S_var_metric", "S_var_H_refit"):
                        value = _num(out["row_updates"].get(key))
                        if value is not None:
                            physical_updates[key] = float(value)
                    physical_statuses["S_var"] = "ok"
                    physical_statuses["S_phys_var"] = "ok"
                out["metrics"]["physical_measurement_work"] = physical_metric
                out["row_updates"].update(physical_updates)
                out["metric_statuses"].update(physical_statuses)
            psi_final, depth_metric, depth_updates, depth_statuses = _reconstruct_state_and_depth(
                str(record["algorithm_id"]), context, row
            )
            out["metrics"]["depth"] = depth_metric
            out["row_updates"].update(depth_updates)
            out["metric_statuses"].update(depth_statuses)
            if psi_final is None:
                same_metric, same_updates, same_statuses = _same_cutoff_unreconstructable_metrics(
                    context,
                    energy,
                    out["metric_statuses"].get("infidelity_same", "not_reconstructable"),
                )
                out["metrics"]["same_cutoff"] = same_metric
                out["row_updates"].update(same_updates)
                out["metric_statuses"].update(same_statuses)
            else:
                same_metric, same_updates, same_statuses = _same_cutoff_metrics(context, psi_final, energy)
                out["metrics"]["same_cutoff"] = same_metric
                out["row_updates"].update(same_updates)
                out["metric_statuses"].update(same_statuses)
            ref_metric, ref_updates, ref_statuses = _reference_cutoff_metrics(spec, context, psi_final)
            out["metrics"]["reference_cutoff"] = ref_metric
            out["row_updates"].update(ref_updates)
            out["metric_statuses"].update(ref_statuses)
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            out["status"] = "failed"
            out["failure_reason"] = str(exc)
            out["metric_statuses"]["failure"] = type(exc).__name__
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / ENRICHMENT_FILENAME).write_text(json.dumps(out, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return out


def run_batch(*, records_path: Path, input_root: Path, output_root: Path, suite_profile: str | None = None) -> dict[str, Any]:
    rows = _load_records(records_path)
    results = []
    for record in rows:
        result = enrich_record(
            record=record,
            input_root=input_root,
            output_dir=output_root / record["record_id"] / "result",
            suite_profile=suite_profile,
        )
        results.append(result)
    summary = {
        "schema": f"{SCHEMA_VERSION}_summary",
        "records_path": str(records_path),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "suite_profile_fallback": suite_profile,
        "record_count": len(rows),
        "status_counts": {status: sum(1 for result in results if result.get("status") == status) for status in sorted({str(result.get("status")) for result in results})},
        "results": [
            {
                "record_id": result.get("record_id"),
                "status": result.get("status"),
                "metric_statuses": result.get("metric_statuses", {}),
            }
            for result in results
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metric_enrichment_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Post-hoc enrich generic static Table-I benchmark metrics.")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-single", action="store_true", default=False)
    parser.add_argument("--record-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--suite-profile",
        default=None,
        help="Fallback Table-I suite profile for records that do not carry a suite_profile column.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = _load_records(args.records)
    if bool(args.run_single):
        if not args.record_id:
            raise ValueError("--record-id is required with --run-single")
        selected = [record for record in records if record["record_id"] == str(args.record_id)]
        if not selected:
            raise ValueError(f"record id not found: {args.record_id}")
        output_dir = args.output_dir
        if output_dir is None:
            if args.output_root is None:
                raise ValueError("--output-dir or --output-root is required with --run-single")
            output_dir = args.output_root / str(args.record_id) / "result"
        result = enrich_record(record=selected[0], input_root=args.input_root, output_dir=output_dir, suite_profile=args.suite_profile)
        print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
        return 0 if result.get("status") != "failed" else 1
    if args.output_root is None:
        raise ValueError("--output-root is required for batch enrichment")
    summary = run_batch(records_path=args.records, input_root=args.input_root, output_root=args.output_root, suite_profile=args.suite_profile)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
