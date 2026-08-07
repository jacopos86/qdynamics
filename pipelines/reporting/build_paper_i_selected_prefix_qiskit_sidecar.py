#!/usr/bin/env python3
"""Build a Paper-I selected-prefix Qiskit resource sidecar.

This is for plateau/prefix resource rows that are not native fixed-threshold
``paper_i_first_crossing`` artifacts.  It reuses the Paper-I Table-I Qiskit
compiler convention (``table_i_basis_gate_transpile_v1``), reconstructs the
accepted prefix from history insertion positions, and writes an additive JSON
sidecar without modifying the raw run artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.snake_table_i_measurement_work import (  # noqa: E402
    snake_algorithmic_work_from_payload,
    snake_mechanism_resolved_work_from_payload,
)
from pipelines.exact_bench.table_i_first_hit_sidecars import (  # noqa: E402
    _reference_state_from_payload,
    _sha256_json_without_snake_sidecars,
)
from pipelines.exact_bench.table_i_qiskit_resource_compile import (  # noqa: E402
    TABLE_I_QISKIT_COMPILE_CONVENTION,
    compile_table_i_ansatz_terms,
)
from pipelines.scaffold.hh_continuation_generators import (  # noqa: E402
    serialize_polynomial_terms_exyz,
)
from pipelines.exact_bench.static_benchmark_runtime import (  # noqa: E402
    _paper_i_history_row_acceptance_status,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial  # noqa: E402
from src.quantum.qubitization_module import PauliTerm  # noqa: E402
from src.quantum.vqe_latex_python_pairs import AnsatzTerm  # noqa: E402

SIDECAR_SCHEMA = "paper_i_selected_prefix_qiskit_cost_sidecar_v1"
SOURCE_KIND = "snake_qiskit_compiled_selected_prefix_ansatz_circuit"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON payload is not an object: {path}")
    return dict(payload)


def _adapt_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = payload.get("adapt_vqe")
    return nested if isinstance(nested, Mapping) else payload


def _history_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    adapt = _adapt_payload(payload)
    rows = adapt.get("history")
    has_explicit_history = isinstance(rows, Sequence) and not isinstance(
        rows, (str, bytes)
    )
    try:
        history_count = int(adapt.get("history_count"))
        tail_count = int(adapt.get("history_tail_count"))
    except (TypeError, ValueError):
        history_count = -1
        tail_count = -1
    if (
        not has_explicit_history
        or not rows
        or (history_count > 0 and len(rows) != history_count)
    ):
        tail = adapt.get("history_tail")
        if (
            isinstance(tail, Sequence)
            and not isinstance(tail, (str, bytes))
            and history_count == tail_count == len(tail)
            and history_count > 0
        ):
            rows = tail
        elif history_count > 0:
            continuation = adapt.get("continuation")
            scaffold = (
                continuation.get("selected_scaffold_history")
                if isinstance(continuation, Mapping)
                else None
            )
            if not isinstance(scaffold, Sequence) or isinstance(
                scaffold, (str, bytes)
            ):
                scaffold = adapt.get("selected_scaffold_history")
            if (
                isinstance(scaffold, Sequence)
                and not isinstance(scaffold, (str, bytes))
                and len(scaffold) == history_count
                and all(isinstance(row, Mapping) for row in scaffold)
            ):
                rows = []
                for position, raw_row in enumerate(scaffold, start=1):
                    row = dict(raw_row)
                    row.setdefault("depth", int(row.get("step_index", position)))
                    rows.append(row)
            elif has_explicit_history:
                rows = []
            else:
                raise ValueError(
                    "payload is missing a complete adapt_vqe.history, "
                    "history_tail, or selected_scaffold_history"
                )
        elif has_explicit_history:
            rows = []
        else:
            raise ValueError(
                "payload is missing a complete adapt_vqe.history or history_tail"
            )
    return [row for row in rows if isinstance(row, Mapping)]


def _terminal_winner_query_work(
    payload: Mapping[str, Any],
    *,
    history_position: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Return the validated terminal winner-lineage audit when it is exact."""

    rows = _history_rows(payload)
    if int(history_position) != len(rows):
        return None, {
            "status": "not_applicable",
            "reason": "requested_prefix_is_not_terminal",
        }

    summary = payload.get("summary")
    audit = summary.get("query_work_audit") if isinstance(summary, Mapping) else None
    if not isinstance(audit, Mapping):
        return None, {"status": "unavailable", "reason": "terminal_query_audit_missing"}
    if audit.get("status") != "ok":
        return None, {"status": "unavailable", "reason": "terminal_query_audit_not_ok"}
    if audit.get("S_alg_work_scope") != "winner_lineage_terminal":
        return None, {
            "status": "unavailable",
            "reason": "terminal_query_audit_scope_mismatch",
        }

    winner_position = audit.get("winner_history_position")
    if winner_position is None:
        winner_position = audit.get("winner_history_count")
    try:
        winner_position_int = int(winner_position)
    except (TypeError, ValueError):
        return None, {
            "status": "unavailable",
            "reason": "terminal_query_audit_winner_position_missing",
        }
    if winner_position_int != int(history_position):
        return None, {
            "status": "unavailable",
            "reason": "terminal_query_audit_winner_position_mismatch",
        }

    components = audit.get("components")
    if not isinstance(components, Mapping):
        components = audit.get("query_work_components")
    component_map = {
        "S_alg_N_H_outer_eval": "N_H_outer_eval",
        "S_alg_N_H_refit_eval": "N_H_refit_eval",
        "S_alg_N_grad_probe": "N_grad",
        "S_alg_N_metric_probe": "N_metric",
    }
    normalized: dict[str, float] = {}
    try:
        for target, source in component_map.items():
            value = float(components[source])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(source)
            normalized[target] = value
        other = float(audit.get("N_other_quantum", 0.0))
        total = float(audit.get("S_alg"))
        if not math.isfinite(other) or other < 0.0 or not math.isfinite(total):
            raise ValueError("nonfinite")
    except (KeyError, TypeError, ValueError):
        return None, {
            "status": "unavailable",
            "reason": "terminal_query_audit_components_invalid",
        }
    normalized["S_alg_N_other_quantum"] = other
    component_sum = float(sum(normalized.values()))
    if not math.isclose(component_sum, total, rel_tol=1e-12, abs_tol=1e-9):
        return None, {
            "status": "unavailable",
            "reason": "terminal_query_audit_component_sum_mismatch",
            "component_sum": component_sum,
            "S_alg": total,
        }

    return {
        "S_alg": total,
        "S_alg_status": "ok",
        "S_alg_work_scope": "winner_lineage_terminal",
        **normalized,
    }, {
        "status": "ok",
        "scope": "winner_lineage_terminal",
        "source": "summary.query_work_audit",
        "winner_history_position": winner_position_int,
        "component_sum": component_sum,
    }


def _list_text(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value if str(item).strip()]
    return []


def _list_int(value: Any) -> list[int]:
    out: list[int] = []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = value
    elif value is not None and value != "":
        values = [value]
    else:
        values = []
    for item in values:
        try:
            parsed = int(float(item))
        except Exception:
            continue
        out.append(parsed)
    return out


def _pauli_group_from_terms(raw_terms: Any) -> list[str]:
    if not isinstance(raw_terms, Sequence) or isinstance(raw_terms, (str, bytes)):
        return []
    group: list[str] = []
    for term in raw_terms:
        if not isinstance(term, Mapping):
            continue
        label = str(term.get("pauli_exyz") or "").strip().lower()
        if label:
            group.append(label)
    return group


def _serialized_terms(raw_terms: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_terms, Sequence) or isinstance(raw_terms, (str, bytes)):
        return []
    out: list[dict[str, Any]] = []
    for raw in raw_terms:
        if not isinstance(raw, Mapping):
            continue
        label = str(raw.get("pauli_exyz") or "").strip().lower()
        if not label:
            continue
        out.append(
            {
                "pauli_exyz": label,
                "coeff_re": float(raw.get("coeff_re", 0.0)),
                "coeff_im": float(raw.get("coeff_im", 0.0)),
                "nq": int(raw.get("nq", len(label))),
            }
        )
    return out


def _same_pauli_direction(
    left_terms: Any,
    right_terms: Any,
    *,
    atol: float = 1.0e-10,
) -> bool:
    """Compare generators modulo one irrelevant nonzero complex scalar."""

    def canonical(raw_terms: Any) -> tuple[tuple[str, ...], np.ndarray] | None:
        coefficients: dict[str, complex] = {}
        for row in _serialized_terms(raw_terms):
            label = str(row["pauli_exyz"])
            coefficients[label] = coefficients.get(label, 0.0j) + complex(
                float(row["coeff_re"]),
                float(row["coeff_im"]),
            )
        coefficients = {
            label: value
            for label, value in coefficients.items()
            if abs(value) > float(atol)
        }
        if not coefficients:
            return None
        labels = tuple(sorted(coefficients))
        vector = np.asarray([coefficients[label] for label in labels], dtype=complex)
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= float(atol):
            return None
        vector = vector / norm
        anchor = next(value for value in vector if abs(value) > float(atol))
        vector = vector / (anchor / abs(anchor))
        return labels, vector

    left = canonical(left_terms)
    right = canonical(right_terms)
    if left is None or right is None or left[0] != right[0]:
        return False
    return bool(np.allclose(left[1], right[1], rtol=0.0, atol=float(atol)))


def _execution_mode_for_label(
    label: str,
    *,
    explicit: Any = None,
) -> str:
    explicit_text = str(explicit or "").strip().lower()
    if explicit_text in {"termwise_product", "grouped_exact"}:
        return explicit_text
    if str(label).endswith("::legal_projected"):
        return "grouped_exact"
    return "termwise_product"


def _ansatz_term_from_serialized(
    *,
    label: str,
    terms: Sequence[Mapping[str, Any]],
    execution_mode: str,
) -> AnsatzTerm:
    serialized = _serialized_terms(terms)
    if not serialized:
        raise ValueError(f"missing serialized Pauli terms for {label}")
    polynomial = PauliPolynomial(
        "JW",
        [
            PauliTerm(
                int(row["nq"]),
                ps=str(row["pauli_exyz"]),
                pc=complex(float(row["coeff_re"]), float(row["coeff_im"])),
            )
            for row in serialized
        ],
    )
    return AnsatzTerm(
        label=str(label),
        polynomial=polynomial,
        execution_mode=_execution_mode_for_label(
            str(label),
            explicit=execution_mode,
        ),
    )


def _final_block_group_map(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    adapt = _adapt_payload(payload)
    parameterization = adapt.get("parameterization") if isinstance(adapt, Mapping) else None
    blocks = parameterization.get("blocks") if isinstance(parameterization, Mapping) else None
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
        return out
    for block_index, block in enumerate(blocks):
        if not isinstance(block, Mapping):
            continue
        label = str(block.get("candidate_label") or "").strip()
        terms = _serialized_terms(block.get("runtime_terms_exyz"))
        group = _pauli_group_from_terms(terms)
        if label and group and label not in out:
            out[label] = {
                "group": group,
                "terms": terms,
                "execution_mode": _execution_mode_for_label(
                    label,
                    explicit=block.get("execution_mode"),
                ),
                "block_index": int(block_index),
            }
    return out


def _record_group_for_label(row: Mapping[str, Any], label: str) -> tuple[list[str], str | None]:
    containers = (
        "admitted_records",
        "selected_feature_rows",
        "retained_shortlist_records",
        "shortlisted_records",
        "scored_surface_records",
    )
    for key in containers:
        records = row.get(key)
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            continue
        for record in records:
            if not isinstance(record, Mapping):
                continue
            record_label = str(record.get("candidate_label") or record.get("selected_op") or "").strip()
            if record_label and record_label != str(label):
                continue
            compile_meta = ((record.get("generator_metadata") or {}).get("compile_metadata") or {})
            group = _pauli_group_from_terms(compile_meta.get("serialized_terms_exyz"))
            if group:
                return group, f"history_row.{key}.generator_metadata.compile_metadata.serialized_terms_exyz"
    return [], None


def _record_term_map(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out = _final_block_group_map(payload)
    for row_index, row in enumerate(_history_rows(payload), start=1):
        for key in (
            "admitted_records",
            "selected_records",
            "selected_feature_rows",
            "retained_shortlist_records",
            "shortlisted_records",
            "scored_surface_records",
        ):
            records = row.get(key)
            if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
                continue
            for record_index, record in enumerate(records):
                if not isinstance(record, Mapping):
                    continue
                label = str(
                    record.get("candidate_label")
                    or record.get("operator_label")
                    or record.get("generator_label")
                    or record.get("selected_op")
                    or ""
                ).strip()
                if not label or label in out:
                    continue
                generator_metadata = record.get("generator_metadata")
                compile_metadata = (
                    generator_metadata.get("compile_metadata")
                    if isinstance(generator_metadata, Mapping)
                    else None
                )
                terms = _serialized_terms(
                    compile_metadata.get("serialized_terms_exyz")
                    if isinstance(compile_metadata, Mapping)
                    else None
                )
                if not terms:
                    terms = _serialized_terms(record.get("runtime_terms_exyz"))
                if not terms:
                    continue
                out[label] = {
                    "group": _pauli_group_from_terms(terms),
                    "terms": terms,
                    "execution_mode": _execution_mode_for_label(
                        label,
                        explicit=(
                            record.get("execution_mode")
                            or (
                                compile_metadata.get("execution_mode")
                                if isinstance(compile_metadata, Mapping)
                                else None
                            )
                        ),
                    ),
                    "source": f"history[{row_index}].{key}[{record_index}]",
                }
    # Beam checkpoints can retain an earlier-prefix operator after it has left
    # the terminal parameterization. Its coefficient-bearing metadata remains
    # in nested candidate/beam telemetry, so index that structured metadata
    # before attempting deterministic regeneration under newer code.
    stack: list[tuple[Any, str]] = [(payload, "payload")]
    while stack:
        value, source_path = stack.pop()
        if isinstance(value, Mapping):
            label = str(
                value.get("candidate_label")
                or value.get("operator_label")
                or value.get("generator_label")
                or value.get("selected_op")
                or ""
            ).strip()
            generator_metadata = value.get("generator_metadata")
            nested_compile_metadata = (
                generator_metadata.get("compile_metadata")
                if isinstance(generator_metadata, Mapping)
                else None
            )
            direct_compile_metadata = value.get("compile_metadata")
            compile_metadata = (
                nested_compile_metadata
                if isinstance(nested_compile_metadata, Mapping)
                else direct_compile_metadata
                if isinstance(direct_compile_metadata, Mapping)
                else None
            )
            terms = _serialized_terms(
                compile_metadata.get("serialized_terms_exyz")
                if isinstance(compile_metadata, Mapping)
                else None
            )
            if not terms:
                terms = _serialized_terms(value.get("runtime_terms_exyz"))
            if label and terms:
                existing = out.get(label)
                if existing is not None:
                    existing_terms = _serialized_terms(existing.get("terms"))
                    if not _same_pauli_direction(existing_terms, terms):
                        raise ValueError(
                            "conflicting coefficient-bearing Pauli metadata for "
                            f"selected label {label!r}"
                        )
                else:
                    out[label] = {
                        "group": _pauli_group_from_terms(terms),
                        "terms": terms,
                        "execution_mode": _execution_mode_for_label(
                            label,
                            explicit=(
                                value.get("execution_mode")
                                or (
                                    compile_metadata.get("execution_mode")
                                    if isinstance(compile_metadata, Mapping)
                                    else None
                                )
                            ),
                        ),
                        "source": source_path,
                    }
            for key, child in value.items():
                if isinstance(child, (Mapping, list, tuple)):
                    stack.append((child, f"{source_path}.{key}"))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                if isinstance(child, (Mapping, list, tuple)):
                    stack.append((child, f"{source_path}[{index}]"))
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regenerate_missing_term_records(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    labels: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Rebuild deterministic projected child terms omitted by compact payloads."""

    plan_path = source_path.with_name("plan.json")
    if not plan_path.is_file():
        return {}, {
            "status": "plan_json_missing",
            "plan_json": str(plan_path),
        }
    plan = _load_json(plan_path)
    scientific = plan.get("scientific_settings")
    run_kwargs = (
        scientific.get("run_kwargs")
        if isinstance(scientific, Mapping)
        else None
    )
    if not isinstance(run_kwargs, Mapping):
        return {}, {
            "status": "plan_run_kwargs_missing",
            "plan_json": str(plan_path),
        }

    settings = payload.get("settings")
    if not isinstance(settings, Mapping):
        return {}, {
            "status": "payload_settings_missing",
            "plan_json": str(plan_path),
        }

    # Keep these imports behind the compact-payload fallback. Importing this
    # reporting module must not resolve a Hamiltonian or build an operator pool.
    from pipelines.contracts.problem import ProblemRequest
    from pipelines.static_adapt.builders.pool_resolution import resolve_pool_plan
    from pipelines.static_adapt.builders.problem_registry import resolve_problem_context
    from pipelines.static_adapt.route_a_child_padding import RouteAChildPaddingConfig
    from pipelines.static_adapt.route_a_shortlists import canonicalize_pauli_child_direction
    from pipelines.static_adapt.runtime_split import build_global_child_records_for_parent
    from src.quantum.hubbard_latex_python_pairs import boson_qubits_per_site

    problem_key = str(settings.get("problem") or run_kwargs.get("problem") or "")
    num_sites = int(settings.get("L") or run_kwargs.get("num_sites") or 0)
    request = ProblemRequest(
        problem_key=problem_key,
        num_sites=num_sites,
        t=float(settings.get("t", run_kwargs.get("t", 1.0))),
        u=float(settings.get("u", run_kwargs.get("u", 0.0))),
        dv=float(settings.get("dv", run_kwargs.get("dv", 0.0))),
        omega0=float(settings.get("omega0", run_kwargs.get("omega0", 1.0))),
        g_ep=float(settings.get("g_ep", run_kwargs.get("g_ep", 0.0))),
        n_ph_max=int(settings.get("n_ph_max", run_kwargs.get("n_ph_max", 1))),
        boson_encoding=str(
            settings.get("boson_encoding")
            or run_kwargs.get("boson_encoding")
            or "binary"
        ),
        ordering=str(settings.get("ordering") or run_kwargs.get("ordering") or "blocked"),
        boundary=str(settings.get("boundary") or run_kwargs.get("boundary") or "open"),
        include_zero_point=bool(
            settings.get(
                "include_zero_point",
                run_kwargs.get("include_zero_point", True),
            )
        ),
    )
    context = resolve_problem_context(request)
    pool_plan = resolve_pool_plan(
        resolved_problem=context,
        continuation_mode=str(
            run_kwargs.get("adapt_continuation_mode")
            or settings.get("continuation_mode")
            or "phase3_v1"
        ),
        adapt_pool=str(run_kwargs.get("adapt_pool") or "full_meta"),
        paop_r=int(run_kwargs.get("paop_r", 0)),
        paop_split_paulis=bool(run_kwargs.get("paop_split_paulis", False)),
        paop_prune_eps=float(run_kwargs.get("paop_prune_eps", 0.0)),
        paop_normalization=str(run_kwargs.get("paop_normalization", "none")),
        phase3_symmetry_mitigation_mode=str(
            run_kwargs.get("phase3_symmetry_mitigation_mode", "off")
        ),
    )
    pool_index_by_label = {
        str(term.label): index for index, term in enumerate(pool_plan.pool)
    }
    funnel_config = run_kwargs.get("route_a_funnel_config")
    child_padding_payload = (
        funnel_config.get("child_padding")
        if isinstance(funnel_config, Mapping)
        else None
    )
    child_padding_config = (
        RouteAChildPaddingConfig(**dict(child_padding_payload))
        if isinstance(child_padding_payload, Mapping)
        else None
    )
    subset_sizes_raw = run_kwargs.get("phase3_runtime_split_subset_sizes")
    subset_sizes = (
        tuple(int(value) for value in subset_sizes_raw)
        if isinstance(subset_sizes_raw, Sequence)
        and not isinstance(subset_sizes_raw, (str, bytes))
        else (1,)
    )
    qpb = boson_qubits_per_site(
        int(request.n_ph_max),
        str(request.boson_encoding),
    )
    required = {str(label) for label in labels}
    parents = sorted(
        {
            label.split("::child_set[", 1)[0]
            for label in required
            if "::child_set[" in label
        }
    )
    regenerated: dict[str, dict[str, Any]] = {}
    missing_parents: list[str] = []
    for parent_label in parents:
        pool_index = pool_index_by_label.get(parent_label)
        if pool_index is None:
            missing_parents.append(parent_label)
            continue
        parent_metadata = pool_plan.pool_generator_registry.get(parent_label)
        records, _telemetry = build_global_child_records_for_parent(
            parent_label=parent_label,
            parent_term=pool_plan.pool[int(pool_index)],
            parent_family_id=str(pool_plan.pool_family_ids[int(pool_index)]),
            parent_generator_metadata=(
                dict(parent_metadata)
                if isinstance(parent_metadata, Mapping)
                else None
            ),
            parent_symmetry_spec=pool_plan.pool_symmetry_specs[int(pool_index)],
            child_set_symmetry_policy=str(
                run_kwargs.get(
                    "phase3_runtime_split_child_set_symmetry_policy",
                    "hard_guard",
                )
            ),
            subset_sizes=subset_sizes,
            num_sites=int(request.num_sites),
            ordering=str(request.ordering),
            qpb=int(qpb),
            problem_key=str(problem_key),
            fixed_num_particles=context.sector.num_particles,
            evaluate_candidate=lambda **kwargs: kwargs,
            child_padding_config=child_padding_config,
            defer_phase1_evaluation=True,
            base_record={},
        )
        for record in records:
            label = str(record.get("candidate_label") or "")
            if label not in required:
                continue
            canonical_term, _normalization = canonicalize_pauli_child_direction(
                record.get("candidate_term")
            )
            terms = _serialized_terms(
                serialize_polynomial_terms_exyz(canonical_term.polynomial)
            )
            regenerated[label] = {
                "group": _pauli_group_from_terms(terms),
                "terms": terms,
                "execution_mode": _execution_mode_for_label(
                    label,
                    explicit=getattr(canonical_term, "execution_mode", None),
                ),
                "source": "deterministic_plan_pool_child_regeneration_v1",
            }
    return regenerated, {
        "status": "ok" if required.issubset(regenerated) else "incomplete",
        "plan_json": str(plan_path),
        "plan_json_sha256": _sha256(plan_path),
        "requested_labels": sorted(required),
        "regenerated_labels": sorted(regenerated),
        "missing_labels": sorted(required - set(regenerated)),
        "missing_parent_labels": missing_parents,
    }


def _beam_prefix_operator_labels(
    payload: Mapping[str, Any],
    *,
    history_position: int,
) -> tuple[list[str] | None, dict[str, Any]]:
    adapt = _adapt_payload(payload)
    history = _history_rows(payload)
    round_index = int(history_position) - 1
    if round_index < 0 or round_index >= len(history):
        return None, {"status": "history_position_out_of_range"}
    branch_id = history[round_index].get("branch_id")
    if int(history_position) == len(history):
        terminal_labels = _list_text(adapt.get("operators"))
        terminal_depth = adapt.get("ansatz_depth")
        if terminal_labels and terminal_depth is not None:
            if int(terminal_depth) != len(terminal_labels):
                raise ValueError(
                    "terminal winner operator label count does not match ansatz depth: "
                    f"{len(terminal_labels)} != {terminal_depth}"
                )
            return terminal_labels, {
                "status": "ok",
                "source": "adapt_vqe.operators_terminal_winner",
                "round_index": round_index,
                "branch_id": adapt.get("branch_id", branch_id),
                "ansatz_depth": len(terminal_labels),
            }
    telemetry = adapt.get("beam_replay_telemetry")
    rounds = telemetry.get("rounds") if isinstance(telemetry, Mapping) else None
    if not isinstance(rounds, Sequence) or isinstance(rounds, (str, bytes)):
        return None, {"status": "beam_round_telemetry_missing"}
    if round_index < 0 or round_index >= len(rounds):
        return None, {"status": "beam_round_telemetry_out_of_range"}
    round_payload = rounds[round_index]
    frontier = round_payload.get("frontier") if isinstance(round_payload, Mapping) else None
    branches = frontier.get("branches") if isinstance(frontier, Mapping) else None
    if not isinstance(branches, Sequence) or isinstance(branches, (str, bytes)):
        return None, {"status": "beam_frontier_branches_missing"}
    for branch in branches:
        if not isinstance(branch, Mapping) or branch.get("branch_id") != branch_id:
            continue
        labels = _list_text(branch.get("operator_labels"))
        if not labels:
            return None, {"status": "beam_operator_labels_missing"}
        ansatz_depth = branch.get("ansatz_depth")
        if ansatz_depth is not None and int(ansatz_depth) != len(labels):
            raise ValueError(
                "beam prefix operator label count does not match ansatz depth: "
                f"{len(labels)} != {ansatz_depth}"
            )
        return labels, {
            "status": "ok",
            "source": "beam_replay_telemetry.rounds.frontier.branch.operator_labels",
            "round_index": round_index,
            "branch_id": branch_id,
            "ansatz_depth": len(labels),
        }
    return None, {
        "status": "beam_history_branch_not_found_in_frontier",
        "branch_id": branch_id,
    }


def _num_qubits_from_groups(groups: Sequence[Sequence[str]], payload: Mapping[str, Any]) -> int:
    for group in groups:
        for label in group:
            if label:
                return len(str(label))
    adapt = _adapt_payload(payload)
    for source in (adapt, payload):
        for key in ("num_qubits", "nq", "total_qubits"):
            raw = source.get(key) if isinstance(source, Mapping) else None
            try:
                parsed = int(float(raw))
            except Exception:
                continue
            if parsed > 0:
                return parsed
    raise ValueError("could not infer num_qubits")


def _selected_prefix_reference_state(
    payload: Mapping[str, Any],
    *,
    num_qubits: int,
) -> tuple[np.ndarray | None, str]:
    state, status = _reference_state_from_payload(
        payload,
        num_qubits=int(num_qubits),
    )
    if state is not None:
        return state, status
    for source_name, source in (
        ("ansatz_input_state", payload.get("ansatz_input_state")),
        ("adapt_vqe.ansatz_input_state", _adapt_payload(payload).get("ansatz_input_state")),
    ):
        if not isinstance(source, Mapping):
            continue
        amplitudes = source.get("amplitudes_qn_to_q0")
        if not isinstance(amplitudes, Mapping):
            continue
        declared_nq = source.get("nq_total")
        if declared_nq is not None and int(declared_nq) != int(num_qubits):
            continue
        vector = np.zeros(1 << int(num_qubits), dtype=complex)
        valid = True
        for bitstring, raw_amplitude in amplitudes.items():
            bits = str(bitstring).strip()
            if len(bits) != int(num_qubits) or set(bits) - {"0", "1"}:
                valid = False
                break
            if isinstance(raw_amplitude, Mapping):
                amplitude = complex(
                    float(raw_amplitude.get("re", 0.0)),
                    float(raw_amplitude.get("im", 0.0)),
                )
            else:
                try:
                    amplitude = complex(raw_amplitude)
                except Exception:
                    valid = False
                    break
            vector[int(bits, 2)] = amplitude
        norm = float(np.linalg.norm(vector))
        if valid and norm > 0.0 and math.isfinite(norm):
            return vector / norm, f"statevector_from_{source_name}"
    return None, status


def reconstruct_prefix_groups(
    payload: Mapping[str, Any],
    *,
    history_position: int,
    source_path: Path | None = None,
) -> tuple[list[str], list[list[str]], dict[str, Any]]:
    labels, ops, meta = reconstruct_prefix_ansatz(
        payload,
        history_position=int(history_position),
        source_path=source_path,
    )
    groups = [
        _pauli_group_from_terms(serialize_polynomial_terms_exyz(op.polynomial))
        for op in ops
    ]
    return labels, groups, meta


def reconstruct_prefix_ansatz(
    payload: Mapping[str, Any],
    *,
    history_position: int,
    source_path: Path | None = None,
) -> tuple[list[str], list[AnsatzTerm], dict[str, Any]]:
    history = _history_rows(payload)
    if int(history_position) < 1 or int(history_position) > len(history):
        raise ValueError(f"history_position={history_position} outside history length {len(history)}")
    exact_beam_labels, beam_meta = _beam_prefix_operator_labels(
        payload,
        history_position=int(history_position),
    )
    ordered_labels: list[str] = list(exact_beam_labels or [])
    committed_operator_count = None
    initial_operator_count = None
    committed_depth = None
    accepted_count = 0
    reason_counts: Counter[str] = Counter()

    if exact_beam_labels is None:
        for row in history[: int(history_position)]:
            accepted, reason = _paper_i_history_row_acceptance_status(
                row,
                committed_operator_count=committed_operator_count,
                initial_operator_count=initial_operator_count,
                committed_depth=committed_depth,
            )
            reason_counts[str(reason)] += 1
            if not accepted:
                row_operator_count = row.get("logical_num_parameters_after_opt")
                if (
                    reason == "preexisting_initial_operator_count"
                    and row_operator_count is not None
                    and initial_operator_count is None
                ):
                    try:
                        initial_operator_count = int(float(row_operator_count))
                    except Exception:
                        pass
                continue
            selected_records = row.get("selected_records")
            records = (
                [record for record in selected_records if isinstance(record, Mapping)]
                if isinstance(selected_records, Sequence)
                and not isinstance(selected_records, (str, bytes))
                else []
            )
            labels = [
                str(
                    record.get("operator_label")
                    or record.get("generator_label")
                    or record.get("candidate_label")
                    or ""
                )
                for record in records
            ]
            labels = [label for label in labels if label]
            positions = [
                int(record.get("position_id"))
                for record in records
                if record.get("position_id") is not None
            ]
            if not labels:
                labels = _list_text(row.get("selected_ops")) or _list_text(
                    row.get("selected_logical_ops")
                )
            if not labels:
                labels = _list_text(
                    row.get("selected_logical_op") or row.get("selected_op")
                )
            if len(positions) != len(labels):
                positions = _list_int(row.get("selected_positions")) or _list_int(
                    row.get("selected_position")
                )
            if len(positions) != len(labels):
                positions = [
                    len(ordered_labels) + offset for offset in range(len(labels))
                ]
            for label, pos in zip(labels, positions):
                insert_at = max(0, min(int(pos), len(ordered_labels)))
                ordered_labels.insert(insert_at, str(label))

            prune = row.get("post_admission_prune")
            if isinstance(prune, Mapping) and int(prune.get("accepted_count", 0)) > 0:
                selected_index = prune.get("selected_index")
                if selected_index is not None:
                    prune_index = int(selected_index)
                    if prune_index < 0 or prune_index >= len(ordered_labels):
                        raise ValueError(
                            "accepted prefix prune index is outside the reconstructed ansatz: "
                            f"{prune_index} not in [0, {len(ordered_labels)})"
                        )
                    expected_label = str(prune.get("selected_label") or "")
                    observed_label = ordered_labels[prune_index]
                    if expected_label and observed_label != expected_label:
                        raise ValueError(
                            "accepted prefix prune label mismatch: "
                            f"index {prune_index} has {observed_label!r}, expected "
                            f"{expected_label!r}"
                        )
                    del ordered_labels[prune_index]

            accepted_count += 1
            row_operator_count = row.get("logical_num_parameters_after_opt")
            if row_operator_count is not None:
                try:
                    committed_operator_count = int(float(row_operator_count))
                except Exception:
                    pass
            row_depth = row.get("depth")
            if row_depth is not None:
                try:
                    committed_depth = int(float(row_depth))
                except Exception:
                    pass

    term_map = _record_term_map(payload)
    missing_labels = sorted(set(ordered_labels) - set(term_map))
    regeneration_meta: dict[str, Any] | None = None
    if missing_labels and source_path is not None:
        regenerated, regeneration_meta = _regenerate_missing_term_records(
            payload,
            source_path=Path(source_path),
            labels=missing_labels,
        )
        term_map.update(regenerated)
        missing_labels = sorted(set(ordered_labels) - set(term_map))
    if missing_labels:
        raise ValueError(
            "missing coefficient-bearing Pauli terms for selected labels: "
            + ", ".join(missing_labels)
        )

    ops: list[AnsatzTerm] = []
    operator_sources: list[dict[str, Any]] = []
    for position, label in enumerate(ordered_labels, start=1):
        term_payload = term_map[label]
        op = _ansatz_term_from_serialized(
            label=label,
            terms=term_payload.get("terms") or [],
            execution_mode=str(term_payload.get("execution_mode") or ""),
        )
        ops.append(op)
        operator_sources.append(
            {
                "position": int(position),
                "label": str(label),
                "source": str(
                    term_payload.get("source")
                    or (
                        "final_parameterization.blocks"
                        if term_payload.get("block_index") is not None
                        else "payload_term_map"
                    )
                ),
                "execution_mode": str(op.execution_mode),
                "serialized_terms_exyz": _serialized_terms(
                    serialize_polynomial_terms_exyz(op.polynomial)
                ),
            }
        )
    meta = {
        "history_position": int(history_position),
        "accepted_replayed_count": int(accepted_count),
        "replayed_operator_count": int(len(ordered_labels)),
        "acceptance_reason_counts": dict(sorted(reason_counts.items())),
        "operator_order": beam_meta,
        "term_regeneration": regeneration_meta,
        "selected_operator_group_sources": operator_sources,
    }
    return ordered_labels, ops, meta


def build_sidecar(
    *,
    result_json: Path,
    history_position: int,
    output_json: Path,
    threshold: float | None,
    result_payload: Mapping[str, Any] | None = None,
    source_result_sha256: str | None = None,
    source_result_hash_convention: str | None = None,
) -> dict[str, Any]:
    payload = (
        dict(result_payload)
        if isinstance(result_payload, Mapping)
        else _load_json(result_json)
    )
    labels, ops, replay_meta = reconstruct_prefix_ansatz(
        payload,
        history_position=int(history_position),
        source_path=result_json,
    )
    groups = [
        _pauli_group_from_terms(serialize_polynomial_terms_exyz(op.polynomial))
        for op in ops
    ]
    num_qubits = _num_qubits_from_groups(groups, payload)
    reference_state, reference_state_status = _selected_prefix_reference_state(
        payload,
        num_qubits=int(num_qubits),
    )
    if reference_state is None:
        raise ValueError(f"reference state unavailable: {reference_state_status}")
    compiled = compile_table_i_ansatz_terms(
        ops=ops,
        num_qubits=int(num_qubits),
        reference_state=reference_state,
        source_kind=SOURCE_KIND,
    )
    runtime_work, runtime_audit = snake_algorithmic_work_from_payload(
        payload,
        scope="display_prefix",
        history_position=int(history_position),
        source_label=str(result_json),
    )
    terminal_runtime_work, terminal_runtime_audit = _terminal_winner_query_work(
        payload,
        history_position=int(history_position),
    )
    if terminal_runtime_work is not None:
        runtime_work = terminal_runtime_work
        runtime_audit = terminal_runtime_audit
        runtime_work_source = "summary.query_work_audit"
    else:
        runtime_work_source = "snake_algorithmic_work_from_payload"
    mechanism_work, mechanism_audit = snake_mechanism_resolved_work_from_payload(
        payload,
        scope="display_prefix",
        history_position=int(history_position),
        source_label=str(result_json),
    )
    history_row = _history_rows(payload)[int(history_position) - 1]
    adapt = _adapt_payload(payload)
    terminal_prefix = int(history_position) == len(_history_rows(payload))
    primary_error = history_row.get("delta_abs_current")
    energy_at_prefix = history_row.get("energy_after_opt")
    prefix_value_source = "adapt_vqe.history"
    if terminal_prefix and adapt.get("abs_delta_e") is not None:
        primary_error = adapt.get("abs_delta_e")
        energy_at_prefix = adapt.get("energy")
        prefix_value_source = "adapt_vqe.terminal_winner"
    if source_result_sha256 is None:
        source_hash = _sha256_json_without_snake_sidecars(result_json)
        hash_convention = "canonical_json_without_snake_sidecars_v1"
    else:
        source_hash = str(source_result_sha256)
        hash_convention = str(
            source_result_hash_convention or "caller_supplied_sha256_v1"
        )
    mechanism_algorithmic = mechanism_work.get("mechanism_algorithmic_work")
    if not isinstance(mechanism_algorithmic, Mapping):
        mechanism_algorithmic = mechanism_work

    sidecar = {
        "schema": SIDECAR_SCHEMA,
        "source_kind": SOURCE_KIND,
        "source_result_path": str(result_json),
        "source_result_sha256": source_hash,
        "source_result_hash_convention": hash_convention,
        "history_position": int(history_position),
        "k_pl": int(history_position),
        "threshold_reference": None if threshold is None else float(threshold),
        "primary_error_at_prefix": primary_error,
        "energy_after_opt_at_prefix": energy_at_prefix,
        "prefix_value_source": prefix_value_source,
        "selected_operator_labels": labels,
        "selected_operator_pauli_labels_exyz": groups,
        "selected_operator_terms_exyz": [
            _serialized_terms(serialize_polynomial_terms_exyz(op.polynomial))
            for op in ops
        ],
        "replay": replay_meta,
        "reference_state_status": reference_state_status,
        "compile_convention_expected": TABLE_I_QISKIT_COMPILE_CONVENTION,
        "compile_input_semantics": "coefficient_bearing_execution_aware_ansatz_terms_v1",
        **compiled,
        "instrumented_runtime_S": runtime_work.get("S_alg"),
        "instrumented_runtime_status": runtime_work.get("S_alg_status"),
        "instrumented_runtime_source": runtime_work_source,
        "instrumented_runtime_terminal_audit": terminal_runtime_audit,
        "instrumented_runtime_scope": (
            runtime_work.get("S_alg_work_scope")
            or runtime_work.get("work_scope")
            or runtime_audit.get("S_alg_work_scope")
            or runtime_audit.get("scope")
        ),
        "instrumented_runtime_components": {
            key: runtime_work.get(key)
            for key in (
                "S_alg_N_grad_probe",
                "S_alg_N_metric_probe",
                "S_alg_N_H_refit_eval",
                "S_alg_N_H_outer_eval",
                "S_alg_N_other_quantum",
            )
            if key in runtime_work
        },
        "instrumented_runtime_audit_status": runtime_audit.get("status") if isinstance(runtime_audit, Mapping) else None,
        "mechanism_formula_S": mechanism_algorithmic.get("S_alg"),
        "mechanism_formula_status": mechanism_algorithmic.get("status") or mechanism_work.get("mechanism_resolution_status"),
        "mechanism_formula_publishable_flag": mechanism_algorithmic.get("publishable"),
        "mechanism_formula_requires_formula_reconstruction": mechanism_algorithmic.get("requires_formula_reconstruction"),
        "mechanism_formula_components": {
            key: mechanism_algorithmic.get(key)
            for key in (
                "S_alg_N_grad_probe",
                "S_alg_N_metric_probe",
                "S_alg_N_H_refit_eval",
                "S_alg_N_H_outer_eval",
                "S_alg_N_other_quantum",
            )
            if key in mechanism_algorithmic
        },
        "mechanism_formula_audit_status": mechanism_audit.get("status") if isinstance(mechanism_audit, Mapping) else None,
        "mechanism_formula_resolution_status": mechanism_work.get("mechanism_resolution_status"),
        "paper_i_main_S_convention": "paper_i_winning_branch_s_alg_v1",
    }
    _write_json(output_json, sidecar)
    return sidecar


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--history-position", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    sidecar = build_sidecar(
        result_json=args.result_json,
        history_position=int(args.history_position),
        output_json=args.output_json,
        threshold=args.threshold,
    )
    summary = {
        "output_json": str(args.output_json),
        "history_position": sidecar.get("history_position"),
        "primary_error_at_prefix": sidecar.get("primary_error_at_prefix"),
        "compiled_count_2q_total": sidecar.get("compiled_count_2q_total"),
        "compiled_depth_2q_total": sidecar.get("compiled_depth_2q_total"),
        "compiled_depth_total": sidecar.get("compiled_depth_total"),
        "instrumented_runtime_S": sidecar.get("instrumented_runtime_S"),
        "mechanism_formula_S": sidecar.get("mechanism_formula_S"),
        "replayed_operator_count": sidecar.get("replay", {}).get("replayed_operator_count"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
