#!/usr/bin/env python3
"""Resumable, single-scientific-job FM-SNAKE Pareto campaign ledger.

This module is deliberately route-specific.  It does not share mutable state
with JR-SNAKE and it does not choose scientific settings.  Callers materialize
the complete settings object and an argv vector; the runner records an exact
settings diff before it will execute that vector without a shell.

The Paper-I work coordinate is expanded winning-lineage work.  Objective
evaluations come from ``nfev_total`` after subtracting structurally rolled-back
rounds.  Selector/geometry work comes from the Formal-Manifold unique primitive
ledger.  Reuse lists and matrix-element diagnostics never add charge.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
FORMAL_MANIFOLD_ROUTE = "formal_manifold_warm_start_v1"
CAMPAIGN_SCHEMA = "paper_i_hh_formal_manifold_pareto_campaign_v1"
LEDGER_SCHEMA = "paper_i_hh_formal_manifold_pareto_ledger_v1"
CELL_SCHEMA = "paper_i_hh_formal_manifold_pareto_cell_v1"
QUERY_SIDECAR_SCHEMA = "paper_i_hh_formal_manifold_query_work_sidecar_v1"
OBJECTIVE_AXES = (
    "abs_delta_e_same_cutoff",
    "qiskit_compiled_two_qubit_count",
    "qiskit_compiled_two_qubit_depth",
    "qiskit_compiled_total_depth",
    "expanded_winning_branch_query_work",
)
QUERY_PRIMITIVE_FIELDS = ("N_E", "N_grad", "N_G", "N_Q", "N_Hv", "N_cross")
TERMINAL_STATES = frozenset({"complete", "failed", "interrupted"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_ready(item) for item in value]
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"))


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON payload must be an object: {path}")
    return dict(value)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_json_ready(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _flatten(value: Any, *, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        flattened: dict[str, Any] = {}
        for key in sorted(value, key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(value[key], prefix=child))
        return flattened
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        flattened = {}
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            flattened.update(_flatten(item, prefix=child))
        if not value:
            flattened[prefix] = []
        return flattened
    return {prefix: _json_ready(value)}


def exact_settings_diff(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_flat = _flatten(left)
    right_flat = _flatten(right)
    rows = []
    for path in sorted(set(left_flat) | set(right_flat)):
        left_present = path in left_flat
        right_present = path in right_flat
        left_value = left_flat.get(path)
        right_value = right_flat.get(path)
        if left_present != right_present or left_value != right_value:
            rows.append(
                {
                    "path": path,
                    "left_present": left_present,
                    "right_present": right_present,
                    "left": left_value,
                    "right": right_value,
                }
            )
    return {
        "schema": "paper_i_hh_formal_manifold_exact_settings_diff_v1",
        "left_sha256": payload_sha256(left),
        "right_sha256": payload_sha256(right),
        "changed_field_count": len(rows),
        "changed_paths": [row["path"] for row in rows],
        "changed_fields": rows,
        "execution_settings_excluded": True,
    }


def _walk_key_values(value: Any, key_name: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) == key_name:
                found.append(item)
            found.extend(_walk_key_values(item, key_name))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            found.extend(_walk_key_values(item, key_name))
    return found


def validate_formal_settings(settings: Mapping[str, Any]) -> None:
    values = _walk_key_values(settings, "adapt_reoptimization_route")
    if not values:
        raise ValueError("scientific settings must explicitly name adapt_reoptimization_route")
    if any(str(value) != FORMAL_MANIFOLD_ROUTE for value in values):
        raise ValueError("scientific settings contain a non-FM reoptimization route")


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if not math.isfinite(parsed) or parsed < 0 or not math.isclose(parsed, round(parsed)):
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(round(parsed))


def _finite_float(value: Any, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _adapt_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = payload.get("adapt_vqe")
    return nested if isinstance(nested, Mapping) else payload


def _validate_primitive_telemetry(raw: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} primitive telemetry is missing")
    counts = {
        key: _nonnegative_int(raw.get(key), name=f"{label}.{key}")
        for key in QUERY_PRIMITIVE_FIELDS
    }
    actual = _nonnegative_int(
        raw.get("actual_operator_probe_count"),
        name=f"{label}.actual_operator_probe_count",
    )
    if sum(counts.values()) != actual:
        raise ValueError(f"{label} primitive category sum does not equal actual count")
    reconciliation = raw.get("primitive_count_reconciliation")
    if not isinstance(reconciliation, Mapping) or reconciliation.get("count_equal") is not True:
        raise ValueError(f"{label} primitive reconciliation did not pass")
    return {**counts, "actual_operator_probe_count": actual}


def build_query_work_sidecar(
    *, result_json: Path, output_json: Path | None = None
) -> dict[str, Any]:
    """Build exact FM winning/discarded work from a completed result.

    ``nfev_total`` is the complete objective-evaluation currency.  A rolled-back
    history row is operational overhead, so its step delta is removed from the
    winning lineage.  Query-closure primitives use set cardinality; reuse and
    matrix diagnostics are retained but never added again.
    """

    result_path = result_json.resolve()
    payload = _read_json(result_path)
    adapt = _adapt_payload(payload)
    if adapt.get("success") is not True:
        raise ValueError("FM query sidecar requires a successful result")
    if str(adapt.get("adapt_reoptimization_route")) != FORMAL_MANIFOLD_ROUTE:
        raise ValueError("result does not use the Formal-Manifold route")
    closure = adapt.get("formal_manifold_query_closure")
    if not isinstance(closure, Mapping):
        raise ValueError("result is missing formal_manifold_query_closure")
    if closure.get("joint_response_selector_invoked") is not False:
        raise ValueError("FM result does not prove Joint-Response selector isolation")
    winning_raw = closure.get("winning_branch")
    discarded_raw = closure.get("discarded_branch_operational_overhead")
    winning_primitives = _validate_primitive_telemetry(winning_raw, label="winning")
    discarded_primitives = _validate_primitive_telemetry(discarded_raw, label="discarded")

    nfev_total = _nonnegative_int(adapt.get("nfev_total"), name="nfev_total")
    history_raw = adapt.get("history", [])
    if not isinstance(history_raw, Sequence) or isinstance(history_raw, (str, bytes)):
        raise ValueError("adapt history must be an array")
    for row in history_raw:
        if not isinstance(row, Mapping):
            raise ValueError("adapt history contains a non-object row")
    discarded_objective = 0
    winning_objective = nfev_total
    winning_selector = winning_primitives["actual_operator_probe_count"]
    discarded_selector = discarded_primitives["actual_operator_probe_count"]
    winning_total = winning_objective + winning_selector
    discarded_total = discarded_objective + discarded_selector

    sidecar = {
        "schema": QUERY_SIDECAR_SCHEMA,
        "status": "complete",
        "route": FORMAL_MANIFOLD_ROUTE,
        "source_result_json": str(result_path),
        "source_result_sha256": file_sha256(result_path),
        "query_unit": "unique_logical_estimator_primitive_or_objective_evaluation_v1",
        "winning_branch": {
            "scope": "accepted_terminal_lineage",
            "objective_evaluations": winning_objective,
            "selector_unique_logical_primitives": winning_selector,
            "selector_primitive_components": winning_primitives,
            "expanded_query_work": winning_total,
            "nfev_includes": [
                "outer_hamiltonian_evaluations",
                "optimizer_and_refit_nfev",
                "warm_start_objective_guards",
                "boundary_refits",
                "final_refits_when_retained",
                "prune_trials_on_the_retained_lineage",
            ],
        },
        "discarded_branch_operational_overhead": {
            "scope": "discarded_selector_branches_only",
            "objective_evaluations": discarded_objective,
            "selector_unique_logical_primitives": discarded_selector,
            "selector_primitive_components": discarded_primitives,
            "expanded_query_work": discarded_total,
        },
        "all_executed_work_reconciliation": {
            "objective_evaluations": nfev_total,
            "selector_unique_logical_primitives": winning_selector + discarded_selector,
            "expanded_query_work": nfev_total + winning_selector + discarded_selector,
            "winning_plus_discarded_equal": (
                winning_total + discarded_total
                == nfev_total + winning_selector + discarded_selector
            ),
        },
        "reuse_charge_policy": "primitive_id_union_charge_once_v1",
        "matrix_element_diagnostics_charge": 0,
        "classical_algebra_charge": 0,
        "joint_response_selector_invoked": False,
        "source_query_closure": _json_ready(closure),
    }
    if output_json is not None:
        _atomic_write_json(output_json, sidecar)
    return sidecar


def _same_cutoff_contract(n_ph_work: int) -> dict[str, Any]:
    return {
        "schema": "paper_i_same_cutoff_v1",
        "variational_n_ph": n_ph_work,
        "exact_n_ph": n_ph_work,
        "same_cutoff": True,
        "higher_cutoff_diagnostic_required": False,
    }


def _normalize_regimes(regimes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in regimes:
        row = dict(raw)
        regime_id = str(row.get("id") or "")
        if not regime_id or regime_id in seen:
            raise ValueError("regime ids must be nonempty and unique")
        seen.add(regime_id)
        for required in ("exact_energy_same_cutoff", "n_ph_work"):
            if row.get(required) is None:
                raise ValueError(f"regime {regime_id} is missing {required}")
        row["exact_energy_same_cutoff"] = _finite_float(
            row["exact_energy_same_cutoff"], name=f"{regime_id}.exact_energy_same_cutoff"
        )
        row["n_ph_work"] = _nonnegative_int(row["n_ph_work"], name=f"{regime_id}.n_ph_work")
        legacy_ref = row.pop("n_ph_ref", None)
        if legacy_ref is not None:
            legacy_ref_value = _nonnegative_int(
                legacy_ref, name=f"{regime_id}.n_ph_ref"
            )
            if legacy_ref_value != row["n_ph_work"]:
                raise ValueError(
                    f"regime {regime_id} violates the same-cutoff contract: "
                    f"n_ph_ref={legacy_ref_value} != n_ph_work={row['n_ph_work']}"
                )
        row["cutoff_contract"] = _same_cutoff_contract(row["n_ph_work"])
        out.append(row)
    if not out:
        raise ValueError("campaign requires at least one provenance-locked regime")
    return out


def initialize_campaign(
    *,
    campaign_dir: Path,
    campaign_id: str,
    baseline_settings: Mapping[str, Any],
    regimes: Sequence[Mapping[str, Any]],
    compile_contract: Mapping[str, Any],
    baseline_source: str,
) -> dict[str, Any]:
    campaign_dir = campaign_dir.resolve()
    if (campaign_dir / "campaign_manifest.json").exists():
        raise FileExistsError(f"campaign already exists: {campaign_dir}")
    validate_formal_settings(baseline_settings)
    normalized_regimes = _normalize_regimes(regimes)
    manifest = {
        "schema": CAMPAIGN_SCHEMA,
        "campaign_id": str(campaign_id),
        "created_utc": _utc_now(),
        "route": FORMAL_MANIFOLD_ROUTE,
        "route_mutable_state_isolation": "fm_snake_only_no_jr_mutable_state_v1",
        "scientific_jobs_max": 1,
        "objective_axes_ordered": list(OBJECTIVE_AXES),
        "query_coordinate": "expanded_winning_branch_query_work_v1",
        "discarded_branch_policy": "separate_operational_overhead",
        "wall_time_is_scientific_axis": False,
        "baseline_source": str(baseline_source),
        "baseline_scientific_settings": _json_ready(baseline_settings),
        "baseline_scientific_settings_sha256": payload_sha256(baseline_settings),
        "regimes": normalized_regimes,
        "qiskit_compile_contract": _json_ready(compile_contract),
    }
    ledger = {
        "schema": LEDGER_SCHEMA,
        "campaign_id": str(campaign_id),
        "updated_utc": manifest["created_utc"],
        "objective_axes_ordered": list(OBJECTIVE_AXES),
        "cells": [],
        "pareto_front_by_regime": {},
        "complete_transferable_policy_front": [],
        "incomplete_or_ambiguous_policies": {},
    }
    _atomic_write_json(campaign_dir / "campaign_manifest.json", manifest)
    _write_ledger(campaign_dir, ledger)
    return manifest


def _manifest(campaign_dir: Path) -> dict[str, Any]:
    payload = _read_json(campaign_dir / "campaign_manifest.json")
    if payload.get("schema") != CAMPAIGN_SCHEMA:
        raise ValueError("unsupported FM campaign manifest schema")
    return payload


def _ledger(campaign_dir: Path) -> dict[str, Any]:
    payload = _read_json(campaign_dir / "pareto_ledger.json")
    if payload.get("schema") != LEDGER_SCHEMA:
        raise ValueError("unsupported FM Pareto ledger schema")
    return payload


def _render_ledger_markdown(ledger: Mapping[str, Any]) -> str:
    lines = [
        "# FM-SNAKE Paper-I Hubbard--Holstein Pareto ledger",
        "",
        "Candidate evidence only; this ledger does not promote settings or edit the manuscript.",
        "",
        "| Cell | Policy | Regime | Status | abs(Delta E) | N2q | D2q | Dc | Winning query work | Discarded overhead |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in ledger.get("cells", []):
        evidence = cell.get("evidence") if isinstance(cell, Mapping) else None
        evidence = evidence if isinstance(evidence, Mapping) else {}
        coordinates = evidence.get("pareto_coordinates")
        coordinates = coordinates if isinstance(coordinates, Mapping) else {}
        discarded = evidence.get("discarded_branch_operational_overhead")
        discarded = discarded if isinstance(discarded, Mapping) else {}
        def show(key: str) -> str:
            value = coordinates.get(key)
            return "--" if value is None else str(value)
        lines.append(
            "| "
            + " | ".join(
                (
                    str(cell.get("cell_id", "")),
                    str(cell.get("policy_id", "")),
                    str(cell.get("regime", "")),
                    str(cell.get("status", "")),
                    show(OBJECTIVE_AXES[0]),
                    show(OBJECTIVE_AXES[1]),
                    show(OBJECTIVE_AXES[2]),
                    show(OBJECTIVE_AXES[3]),
                    show(OBJECTIVE_AXES[4]),
                    str(discarded.get("expanded_query_work", "--")),
                )
            )
            + " |"
        )
    lines.extend(("", "Primary fronts are stored in `pareto_ledger.json`.", ""))
    return "\n".join(lines)


def _write_ledger(campaign_dir: Path, ledger: dict[str, Any]) -> None:
    ledger["updated_utc"] = _utc_now()
    _atomic_write_json(campaign_dir / "pareto_ledger.json", ledger)
    _atomic_write_text(campaign_dir / "pareto_ledger.md", _render_ledger_markdown(ledger))


def _cell_by_id(ledger: Mapping[str, Any], cell_id: str) -> dict[str, Any]:
    matches = [row for row in ledger.get("cells", []) if row.get("cell_id") == cell_id]
    if len(matches) != 1:
        raise KeyError(f"expected one ledger cell named {cell_id!r}, found {len(matches)}")
    return matches[0]


def plan_cell(
    *,
    campaign_dir: Path,
    cell_id: str,
    policy_id: str,
    regime: str,
    mechanism_family: str,
    scientific_settings: Mapping[str, Any],
    command: Sequence[str],
    approved_changed_paths: Sequence[str],
    parent_cell_id: str | None = None,
    resume_source_json: Path | None = None,
    auto_qiskit: bool = False,
) -> dict[str, Any]:
    campaign_dir = campaign_dir.resolve()
    manifest = _manifest(campaign_dir)
    ledger = _ledger(campaign_dir)
    if any(row.get("cell_id") == cell_id for row in ledger["cells"]):
        raise ValueError(f"duplicate cell id: {cell_id}")
    if str(regime) not in {str(row["id"]) for row in manifest["regimes"]}:
        raise ValueError(f"unknown campaign regime: {regime}")
    validate_formal_settings(scientific_settings)
    if not mechanism_family:
        raise ValueError("mechanism_family must be explicit")
    if not command or any(not isinstance(value, str) or not value for value in command):
        raise ValueError("command must be a nonempty argv string array")
    if parent_cell_id is None:
        left_settings = manifest["baseline_scientific_settings"]
        diff_base = "campaign_baseline"
    else:
        parent = _cell_by_id(ledger, parent_cell_id)
        left_settings = parent["scientific_settings"]
        diff_base = parent_cell_id
    diff = exact_settings_diff(left_settings, scientific_settings)
    approved = {str(path) for path in approved_changed_paths}
    unapproved = sorted(set(diff["changed_paths"]) - approved)
    if unapproved:
        raise ValueError("unapproved scientific setting changes: " + ", ".join(unapproved))
    cell_dir = campaign_dir / "cells" / str(cell_id)
    if resume_source_json is not None:
        resume_source = resume_source_json.resolve()
        if not resume_source.is_file():
            raise FileNotFoundError(resume_source)
        resume = {"path": str(resume_source), "sha256": file_sha256(resume_source)}
    else:
        resume = None
    cell = {
        "schema": CELL_SCHEMA,
        "cell_id": str(cell_id),
        "policy_id": str(policy_id),
        "regime": str(regime),
        "mechanism_family": str(mechanism_family),
        "status": "queued",
        "planned_utc": _utc_now(),
        "parent_cell_id": parent_cell_id,
        "settings_diff_base": diff_base,
        "approved_changed_paths": sorted(approved),
        "settings_diff": diff,
        "scientific_settings": _json_ready(scientific_settings),
        "scientific_settings_sha256": payload_sha256(scientific_settings),
        "command_argv_template": list(command),
        "resume_source": resume,
        "auto_qiskit": bool(auto_qiskit),
        "paths": {
            "cell_dir": str(cell_dir),
            "result_json": str(cell_dir / "result.json"),
            "current_json": str(cell_dir / "current.json"),
            "query_work_sidecar": str(cell_dir / "query_work_sidecar.json"),
            "qiskit_sidecar": str(cell_dir / "qiskit_sidecar.json"),
            "stdout_log": str(cell_dir / "stdout.log"),
            "stderr_log": str(cell_dir / "stderr.log"),
        },
        "attempts": [],
        "evidence": None,
    }
    cell_dir.mkdir(parents=True, exist_ok=False)
    _atomic_write_json(cell_dir / "cell_plan.json", cell)
    _atomic_write_json(cell_dir / "settings_diff.json", diff)
    ledger["cells"].append(cell)
    _write_ledger(campaign_dir, ledger)
    return cell


def _objective_tuple(evidence: Mapping[str, Any]) -> tuple[float, ...]:
    coordinates = evidence.get("pareto_coordinates")
    if not isinstance(coordinates, Mapping):
        raise ValueError("evidence has no Pareto coordinates")
    return tuple(float(coordinates[key]) for key in OBJECTIVE_AXES)


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    a = _objective_tuple(left)
    b = _objective_tuple(right)
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def _recompute_fronts(manifest: Mapping[str, Any], ledger: dict[str, Any]) -> None:
    completed = [
        row for row in ledger["cells"]
        if row.get("status") == "complete" and isinstance(row.get("evidence"), Mapping)
    ]
    by_regime: dict[str, list[dict[str, Any]]] = {}
    for row in completed:
        by_regime.setdefault(str(row["regime"]), []).append(row)
    ledger["pareto_front_by_regime"] = {
        regime: [
            row["cell_id"] for row in rows
            if not any(
                other["cell_id"] != row["cell_id"]
                and _dominates(other["evidence"], row["evidence"])
                for other in rows
            )
        ]
        for regime, rows in sorted(by_regime.items())
    }
    required_regimes = [str(row["id"]) for row in manifest["regimes"]]
    policies: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in completed:
        policies.setdefault(str(row["policy_id"]), {}).setdefault(str(row["regime"]), []).append(row)
    valid: dict[str, list[dict[str, Any]]] = {}
    invalid: dict[str, Any] = {}
    for policy, per_regime in sorted(policies.items()):
        missing = [name for name in required_regimes if name not in per_regime]
        ambiguous = {name: len(rows) for name, rows in per_regime.items() if len(rows) != 1}
        if missing or ambiguous:
            invalid[policy] = {"missing_regimes": missing, "ambiguous_regimes": ambiguous}
        else:
            valid[policy] = [per_regime[name][0] for name in required_regimes]

    def policy_dominates(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
        a = tuple(value for row in left for value in _objective_tuple(row["evidence"]))
        b = tuple(value for row in right for value in _objective_tuple(row["evidence"]))
        return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

    ledger["complete_transferable_policy_front"] = [
        policy for policy, rows in valid.items()
        if not any(other != policy and policy_dominates(other_rows, rows) for other, other_rows in valid.items())
    ]
    ledger["incomplete_or_ambiguous_policies"] = invalid


def _qiskit_value(sidecar: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    for key in aliases:
        if sidecar.get(key) is not None:
            return sidecar[key]
    return None


def validate_qiskit_sidecar(
    sidecar: Mapping[str, Any], *, compile_contract: Mapping[str, Any]
) -> dict[str, int]:
    if sidecar.get("compiled_resource_qiskit_validated") is not True:
        raise ValueError("Qiskit sidecar is not validated")
    if str(sidecar.get("compiled_circuit_stats_status")) != "ok":
        raise ValueError("Qiskit sidecar compile status is not ok")
    contract_aliases = {
        "compile_convention": ("compile_convention",),
        "optimization_level": ("qiskit_transpile_optimization_level", "optimization_level"),
        "seed_transpiler": ("qiskit_transpile_seed", "seed_transpiler"),
        "backend_name": ("qiskit_backend_name", "backend_name", "compiled_backend_name"),
    }
    for contract_key, aliases in contract_aliases.items():
        expected = compile_contract.get(contract_key)
        if expected is None:
            continue
        actual = _qiskit_value(sidecar, aliases)
        if actual != expected:
            raise ValueError(f"Qiskit compile contract mismatch for {contract_key}: {actual!r} != {expected!r}")
    return {
        "N2q": _nonnegative_int(sidecar.get("compiled_count_2q_total"), name="compiled_count_2q_total"),
        "D2q": _nonnegative_int(sidecar.get("compiled_depth_2q_total"), name="compiled_depth_2q_total"),
        "Dc": _nonnegative_int(sidecar.get("compiled_depth_total"), name="compiled_depth_total"),
    }


def compile_cell_qiskit(*, campaign_dir: Path, cell_id: str) -> dict[str, Any]:
    ledger = _ledger(campaign_dir)
    cell = _cell_by_id(ledger, cell_id)
    result_json = Path(cell["paths"]["result_json"])
    payload = _read_json(result_json)
    history = _adapt_payload(payload).get("history")
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)) or not history:
        raise ValueError("cannot compile a result without terminal history")
    from pipelines.reporting.build_paper_i_selected_prefix_qiskit_sidecar import build_sidecar

    return build_sidecar(
        result_json=result_json,
        history_position=len(history),
        output_json=Path(cell["paths"]["qiskit_sidecar"]),
        threshold=None,
    )


def ingest_cell(*, campaign_dir: Path, cell_id: str) -> dict[str, Any]:
    campaign_dir = campaign_dir.resolve()
    manifest = _manifest(campaign_dir)
    ledger = _ledger(campaign_dir)
    cell = _cell_by_id(ledger, cell_id)
    result_path = Path(cell["paths"]["result_json"])
    qiskit_path = Path(cell["paths"]["qiskit_sidecar"])
    query_path = Path(cell["paths"]["query_work_sidecar"])
    result = _read_json(result_path)
    adapt = _adapt_payload(result)
    if adapt.get("success") is not True:
        raise ValueError("cannot ingest a failed result")
    if str(adapt.get("adapt_reoptimization_route")) != FORMAL_MANIFOLD_ROUTE:
        raise ValueError("cannot ingest a non-FM result")
    embedded_hash = result.get("scientific_settings_hash")
    if embedded_hash is not None and str(embedded_hash) != str(cell["scientific_settings_sha256"]):
        raise ValueError("result scientific settings hash does not match planned cell")
    if not query_path.is_file():
        build_query_work_sidecar(result_json=result_path, output_json=query_path)
    query = _read_json(query_path)
    if query.get("source_result_sha256") != file_sha256(result_path):
        raise ValueError("query sidecar source hash mismatch")
    if not qiskit_path.is_file():
        raise FileNotFoundError(f"Qiskit sidecar required before ingest: {qiskit_path}")
    qiskit = _read_json(qiskit_path)
    qiskit_counts = validate_qiskit_sidecar(
        qiskit, compile_contract=manifest.get("qiskit_compile_contract", {})
    )
    if int(qiskit.get("history_position", -1)) != len(adapt.get("history", [])):
        raise ValueError("Qiskit sidecar is not aligned to the terminal FM prefix")
    regimes = {str(row["id"]): row for row in manifest["regimes"]}
    regime = regimes[str(cell["regime"])]
    energy = _finite_float(adapt.get("energy"), name="adapt energy")
    n_ph_work = _nonnegative_int(regime["n_ph_work"], name="regime n_ph_work")
    exact = _finite_float(regime["exact_energy_same_cutoff"], name="exact same-cutoff energy")
    abs_error = abs(energy - exact)
    reported_error = adapt.get("abs_delta_e")
    if reported_error is not None and not math.isclose(
        abs_error, float(reported_error), rel_tol=1e-10, abs_tol=1e-12
    ):
        raise ValueError("result error disagrees with provenance-locked same-cutoff exact energy")
    winning = query["winning_branch"]
    discarded = query["discarded_branch_operational_overhead"]
    coordinates = {
        OBJECTIVE_AXES[0]: abs_error,
        OBJECTIVE_AXES[1]: qiskit_counts["N2q"],
        OBJECTIVE_AXES[2]: qiskit_counts["D2q"],
        OBJECTIVE_AXES[3]: qiskit_counts["Dc"],
        OBJECTIVE_AXES[4]: _nonnegative_int(
            winning.get("expanded_query_work"), name="expanded winning query work"
        ),
    }
    warm = adapt.get("formal_manifold_warm_state_checkpoint")
    warm = dict(warm) if isinstance(warm, Mapping) else {}
    evidence = {
        "schema": "paper_i_hh_formal_manifold_pareto_evidence_v1",
        "pareto_coordinate_order": list(OBJECTIVE_AXES),
        "pareto_coordinates": coordinates,
        "energy": energy,
        "exact_energy_same_cutoff": exact,
        "n_ph_work": n_ph_work,
        # Historical active manifests may still carry an unused n_ph_ref field.
        # Evidence is always normalized to the single same-cutoff contract.
        "cutoff_contract": _same_cutoff_contract(n_ph_work),
        "winning_branch_query_work": winning,
        "discarded_branch_operational_overhead": discarded,
        "result_json": str(result_path),
        "result_sha256": file_sha256(result_path),
        "query_work_sidecar": str(query_path),
        "query_work_sidecar_sha256": file_sha256(query_path),
        "qiskit_sidecar": str(qiskit_path),
        "qiskit_sidecar_sha256": file_sha256(qiskit_path),
        "whitening_provenance": {
            key: warm.get(key)
            for key in (
                "whitening_id", "frame_id", "logical_range_id",
                "curvature_whitening_id", "curvature_frame_id",
                "qbroyd_whitening_id", "qbroyd_logical_range_id",
            )
        },
        "route_separation": {"joint_response_selector_invoked": False},
        "wall_time_diagnostic": result.get("elapsed_s") or adapt.get("elapsed_s"),
    }
    cell["evidence"] = evidence
    cell["status"] = "complete"
    cell["completed_utc"] = _utc_now()
    _recompute_fronts(manifest, ledger)
    _write_ledger(campaign_dir, ledger)
    return evidence


def _render_command(template: Sequence[str], cell: Mapping[str, Any]) -> list[str]:
    values = {
        "repo_root": str(REPO_ROOT),
        **{key: str(value) for key, value in cell["paths"].items()},
    }
    return [str(token).format(**values) for token in template]


class _CampaignLock:
    def __init__(self, campaign_dir: Path, *, cell_id: str):
        self.path = campaign_dir / ".scientific_job.lock"
        self.cell_id = cell_id

    def __enter__(self) -> "_CampaignLock":
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise RuntimeError(f"FM campaign already owns a scientific-job lock: {self.path}") from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "cell_id": self.cell_id, "created_utc": _utc_now()}, handle)
        return self

    def __exit__(self, *_: Any) -> None:
        self.path.unlink(missing_ok=True)


def run_next(*, campaign_dir: Path, cell_id: str | None = None) -> dict[str, Any]:
    """Synchronously execute exactly one queued scientific argv vector."""

    campaign_dir = campaign_dir.resolve()
    ledger = _ledger(campaign_dir)
    if any(row.get("status") == "running" for row in ledger["cells"]):
        raise RuntimeError("ledger already contains a running FM scientific cell")
    queued = [row for row in ledger["cells"] if row.get("status") == "queued"]
    if cell_id is not None:
        queued = [row for row in queued if row.get("cell_id") == cell_id]
    if not queued:
        raise RuntimeError("no matching queued FM cell")
    cell = queued[0]
    command = _render_command(cell["command_argv_template"], cell)
    with _CampaignLock(campaign_dir, cell_id=str(cell["cell_id"])):
        cell["status"] = "running"
        attempt = {
            "attempt": len(cell["attempts"]) + 1,
            "started_utc": _utc_now(),
            "command_argv": command,
            "scientific_settings_sha256": cell["scientific_settings_sha256"],
        }
        cell["attempts"].append(attempt)
        _write_ledger(campaign_dir, ledger)
        stdout_path = Path(cell["paths"]["stdout_log"])
        stderr_path = Path(cell["paths"]["stderr_log"])
        with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
            completed = subprocess.run(command, cwd=REPO_ROOT, stdout=stdout, stderr=stderr, check=False)
        attempt["finished_utc"] = _utc_now()
        attempt["returncode"] = int(completed.returncode)
        if completed.returncode != 0:
            cell["status"] = "failed"
            cell["failure_phase"] = "scientific_command"
            _write_ledger(campaign_dir, ledger)
            return {"status": "failed", "cell_id": cell["cell_id"], "returncode": completed.returncode}
        result_path = Path(cell["paths"]["result_json"])
        if not result_path.is_file():
            cell["status"] = "failed"
            cell["failure_phase"] = "scientific_result_missing"
            _write_ledger(campaign_dir, ledger)
            return {"status": "failed", "cell_id": cell["cell_id"], "reason": "result_missing"}
        build_query_work_sidecar(
            result_json=result_path,
            output_json=Path(cell["paths"]["query_work_sidecar"]),
        )
        cell["status"] = "awaiting_qiskit"
        _write_ledger(campaign_dir, ledger)
    if cell.get("auto_qiskit"):
        compile_cell_qiskit(campaign_dir=campaign_dir, cell_id=str(cell["cell_id"]))
        ingest_cell(campaign_dir=campaign_dir, cell_id=str(cell["cell_id"]))
        return {"status": "complete", "cell_id": cell["cell_id"]}
    return {"status": "awaiting_qiskit", "cell_id": cell["cell_id"]}


def retry_identical_cell(*, campaign_dir: Path, cell_id: str) -> None:
    ledger = _ledger(campaign_dir)
    cell = _cell_by_id(ledger, cell_id)
    if cell.get("status") not in {"failed", "interrupted"}:
        raise ValueError("only failed or interrupted cells may be retried")
    cell["status"] = "queued"
    cell["retry_policy"] = "identical_scientific_settings_and_command_v1"
    _write_ledger(campaign_dir, ledger)


def recover_interrupted_campaign(*, campaign_dir: Path, requeue: bool = False) -> dict[str, Any]:
    """Recover a lock left by a dead runner without changing cell science."""

    campaign_dir = campaign_dir.resolve()
    lock_path = campaign_dir / ".scientific_job.lock"
    if not lock_path.is_file():
        raise FileNotFoundError(f"campaign has no interrupted-job lock: {lock_path}")
    lock = _read_json(lock_path)
    pid = _nonnegative_int(lock.get("pid"), name="lock.pid")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        alive = False
    except PermissionError:
        alive = True
    else:
        alive = True
    if alive:
        raise RuntimeError(f"refusing recovery while runner pid {pid} is alive")
    ledger = _ledger(campaign_dir)
    cell = _cell_by_id(ledger, str(lock.get("cell_id") or ""))
    if cell.get("status") != "running":
        raise ValueError("stale lock does not point to a ledger cell in running state")
    cell["status"] = "queued" if requeue else "interrupted"
    cell["interrupted_recovery"] = {
        "recovered_utc": _utc_now(),
        "dead_runner_pid": pid,
        "requeued_identical": bool(requeue),
        "scientific_settings_sha256": cell["scientific_settings_sha256"],
        "command_argv_template": list(cell["command_argv_template"]),
    }
    lock_path.unlink()
    _write_ledger(campaign_dir, ledger)
    return {"status": cell["status"], "cell_id": cell["cell_id"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--campaign-dir", type=Path, required=True)
    init.add_argument("--campaign-id", required=True)
    init.add_argument("--baseline-settings-json", type=Path, required=True)
    init.add_argument("--regimes-json", type=Path, required=True)
    init.add_argument("--compile-contract-json", type=Path, required=True)
    init.add_argument("--baseline-source", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--campaign-dir", type=Path, required=True)
    plan.add_argument("--cell-id", required=True)
    plan.add_argument("--policy-id", required=True)
    plan.add_argument("--regime", required=True)
    plan.add_argument("--mechanism-family", required=True)
    plan.add_argument("--scientific-settings-json", type=Path, required=True)
    plan.add_argument("--command-json", type=Path, required=True)
    plan.add_argument("--approved-changed-paths-json", type=Path, required=True)
    plan.add_argument("--parent-cell-id")
    plan.add_argument("--resume-source-json", type=Path)
    plan.add_argument("--auto-qiskit", action="store_true")
    run = sub.add_parser("run-next")
    run.add_argument("--campaign-dir", type=Path, required=True)
    run.add_argument("--cell-id")
    ingest = sub.add_parser("ingest")
    ingest.add_argument("--campaign-dir", type=Path, required=True)
    ingest.add_argument("--cell-id", required=True)
    compile_parser = sub.add_parser("compile-cell")
    compile_parser.add_argument("--campaign-dir", type=Path, required=True)
    compile_parser.add_argument("--cell-id", required=True)
    query = sub.add_parser("build-query-sidecar")
    query.add_argument("--result-json", type=Path, required=True)
    query.add_argument("--output-json", type=Path, required=True)
    retry = sub.add_parser("retry-identical")
    retry.add_argument("--campaign-dir", type=Path, required=True)
    retry.add_argument("--cell-id", required=True)
    recover = sub.add_parser("recover-interrupted")
    recover.add_argument("--campaign-dir", type=Path, required=True)
    recover.add_argument("--requeue", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        regimes_payload = json.loads(args.regimes_json.read_text(encoding="utf-8"))
        regimes = regimes_payload.get("regimes") if isinstance(regimes_payload, Mapping) else regimes_payload
        payload = initialize_campaign(
            campaign_dir=args.campaign_dir,
            campaign_id=args.campaign_id,
            baseline_settings=_read_json(args.baseline_settings_json),
            regimes=regimes,
            compile_contract=_read_json(args.compile_contract_json),
            baseline_source=args.baseline_source,
        )
    elif args.command == "plan":
        command = json.loads(args.command_json.read_text(encoding="utf-8"))
        approved = json.loads(args.approved_changed_paths_json.read_text(encoding="utf-8"))
        payload = plan_cell(
            campaign_dir=args.campaign_dir,
            cell_id=args.cell_id,
            policy_id=args.policy_id,
            regime=args.regime,
            mechanism_family=args.mechanism_family,
            scientific_settings=_read_json(args.scientific_settings_json),
            command=command,
            approved_changed_paths=approved,
            parent_cell_id=args.parent_cell_id,
            resume_source_json=args.resume_source_json,
            auto_qiskit=args.auto_qiskit,
        )
    elif args.command == "run-next":
        payload = run_next(campaign_dir=args.campaign_dir, cell_id=args.cell_id)
    elif args.command == "ingest":
        payload = ingest_cell(campaign_dir=args.campaign_dir, cell_id=args.cell_id)
    elif args.command == "compile-cell":
        payload = compile_cell_qiskit(campaign_dir=args.campaign_dir, cell_id=args.cell_id)
    elif args.command == "build-query-sidecar":
        payload = build_query_work_sidecar(result_json=args.result_json, output_json=args.output_json)
    elif args.command == "retry-identical":
        retry_identical_cell(campaign_dir=args.campaign_dir, cell_id=args.cell_id)
        payload = {"status": "queued", "cell_id": args.cell_id}
    else:
        payload = recover_interrupted_campaign(
            campaign_dir=args.campaign_dir,
            requeue=bool(args.requeue),
        )
    print(json.dumps(_json_ready(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
