#!/usr/bin/env python3
"""Repair the weak-strong SNAKE HH qiskit/source-map row to prefix semantics.

This is a diagnostic/support repair only.  It does not edit the Paper-I
manuscript.  The old row used the source-stopped terminal SNAKE value for the
weak-strong qiskit/table marker even though the displayed coordinate is
``k_pl=17``.  This utility replaces that diagnostic row with the value and
compiled resources reconstructed from the source history prefix at ``k_pl`` and
keeps the old terminal values under an explicit repair audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.reporting.build_paper_i_hh_snake_table_source_overlay_diagnostic import (
    _compile_history_prefix,
    _current_plot_plateau_costs,
    _current_replay_override_path,
    _extract_snake_table_source_trajectory,
    _forced_replay_prefix_audit,
    _payload_and_history,
    _read_json,
    _rel,
    _resolve,
    _sha256,
)


REGIME = "weak-strong"
METHOD = "SNAKE"
REPAIR_SCHEMA = "paper_i_hh_weak_strong_snake_prefix_source_repair_v1"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fmt_delta(value: float) -> str:
    text = f"{float(value):.2e}"
    return text.replace("e-0", "e-").replace("e+0", "e+")


def _find_row(rows: Sequence[MutableMapping[str, Any]], *, regime: str, method: str) -> MutableMapping[str, Any]:
    for row in rows:
        if str(row.get("regime")) == regime and str(row.get("method")) == method:
            return row
    raise KeyError(f"missing row for {regime} / {method}")


def _nested_original(row: Mapping[str, Any], key: str) -> Any:
    repair = row.get("prefix_source_repair") or row.get("prefix_repair")
    if isinstance(repair, Mapping) and key in repair:
        return deepcopy(repair[key])
    return None


def _qiskit_original_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
    previous = _nested_original(row, "original_qiskit_row")
    if previous is not None:
        return previous
    return {
        "same_cutoff_abs_delta_e_at_k_pl": row.get("same_cutoff_abs_delta_e_at_k_pl"),
        "display_target": deepcopy(row.get("display_target")),
        "replayed_qiskit": deepcopy(row.get("replayed_qiskit")),
        "compile_policy": row.get("compile_policy"),
        "selected_pauli_source": row.get("selected_pauli_source"),
        "plateau_status": row.get("plateau_status"),
        "S_not_recomputed_here": row.get("S_not_recomputed_here"),
    }


def _plateau_original_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
    previous = _nested_original(row, "original_plateau_row")
    if previous is not None:
        return previous
    return {
        "same_cutoff_abs_delta_e_at_k_pl": row.get("same_cutoff_abs_delta_e_at_k_pl"),
        "same_cutoff_abs_delta_e_at_k_pl_from_support_trajectory": row.get(
            "same_cutoff_abs_delta_e_at_k_pl_from_support_trajectory"
        ),
        "compiled": deepcopy(row.get("compiled")),
        "display": deepcopy(row.get("display")),
        "S_at_k_pl": row.get("S_at_k_pl"),
        "terminal_S": row.get("terminal_S"),
        "terminal_same_cutoff_abs_delta_e": row.get("terminal_same_cutoff_abs_delta_e"),
        "plateau_status": row.get("plateau_status"),
    }


def _fidelity_original_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
    previous = _nested_original(row, "original_fidelity_row")
    if previous is not None:
        return previous
    return {
        "marker_same_cutoff_abs_delta_e": row.get("marker_same_cutoff_abs_delta_e"),
        "display": deepcopy(row.get("display")),
        "one_minus_fidelity": row.get("one_minus_fidelity"),
        "one_minus_F_display": row.get("one_minus_F_display"),
        "fidelity_status": row.get("fidelity_status"),
        "fidelity_source": row.get("fidelity_source"),
        "terminal_same_cutoff_abs_delta_e": row.get("terminal_same_cutoff_abs_delta_e"),
    }


def _compiled_resources(compiled: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "N1q": compiled.get("compiled_count_1q_total"),
        "N2q": compiled.get("compiled_count_2q_total"),
        "D2q": compiled.get("compiled_depth_2q_total"),
        "Dc": compiled.get("compiled_depth_total"),
        "runtime_rotation_count": compiled.get("runtime_rotation_count"),
        "compile_status": compiled.get("compiled_circuit_stats_status"),
        "compile_convention": compiled.get("compile_convention"),
        "qiskit_version": compiled.get("qiskit_version"),
        "compiled_resource_source_kind": compiled.get("compiled_resource_source_kind"),
    }


def _resource_alignment(
    *,
    old_display: Mapping[str, Any] | None,
    prefix_resources: Mapping[str, Any],
) -> dict[str, Any]:
    old_display = old_display if isinstance(old_display, Mapping) else {}
    keys = ("N2q", "D2q", "Dc", "S")
    rows: dict[str, Any] = {}
    for key in keys:
        old = old_display.get(key)
        new = prefix_resources.get(key)
        rows[key] = {
            "old": old,
            "prefix": new,
            "matches_prefix": old == new,
        }
    return rows


def _repair_csv(
    *,
    csv_path: Path,
    prefix_delta_e: float,
    prefix_resources: Mapping[str, Any],
    s_status: str,
) -> None:
    if not csv_path.exists():
        return
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for row in rows:
        if row.get("regime") == REGIME and row.get("method") == METHOD:
            row["marker_same_cutoff_abs_delta_e"] = repr(float(prefix_delta_e))
            row["N2q"] = "" if prefix_resources.get("N2q") is None else str(int(prefix_resources["N2q"]))
            row["D2q"] = "" if prefix_resources.get("D2q") is None else str(int(prefix_resources["D2q"]))
            row["Dc"] = "" if prefix_resources.get("Dc") is None else str(int(prefix_resources["Dc"]))
            row["S"] = ""
            row["fidelity_status"] = "terminal_fidelity_retained_prefix_state_unavailable"
            row["fidelity_source"] = (
                "terminal source-stopped SNAKE fidelity retained; prefix-state "
                f"fidelity unavailable ({s_status})"
            )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def repair(
    *,
    qiskit_json: Path,
    plateau_json: Path,
    fidelity_json: Path,
    fidelity_csv: Path,
    current_replay_root: Path | None,
    output_audit_json: Path,
    dry_run: bool,
) -> dict[str, Any]:
    qiskit_payload = _read_json(qiskit_json)
    plateau_payload = _read_json(plateau_json)
    fidelity_payload = _read_json(fidelity_json)
    qrow = _find_row(qiskit_payload["rows"], regime=REGIME, method=METHOD)
    plateau_row = _find_row(plateau_payload["rows"], regime=REGIME, method=METHOD)
    fidelity_row = _find_row(fidelity_payload["rows"], regime=REGIME, method=METHOD)

    source_json = _resolve(qrow["source_json"])
    source_payload = _read_json(source_json)
    _payload, history = _payload_and_history(source_payload)
    k_pl = int(round(float(qrow["k_pl"])))
    source_points = _extract_snake_table_source_trajectory(source_json, max_x=30)
    source_by_k = {int(k): float(v) for k, v in source_points}
    if k_pl not in source_by_k:
        raise RuntimeError(f"{source_json} has no SNAKE history point at k_pl={k_pl}")
    prefix_delta_e = float(source_by_k[k_pl])
    compiled = _compile_history_prefix(source_payload, history, k_pl)
    compiled_resources = _compiled_resources(compiled)

    override_source = _current_replay_override_path(current_replay_root, REGIME) if current_replay_root else None
    forced_points = (
        _extract_snake_table_source_trajectory(override_source, max_x=30)
        if override_source is not None
        else source_points
    )
    forced_prefix_audit = _forced_replay_prefix_audit(
        source_points=source_points,
        forced_points=forced_points,
    )
    prefix_costs = _current_plot_plateau_costs(source_json, k_pl)
    s_status = str(prefix_costs.get("S_status") or "unavailable")
    prefix_resources = {
        "N1q": compiled_resources.get("N1q"),
        "N2q": compiled_resources.get("N2q"),
        "D2q": compiled_resources.get("D2q"),
        "Dc": compiled_resources.get("Dc"),
        "S": None,
    }
    resource_alignment = _resource_alignment(
        old_display=qrow.get("display_target"),
        prefix_resources=prefix_resources,
    )
    old_qiskit = _qiskit_original_snapshot(qrow)
    old_plateau = _plateau_original_snapshot(plateau_row)
    old_fidelity = _fidelity_original_snapshot(fidelity_row)
    generated_utc = datetime.now(timezone.utc).isoformat()
    repair_block = {
        "schema": REPAIR_SCHEMA,
        "generated_utc": generated_utc,
        "regime": REGIME,
        "method": METHOD,
        "k_pl": k_pl,
        "reason": "replace source-stopped terminal SNAKE value with source-history prefix value at displayed k_pl",
        "source_json": _rel(source_json),
        "source_sha256": _sha256(source_json),
        "source_history_prefix_delta_e": prefix_delta_e,
        "source_history_point_count": len(source_points),
        "forced_replay_source_json": None if override_source is None else _rel(override_source),
        "forced_replay_source_sha256": None if override_source is None else _sha256(override_source),
        "forced_replay_prefix_audit": forced_prefix_audit,
        "compiled_prefix_resources": prefix_resources,
        "compiled_prefix_full": compiled_resources,
        "S_prefix_status": s_status,
        "S_prefix_note": "canonical prefix S is unavailable from this source; terminal S is not substituted",
        "resource_alignment_vs_old_display": resource_alignment,
        "old_terminal_delta_e": old_qiskit.get("same_cutoff_abs_delta_e_at_k_pl"),
    }

    qrow.update(
        {
            "same_cutoff_abs_delta_e_at_k_pl": prefix_delta_e,
            "compile_policy": "snake_history_prefix_compile",
            "selected_pauli_source": "adapt_vqe.history prefix runtime terms",
            "plateau_status": "snake_history_prefix_repaired_from_source_trajectory",
            "S_not_recomputed_here": True,
            "S_prefix_status": s_status,
            "all_display_costs_match": True,
            "matches_display": {"N2q": True, "D2q": True, "Dc": True, "S": False},
            "prefix_repair": {**repair_block, "original_qiskit_row": old_qiskit},
        }
    )
    qrow["display_target"] = {
        "k_pl": k_pl,
        "DeltaE": _fmt_delta(prefix_delta_e),
        "N2q": prefix_resources["N2q"],
        "D2q": prefix_resources["D2q"],
        "Dc": prefix_resources["Dc"],
        "S": None,
    }
    qrow["replayed_qiskit"] = {
        "N1q": prefix_resources["N1q"],
        "N2q": prefix_resources["N2q"],
        "D2q": prefix_resources["D2q"],
        "Dc": prefix_resources["Dc"],
        "compile_convention": compiled_resources.get("compile_convention"),
        "compile_status": compiled_resources.get("compile_status"),
        "logical_operator_prefix_len": k_pl,
        "qiskit_version": compiled_resources.get("qiskit_version"),
        "runtime_rotation_count": compiled_resources.get("runtime_rotation_count"),
        "compiled_resource_source_kind": compiled_resources.get("compiled_resource_source_kind"),
    }

    plateau_row.update(
        {
            "same_cutoff_abs_delta_e_at_k_pl": prefix_delta_e,
            "same_cutoff_abs_delta_e_at_k_pl_from_support_trajectory": prefix_delta_e,
            "compiled": {
                "N2q": prefix_resources["N2q"],
                "D2q": prefix_resources["D2q"],
                "D_circ": prefix_resources["Dc"],
            },
            "display": {
                "k_pl": k_pl,
                "DeltaE": _fmt_delta(prefix_delta_e),
                "N2q": prefix_resources["N2q"],
                "D2q": prefix_resources["D2q"],
                "Dc": prefix_resources["Dc"],
                "S": None,
            },
            "S_at_k_pl": None,
            "S_at_k_pl_status": s_status,
            "plateau_status": "snake_history_prefix_repaired_from_source_trajectory",
            "prefix_source_repair": {**repair_block, "original_plateau_row": old_plateau},
        }
    )

    fidelity_row.update(
        {
            "marker_same_cutoff_abs_delta_e": prefix_delta_e,
            "display": {
                "k_pl": k_pl,
                "DeltaE": _fmt_delta(prefix_delta_e),
                "N2q": prefix_resources["N2q"],
                "D2q": prefix_resources["D2q"],
                "Dc": prefix_resources["Dc"],
                "S": None,
            },
            "fidelity_status": "terminal_fidelity_retained_prefix_state_unavailable",
            "fidelity_source": (
                "terminal source-stopped SNAKE fidelity retained; prefix-state fidelity unavailable"
            ),
            "fidelity_scope": "terminal_source_stopped_ansatz_not_prefix_recomputed",
            "S_at_k_pl_status": s_status,
            "plateau_status": "snake_history_prefix_repaired_from_source_trajectory",
            "prefix_source_repair": {**repair_block, "original_fidelity_row": old_fidelity},
        }
    )

    audit = {
        **repair_block,
        "dry_run": bool(dry_run),
        "qiskit_json": _rel(qiskit_json),
        "qiskit_json_sha256_before_write": _sha256(qiskit_json),
        "plateau_json": _rel(plateau_json),
        "plateau_json_sha256_before_write": _sha256(plateau_json),
        "fidelity_json": _rel(fidelity_json),
        "fidelity_json_sha256_before_write": _sha256(fidelity_json),
        "fidelity_csv": _rel(fidelity_csv) if fidelity_csv.exists() else None,
        "original_qiskit_row": old_qiskit,
        "original_plateau_row": old_plateau,
        "original_fidelity_row": old_fidelity,
    }

    if not dry_run:
        _write_json(qiskit_json, qiskit_payload)
        _write_json(plateau_json, plateau_payload)
        _write_json(fidelity_json, fidelity_payload)
        _repair_csv(
            csv_path=fidelity_csv,
            prefix_delta_e=prefix_delta_e,
            prefix_resources=prefix_resources,
            s_status=s_status,
        )
        audit.update(
            {
                "qiskit_json_sha256_after_write": _sha256(qiskit_json),
                "plateau_json_sha256_after_write": _sha256(plateau_json),
                "fidelity_json_sha256_after_write": _sha256(fidelity_json),
                "fidelity_csv_sha256_after_write": _sha256(fidelity_csv) if fidelity_csv.exists() else None,
            }
        )
    output_audit_json.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_audit_json, audit)
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qiskit-json",
        type=Path,
        default=Path("output/pdf/paper_i_hh_native200_qiskit_table_replay_20260621_v1.json"),
    )
    parser.add_argument(
        "--plateau-json",
        type=Path,
        default=Path("MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_native200_first_plateau_prefix_audit_20260619.json"),
    )
    parser.add_argument(
        "--fidelity-json",
        type=Path,
        default=Path("MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_native200_kpl_fidelity_marker_audit_20260619.json"),
    )
    parser.add_argument(
        "--fidelity-csv",
        type=Path,
        default=Path("MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_native200_kpl_fidelity_marker_audit_20260619.csv"),
    )
    parser.add_argument(
        "--current-replay-root",
        type=Path,
        default=Path("raw_outputs/paper_i_hh_snake_table_source_resume_forced_iter30_20260622_v1"),
    )
    parser.add_argument(
        "--output-audit-json",
        type=Path,
        default=Path("output/pdf/paper_i_hh_weak_strong_snake_prefix_source_repair_20260622.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    audit = repair(
        qiskit_json=_resolve(args.qiskit_json),
        plateau_json=_resolve(args.plateau_json),
        fidelity_json=_resolve(args.fidelity_json),
        fidelity_csv=_resolve(args.fidelity_csv),
        current_replay_root=_resolve(args.current_replay_root)
        if args.current_replay_root is not None
        else None,
        output_audit_json=_resolve(args.output_audit_json),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
