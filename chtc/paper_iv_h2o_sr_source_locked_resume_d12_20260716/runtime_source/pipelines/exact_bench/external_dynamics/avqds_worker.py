#!/usr/bin/env python3
"""Isolated source/API probe for the GQCE/Ames base AVQDS reference."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

WORKER_SCHEMA = "avqds_external_worker_v1"


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _class_methods(tree: ast.AST, class_name: str) -> set[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {item.name for item in node.body if isinstance(item, ast.FunctionDef)}
    return set()


def run_probe(checkout: Path) -> dict[str, Any]:
    ansatz_source = checkout / "avqds" / "ansatz.py"
    dyn_source = checkout / "avqds" / "avaridyn.py"
    if not ansatz_source.exists() or not dyn_source.exists():
        return {
            "schema": WORKER_SCHEMA,
            "status": "skipped_incompatible_api",
            "passed": None,
            "reason": f"missing avqds/ansatz.py or avqds/avaridyn.py in {checkout}",
        }
    ansatz_text = ansatz_source.read_text(encoding="utf-8")
    dyn_text = dyn_source.read_text(encoding="utf-8")
    try:
        ansatz_tree = ast.parse(ansatz_text)
        dyn_tree = ast.parse(dyn_text)
    except SyntaxError as exc:
        return {
            "schema": WORKER_SCHEMA,
            "status": "skipped_incompatible_api",
            "passed": None,
            "reason": f"AVQDS source syntax parse failed: {exc}",
        }
    ansatz_methods = _class_methods(ansatz_tree, "ansatz")
    dyn_methods = _class_methods(dyn_tree, "avaridynBase")
    required_ansatz = {"one_step", "add_ops_dyn", "get_dist", "set_par_states"}
    required_dyn = {"run", "set_initial_state", "init_records", "update_records"}
    missing_ansatz = sorted(required_ansatz - ansatz_methods)
    missing_dyn = sorted(required_dyn - dyn_methods)
    has_mclachlan_distance = "McLachlan distance" in ansatz_text and "self._rcut" in ansatz_text
    has_adaptive_growth = "add_ops_dyn" in ansatz_text and "self._ansatz" in ansatz_text and "append" in ansatz_text
    passed = not missing_ansatz and not missing_dyn and has_mclachlan_distance and has_adaptive_growth
    return {
        "schema": WORKER_SCHEMA,
        "status": "completed_source_probe" if passed else "skipped_incompatible_api",
        "passed": None,
        "reason": "source/API conformance probe completed; numeric external parity not executed",
        "numeric_parity_executed": False,
        "source_conformance_passed": bool(passed),
        "features": {
            "ansatz_class_methods": sorted(ansatz_methods),
            "avaridyn_base_methods": sorted(dyn_methods),
            "required_ansatz_methods_missing": missing_ansatz,
            "required_dynamics_methods_missing": missing_dyn,
            "uses_mclachlan_distance_cutoff": bool(has_mclachlan_distance),
            "uses_adaptive_ansatz_growth": bool(has_adaptive_growth),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = run_probe(Path(args.checkout).expanduser())
    _write(Path(args.output), payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
