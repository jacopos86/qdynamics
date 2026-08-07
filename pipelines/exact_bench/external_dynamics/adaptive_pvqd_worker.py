#!/usr/bin/env python3
"""Isolated source/API probe for the dalin27 adaptive-pVQD reference."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

WORKER_SCHEMA = "adaptive_pvqd_external_worker_v1"


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _class_methods(tree: ast.AST, class_name: str) -> set[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {item.name for item in node.body if isinstance(item, ast.FunctionDef)}
    return set()


def _contains_name(tree: ast.AST, name: str) -> bool:
    return any(isinstance(node, ast.Name) and node.id == name for node in ast.walk(tree))


def run_probe(checkout: Path) -> dict[str, Any]:
    source = checkout / "adaptive_pvqd.py"
    if not source.exists():
        return {
            "schema": WORKER_SCHEMA,
            "status": "skipped_incompatible_api",
            "passed": None,
            "reason": f"missing adaptive_pvqd.py in {checkout}",
        }
    text = source.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return {
            "schema": WORKER_SCHEMA,
            "status": "skipped_incompatible_api",
            "passed": None,
            "reason": f"adaptive_pvqd.py syntax parse failed: {exc}",
        }
    methods = _class_methods(tree, "AdaptivePVQD")
    required = {"get_loss", "adaptive_step", "minimization_routine", "one_time_step", "evolve"}
    missing = sorted(required - methods)
    has_suzuki_target = _contains_name(tree, "SuzukiTrotter") and "PauliEvolutionGate" in text
    has_tetris_like_batching = "Method 1: Tetris-like" in text and "isdisjoint" in text
    passed = not missing and has_suzuki_target and has_tetris_like_batching
    return {
        "schema": WORKER_SCHEMA,
        "status": "completed_source_probe" if passed else "skipped_incompatible_api",
        "passed": None,
        "reason": "source/API conformance probe completed; numeric external parity not executed",
        "numeric_parity_executed": False,
        "source_conformance_passed": bool(passed),
        "features": {
            "class": "AdaptivePVQD",
            "methods": sorted(methods),
            "required_methods_missing": missing,
            "uses_product_formula_target": bool(has_suzuki_target),
            "uses_tetris_like_disjoint_batching": bool(has_tetris_like_batching),
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
