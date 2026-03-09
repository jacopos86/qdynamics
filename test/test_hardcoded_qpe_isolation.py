#!/usr/bin/env python3
"""Guard that the live hardcoded runtime does not import archive compare code."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
HARD_CODED_PIPELINE = REPO_ROOT / "pipelines" / "hardcoded" / "hubbard_pipeline.py"


def _import_targets(tree: ast.AST) -> list[str]:
    targets: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            targets.append(node.module)
    return targets


def test_hardcoded_pipeline_does_not_import_qiskit_archive() -> None:
    tree = ast.parse(HARD_CODED_PIPELINE.read_text(encoding="utf-8"))
    offenders = [
        target
        for target in _import_targets(tree)
        if target == "pipelines.qiskit_archive" or target.startswith("pipelines.qiskit_archive.")
    ]
    assert offenders == [], (
        "pipelines/hardcoded/hubbard_pipeline.py must not import archive compare modules; "
        f"found {offenders!r}"
    )
