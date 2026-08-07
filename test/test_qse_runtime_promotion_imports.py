from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE = REPO_ROOT / "pipelines" / "scaffold" / "qse_runtime_promotion.py"


def _import_targets(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                targets.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if int(node.level) > 0:
                targets.add("<relative>")
            if node.module is not None:
                targets.add(node.module)
    return sorted(targets)


def test_qse_runtime_promotion_uses_absolute_imports_only() -> None:
    assert "<relative>" not in _import_targets(MODULE)


def test_qse_runtime_promotion_has_no_realtime_qiskit_optuna_hardcoded_or_runner_imports() -> None:
    forbidden_roots = {
        "qiskit",
        "optuna",
        "pipelines.time_dynamics",
        "pipelines.hardcoded",
        "pipelines.shell",
        "pipelines.exact_bench",
        "launchctl",
    }
    forbidden_fragments = (
        "realtime",
        "controller",
        "chtc",
        "remote_runner",
        "runner",
    )
    explicitly_allowed = {
        "pipelines.scaffold.runtime_loader",  # lazy offline contract validator only
    }
    offenders: list[str] = []
    for target in _import_targets(MODULE):
        if target in explicitly_allowed:
            continue
        for root in forbidden_roots:
            if target == root or target.startswith(root + "."):
                offenders.append(target)
        if any(fragment in target for fragment in forbidden_fragments):
            offenders.append(target)
    assert offenders == []


def test_runtime_loader_import_is_lazy_function_level_only() -> None:
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    top_level_targets: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            top_level_targets.add(node.module)
    assert "pipelines.scaffold.runtime_loader" not in top_level_targets
    assert "pipelines.scaffold.runtime_loader" in _import_targets(MODULE)
