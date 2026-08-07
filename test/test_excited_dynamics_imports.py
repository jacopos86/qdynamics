from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "pipelines" / "excited_dynamics"
TIME_DYNAMICS = REPO_ROOT / "pipelines" / "time_dynamics"


def _python_files(path: Path) -> list[Path]:
    return sorted(path.glob("*.py"))


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


def test_excited_dynamics_package_uses_absolute_imports_only() -> None:
    for path in _python_files(PACKAGE):
        targets = _import_targets(path)
        assert "<relative>" not in targets, f"{path.relative_to(REPO_ROOT)} uses relative imports"


def test_excited_dynamics_package_does_not_import_controller_or_qiskit() -> None:
    forbidden_roots = {
        "qiskit",
        "pipelines.time_dynamics",
        "pipelines.hardcoded",
        "pipelines.shell",
        "pipelines.exact_bench",
    }
    forbidden_fragments = (
        "realtime",
        "controller",
        "chtc",
        "remote_runner",
        "runner",
    )
    offenders_by_path: dict[Path, list[str]] = {}
    for path in _python_files(PACKAGE):
        offenders: list[str] = []
        for target in _import_targets(path):
            for root in forbidden_roots:
                if target == root or target.startswith(root + "."):
                    offenders.append(target)
            if any(fragment in target for fragment in forbidden_fragments):
                offenders.append(target)
        if offenders:
            offenders_by_path[path.relative_to(REPO_ROOT)] = offenders
    assert offenders_by_path == {}


def test_realtime_modules_do_not_import_excited_dynamics_sidecar() -> None:
    for path in _python_files(TIME_DYNAMICS):
        targets = _import_targets(path)
        offenders = [target for target in targets if target == "pipelines.excited_dynamics" or target.startswith("pipelines.excited_dynamics.")]
        assert offenders == [], f"{path.relative_to(REPO_ROOT)} imports excited_dynamics {offenders!r}"
