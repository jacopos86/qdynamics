from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PLOTS_ROOT = REPO_ROOT / "plots" / "adapt_scaffold"
PYTHON_FILES = sorted(path for path in PLOTS_ROOT.glob("*.py"))
ORQVIZ_ADAPTER = PLOTS_ROOT / "orqviz_adapter.py"


def _import_targets(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                targets.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            targets.add(node.module)
            for alias in node.names:
                if alias.name == "*":
                    continue
                targets.add(f"{node.module}.{alias.name}")
    return sorted(targets)


def _relative_import_offenders(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sorted(
        int(node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and int(node.level) > 0
    )


def _root_descendant_offenders(path: Path, *, forbidden_root: str) -> list[str]:
    return [
        target
        for target in _import_targets(path)
        if target == forbidden_root or target.startswith(forbidden_root + ".")
    ]


def _exact_offenders(path: Path, *, forbidden: set[str]) -> list[str]:
    return [target for target in _import_targets(path) if target in forbidden]


def test_new_plot_files_use_absolute_imports_only() -> None:
    for path in PYTHON_FILES:
        assert _relative_import_offenders(path) == [], f"{path.relative_to(REPO_ROOT)} must use absolute imports only."


def test_reader_renderer_landscape_main_avoid_pipeline_wrappers() -> None:
    guarded = [
        PLOTS_ROOT / "artifact_reader.py",
        PLOTS_ROOT / "renderers.py",
        PLOTS_ROOT / "landscape.py",
        PLOTS_ROOT / "main.py",
    ]
    forbidden_exact = {
        "pipelines.static_adapt.adapt_pipeline",
        "pipelines.static_adapt.output_artifacts",
        "pipelines.time_dynamics.runners.hh_from_adapt_artifact",
    }
    for path in guarded:
        offenders = _exact_offenders(path, forbidden=forbidden_exact)
        assert offenders == [], f"{path.relative_to(REPO_ROOT)} imports forbidden pipeline wrappers: {offenders!r}"
        report_offenders = _root_descendant_offenders(path, forbidden_root="pipelines.reporting")
        assert report_offenders == [], f"{path.relative_to(REPO_ROOT)} must not import reporting modules: {report_offenders!r}"


def test_only_orqviz_adapter_imports_external_orqviz() -> None:
    for path in PYTHON_FILES:
        offenders = _root_descendant_offenders(path, forbidden_root="orqviz")
        if path == ORQVIZ_ADAPTER:
            continue
        assert offenders == [], f"{path.relative_to(REPO_ROOT)} must not import external orqviz modules."


def test_orqviz_adapter_has_no_module_level_orqviz_import() -> None:
    tree = ast.parse(ORQVIZ_ADAPTER.read_text(encoding="utf-8"))
    module_level_offenders: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "orqviz" or alias.name.startswith("orqviz."):
                    module_level_offenders.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module == "orqviz" or node.module.startswith("orqviz."):
                module_level_offenders.append(node.module)
    assert module_level_offenders == [], "orqviz_adapter.py must not import orqviz at module scope."

