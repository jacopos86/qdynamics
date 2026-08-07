from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
QSE_PACKAGE = REPO_ROOT / "pipelines" / "qse_spectra"


def _python_files() -> list[Path]:
    return sorted(QSE_PACKAGE.glob("*.py"))


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


def _lazy_symbol_modules() -> dict[str, str]:
    tree = ast.parse((QSE_PACKAGE / "__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        value_node: ast.AST | None = None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_SYMBOL_MODULES" for target in node.targets
        ):
            value_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "_SYMBOL_MODULES":
            value_node = node.value
        if value_node is None:
            continue
        value = ast.literal_eval(value_node)
        assert isinstance(value, dict)
        return {str(key): str(module) for key, module in value.items()}
    raise AssertionError("pipelines/qse_spectra/__init__.py does not define _SYMBOL_MODULES")


def _top_level_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".", maxsplit=1)[0])
    return names


def test_qse_spectra_package_uses_absolute_imports_only() -> None:
    for path in _python_files():
        targets = _import_targets(path)
        assert "<relative>" not in targets, f"{path.relative_to(REPO_ROOT)} uses relative imports."


def test_qse_spectra_package_does_not_import_qiskit_or_pipeline_facades() -> None:
    forbidden_roots = {
        "qiskit",
        "optuna",
        "pipelines.excited_dynamics",
        "pipelines.hardcoded",
        "pipelines.time_dynamics",
        "pipelines.static_adapt",
    }
    allowed_targets = {
        (
            Path("pipelines/qse_spectra/optuna_tune.py"),
            "optuna",
        ),
        (
            Path("pipelines/qse_spectra/static_adapt_adapter.py"),
            "pipelines.static_adapt.builders.hh_pool_presets",
        ),
        (
            Path("pipelines/qse_spectra/static_adapt_adapter.py"),
            "pipelines.static_adapt.builders.problem_setup",
        ),
    }
    for path in _python_files():
        offenders: list[str] = []
        rel_path = path.relative_to(REPO_ROOT)
        for target in _import_targets(path):
            if (rel_path, target) in allowed_targets:
                continue
            for root in forbidden_roots:
                if target == root or target.startswith(root + "."):
                    offenders.append(target)
        assert offenders == [], f"{path.relative_to(REPO_ROOT)} has forbidden imports {offenders!r}"


def test_qse_spectra_lazy_package_exports_match_symbol_map() -> None:
    import pipelines.qse_spectra as qse_spectra

    assert qse_spectra.__all__ == list(_lazy_symbol_modules())


def test_qse_spectra_lazy_symbol_map_targets_existing_top_level_names() -> None:
    for symbol, module_name in _lazy_symbol_modules().items():
        prefix = "pipelines.qse_spectra."
        assert module_name.startswith(prefix), f"{symbol!r} maps outside QSE package: {module_name!r}"
        module_leaf = module_name.removeprefix(prefix)
        assert "." not in module_leaf, f"{symbol!r} maps to unsupported nested module: {module_name!r}"
        module_path = QSE_PACKAGE / f"{module_leaf}.py"
        assert module_path.exists(), f"{symbol!r} maps to missing module: {module_name!r}"
        assert symbol in _top_level_names(module_path), (
            f"{symbol!r} is listed in lazy QSE exports but is not a top-level name in "
            f"{module_path.relative_to(REPO_ROOT)}"
        )
