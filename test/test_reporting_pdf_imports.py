from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTING_DIR = REPO_ROOT / "pipelines" / "reporting"
IMPORT_TIMEOUT_SECONDS = 5.0

# These modules predate the LaTeX-first paper-facing PDF rule and still use
# ReportLab at module import time. Keep this list explicit so new paper-facing
# PDF builders do not quietly inherit the legacy path.
LEGACY_REPORTLAB_PDF_MODULES = frozenset(
    {
        "build_paper_i_ablation_results_tables_pdf",
        "build_paper_i_hh_intermediate_evidence_pdf",
        "build_paper_i_repeat_policy_suite_comparison_pdf",
        "build_phase3_fullstack_results_summary_pdf",
    }
)

_REPORTLAB_TOP_LEVEL_IMPORT_RE = re.compile(r"^(?:from\s+reportlab\b|import\s+reportlab\b)", re.MULTILINE)
IMPORT_LIGHT_PAPER_I_BUILDERS = (
    "pipelines.reporting.build_paper_i_hh_tracking_plateau_costs",
    "pipelines.reporting.build_paper_i_hh_macro_common_accuracy_pdf",
    "pipelines.reporting.add_paper_i_hh_singleton_common_accuracy_page",
    "pipelines.reporting.add_paper_i_hh_singleton_own_plateau_page",
    "pipelines.reporting.build_paper_i_ra_adapt_stationary_core_master_pdf",
)


def _pdf_module_paths() -> list[Path]:
    return sorted(path for path in REPORTING_DIR.glob("*pdf*.py") if path.name != "__init__.py")


def _module_name(path: Path) -> str:
    return ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)


def _has_top_level_reportlab_import(path: Path) -> bool:
    return _REPORTLAB_TOP_LEVEL_IMPORT_RE.search(path.read_text(encoding="utf-8")) is not None


def test_reporting_pdf_entrypoints_import_without_latency_regression() -> None:
    """PDF builders must not do heavy artifact/PDF work at import time."""

    reportlab_available = importlib.util.find_spec("reportlab") is not None
    failures: list[str] = []
    skipped_legacy_reportlab: list[str] = []
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    for path in _pdf_module_paths():
        module_name = _module_name(path)
        if not reportlab_available and _has_top_level_reportlab_import(path):
            assert path.stem in LEGACY_REPORTLAB_PDF_MODULES
            skipped_legacy_reportlab.append(module_name)
            continue

        code = (
            "import importlib, time\n"
            "started = time.perf_counter()\n"
            f"importlib.import_module({module_name!r})\n"
            "print(f'{time.perf_counter() - started:.6f}')\n"
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=IMPORT_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            failures.append(f"{module_name}: import timed out after {IMPORT_TIMEOUT_SECONDS:.1f}s")
            continue

        if proc.returncode != 0:
            failures.append(
                f"{module_name}: import failed with exit {proc.returncode}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
            continue

        try:
            import_seconds = float(proc.stdout.strip().splitlines()[-1])
        except (IndexError, ValueError):
            failures.append(f"{module_name}: import did not report elapsed seconds; stdout={proc.stdout!r}")
            continue

        if import_seconds > IMPORT_TIMEOUT_SECONDS:
            failures.append(f"{module_name}: import took {import_seconds:.3f}s")

    assert not failures, "\n\n".join(failures)
    assert set(skipped_legacy_reportlab) <= {_module_name(REPORTING_DIR / f"{stem}.py") for stem in LEGACY_REPORTLAB_PDF_MODULES}


def test_reportlab_top_level_imports_stay_in_legacy_pdf_allowlist() -> None:
    offenders = sorted(
        path.name
        for path in _pdf_module_paths()
        if _has_top_level_reportlab_import(path) and path.stem not in LEGACY_REPORTLAB_PDF_MODULES
    )
    assert offenders == []


def test_paper_i_summary_consumers_do_not_import_optional_heavy_stacks() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    for module_name in IMPORT_LIGHT_PAPER_I_BUILDERS:
        code = (
            "import importlib, json, sys\n"
            f"importlib.import_module({module_name!r})\n"
            "heavy = sorted(name for name in sys.modules if "
            "name == 'numpy' or name == 'matplotlib' or "
            "name.startswith('matplotlib.') or name == 'qiskit' or "
            "name.startswith('qiskit.'))\n"
            "print(json.dumps(heavy))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=IMPORT_TIMEOUT_SECONDS,
            check=True,
        )
        assert json.loads(proc.stdout) == [], module_name


def test_legacy_reportlab_pdf_allowlist_matches_existing_modules() -> None:
    existing_pdf_modules = {path.stem for path in _pdf_module_paths()}
    assert LEGACY_REPORTLAB_PDF_MODULES <= existing_pdf_modules
