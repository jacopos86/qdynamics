#!/usr/bin/env python3
"""Local preflight for the two-arm package. No SSH, no submission.

Checks:
  help_lists_run_script_flags   `h2o_application --help`, run from the archived
                                source.tgz (the code that ships), lists every
                                flag run_segment.sh passes
  help_lists_both_arms          both candidate representations are choices
  run_script_flags_in_sync      RUN_SCRIPT_FLAGS == the flags found in the runner
  fixture_sha_matches_manifest  package/<fixture> sha256 == input_manifest.json
  runner_sha_matches_manifest   package/run_segment.sh sha256 == manifest
  manifest_sha256_file_agrees   input_manifest.sha256 lines == manifest digests
  source_archive_imports        pipelines.static_adapt.ra_adapt.h2o_application
                                imports from the extracted archive
  help_names_wall_clock_stop_reason
  archive_controller_stop_reasons_match_next_segment
  archive_resume_ledger_schemas_match_next_segment
  archive_runtime_skips_paper_i_summary_for_h2o
  archive_h2o_application_writes_result_and_summary
                                the archived engine spells the stop reasons and
                                sidecar schemas next_segment.py hard-codes, the
                                H2O family skips the Paper-I summary at exit
                                (929f1b47), and a wall stop writes result.json
                                and summary.json
  runner_stages_resume_outside_output_dir
                                run_segment.sh moves resume files to resume/,
                                never inside --output-dir
  submit_files_parse            submit_*.sub and every <arm>/seg*/submit.sub are
                                key = value lines with the required keys
  scripts_compile               py_compile on *.py, bash -n on *.sh

Exit 0 on pass, 2 on any failure; the JSON verdict is printed either way.
"""

from __future__ import annotations

import argparse
import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_package as bp  # noqa: E402
import next_segment as ns  # noqa: E402

VENV_PYTHON = Path("/Users/lordskyler/Holstein_test/.venv/bin/python")

ARCHIVED_SOURCE_FILES = {
    "controller": "pipelines/static_adapt/sr_snake/_controller.py",
    "resume": "pipelines/static_adapt/sr_snake/_resume.py",
    "runtime": "pipelines/static_adapt/ra_adapt/runtime.py",
    "h2o_application": "pipelines/static_adapt/ra_adapt/h2o_application.py",
}


def archived_source_texts(workdir: Path) -> dict[str, str]:
    """Text of the archived engine files next_segment.py's constants mirror."""
    texts: dict[str, str] = {}
    for key, rel in ARCHIVED_SOURCE_FILES.items():
        path = workdir / rel
        texts[key] = path.read_text(encoding="utf-8") if path.is_file() else ""
    return texts


def _python() -> str:
    return str(VENV_PYTHON if VENV_PYTHON.is_file() else Path(sys.executable))


def flags_in_runner(text: str) -> set[str]:
    """Flags run_segment.sh passes to h2o_application.

    Only the `CMD=( ... )` block and the `RESUME_ARGS=(...)` assignment are
    scanned, so apptainer's own options (--bind, --cleanenv, ...) and
    sha256sum's --strict do not count.
    """
    blocks = re.findall(r"CMD=\((.*?)\n\)", text, flags=re.S)
    blocks += re.findall(r"RESUME_ARGS=\((.*?)\)", text)
    return set(re.findall(r"(--[a-z][a-z0-9-]+)", "\n".join(blocks)))


def run_help_from_archive(archive: Path, workdir: Path) -> tuple[int, str]:
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(workdir, filter="data")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(workdir)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [_python(), "-m", "pipelines.static_adapt.ra_adapt.h2o_application", "--help"],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return completed.returncode, completed.stdout + completed.stderr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-json", type=Path, default=bp.PACKAGE_DIR / "package" / "preflight.json")
    parser.add_argument("--skip-import", action="store_true", help="skip the archive --help/import check (fast mode)")
    args = parser.parse_args(argv)

    package = bp.PACKAGE_DIR / "package"
    checks: dict[str, bool] = {}
    detail: dict[str, Any] = {}

    manifest_path = package / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    checks["manifest_present"] = manifest is not None
    if manifest is None:
        detail["manifest"] = "run build_package.py first"

    runner_text = (bp.PACKAGE_DIR / "run_segment.sh").read_text(encoding="utf-8")
    runner_flags = flags_in_runner(runner_text)
    expected_flags = set(bp.RUN_SCRIPT_FLAGS)
    checks["run_script_flags_in_sync"] = runner_flags == expected_flags
    detail["runner_flags_not_in_list"] = sorted(runner_flags - expected_flags)
    detail["list_flags_not_in_runner"] = sorted(expected_flags - runner_flags)

    if manifest is not None:
        files = manifest["files"]
        fixture = package / bp.FIXTURE_NAME
        checks["fixture_present"] = fixture.is_file()
        checks["fixture_sha_matches_manifest"] = fixture.is_file() and bp.sha256(fixture) == files[bp.FIXTURE_NAME]["sha256"]
        checks["fixture_bytes_expected"] = fixture.is_file() and fixture.stat().st_size == bp.FIXTURE_EXPECTED_BYTES
        detail["fixture_bytes"] = fixture.stat().st_size if fixture.is_file() else None
        runner_pkg = package / "run_segment.sh"
        checks["runner_sha_matches_manifest"] = (
            runner_pkg.is_file()
            and bp.sha256(runner_pkg) == files["run_segment.sh"]["sha256"]
            and bp.sha256(bp.PACKAGE_DIR / "run_segment.sh") == files["run_segment.sh"]["sha256"]
        )
        archive = package / "source.tgz"
        checks["source_sha_matches_manifest"] = archive.is_file() and bp.sha256(archive) == files["source.tgz"]["sha256"]
        sums = (package / "input_manifest.sha256").read_text(encoding="utf-8").split() if (package / "input_manifest.sha256").is_file() else []
        pairs = dict(zip(sums[1::2], sums[0::2]))
        checks["manifest_sha256_file_agrees"] = all(pairs.get(name) == files[name]["sha256"] for name in ("source.tgz", bp.FIXTURE_NAME))
        detail["git_head"] = manifest["source_archive"]["git_head"]
        detail["worktree_dirty_within_archived_paths"] = manifest["source_archive"]["worktree_dirty_within_archived_paths"]
        checks["max_wall_below_periodic_hold"] = int(manifest["segment_budget"]["max_wall_seconds"]) < int(manifest["segment_budget"]["periodic_hold_seconds"]) < bp.POOL_WALL_SECONDS

        if archive.is_file() and not args.skip_import:
            with tempfile.TemporaryDirectory(prefix="paper_iv_preflight_") as tmp:
                rc, out = run_help_from_archive(archive, Path(tmp))
                archived = archived_source_texts(Path(tmp))
            checks["source_archive_imports"] = rc == 0
            missing = sorted(f for f in bp.RUN_SCRIPT_FLAGS if f not in out)
            checks["help_lists_run_script_flags"] = rc == 0 and not missing
            detail["help_missing_flags"] = missing
            checks["help_lists_both_arms"] = all(bp.ARMS[a]["candidate_representation"] in out for a in bp.ARMS)
            # Stop semantics next_segment.py relies on, checked against the
            # code that ships rather than against this worktree.
            checks["help_names_wall_clock_stop_reason"] = rc == 0 and ns.WALL_CLOCK_STOP_REASON in out
            controller = archived.get("controller", "")
            checks["archive_controller_stop_reasons_match_next_segment"] = (
                f'WALL_CLOCK_BUDGET_STOP_REASON = "{ns.WALL_CLOCK_STOP_REASON}"' in controller
                and f'"{ns.EXACT_TARGET_STOP_REASON}"' in controller
                and f'"{ns.HORIZON_STOP_REASON}"' in controller
            )
            resume = archived.get("resume", "")
            checks["archive_resume_ledger_schemas_match_next_segment"] = (
                f'"{ns.LEDGER_POINTER_SCHEMA}"' in resume and f'"{ns.LEDGER_SIDECAR_SCHEMA}"' in resume
            )
            # 929f1b47: the H2O family skips the Paper-I canonical summary that
            # crashed every H2O run at exit; without it a horizon or exact
            # stop returns a checkpoint but no result.json.
            checks["archive_runtime_skips_paper_i_summary_for_h2o"] = (
                '"molecular_vibronic_h2o_linear_fd"' in archived.get("runtime", "")
            )
            # Segment boundary: a wall stop must write result.json + summary.json.
            app = archived.get("h2o_application", "")
            checks["archive_h2o_application_writes_result_and_summary"] = (
                "_write_json(result_path, result.to_dict())" in app
                and 'output_dir / "summary.json"' in app
                and "maximum_wall_seconds=engine_wall_budget_seconds" in app
                and "Refusing to resume from a checkpoint inside --output-dir" in app
            )
            if rc != 0:
                detail["help_tail"] = out[-1500:]
        elif args.skip_import:
            detail["import_check"] = "skipped"

    # run_segment.sh must stage resume files outside --output-dir
    # (h2o_application refuses a checkpoint inside it) and pass the staged path.
    checks["runner_stages_resume_outside_output_dir"] = (
        'mv -- "$name" "resume/$name"' in runner_text
        and '--resume-checkpoint "/work/resume/$RESUME_NAME"' in runner_text
        and "--output-dir /work/results" in runner_text
    )

    submit_problems: list[str] = []
    submit_files = sorted(bp.PACKAGE_DIR.glob("submit_*.sub")) + sorted(bp.PACKAGE_DIR.glob("*/seg*/submit.sub"))
    for path in submit_files:
        submit_problems.extend(bp.check_submit_text(path.read_text(encoding="utf-8"), label=str(path.relative_to(bp.PACKAGE_DIR))))
    checks["submit_files_present"] = any(p.name == "submit_macro.sub" for p in submit_files) and any(p.name == "submit_singleton.sub" for p in submit_files)
    checks["submit_files_parse"] = not submit_problems
    detail["submit_problems"] = submit_problems
    detail["submit_files_checked"] = [str(p.relative_to(bp.PACKAGE_DIR)) for p in submit_files]

    compile_problems: list[str] = []
    for py in sorted(bp.PACKAGE_DIR.glob("*.py")):
        try:
            py_compile.compile(str(py), doraise=True)
        except py_compile.PyCompileError as exc:
            compile_problems.append(str(exc))
    bash = shutil.which("bash")
    for sh in sorted(bp.PACKAGE_DIR.glob("*.sh")):
        completed = subprocess.run([bash or "bash", "-n", str(sh)], capture_output=True, text=True)
        if completed.returncode != 0:
            compile_problems.append(f"{sh.name}: {completed.stderr.strip()}")
    checks["scripts_compile"] = not compile_problems
    detail["compile_problems"] = compile_problems

    payload = {
        "schema": "paper_iv_h2o_two_arm_runtime_preflight_v1",
        "campaign": bp.CAMPAIGN,
        "status": "pass" if all(checks.values()) else "blocked",
        "checks": checks,
        "detail": detail,
        "python": _python(),
        "submitted": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
