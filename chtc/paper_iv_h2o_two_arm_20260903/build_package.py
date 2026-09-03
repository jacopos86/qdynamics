#!/usr/bin/env python3
"""Build the non-submitting CHTC package for the two Paper-IV H2O arms.

This script never submits, never opens SSH, and never touches the pool. It
produces, under this directory:

  package/source.tgz              git archive of the current worktree HEAD
  package/<fixture>.json          the (3,7,3)/19-qubit H2O fixture
  package/run_segment.sh          copied runner (hashed into the manifest)
  package/input_manifest.json     sha256 of every input, git sha, build args
  package/input_manifest.sha256   the same digests in `sha256sum -c` form
  package/build_record.json       build provenance
  submit_macro.sub                segment 00 of the macro arm
  submit_singleton.sub            segment 00 of the singleton arm
  <arm>/seg00/logs/.keep          HTCondor needs the log directory to exist

`next_segment.py` imports `render_submit` from here so every segment of both
arms is written from one template.

Segment arithmetic (see README.md):

  pool wall (SYSTEM_PERIODIC_HOLD, no stage-out)   259200 s = 72 h
  submit-file periodic_hold (belt and braces)      172800 s = 48 h
  in-process --max-wall-seconds (the real stop)    129600 s = 36 h  (default)

The 12 h between the in-process stop and the periodic_hold is the budget for
the last controller round (the most expensive one of a segment), the final
checkpoint write, the tarball, and output transfer. Override with
`--max-wall-seconds` once increment 2 calibration gives real per-round times.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CAMPAIGN = "paper_iv_h2o_two_arm_20260903"
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]

REMOTE_USER = "jsstrobel"
REMOTE_HOME_ROOT = f"/home/{REMOTE_USER}/chtc/{CAMPAIGN}"
STAGING_ROOT = f"/staging/{REMOTE_USER}/{CAMPAIGN}"
DEFAULT_IMAGE_PATH = (
    f"/home/{REMOTE_USER}/paper_i_hh_b3_mr_measured_costexp_20260824/image.sif"
)

FIXTURE_NAME = "h2o_linear_fd_work373_ref484_fixture_20260826.json"
FIXTURE_EXPECTED_BYTES = 28_025_185
FIXTURE_SEARCH_PATHS = (
    REPO_ROOT / "test_support" / "paper_iv_h2o" / FIXTURE_NAME,
    Path.home()
    / "chtc_rescue/home/chtc/paper_iv_h2o_reencode_20260826/results"
    / FIXTURE_NAME,
)

ARMS: dict[str, dict[str, str]] = {
    "macro": {"candidate_representation": "macro_generator_v1"},
    "singleton": {
        "candidate_representation": "sector_complete_pauli_block_v1"
    },
}
COST_MODES = ("proxy", "marrakesh_graph_span_v1")
RESULTS_MODES = ("home", "staging")

HORIZON_ROUNDS = 50
POOL_WALL_SECONDS = 259_200
PERIODIC_HOLD_SECONDS = 172_800
DEFAULT_MAX_WALL_SECONDS = 129_600
REQUEST_CPUS = 4
REQUEST_MEMORY = "8GB"
REQUEST_DISK = "10GB"

SOURCE_ARCHIVE_PATHS = ("src", "pipelines", "docs/reports")

# Every h2o_application flag that run_segment.sh passes. runtime_preflight.py
# checks each against `--help` of the archived source and cross-checks this
# list against the runner text so the two cannot drift apart silently.
RUN_SCRIPT_FLAGS = (
    "--fixture",
    "--output-dir",
    "--maximum-controller-rounds",
    "--insertion",
    "--checkpoint-history-tail",
    "--candidate-representation",
    "--cost-mode",
    "--max-wall-seconds",
    "--resume-checkpoint",
)

_SUBMIT_LINE = re.compile(r"^\s*(\+?[A-Za-z_][A-Za-z0-9_.]*)\s*=\s*(.*)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def segment_tag(arm: str, segment: int) -> str:
    return f"{arm}_seg{int(segment):02d}"


def segment_dirname(segment: int) -> str:
    return f"seg{int(segment):02d}"


def render_submit(
    *,
    arm: str,
    segment: int,
    cost_mode: str,
    max_wall_seconds: int,
    results_mode: str,
    resume_files: tuple[str, ...] = (),
    remote_root: str = REMOTE_HOME_ROOT,
    staging_root: str = STAGING_ROOT,
    image_path: str = DEFAULT_IMAGE_PATH,
    horizon_rounds: int = HORIZON_ROUNDS,
) -> str:
    """Render one segment's submit file.

    `resume_files` are basenames under `<arm>/segNN/input/` (current.json,
    its content-addressed sidecars, and resume_manifest.sha256). They are
    transferred without preserve_relative_paths, so they land in the scratch
    root as basenames; run_segment.sh moves them into `resume/` before the
    run so the checkpoint never sits inside --output-dir.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}")
    if cost_mode not in COST_MODES:
        raise ValueError(f"unknown cost mode {cost_mode!r}")
    if results_mode not in RESULTS_MODES:
        raise ValueError(f"unknown results mode {results_mode!r}")
    if int(max_wall_seconds) <= 0 or int(max_wall_seconds) >= PERIODIC_HOLD_SECONDS:
        raise ValueError(
            "--max-wall-seconds must be positive and strictly below the "
            f"{PERIODIC_HOLD_SECONDS}s periodic_hold"
        )
    if resume_files and "current.json" not in resume_files:
        raise ValueError("resume_files must include current.json")
    if int(segment) > 0 and not resume_files:
        raise ValueError("segments after seg00 need resume files")
    if int(segment) == 0 and resume_files:
        raise ValueError("seg00 is a fresh start and cannot carry resume files")

    tag = segment_tag(arm, segment)
    seg_dir = f"{remote_root}/{arm}/{segment_dirname(segment)}"
    rep = ARMS[arm]["candidate_representation"]
    resume_arg = "current.json" if resume_files else "-"
    inputs = [
        f"{remote_root}/package/source.tgz",
        f"{remote_root}/package/{FIXTURE_NAME}",
        f"{remote_root}/package/input_manifest.sha256",
        image_path,
    ]
    inputs.extend(f"input/{name}" for name in resume_files)
    tarball = f"transfer/{tag}_results.tar.gz"

    lines = [
        f"# {CAMPAIGN} -- {tag} -- generated by build_package.py; do not hand-edit",
        "universe = vanilla",
        f"initialdir = {seg_dir}",
        f"executable = {remote_root}/package/run_segment.sh",
        "transfer_executable = True",
        (
            'arguments = "'
            f"{arm} {int(segment):02d} {rep} {cost_mode} {int(max_wall_seconds)} "
            f'{int(horizon_rounds)} {results_mode} {resume_arg}"'
        ),
        "should_transfer_files = YES",
        # ON_EXIT: a normal exit is the only event this design relies on. A
        # hold (pool 72h SYSTEM_PERIODIC_HOLD or our own periodic_hold) never
        # stages out, so the in-process stop is what makes results transfer.
        "when_to_transfer_output = ON_EXIT",
        f"transfer_input_files = {', '.join(inputs)}",
    ]
    if results_mode == "home":
        lines.append("transfer_output_files = results")
    else:
        # The only remap this pool has been shown to honor is a single file
        # remapped to /staging (b3mr_18cell_chtc/submit_fresh.sub,
        # ..._four_arm_20260817_v2_chtc/submit_home.sub). Directory remaps
        # have no precedent here, so the runner tars results/ first.
        lines.append(f"transfer_output_files = {tarball}")
        lines.append(
            f'transfer_output_remaps = "{tarball}='
            f'{staging_root}/{arm}/{segment_dirname(segment)}/{tag}_results.tar.gz"'
        )
    lines.extend(
        [
            f"request_cpus = {REQUEST_CPUS}",
            f"request_memory = {REQUEST_MEMORY}",
            f"request_disk = {REQUEST_DISK}",
            "max_retries = 0",
            f"+MaxRuntime = {PERIODIC_HOLD_SECONDS}",
            (
                "periodic_hold = (JobStatus == 2) && "
                f"((time() - EnteredCurrentStatus) > {PERIODIC_HOLD_SECONDS})"
            ),
            (
                'periodic_hold_reason = "'
                f"{CAMPAIGN} {tag}: exceeded the 48h segment budget before the "
                "72h pool wall; the in-process --max-wall-seconds stop did not "
                'fire. A hold never stages out: this segment is lost, but 24h '
                'of pool time is not."'
            ),
            "periodic_hold_subcode = 48",
            "kill_sig = SIGTERM",
            "kill_sig_timeout = 900",
            "notification = Never",
            "getenv = False",
            "stream_output = False",
            "stream_error = False",
            f"log = logs/{tag}.$(Cluster).$(Process).log",
            f"output = logs/{tag}.$(Cluster).$(Process).out",
            f"error = logs/{tag}.$(Cluster).$(Process).err",
            f'+JobBatchName = "{CAMPAIGN}_{arm}"',
            f'environment = "PAPER_IV_SEGMENT_TAG={tag}"',
            "queue 1",
        ]
    )
    return "\n".join(lines) + "\n"


def check_submit_text(text: str, *, label: str) -> list[str]:
    """Self-check: every non-comment line is `key = value` or `queue ...`."""
    problems: list[str] = []
    saw_queue = False
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("queue"):
            saw_queue = True
            continue
        if saw_queue:
            problems.append(f"{label}:{number}: content after queue")
        match = _SUBMIT_LINE.match(line)
        if match is None:
            problems.append(f"{label}:{number}: not `key = value`: {line}")
            continue
        value = match.group(2)
        if value.count('"') % 2:
            problems.append(f"{label}:{number}: unbalanced quotes")
    if not saw_queue:
        problems.append(f"{label}: no queue statement")
    required = {
        "when_to_transfer_output",
        "transfer_output_files",
        "request_cpus",
        "request_memory",
        "request_disk",
        "max_retries",
        "periodic_hold",
        "periodic_hold_reason",
        "+JobBatchName",
    }
    present = {
        m.group(1)
        for m in (_SUBMIT_LINE.match(l) for l in text.splitlines())
        if m is not None
    }
    for key in sorted(required - present):
        problems.append(f"{label}: missing {key}")
    return problems


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_source_archive(destination: Path) -> dict[str, Any]:
    head = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--", *SOURCE_ARCHIVE_PATHS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        subprocess.run(
            [
                "git",
                "archive",
                "--format=tar.gz",
                f"--prefix=",
                head,
                "--",
                *SOURCE_ARCHIVE_PATHS,
            ],
            cwd=REPO_ROOT,
            check=True,
            stdout=handle,
        )
    return {
        "git_head": head,
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "archived_paths": list(SOURCE_ARCHIVE_PATHS),
        "worktree_dirty_within_archived_paths": bool(dirty),
        "dirty_entries": dirty.splitlines(),
        "note": (
            "git archive packages HEAD, not the working tree: uncommitted "
            "edits in this worktree are NOT in source.tgz."
        ),
    }


def locate_fixture(explicit: Path | None) -> Path:
    candidates = [explicit] if explicit is not None else list(FIXTURE_SEARCH_PATHS)
    for candidate in candidates:
        if candidate is not None and candidate.expanduser().is_file():
            return candidate.expanduser().resolve()
    raise SystemExit(
        "H2O fixture not found; pass --fixture. Searched: "
        + ", ".join(str(c) for c in candidates if c is not None)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixture", type=Path, default=None)
    parser.add_argument(
        "--macro-cost-mode", choices=COST_MODES, default="proxy"
    )
    parser.add_argument(
        "--singleton-cost-mode", choices=COST_MODES, default="proxy"
    )
    parser.add_argument(
        "--max-wall-seconds", type=int, default=DEFAULT_MAX_WALL_SECONDS
    )
    parser.add_argument(
        "--results-mode",
        choices=RESULTS_MODES,
        default="home",
        help=(
            "home: results/ lands in <initialdir>/results under /home "
            "(default; /staging/jsstrobel did not exist on 2026-09-03). "
            "staging: results are tarred and remapped to "
            f"{STAGING_ROOT}/<arm>/segNN/ -- only once /staging exists."
        ),
    )
    parser.add_argument("--image-path", default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--remote-root", default=REMOTE_HOME_ROOT)
    args = parser.parse_args(argv)

    if args.results_mode == "staging":
        print(
            "WARNING: staging mode remaps output to "
            f"{STAGING_ROOT}. Confirm `ls -ld /staging/{REMOTE_USER}` on "
            "ap2001 first: a remap into a missing directory fails the "
            "output transfer.",
            file=sys.stderr,
        )

    package = PACKAGE_DIR / "package"
    package.mkdir(exist_ok=True)

    archive = package / "source.tgz"
    archive_info = build_source_archive(archive)

    fixture_src = locate_fixture(args.fixture)
    fixture_dst = package / FIXTURE_NAME
    if fixture_src != fixture_dst:
        shutil.copyfile(fixture_src, fixture_dst)
    fixture_bytes = fixture_dst.stat().st_size
    if fixture_bytes != FIXTURE_EXPECTED_BYTES:
        print(
            f"WARNING: fixture is {fixture_bytes} bytes, expected "
            f"{FIXTURE_EXPECTED_BYTES}; recording what was given.",
            file=sys.stderr,
        )

    runner_src = PACKAGE_DIR / "run_segment.sh"
    runner_dst = package / "run_segment.sh"
    shutil.copyfile(runner_src, runner_dst)
    runner_dst.chmod(0o755)

    manifest_files = {
        "source.tgz": archive,
        FIXTURE_NAME: fixture_dst,
        "run_segment.sh": runner_dst,
    }
    digests = {name: sha256(path) for name, path in manifest_files.items()}
    (package / "input_manifest.sha256").write_text(
        "".join(
            f"{digests[name]}  {name}\n"
            for name in ("source.tgz", FIXTURE_NAME)
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema": "paper_iv_h2o_two_arm_input_manifest_v1",
        "campaign": CAMPAIGN,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            name: {"sha256": digests[name], "bytes": path.stat().st_size}
            for name, path in manifest_files.items()
        },
        "source_archive": archive_info,
        "fixture_source_path": str(fixture_src),
        "image_path": args.image_path,
        "image_sha256": None,
        "image_note": (
            "image.sif lives only on ap2001; run_segment.sh records its "
            "sha256 into results/segment_receipt.json at run time."
        ),
        "arms": {
            arm: {
                **ARMS[arm],
                "cost_mode": getattr(args, f"{arm}_cost_mode"),
            }
            for arm in ARMS
        },
        "segment_budget": {
            "pool_wall_seconds": POOL_WALL_SECONDS,
            "periodic_hold_seconds": PERIODIC_HOLD_SECONDS,
            "max_wall_seconds": args.max_wall_seconds,
            "horizon_rounds": HORIZON_ROUNDS,
        },
        "results_mode": args.results_mode,
        "remote_root": args.remote_root,
        "staging_root": STAGING_ROOT,
        "run_script_flags": list(RUN_SCRIPT_FLAGS),
        "requests": {
            "cpus": REQUEST_CPUS,
            "memory": REQUEST_MEMORY,
            "disk": REQUEST_DISK,
        },
    }
    write_json(package / "input_manifest.json", manifest)

    problems: list[str] = []
    for arm in ARMS:
        text = render_submit(
            arm=arm,
            segment=0,
            cost_mode=getattr(args, f"{arm}_cost_mode"),
            max_wall_seconds=args.max_wall_seconds,
            results_mode=args.results_mode,
            remote_root=args.remote_root,
            image_path=args.image_path,
        )
        submit_path = PACKAGE_DIR / f"submit_{arm}.sub"
        submit_path.write_text(text, encoding="utf-8")
        problems.extend(check_submit_text(text, label=submit_path.name))
        seg_dir = PACKAGE_DIR / arm / segment_dirname(0)
        (seg_dir / "logs").mkdir(parents=True, exist_ok=True)
        (seg_dir / "logs" / ".keep").touch()
        shutil.copyfile(submit_path, seg_dir / "submit.sub")
        write_json(
            seg_dir / "segment.json",
            {
                "schema": "paper_iv_h2o_two_arm_segment_v1",
                "campaign": CAMPAIGN,
                "arm": arm,
                "segment": 0,
                "candidate_representation": ARMS[arm][
                    "candidate_representation"
                ],
                "cost_mode": getattr(args, f"{arm}_cost_mode"),
                "max_wall_seconds": args.max_wall_seconds,
                "horizon_rounds": HORIZON_ROUNDS,
                "results_mode": args.results_mode,
                "resume_from": None,
                "submitted_cluster": None,
                "submit_file": "submit.sub",
            },
        )

    write_json(
        package / "build_record.json",
        {
            "schema": "paper_iv_h2o_two_arm_build_record_v1",
            "campaign": CAMPAIGN,
            "built_utc": datetime.now(timezone.utc).isoformat(),
            "argv": sys.argv[1:] if argv is None else list(argv),
            "python": sys.executable,
            "cwd": os.getcwd(),
            "git_head": archive_info["git_head"],
            "digests": digests,
            "submit_files": [f"submit_{arm}.sub" for arm in ARMS],
            "submitted": False,
        },
    )

    if problems:
        for problem in problems:
            print(f"SUBMIT-CHECK FAIL: {problem}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "campaign": CAMPAIGN,
                "git_head": archive_info["git_head"],
                "dirty_archived_paths": archive_info[
                    "worktree_dirty_within_archived_paths"
                ],
                "digests": digests,
                "results_mode": args.results_mode,
                "max_wall_seconds": args.max_wall_seconds,
                "submit_files": [f"submit_{arm}.sub" for arm in ARMS],
                "submitted": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
