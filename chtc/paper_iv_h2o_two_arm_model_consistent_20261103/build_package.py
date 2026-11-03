#!/usr/bin/env python3
"""Build the non-submitting CHTC package for the two Paper-IV H2O arms.

This script never submits, never opens SSH, and never touches the pool. It
produces, under this directory:

  package/source.tgz              git archive of the current worktree HEAD
  package/<fixture>.json          the (3,3,3)/18-qubit H2O fixture of the model
                                  decided 2026-11-02 (K_PES + chi oscillators,
                                  model-consistent modes), identity-checked
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


CAMPAIGN = "paper_iv_h2o_two_arm_model_consistent_20261103"
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]

REMOTE_USER = "jsstrobel"
REMOTE_HOME_ROOT = f"/home/{REMOTE_USER}/chtc/{CAMPAIGN}"
STAGING_ROOT = f"/staging/{REMOTE_USER}/{CAMPAIGN}"
DEFAULT_IMAGE_PATH = (
    f"/home/{REMOTE_USER}/paper_i_hh_b3_mr_measured_costexp_20260824/image.sif"
)

FIXTURE_NAME = "h2o_builder_model_consistent_work333_ref444_20261103_fixture.json"
FIXTURE_EXPECTED_BYTES = 15_673_312
# The identity every Paper IV checkpoint on this fixture attests (sha256 of the
# bytes, pinned in test/test_paper_iv_h2o_fixture_from_builder.py as well).
FIXTURE_EXPECTED_SHA256 = (
    "de32cd4d1e29b8dcc07e94ad132ee43893fe03fd36cf4538eb6dadf6d5e194df"
)
# What the fixture must say about itself before it is packaged: the decided
# model (producer src/quantum/chemistry/paper_iv_h2o_fixture_from_builder.py),
# 18 qubits, work cutoffs (3,3,3), lane mode order, production evidence.
FIXTURE_EXPECTED_IDENTITY = {
    "n_total_qubits": 18,
    "n_ph_max_by_mode": [3, 3, 3],
    "mode_labels": ["bend", "symmetric_stretch", "antisymmetric_stretch"],
    "production_status": "production_validated",
    "model_definition": {
        "decided_model": True,
        "harmonic_curvature": "pes_consistent",
        "alignment": "block_diagonal",
        "mode_basis": "model_consistent",
    },
}
FIXTURE_SEARCH_PATHS = (
    REPO_ROOT / "test_support" / "paper_iv_h2o" / FIXTURE_NAME,
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
# Memory: the 19-qubit (3,7,3) campaign (sector 28,800) was raised to 22,528 MB
# after a measured true peak of 14,792 MB at round 25 (append-only ADAPT grows
# per round). This model has half that sector (14,400) and 18 qubits; 16 GB is
# that peak scaled by the sector ratio with a 2x margin. No peak has been
# measured for this model yet: override with --request-memory once one exists.
REQUEST_MEMORY = "16GB"
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
    request_memory: str = REQUEST_MEMORY,
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
            f"request_memory = {request_memory}",
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


def verify_fixture_identity(path: Path) -> dict[str, Any]:
    """Refuse any fixture but the decided model's: bytes, sha256 and what it says about itself.

    Returns the identity record written into the input manifest. Raises
    ValueError naming every mismatch, so a package can never be built on the
    old (3,7,3)/19-qubit fixture or on a fixture the producer did not mark as
    the decided model.
    """
    problems: list[str] = []
    size = path.stat().st_size
    digest = sha256(path)
    if size != FIXTURE_EXPECTED_BYTES:
        problems.append(f"bytes {size} != {FIXTURE_EXPECTED_BYTES}")
    if digest != FIXTURE_EXPECTED_SHA256:
        problems.append(f"sha256 {digest} != {FIXTURE_EXPECTED_SHA256}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = FIXTURE_EXPECTED_IDENTITY
    found = {
        "n_total_qubits": int(payload["register_layout"]["n_total_qubits"]),
        "n_ph_max_by_mode": [int(v) for v in payload["physical_sector"]["n_ph_max_by_mode"]],
        "mode_labels": [str(v) for v in payload["physical_sector"]["mode_labels"]],
        "production_status": str(payload["manifest"]["production_status"]),
        "model_definition": {
            key: payload["provenance"]["backend"].get("model_definition", {}).get(key)
            for key in expected["model_definition"]
        },
    }
    for key, value in expected.items():
        if found[key] != value:
            problems.append(f"{key}: {found[key]!r} != {value!r}")
    if problems:
        raise ValueError(
            f"{path.name} is not the fixture this package is for: " + "; ".join(problems)
        )
    exact = payload["exact_reference"]
    return {
        "sha256": digest,
        "bytes": size,
        "schema": str(payload["manifest"]["schema"]),
        "exact_ground_energy_hartree": exact.get("ground_energy_hartree"),
        "sector_dimension": exact.get("sector_dimension"),
        "mode_frequencies_cm1": [float(m["frequency_cm1"]) for m in payload["normal_modes"]],
        "model_decided_on": payload["provenance"]["backend"]["model_definition"].get("decided_on"),
        **found,
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
    parser.add_argument(
        "--request-memory",
        default=REQUEST_MEMORY,
        help=f"HTCondor request_memory for every segment (default {REQUEST_MEMORY}; see the constant's comment)",
    )
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9]+(MB|GB)", str(args.request_memory)):
        print(f"--request-memory must look like 16GB or 22528MB, got {args.request_memory!r}", file=sys.stderr)
        return 2

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
    try:
        fixture_identity = verify_fixture_identity(fixture_dst)
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"FIXTURE-CHECK FAIL: {exc}", file=sys.stderr)
        fixture_dst.unlink(missing_ok=True)
        return 2

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
        "fixture_identity": fixture_identity,
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
            "memory": str(args.request_memory),
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
            request_memory=str(args.request_memory),
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
