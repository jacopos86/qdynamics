#!/usr/bin/env python3
"""Validate the exact candidate CHTC package before submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--require-clean-source", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from chtc.validate_condor_submit_lifecycle import validate_submit_lifecycle
    from pipelines.qse_spectra.matched_route_acquisition import (
        METHOD_DIRECT_DAVIDSON,
        METHOD_SELECTED_QSE,
        locked_hh_profile,
    )

    package = root / "chtc" / "paper_iii_selected_vs_direct_hh_20260828_v1"
    required = (
        package / "input" / "cases.txt",
        package / "run_case.sh",
        package / "run_case_apptainer.sh",
        package / "submit_cases.sub",
        root / "pipelines" / "exact_bench" / "paper_iii_selected_vs_direct_case.py",
        root / "pipelines" / "exact_bench" / "paper_iii_selected_vs_direct_collect.py",
        root / "pipelines" / "qse_spectra" / "matched_route_acquisition.py",
    )
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing required package files: {missing}")

    profile = locked_hh_profile()
    cases = tuple(
        line.strip()
        for line in required[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    expected_cases = tuple(case.case_key for case in profile.cases)
    if cases != expected_cases:
        raise SystemExit("case queue does not match the locked profile")
    if profile.methods != (METHOD_SELECTED_QSE, METHOD_DIRECT_DAVIDSON):
        raise SystemExit("method set is not the locked selected/direct pair")

    submit_text = (package / "submit_cases.sub").read_text(encoding="utf-8")
    validate_submit_lifecycle(submit_text)
    required_submit_fragments = (
        "arguments = $(case_key)",
        "when_to_transfer_output = ON_EXIT_OR_EVICT",
        "transfer_output_files = case_result.json",
        "stream_output = False",
        "stream_error = False",
        "queue case_key from",
    )
    if any(fragment not in submit_text for fragment in required_submit_fragments):
        raise SystemExit("submit descriptor is missing a required contract field")
    lowered = submit_text.lower()
    if "hybrid" in lowered or "fallback" in lowered:
        raise SystemExit("submit descriptor exposes a forbidden route")

    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    relative_required = [str(path.relative_to(root)) for path in required]
    try:
        unstaged = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", *relative_required],
            cwd=root,
            check=False,
        ).returncode
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet", "HEAD", "--", *relative_required],
            cwd=root,
            check=False,
        ).returncode
        source_tree_clean = revision != "unknown" and unstaged == 0 and staged == 0
    except OSError:
        source_tree_clean = False
    if args.require_clean_source and not source_tree_clean:
        raise SystemExit("required package/source paths are not clean at source_revision")
    payload = {
        "schema_version": "paper_iii_selected_vs_direct_chtc_preflight_v1",
        "ok": True,
        "run_classification": "candidate",
        "case_count": len(cases),
        "route_count": len(cases) * len(profile.methods),
        "case_keys": list(cases),
        "methods": list(profile.methods),
        "profile_digest": profile.digest(),
        "source_revision": revision,
        "source_tree_clean": source_tree_clean,
        "implicit_fallback_enabled": False,
        "hybrid_enabled": False,
        "package_file_sha256": {
            str(path.relative_to(root)): _sha256(path) for path in required
        },
    }
    print(json.dumps(payload, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
