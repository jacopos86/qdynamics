"""Contract tests for the distributed Paper-III HH CHTC package."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from chtc.validate_condor_submit_lifecycle import validate_submit_lifecycle
from pipelines.qse_spectra.matched_route_acquisition import locked_hh_profile


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "chtc" / "paper_iii_selected_vs_direct_hh_20260828_v1"


def test_case_queue_is_exactly_the_locked_twelve_case_matrix() -> None:
    queued = tuple(
        line.strip()
        for line in (PACKAGE / "input" / "cases.txt").read_text().splitlines()
        if line.strip()
    )
    expected = tuple(case.case_key for case in locked_hh_profile().cases)

    assert queued == expected
    assert len(set(queued)) == 12


def test_submit_descriptor_is_bounded_explicit_and_has_no_hidden_route() -> None:
    text = (PACKAGE / "submit_cases.sub").read_text()

    validate_submit_lifecycle(text)
    assert "arguments = $(case_key)" in text
    assert "queue case_key from" in text
    assert "when_to_transfer_output = ON_EXIT_OR_EVICT" in text
    assert "transfer_output_files = case_result.json" in text
    assert "stream_output = False" in text
    assert "stream_error = False" in text
    assert '+JobBatchName = "paper-iii-selected-vs-direct-hh-20260828-v1"' in text
    assert "pipelines" in text
    assert "src" in text
    assert "image.sif" in text
    assert "hybrid" not in text.lower()
    assert "fallback" not in text.lower()


def test_package_validator_accepts_the_exact_materialized_descriptor() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(PACKAGE / "validate_package.py"),
            "--repo-root",
            str(ROOT),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["case_count"] == 12
    assert payload["route_count"] == 24
    assert payload["run_classification"] == "candidate"
    assert payload["implicit_fallback_enabled"] is False
    assert payload["hybrid_enabled"] is False
