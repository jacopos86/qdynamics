from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_continuation_symmetry import (
    build_symmetry_spec,
    leakage_penalty_from_spec,
    verify_symmetry_sequence,
)


def test_build_symmetry_spec_shares_metadata_for_gating_and_verification() -> None:
    spec = build_symmetry_spec(family_id="paop_lf_std", mitigation_mode="verify_only")
    leak = leakage_penalty_from_spec(spec)
    verify = verify_symmetry_sequence(
        generator_metadata=[{"symmetry_spec": spec.__dict__}],
        mitigation_mode="verify_only",
    )
    assert leak == 0.0
    assert verify["executed"] is True
    assert verify["passed"] is True
    assert verify["max_leakage_risk"] == 0.0


def test_verify_symmetry_sequence_fails_for_high_risk_metadata() -> None:
    verify = verify_symmetry_sequence(
        generator_metadata=[{"symmetry_spec": {"leakage_risk": 0.9}}],
        mitigation_mode="verify_only",
    )
    assert verify["executed"] is True
    assert verify["passed"] is False
    assert verify["high_risk_count"] == 1


def test_off_mode_preserves_legacy_behavior() -> None:
    verify = verify_symmetry_sequence(
        generator_metadata=[{"symmetry_spec": {"leakage_risk": 0.9}}],
        mitigation_mode="off",
    )
    assert verify["executed"] is False
    assert verify["passed"] is True
    assert verify["mode"] == "off"
