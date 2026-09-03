"""Compact-layout checkpoints must resume through the canonical reader.

Why (Paper-IV H2O campaign, 2026-09-03): the campaign runs bounded segments
that exit normally and continue through ``--resume-checkpoint``.  The opt-in
``STATIC_ADAPT_CHECKPOINT_JSON_LAYOUT=compact`` writer (3b1b86f4) changes the
bytes of ``current.json`` and of the content-addressed estimator-ledger
sidecar.  ``sr_snake/_resume.py`` authenticates the sidecar by file sha256
against the pointer and rejects on mismatch, and no existing test wrote a
checkpoint through the production writer under ``compact`` and loaded it
back.  These tests republish the resume-adapter fixture bundle through the
production writers (sidecar first, pointer, then ``current.json``, mirroring
``_write_current_checkpoint``) and require the canonical reader to accept
it under every layout combination a multi-segment campaign can produce.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.static_adapt import current_checkpoint as cc
from pipelines.static_adapt.sr_snake._resume import (
    CanonicalAcceptedStateHydration,
    CanonicalResumeError,
    load_canonical_accepted_state_resume,
)
from test_paper_i_sr_snake_resume_adapter import (
    _resume,
    _write_checkpoint_bundle,
)

_POINTER_FIELD = "estimator_call_ledger_checkpoint"


def _set_layout(monkeypatch: pytest.MonkeyPatch, layout: str | None) -> None:
    if layout is None:
        monkeypatch.delenv(cc._CHECKPOINT_JSON_LAYOUT_ENV, raising=False)
    else:
        monkeypatch.setenv(cc._CHECKPOINT_JSON_LAYOUT_ENV, layout)


def _is_single_line_document(path: Path) -> bool:
    data = path.read_bytes()
    return data.endswith(b"\n") and b"\n" not in data[:-1]


def _republish_through_production_writers(
    current: Path,
    sidecar: Path,
    *,
    sidecar_layout: str | None,
    current_layout: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, str]:
    """Rewrite the fixture bundle with ``current_checkpoint``'s writers.

    Order mirrors ``_write_current_checkpoint``: the sidecar is written
    content-addressed and its (path, sha256) is placed into both pointer
    copies before ``current.json`` is written.
    """

    envelope = json.loads(current.read_text(encoding="utf-8"))
    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    sidecar.unlink()

    _set_layout(monkeypatch, sidecar_layout)
    new_sidecar, sidecar_sha256 = cc._atomic_write_content_addressed_json(
        current,
        filename_role=_POINTER_FIELD,
        payload=sidecar_payload,
    )
    for owner in (envelope["adapt_vqe"], envelope["checkpoint"]):
        pointer = dict(owner[_POINTER_FIELD])
        pointer["path"] = new_sidecar.name
        pointer["sha256"] = sidecar_sha256
        owner[_POINTER_FIELD] = pointer

    _set_layout(monkeypatch, current_layout)
    cc._atomic_write_json(current, envelope)
    _set_layout(monkeypatch, None)
    return new_sidecar, sidecar_sha256


def _load(problem, profile, digest, current: Path) -> CanonicalAcceptedStateHydration:
    return load_canonical_accepted_state_resume(
        _resume(current),
        expected_problem=problem,
        expected_route_profile=profile,
        expected_route_contract_sha256=digest,
    )


def test_compact_layout_checkpoint_and_sidecar_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem, profile, digest, current, sidecar = _write_checkpoint_bundle(tmp_path)
    pretty_sidecar_size = sidecar.stat().st_size
    pretty_current_size = current.stat().st_size

    new_sidecar, sidecar_sha256 = _republish_through_production_writers(
        current,
        sidecar,
        sidecar_layout="compact",
        current_layout="compact",
        monkeypatch=monkeypatch,
    )

    # The files really are compact, and the pointer digest is the file digest.
    assert _is_single_line_document(new_sidecar)
    assert _is_single_line_document(current)
    assert new_sidecar.stat().st_size < pretty_sidecar_size
    assert current.stat().st_size < pretty_current_size
    assert cc._sha256_file(new_sidecar) == sidecar_sha256
    assert new_sidecar.name == (
        f"current.{_POINTER_FIELD}.{sidecar_sha256[:16]}.json"
    )
    assert not sidecar.exists()

    hydration = _load(problem, profile, digest, current)

    assert isinstance(hydration, CanonicalAcceptedStateHydration)
    assert hydration.controller_round == 1
    assert hydration.s_alg == 1
    assert hydration.s_unique == 1
    assert hydration.estimator_prefix_checkpoint_cursor["raw_occurrence_count"] == 1
    assert hydration.mutable_estimator_call_ledger_payload()["ledger_fingerprint"]


def test_default_layout_reproduces_the_fixture_bytes_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem, profile, digest, current, sidecar = _write_checkpoint_bundle(tmp_path)
    original_sidecar_name = sidecar.name
    original_sidecar_bytes = sidecar.read_bytes()
    original_current_bytes = current.read_bytes()

    new_sidecar, _sidecar_sha256 = _republish_through_production_writers(
        current,
        sidecar,
        sidecar_layout=None,
        current_layout=None,
        monkeypatch=monkeypatch,
    )

    # Default (unset) layout is the historical indent=2 document byte for byte,
    # so the content-addressed name and both files are unchanged.
    assert new_sidecar.name == original_sidecar_name
    assert new_sidecar.read_bytes() == original_sidecar_bytes
    assert current.read_bytes() == original_current_bytes
    assert _load(problem, profile, digest, current).controller_round == 1


@pytest.mark.parametrize(
    ("sidecar_layout", "current_layout"),
    [("compact", None), (None, "compact"), ("pretty", "compact")],
)
def test_layout_switch_between_campaign_segments_still_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sidecar_layout: str | None,
    current_layout: str | None,
) -> None:
    """A segment written under one layout resumes under the other.

    The reader authenticates bytes actually on disk, never a re-encoding, so
    the layout in force when a segment resumes is irrelevant to acceptance.
    """

    problem, profile, digest, current, sidecar = _write_checkpoint_bundle(tmp_path)
    new_sidecar, sidecar_sha256 = _republish_through_production_writers(
        current,
        sidecar,
        sidecar_layout=sidecar_layout,
        current_layout=current_layout,
        monkeypatch=monkeypatch,
    )
    assert _is_single_line_document(new_sidecar) == (sidecar_layout == "compact")
    assert _is_single_line_document(current) == (current_layout == "compact")
    assert cc._sha256_file(new_sidecar) == sidecar_sha256

    # Resume with the *other* layout in force for any subsequent writes.
    _set_layout(monkeypatch, None if current_layout == "compact" else "compact")
    assert _load(problem, profile, digest, current).controller_round == 1


def test_compact_sidecar_tampering_is_still_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem, profile, digest, current, sidecar = _write_checkpoint_bundle(tmp_path)
    new_sidecar, _sidecar_sha256 = _republish_through_production_writers(
        current,
        sidecar,
        sidecar_layout="compact",
        current_layout="compact",
        monkeypatch=monkeypatch,
    )
    new_sidecar.write_bytes(new_sidecar.read_bytes() + b" ")

    with pytest.raises(CanonicalResumeError, match="sidecar SHA-256"):
        _load(problem, profile, digest, current)


def test_compact_current_json_outer_hash_is_the_file_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``h2o_application`` hands the reader sha256(file bytes); it must match."""

    problem, profile, digest, current, sidecar = _write_checkpoint_bundle(tmp_path)
    _republish_through_production_writers(
        current,
        sidecar,
        sidecar_layout="compact",
        current_layout="compact",
        monkeypatch=monkeypatch,
    )
    resume = _resume(current)
    assert resume.checkpoint_sha256 == cc._sha256_file(current)
    hydration = load_canonical_accepted_state_resume(
        resume,
        expected_problem=problem,
        expected_route_profile=profile,
        expected_route_contract_sha256=digest,
    )
    assert hydration.controller_round == 1
