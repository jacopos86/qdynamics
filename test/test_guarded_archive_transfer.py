"""Fail-closed tests for resumable incremental archive transfers."""

from __future__ import annotations

from dataclasses import replace
import errno
import hashlib
import io
import json
import os
import stat
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipelines.time_dynamics.guarded_archive_transfer import (
    ArchiveTransferSpec,
    TRANSFER_FINALIZATION_INTENT_V1,
    TRANSFER_QUARANTINE_INTENT_V1,
    TransferRecoveryError,
    finalize_archive_transfer,
    prepare_partial_transfer,
    stream_archive_transfer,
)
from pipelines.time_dynamics import guarded_archive_transfer as guarded


def _archive_bytes() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        payload = b'{"status":"complete"}\n'
        info = tarfile.TarInfo("raw_outputs/example/run.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _spec(tmp_path: Path, payload: bytes) -> ArchiveTransferSpec:
    return ArchiveTransferSpec(
        remote_path="/home/jsstrobel/frozen/delta.tar.gz",
        local_final_path=(tmp_path / "evidence" / "delta.tar.gz").resolve(),
        expected_bytes=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_member_count=1,
    )


def _signed_intent(spec: ArchiveTransferSpec) -> dict[str, object]:
    intent: dict[str, object] = {
        "schema": TRANSFER_FINALIZATION_INTENT_V1,
        "status": "verified_part_ready_for_atomic_rename",
        "remote_path": spec.remote_path,
        "local_part_path": str(spec.part_path),
        "local_final_path": str(spec.local_final_path),
        "archive_bytes": spec.expected_bytes,
        "archive_sha256": spec.expected_sha256,
        "tar_member_count": spec.expected_member_count,
        "remote_preservation_required": True,
    }
    intent["intent_sha256"] = hashlib.sha256(
        json.dumps(
            intent, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    return intent


def test_absent_partial_starts_from_zero_without_inventing_proof(tmp_path: Path) -> None:
    payload = _archive_bytes()
    decision = prepare_partial_transfer(
        _spec(tmp_path, payload),
        remote_prefix_sha256=lambda _length: pytest.fail("callback was invoked"),
    )
    assert decision.status == "start_empty"
    assert decision.resume_offset == 0
    assert decision.receipt_path is None


def test_first_use_fsyncs_each_new_confinement_parent_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _archive_bytes()
    spec = ArchiveTransferSpec(
        remote_path="/home/jsstrobel/frozen/delta.tar.gz",
        local_final_path=(
            tmp_path / "new-a" / "new-b" / "evidence" / "delta.tar.gz"
        ).absolute(),
        expected_bytes=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_member_count=1,
    )
    events: list[tuple[str, tuple[int, int], str | None]] = []
    real_mkdir = os.mkdir
    real_fsync = os.fsync

    def traced_mkdir(path, mode=0o777, *, dir_fd=None):
        if dir_fd is None:
            return real_mkdir(path, mode)
        metadata = os.fstat(dir_fd)
        identity = (metadata.st_dev, metadata.st_ino)
        result = real_mkdir(path, mode, dir_fd=dir_fd)
        events.append(("mkdir", identity, os.fspath(path)))
        return result

    def traced_fsync(descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        if stat.S_ISDIR(metadata.st_mode):
            events.append(
                ("fsync", (metadata.st_dev, metadata.st_ino), None)
            )
        real_fsync(descriptor)

    monkeypatch.setattr(guarded.os, "mkdir", traced_mkdir)
    monkeypatch.setattr(guarded.os, "fsync", traced_fsync)

    guarded.append_transfer_chunk(
        spec,
        expected_offset=0,
        chunk=payload[:1],
        chunk_sha256=hashlib.sha256(payload[:1]).hexdigest(),
    )

    mkdir_events = [
        (index, identity)
        for index, (kind, identity, _component) in enumerate(events)
        if kind == "mkdir"
    ]
    assert len(mkdir_events) == 3
    for mkdir_index, parent_identity in mkdir_events:
        assert any(
            index > mkdir_index
            and kind == "fsync"
            and identity == parent_identity
            for index, (kind, identity, _component) in enumerate(events)
        )


def test_exact_remote_prefix_is_the_only_resumable_partial(tmp_path: Path) -> None:
    payload = _archive_bytes()
    spec = _spec(tmp_path, payload)
    spec.part_path.parent.mkdir(parents=True)
    prefix = payload[: len(payload) // 2]
    spec.part_path.write_bytes(prefix)

    decision = prepare_partial_transfer(
        spec,
        remote_prefix_sha256=lambda length: hashlib.sha256(
            payload[:length]
        ).hexdigest(),
    )

    assert decision.status == "resume_verified_prefix"
    assert decision.resume_offset == len(prefix)
    assert spec.part_path.read_bytes() == prefix
    receipt = json.loads(decision.receipt_path.read_text())  # type: ignore[union-attr]
    assert receipt["prefix_proof"]["matches"] is True
    assert receipt["prefix_proof"]["length"] == len(prefix)


def test_mismatched_partial_is_quarantined_before_restart(tmp_path: Path) -> None:
    payload = _archive_bytes()
    spec = _spec(tmp_path, payload)
    spec.part_path.parent.mkdir(parents=True)
    bad = b"not the remote prefix"
    spec.part_path.write_bytes(bad)

    decision = prepare_partial_transfer(
        spec,
        remote_prefix_sha256=lambda length: hashlib.sha256(
            payload[:length]
        ).hexdigest(),
    )

    assert decision.status == "restart_after_quarantine"
    assert decision.resume_offset == 0
    assert not spec.part_path.exists()
    assert decision.quarantine_path is not None
    assert decision.quarantine_path.read_bytes() == bad
    receipt = json.loads(decision.receipt_path.read_text())  # type: ignore[union-attr]
    assert receipt["status"] == "quarantined_unverified_partial"
    assert receipt["prefix_proof"]["matches"] is False
    assert receipt["quarantine_path"] == str(decision.quarantine_path)
    unsigned = dict(receipt)
    declared = unsigned.pop("recovery_receipt_sha256")
    assert declared == hashlib.sha256(
        json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def test_unavailable_prefix_proof_quarantines_instead_of_resuming(
    tmp_path: Path,
) -> None:
    payload = _archive_bytes()
    spec = _spec(tmp_path, payload)
    spec.part_path.parent.mkdir(parents=True)
    spec.part_path.write_bytes(payload[:17])

    decision = prepare_partial_transfer(
        spec,
        remote_prefix_sha256=lambda _length: (_ for _ in ()).throw(
            OSError("transport unavailable")
        ),
    )

    assert decision.status == "restart_after_quarantine"
    receipt = json.loads(decision.receipt_path.read_text())  # type: ignore[union-attr]
    assert receipt["prefix_proof"]["remote_prefix_sha256"] is None
    assert "transport unavailable" in receipt["detail"]


def test_quarantine_never_overwrites_an_existing_target(tmp_path: Path) -> None:
    payload = _archive_bytes()
    spec = _spec(tmp_path, payload)
    bad = b"wrong"
    digest = hashlib.sha256(bad).hexdigest()
    expected_target = spec.effective_quarantine_root / (
        f"{spec.part_path.name}.{digest[:16]}"
    )
    expected_target.parent.mkdir(parents=True)
    expected_target.write_bytes(b"older evidence")
    spec.part_path.parent.mkdir(parents=True, exist_ok=True)
    spec.part_path.write_bytes(bad)

    decision = prepare_partial_transfer(
        spec,
        remote_prefix_sha256=lambda _length: "0" * 64,
    )

    assert expected_target.read_bytes() == b"older evidence"
    assert decision.quarantine_path != expected_target
    assert decision.quarantine_path is not None
    assert decision.quarantine_path.read_bytes() == bad


def test_quarantine_allocation_survives_a_racing_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _archive_bytes()
    spec = _spec(tmp_path, payload)
    spec.part_path.parent.mkdir(parents=True)
    bad = b"racing partial"
    spec.part_path.write_bytes(bad)
    real_link = os.link
    collisions = 0

    def collide_once(
        source,
        destination,
        *,
        src_dir_fd=None,
        dst_dir_fd=None,
        follow_symlinks=True,
    ):
        nonlocal collisions
        if os.fspath(source) in {
            os.fspath(spec.part_path),
            spec.part_path.name,
        } and collisions == 0:
            collisions += 1
            target = (
                Path(destination)
                if dst_dir_fd is None
                else spec.effective_quarantine_root / os.fspath(destination)
            )
            target.write_bytes(b"concurrent evidence")
            raise FileExistsError(errno.EEXIST, "simulated collision", destination)
        kwargs = {"follow_symlinks": follow_symlinks}
        if src_dir_fd is not None:
            kwargs["src_dir_fd"] = src_dir_fd
        if dst_dir_fd is not None:
            kwargs["dst_dir_fd"] = dst_dir_fd
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(guarded.os, "link", collide_once)
    decision = prepare_partial_transfer(
        spec, remote_prefix_sha256=lambda _length: "0" * 64
    )

    assert collisions == 1
    assert decision.quarantine_path is not None
    assert decision.quarantine_path.read_bytes() == bad
    sibling = decision.quarantine_path.with_name(
        decision.quarantine_path.name.rsplit(".", 1)[0]
    )
    assert sibling.read_bytes() == b"concurrent evidence"


def test_quarantine_move_has_durable_intent_before_source_can_be_unlinked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _archive_bytes()
    spec = _spec(tmp_path, payload)
    spec.part_path.parent.mkdir(parents=True)
    spec.part_path.write_bytes(b"unsafe partial")
    real_link = os.link

    def fail_quarantine_link(
        source,
        destination,
        *,
        src_dir_fd=None,
        dst_dir_fd=None,
        follow_symlinks=True,
    ):
        if os.fspath(source) in {
            os.fspath(spec.part_path),
            spec.part_path.name,
        }:
            raise OSError(errno.EIO, "simulated crash before quarantine link")
        kwargs = {"follow_symlinks": follow_symlinks}
        if src_dir_fd is not None:
            kwargs["src_dir_fd"] = src_dir_fd
        if dst_dir_fd is not None:
            kwargs["dst_dir_fd"] = dst_dir_fd
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(guarded.os, "link", fail_quarantine_link)
    with pytest.raises(OSError, match="simulated crash"):
        prepare_partial_transfer(
            spec, remote_prefix_sha256=lambda _length: "0" * 64
        )

    assert spec.part_path.read_bytes() == b"unsafe partial"
    intents = tuple(
        (spec.local_final_path.parent / "quarantine_intents").glob("*.json")
    )
    assert len(intents) == 1
    intent = json.loads(intents[0].read_text())
    unsigned = dict(intent)
    declared = unsigned.pop("intent_sha256")
    assert intent["schema"] == TRANSFER_QUARANTINE_INTENT_V1
    assert intent["source_path"] == str(spec.part_path)
    assert declared == hashlib.sha256(
        json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def test_quarantine_fsyncs_linked_bytes_before_source_unlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _archive_bytes()
    spec = _spec(tmp_path, payload)
    spec.part_path.parent.mkdir(parents=True)
    spec.part_path.write_bytes(b"unsafe partial")
    source_identity = (
        spec.part_path.stat().st_dev,
        spec.part_path.stat().st_ino,
    )
    events: list[str] = []
    real_fsync = os.fsync
    real_unlink = os.unlink

    def traced_fsync(descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        if stat.S_ISREG(metadata.st_mode) and (
            metadata.st_dev,
            metadata.st_ino,
        ) == source_identity:
            events.append("quarantine_bytes_fsync")
        real_fsync(descriptor)

    def traced_unlink(path, *args, **kwargs):
        if os.fspath(path) in {
            os.fspath(spec.part_path),
            spec.part_path.name,
        }:
            events.append("source_unlink")
            assert "quarantine_bytes_fsync" in events
            assert events.index("quarantine_bytes_fsync") < events.index(
                "source_unlink"
            )
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(guarded.os, "fsync", traced_fsync)
    monkeypatch.setattr(guarded.os, "unlink", traced_unlink)

    decision = prepare_partial_transfer(
        spec, remote_prefix_sha256=lambda _length: "0" * 64
    )

    assert decision.status == "restart_after_quarantine"
    assert events.count("quarantine_bytes_fsync") >= 1
    assert events.count("source_unlink") == 1


def test_complete_part_is_verified_renamed_and_reverified(tmp_path: Path) -> None:
    payload = _archive_bytes()
    spec = _spec(tmp_path, payload)
    spec.part_path.parent.mkdir(parents=True)
    spec.part_path.write_bytes(payload)

    receipt = finalize_archive_transfer(spec)

    assert not spec.part_path.exists()
    assert spec.local_final_path.read_bytes() == payload
    assert receipt["archive_sha256"] == spec.expected_sha256
    assert receipt["archive_bytes"] == spec.expected_bytes
    assert receipt["gzip_integrity"] is True
    assert receipt["tar_integrity"] is True
    assert receipt["tar_member_count"] == 1
    assert receipt["remote_preservation_required"] is True


def test_preexisting_complete_part_is_fsynced_before_finalization_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _archive_bytes()
    spec = _spec(tmp_path, payload)
    spec.part_path.parent.mkdir(parents=True)
    spec.part_path.write_bytes(payload)
    part_identity = (spec.part_path.stat().st_dev, spec.part_path.stat().st_ino)
    events: list[str] = []
    real_fsync = os.fsync
    real_write = guarded._write_json_exclusive

    def traced_fsync(descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        if stat.S_ISREG(metadata.st_mode) and (
            metadata.st_dev,
            metadata.st_ino,
        ) == part_identity:
            events.append("part_bytes_fsync")
        real_fsync(descriptor)

    def stop_after_intent(
        path: Path, body: dict[str, object], **kwargs
    ) -> None:
        if path == spec.effective_finalization_intent_path:
            events.append("intent_publication")
            assert "part_bytes_fsync" in events
            assert events.index("part_bytes_fsync") < events.index(
                "intent_publication"
            )
            raise RuntimeError("stop after ordering assertion")
        real_write(path, body, **kwargs)

    monkeypatch.setattr(guarded.os, "fsync", traced_fsync)
    monkeypatch.setattr(guarded, "_write_json_exclusive", stop_after_intent)

    with pytest.raises(RuntimeError, match="ordering assertion"):
        finalize_archive_transfer(spec)
    assert spec.part_path.read_bytes() == payload
    assert not spec.local_final_path.exists()


def test_final_publication_never_overwrites_racing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _archive_bytes()
    spec = _spec(tmp_path, payload)
    spec.part_path.parent.mkdir(parents=True)
    spec.part_path.write_bytes(payload)
    competitor = b"concurrent immutable evidence"
    real_write = guarded._write_json_exclusive

    def publish_competitor_after_intent(
        path: Path, body: dict[str, object], **kwargs
    ) -> None:
        real_write(path, body, **kwargs)
        if path == spec.effective_finalization_intent_path:
            spec.local_final_path.write_bytes(competitor)

    monkeypatch.setattr(
        guarded, "_write_json_exclusive", publish_competitor_after_intent
    )

    with pytest.raises(TransferRecoveryError, match="final archive.*exists"):
        finalize_archive_transfer(spec)

    assert spec.local_final_path.read_bytes() == competitor
    assert spec.part_path.read_bytes() == payload


def test_chunk_append_rejects_terminal_part_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    payload = _archive_bytes()
    spec = _spec(tmp_path, payload)
    spec.part_path.parent.mkdir(parents=True)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    spec.part_path.symlink_to(outside)

    with pytest.raises(TransferRecoveryError, match="regular|symlink|nofollow"):
        guarded.append_transfer_chunk(
            spec,
            expected_offset=len(b"outside"),
            chunk=b"x",
            chunk_sha256=hashlib.sha256(b"x").hexdigest(),
        )

    assert outside.read_bytes() == b"outside"


def test_prefix_proof_rejects_a_same_length_partial_name_swap(
    tmp_path: Path,
) -> None:
    payload = _archive_bytes()
    spec = _spec(tmp_path, payload)
    prefix = payload[: max(1, len(payload) // 3)]
    replacement = bytes(value ^ 0xFF for value in prefix)
    spec.part_path.parent.mkdir(parents=True)
    spec.part_path.write_bytes(prefix)

    def swap_after_remote_proof(length: int) -> str:
        assert length == len(prefix)
        spec.part_path.rename(spec.part_path.with_suffix(".pinned"))
        spec.part_path.write_bytes(replacement)
        return hashlib.sha256(prefix).hexdigest()

    with pytest.raises(TransferRecoveryError, match="changed.*prefix"):
        prepare_partial_transfer(
            spec, remote_prefix_sha256=swap_after_remote_proof
        )

    assert spec.part_path.read_bytes() == replacement
    assert not spec.effective_recovery_receipt_path.exists()


def test_finalization_rejects_wrong_complete_bytes(tmp_path: Path) -> None:
    payload = _archive_bytes()
    spec = _spec(tmp_path, payload)
    spec.part_path.parent.mkdir(parents=True)
    spec.part_path.write_bytes(payload + b"tamper")

    with pytest.raises(TransferRecoveryError, match="binding mismatch"):
        finalize_archive_transfer(spec)
    assert not spec.part_path.exists()
    assert not spec.local_final_path.exists()
    quarantined = tuple(spec.effective_quarantine_root.iterdir())
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == payload + b"tamper"
    receipt = json.loads(spec.effective_recovery_receipt_path.read_text())
    assert receipt["status"] == "quarantined_invalid_complete_partial"


def test_existing_final_must_match_and_may_not_coexist_with_part(
    tmp_path: Path,
) -> None:
    payload = _archive_bytes()
    spec = _spec(tmp_path, payload)
    spec.local_final_path.parent.mkdir(parents=True)
    spec.local_final_path.write_bytes(payload)
    spec.part_path.write_bytes(payload[:3])

    with pytest.raises(TransferRecoveryError, match="coexist"):
        prepare_partial_transfer(
            spec,
            remote_prefix_sha256=lambda length: hashlib.sha256(
                payload[:length]
            ).hexdigest(),
        )


def test_repeated_quarantines_allocate_immutable_attempt_receipts(
    tmp_path: Path,
) -> None:
    payload = _archive_bytes()
    spec = _spec(tmp_path, payload)
    spec.part_path.parent.mkdir(parents=True)
    spec.part_path.write_bytes(b"first bad partial")
    first = prepare_partial_transfer(
        spec, remote_prefix_sha256=lambda _length: "0" * 64
    )
    spec.part_path.write_bytes(b"second bad partial")
    second = prepare_partial_transfer(
        spec, remote_prefix_sha256=lambda _length: "0" * 64
    )
    assert first.receipt_path != second.receipt_path
    assert first.receipt_path is not None and first.receipt_path.is_file()
    assert second.receipt_path is not None and second.receipt_path.is_file()
    assert json.loads(first.receipt_path.read_text())["local"]["part_sha256"] != (
        json.loads(second.receipt_path.read_text())["local"]["part_sha256"]
    )


def test_invalid_existing_final_is_quarantined_without_overwrite(
    tmp_path: Path,
) -> None:
    payload = _archive_bytes()
    spec = _spec(tmp_path, payload)
    spec.local_final_path.parent.mkdir(parents=True)
    spec.local_final_path.write_bytes(b"wrong final")

    decision = prepare_partial_transfer(
        spec, remote_prefix_sha256=lambda _length: pytest.fail("not needed")
    )

    assert decision.status == "restart_after_quarantine"
    assert not spec.local_final_path.exists()
    assert decision.quarantine_path is not None
    assert decision.quarantine_path.read_bytes() == b"wrong final"
    receipt = json.loads(decision.receipt_path.read_text())  # type: ignore[union-attr]
    assert receipt["status"] == "quarantined_invalid_final"
    assert receipt["local"]["source_kind"] == "final"


def test_finalization_recovers_crash_after_intent_and_atomic_rename(
    tmp_path: Path,
) -> None:
    payload = _archive_bytes()
    spec = _spec(tmp_path, payload)
    spec.local_final_path.parent.mkdir(parents=True)
    spec.local_final_path.write_bytes(payload)
    spec.effective_finalization_intent_path.write_text(
        json.dumps(_signed_intent(spec)) + "\n"
    )

    receipt = finalize_archive_transfer(spec)
    assert receipt["recovered_after_atomic_rename"] is True
    assert spec.effective_finalization_receipt_path.is_file()
    replay = finalize_archive_transfer(spec)
    assert replay == receipt


def test_finalization_replay_consumes_same_inode_partial_hardlink(
    tmp_path: Path,
) -> None:
    payload = _archive_bytes()
    spec = _spec(tmp_path, payload)
    spec.part_path.parent.mkdir(parents=True)
    spec.part_path.write_bytes(payload)
    spec.effective_finalization_intent_path.write_text(
        json.dumps(_signed_intent(spec)) + "\n"
    )
    os.link(spec.part_path, spec.local_final_path)

    receipt = finalize_archive_transfer(spec)

    assert receipt["recovered_after_atomic_rename"] is True
    assert not spec.part_path.exists()
    assert spec.local_final_path.read_bytes() == payload


@pytest.mark.parametrize("kind", ("intent", "receipt"))
def test_finalization_rejects_terminal_evidence_symlinks_without_reading_target(
    tmp_path: Path,
    kind: str,
) -> None:
    payload = _archive_bytes()
    spec = _spec(tmp_path, payload)
    spec.part_path.parent.mkdir(parents=True)
    spec.part_path.write_bytes(payload)
    finalize_archive_transfer(spec)
    target_path = (
        spec.effective_finalization_intent_path
        if kind == "intent"
        else spec.effective_finalization_receipt_path
    )
    outside = tmp_path / f"outside-{kind}.json"
    outside.write_bytes(target_path.read_bytes())
    expected = outside.read_bytes()
    target_path.unlink()
    target_path.symlink_to(outside)

    with pytest.raises(TransferRecoveryError, match="nofollow|regular"):
        finalize_archive_transfer(spec)

    assert outside.read_bytes() == expected


def test_existing_final_rejects_tampered_intent_and_receipt(
    tmp_path: Path,
) -> None:
    payload = _archive_bytes()
    spec = _spec(tmp_path, payload)
    spec.part_path.parent.mkdir(parents=True)
    spec.part_path.write_bytes(payload)
    finalize_archive_transfer(spec)

    intent = json.loads(spec.effective_finalization_intent_path.read_text())
    intent["archive_bytes"] += 1
    spec.effective_finalization_intent_path.write_text(json.dumps(intent) + "\n")
    with pytest.raises(TransferRecoveryError, match="intent binding"):
        finalize_archive_transfer(spec)


def test_existing_final_rejects_malformed_receipt_as_typed_error(
    tmp_path: Path,
) -> None:
    payload = _archive_bytes()
    spec = _spec(tmp_path, payload)
    spec.part_path.parent.mkdir(parents=True)
    spec.part_path.write_bytes(payload)
    finalize_archive_transfer(spec)
    spec.effective_finalization_receipt_path.write_text("{not-json\n")

    with pytest.raises(TransferRecoveryError, match="receipt is unreadable"):
        finalize_archive_transfer(spec)


def test_existing_final_rejects_self_digest_valid_json_receipt_tamper(
    tmp_path: Path,
) -> None:
    payload = _archive_bytes()
    spec = _spec(tmp_path, payload)
    spec.part_path.parent.mkdir(parents=True)
    spec.part_path.write_bytes(payload)
    finalize_archive_transfer(spec)
    receipt = json.loads(spec.effective_finalization_receipt_path.read_text())
    receipt["archive_bytes"] += 1
    spec.effective_finalization_receipt_path.write_text(
        json.dumps(receipt) + "\n"
    )

    with pytest.raises(TransferRecoveryError, match="receipt binding"):
        finalize_archive_transfer(spec)


def test_stream_transfer_starts_at_zero_and_is_binary_safe(tmp_path: Path) -> None:
    payload = _archive_bytes()
    spec = _spec(tmp_path, payload)
    offsets: list[tuple[int, int]] = []

    def chunk(offset: int, length: int) -> bytes:
        offsets.append((offset, length))
        return payload[offset : offset + length]

    receipt = stream_archive_transfer(
        spec,
        remote_prefix_sha256=lambda length: hashlib.sha256(
            payload[:length]
        ).hexdigest(),
        remote_chunk=chunk,
        chunk_bytes=17,
        free_floor_bytes=0,
    )
    assert offsets[0] == (0, 17)
    assert b"".join(payload[offset : offset + length] for offset, length in offsets) == payload
    assert spec.local_final_path.read_bytes() == payload
    assert receipt["archive_sha256"] == spec.expected_sha256
    assert receipt["remote_preservation_required"] is True


def test_stream_transfer_resumes_only_after_exact_prefix_proof(tmp_path: Path) -> None:
    payload = _archive_bytes()
    spec = _spec(tmp_path, payload)
    prefix_length = len(payload) // 3
    spec.part_path.parent.mkdir(parents=True)
    spec.part_path.write_bytes(payload[:prefix_length])
    offsets: list[int] = []

    stream_archive_transfer(
        spec,
        remote_prefix_sha256=lambda length: hashlib.sha256(
            payload[:length]
        ).hexdigest(),
        remote_chunk=lambda offset, length: (
            offsets.append(offset) or payload[offset : offset + length]
        ),
        chunk_bytes=19,
        free_floor_bytes=0,
    )
    assert offsets[0] == prefix_length
    assert spec.local_final_path.read_bytes() == payload


def test_stream_transfer_enforces_free_space_floor_before_writing(
    tmp_path: Path,
) -> None:
    payload = _archive_bytes()
    spec = _spec(tmp_path, payload)
    with pytest.raises(TransferRecoveryError, match="free-space floor"):
        stream_archive_transfer(
            spec,
            remote_prefix_sha256=lambda length: hashlib.sha256(
                payload[:length]
            ).hexdigest(),
            remote_chunk=lambda offset, length: payload[offset : offset + length],
            free_floor_bytes=10**30,
        )
    assert not spec.part_path.exists()


def test_storage_guard_uses_remaining_bytes_and_extraction_reserve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _archive_bytes()
    spec = _spec(tmp_path, payload)
    split = len(payload) - 5
    spec.part_path.parent.mkdir(parents=True)
    spec.part_path.write_bytes(payload[:split])
    free = 1000
    floor = 900
    monkeypatch.setattr(
        guarded.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=free, used=0, free=free),
    )

    stream_archive_transfer(
        spec,
        remote_prefix_sha256=lambda length: hashlib.sha256(
            payload[:length]
        ).hexdigest(),
        remote_chunk=lambda offset, length: payload[offset : offset + length],
        free_floor_bytes=floor,
        extraction_reserve_bytes=90,
    )
    assert spec.local_final_path.read_bytes() == payload

    other = _spec(tmp_path / "blocked", payload)
    with pytest.raises(TransferRecoveryError, match="free-space floor"):
        stream_archive_transfer(
            other,
            remote_prefix_sha256=lambda length: hashlib.sha256(
                payload[:length]
            ).hexdigest(),
            remote_chunk=lambda offset, length: payload[offset : offset + length],
            free_floor_bytes=floor,
            extraction_reserve_bytes=90,
        )


def test_storage_guard_cannot_ignore_spec_bound_extraction_reserve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _archive_bytes()
    spec = replace(_spec(tmp_path, payload), extraction_reserve_bytes=90)
    split = len(payload) - 5
    spec.part_path.parent.mkdir(parents=True)
    spec.part_path.write_bytes(payload[:split])
    monkeypatch.setattr(
        guarded.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=994, used=0, free=994),
    )

    with pytest.raises(TransferRecoveryError, match="free-space floor"):
        stream_archive_transfer(
            spec,
            remote_prefix_sha256=lambda length: hashlib.sha256(
                payload[:length]
            ).hexdigest(),
            remote_chunk=lambda offset, length: payload[offset : offset + length],
            free_floor_bytes=900,
        )


def test_storage_guard_is_rechecked_during_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _archive_bytes()
    spec = _spec(tmp_path, payload)
    observations = iter((10_000, 10_000, 0))
    monkeypatch.setattr(
        guarded.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(
            total=10_000, used=0, free=next(observations)
        ),
    )
    with pytest.raises(TransferRecoveryError, match="free-space floor"):
        stream_archive_transfer(
            spec,
            remote_prefix_sha256=lambda length: hashlib.sha256(
                payload[:length]
            ).hexdigest(),
            remote_chunk=lambda offset, length: payload[offset : offset + length],
            chunk_bytes=17,
            free_floor_bytes=1,
        )
    assert spec.part_path.stat().st_size == 17


def test_already_complete_replay_does_not_apply_a_false_download_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _archive_bytes()
    spec = _spec(tmp_path, payload)
    spec.part_path.parent.mkdir(parents=True)
    spec.part_path.write_bytes(payload)
    first = finalize_archive_transfer(spec)
    monkeypatch.setattr(
        guarded.shutil,
        "disk_usage",
        lambda _path: pytest.fail("complete replay must not subtract download bytes"),
    )
    replay = stream_archive_transfer(
        spec,
        remote_prefix_sha256=lambda _length: pytest.fail("no prefix request"),
        remote_chunk=lambda _offset, _length: pytest.fail("no chunk request"),
        free_floor_bytes=10**30,
        extraction_reserve_bytes=10**30,
    )
    assert replay == first


def test_corrupt_gzip_with_registered_full_digest_is_quarantined(
    tmp_path: Path,
) -> None:
    payload = b"not a gzip archive"
    spec = ArchiveTransferSpec(
        remote_path="/home/jsstrobel/frozen/corrupt.tar.gz",
        local_final_path=(tmp_path / "corrupt.tar.gz").resolve(),
        expected_bytes=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_member_count=1,
    )
    spec.part_path.write_bytes(payload)
    with pytest.raises(TransferRecoveryError, match="integrity"):
        finalize_archive_transfer(spec)
    assert not spec.part_path.exists()
    assert not spec.local_final_path.exists()
    assert tuple(spec.effective_quarantine_root.iterdir())


def test_member_count_is_part_of_the_immutable_binding(tmp_path: Path) -> None:
    payload = _archive_bytes()
    base = _spec(tmp_path, payload)
    spec = replace(base, expected_member_count=2)
    spec.part_path.parent.mkdir(parents=True)
    spec.part_path.write_bytes(payload)
    with pytest.raises(TransferRecoveryError, match="member-count"):
        finalize_archive_transfer(spec)
    assert not spec.local_final_path.exists()
