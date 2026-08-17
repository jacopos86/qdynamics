from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pipelines.reporting import append_paper_i_phase0_route_pages as phase0


def test_retrieved_full_archive_closes_to_preserved_remote_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"authenticated full archive fixture"
    archive = tmp_path / "9605157.3_full.tar.gz"
    archive.write_bytes(payload)
    monkeypatch.setattr(phase0, "COMPLETED_DIR", tmp_path)
    completed = {
        "cluster_id": 9605157,
        "proc_id": 3,
        "source": {
            "full_archive": {
                "path": "/staging/jsstrobel/original.tar.gz",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        },
    }

    source = phase0.bind_retrieved_full_archive(
        completed, adapter_name="fixture"
    )

    assert source["local_archive"]["path"] == str(archive.resolve())
    assert source["remote_archive_at_retrieval"]["path"].startswith(
        "/staging/jsstrobel/"
    )
    assert source["remote_local_sha256_size_identity"] == "passed"


def test_retrieved_full_archive_rejects_identity_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "9605157.3_full.tar.gz").write_bytes(b"local")
    monkeypatch.setattr(phase0, "COMPLETED_DIR", tmp_path)
    completed = {
        "cluster_id": 9605157,
        "proc_id": 3,
        "source": {
            "full_archive": {
                "path": "/staging/jsstrobel/original.tar.gz",
                "sha256": hashlib.sha256(b"remote").hexdigest(),
                "size_bytes": len(b"remote"),
            }
        },
    }

    with pytest.raises(phase0.UpdateError, match="differs from remote"):
        phase0.bind_retrieved_full_archive(completed, adapter_name="fixture")
