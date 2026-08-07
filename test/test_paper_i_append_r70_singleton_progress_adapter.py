from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from pipelines.reporting import (
    build_paper_i_append_r70_singleton_progress_adapter as adapter,
)


def _tar_member(bundle: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    bundle.addfile(info, io.BytesIO(payload))


def test_stream_compact_members_reads_full_closure_without_retaining_bulk(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "attempt.tar.gz"
    payloads = {
        name: (f"payload:{name}").encode("utf-8")
        for name in adapter.EXPECTED_ARCHIVE_MEMBERS
    }
    payloads["worker_outputs/payload/result.json"] = b"x" * 4096
    payloads["worker_outputs/payload/estimator_ledger.json"] = b"y" * 8192
    with tarfile.open(archive, "w:gz") as bundle:
        for name in sorted(payloads):
            _tar_member(bundle, name, payloads[name])

    captured, observed = adapter._stream_compact_members(archive)

    assert set(observed) == adapter.EXPECTED_ARCHIVE_MEMBERS
    assert set(captured) == adapter.COMPACT_MEMBERS
    assert "worker_outputs/payload/result.json" not in captured
    assert "worker_outputs/payload/estimator_ledger.json" not in captured
    for name, payload in payloads.items():
        assert observed[name] == {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }


def test_stream_compact_mode_accepts_only_the_exact_retained_member_set(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "compact.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name in sorted(adapter.COMPACT_MEMBERS):
            _tar_member(bundle, name, name.encode("utf-8"))

    captured, observed = adapter._stream_compact_members(
        archive, expected_members=adapter.COMPACT_MEMBERS
    )

    assert set(captured) == adapter.COMPACT_MEMBERS
    assert set(observed) == adapter.COMPACT_MEMBERS
    with pytest.raises(adapter.AdapterInputError, match="member closure"):
        adapter._stream_compact_members(archive)


def _identity_payloads() -> tuple[dict[str, object], dict[str, object], str]:
    regime = "strong_weak_u8"
    execution_id = adapter._execution_id(regime)
    digest = "a" * 64
    size = 1234
    execution = {
        "execution_id": execution_id,
        "cluster_id": adapter.CLUSTER_ID,
        "proc_id": adapter.PROC_BY_REGIME[regime],
        "attempt_ordinal": 1,
        "source_horizon": 50,
        "target_horizon": 70,
        "fresh_start": True,
        "resume_claimed": False,
    }
    retrieval = {
        "local_archive": {"sha256": digest, "size_bytes": size},
        "remote_archive_sha256": digest,
        "remote_archive_size_bytes": size,
        "remote_local_hash_size_match": True,
        "expected_final_basename_match": True,
    }
    name = (
        f"{execution_id}__cluster_{adapter.CLUSTER_ID}__proc_"
        f"{adapter.PROC_BY_REGIME[regime]}.tar.gz"
    )
    return execution, retrieval, name


def test_archive_identity_requires_exact_receipt_hash_size_and_attempt() -> None:
    execution, retrieval, name = _identity_payloads()

    assert adapter._validate_archive_identity(
        execution=execution,
        retrieval=retrieval,
        archive_name=name,
        archive_sha256="a" * 64,
        archive_size_bytes=1234,
    ) == "strong_weak_u8"

    tampered = dict(retrieval)
    tampered["remote_archive_sha256"] = "b" * 64
    with pytest.raises(adapter.AdapterInputError, match="archive/retrieval"):
        adapter._validate_archive_identity(
            execution=execution,
            retrieval=tampered,
            archive_name=name,
            archive_sha256="a" * 64,
            archive_size_bytes=1234,
        )

    wrong_attempt = dict(execution)
    wrong_attempt["proc_id"] = 4
    with pytest.raises(adapter.AdapterInputError, match="execution identity"):
        adapter._validate_archive_identity(
            execution=wrong_attempt,
            retrieval=retrieval,
            archive_name=name,
            archive_sha256="a" * 64,
            archive_size_bytes=1234,
        )


def _summary(exact: float) -> dict[str, object]:
    history = []
    before = exact + 1.0
    for controller_round in range(1, 71):
        after = exact + 1.0 / (controller_round + 1)
        history.append(
            {
                "controller_round": controller_round,
                "energy_before": before,
                "energy_after": after,
            }
        )
        before = after
    return {
        "accepted_history": history,
        "final_energy": before,
    }


def test_trace_is_exact_zero_through_seventy_and_rejects_discontinuity() -> None:
    summary = _summary(-0.5)
    points = adapter._trace(
        regime="intermediate_weak", summary=summary, exact_energy=-0.5
    )
    assert [row["round"] for row in points] == list(range(71))
    assert points[50]["delta_e"] == pytest.approx(1.0 / 51.0)
    assert points[70]["delta_e"] == pytest.approx(1.0 / 71.0)

    broken = json.loads(json.dumps(summary))
    broken["accepted_history"][20]["energy_before"] += 0.1
    with pytest.raises(adapter.AdapterInputError, match="continuity"):
        adapter._trace(
            regime="intermediate_weak",
            summary=broken,
            exact_energy=-0.5,
        )


def test_complete_singleton_closure_uses_all_six_expected_cluster_procs() -> None:
    assert adapter.COMPLETED_REGIMES == adapter.REGIME_ORDER
    assert adapter.PENDING_REGIMES == ()
    assert adapter.PROC_BY_REGIME == {
        "weak_weak": 1,
        "intermediate_weak": 3,
        "strong_weak_u8": 5,
        "weak_strong": 7,
        "intermediate_strong": 9,
        "strong_strong_u8": 11,
    }


def test_build_adapter_emits_six_completed_zero_pending_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exact = {regime: -float(index + 1) for index, regime in enumerate(adapter.REGIME_ORDER)}

    monkeypatch.setattr(
        adapter,
        "_ed_reference",
        lambda: (
            exact,
            {
                "path": "locked-ed.json",
                "sha256": adapter.ED_REFERENCE_SHA256,
                "schema": "test_ed_v1",
                "cutoff_rule": "job_nph_equals_working_cutoff_v1",
            },
        ),
    )

    def fake_source(*, receipt_path: Path, archive_path: Path) -> dict[str, object]:
        del archive_path
        regime = receipt_path.stem
        return {
            "regime": regime,
            "job": {
                "execution_id": adapter._execution_id(regime),
                "regime_id": regime,
            },
            "summary": _summary(exact[regime]),
            "checkpoint": {"regime": regime},
            "source": {"receipt": regime},
        }

    monkeypatch.setattr(adapter, "_validate_receipt_and_archive", fake_source)

    def fake_compile(cells: list[dict[str, object]]) -> dict[str, object]:
        result = {}
        for index, cell in enumerate(cells, start=1):
            execution_id = cell["job"]["execution_id"]
            rounds = {}
            for controller_round in (50, 70):
                rounds[f"round_{controller_round}"] = {
                    "checkpoint_sha256": f"{index:x}" * 64,
                    "costs": {
                        "N2q": 100 + controller_round,
                        "D2q": 80 + controller_round,
                        "Dc": 200 + controller_round,
                        "W1q": 90 + controller_round,
                        "S_alg": index * 100_000 + controller_round,
                    },
                    "compile": {
                        "compile_convention": "table_i_basis_gate_transpile_v1"
                    },
                }
            result[execution_id] = {"execution_id": execution_id, **rounds}
        return result

    monkeypatch.setattr(adapter, "_compile_costs_for_cells", fake_compile)
    sources = [
        (Path(f"{regime}.json"), Path(f"{regime}.tar.gz"))
        for regime in adapter.COMPLETED_REGIMES
    ]
    output = tmp_path / "adapter.json"

    first = adapter.build_adapter(sources=sources, output=output)
    second = adapter.build_adapter(sources=sources, output=output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert first["status"] == "written"
    assert second["status"] == "already_current"
    assert adapter.verify_self_digest(payload, label="adapter") == payload["sha256"]
    assert payload["completed_regimes"] == list(adapter.COMPLETED_REGIMES)
    assert payload["pending_regimes"] == list(adapter.PENDING_REGIMES)
    assert [row["regime_id"] for row in payload["cells"]] == [
        regime for regime in adapter.REGIME_ORDER if regime in adapter.COMPLETED_REGIMES
    ]
    assert payload["cost_policy"]["round_50"]["classification"] == (
        "canonical_paper_comparable"
    )
    assert payload["cost_policy"]["round_70"]["classification"] == (
        "diagnostic_extension"
    )
    assert all(len(row["points"]) == 71 for row in payload["cells"])


def test_mixed_source_dispatch_keeps_compact_transport_limit_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        adapter,
        "_validate_receipt_and_archive",
        lambda **kwargs: {"mode": "receipt", **kwargs},
    )
    monkeypatch.setattr(
        adapter,
        "_validate_direct_full_archive",
        lambda **kwargs: {"mode": "direct", **kwargs},
    )
    monkeypatch.setattr(
        adapter,
        "_validate_compact_archive",
        lambda **kwargs: {"mode": "compact", **kwargs},
    )

    receipt = adapter._load_mixed_source(
        {
            "mode": adapter.SOURCE_MODE_RECEIPT_FULL,
            "receipt": "receipt.json",
            "archive": "full.tar.gz",
        }
    )
    direct = adapter._load_mixed_source(
        {
            "mode": adapter.SOURCE_MODE_DIRECT_FULL,
            "archive": "direct.tar.gz",
        }
    )
    compact = adapter._load_mixed_source(
        {
            "mode": adapter.SOURCE_MODE_COMPACT,
            "archive": "compact.tar.gz",
            "compact_sha256": "a" * 64,
            "remote_full_archive_sha256": "b" * 64,
            "remote_full_archive_size_bytes": 235895048,
            "remote_observed_utc": "2026-08-01T14:29:52Z",
        }
    )

    assert receipt["mode"] == "receipt"
    assert direct["mode"] == "direct"
    assert compact == {
        "mode": "compact",
        "archive_path": Path("compact.tar.gz"),
        "compact_sha256": "a" * 64,
        "remote_full_archive_sha256": "b" * 64,
        "remote_full_archive_size_bytes": 235895048,
        "remote_observed_utc": "2026-08-01T14:29:52Z",
    }


def test_remote_compact_observation_requires_strict_utc() -> None:
    assert adapter._validated_observed_utc("2026-08-01T14:29:52Z") == (
        "2026-08-01T14:29:52Z"
    )
    with pytest.raises(adapter.AdapterInputError, match="end in Z"):
        adapter._validated_observed_utc("2026-08-01T14:29:52-05:00")
