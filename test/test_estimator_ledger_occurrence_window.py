"""Windowed occurrence accessors must agree with the full payload projection.

``EstimatorCallLedger.occurrence_count`` and ``occurrence_rows`` exist so the
selection kernel can ask "how many occurrences so far?" and "give me rows
``[start:end]``" without building ``to_payload()`` — three full passes, a
fingerprint digest, and two summaries — on every phase boundary.  They are a
cost strategy, never a semantics change, so the contract proven here is
equality with ``to_payload()["occurrences"]`` in both RAM and streaming mode,
across restore, and through the ``adapt_pipeline`` helpers and receipt
adapters that call them.

The stage-timing telemetry added alongside them is fail-soft and stdout-only:
it never feeds a selector decision, a receipt, or a checkpoint, so it must
never raise and never change a returned value.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any

import pytest

from pipelines.static_adapt import adapt_pipeline
from pipelines.static_adapt.estimator_call_ledger import (
    CALL_KEY_SCHEMA_V2,
    EstimatorCallKey,
    EstimatorCallLedger,
)

# The streaming (JSONL occurrence stream) ledger mode exists only on the
# Paper-I integration lineage; this branch's ledger is RAM-only.  The
# streaming-mode cases below skip rather than fail so they stay in force
# unchanged should that mode be ported here.
_STREAMING_SUPPORTED = "occurrence_stream_path" in inspect.signature(
    EstimatorCallLedger.__init__
).parameters
_requires_streaming = pytest.mark.skipif(
    not _STREAMING_SUPPORTED,
    reason=(
        "EstimatorCallLedger on this branch has no occurrence_stream_path "
        "(streaming) mode"
    ),
)


# Occurrence count driven into every ledger under test.  Small enough to stay
# fast and memory-light, large enough for the (3, 7) and (N - 2) windows.
N_CALLS = 12

_COMPONENTS = ("N_grad", "N_H_outer", "N_H_refit", "N_metric")


def _hex(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def _key(index: int) -> EstimatorCallKey:
    """Build a v2 call key the way test_estimator_ledger_streaming.py does.

    The modular arithmetic makes identities repeat inside ``N_CALLS`` calls,
    so the driven ledger mixes charged first-sightings with uncharged reuse
    occurrences.
    """

    return EstimatorCallKey(
        backend_fingerprint="compiled_exact_statevector_v1",
        hamiltonian_fingerprint=_hex(f"ham:{index % 2}"),
        observable_or_formula_identity="adapt_commutator_gradient_v1",
        operand_identity=f"candidate:{index % 3}",
        precision_contract="complex128_float64_deterministic_v1",
        primitive_kind="coordinate_gradient",
        projective_state_fingerprint=_hex(f"state:{index % 2}"),
        schema=CALL_KEY_SCHEMA_V2,
    )


def _drive(ledger: EstimatorCallLedger, calls: int = N_CALLS) -> None:
    """Record ``calls`` occurrences mixing identities, branches, and scopes."""

    for index in range(calls):
        ledger.record_call(
            _key(index),
            component=_COMPONENTS[index % 4],
            consumer_scope=f"round_{index % 3}:scope_{index % 2}",
            branch_id=None if index % 4 == 0 else f"branch_{index % 3}",
        )


def _payload_occurrences(ledger: EstimatorCallLedger) -> list[dict[str, Any]]:
    return list(ledger.to_payload()["occurrences"])


_WINDOWS: tuple[tuple[int, int | None], ...] = (
    (0, None),
    (3, 7),
    (N_CALLS, None),
    (0, N_CALLS + 5),
    (5, 5),
)


def _assert_window_equivalence(ledger: EstimatorCallLedger) -> None:
    """occurrence_count/occurrence_rows must equal the payload projection."""

    occurrences = _payload_occurrences(ledger)
    assert ledger.occurrence_count() == len(occurrences) == N_CALLS
    for start, end in _WINDOWS:
        assert ledger.occurrence_rows(start, end) == occurrences[start:end], (
            f"window ({start}, {end}) diverged from the payload slice"
        )


@pytest.fixture()
def ram_ledger() -> EstimatorCallLedger:
    ledger = EstimatorCallLedger()
    _drive(ledger)
    return ledger


@pytest.fixture()
def stream_path(tmp_path):
    return tmp_path / "occurrences.jsonl"


@pytest.fixture()
def streamed_ledger(stream_path) -> EstimatorCallLedger:
    if not _STREAMING_SUPPORTED:
        pytest.skip("streaming ledger mode is not available on this branch")
    ledger = EstimatorCallLedger(occurrence_stream_path=stream_path)
    _drive(ledger)
    return ledger


# ---------------------------------------------------------------------------
# RAM-mode ledger
# ---------------------------------------------------------------------------


def test_ram_ledger_windows_match_the_full_payload_projection(
    ram_ledger: EstimatorCallLedger,
) -> None:
    _assert_window_equivalence(ram_ledger)


def test_ram_ledger_default_window_is_every_row(
    ram_ledger: EstimatorCallLedger,
) -> None:
    assert ram_ledger.occurrence_rows() == _payload_occurrences(ram_ledger)


def test_ram_ledger_records_both_charged_and_reused_occurrences(
    ram_ledger: EstimatorCallLedger,
) -> None:
    # Guards the fixture itself: an all-charged ledger would not exercise the
    # repeated-identity path the windows are supposed to preserve.
    charged = [row["charged"] for row in _payload_occurrences(ram_ledger)]
    assert any(charged) and not all(charged)


def test_ram_ledger_rejects_a_negative_start(
    ram_ledger: EstimatorCallLedger,
) -> None:
    with pytest.raises(ValueError):
        ram_ledger.occurrence_rows(-1)


def test_ram_ledger_rejects_an_end_before_the_start(
    ram_ledger: EstimatorCallLedger,
) -> None:
    with pytest.raises(ValueError):
        ram_ledger.occurrence_rows(5, 4)


def test_empty_ram_ledger_reports_zero_occurrences() -> None:
    ledger = EstimatorCallLedger()
    assert ledger.occurrence_count() == 0
    assert ledger.occurrence_rows(0, None) == []
    assert ledger.occurrence_rows() == _payload_occurrences(ledger) == []


# ---------------------------------------------------------------------------
# Streaming-mode ledger (CHTC memory diet)
# ---------------------------------------------------------------------------


@_requires_streaming
def test_streamed_ledger_windows_match_the_full_payload_projection(
    streamed_ledger: EstimatorCallLedger,
) -> None:
    _assert_window_equivalence(streamed_ledger)


@_requires_streaming
def test_streamed_ledger_tail_window_returns_the_last_two_rows(
    streamed_ledger: EstimatorCallLedger,
) -> None:
    occurrences = _payload_occurrences(streamed_ledger)
    tail = streamed_ledger.occurrence_rows(N_CALLS - 2)
    assert tail == occurrences[-2:]
    assert [row["sequence"] for row in tail] == [N_CALLS - 1, N_CALLS]


@_requires_streaming
def test_streamed_ledger_reads_do_not_mutate_the_stream_file(
    streamed_ledger: EstimatorCallLedger,
    stream_path,
) -> None:
    before = stream_path.read_bytes()
    streamed_ledger.occurrence_count()
    for start, end in _WINDOWS:
        streamed_ledger.occurrence_rows(start, end)
    streamed_ledger.occurrence_rows(N_CALLS - 2)
    assert stream_path.read_bytes() == before


@_requires_streaming
def test_streamed_and_ram_ledgers_agree_row_for_row(
    ram_ledger: EstimatorCallLedger,
    streamed_ledger: EstimatorCallLedger,
) -> None:
    assert ram_ledger.occurrence_count() == streamed_ledger.occurrence_count()
    for start, end in _WINDOWS:
        assert ram_ledger.occurrence_rows(start, end) == (
            streamed_ledger.occurrence_rows(start, end)
        )


@_requires_streaming
def test_streamed_ledger_rejects_invalid_windows(
    streamed_ledger: EstimatorCallLedger,
) -> None:
    with pytest.raises(ValueError):
        streamed_ledger.occurrence_rows(-1)
    with pytest.raises(ValueError):
        streamed_ledger.occurrence_rows(5, 4)


# ---------------------------------------------------------------------------
# from_payload round trip
# ---------------------------------------------------------------------------


def test_restored_ledger_reports_the_same_count_and_rows(
    ram_ledger: EstimatorCallLedger,
) -> None:
    restored = EstimatorCallLedger.from_payload(ram_ledger.to_payload())
    assert restored.occurrence_count() == ram_ledger.occurrence_count()
    for start, end in _WINDOWS:
        assert restored.occurrence_rows(start, end) == (
            ram_ledger.occurrence_rows(start, end)
        )
    assert _payload_occurrences(restored) == _payload_occurrences(ram_ledger)


@_requires_streaming
def test_restored_streamed_ledger_matches_its_source(
    streamed_ledger: EstimatorCallLedger,
) -> None:
    restored = EstimatorCallLedger.from_payload(streamed_ledger.to_payload())
    assert restored.occurrence_count() == N_CALLS
    assert restored.occurrence_rows(0, None) == (
        streamed_ledger.occurrence_rows(0, None)
    )


def test_record_call_after_restore_continues_the_occurrence_count(
    ram_ledger: EstimatorCallLedger,
) -> None:
    restored = EstimatorCallLedger.from_payload(ram_ledger.to_payload())
    restored.record_call(
        _key(N_CALLS),
        component="N_H_outer",
        consumer_scope="post_restore_scope",
        branch_id="post_restore_branch",
    )
    assert restored.occurrence_count() == N_CALLS + 1
    occurrences = _payload_occurrences(restored)
    assert len(occurrences) == N_CALLS + 1
    assert restored.occurrence_rows(0, None) == occurrences
    appended = restored.occurrence_rows(N_CALLS)
    assert appended == occurrences[-1:]
    assert appended[0]["sequence"] == N_CALLS + 1
    assert appended[0]["consumer_scope"] == "post_restore_scope"


# ---------------------------------------------------------------------------
# adapt_pipeline ledger helpers
# ---------------------------------------------------------------------------


def test_pipeline_occurrence_helper_matches_the_payload_projection(
    ram_ledger: EstimatorCallLedger,
) -> None:
    rows = adapt_pipeline._default_no_prune_ledger_occurrences(ram_ledger)
    assert rows == _payload_occurrences(ram_ledger)
    assert all(isinstance(row, dict) for row in rows)


def test_pipeline_occurrence_helpers_absorb_a_missing_ledger() -> None:
    assert adapt_pipeline._default_no_prune_ledger_occurrences(None) == []
    assert adapt_pipeline._default_no_prune_ledger_occurrence_count(None) == 0
    assert adapt_pipeline._default_no_prune_ledger_occurrence_rows(None) == []
    assert (
        adapt_pipeline._default_no_prune_ledger_occurrence_rows(None, 3, 7) == []
    )


def test_pipeline_occurrence_count_and_rows_track_the_ledger(
    ram_ledger: EstimatorCallLedger,
) -> None:
    occurrences = _payload_occurrences(ram_ledger)
    assert adapt_pipeline._default_no_prune_ledger_occurrence_count(
        ram_ledger
    ) == len(occurrences)
    for start, end in _WINDOWS:
        assert adapt_pipeline._default_no_prune_ledger_occurrence_rows(
            ram_ledger, start, end
        ) == occurrences[start:end]


# ---------------------------------------------------------------------------
# Selection-receipt adapter accessors
# ---------------------------------------------------------------------------


def _unused(*_args: object, **_kwargs: object) -> Any:
    raise AssertionError("adapter entry point was not expected in this test")


def _receipts(**overrides: Any) -> Any:
    """Minimal _DefaultSelectionReceiptAdapters, as the route tests build it."""

    fields: dict[str, Any] = {
        "record_from_live": _unused,
        "shortlist_ranks": _unused,
        "ledger_occurrences": _unused,
        "restore_singleton_coordinates": _unused,
        "phase1_score_key": "phase1",
        "phase3_score_key": "phase3",
        "phase3_tie_break_score_key": "phase3_tie",
        "coordinate_solve_policy": "fixture",
    }
    fields.update(overrides)
    return adapt_pipeline._DefaultSelectionReceiptAdapters(**fields)


def test_receipt_accessors_fall_back_to_ledger_occurrences(
    ram_ledger: EstimatorCallLedger,
) -> None:
    occurrences = _payload_occurrences(ram_ledger)
    receipts = _receipts(ledger_occurrences=lambda: list(occurrences))

    # Built without the optional accessors, the adapter still answers through
    # the legacy full-materialization path.
    assert receipts.ledger_occurrence_count is None
    assert receipts.ledger_occurrence_rows is None
    assert adapt_pipeline._selection_receipt_occurrence_count(receipts) == (
        len(occurrences)
    )
    for start, end in _WINDOWS:
        assert adapt_pipeline._selection_receipt_occurrence_rows(
            receipts, start, end
        ) == occurrences[start:end]
    assert adapt_pipeline._selection_receipt_occurrence_rows(receipts, 4) == (
        occurrences[4:]
    )


def test_receipt_accessors_use_the_windowed_adapters_when_present(
    ram_ledger: EstimatorCallLedger,
) -> None:
    occurrences = _payload_occurrences(ram_ledger)
    calls: list[tuple[int, int | None]] = []

    def _rows(start: int, end: int | None) -> list[dict[str, Any]]:
        calls.append((start, end))
        return adapt_pipeline._default_no_prune_ledger_occurrence_rows(
            ram_ledger, start, end
        )

    receipts = _receipts(
        # ledger_occurrences stays forbidden: reaching it would mean the
        # windowed accessors were bypassed.
        ledger_occurrences=_unused,
        ledger_occurrence_count=lambda: (
            adapt_pipeline._default_no_prune_ledger_occurrence_count(ram_ledger)
        ),
        ledger_occurrence_rows=_rows,
    )

    assert adapt_pipeline._selection_receipt_occurrence_count(receipts) == (
        len(occurrences)
    )
    for start, end in _WINDOWS:
        assert adapt_pipeline._selection_receipt_occurrence_rows(
            receipts, start, end
        ) == occurrences[start:end]
    assert calls == list(_WINDOWS)


def test_receipt_row_accessor_defaults_its_end_to_none(
    ram_ledger: EstimatorCallLedger,
) -> None:
    seen: list[tuple[int, int | None]] = []

    def _rows(start: int, end: int | None) -> list[dict[str, Any]]:
        seen.append((start, end))
        return adapt_pipeline._default_no_prune_ledger_occurrence_rows(
            ram_ledger, start, end
        )

    receipts = _receipts(ledger_occurrence_rows=_rows)
    rows = adapt_pipeline._selection_receipt_occurrence_rows(receipts, 9)
    assert seen == [(9, None)]
    assert rows == _payload_occurrences(ram_ledger)[9:]


# ---------------------------------------------------------------------------
# Fail-soft stage telemetry
# ---------------------------------------------------------------------------


def test_stage_timing_never_raises_when_the_logger_raises(monkeypatch) -> None:
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("stdout is gone")

    monkeypatch.setattr(adapt_pipeline, "_ai_log", _boom)
    # Returning None rather than propagating is the whole contract: telemetry
    # sits inside the accepted-transition path and must never fail a run.
    assert (
        adapt_pipeline._emit_round_stage_timing(
            "hardcoded_adapt_test_event", depth=3, elapsed_s=0.125
        )
        is None
    )


def test_stage_timing_carries_its_fields_and_an_rss_sample(monkeypatch) -> None:
    captured: list[tuple[str, dict[str, Any]]] = []

    def _capture(event: str, **fields: Any) -> None:
        captured.append((event, dict(fields)))

    monkeypatch.setattr(adapt_pipeline, "_ai_log", _capture)
    adapt_pipeline._emit_round_stage_timing(
        "hardcoded_adapt_test_event", depth=2, elapsed_s=0.5
    )
    assert len(captured) == 1
    event, fields = captured[0]
    assert event == "hardcoded_adapt_test_event"
    assert fields["depth"] == 2
    assert fields["elapsed_s"] == 0.5
    assert "rss_mb" in fields and "rss_high_water_mb" in fields
    # The emitted payload must stay JSON-serializable; _ai_log dumps it.
    json.dumps(fields, sort_keys=True, default=str)


def test_process_rss_sample_reports_none_or_a_non_negative_float() -> None:
    sample = adapt_pipeline._process_rss_mb()
    assert set(sample) == {"rss_mb", "rss_high_water_mb"}
    for key, value in sample.items():
        assert value is None or isinstance(value, float), key
        if value is not None:
            assert value >= 0.0, key


def test_ledger_materialization_event_fires_only_above_the_threshold(
    monkeypatch,
    ram_ledger: EstimatorCallLedger,
) -> None:
    captured: list[tuple[str, dict[str, Any]]] = []

    def _capture(event: str, **fields: Any) -> None:
        captured.append((event, dict(fields)))

    monkeypatch.setattr(adapt_pipeline, "_ai_log", _capture)
    occurrences = _payload_occurrences(ram_ledger)

    monkeypatch.setattr(
        adapt_pipeline, "_LEDGER_MATERIALIZATION_LOG_THRESHOLD_S", 1.0e9
    )
    quiet_rows = adapt_pipeline._default_no_prune_ledger_occurrence_rows(
        ram_ledger, 3, 7
    )
    assert quiet_rows == occurrences[3:7]
    assert captured == []

    monkeypatch.setattr(
        adapt_pipeline, "_LEDGER_MATERIALIZATION_LOG_THRESHOLD_S", 0.0
    )
    loud_rows = adapt_pipeline._default_no_prune_ledger_occurrence_rows(
        ram_ledger, 3, 7
    )
    # The telemetry decision must not change the returned window.
    assert loud_rows == quiet_rows
    assert [event for event, _ in captured] == [
        "hardcoded_adapt_ledger_materialization"
    ]
    fields = captured[0][1]
    assert fields["start"] == 3
    assert fields["end"] == 7
    assert fields["row_count"] == 4
    assert isinstance(fields["elapsed_s"], float)
    assert fields["elapsed_s"] >= 0.0
    assert "rss_mb" in fields and "rss_high_water_mb" in fields


def test_ledger_materialization_event_reports_an_open_ended_window(
    monkeypatch,
    ram_ledger: EstimatorCallLedger,
) -> None:
    captured: list[dict[str, Any]] = []

    monkeypatch.setattr(
        adapt_pipeline,
        "_ai_log",
        lambda event, **fields: captured.append({"event": event, **fields}),
    )
    monkeypatch.setattr(
        adapt_pipeline, "_LEDGER_MATERIALIZATION_LOG_THRESHOLD_S", 0.0
    )
    rows = adapt_pipeline._default_no_prune_ledger_occurrence_rows(ram_ledger)
    assert rows == _payload_occurrences(ram_ledger)
    assert len(captured) == 1
    assert captured[0]["event"] == "hardcoded_adapt_ledger_materialization"
    assert captured[0]["start"] == 0
    assert captured[0]["end"] is None
    assert captured[0]["row_count"] == N_CALLS


def test_missing_ledger_emits_no_materialization_event(monkeypatch) -> None:
    captured: list[str] = []

    monkeypatch.setattr(
        adapt_pipeline,
        "_ai_log",
        lambda event, **_fields: captured.append(event),
    )
    monkeypatch.setattr(
        adapt_pipeline, "_LEDGER_MATERIALIZATION_LOG_THRESHOLD_S", 0.0
    )
    assert adapt_pipeline._default_no_prune_ledger_occurrence_rows(None) == []
    assert captured == []
