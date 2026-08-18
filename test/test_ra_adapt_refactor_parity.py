"""Characterization parity tests for the adapt_pipeline refactor.

Golden-snapshot tests that pin observable RA-ADAPT behavior before any
refactor code movement (retirement of legacy routes and archaic
comparators, then module extraction). Each test runs a bounded real
RA-ADAPT continuation on the small Hubbard--Holstein fixture problem and
compares a normalized behavior payload against a committed baseline.

On first execution a missing baseline is recorded under
``test/fixtures/ra_refactor_parity/`` and the test is skipped; every
later execution must reproduce the baseline exactly. If behavior changes
*intentionally*, delete the affected baseline file and re-run to
re-record — that deletion is a deliberate scientific decision, not a
routine fix.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from pipelines.static_adapt.ra_adapt import run_ra_adapt
from test_ra_adapt_facade import (
    _hh_problem,
    _validated_macro_protocol,
    _validated_singleton_protocol,
)

BASELINE_DIR = Path(__file__).parent / "fixtures" / "ra_refactor_parity"

# Keys whose values vary between identical runs (timing, machine, file
# locations) and therefore carry no parity information.
_VOLATILE_KEY_MARKERS = (
    "wallclock",
    "timestamp",
    "time_s",
    "duration",
    "elapsed",
    "hostname",
    "path",
    "pid",
)

_FLOAT_DECIMALS = 10


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _normalize(item)
            for key, item in sorted(value.items())
            if not any(
                marker in key.lower() for marker in _VOLATILE_KEY_MARKERS
            )
        }
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return round(value, _FLOAT_DECIMALS)
    if isinstance(value, (str, int)):
        return value
    return repr(value)


def _stop_payload(stop: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name in dir(stop):
        if name.startswith("_"):
            continue
        attribute = getattr(stop, name)
        if isinstance(attribute, (str, int, float, bool)):
            payload[name] = attribute
    return payload


def _parity_payload(result: Any) -> dict[str, Any]:
    return _normalize(
        {
            "result_schema": result.schema,
            "protocol_sha256": result.protocol.sha256,
            "completed_controller_rounds": (
                result.run.stop.completed_controller_rounds
            ),
            "stop": _stop_payload(result.run.stop),
            "accepted_trajectory": [
                row.to_dict() for row in result.run.accepted_trajectory
            ],
        }
    )


def _assert_matches_baseline(name: str, payload: dict[str, Any]) -> None:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    baseline_file = BASELINE_DIR / f"{name}.json"
    if not baseline_file.exists():
        baseline_file.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        pytest.skip(f"parity baseline recorded: {baseline_file.name}")
    baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
    assert payload == baseline, (
        f"RA parity deviation against {baseline_file.name}; if this "
        "change is intentional, delete the baseline and re-record"
    )


def test_parity_macro_append_only_bounded() -> None:
    problem = _hh_problem()
    protocol = _validated_macro_protocol(problem, rounds=3)
    result = run_ra_adapt(problem, protocol)
    _assert_matches_baseline(
        "macro_append_only_r3", _parity_payload(result)
    )


def test_parity_singleton_plateau_bounded() -> None:
    problem = _hh_problem()
    protocol = _validated_singleton_protocol(problem, rounds=3)
    result = run_ra_adapt(problem, protocol)
    _assert_matches_baseline(
        "singleton_plateau_r3", _parity_payload(result)
    )
