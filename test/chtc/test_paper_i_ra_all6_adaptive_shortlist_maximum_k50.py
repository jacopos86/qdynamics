from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.static_adapt import adapt_pipeline
from pipelines.static_adapt.ra_adapt.contracts import (
    RAAdaptOperationalControls,
    canonical_sha256,
)
from pipelines.static_adapt.ra_adapt.engine import run_ra_adapt
from pipelines.static_adapt.ra_adapt.semantic_closure_routes import (
    build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request,
    build_paper_i_ra_hh_regime_problem,
    materialize_paper_i_ra_semantic_protocol,
)
from pipelines.static_adapt.sr_snake.contracts import (
    CheckpointObservation,
    EstimatorLedgerObservation,
    SRObservationPolicy,
)


RUNNER_PATH = (
    REPO_ROOT
    / "chtc/paper_i_ra_adapt_repair_20260727/"
    "run_local_paper_i_ra_all6_adaptive_shortlist_append_then_plateau_"
    "maximum_k50_20260817.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "paper_i_ra_all6_adaptive_maximum_k50",
        RUNNER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_public_cell_completion_accepts_authenticated_round_zero_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A real V2 terminal is complete at m=0 without a fabricated row."""

    runner = _load_runner()
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(adapt_pipeline, "_ai_log", lambda *_a, **_k: None)
    real_shortlist = adapt_pipeline._adaptive_phase_shortlist_with_receipt
    phase3_calls = 0

    def force_first_phase3_to_zero(
        records: Sequence[Mapping[str, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        nonlocal phase3_calls
        if kwargs.get("phase") == "phase_iii":
            phase3_calls += 1
            assert phase3_calls == 1
            assert records
            score_key = str(kwargs["score_key"])
            for record in records:
                assert isinstance(record, dict)
                record[score_key] = 0.0
        return real_shortlist(records, *args, **kwargs)

    monkeypatch.setattr(
        adapt_pipeline,
        "_adaptive_phase_shortlist_with_receipt",
        force_first_phase3_to_zero,
    )
    checkpoint_path = tmp_path / "current.json"
    ledger_path = tmp_path / "ledger.json"
    problem = build_paper_i_ra_hh_regime_problem("weak_weak")
    protocol = materialize_paper_i_ra_semantic_protocol(
        problem,
        build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request(
            insertion_policy="append_only",
            maximum_controller_rounds=50,
        ),
    )
    result = run_ra_adapt(
        problem,
        protocol,
        operational_controls=RAAdaptOperationalControls(
            maximum_controller_rounds=50,
            observation=SRObservationPolicy(
                checkpoint=CheckpointObservation(
                    path=checkpoint_path,
                    every_controller_rounds=1,
                    keep_history_tail=50,
                ),
                estimator_ledger=EstimatorLedgerObservation(path=ledger_path),
                resource_rounds=tuple(range(1, 51)),
            ),
        ),
    )
    assert result.run.paper_i_summary is None

    completion = runner.validate_cell_completion(
        runner.CELL_SPECS[0],
        result=result.to_dict(),
        summary=None,
        checkpoint_path=checkpoint_path,
    )

    assert completion["schema"] == (
        "paper_i_ra_all6_adaptive_maximum_k50_cell_completion_v1"
    )
    assert completion["completion_kind"] == (
        "authenticated_phase3_no_positive_natural_terminal_v1"
    )
    assert completion["maximum_controller_rounds"] == 50
    assert completion["accepted_controller_rounds"] == 0
    assert completion["terminal_attempted_controller_round"] == 1
    assert completion["summary_artifact_status"] == "not_applicable_round_zero"
    assert completion["checkpoint_file_sha256"] == hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()
    assert completion["accepted_trajectory_sha256"] == canonical_sha256([])
    assert completion["sha256"] == canonical_sha256(
        {key: value for key, value in completion.items() if key != "sha256"}
    )
