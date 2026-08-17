from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
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
    build_paper_i_ra_all_phase_position_adaptive_request,
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


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "paper_i_ra_all6_maximum_k50_completion_contract_runner",
        RUNNER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


def _rehash(mapping: dict[str, Any]) -> None:
    mapping["sha256"] = canonical_sha256(
        {key: value for key, value in mapping.items() if key != "sha256"}
    )


@pytest.fixture(scope="module")
def authenticated_round_zero_terminal(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    """Produce one real V2 terminal and reuse its authenticated evidence."""

    artifact_dir = tmp_path_factory.mktemp("maximum-k50-round-zero")
    checkpoint_path = artifact_dir / "current.json"
    ledger_path = artifact_dir / "ledger.json"
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

    problem = build_paper_i_ra_hh_regime_problem("weak_weak")
    protocol = materialize_paper_i_ra_semantic_protocol(
        problem,
        build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request(
            insertion_policy="append_only",
            maximum_controller_rounds=50,
        ),
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
        monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
        monkeypatch.setattr(adapt_pipeline, "_ai_log", lambda *_a, **_k: None)
        monkeypatch.setattr(
            adapt_pipeline,
            "_adaptive_phase_shortlist_with_receipt",
            force_first_phase3_to_zero,
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

    assert phase3_calls == 1
    assert result.run.paper_i_summary is None
    return {
        "result": result.to_dict(),
        "checkpoint_path": checkpoint_path,
        "artifact_dir": artifact_dir,
    }


@pytest.fixture(scope="module")
def authenticated_round_one_terminal(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    """Accept round one, then terminate on the real second Phase III."""

    artifact_dir = tmp_path_factory.mktemp("maximum-k50-round-one")
    checkpoint_path = artifact_dir / "current.json"
    ledger_path = artifact_dir / "ledger.json"
    real_shortlist = adapt_pipeline._adaptive_phase_shortlist_with_receipt
    phase3_calls = 0

    def force_second_phase3_to_zero(
        records: Sequence[Mapping[str, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        nonlocal phase3_calls
        if kwargs.get("phase") == "phase_iii":
            phase3_calls += 1
            assert records
            if phase3_calls == 2:
                score_key = str(kwargs["score_key"])
                for record in records:
                    assert isinstance(record, dict)
                    record[score_key] = 0.0
        return real_shortlist(records, *args, **kwargs)

    problem = build_paper_i_ra_hh_regime_problem("weak_weak")
    protocol = materialize_paper_i_ra_semantic_protocol(
        problem,
        build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request(
            insertion_policy="append_only",
            maximum_controller_rounds=50,
        ),
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
        monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
        monkeypatch.setattr(adapt_pipeline, "_ai_log", lambda *_a, **_k: None)
        monkeypatch.setattr(
            adapt_pipeline,
            "_adaptive_phase_shortlist_with_receipt",
            force_second_phase3_to_zero,
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

    assert phase3_calls == 2
    assert result.run.paper_i_summary is not None
    result_mapping = result.to_dict()
    return {
        "result": result_mapping,
        "summary": result_mapping["run"]["paper_i_summary"],
        "checkpoint_path": checkpoint_path,
    }


def _validate(
    evidence: Mapping[str, Any],
    *,
    result: Mapping[str, Any] | None = None,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    return RUNNER.validate_cell_completion(
        RUNNER.CELL_SPECS[0],
        result=(evidence["result"] if result is None else result),
        summary=None,
        checkpoint_path=(
            evidence["checkpoint_path"]
            if checkpoint_path is None
            else checkpoint_path
        ),
    )


def test_completion_contract_accepts_real_authenticated_round_zero_terminal(
    authenticated_round_zero_terminal: Mapping[str, Any],
) -> None:
    completion = _validate(authenticated_round_zero_terminal)

    assert completion["completion_kind"] == (
        "authenticated_phase3_no_positive_natural_terminal_v1"
    )
    assert completion["accepted_controller_rounds"] == 0
    assert completion["terminal_attempted_controller_round"] == 1
    assert completion["summary_artifact_status"] == (
        "not_applicable_round_zero"
    )
    assert completion["sha256"] == canonical_sha256(
        {key: value for key, value in completion.items() if key != "sha256"}
    )


def test_completion_contract_accepts_real_round_one_then_terminal_without_fake_row_two(
    authenticated_round_one_terminal: Mapping[str, Any],
) -> None:
    result = authenticated_round_one_terminal["result"]
    summary = authenticated_round_one_terminal["summary"]
    completion = RUNNER.validate_cell_completion(
        RUNNER.CELL_SPECS[0],
        result=result,
        summary=summary,
        checkpoint_path=authenticated_round_one_terminal["checkpoint_path"],
    )

    assert completion["accepted_controller_rounds"] == 1
    assert completion["terminal_attempted_controller_round"] == 2
    assert completion["summary_artifact_status"] == "present"
    assert completion["paper_i_summary_sha256"] == canonical_sha256(summary)
    run = result["run"]
    for accepted_rows in (
        run["accepted_trajectory"],
        run["accepted_transitions"],
        run["scientific_replay"],
    ):
        assert [row["controller_round"] for row in accepted_rows] == [1]
    assert [
        row["accepted_round_ordinal"]
        for row in result["scientific_receipts"]["accepted_round_receipts"]
    ] == [1]
    assert [
        row["controller_round"] for row in summary["accepted_error_trace"]
    ] == [1]
    assert len(run["canonical_reporting"]["accepted_prefix_work"]) == 1
    terminal_work = run["estimator_accounting"]["all_work"]
    accepted_prefix_work = run["canonical_reporting"][
        "accepted_prefix_work"
    ][0]
    assert terminal_work["s_alg"] >= accepted_prefix_work["s_alg"]
    for component in ("n_h_outer", "n_h_refit", "n_grad", "n_metric"):
        assert terminal_work["components"][component] >= (
            accepted_prefix_work["components"][component]
        )


def test_completion_contract_rejects_rehashed_terminal_round_tamper(
    authenticated_round_zero_terminal: Mapping[str, Any],
) -> None:
    tampered = copy.deepcopy(authenticated_round_zero_terminal["result"])
    terminal = tampered["scientific_receipts"][
        "terminal_phase3_selection_receipt"
    ]
    terminal["attempted_controller_round"] = 2
    _rehash(terminal)

    with pytest.raises(RUNNER.RunnerError):
        _validate(authenticated_round_zero_terminal, result=tampered)


def test_completion_contract_rejects_rehashed_terminal_count_tamper(
    authenticated_round_zero_terminal: Mapping[str, Any],
) -> None:
    tampered = copy.deepcopy(authenticated_round_zero_terminal["result"])
    terminal = tampered["scientific_receipts"][
        "terminal_phase3_selection_receipt"
    ]
    terminal["accepted_operator_count"] = 1
    _rehash(terminal)

    with pytest.raises(RUNNER.RunnerError):
        _validate(authenticated_round_zero_terminal, result=tampered)


def test_completion_contract_rejects_fully_rehashed_replay_state_tamper(
    authenticated_round_zero_terminal: Mapping[str, Any],
) -> None:
    tampered = copy.deepcopy(authenticated_round_zero_terminal["result"])
    scientific = tampered["scientific_receipts"]
    replay = scientific["controller_replay_evidence"]
    replay_terminal = replay["phase3_no_positive_terminal"]
    replay_terminal["round_zero_accepted_state"]["energy"] += 1.0
    _rehash(replay_terminal)
    sidecar = replay["resume_sidecar_closure"]
    sidecar["phase3_no_positive_terminal_sha256"] = replay_terminal["sha256"]
    _rehash(sidecar)
    _rehash(replay)
    scientific["controller_replay_evidence_sha256"] = replay["sha256"]

    with pytest.raises(RUNNER.RunnerError):
        _validate(authenticated_round_zero_terminal, result=tampered)


def test_completion_contract_rejects_result_final_state_tamper(
    authenticated_round_zero_terminal: Mapping[str, Any],
) -> None:
    tampered = copy.deepcopy(authenticated_round_zero_terminal["result"])
    tampered["run"]["final_state"]["energy"] += 1.0

    with pytest.raises(RUNNER.RunnerError):
        _validate(authenticated_round_zero_terminal, result=tampered)


def test_completion_contract_rejects_rehashed_checkpoint_sidecar_tamper(
    authenticated_round_zero_terminal: Mapping[str, Any],
) -> None:
    source_checkpoint_path = Path(
        authenticated_round_zero_terminal["checkpoint_path"]
    )
    checkpoint = json.loads(source_checkpoint_path.read_text(encoding="utf-8"))
    source_pointer = checkpoint["adapt_vqe"][
        "estimator_call_ledger_checkpoint"
    ]
    source_sidecar_path = source_checkpoint_path.parent / source_pointer["path"]
    sidecar = json.loads(source_sidecar_path.read_text(encoding="utf-8"))
    sidecar["S_alg"] += 1
    sidecar_bytes = json.dumps(sidecar, sort_keys=True).encode("utf-8")
    sidecar_sha256 = hashlib.sha256(sidecar_bytes).hexdigest()
    tampered_dir = (
        Path(authenticated_round_zero_terminal["artifact_dir"])
        / "rehashed-sidecar-tamper"
    )
    tampered_dir.mkdir()
    sidecar_name = (
        "current.estimator_call_ledger_checkpoint."
        f"{sidecar_sha256[:16]}.json"
    )
    tampered_sidecar_path = tampered_dir / sidecar_name
    tampered_sidecar_path.write_bytes(sidecar_bytes)
    for owner in (checkpoint["adapt_vqe"], checkpoint["checkpoint"]):
        owner["estimator_call_ledger_checkpoint"]["path"] = sidecar_name
        owner["estimator_call_ledger_checkpoint"]["sha256"] = sidecar_sha256
    tampered_checkpoint_path = tampered_dir / "current.json"
    tampered_checkpoint_path.write_text(
        json.dumps(checkpoint, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(RUNNER.RunnerError):
        _validate(
            authenticated_round_zero_terminal,
            checkpoint_path=tampered_checkpoint_path,
        )


def test_completion_contract_rejects_short_result_without_typed_terminal(
    authenticated_round_zero_terminal: Mapping[str, Any],
) -> None:
    tampered = copy.deepcopy(authenticated_round_zero_terminal["result"])
    tampered["run"]["stop"]["terminal_controller_outcome"] = None
    tampered["run"]["stop"]["primary_reason"] = "maximum_controller_rounds"
    tampered["scientific_receipts"].pop(
        "terminal_phase3_selection_receipt", None
    )

    with pytest.raises(RUNNER.RunnerError):
        _validate(authenticated_round_zero_terminal, result=tampered)


def test_completion_contract_rejects_v1_route_with_terminal_shaped_result(
    authenticated_round_zero_terminal: Mapping[str, Any],
) -> None:
    tampered = copy.deepcopy(authenticated_round_zero_terminal["result"])
    problem = build_paper_i_ra_hh_regime_problem("weak_weak")
    v1_protocol = materialize_paper_i_ra_semantic_protocol(
        problem,
        build_paper_i_ra_all_phase_position_adaptive_request(
            insertion_policy="append_only",
            maximum_controller_rounds=50,
        ),
    )
    tampered["protocol"] = v1_protocol.to_dict()
    tampered["scientific_receipts"]["resolved_route_contract"] = copy.deepcopy(
        dict(v1_protocol.route_contract)
    )

    with pytest.raises(RUNNER.RunnerError):
        _validate(authenticated_round_zero_terminal, result=tampered)
