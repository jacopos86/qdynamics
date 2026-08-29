"""Behavior tests for the automated Paper-III matched-route acquisition."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pipelines.qse_spectra.matched_route_acquisition import (
    DIRECT_STATUS_DIMENSION_CAP_REACHED,
    DIRECT_TERMINAL_STATUSES,
    LocalScientificRouteExecutor,
    METHOD_DIRECT_DAVIDSON,
    METHOD_SELECTED_QSE,
    MatchedRouteAcquisitionError,
    RouteExecutionResult,
    SELECTED_STATUS_CERTIFIED,
    SELECTED_STATUS_POOL_EXHAUSTED,
    acquire_selected_vs_direct_hh_case,
    collect_selected_vs_direct_hh_cases,
    ensure_selected_vs_direct_hh_campaign,
    locked_hh_profile,
)


class _LiteralExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def execute(self, case, method_id, profile) -> RouteExecutionResult:
        self.calls.append((case.case_key, method_id))
        selected = method_id == METHOD_SELECTED_QSE
        completion = {
            "schema_version": "literal_online_completion_v1",
            "method_id": method_id,
            "completion_route": (
                "selected_qse_fixed_pool" if selected else "direct_davidson"
            ),
            "davidson_invoked": not selected,
            "online_only": True,
            "terminal_status": (
                SELECTED_STATUS_CERTIFIED
                if selected
                else DIRECT_STATUS_DIMENSION_CAP_REACHED
            ),
        }
        row = {
            "k": 8,
            "support_size": 8,
            "retained_rank": 8,
            "n2q": 12.0 if selected else 24.0,
            "d2q": 8.0 if selected else 16.0,
            "dc": 18.0 if selected else 36.0,
            "root_energies": [
                0.1000005,
                0.2000005,
                0.3000005,
                0.4000005,
                0.5000005,
                0.6000005,
            ],
            "estimator": {
                "pair_count": 36,
                "queries": 612,
                "convention": "literal_test_estimator_v1",
            },
        }
        if selected:
            row["support_indices"] = list(range(8))
        return RouteExecutionResult(
            online_rows=(row,),
            completion=completion,
            audit_rows=(
                {
                    "k": 8,
                    "max_root_abs_error": 5.0e-7,
                    "per_root_abs_error": [5.0e-7] * 6,
                },
            ),
            reference_excitations=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
            pool_digest="ab" * 32,
            elapsed_seconds=0.01,
        )


def _replace_result(
    result: RouteExecutionResult,
    *,
    online_rows=None,
    completion=None,
) -> RouteExecutionResult:
    return RouteExecutionResult(
        online_rows=result.online_rows if online_rows is None else tuple(online_rows),
        completion=result.completion if completion is None else completion,
        audit_rows=result.audit_rows,
        reference_excitations=result.reference_excitations,
        pool_digest=result.pool_digest,
        elapsed_seconds=result.elapsed_seconds,
    )


def test_locked_profile_declares_only_selected_and_direct_over_twelve_cases() -> None:
    profile = locked_hh_profile()

    assert len(profile.cases) == 12
    assert profile.methods == (METHOD_SELECTED_QSE, METHOD_DIRECT_DAVIDSON)
    assert {case.n_ph_max for case in profile.cases} == {1, 3}
    assert len({case.case_key for case in profile.cases}) == 12
    assert profile.target_roots == tuple(range(1, 7))
    assert profile.dimension_cap == 64
    assert profile.trace_stride == 2
    assert profile.selector_policy_id.startswith("full_pencil_ky_fan")
    assert "hybrid" not in profile.canonical_json().lower()
    assert profile.payload()["controller_boundary"]["implicit_fallback_enabled"] is False


def test_one_command_completes_and_exact_replay_resumes_without_rerun(
    tmp_path: Path,
) -> None:
    executor = _LiteralExecutor()
    output_root = tmp_path / "matched"

    first = ensure_selected_vs_direct_hh_campaign(output_root, executor=executor)
    assert first.execution_status == "COMPLETE"
    assert first.completed_route_count == 24
    assert first.total_route_count == 24
    assert len(executor.calls) == 24

    first_status = (output_root / "status.json").read_bytes()
    first_matched = (output_root / "matched_routes.json").read_bytes()
    second = ensure_selected_vs_direct_hh_campaign(output_root, executor=executor)

    assert second == first
    assert len(executor.calls) == 24
    assert (output_root / "status.json").read_bytes() == first_status
    assert (output_root / "matched_routes.json").read_bytes() == first_matched
    decision = json.loads((output_root / "decision.json").read_text())
    assert decision["route_recommendation_counts"] == {"SELECTED_QSE": 36}


def test_local_executor_keeps_online_completion_separate_from_exact_audit() -> None:
    profile = locked_hh_profile()
    case = next(
        case
        for case in profile.cases
        if case.case_key == "weak_weak_nph1"
    )
    executor = LocalScientificRouteExecutor()

    selected = executor.execute(case, METHOD_SELECTED_QSE, profile)
    direct = executor.execute(case, METHOD_DIRECT_DAVIDSON, profile)

    assert selected.completion["terminal_status"] == SELECTED_STATUS_CERTIFIED
    assert selected.completion["davidson_invoked"] is False
    assert direct.completion["terminal_status"] in DIRECT_TERMINAL_STATUSES
    assert direct.completion["davidson_invoked"] is True
    assert direct.completion["native_stop_reason"]
    assert all("support_indices" not in row for row in direct.online_rows)
    for result in (selected, direct):
        assert result.online_rows
        assert len(result.online_rows) == len(result.audit_rows)
        assert all("max_root_abs_error" not in row for row in result.online_rows)
        assert all("per_root_abs_error" not in row for row in result.online_rows)
        assert all("max_root_abs_error" in row for row in result.audit_rows)
        assert all(
            set(("n2q", "d2q", "dc", "estimator")) <= set(row)
            for row in result.online_rows
        )


def test_cli_exposes_one_locked_acquisition_command_without_route_tuning() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pipelines.exact_bench.paper_iii_selected_vs_direct_campaign",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    help_text = proc.stdout.lower()
    assert "--output-root" in help_text
    assert "--methods" not in help_text
    assert "--hybrid" not in help_text
    assert "--fallback" not in help_text
    assert "--selector" not in help_text


def test_resume_rejects_tampered_post_run_crossings(tmp_path: Path) -> None:
    output_root = tmp_path / "matched"
    ensure_selected_vs_direct_hh_campaign(output_root, executor=_LiteralExecutor())
    route_path = (
        output_root
        / "routes"
        / "weak_weak_nph1"
        / f"{METHOD_SELECTED_QSE}.json"
    )
    artifact = json.loads(route_path.read_text())
    artifact["post_run_exact_audit"]["crossings"][0]["n2q"] = 999.0
    route_path.write_text(json.dumps(artifact))

    with pytest.raises(MatchedRouteAcquisitionError, match="crossing"):
        ensure_selected_vs_direct_hh_campaign(output_root, executor=_LiteralExecutor())


def test_resume_recomputes_audit_errors_from_sealed_ritz_energies(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "matched"
    ensure_selected_vs_direct_hh_campaign(output_root, executor=_LiteralExecutor())
    route_path = (
        output_root
        / "routes"
        / "weak_weak_nph1"
        / f"{METHOD_SELECTED_QSE}.json"
    )
    artifact = json.loads(route_path.read_text())
    audit = artifact["post_run_exact_audit"]
    audit["rungs"][0]["max_root_abs_error"] = 0.1
    audit["rungs"][0]["per_root_abs_error"] = [0.1] * 6
    for crossing in audit["crossings"]:
        crossing.update(
            {
                "status": "NOT_REACHED_WITHIN_CAP",
                "k": None,
                "n2q": None,
                "d2q": None,
                "dc": None,
                "estimator_queries": None,
            }
        )
    route_path.write_text(json.dumps(artifact))

    with pytest.raises(MatchedRouteAcquisitionError, match="audit rung drift"):
        ensure_selected_vs_direct_hh_campaign(output_root, executor=_LiteralExecutor())


def test_online_exact_leak_and_hidden_hybrid_fail_closed(tmp_path: Path) -> None:
    class LeakyExecutor(_LiteralExecutor):
        def execute(self, case, method_id, profile):
            result = super().execute(case, method_id, profile)
            row = {**result.online_rows[0], "max_root_abs_error": 0.0}
            return _replace_result(result, online_rows=(row,))

    with pytest.raises(MatchedRouteAcquisitionError, match="EXACT_CONTROL_LEAK"):
        ensure_selected_vs_direct_hh_campaign(
            tmp_path / "leak", executor=LeakyExecutor()
        )

    class HybridExecutor(_LiteralExecutor):
        def execute(self, case, method_id, profile):
            result = super().execute(case, method_id, profile)
            if method_id != METHOD_SELECTED_QSE:
                return result
            completion = {**result.completion, "davidson_invoked": True}
            return _replace_result(result, completion=completion)

    with pytest.raises(MatchedRouteAcquisitionError, match="HIDDEN_HYBRID_VIOLATION"):
        ensure_selected_vs_direct_hh_campaign(
            tmp_path / "hybrid", executor=HybridExecutor()
        )


def test_pool_exhaustion_is_completed_evidence_not_pending_or_fallback(
    tmp_path: Path,
) -> None:
    class ExhaustedExecutor(_LiteralExecutor):
        def execute(self, case, method_id, profile):
            result = super().execute(case, method_id, profile)
            if method_id != METHOD_SELECTED_QSE:
                return result
            completion = {
                **result.completion,
                "terminal_status": SELECTED_STATUS_POOL_EXHAUSTED,
            }
            return _replace_result(result, completion=completion)

    output_root = tmp_path / "exhausted"
    snapshot = ensure_selected_vs_direct_hh_campaign(
        output_root, executor=ExhaustedExecutor()
    )
    decision = json.loads((output_root / "decision.json").read_text())

    assert snapshot.execution_status == "COMPLETE"
    assert snapshot.completed_route_count == 24
    assert decision["route_recommendation_counts"] == {"DIRECT_DAVIDSON": 36}
    assert all(
        row["selected_qse"]["algorithm_status"] == SELECTED_STATUS_POOL_EXHAUSTED
        for row in decision["rows"]
    )
    assert decision["hybrid_enabled"] is False


def test_case_worker_and_collector_reproduce_the_locked_campaign(
    tmp_path: Path,
) -> None:
    executor = _LiteralExecutor()
    profile = locked_hh_profile()
    case_paths: list[Path] = []
    for case in profile.cases:
        path = tmp_path / "case_results" / f"{case.case_key}.json"
        acquire_selected_vs_direct_hh_case(
            case.case_key,
            path,
            executor=executor,
        )
        case_paths.append(path)

    first_bytes = case_paths[0].read_bytes()
    acquire_selected_vs_direct_hh_case(
        profile.cases[0].case_key,
        case_paths[0],
        executor=executor,
    )
    assert len(executor.calls) == 24
    assert case_paths[0].read_bytes() == first_bytes

    snapshot = collect_selected_vs_direct_hh_cases(
        case_paths,
        tmp_path / "campaign_result",
    )

    assert len(executor.calls) == 24
    assert snapshot.execution_status == "COMPLETE"
    assert snapshot.completed_route_count == 24
    decision = json.loads((tmp_path / "campaign_result" / "decision.json").read_text())
    assert decision["route_recommendation_counts"] == {"SELECTED_QSE": 36}
    assert decision["hybrid_enabled"] is False


def test_collector_fails_closed_on_an_incomplete_case_matrix(tmp_path: Path) -> None:
    profile = locked_hh_profile()
    first_case = profile.cases[0]
    path = tmp_path / f"{first_case.case_key}.json"
    acquire_selected_vs_direct_hh_case(
        first_case.case_key,
        path,
        executor=_LiteralExecutor(),
    )

    with pytest.raises(MatchedRouteAcquisitionError, match="CASE_MATRIX_INCOMPLETE"):
        collect_selected_vs_direct_hh_cases([path], tmp_path / "campaign_result")


@pytest.mark.parametrize(
    ("module", "required_flags"),
    (
        (
            "pipelines.exact_bench.paper_iii_selected_vs_direct_case",
            ("--case-key", "--output-json"),
        ),
        (
            "pipelines.exact_bench.paper_iii_selected_vs_direct_collect",
            ("--case-results-dir", "--output-root"),
        ),
    ),
)
def test_distributed_clis_expose_only_locked_acquisition_inputs(
    module: str,
    required_flags: tuple[str, ...],
) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    help_text = proc.stdout.lower()
    assert all(flag in help_text for flag in required_flags)
    assert "--method" not in help_text
    assert "--hybrid" not in help_text
    assert "--fallback" not in help_text
