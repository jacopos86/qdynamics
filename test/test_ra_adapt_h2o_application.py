from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from pipelines.static_adapt import adapt_pipeline
from pipelines.static_adapt.ra_adapt.adapters import (
    H2OLinearFDSectorCompletePauliBlockCandidateAdapter,
    H2OLinearFDSymmetryCompleteCandidateAdapter,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    CANDIDATE_REPRESENTATION_MACRO,
    CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    RAAdaptOperationalControls,
    ra_adapt_request_from_mapping,
)
from pipelines.static_adapt.ra_adapt.engine import (
    _sr_request,
    build_resolved_ra_protocol,
)
from pipelines.static_adapt.ra_adapt.h2o_application import (
    H2O_INSERTION_APPEND_ONLY,
    _engine_wall_budget_seconds,
    _problem,
    build_h2o_ra_request,
)
from pipelines.static_adapt.route_a_child_padding import (
    ROUTE_A_CHILD_PADDING_FULL_BINARY_CODE_SPACE_V1,
)
from pipelines.static_adapt.sr_snake._context import (
    _resolve_execution_context,
)


_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = (
    _ROOT
    / "chtc"
    / "paper_iv_h2o_sr_paper_i_noprune_nobeam_derivative_resolved_d50_20260724"
    / "input"
    / "h2o_fixture.json"
)


@pytest.fixture(scope="module")
def h2o_protocol(tmp_path_factory: pytest.TempPathFactory):
    problem = _problem(_FIXTURE)
    request = build_h2o_ra_request(
        problem=problem,
        output_dir=tmp_path_factory.mktemp("h2o_ra"),
        maximum_controller_rounds=50,
        gradient_tolerance=5.0e-7,
        exact_absolute_tolerance=1.6e-3,
        exact_confirmation_controller_rounds=2,
    )
    return problem, request, build_resolved_ra_protocol(problem, request)


def test_h2o_ra_defaults_to_staged_sector_complete_pauli_blocks(
    h2o_protocol,
) -> None:
    _, request, protocol = h2o_protocol
    contract = protocol.route_contract
    execution = contract["execution_settings"]
    invariants = contract["semantic_invariants"]

    assert isinstance(
        request.adapter,
        H2OLinearFDSectorCompletePauliBlockCandidateAdapter,
    )
    assert protocol.candidate_representation == CANDIDATE_REPRESENTATION_MACRO
    assert protocol.parent_inventory.count == 448
    assert protocol.executable_pool.count == 448
    assert any(
        label.startswith("el::")
        for label in protocol.executable_pool.ordered_labels
    )
    assert any(
        label.startswith("coupled::")
        for label in protocol.executable_pool.ordered_labels
    )
    assert any(
        label.startswith("conditional::")
        for label in protocol.executable_pool.ordered_labels
    )
    assert execution["phase3_runtime_split_mode"] == "off"
    assert execution["phase3_runtime_split_selection_mode"] == "off"
    assert invariants["singleton_admission"] is True
    assert invariants["staged_singleton_exposure"] is False
    assert invariants["staged_sector_complete_pauli_block_exposure"] is True
    assert invariants["raw_single_pauli_child_exposure"] is False
    assert invariants["guarded_single_pauli_child_exposure"] is False
    assert invariants["sector_complete_pauli_block_exposure"] is True
    assert invariants["candidate_generator_semantics"] == (
        "sector_complete_pauli_block_v1"
    )
    assert protocol.stopping_rule["gradient_tolerance"] == pytest.approx(
        5.0e-7
    )
    assert protocol.stopping_rule["exact_ed_target"][
        "confirmation_controller_rounds"
    ] == 2
    assert protocol.problem.n_ph_max == 1


def test_h2o_ra_sector_complete_block_adapter_round_trips(
    h2o_protocol,
) -> None:
    _, request, _ = h2o_protocol
    restored = ra_adapt_request_from_mapping(request.to_dict())

    assert isinstance(
        restored.adapter,
        H2OLinearFDSectorCompletePauliBlockCandidateAdapter,
    )
    assert restored == request


def test_h2o_sector_complete_blocks_group_unsafe_jw_children(
    h2o_protocol,
) -> None:
    problem, request, _ = h2o_protocol
    parent_inventory = request.adapter.parent_inventory(problem)
    electronic_single = next(
        candidate
        for candidate in parent_inventory.candidates
        if candidate.label.startswith("el::uccsd_sing(")
    )

    exposed = request.adapter.expose_children(
        (electronic_single,),
        problem=problem,
    )

    assert exposed.receipt.count == 1
    block = exposed.candidates[0]
    assert len(block.serialized_terms_exyz) == 2
    assert block.execution_mode == "grouped_exact"
    assert block.symmetry_receipt["grouped_preserves_fixed_counts"] is True
    assert block.symmetry_receipt["execution_preserves_fixed_counts"] is True
    assert exposed.metadata["raw_unsafe_single_pauli_words_rejected"] is True


def test_h2o_raw_single_pauli_route_fails_closed(tmp_path: Path) -> None:
    problem = _problem(_FIXTURE)

    with pytest.raises(ValueError, match="sector-complete"):
        build_h2o_ra_request(
            problem=problem,
            output_dir=tmp_path,
            maximum_controller_rounds=1,
            gradient_tolerance=5.0e-7,
            exact_absolute_tolerance=1.6e-3,
            candidate_representation=CANDIDATE_REPRESENTATION_SINGLE_PAULI,
        )


def test_h2o_ra_numerical_gate_accepts_only_the_named_application(
    h2o_protocol,
) -> None:
    problem, request, protocol = h2o_protocol
    route_contract = dict(protocol.route_contract)
    route_sha256 = route_contract.pop("sha256")
    route_profile = route_contract["route_profile"]
    context = _resolve_execution_context(
        problem,
        _sr_request(request),
        route_override=(
            route_profile,
            route_profile,
            route_contract,
            route_sha256,
        ),
        candidate_adapter=request.adapter,
    )

    runtime_kwargs = context.canonical_runtime_kwargs()
    validated = adapt_pipeline._validated_ra_adapt_route_contract(
        runtime_kwargs
    )
    assert validated is not None
    assert validated["semantic_invariants"][
        "application_lane"
    ] == (
        "paper_iv_h2o_linear_fd_ra_adapt_sector_complete_pauli_block_v1"
    )
    assert adapt_pipeline._default_no_prune_active_phase_flags(
        runtime_kwargs
    ) == (True, True, True)
    child_padding = adapt_pipeline._default_no_prune_child_padding_config(
        runtime_kwargs
    )
    assert child_padding.policy == (
        ROUTE_A_CHILD_PADDING_FULL_BINARY_CODE_SPACE_V1
    )


def test_h2o_ra_macro_representation_remains_explicitly_available(
    tmp_path: Path,
) -> None:
    problem = _problem(_FIXTURE)
    request = build_h2o_ra_request(
        problem=problem,
        output_dir=tmp_path,
        maximum_controller_rounds=1,
        gradient_tolerance=5.0e-7,
        exact_absolute_tolerance=1.6e-3,
        candidate_representation=CANDIDATE_REPRESENTATION_MACRO,
    )
    protocol = build_resolved_ra_protocol(problem, request)

    assert isinstance(
        request.adapter,
        H2OLinearFDSymmetryCompleteCandidateAdapter,
    )
    assert protocol.candidate_representation == CANDIDATE_REPRESENTATION_MACRO
    invariants = protocol.route_contract["semantic_invariants"]
    assert invariants["staged_singleton_exposure"] is False
    assert invariants["guarded_single_pauli_child_exposure"] is False


# --- Segmented-resume CLI guards (Paper IV, 2026-09-03) ------------------
#
# Cluster 9715945 lost 70 h of compute because CHTC's 72 h SYSTEM_PERIODIC_HOLD
# terminates without stage-out, and the entrypoint had no way to resume the
# per-round accepted-state checkpoint it had been writing all along. These
# tests pin the resume seam and the two overwrite guards that protect it.


def test_h2o_cli_exposes_resume_checkpoint_flag() -> None:
    """--resume-checkpoint must exist; without it a segment cannot continue."""
    import subprocess
    import sys

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pipelines.static_adapt.ra_adapt.h2o_application",
            "--help",
        ],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--resume-checkpoint" in completed.stdout


def test_h2o_resume_rejects_missing_checkpoint(tmp_path: Path) -> None:
    """A nonexistent checkpoint must fail loudly, not silently fresh-start."""
    import subprocess
    import sys

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pipelines.static_adapt.ra_adapt.h2o_application",
            "--fixture",
            str(_FIXTURE),
            "--output-dir",
            str(tmp_path / "out"),
            "--resume-checkpoint",
            str(tmp_path / "absent" / "current.json"),
        ],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "Resume checkpoint does not exist" in completed.stderr


def test_h2o_resume_rejects_checkpoint_inside_output_dir(
    tmp_path: Path,
) -> None:
    """Resuming from --output-dir/current.json would destroy its own source.

    The resumed run rewrites current.json every controller round, so a
    checkpoint living in the output directory is overwritten mid-run.
    """
    import subprocess
    import sys

    out = tmp_path / "out"
    out.mkdir()
    checkpoint = out / "current.json"
    checkpoint.write_text("{}", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pipelines.static_adapt.ra_adapt.h2o_application",
            "--fixture",
            str(_FIXTURE),
            "--output-dir",
            str(out),
            "--resume-checkpoint",
            str(checkpoint),
        ],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "inside --output-dir" in completed.stderr


def test_h2o_fresh_start_refuses_to_clobber_existing_checkpoint(
    tmp_path: Path,
) -> None:
    """The 9715945 data-loss hole: only result.json used to be guarded.

    A fresh start into a directory already holding an accepted-state
    checkpoint silently overwrote it at round 1, discarding prior progress.
    """
    import subprocess
    import sys

    out = tmp_path / "out"
    out.mkdir()
    (out / "current.json").write_text("{}", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pipelines.static_adapt.ra_adapt.h2o_application",
            "--fixture",
            str(_FIXTURE),
            "--output-dir",
            str(out),
        ],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "accepted-state" in completed.stderr
    assert (out / "current.json").read_text(encoding="utf-8") == "{}"


# --- Wall-clock segment stop (Paper IV, 2026-09-03) ------------------------
#
# CHTC's 72 h SYSTEM_PERIODIC_HOLD gives no stage-out, so a segment must exit
# normally before the wall whatever a round costs.  --max-wall-seconds is an
# operational control (RAAdaptOperationalControls.maximum_wall_seconds): it
# never enters the resolved protocol, is checked only at controller-round
# boundaries after each completed round, and stops with wall_clock_budget_v1
# exactly the way a maximum_controller_rounds stop does.


def test_h2o_cli_exposes_max_wall_seconds_flag() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pipelines.static_adapt.ra_adapt.h2o_application",
            "--help",
        ],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--max-wall-seconds" in completed.stdout
    assert "wall_clock_budget_v1" in completed.stdout


def test_h2o_cli_rejects_negative_wall_budget(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pipelines.static_adapt.ra_adapt.h2o_application",
            "--fixture",
            str(_FIXTURE),
            "--output-dir",
            str(tmp_path / "out"),
            "--max-wall-seconds",
            "-1",
        ],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "nonnegative" in completed.stderr
    assert not (tmp_path / "out" / "protocol.json").exists()


def test_wall_budget_is_operational_and_deducts_cli_startup() -> None:
    """The typed control defaults off and serializes nothing when unset."""

    controls = RAAdaptOperationalControls(maximum_controller_rounds=2)
    assert controls.maximum_wall_seconds is None
    assert "maximum_wall_seconds" not in controls.to_dict()
    budgeted = RAAdaptOperationalControls(
        maximum_controller_rounds=2,
        maximum_wall_seconds=0,
    )
    assert budgeted.to_dict()["maximum_wall_seconds"] == 0.0
    with pytest.raises(ValueError, match="nonnegative"):
        RAAdaptOperationalControls(
            maximum_controller_rounds=2,
            maximum_wall_seconds=-1.0,
        )
    assert _engine_wall_budget_seconds(None, startup_seconds=30.0) is None
    assert _engine_wall_budget_seconds(100.0, startup_seconds=30.0) == 70.0
    assert _engine_wall_budget_seconds(10.0, startup_seconds=30.0) == 0.0


# The end-to-end segment chain below executes the production H2O route
# (chtc/...20260724 fixture, macro_generator_v1 arm, append-only insertion).
# One controller round costs minutes on the M4 host, so the chain is opt-in:
#     PAPER_IV_H2O_CLI_RUNS=1 .venv/bin/python -m pytest \
#         test/test_ra_adapt_h2o_application.py -k segment_chain
_CLI_RUNS_ENV = "PAPER_IV_H2O_CLI_RUNS"
_cli_runs = pytest.mark.skipif(
    os.environ.get(_CLI_RUNS_ENV) != "1",
    reason=(
        f"set {_CLI_RUNS_ENV}=1 to run the multi-minute production H2O "
        "segment chain"
    ),
)


def _run_h2o_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    started = time.monotonic()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pipelines.static_adapt.ra_adapt.h2o_application",
            "--fixture",
            str(_FIXTURE),
            "--candidate-representation",
            CANDIDATE_REPRESENTATION_MACRO,
            "--insertion",
            H2O_INSERTION_APPEND_ONLY,
            "--maximum-controller-rounds",
            "2",
            *arguments,
        ],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
    )
    completed.elapsed_seconds = time.monotonic() - started  # type: ignore[attr-defined]
    return completed


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sidecar_of(checkpoint: Path) -> Path:
    pointer = _load_json(checkpoint)["checkpoint"][
        "estimator_call_ledger_checkpoint"
    ]
    sidecar = checkpoint.with_name(pointer["path"])
    assert sidecar.is_file(), sidecar
    assert (
        hashlib.sha256(sidecar.read_bytes()).hexdigest() == pointer["sha256"]
    )
    return sidecar


@pytest.fixture(scope="module")
def h2o_segment_chain(tmp_path_factory: pytest.TempPathFactory):
    """Segment 1 of the campaign chain: a zero wall budget, fresh start."""

    if os.environ.get(_CLI_RUNS_ENV) != "1":
        pytest.skip(f"set {_CLI_RUNS_ENV}=1 to run the H2O segment chain")
    root = tmp_path_factory.mktemp("h2o_segments")
    segment_one = root / "segment_1"
    first = _run_h2o_cli(
        "--output-dir",
        str(segment_one),
        "--max-wall-seconds",
        "0",
    )
    assert first.returncode == 0, first.stderr[-4000:]
    return {
        "root": root,
        "segment_one": segment_one,
        "first": first,
    }


@_cli_runs
def test_h2o_cli_wall_budget_stops_after_the_round_in_flight(
    h2o_segment_chain,
) -> None:
    """(a) zero budget: exit 0, one completed round, wall-clock reason."""

    segment_one = h2o_segment_chain["segment_one"]
    checkpoint = segment_one / "current.json"
    assert checkpoint.is_file()
    _sidecar_of(checkpoint)
    assert (segment_one / "estimator_call_ledger.json").is_file()

    manifest = _load_json(segment_one / "launch_manifest.json")
    assert manifest["max_wall_seconds"] == 0.0
    assert manifest["max_wall_seconds_engine_budget"] == 0.0
    assert manifest["fresh_start"] is True

    summary = _load_json(segment_one / "summary.json")
    assert summary["stop_reason"] == "wall_clock_budget_v1"
    assert summary["completed_controller_rounds"] == 1
    assert summary["max_wall_seconds"] == 0.0
    assert summary["wall_clock_budget_seconds"] == 0.0
    assert summary["wall_clock_elapsed_seconds"] >= 0.0

    result = _load_json(segment_one / "result.json")
    stop = result["run"]["stop"]
    assert stop["primary_reason"] == "wall_clock_budget_v1"
    assert stop["fired_reasons"] == ["wall_clock_budget_v1"]
    assert stop["completed_controller_rounds"] == 1
    reasons = {row["reason"]: row["fired"] for row in stop["conditions"]}
    assert reasons["maximum_controller_rounds"] is False
    assert reasons["wall_clock_budget_v1"] is True
    assert (
        result["run"]["canonical_reporting"]["horizon_scope"]
        == "deliberately_stopped_prefix"
    )
    checkpoint_payload = _load_json(checkpoint)
    assert checkpoint_payload["checkpoint"]["depth"] == 1
    assert checkpoint_payload["adapt_vqe"]["stop_reason"] is None


@_cli_runs
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "Blocked upstream of the wall-clock seam: --resume-checkpoint on the "
        "H2O route fails in sr_snake/_resume.py _validate_problem_binding "
        "(CanonicalResumeError: accepted-state resume is restricted to "
        "canonical L=2 Hubbard--Holstein or the named L=2 pure-Hubbard "
        "application; it also rejects any request carrying "
        "molecular_vibronic_h2o_linear_fd_fixture_json and requires "
        "adapt_pool == 'full_meta'). Admitting the Paper-IV route to "
        "canonical resume is a scientific-surface decision; this test "
        "flips to a hard failure (strict xfail) once it is made."
    ),
)
def test_h2o_cli_segment_chain_resumes_from_the_wall_clock_stop(
    h2o_segment_chain,
) -> None:
    """(b) the campaign design: segment 2 continues from segment 1."""

    segment_one = h2o_segment_chain["segment_one"]
    segment_two = h2o_segment_chain["root"] / "segment_2"
    second = _run_h2o_cli(
        "--output-dir",
        str(segment_two),
        "--resume-checkpoint",
        str(segment_one / "current.json"),
    )
    assert second.returncode == 0, second.stderr[-4000:]
    manifest = _load_json(segment_two / "launch_manifest.json")
    assert manifest["fresh_start"] is False
    assert manifest["resume_checkpoint"] == str(segment_one / "current.json")
    assert manifest["max_wall_seconds"] is None
    assert manifest["max_wall_seconds_engine_budget"] is None

    summary = _load_json(segment_two / "summary.json")
    assert summary["completed_controller_rounds"] == 2
    assert summary["stop_reason"] == "maximum_controller_rounds"
    assert summary["wall_clock_budget_seconds"] is None

    first_result = _load_json(segment_one / "result.json")
    second_result = _load_json(segment_two / "result.json")
    first_trajectory = first_result["run"]["accepted_trajectory"]
    second_trajectory = second_result["run"]["accepted_trajectory"]
    assert len(first_trajectory) == 1
    assert len(second_trajectory) == 2
    assert second_trajectory[0] == first_trajectory[0]
    assert (
        second_trajectory[1]["controller_round"]
        > second_trajectory[0]["controller_round"]
    )
    checkpoint = segment_two / "current.json"
    assert _load_json(checkpoint)["checkpoint"]["depth"] == 2
    _sidecar_of(checkpoint)
    # Segment 1's checkpoint is untouched by the resumed segment.
    assert _load_json(segment_one / "current.json")["checkpoint"]["depth"] == 1


@_cli_runs
def test_h2o_cli_wall_budget_leaves_the_protocol_digest_unchanged(
    h2o_segment_chain,
) -> None:
    """(c) protocol.json is byte-identical with and without the budget.

    The resolved protocol binds the observation paths (checkpoint and
    ledger live under --output-dir), so the flag-free reference protocol
    is resolved in-process for the same output directory: the budgeted
    CLI launch must have written exactly that protocol and digest.
    """

    segment_one = h2o_segment_chain["segment_one"]
    manifest = _load_json(segment_one / "launch_manifest.json")
    assert manifest["max_wall_seconds"] == 0.0
    problem = _problem(_FIXTURE)
    request = build_h2o_ra_request(
        problem=problem,
        output_dir=segment_one,
        maximum_controller_rounds=2,
        gradient_tolerance=manifest["gradient_tolerance"],
        exact_absolute_tolerance=manifest["exact_absolute_tolerance"],
        candidate_representation=CANDIDATE_REPRESENTATION_MACRO,
        insertion=H2O_INSERTION_APPEND_ONLY,
    )
    reference = build_resolved_ra_protocol(problem, request)
    assert reference.sha256 == manifest["protocol_sha256"]
    with_budget = _load_json(segment_one / "protocol.json")
    assert with_budget == reference.to_dict()
    assert with_budget["sha256"] == reference.sha256
    result = _load_json(segment_one / "result.json")
    assert result["protocol"]["sha256"] == reference.sha256
    assert "maximum_wall_seconds" not in json.dumps(with_budget)

# --- Opt-in Marrakesh graph-span selector cost (Paper IV, 2026-09-03) -------
#
# The H2O route contract resolves phase3_backend_cost_mode="proxy" because the
# Paper-I FakeMarrakesh embedding table stopped at 12 qubits.  The table now
# carries the 19-qubit continuation of the same rule, and --cost-mode
# marrakesh_graph_span_v1 opts the H2O application into it.  These tests pin
# the default identity byte-for-byte and the variant as a distinct, stable
# identity that passes every graph-span guard and constructs the 19q oracle.

# Pinned on the 20260724 fixture addressed by its repo-relative path string
# (the problem_request_sha256 hashes the fixture path as given, so the
# protocol digest is only portable for a fixed path string) with
# output_dir=/paper_iv/h2o_ra (the digest also covers the checkpoint path
# through the request).  The route-contract digests are path-independent.
_PINNED_FIXTURE_RELATIVE = Path(
    "chtc/paper_iv_h2o_sr_paper_i_noprune_nobeam_derivative_resolved_d50_"
    "20260724/input/h2o_fixture.json"
)
_PINNED_OUTPUT_DIR = Path("/paper_iv/h2o_ra")
_PINNED_DEFAULT_PROTOCOL_SHA256 = (
    "34683ac8b80100b2227f3280d3d2ed3ef1dd54d0eb18bf9789db0faa2765f4cc"
)
_PINNED_DEFAULT_ROUTE_CONTRACT_SHA256 = (
    "6ed3f6a4f69106d09e735415d0c504d81213869df15ea8e1e0991aa320019c55"
)
_PINNED_GRAPH_SPAN_PROTOCOL_SHA256 = (
    "daa5b457cbd866bf049a4fec19b9d75bb70224c668a5b9d63945627a8046dd2e"
)
_PINNED_GRAPH_SPAN_ROUTE_CONTRACT_SHA256 = (
    "e91724f5bb8ffe51ea6d0f71fdb8ba6c075789345b90d8aa86d338ea0c92b1f1"
)
_PINNED_DEFAULT_ROUTE_PROFILE = (
    "paper_iv_h2o_ra_adapt__macro_generator_v1"
    "__sector_complete_pauli_block_exposure_v1"
    "__insertion_commutation_plateau_v1__measured_residual_response_v1"
    "__all_phase_resource_weighting_v1"
    "__incremental_active_baseline__exact_guarded_full_response"
)
_PINNED_GRAPH_SPAN_ROUTE_PROFILE = (
    "paper_iv_h2o_ra_adapt__macro_generator_v1"
    "__sector_complete_pauli_block_exposure_v1"
    "__cost-marrakesh_graph_span_v1"
    "__insertion_commutation_plateau_v1__measured_residual_response_v1"
    "__all_phase_resource_weighting_v1"
    "__incremental_active_baseline__exact_guarded_full_response"
)


def _pinned_request_kwargs(problem):
    return dict(
        problem=problem,
        output_dir=_PINNED_OUTPUT_DIR,
        maximum_controller_rounds=50,
        gradient_tolerance=5.0e-7,
        exact_absolute_tolerance=1.6e-3,
        exact_confirmation_controller_rounds=2,
    )


@pytest.fixture(scope="module")
def h2o_pinned_protocols():
    """Default and graph-span protocols on the pinned fixture path string."""
    import os

    previous = os.getcwd()
    os.chdir(_ROOT)
    try:
        assert _PINNED_FIXTURE_RELATIVE.is_file()
        problem = _problem(_PINNED_FIXTURE_RELATIVE)
    finally:
        os.chdir(previous)
    default_request = build_h2o_ra_request(**_pinned_request_kwargs(problem))
    graph_span_request = build_h2o_ra_request(
        **_pinned_request_kwargs(problem),
        cost_mode="marrakesh_graph_span_v1",
    )
    return (
        problem,
        default_request,
        build_resolved_ra_protocol(problem, default_request),
        graph_span_request,
        build_resolved_ra_protocol(problem, graph_span_request),
    )


@pytest.fixture(scope="module")
def h2o_graph_span_protocol(h2o_pinned_protocols):
    problem, _, _, request, protocol = h2o_pinned_protocols
    return problem, request, protocol


def test_h2o_default_protocol_identity_is_byte_identical(
    h2o_pinned_protocols,
) -> None:
    problem, request, protocol, _, _ = h2o_pinned_protocols

    assert request.method.compile_cost is None
    assert "compile_cost" not in request.to_dict()["method"]
    assert protocol.sha256 == _PINNED_DEFAULT_PROTOCOL_SHA256
    assert protocol.route_contract["sha256"] == (
        _PINNED_DEFAULT_ROUTE_CONTRACT_SHA256
    )
    assert protocol.route_contract["route_profile"] == (
        _PINNED_DEFAULT_ROUTE_PROFILE
    )
    execution = protocol.route_contract["execution_settings"]
    assert execution["phase3_backend_cost_mode"] == "proxy"
    # An explicit cost_mode="proxy" is the same request, not a new identity.
    explicit = build_h2o_ra_request(
        **_pinned_request_kwargs(problem), cost_mode="proxy"
    )
    assert explicit == request


def test_h2o_graph_span_variant_is_a_distinct_stable_identity(
    h2o_pinned_protocols,
) -> None:
    from pipelines.static_adapt.hh_backend_compile_oracle import (
        MARRAKESH_GRAPH_SPAN_MODE,
    )
    from pipelines.static_adapt.sr_snake.contracts import (
        MarrakeshGraphSpanCompileCost,
    )

    _, _, default_protocol, request, protocol = h2o_pinned_protocols
    contract = protocol.route_contract
    execution = contract["execution_settings"]
    invariants = contract["semantic_invariants"]

    assert isinstance(request.method.compile_cost, MarrakeshGraphSpanCompileCost)
    assert protocol.sha256 == _PINNED_GRAPH_SPAN_PROTOCOL_SHA256
    assert protocol.sha256 != _PINNED_DEFAULT_PROTOCOL_SHA256
    assert contract["sha256"] == _PINNED_GRAPH_SPAN_ROUTE_CONTRACT_SHA256
    assert contract["sha256"] != _PINNED_DEFAULT_ROUTE_CONTRACT_SHA256
    assert contract["route_profile"] == _PINNED_GRAPH_SPAN_ROUTE_PROFILE
    # The engine.py / adapt_pipeline.py graph-span guards: FakeMarrakesh,
    # optimization level 1, transpile seed 7, no backend shortlist scope.
    assert execution["phase3_backend_cost_mode"] == MARRAKESH_GRAPH_SPAN_MODE
    assert execution["phase3_backend_name"] == "FakeMarrakesh"
    assert execution["phase3_backend_optimization_level"] == 1
    assert execution["phase3_backend_transpile_seed"] == 7
    assert "phase3_backend_cost_scope" not in execution
    assert invariants["phase3_backend_cost_mode"] == MARRAKESH_GRAPH_SPAN_MODE
    assert invariants["compile_cost_policy"] == "marrakesh_graph_span_v1"
    assert invariants["selector_qiskit_compile_cost_active"] is False
    assert invariants["application_lane"] == (
        "paper_iv_h2o_linear_fd_ra_adapt_sector_complete_pauli_block_v1"
    )
    assert (
        "selector_cost_structural_proxy_to_marrakesh_graph_span_"
        "fixed_embedding_no_transpile"
        in contract["lineage_authority"]["only_intended_scientific_changes"]
    )
    # The lineage parent stays the proxy H2O route contract.
    assert contract["lineage_authority"]["parent_route_profile"] == (
        default_protocol.route_contract["lineage_authority"][
            "parent_route_profile"
        ]
    )
    assert contract["lineage_authority"]["parent_contract_sha256"] == (
        default_protocol.route_contract["lineage_authority"][
            "parent_contract_sha256"
        ]
    )
    # Every other execution setting is untouched relative to the default.
    default_execution = default_protocol.route_contract["execution_settings"]
    differences = {
        key
        for key in set(execution) | set(default_execution)
        if execution.get(key) != default_execution.get(key)
    }
    assert differences == {"phase3_backend_cost_mode"}


def test_h2o_graph_span_request_round_trips(h2o_graph_span_protocol) -> None:
    _, request, _ = h2o_graph_span_protocol
    restored = ra_adapt_request_from_mapping(request.to_dict())

    assert restored == request
    assert request.to_dict()["method"]["compile_cost"] == {
        "kind": "marrakesh_graph_span_v1"
    }


def test_h2o_graph_span_variant_passes_runtime_guards_and_builds_19q_oracle(
    h2o_graph_span_protocol,
) -> None:
    from types import SimpleNamespace

    import numpy as np

    from pipelines.static_adapt.hh_backend_compile_oracle import (
        BACKEND_COMPILE_SCOPE_SHARED_ALL_PHASES_V1,
        MARRAKESH_GRAPH_SPAN_MODE,
        MarrakeshGraphSpanCostOracle,
        marrakesh_logical_embedding_v1,
    )

    problem, request, protocol = h2o_graph_span_protocol
    route_contract = dict(protocol.route_contract)
    route_sha256 = route_contract.pop("sha256")
    route_profile = route_contract["route_profile"]
    context = _resolve_execution_context(
        problem,
        _sr_request(request),
        route_override=(
            route_profile,
            route_profile,
            route_contract,
            route_sha256,
        ),
        candidate_adapter=request.adapter,
    )
    runtime_kwargs = context.canonical_runtime_kwargs()

    validated = adapt_pipeline._validated_ra_adapt_route_contract(
        runtime_kwargs
    )
    assert validated is not None
    assert validated["semantic_invariants"]["application_lane"] == (
        "paper_iv_h2o_linear_fd_ra_adapt_sector_complete_pauli_block_v1"
    )
    assert adapt_pipeline._default_no_prune_active_phase_flags(
        runtime_kwargs
    ) == (True, True, True)
    assert runtime_kwargs["phase3_backend_cost_mode"] == (
        MARRAKESH_GRAPH_SPAN_MODE
    )
    assert runtime_kwargs["phase3_backend_name"] == "FakeMarrakesh"
    assert runtime_kwargs["phase3_backend_optimization_level"] == 1
    assert runtime_kwargs["backend_transpile_seed"] == 7
    assert runtime_kwargs["phase3_backend_cost_scope"] == (
        BACKEND_COMPILE_SCOPE_SHARED_ALL_PHASES_V1
    )

    # The production construction path, on a 19-qubit reference register
    # (the (3,7,3) H2O model: 12 fermion + 7 boson qubits).
    core = SimpleNamespace(reference_state=np.zeros(2**19, dtype=complex))
    oracle = adapt_pipeline._default_no_prune_backend_compile_oracle(
        core=core, kwargs=runtime_kwargs
    )
    assert isinstance(oracle, MarrakeshGraphSpanCostOracle)
    assert oracle.num_qubits == 19
    assert oracle.embedding == marrakesh_logical_embedding_v1(19)
    assert oracle.targets[0].resolved_name == "FakeMarrakesh"
    assert (
        adapt_pipeline._default_no_prune_phase3_backend_compile_oracle(
            core=core, kwargs=runtime_kwargs
        )
        is None
    )


def test_h2o_default_runtime_keeps_no_backend_oracle(h2o_protocol) -> None:
    from types import SimpleNamespace

    import numpy as np

    problem, request, protocol = h2o_protocol
    route_contract = dict(protocol.route_contract)
    route_sha256 = route_contract.pop("sha256")
    route_profile = route_contract["route_profile"]
    runtime_kwargs = _resolve_execution_context(
        problem,
        _sr_request(request),
        route_override=(
            route_profile,
            route_profile,
            route_contract,
            route_sha256,
        ),
        candidate_adapter=request.adapter,
    ).canonical_runtime_kwargs()

    assert runtime_kwargs["phase3_backend_cost_mode"] == "proxy"
    assert (
        adapt_pipeline._default_no_prune_backend_compile_oracle(
            core=SimpleNamespace(
                reference_state=np.zeros(2**19, dtype=complex)
            ),
            kwargs=runtime_kwargs,
        )
        is None
    )


def test_graph_span_compile_cost_is_rejected_off_the_h2o_application() -> None:
    from pipelines.static_adapt.ra_adapt.adapters import (
        MacroCandidateAdapter,
    )
    from pipelines.static_adapt.ra_adapt.contracts import (
        ACTIVE_GRADIENT_MEASURED,
        RESOURCE_WEIGHTING_ALL_PHASE,
        RAAdaptRequest,
    )
    from pipelines.static_adapt.ra_adapt.engine import (
        RA_ADAPT_ALGORITHM_ID,
        _repaired_route_contract,
    )
    from pipelines.static_adapt.sr_snake.contracts import (
        MarrakeshGraphSpanCompileCost,
        SRMethodPolicy,
    )

    request = RAAdaptRequest(
        adapter=MacroCandidateAdapter(),
        method=SRMethodPolicy(
            compile_cost=MarrakeshGraphSpanCompileCost(),
        ),
    )
    with pytest.raises(ValueError, match="admissible only"):
        _repaired_route_contract(
            request,
            active_gradient_policy=ACTIVE_GRADIENT_MEASURED,
            resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
            algorithm_id=RA_ADAPT_ALGORITHM_ID,
        )


def test_h2o_rejects_unknown_cost_mode(h2o_protocol) -> None:
    problem, _, _ = h2o_protocol
    with pytest.raises(ValueError, match="Unknown H2O cost mode"):
        build_h2o_ra_request(
            **_pinned_request_kwargs(problem), cost_mode="transpile_single_v1"
        )


def test_h2o_cli_exposes_cost_mode_flag() -> None:
    import subprocess
    import sys

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pipelines.static_adapt.ra_adapt.h2o_application",
            "--help",
        ],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--cost-mode {proxy,marrakesh_graph_span_v1}" in completed.stdout


# --- Early fail-closed on an unpriceable register (Paper IV, 2026-09-03) ---
#
# --cost-mode marrakesh_graph_span_v1 prices supports on one fixed
# logical-to-physical FakeMarrakesh embedding, and that embedding exists only
# for the sizes in _MARRAKESH_GRAPH_SPAN_EMBEDDINGS_V1. Before this check the
# missing size surfaced only when the oracle was constructed -- minutes into a
# production round, after protocol.json and launch_manifest.json had already
# been written, so a launch looked healthy and then died. The register size is
# known from the fixture, so the launcher rejects it in seconds, before it
# writes anything. The 20260724 fixture is a 15-qubit register and 15 is
# deliberately absent from the table.


def test_graph_span_cost_mode_is_rejected_for_an_unsupported_register() -> None:
    from pipelines.static_adapt.ra_adapt.h2o_application import (
        validate_cost_mode_register_size,
    )

    # The default proxy mode is register-size agnostic.
    validate_cost_mode_register_size("proxy", num_qubits=15)
    # Supported sizes stay supported.
    validate_cost_mode_register_size("marrakesh_graph_span_v1", num_qubits=19)
    with pytest.raises(ValueError, match=r"15-qubit register"):
        validate_cost_mode_register_size(
            "marrakesh_graph_span_v1", num_qubits=15
        )


def test_h2o_cli_graph_span_fails_before_writing_protocol_json(
    tmp_path: Path,
) -> None:
    """The 15q fixture must be refused in seconds, with nothing written."""
    import subprocess
    import sys

    out = tmp_path / "out"
    started = time.monotonic()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pipelines.static_adapt.ra_adapt.h2o_application",
            "--fixture",
            str(_FIXTURE),
            "--output-dir",
            str(out),
            "--cost-mode",
            "marrakesh_graph_span_v1",
            "--maximum-controller-rounds",
            "1",
        ],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - started

    assert completed.returncode != 0
    assert "marrakesh_graph_span_v1" in completed.stderr
    assert "15-qubit register" in completed.stderr
    assert "Supported register sizes" in completed.stderr
    # Nothing was written: not the protocol, not the manifest, not the dir.
    assert not (out / "protocol.json").exists()
    assert not (out / "launch_manifest.json").exists()
    assert not out.exists()
    # Seconds, not the minutes an oracle-construction failure used to cost.
    assert elapsed < 120.0
