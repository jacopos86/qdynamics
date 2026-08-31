from __future__ import annotations

import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipelines.static_adapt.adapt_pipeline as adapt_pipeline
from pipelines.static_adapt.cli_config import (
    _build_adapt_arg_parser,
    _build_run_hardcoded_adapt_vqe_kwargs,
)
from pipelines.static_adapt.sr_snake_route_profile import (
    CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_ONE_SIDED_COST_V1_EXECUTION_SETTINGS,
    CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_V1_EXECUTION_SETTINGS,
    SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_ONE_SIDED_COST_V1,
    SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_V1,
    canonical_sr_snake_macro_only_physical_lanes_fs_prune_beam_one_sided_cost_v1_contract,
    canonical_sr_snake_macro_only_physical_lanes_fs_prune_beam_v1_contract,
)
from src.quantum.hubbard_latex_python_pairs import (
    build_hubbard_holstein_hamiltonian,
)


SYMMETRIC_ALIAS = "sr_snake_macro_only_physical_lanes_fs_prune_beam3x2_v1"
ONE_SIDED_ALIAS = (
    "sr_snake_macro_only_physical_lanes_fs_prune_beam3x2_one_sided_cost_v1"
)


def _parser():
    return _build_adapt_arg_parser(adapt_gradient_parity_rtol=1.0e-7)


def _args(alias: str, *extra: str):
    return _parser().parse_args(
        ["--sr-route-profile", alias, "--adapt-max-depth", "50", *extra]
    )


def _runtime_kwargs(args) -> dict[str, object]:
    return _build_run_hardcoded_adapt_vqe_kwargs(
        args,
        h_poly=None,
        resolved_problem_context=SimpleNamespace(
            layout=SimpleNamespace(total_qubits=6)
        ),
        cli_adapt_continuation_mode="phase3_v1",
        adapt_ref_base_depth=0,
        psi_ref_override=None,
        psi_ref_source=None,
        psi_ref_handoff_state_kind=None,
        exact_gs_override=0.0,
        phase3_oracle_gradient_config=None,
        final_noise_audit_config=None,
    )


@pytest.mark.parametrize(
    ("alias", "resolved", "contract_factory", "cost_mode"),
    [
        (
            SYMMETRIC_ALIAS,
            SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_V1,
            canonical_sr_snake_macro_only_physical_lanes_fs_prune_beam_v1_contract,
            "family_robust_symmetric_arctan_v1",
        ),
        (
            ONE_SIDED_ALIAS,
            SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_ONE_SIDED_COST_V1,
            canonical_sr_snake_macro_only_physical_lanes_fs_prune_beam_one_sided_cost_v1_contract,
            "family_robust_v1",
        ),
    ],
)
def test_macro_beam_prune_cost_profiles_materialize_exact_contract(
    alias: str,
    resolved: str,
    contract_factory,
    cost_mode: str,
) -> None:
    args = _args(alias)

    assert args.sr_route_profile_request == resolved
    assert args.sr_route_profile_resolved == resolved
    assert args.sr_route_profile_contract == contract_factory()
    assert args.adapt_max_depth == 50
    assert args.static_lane_route == "physical_operator_type"
    assert args.physical_lane_shortlist_aggressiveness == 3
    assert args.phase3_runtime_split_mode == "off"
    assert args.adapt_child_pool_expansion_mode == "off"
    assert args.shared_pauli_pool_mode == "off"
    assert args.adapt_beam_live_branches == 3
    assert args.adapt_beam_children_per_parent == 2
    assert args.adapt_beam_terminated_keep == 3
    assert args.adapt_beam_terminal_archive_mode == "legacy"
    assert args.adapt_beam_lambda == pytest.approx(0.005)
    assert args.phase1_prune_enabled is True
    assert args.phase1_prune_mode == "live"
    assert args.phase1_prune_max_candidates == 1
    assert args.phase1_prune_local_window_size == 0
    assert args.phase1_prune_recovery_trust_radius == pytest.approx(0.125)
    assert args.phase1_prune_schur_nomination_route == (
        "full_logical_fs_trust_delete_refit_v1"
    )
    assert args.phase1_prune_metric_schur_solve_mode == (
        "affine_deletion_global_trust_v1"
    )
    assert args.phase1_prune_metric_schur_mu == 0.0
    assert args.phase1_prune_metric_mu_update_policy == "off"
    assert args.phase1_prune_metric_schur_cost_weighting == "off"
    assert args.phase1_prune_endpoint_overlap_policy == "off"
    assert args.adapt_final_full_refit == "false"
    assert args.phase3_hardware_cost_normalization_mode == cost_mode


def test_macro_beam_prune_cost_arms_differ_only_in_cost_normalization() -> None:
    symmetric = dict(
        CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_V1_EXECUTION_SETTINGS
    )
    one_sided = dict(
        CANONICAL_SR_SNAKE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_ONE_SIDED_COST_V1_EXECUTION_SETTINGS
    )

    assert symmetric.pop("phase3_hardware_cost_normalization_mode") == (
        "family_robust_symmetric_arctan_v1"
    )
    assert one_sided.pop("phase3_hardware_cost_normalization_mode") == (
        "family_robust_v1"
    )
    assert one_sided == symmetric

    symmetric_contract = (
        canonical_sr_snake_macro_only_physical_lanes_fs_prune_beam_v1_contract()
    )
    one_sided_contract = (
        canonical_sr_snake_macro_only_physical_lanes_fs_prune_beam_one_sided_cost_v1_contract()
    )
    assert symmetric_contract["semantic_invariants"]["hardware_cost_policy"] == (
        "family_robust_symmetric_arctan_v1"
    )
    assert one_sided_contract["semantic_invariants"]["hardware_cost_policy"] == (
        "family_robust_v1"
    )
    assert one_sided_contract["semantic_invariants"][
        "all_energy_models_infeasible_novelty_fallback_policy"
    ] == "collective_span_novelty_over_cost_v1"


@pytest.mark.parametrize("alias", [SYMMETRIC_ALIAS, ONE_SIDED_ALIAS])
def test_macro_beam_prune_cost_profiles_are_registered_at_runtime_startup(
    alias: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both complete contracts must cross runtime registration successfully."""

    class _RuntimeProfileAccepted(RuntimeError):
        pass

    kwargs = _runtime_kwargs(
        _args(
            alias,
            "--problem",
            "hh",
            "--L",
            "2",
            "--u",
            "0.25",
            "--g-ep",
            "0.353553390593",
            "--n-ph-max",
            "2",
        )
    )
    kwargs["resolved_problem_context"] = None
    kwargs["h_poly"] = build_hubbard_holstein_hamiltonian(
        dims=2,
        J=1.0,
        U=0.25,
        omega0=1.0,
        g=0.353553390593,
        n_ph_max=2,
        boson_encoding="binary",
        repr_mode="JW",
        indexing="blocked",
        pbc=False,
        include_zero_point=True,
    )
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(adapt_pipeline, "_ai_log", lambda *_a, **_k: None)

    def _runtime_profile_accepted(_payload: dict[str, object]) -> str:
        raise _RuntimeProfileAccepted

    monkeypatch.setattr(
        adapt_pipeline,
        "_candidate_record_payload_digest",
        _runtime_profile_accepted,
    )

    with pytest.raises(_RuntimeProfileAccepted):
        adapt_pipeline._run_hardcoded_adapt_vqe(**kwargs)


def test_macro_beam_prune_cost_profiles_activate_prune_trial_accounting() -> None:
    for alias in (SYMMETRIC_ALIAS, ONE_SIDED_ALIAS):
        kwargs = _runtime_kwargs(_args(alias))
        assert kwargs["phase1_prune_enabled"] is True
        assert kwargs["phase1_prune_mode"] == "live"

    runtime_source = inspect.getsource(adapt_pipeline._run_hardcoded_adapt_vqe)
    accounting_profile_set = runtime_source.split(
        "sr_v4_prune_accounting_active = bool(", 1
    )[1].split("sr_v4_prune_accounting_views", 1)[0]
    assert (
        "SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_V1"
        in accounting_profile_set
    )
    assert (
        "SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_FS_PRUNE_BEAM_ONE_SIDED_COST_V1"
        in accounting_profile_set
    )


@pytest.mark.parametrize(
    "override",
    [
        ("--adapt-beam-live-branches", "2"),
        ("--adapt-beam-children-per-parent", "3"),
        ("--phase1-prune-metric-schur-mu", "0.01"),
        ("--phase1-prune-mode", "both"),
        ("--phase3-runtime-split-mode", "shortlist_pauli_children_v1"),
    ],
)
def test_macro_beam_prune_profiles_fail_closed_on_drift(
    override: tuple[str, str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _args(SYMMETRIC_ALIAS, *override)
    assert exc_info.value.code == 2
