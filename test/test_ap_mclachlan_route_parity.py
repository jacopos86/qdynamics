"""Golden-run parity locks for the AP-McLachlan dynamics route.

These tests pin end-to-end behavior of the operating configuration so that
refactoring the launch surface cannot silently change the science.  They run
tiny deterministic trajectories through the ordinary runner entry point and
assert the decision sequence, the committed support size, and the energy
trajectory to tight tolerance.  A deliberate algorithm change should update
the recorded values in one place, with the change visible in review.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.contracts.scaffold import CandidatePoolSource, ScaffoldRuntimeInput
from pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory import (
    SupportPatchControllerConfig,
)
from pipelines.time_dynamics.ap_mclachlan.fixed_step import SolveRepairConfig
from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import (
    AppendControllerConfig,
    run_append_ap_mclachlan_from_runtime_input,
)
from src.quantum.ansatz_parameterization import build_parameter_layout
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _poly(*components: tuple[str, float], nq: int = 2) -> PauliPolynomial:
    poly = PauliPolynomial("JW")
    for word, coeff in components:
        poly.add_term(PauliTerm(int(nq), ps=str(word), pc=float(coeff)))
    poly._reduce()
    return poly


X0 = AnsatzTerm(label="sx0", polynomial=_poly(("ex", 1.0)))
Z0 = AnsatzTerm(label="sz0", polynomial=_poly(("ez", 1.0)))
CA = AnsatzTerm(label="cand_a", polynomial=_poly(("xe", 0.7)))
CB = AnsatzTerm(label="cand_b", polynomial=_poly(("ye", 0.4)))
HAM_POLY = _poly(("ez", 2.0), ("xx", 0.6))


def _runtime_input(theta=(0.35, -0.22)):
    selected = (X0, Z0)
    layout = build_parameter_layout(selected)
    executor = CompiledAnsatzExecutor(
        selected, parameterization_layout=layout,
        parameterization_mode="per_pauli_term",
    )
    psi_ref = np.zeros(4, dtype=complex)
    psi_ref[0] = 1.0
    return ScaffoldRuntimeInput(
        resolved_problem=SimpleNamespace(family_key="toy", hamiltonian=HAM_POLY),
        psi_ref=psi_ref,
        psi_initial=np.asarray(
            executor.prepare_state(np.asarray(theta, dtype=float), psi_ref),
            dtype=complex,
        ),
        base_layout=layout,
        theta_runtime=np.asarray(theta, dtype=float),
        theta_logical=np.zeros(int(layout.logical_parameter_count), dtype=float),
        structure_locked=False,
        exact_energy=None,
        selected_terms=selected,
        candidate_pool_terms=(CA, CB),
        candidate_pool_source=CandidatePoolSource(
            source_kind="resolved_pool", pool_key="toy_pool", completeness="complete",
        ),
        provenance={"artifact_json": "parity_toy.json"},
    )


def _run(**support_overrides):
    support = dict(
        residual_ratio_threshold=0.02,
        max_structural_pool_size=2,
        max_joint_patch_evaluations=5000,
        max_certification_attempts_per_level=6,
        max_certification_attempts_per_deletion_branch=2,
        max_insertion_batch_size=1,
        certification_refit_enabled=True,
        certification_refit_trust_radius=0.6,
        certification_refit_max_iterations=15,
        prune_ray_distance_tol=2.0e-3,
        prune_history_lambda=0.0,
        min_runtime_parameter_count=1,
    )
    support.update(support_overrides)
    return run_append_ap_mclachlan_from_runtime_input(
        _runtime_input(),
        times=tuple(0.05 * k for k in range(5)),
        integrator_method="rk4",
        pinv_rcond=1.0e-10,
        ridge_lambda=1.0e-7,
        controller_config=AppendControllerConfig(max_append_candidates=0),
        support_patch_config=SupportPatchControllerConfig(**support),
        solve_repair_config=SolveRepairConfig.minimal_profile(),
    )


def _decisions(payload):
    out = []
    for point in payload["trajectory"]["points"]:
        meta = (point.get("patch_decision") or {}).get("metadata") or {}
        if meta.get("kind"):
            out.append(meta["kind"])
    return out


def test_exchange_route_golden_trajectory() -> None:
    payload = _run()
    summary = payload["summary"]
    energies = [row["energy_expectation"] for row in payload["plot_rows"]]

    # Decision sequence and support evolution are the scientific content.
    #
    # DELIBERATE GOLDEN REBASELINE (user ruling, 2026-09-01; ported with
    # 259ef926 "Fix logical SIAM controller geometry").  The final checkpoint
    # is a certified stay, not an insertion.  Candidate suffix propagation in
    # the structural insertion cache must use the evolved checkpoint angles,
    # not the angles the state retained at the preceding support rebuild;
    # `build_selector_inputs` now passes `theta_runtime` explicitly.  The
    # stale-angle cache reported a spurious final insertion here.
    #
    # Measured before/after of that one-line plumbing change, this fixture:
    #   before: ["insert", "insert", "delete", "insert"], final count 4
    #   after:  ["insert", "insert", "delete", "stay"],   final count 3
    # The pinned energies and residual ratio below are unchanged by it.
    assert _decisions(payload) == ["insert", "insert", "delete", "stay"]
    assert summary["runtime_parameter_count_initial"] == 2
    assert summary["runtime_parameter_count_final"] == 3
    assert summary["accepted_append_count"] == 2
    # The measurement-free permission changes the sequence deliberately: only
    # one of 28 proposed deletion sets enters finalist certification here.
    assert summary["deletion_permission_evaluated_set_count"] == 28
    assert summary["deletion_permission_permitted_set_count"] == 1
    assert summary["deletion_permission_rejected_set_count"] == 27
    assert summary["accepted_delete_count"] == 1
    # NOTE: 259ef926 also asserts accepted_deleted_coordinate_count == 1 here.
    # That is deliberately NOT ported: on this tree the per-coordinate patch
    # counters (patch_appended_count / patch_deleted_count / patch_removed_count)
    # read 0 on every accepted row of this route, so the derived summary fields
    # accepted_appended_coordinate_count and accepted_deleted_coordinate_count
    # are 0 while the per-patch counts above are correct.  That is a separate
    # pre-existing telemetry gap, not a consequence of the checkpoint-angle
    # fix, and pinning it here would assert it fixed.
    # Energy trajectory pinned; a refactor that changes physics fails here.
    assert energies[0] == pytest.approx(1.5296843746, abs=1e-9)
    assert energies[-1] == pytest.approx(1.5296843746, abs=1e-7)
    assert summary["max_mclachlan_residual_ratio"] == pytest.approx(
        0.1780937493, abs=1e-6
    )


def test_prune_only_below_threshold_is_measurement_free() -> None:
    # A threshold above every residual keeps insertions unenumerated while
    # deletions stay eligible: the measurement-economics contract.
    payload = _run(residual_ratio_threshold=1.0e9)
    for point in payload["trajectory"]["points"]:
        meta = (point.get("patch_decision") or {}).get("metadata") or {}
        if not meta:
            continue
        assert meta.get("insertions_enabled") is False
        assert meta.get("candidate_pool_deduplicated") == 0


def test_avqds_policy_golden_trajectory() -> None:
    payload = _run(
        dynamics_policy="avqds",
        avqds_l2_cut=1.0e-6,
        avqds_max_appends_per_checkpoint=1,
    )
    summary = payload["summary"]
    kinds = _decisions(payload)
    assert kinds and set(kinds) <= {"insert", "stay"}
    # AVQDS never deletes: support is monotone non-decreasing.
    counts = [p["runtime_parameter_count"] for p in payload["trajectory"]["points"]]
    assert all(b >= a for a, b in zip(counts, counts[1:]))
    assert summary["runtime_parameter_count_final"] >= 2


@pytest.mark.parametrize(
    (
        "integrator",
        "solve_repair",
        "parameter_displacement_max",
        "retained_eigenmode_step_max",
    ),
    (
        ("rk4", SolveRepairConfig.minimal_profile(), None, None),
        ("euler", SolveRepairConfig(), 5.0e-3, None),
        ("euler", SolveRepairConfig(), None, 5.0e-3),
        (
            "rk4",
            replace(SolveRepairConfig.minimal_profile(), parameter_step_max=5.0e-3),
            None,
            None,
        ),
    ),
)
def test_structural_policies_share_the_evolution_kernel_when_no_patch_commits(
    integrator,
    solve_repair,
    parameter_displacement_max,
    retained_eigenmode_step_max,
) -> None:
    from pipelines.time_dynamics.run_lock import assert_comparable

    common = dict(
        deletion_enabled=False,
        insertion_gate_mode="mclachlan_l2",
        insertion_l2_cut=1.0e9,
        avqds_l2_cut=1.0e9,
        avqds_delta_theta_max=parameter_displacement_max,
        retained_eigenmode_step_max=retained_eigenmode_step_max,
        max_structural_pool_size=2,
    )
    payloads = []
    for policy in ("exchange", "avqds"):
        payloads.append(
            run_append_ap_mclachlan_from_runtime_input(
                _runtime_input(),
                times=tuple(0.05 * k for k in range(5)),
                integrator_method=integrator,
                pinv_rcond=1.0e-10,
                ridge_lambda=1.0e-7,
                controller_config=AppendControllerConfig(max_append_candidates=0),
                support_patch_config=SupportPatchControllerConfig(
                    **common, dynamics_policy=policy
                ),
                solve_repair_config=solve_repair,
            )
        )

    assert assert_comparable([payload["run_lock"] for payload in payloads])
    exchange_points, avqds_points = (
        payload["trajectory"]["points"] for payload in payloads
    )
    if retained_eigenmode_step_max is not None:
        for payload in payloads:
            assert payload["run_lock"]["shared_protocol"]["time_step_controller"][
                "retained_eigenmode_displacement_max"
            ] == pytest.approx(retained_eigenmode_step_max)
    assert len(exchange_points) == len(avqds_points)
    for exchange_point, avqds_point in zip(exchange_points, avqds_points):
        assert exchange_point["theta_runtime"] == pytest.approx(
            avqds_point["theta_runtime"], abs=1.0e-12
        )
        assert exchange_point["energy_expectation"] == pytest.approx(
            avqds_point["energy_expectation"], abs=1.0e-12
        )
        assert exchange_point["integration_to_next"] == avqds_point[
            "integration_to_next"
        ]
        assert exchange_point["parameter_stepping_to_next"] == avqds_point[
            "parameter_stepping_to_next"
        ]
        if retained_eigenmode_step_max is not None and exchange_point[
            "parameter_stepping_to_next"
        ] is not None:
            assert exchange_point["parameter_stepping_to_next"]["policy"] == (
                "retained_metric_eigenmode_controlled_euler_v1"
            )


def test_declared_exchange_round_cap_is_not_off_by_one() -> None:
    payload = _run(
        deletion_enabled=False,
        insertion_gate_mode="mclachlan_l2",
        insertion_l2_cut=1.0e-15,
        max_insertion_rounds_per_checkpoint=2,
    )
    for point in payload["trajectory"]["points"]:
        metadata = dict((point.get("patch_decision") or {}).get("metadata") or {})
        assert int(metadata.get("checkpoint_commit_count", 0)) <= 2
