"""Named molecular closed-shell RA-ADAPT application: admission, identity, route, and a real run.

The problems are the bridge's synthetic witnesses (``pipelines.chemistry_bridge``):
small closed-shell contexts produced by the same strict intake as Doron's export,
so every test here runs in seconds without the dense 4096-dimensional exact
reference of the 12-qubit H2O case.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipelines.chemistry_bridge.methods import resolve_bridge_context, run_exact_reference
from pipelines.chemistry_bridge.qdynamics_export import (
    bridge_problem_from_qdynamics,
    qdynamics_matrix_elements_from_mapping,
    write_bridge_problem_json,
)
from pipelines.chemistry_bridge.synthetic_witnesses import synthetic_qdynamics_mapping
from pipelines.static_adapt.ra_adapt.adapters import MacroCandidateAdapter
from pipelines.static_adapt.ra_adapt.contracts import RAAdaptRequest
from pipelines.static_adapt.ra_adapt.engine import build_resolved_ra_protocol, run_ra_adapt
from pipelines.static_adapt.ra_adapt.molecular_closed_shell import (
    MOLECULAR_CLOSED_SHELL_APPLICATION_LANE,
    MOLECULAR_CLOSED_SHELL_FAMILY,
    MOLECULAR_CLOSED_SHELL_MACRO_ADAPTER_ID,
    MOLECULAR_CLOSED_SHELL_ROUTE_PREFIX,
    MolecularClosedShellMacroCandidateAdapter,
    build_molecular_closed_shell_ra_request,
    is_molecular_closed_shell_application,
    require_molecular_closed_shell_problem,
)
from pipelines.static_adapt.sr_snake.contracts import (
    MetricPruning,
    SRExecutionPolicy,
    SRMethodPolicy,
    SRStopPolicy,
)


def _context(tmp_path: Path, kind: str):
    mapping = synthetic_qdynamics_mapping(kind)
    elements = qdynamics_matrix_elements_from_mapping(mapping, source={"kind": kind, "simulation": True, "producer_generated": False}, declared_origin="synthetic")
    manifest = write_bridge_problem_json(bridge_problem_from_qdynamics(elements), tmp_path / f"{kind}.json")
    return resolve_bridge_context(manifest["manifest_path"])


def test_default_request_stays_locked_and_foreign_adapters_are_refused(tmp_path):
    bridge = _context(tmp_path, "random6")
    with pytest.raises(ValueError, match="locked"):
        run_ra_adapt(bridge.context)
    with pytest.raises(ValueError, match="locked"):
        run_ra_adapt(bridge.context, RAAdaptRequest())
    with pytest.raises(ValueError, match="locked"):
        build_resolved_ra_protocol(bridge.context, RAAdaptRequest(adapter=MacroCandidateAdapter(pool_key="uccsd")))
    assert is_molecular_closed_shell_application(bridge.context, RAAdaptRequest()) is False


def test_adapter_identity_pool_and_policy_are_locked(tmp_path):
    bridge = _context(tmp_path, "random6")
    adapter = MolecularClosedShellMacroCandidateAdapter()
    assert adapter.application_family_key == MOLECULAR_CLOSED_SHELL_FAMILY and adapter.adapter_id == MOLECULAR_CLOSED_SHELL_MACRO_ADAPTER_ID
    assert adapter.pool_key == "uccsd"
    with pytest.raises(ValueError, match="identity is fixed"):
        MolecularClosedShellMacroCandidateAdapter(adapter_id="something_else")
    with pytest.raises(ValueError, match="explicit pool_key"):
        MolecularClosedShellMacroCandidateAdapter(pool_key=None)
    with pytest.raises(ValueError, match="not admissible"):
        MolecularClosedShellMacroCandidateAdapter(pool_key="no_such_pool").parent_inventory(bridge.context)
    request = build_molecular_closed_shell_ra_request(maximum_controller_rounds=2)
    assert is_molecular_closed_shell_application(bridge.context, request) is True
    pruned = RAAdaptRequest(adapter=adapter, method=SRMethodPolicy(pruning=MetricPruning(small_theta_abs=1.0e-10)), execution=SRExecutionPolicy(stop=SRStopPolicy(maximum_controller_rounds=2)))
    with pytest.raises(ValueError, match="locked to singleton admission"):
        is_molecular_closed_shell_application(bridge.context, pruned)
    with pytest.raises(ValueError, match="requires its problem"):
        is_molecular_closed_shell_application(None, request)
    with pytest.raises(ValueError, match="insertion must be one of"):
        build_molecular_closed_shell_ra_request(maximum_controller_rounds=2, insertion="beam")
    with pytest.raises(ValueError, match="positive"):
        build_molecular_closed_shell_ra_request(maximum_controller_rounds=0)


def test_executable_pool_equals_audited_parent_pool(tmp_path):
    bridge = _context(tmp_path, "random8")
    adapter = MolecularClosedShellMacroCandidateAdapter()
    parent = adapter.parent_inventory(bridge.context)
    executable = adapter.executable_pool(bridge.context)
    assert len(parent.candidates) == len(executable.candidates) > 0  # UCCSD generators preserve the fixed counts
    assert [c.label for c in executable.candidates] == [c.label for c in parent.candidates]
    assert executable.metadata["sector_audit_removed_count"] == 0 and parent.metadata["pool_key"] == "uccsd"
    assert all(c.execution_mode == "termwise_product" for c in executable.candidates)
    require_molecular_closed_shell_problem(bridge.context)


def test_named_application_runs_and_names_its_lane(tmp_path):
    bridge = _context(tmp_path, "random6")
    exact = run_exact_reference(bridge, "numpy:float64", n_roots=2).results["low_roots"][0]
    hf_energy = float(bridge.intake_receipt["checks"]["input_tensor_determinant_energy"])
    for insertion, rounds in (("append_only", 2), ("plateau_commutation", 4)):
        request = build_molecular_closed_shell_ra_request(maximum_controller_rounds=rounds, insertion=insertion)
        result = run_ra_adapt(bridge.context, request)
        energies = [float(s.energy) for s in result.run.accepted_trajectory]
        assert len(energies) == rounds
        assert all(b <= a for a, b in zip([hf_energy] + energies, energies))  # measured: 1.0634 -> 0.4515 -> -0.5150 (-> -1.1168 -> -1.6086)
        assert energies[-1] >= exact - 1.0e-9  # variational against the exact sector ground energy (-1.750066)
        route = result.scientific_receipts["resolved_route_contract"]
        assert route["route_profile"].startswith(MOLECULAR_CLOSED_SHELL_ROUTE_PREFIX)
        execution, invariants = route["execution_settings"], route["semantic_invariants"]
        assert execution["problem"] == MOLECULAR_CLOSED_SHELL_FAMILY and execution["adapt_pool"] == "uccsd" and execution["phase3_backend_cost_mode"] == "proxy"
        assert "phase3_backend_cost_scope" not in execution
        assert invariants["application_lane"] == MOLECULAR_CLOSED_SHELL_APPLICATION_LANE and invariants["problem_family"] == MOLECULAR_CLOSED_SHELL_FAMILY
        assert invariants["operator_pool_identity"] == "uccsd" and "hh_preseed_policy" not in invariants and "candidate_parent_pool" not in invariants
        assert result.run.stop.primary_reason == "maximum_controller_rounds" and result.run.stop.exact_target_energy is None
        assert result.run.final_state.operators[0] == "uccsd_dbl(ab:0,3->2,5)"
