from __future__ import annotations

from pathlib import Path

import numpy as np

from pipelines.contracts.problem import ProblemRequest
from pipelines.exact_bench.audit_h2o_linear_fd_fixture import (
    _closed_shell_energy_from_tensors,
    audit_h2o_linear_fd_fixture,
)
from pipelines.static_adapt.builders.problem_registry import resolve_problem_context
from pipelines.static_adapt.sector_invariants import FixedCountSectorStateAuditor
from src.quantum.chemistry.generate_h2o_linear_fd_fixture import (
    build_h2o_linear_fd_fixture_from_record,
)
from src.quantum.chemistry.vibronic_h2o_linear_fd import (
    load_cached_production_vibronic_h2o_linear_fd_fixture,
    write_production_vibronic_h2o_fixture,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from test_support.h2o_linear_fd_fixture_factory import (
    synthetic_three_mode_h2o_linear_fd_backend_record,
)


def test_closed_shell_tensor_energy_matches_synthetic_reference() -> None:
    fixture = build_h2o_linear_fd_fixture_from_record(
        synthetic_three_mode_h2o_linear_fd_backend_record(),
        mode_cutoffs=(1, 1, 1),
        reference_cutoffs=(1, 1, 1),
        dense_full_dim_cap=512,
        embed_exact_state=True,
    )
    active = fixture.active_space
    energy = _closed_shell_energy_from_tensors(
        active.scalar_energy_hartree,
        active.one_body_integrals,
        active.two_body_integrals,
        n_occupied=active.num_particles[0],
    )
    assert energy < 0.0


def test_small_fixture_correctness_audit_passes(tmp_path: Path) -> None:
    fixture = build_h2o_linear_fd_fixture_from_record(
        synthetic_three_mode_h2o_linear_fd_backend_record(),
        mode_cutoffs=(1, 1, 1),
        reference_cutoffs=(1, 1, 1),
        dense_full_dim_cap=512,
        embed_exact_state=True,
    )
    fixture_path = tmp_path / "fixture.json"
    write_production_vibronic_h2o_fixture(fixture_path, fixture)

    audit = audit_h2o_linear_fd_fixture(fixture_path)

    assert audit["implementation_checks_passed"] is True
    assert audit["vibronic_reference_checks"]["pool_sector_contract"]["execution_passed"] is True
    assert audit["vibronic_reference_checks"]["eigenstate_residual_l2"] < 1.0e-9


def test_grouped_coupled_macro_preserves_declared_sector_at_finite_angle(
    tmp_path: Path,
) -> None:
    fixture = build_h2o_linear_fd_fixture_from_record(
        synthetic_three_mode_h2o_linear_fd_backend_record(),
        mode_cutoffs=(1, 1, 1),
        reference_cutoffs=(1, 1, 1),
        dense_full_dim_cap=512,
        embed_exact_state=True,
    )
    fixture_path = tmp_path / "fixture.json"
    write_production_vibronic_h2o_fixture(fixture_path, fixture)
    cached = load_cached_production_vibronic_h2o_linear_fd_fixture(fixture_path)
    resolved = resolve_problem_context(
        ProblemRequest(
            problem_key="molecular_vibronic_h2o_linear_fd",
            num_sites=2,
            t=0.0,
            u=0.0,
            dv=0.0,
            omega0=0.0,
            g_ep=1.0,
            n_ph_max=1,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
            molecular_vibronic_h2o_linear_fd_fixture_json=str(fixture_path),
        )
    )
    coupled = next(
        term
        for term in cached.model.pool
        if term.label.startswith("coupled::")
    )

    assert coupled.execution_mode == "grouped_exact"
    executor = CompiledAnsatzExecutor(
        [coupled],
        parameterization_mode="logical_shared",
    )
    state = executor.prepare_state(np.asarray([0.137]), cached.model.psi_ref)
    sector_audit = FixedCountSectorStateAuditor(resolved).audit(
        state,
        source="finite_angle_grouped_coupled_macro",
    )

    assert sector_audit["passed"] is True
    assert abs(np.linalg.norm(state) - 1.0) < 1.0e-10
