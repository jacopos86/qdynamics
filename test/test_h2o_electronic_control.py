from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipelines.exact_bench.export_h2o_electronic_control import (
    ELECTRONIC_CONTROL_SCHEMA,
    write_h2o_electronic_control,
)
from src.quantum.chemistry.generate_h2o_linear_fd_fixture import (
    build_h2o_linear_fd_fixture_from_record,
)
from src.quantum.chemistry.molecular_hamiltonian import (
    build_restricted_closed_shell_molecular_hamiltonian,
)
from src.quantum.chemistry.psi4_adapter import (
    load_restricted_closed_shell_problem_from_json,
)
from src.quantum.chemistry.vibronic_h2o_linear_fd import (
    write_production_vibronic_h2o_fixture,
)
from src.quantum.vqe_latex_python_pairs import exact_ground_energy_sector
from test_support.h2o_linear_fd_fixture_factory import (
    synthetic_three_mode_h2o_linear_fd_backend_record,
)


def test_electronic_control_reproduces_source_fixture_hamiltonian_and_pool(
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
    control_path = tmp_path / "electronic_control.json"
    write_production_vibronic_h2o_fixture(fixture_path, fixture)

    write_h2o_electronic_control(fixture_path, control_path)
    problem = load_restricted_closed_shell_problem_from_json(control_path)
    hamiltonian = build_restricted_closed_shell_molecular_hamiltonian(problem)

    payload = json.loads(control_path.read_text(encoding="utf-8"))
    assert payload["schema"] == ELECTRONIC_CONTROL_SCHEMA
    assert payload["problem"]["n_spatial_orbitals"] == 2
    assert payload["problem"]["n_alpha"] == 1
    assert payload["problem"]["n_beta"] == 1
    assert payload["hamiltonian_contract"]["n_fermion_qubits"] == 4
    assert payload["hamiltonian_contract"]["n_boson_qubits"] == 0
    assert payload["hamiltonian_contract"]["source_electronic_block_parity"][
        "coefficient_delta_max_abs"
    ] == 0.0
    assert payload["hamiltonian_contract"]["source_electronic_block_parity"][
        "left_term_count"
    ] == payload["hamiltonian_contract"]["source_electronic_block_parity"][
        "right_term_count"
    ]
    assert payload["pool_contract"]["generator_count"] == 3
    assert payload["pool_contract"]["matches_source_fixture_electronic_pool"] is True
    assert payload["source"]["fixture_sha256"] == hashlib.sha256(
        fixture_path.read_bytes()
    ).hexdigest()
    assert payload["exact_energy"] == pytest.approx(
        exact_ground_energy_sector(
            hamiltonian,
            num_sites=2,
            num_particles=(1, 1),
            indexing="blocked",
            tol=0.0,
        ),
        abs=1.0e-12,
    )
