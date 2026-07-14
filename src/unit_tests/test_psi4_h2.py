from __future__ import annotations
import shutil
import sys
from pathlib import Path
import numpy as np
import pytest
import yaml

psi4 = pytest.importorskip("psi4")

#
#   write temp H2 file
#

def _write_temp_h2_input(tmp_path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    source_dir = repo_root / "TESTS" / "2"
    for filename in ("h2.xyz", "h_basis.gbs"):
        shutil.copy2(source_dir / filename, tmp_path / filename)
    with (source_dir / "input.yml").open("r") as f:
        data = yaml.safe_load(f)
    data["working_dir"] = str(tmp_path)
    data["output_dir"] = str(tmp_path / "PSI4-OUT")
    data["coordinate_file"] = "h2.xyz"
    data["optimized_coordinate_file"] = "h2_optimized.xyz"
    data["basis_set_file"] = "h_basis.gbs"
    input_file = tmp_path / "input.yml"
    with input_file.open("w") as f:
        yaml.safe_dump(data, f)
    return input_file

#
#  test PSI4 energies calculation
#

def test_psi4_h2_expectation_values_match_psi4_energies(tmp_path, monkeypatch):
    input_file = _write_temp_h2_input(tmp_path)
    monkeypatch.delenv("PYDEPHASING_TESTING", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pytest",
            "-ct1",
            "GS",
            "-co",
            "elec-sys",
            "-ct2",
            "PSI4",
            "-yml_inp",
            str(input_file),
        ],
    )
    # import time
    from src.chemistry.psi4.electron_solver import Psi4Driver
    from src.parameters.set_param_object import p
    p._real_p = None
    p.read_yml_data(input_file)
    psi4.core.clean()
    psi4_driver = Psi4Driver(p.basis_set_file, p.psi4_calc_parameters)
    geometry = psi4_driver.psi4_geometry_driver(
        p.coordinate_file,
        p.optimized_coordinate_file,
        p.charge,
        p.multiplicity,
    )
    wavefunction = psi4_driver.psi4_elec_struct_driver(geometry)
    _, mo, density_matrix, hamiltonian = psi4_driver.set_electronic_operators(
        wavefunction
    )
    psi4_driver.build_operators_MO_basis(mo, density_matrix, hamiltonian)
    one_particle_energy = np.trace(density_matrix.Dae @ hamiltonian.hij)
    occupations = np.diag(density_matrix.Dae)
    two_particle_energy = 0.0
    for p_index, p_occ in enumerate(occupations):
        if p_occ == 0.0:
            continue
        for q_index, q_occ in enumerate(occupations):
            if q_occ == 0.0:
                continue
            two_particle_energy += 0.5 * p_occ * q_occ * (
                hamiltonian.Iijkl[p_index, p_index, q_index, q_index]
                - hamiltonian.Iijkl[p_index, q_index, q_index, p_index]
            )
    total_energy = (
        one_particle_energy
        + two_particle_energy
        + hamiltonian.Vnn.magnitude
    )
    assert one_particle_energy == pytest.approx(
        wavefunction.energy_1p.magnitude, abs=1.0e-6
    )
    assert two_particle_energy == pytest.approx(
        wavefunction.energy_2p.magnitude, abs=1.0e-6
    )
    assert total_energy == pytest.approx(wavefunction.energy.magnitude, abs=1.0e-6)
