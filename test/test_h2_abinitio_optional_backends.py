from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_OPTIONAL_CHEMISTRY") != "1",
    reason="Optional chemistry backend checks require RUN_OPTIONAL_CHEMISTRY=1.",
)


def test_optional_pyscf_fci_cross_check_smoke() -> None:
    pytest.importorskip("pyscf")
    from src.quantum.chemistry.generate_h2_abinitio_fixture import pyscf_fci_h2_energy

    result = pyscf_fci_h2_energy(bond_length_angstrom=0.7414, basis="sto-3g")
    assert result["backend"] == "pyscf"
    assert result["fci_total_energy_hartree"] < result["hf_total_energy_hartree"]


def test_optional_psi4_three_point_curve_generation_smoke() -> None:
    pytest.importorskip("psi4")
    from src.quantum.chemistry.generate_h2_abinitio_fixture import build_h2_curve_with_psi4
    from src.quantum.chemistry.h2_abinitio_curve import derive_local_taylor_product

    curve = build_h2_curve_with_psi4(
        r_values_angstrom=[0.7314, 0.7414, 0.7514],
        center_r_angstrom=0.7414,
        basis="sto-3g",
    )
    local = derive_local_taylor_product(
        curve,
        center_R_angstrom=0.7414,
        step_angstrom=0.01,
    )
    assert local["schema"] == "h2_local_taylor_product_v1"
    assert local["omega_au"] > 0.0
