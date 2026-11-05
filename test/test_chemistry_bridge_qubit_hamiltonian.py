"""qdynamics JSON -> our solvers entry point (pipelines.chemistry_bridge.qubit_hamiltonian) on a synthetic vibronic model.

Two spatial orbitals, one electron per spin, one mode: 4 fermion qubits + 1 boson qubit (cutoff 1) or 2 (cutoff 3).
Nothing here needs PySCF or Doron's bytes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.chemistry_bridge import qubit_hamiltonian as qh  # noqa: E402
from src.quantum.chemistry.molecular_vibronic import (  # noqa: E402
    ActiveSpace,
    ActiveTensors,
    LinearCoupling,
    MolecularGeometry,
    NormalMode,
    assemble_vibronic_model,
)
from src.quantum.chemistry.qdynamics_interop import physical_sector_energies, save_qdynamics_qubit_hamiltonian  # noqa: E402


def _synthetic_model(cutoff: int, *, coupling_scale: float = 0.08):
    rng = np.random.default_rng(7)
    h1 = np.array([[-1.2, -0.15], [-0.15, -0.4]])
    eri = np.zeros((2, 2, 2, 2))
    for p in range(2):
        for q in range(2):
            for r in range(2):
                for s in range(2):
                    eri[p, q, r, s] = 0.05 * (1.0 + 0.3 * ((p == q) + (r == s))) + (0.4 if p == q == r == s else 0.0)
    tensors = ActiveTensors(0.7, h1, eri)
    space = ActiveSpace((), (0, 1), (), 1, 1, "full")
    mode = NormalMode(0, 0.02, np.zeros((2, 3)), "mode_0")
    c = coupling_scale * np.array([[0.6, 0.2], [0.2, -0.3]])
    coupling = LinearCoupling(0, "mean_field_pyscf_eph", 0.0, c, None, 0.0)
    geometry = MolecularGeometry(("H", "H"), np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]))
    return assemble_vibronic_model(geometry=geometry, basis="synthetic", active_space=space, active_tensors=tensors,
                                   modes=(mode,), couplings=(coupling,), cutoffs=int(cutoff), encoding="binary")


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    base = tmp_path_factory.mktemp("qh")
    out = {}
    for cutoff in (1, 3):
        model = _synthetic_model(cutoff)
        out[cutoff] = (save_qdynamics_qubit_hamiltonian(model, base / f"synthetic_c{cutoff}.json", source_files=["synthetic"], metadata={"case": "synthetic"}), model)
    return out


def test_load_problem_identity_and_context(files):
    path, model = files[1]
    problem = qh.load_problem(path)
    assert problem.nq == 5 and problem.n_spatial == 2 and problem.particles == (1, 1) and problem.cutoffs == (1,)
    assert problem.identity()["raw_sha256"] == __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    assert problem.context.family_key == qh.FAMILY_KEY and problem.context.layout.boson_qubits == 1
    assert [b.name for b in problem.context.layout.blocks] == ["fermion", "boson", "boson_mode_0"]
    psi = problem.reference_state()
    assert psi[problem.reference_index] == 1.0 and np.count_nonzero(psi) == 1


@pytest.mark.parametrize("cutoff", [1, 3])
def test_exact_matches_the_sector_solver(files, cutoff):
    path, model = files[cutoff]
    problem = qh.load_problem(path)
    receipt = qh.run_exact(problem, n_low=2)
    assert receipt.status == qh.STATUS_VERIFIED
    assert receipt.results["sector_energies"] == pytest.approx(list(physical_sector_energies(model, n_low=2)), abs=1e-10)
    assert receipt.checks["sector_leakage_one_apply"] < 1e-12
    assert receipt.results["ground_energy"] <= receipt.results["hartree_fock_vacuum_energy"]
    assert len(receipt.results["ground_boson_occupations"]) == 1


@pytest.mark.parametrize("kind", qh.QSE_BASIS_KINDS)
def test_qse_ritz_values_are_upper_bounds(files, kind):
    path, _model = files[3]
    problem = qh.load_problem(path)
    exact = qh.run_exact(problem).results["sector_energies"]
    receipt = qh.run_qse(problem, basis_kind=kind, n_roots=3, exact_roots=exact)
    assert receipt.status == qh.STATUS_VERIFIED, receipt.checks
    assert receipt.results["ritz_roots"][0] >= exact[0] - 1e-9
    assert receipt.results["reference_energy"] == pytest.approx(qh.run_exact(problem).results["hartree_fock_vacuum_energy"], abs=1e-10)
    assert receipt.results["basis_counts"]["uccsd_lifted"] == 3
    if kind != "uccsd":
        assert receipt.results["basis_size"] > 4
        assert receipt.results["ritz_roots"][0] <= qh.run_qse(problem, basis_kind="uccsd", n_roots=3).results["ritz_roots"][0] + 1e-9


def test_apm_runs_without_a_dense_matrix(files):
    path, _model = files[1]
    problem = qh.load_problem(path)
    receipt = qh.run_apm(problem, times=(0.0, 0.05, 0.10), pool_kind="hermitian_excitations_and_displacements", parameterization_mode="logical_shared")
    assert receipt.status == qh.STATUS_VERIFIED, receipt.checks
    assert receipt.checks["sector_leakage_max_norm"] <= 1e-8
    assert receipt.checks["initial_energy_minus_reference_abs"] <= 1e-10
    assert receipt.checks["point_count"] == 3 and receipt.results["pool_counts"]["mode_coordinates"] == 1
    assert 0.0 < receipt.checks["min_fidelity_vs_exact"] <= 1.0 + 1e-12
    assert receipt.results["exact_energies_used_online"] is False if "exact_energies_used_online" in receipt.results else True


def test_apm_refuses_padded_boson_codes(tmp_path):
    model = _synthetic_model(2)   # cutoff 2 on 2 qubits leaves code 3 unphysical
    path = save_qdynamics_qubit_hamiltonian(model, tmp_path / "padded.json")
    with pytest.raises(ValueError, match="padded"):
        qh.run_apm(qh.load_problem(path), times=(0.0, 0.05))


def test_cli_report_and_exit_codes(files, tmp_path):
    path, _model = files[1]
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    out_dir = tmp_path / "out"
    completed = subprocess.run([sys.executable, "-m", "pipelines.chemistry_bridge.qubit_hamiltonian", "--hamiltonian", str(path), "--out-dir", str(out_dir),
                                "--methods", "exact,qse", "--qse-basis", "uccsd", "--declared-origin", "synthetic", "--run-class", "smoke"],
                               cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stderr[-2000:]
    report = json.loads((out_dir / "qubit_hamiltonian_report.json").read_text())
    assert report["schema"] == qh.REPORT_SCHEMA and report["status_by_method"] == {"exact": "verified", "qse": "verified"}
    assert report["inputs"]["raw_sha256"] and report["repository"]["commit"] and report["environment"]["python"]
    assert report["origin"]["declared_origin"] == "synthetic" and report["origin"]["producer_origin_established"] is False
    assert report["exact_energies_used_online"] is False and report["run_class"] == "smoke"
    assert [m["method"] for m in report["methods"]] == ["exact", "qse"] and all("requested" in m and "executed" in m for m in report["methods"])
    refused = subprocess.run([sys.executable, "-m", "pipelines.chemistry_bridge.qubit_hamiltonian", "--hamiltonian", str(tmp_path / "missing.json"), "--out-dir", str(tmp_path / "x")],
                             cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=120)
    assert refused.returncode == 2 and "refused input" in refused.stderr
    bad = subprocess.run([sys.executable, "-m", "pipelines.chemistry_bridge.qubit_hamiltonian", "--hamiltonian", str(path), "--out-dir", str(tmp_path / "y"), "--methods", "ra"],
                         cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=120)
    assert bad.returncode == 2
