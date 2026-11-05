"""Doron Wu's vendored pipeline (third_party/qdynamics_chemistry, jacopos86/qdynamics@38f40805) against our builder.

The integrity test always runs. The H2O cross-check runs his PySCF driver and his qubitizer as subprocesses
(pyscf + h5py needed, e.g. the isolated venv ~/.venvs/doron-bridge-gpu) and compares his JSON with our
builder set to his model, term by term, after matching mode order and mode signs."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.chemistry.molecular_vibronic import MolecularGeometry  # noqa: E402
from src.quantum.chemistry.qdynamics_interop import (  # noqa: E402
    builder_model_in_qdynamics_conventions,
    classify_term_differences,
    hartree_fock_vacuum_energy,
    load_qdynamics_qubit_hamiltonian,
    match_modes_to_reference,
    physical_sector_energies,
    qdynamics_qubit_hamiltonian_from_payload,
)
from src.quantum.chemistry.qdynamics_vendored import (  # noqa: E402
    UPSTREAM_COMMIT,
    VENDORED_ROOT,
    load_manifest,
    read_h5_metadata,
    run_pyscf_driver,
    run_qubitizer,
    verify_vendored_files,
    write_input_deck,
)

# H2O at the RHF/STO-3G optimum (our Paper IV geometry), Bohr.
H2O_OPT_BOHR = np.array([[0.0, 0.0, -0.06340060329635555],
                         [0.0, 1.432278617817422, 1.1388573462159217],
                         [0.0, -1.4322786178174232, 1.1388573462159226]])
N_FERMION_QUBITS = 14


def test_vendored_files_match_manifest():
    digests = verify_vendored_files()
    manifest = load_manifest()
    assert manifest["commit"] == UPSTREAM_COMMIT and len(digests) == 22
    assert all(entry["matches_upstream_tree"] for entry in manifest["files"].values())
    assert (VENDORED_ROOT / "LICENSE").read_text().lstrip().startswith("GNU GENERAL PUBLIC LICENSE")
    assert (VENDORED_ROOT / "conftest.py").is_file()   # keeps his tests out of our collection


@pytest.fixture(scope="module")
def h2o(tmp_path_factory):
    pytest.importorskip("pyscf")
    pytest.importorskip("h5py")
    geometry = MolecularGeometry(("O", "H", "H"), H2O_OPT_BOHR)
    model, computation = builder_model_in_qdynamics_conventions(geometry, cutoffs=1)
    ours_frequencies = [m.frequency_hartree for m in model.modes]
    base = tmp_path_factory.mktemp("doron")
    runs = {}
    for label, masses in (("isotope", "ours"), ("default", None)):
        deck = write_input_deck(geometry, base / label, name="h2o", isotope_masses_amu=masses)
        run = run_pyscf_driver(deck)
        json_path = run_qubitizer([deck.ele_h5, deck.eph_h5], base / label / "h2o_c1.json", cutoffs=1)
        payload = json.loads(json_path.read_text())
        matched, order, signs = match_modes_to_reference(payload, ours_frequencies, model.h_vibronic)
        runs[label] = {"deck": deck, "run": run, "raw": load_qdynamics_qubit_hamiltonian(json_path),
                       "matched": qdynamics_qubit_hamiltonian_from_payload(matched), "order": order, "signs": signs,
                       "meta": read_h5_metadata(deck.ele_h5)}
    return {"model": model, "computation": computation, "frequencies": ours_frequencies, "runs": runs}


def test_his_driver_reproduces_our_scf(h2o):
    for run in h2o["runs"].values():
        meta = run["meta"]
        assert meta["converged"] and int(meta["nelec"]) == 10 and int(meta["nmo"]) == 7
        assert float(meta["e_scf"]) == pytest.approx(h2o["computation"].center.hf_energy, abs=1.0e-9)
        assert run["raw"].n_qubits == 17 and run["raw"].num_particles == (5, 5) and run["raw"].mode_cutoffs == (1, 1, 1)


def test_his_file_equals_ours_with_isotope_masses(h2o):
    model, run = h2o["model"], h2o["runs"]["isotope"]
    assert sorted(run["order"]) == [0, 1, 2]
    np.testing.assert_allclose(run["matched"].frequencies_hartree, h2o["frequencies"], rtol=1.0e-7)
    classes = classify_term_differences(run["matched"].operator, model.h_vibronic, N_FERMION_QUBITS)
    assert classes["electronic"]["n_terms"] == 1085 and classes["electronic"]["max_abs_difference"] < 1.0e-12
    assert classes["identity"]["max_abs_difference"] < 1.0e-9
    assert classes["oscillator"]["max_abs_difference"] < 1.0e-9
    # Residual on the coupling terms: his Hessian keeps translations and rotations, ours projects them out.
    assert classes["coupling"]["n_terms"] == 116 and classes["coupling"]["max_abs_difference"] < 2.0e-5
    assert hartree_fock_vacuum_energy(run["matched"]) == pytest.approx(hartree_fock_vacuum_energy(model), abs=1.0e-9)
    his_e0 = physical_sector_energies(run["matched"])[0]
    assert his_e0 == pytest.approx(physical_sector_energies(model)[0], abs=1.0e-6)
    assert his_e0 < hartree_fock_vacuum_energy(model) - 0.15        # the HF (x) vacuum energy is not a ground state


def test_his_default_masses_overcorrect_every_frequency(h2o):
    from pyscf.data.nist import AMU2AU, MP_ME  # type: ignore

    model, run = h2o["model"], h2o["runs"]["default"]
    ratio = np.asarray(run["matched"].frequencies_hartree) / np.asarray(h2o["frequencies"])
    # PySCF's eph uses integer mass numbers x MP_ME; his EPH_FREQUENCY_SCALE undoes the MP_ME part only.
    np.testing.assert_allclose(ratio, np.sqrt(MP_ME / AMU2AU), atol=2.0e-4)
    classes = classify_term_differences(run["matched"].operator, model.h_vibronic, N_FERMION_QUBITS)
    assert classes["electronic"]["max_abs_difference"] < 1.0e-12
    assert classes["coupling"]["max_abs_difference"] > 1.0e-4
    zpe_shift = 0.5 * (sum(run["matched"].frequencies_hartree) - sum(h2o["frequencies"]))
    assert hartree_fock_vacuum_energy(run["matched"]) - hartree_fock_vacuum_energy(model) == pytest.approx(zpe_shift, abs=1.0e-9)
