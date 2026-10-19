"""Convention mapping from Doron's qdynamics export schema to the strict intake.

Every fixture here is synthetic and labelled so; nothing is a molecule or a
producer output.  The positive witnesses prove the converter reproduces the
spatial tensors it was given and that its output passes the strict intake with
a zero HF-witness defect; the negative controls prove each named convention
check fails closed with its code.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.chemistry_bridge import qdynamics_export as qx
from pipelines.chemistry_bridge.methods import resolve_bridge_context
from pipelines.chemistry_bridge.polynomial_action import (
    BridgePolynomialAction,
    fixed_sector_basis_indices,
    sector_matrix,
    sector_spectrum,
)
from pipelines.chemistry_bridge.array_backend import NumpyBackend
from pipelines.chemistry_bridge.synthetic_witnesses import (
    SYNTHETIC_KINDS,
    contract_four_qubit_tensors,
    qdynamics_mapping_from_spatial,
    random_symmetric_tensors,
    synthetic_qdynamics_mapping,
)
from pipelines.contracts.problem import ProblemRequest
from pipelines.static_adapt.builders.molecular_diagnostic_intake import resolve_molecular_diagnostic_intake


def _elements(mapping, **source):
    return qx.qdynamics_matrix_elements_from_mapping(mapping, source={"payload_kind": "synthetic_witness", **source})


def _intake(bridge_problem: qx.BridgeProblem, tmp_path: Path, name: str = "problem.json"):
    manifest = qx.write_bridge_problem_json(bridge_problem, tmp_path / name)
    request = ProblemRequest(
        problem_key="molecular_restricted_closed_shell",
        num_sites=int(bridge_problem.problem.n_spatial_orbitals),
        t=0.0, u=0.0, dv=0.0, omega0=0.0, g_ep=0.0, n_ph_max=0,
        boson_encoding="binary", ordering="blocked", boundary="open",
        molecular_problem_json=manifest["problem_json_path"],
    )
    context = resolve_molecular_diagnostic_intake(
        request,
        expected_raw_sha256=manifest["raw_sha256"],
        declaration=qx.declaration_from_mapping(manifest["declaration"]),
    )
    return context, manifest


def test_interleaved_extraction_round_trips_spatial_tensors():
    h, g = random_symmetric_tensors(4, seed=3)
    mapping = qdynamics_mapping_from_spatial(h, g, nuclear_repulsion=0.5, n_occupied=2)
    extracted = qx.spatial_tensors_from_interleaved(mapping["h1"], mapping["h2"])
    assert np.array_equal(extracted.h_mo, h)
    assert np.array_equal(extracted.eri_mo, g)
    assert all(value == 0.0 for value in extracted.checks.values())


@pytest.mark.parametrize("kind", SYNTHETIC_KINDS)
def test_converter_output_passes_strict_intake_with_zero_witness_defect(kind, tmp_path):
    mapping = synthetic_qdynamics_mapping(kind)
    bridge_problem = qx.bridge_problem_from_qdynamics(_elements(mapping, kind=kind))
    context, manifest = _intake(bridge_problem, tmp_path)
    receipt = context.runtime_data["molecular_diagnostic_intake_receipt"]
    assert receipt["raw_sha256"] == manifest["raw_sha256"]
    assert receipt["checks"]["hf_witness_abs_defect"] == 0.0
    assert context.layout.total_qubits == 2 * int(mapping["metadata"]["nmo"])
    assert bridge_problem.declaration.scalar_semantics == "nuclear_only"
    assert bridge_problem.provenance["notation_binding"].startswith("declared_by_producer_metadata")
    document = json.loads(Path(manifest["problem_json_path"]).read_text())
    assert document["schema"] == qx.BRIDGE_PROBLEM_SCHEMA
    assert "n_spatial_orbitals" in document["problem"] and "n_spatial_orbitals" not in document


def _base_mapping():
    return synthetic_qdynamics_mapping("random6")


@pytest.mark.parametrize(
    "mutate, code",
    [
        (lambda m: m["metadata"].__setitem__("h1_convention", "spin_orbital_blocked"), "h1_convention"),
        (lambda m: m["metadata"].__setitem__("h2_convention", "physicist_spin_orbital"), "h2_convention"),
        (lambda m: m["metadata"].__setitem__("h2_antisymmetrized", True), "h2_antisymmetrized"),
        (lambda m: m["metadata"].pop("h2_antisymmetrized"), "h2_antisymmetrized_missing"),
        (lambda m: m["metadata"].__setitem__("method", "RKS"), "method"),
        (lambda m: m["metadata"].__setitem__("method", "UHF"), "method"),
        (lambda m: (m["metadata"].__setitem__("spin", 1), m["metadata"].__setitem__("ms2", 1)), "open_shell"),
        (lambda m: m["metadata"].__setitem__("converged", False), "scf_not_converged"),
        (lambda m: m["metadata"].__setitem__("e_scf", m["metadata"]["e_scf"] + 1.0e-3), "hf_witness"),
        (lambda m: m["metadata"].__setitem__("nmo", 5), "nmo"),
        (lambda m: m["metadata"].__setitem__("nelec", 3), "nelec"),
        (lambda m: m["h1"].__setitem__((0, 1), 1.0e-6), "h1_cross_spin_nonzero"),
        (lambda m: m["h1"].__setitem__((1, 1), m["h1"][1, 1] + 1.0e-6), "h1_spin_blocks_differ"),
        (lambda m: m["h2"].__setitem__((0, 1, 0, 0), 1.0e-6), "h2_spin_forbidden_nonzero"),
        (lambda m: m["h2"].__setitem__((1, 1, 1, 1), m["h2"][1, 1, 1, 1] + 1.0e-6), "h2_spin_block_placement"),
        (lambda m: m.__setitem__("mo_occ", np.array([0.0, 2.0, 0.0])), "occupied_prefix"),
    ],
)
def test_negative_controls_fail_closed_with_named_codes(mutate, code):
    mapping = _base_mapping()
    mutate(mapping)
    with pytest.raises(qx.QdynamicsConventionError) as excinfo:
        qx.bridge_problem_from_qdynamics(_elements(mapping))
    assert excinfo.value.code == code


def test_physicists_ordered_two_body_tensor_fails_symmetry_for_generic_tensors():
    """A physicists' <pq|rs> = (pr|qs) tensor breaks the first-pair swap for generic
    integrals.  This is a necessary check only: notation is bound by the producer's
    declared h2_convention, because degenerate tensors can pass symmetry."""

    h, g = random_symmetric_tensors(3, seed=11)
    physicists = g.transpose(0, 2, 1, 3)
    mapping = qdynamics_mapping_from_spatial(h, physicists, nuclear_repulsion=0.25, n_occupied=1)
    with pytest.raises(qx.QdynamicsConventionError) as excinfo:
        qx.bridge_problem_from_qdynamics(_elements(mapping))
    assert excinfo.value.code == "eri_symmetry"


def test_complex_tensor_with_nonzero_imaginary_part_is_rejected():
    mapping = _base_mapping()
    mapping["h1"] = mapping["h1"].astype(complex)
    mapping["h1"][0, 0] += 1.0e-9j
    with pytest.raises(qx.QdynamicsConventionError) as excinfo:
        qx.qdynamics_matrix_elements_from_mapping(mapping)
    assert excinfo.value.code == "complex_tensor"


def test_frozen_core_contract_matches_repository_helper():
    from src.quantum.chemistry.generate_h2o_linear_fd_fixture import _freeze_core_active_tensors
    from src.quantum.chemistry.psi4_adapter import RestrictedClosedShellMolecularProblem

    h, g = random_symmetric_tensors(5, seed=5)
    problem = RestrictedClosedShellMolecularProblem(
        geometry_spec="synthetic", basis="synthetic", charge=0, multiplicity=1, reference="rhf",
        n_spatial_orbitals=5, n_alpha=3, n_beta=3, hf_energy=0.0, nuclear_repulsion_energy=0.7,
        one_body_integrals_mo=h, two_body_integrals_mo=g,
    )
    expected_scalar, expected_h, expected_g = _freeze_core_active_tensors(problem, active_indices=(1, 2, 3, 4), frozen_core_indices=(0,))
    scalar, h_active, g_active = qx.frozen_core_contract(h, g, 0.7, frozen=(0,), active=(1, 2, 3, 4))
    assert abs(scalar - expected_scalar) < 1.0e-12
    assert np.max(np.abs(h_active - expected_h)) < 1.0e-12
    assert np.array_equal(g_active, expected_g)


def test_frozen_core_reduction_reproduces_core_occupied_spectrum(tmp_path):
    """Freezing orbital 0 of an m=3 (2,2) problem gives an m=2 (1,1) problem whose
    sector spectrum equals the full Hamiltonian restricted to determinants with
    orbital 0 doubly occupied.  The HF witness is preserved across the contraction."""

    mapping = synthetic_qdynamics_mapping("random6_two_occupied")
    full = qx.bridge_problem_from_qdynamics(_elements(mapping))
    reduced = qx.bridge_problem_from_qdynamics(_elements(mapping), frozen_core=qx.FrozenCoreReduction(1))
    assert reduced.declaration.scalar_semantics == "nuclear_plus_frozen_core"
    assert reduced.declaration.one_body_core_policy == "already_contracted"
    assert reduced.problem.hf_energy == full.problem.hf_energy
    assert reduced.provenance["reduction"]["kind"] == "frozen_core_occupied_prefix"
    full_context, _ = _intake(full, tmp_path, "full.json")
    reduced_context, _ = _intake(reduced, tmp_path, "reduced.json")
    assert full_context.layout.total_qubits == 6 and reduced_context.layout.total_qubits == 4
    backend = NumpyBackend()
    full_action = BridgePolynomialAction.from_polynomial(full_context.hamiltonian).bind(backend)
    reduced_action = BridgePolynomialAction.from_polynomial(reduced_context.hamiltonian).bind(backend)
    full_basis = fixed_sector_basis_indices(3, (2, 2))
    core_mask = (1 << 0) | (1 << 3)  # alpha0 (qubit 0) and beta0 (qubit 3) occupied
    core_basis = np.asarray([index for index in full_basis if (int(index) & core_mask) == core_mask], dtype=np.int64)
    full_block, _ = sector_matrix(full_action, core_basis)
    reduced_block, leak = sector_matrix(reduced_action, fixed_sector_basis_indices(2, (1, 1)))
    assert leak == 0.0
    assert core_basis.size == reduced_block.shape[0] == 4
    np.testing.assert_allclose(sector_spectrum(full_block), sector_spectrum(reduced_block), atol=1.0e-10, rtol=0.0)


def test_editing_the_problem_document_fails_the_strict_intake(tmp_path):
    bridge_problem = qx.bridge_problem_from_qdynamics(_elements(synthetic_qdynamics_mapping("contract4")))
    manifest = qx.write_bridge_problem_json(bridge_problem, tmp_path / "problem.json")
    path = Path(manifest["problem_json_path"])
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        resolve_bridge_context(manifest["manifest_path"])


def _fcidump_text(h: np.ndarray, g: np.ndarray, ecore: float, nelec: int) -> str:
    n = h.shape[0]
    lines = [f" &FCI NORB={n},NELEC={nelec},MS2=0,", "  ORBSYM=" + ",".join(["1"] * n) + ",", "  ISYM=1,", " &END"]
    for p in range(n):
        for q in range(p + 1):
            pq = p * (p + 1) // 2 + q
            for r in range(n):
                for s in range(r + 1):
                    rs = r * (r + 1) // 2 + s
                    if rs <= pq and g[p, q, r, s] != 0.0:
                        lines.append(f" {g[p, q, r, s]: .16e} {p + 1} {q + 1} {r + 1} {s + 1}")
    for p in range(n):
        for q in range(p + 1):
            if h[p, q] != 0.0:
                lines.append(f" {h[p, q]: .16e} {p + 1} {q + 1} 0 0")
    lines.append(f" {ecore: .16e} 0 0 0 0")
    return "\n".join(lines) + "\n"


def test_fcidump_round_trip_and_intake(tmp_path):
    h, g, scalar, hf = contract_four_qubit_tensors()
    text = _fcidump_text(h, g, scalar, 2)
    data = qx.read_fcidump(text)
    assert data.norb == 2 and data.nelec == 2 and data.ms2 == 0
    assert np.array_equal(data.h, h) and np.array_equal(data.g, g) and data.ecore == scalar
    sidecar = qx.FcidumpSidecar(geometry_spec="synthetic mathematical fixture", basis="synthetic", e_scf=hf)
    bridge_problem = qx.bridge_problem_from_fcidump(data, sidecar)
    context, _ = _intake(bridge_problem, tmp_path)
    assert context.runtime_data["molecular_diagnostic_intake_receipt"]["checks"]["hf_witness_abs_defect"] == 0.0
    assert bridge_problem.provenance["fcidump_is_current_producer_output"] is False


def test_fcidump_rejects_wrong_declaration_and_complex_entries():
    h, g, scalar, hf = contract_four_qubit_tensors()
    data = qx.read_fcidump(_fcidump_text(h, g, scalar, 2))
    with pytest.raises(qx.QdynamicsConventionError) as excinfo:
        qx.bridge_problem_from_fcidump(data, qx.FcidumpSidecar(geometry_spec="s", basis="s", e_scf=hf, integral_notation="physicists_spin_orbital"))
    assert excinfo.value.code == "fcidump_notation"
    with pytest.raises(qx.QdynamicsConventionError) as excinfo:
        qx.bridge_problem_from_fcidump(data, qx.FcidumpSidecar(geometry_spec="s", basis="s", e_scf=hf + 1.0e-2))
    assert excinfo.value.code == "hf_witness"
    complex_text = _fcidump_text(h, g, scalar, 2).replace(" 1 1 1 1", " 1 1 1 1", 1)
    complex_text = "\n".join(line if "0 0 0 0" not in line else " (0.3, 0.1) 0 0 0 0" for line in complex_text.splitlines()) + "\n"
    with pytest.raises(qx.QdynamicsConventionError) as excinfo:
        qx.read_fcidump(complex_text)
    assert excinfo.value.code == "fcidump_complex"


def test_h5_loader_is_explicit_about_its_dependency(tmp_path):
    try:
        import h5py  # type: ignore
    except ModuleNotFoundError:
        (tmp_path / "x_ele.h5").write_bytes(b"not an hdf5 file")
        with pytest.raises(qx.BridgeDependencyUnavailable) as excinfo:
            qx.load_qdynamics_matrix_elements_h5(tmp_path / "x_ele.h5")
        assert excinfo.value.dependency == "h5py"
        return
    mapping = synthetic_qdynamics_mapping("contract4")
    path = tmp_path / "synthetic_ele.h5"
    with h5py.File(path, "w") as handle:  # mirrors write_matrix_elements_h5 at e5ec5bc4
        group = handle.create_group("metadata")
        for key, value in mapping["metadata"].items():
            group.attrs[key] = value if value is not None else ""
        handle.create_dataset("h1", data=np.asarray(mapping["h1"]), compression="gzip", chunks=True)
        handle.create_dataset("h2", data=np.asarray(mapping["h2"]), compression="gzip", chunks=True)
        handle.create_dataset("nuclear_repulsion", data=float(mapping["nuclear_repulsion"]))
        handle.create_dataset("mo_occ", data=np.asarray(mapping["mo_occ"]))
    elements = qx.load_qdynamics_matrix_elements_h5(path)
    assert elements.source["payload_kind"] == qx.QDYNAMICS_ELE_H5_SCHEMA
    assert len(elements.source["file_sha256"]) == 64
    bridge_problem = qx.bridge_problem_from_qdynamics(elements)
    assert np.array_equal(bridge_problem.problem.two_body_integrals_mo, contract_four_qubit_tensors()[1])
