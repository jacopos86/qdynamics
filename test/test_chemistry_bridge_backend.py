"""Backend and matrix-free action witnesses for the chemistry bridge.

The action is checked against the repository kernel, the repository dense
matrix, and an independent Kronecker-product construction written here; the
sector bookkeeping against the repository sector solver; the status logic
against the rule that a CPU-exercised device is never reported as GPU.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.chemistry_bridge import array_backend as ab
from pipelines.chemistry_bridge import polynomial_action as pa
from pipelines.chemistry_bridge.methods import (
    STATUS_CPU_VERIFIED,
    STATUS_FAILED,
    STATUS_GPU_VERIFIED,
    STATUS_SKIPPED_UNSUPPORTED,
    verified_status,
)
from pipelines.chemistry_bridge.qdynamics_export import bridge_problem_from_qdynamics, qdynamics_matrix_elements_from_mapping, write_bridge_problem_json, declaration_from_mapping
from pipelines.chemistry_bridge.synthetic_witnesses import CONTRACT_FOUR_QUBIT_ROOTS, synthetic_qdynamics_mapping
from pipelines.contracts.problem import ProblemRequest
from pipelines.static_adapt.builders.molecular_diagnostic_intake import resolve_molecular_diagnostic_intake
from src.quantum.compiled_polynomial import apply_compiled_polynomial, compile_polynomial_action
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import _sector_basis_indices, _spin_orbital_index_sets, exact_ground_energy_sector, hamiltonian_matrix


_PAULI = {
    "e": np.eye(2, dtype=complex),
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _independent_matrix(poly: PauliPolynomial) -> np.ndarray:
    """Kronecker construction written here, leftmost letter = highest qubit."""

    terms = poly.return_polynomial()
    nq = int(terms[0].nqubit())
    matrix = np.zeros((1 << nq, 1 << nq), dtype=complex)
    for term in terms:
        word = np.array([[1.0 + 0.0j]])
        for letter in str(term.pw2strng()):
            word = np.kron(word, _PAULI[letter])
        matrix += complex(term.p_coeff) * word
    return matrix


def _random_polynomial(rng: np.random.Generator, nq: int, count: int) -> PauliPolynomial:
    poly = PauliPolynomial("JW")
    for _ in range(count):
        word = "".join(rng.choice(list("exyz"), size=nq))
        poly.add_term(PauliTerm(nq, ps=word, pc=complex(rng.standard_normal(), rng.standard_normal())))
    return poly


@pytest.mark.parametrize("spec", ["numpy:float64", "numpy:float32"])
def test_action_matches_repository_kernel_dense_and_independent_kron(spec):
    rng = np.random.default_rng(2026)
    backend = ab.resolve_backend(spec)
    tol = ab.parity_tolerances(backend.report().real_dtype)["vector_abs"]
    for _ in range(12):
        nq = int(rng.integers(1, 7))
        poly = _random_polynomial(rng, nq, int(rng.integers(1, 10)))
        psi = rng.standard_normal(1 << nq) + 1j * rng.standard_normal(1 << nq)
        out = pa.BridgePolynomialAction.from_polynomial(poly).bind(backend).apply_numpy(psi)
        kernel = apply_compiled_polynomial(psi, compile_polynomial_action(poly, tol=1.0e-15))
        dense = hamiltonian_matrix(poly, tol=0.0) @ psi
        independent = _independent_matrix(poly) @ psi
        scale = max(1.0, float(np.max(np.abs(kernel))))
        assert np.max(np.abs(out - kernel)) <= tol * scale
        assert np.max(np.abs(out - dense)) <= tol * scale
        assert np.max(np.abs(out - independent)) <= tol * scale


def test_action_identity_and_expectation():
    poly = PauliPolynomial("JW")
    poly.add_term(PauliTerm(3, ps="eee", pc=0.75))
    poly.add_term(PauliTerm(3, ps="zez", pc=-0.25))
    action = pa.BridgePolynomialAction.from_polynomial(poly)
    assert action.identity_coefficient == 0.75
    bound = action.bind(ab.NumpyBackend())
    psi = np.zeros(8, dtype=complex)
    psi[0b101] = 1.0
    assert abs(bound.expectation_numpy(psi) - (0.75 - 0.25)) < 1.0e-14


def test_sector_indices_and_spectrum_match_repository():
    rng = np.random.default_rng(5)
    n_spatial, particles = 3, (2, 1)
    alpha, beta = _spin_orbital_index_sets(n_spatial, "blocked")
    expected = np.asarray(_sector_basis_indices(2 * n_spatial, alpha, beta, *particles), dtype=np.int64)
    assert np.array_equal(pa.fixed_sector_basis_indices(n_spatial, particles), expected)
    # number-conserving random Hamiltonian: diagonal Z words plus XX+YY hopping pairs
    poly = PauliPolynomial("JW")
    for q in range(6):
        poly.add_term(PauliTerm(6, ps="".join("z" if k == 5 - q else "e" for k in range(6)), pc=float(rng.standard_normal())))
    for (a, b) in [(0, 1), (1, 2), (3, 4), (4, 5)]:
        c = float(rng.standard_normal())
        for letters in ("xx", "yy"):
            word = ["e"] * 6
            word[5 - a] = letters[0]
            word[5 - b] = letters[1]
            poly.add_term(PauliTerm(6, ps="".join(word), pc=0.5 * c))
    bound = pa.BridgePolynomialAction.from_polynomial(poly).bind(ab.NumpyBackend())
    block, leakage = pa.sector_matrix(bound, expected)
    assert leakage == 0.0
    ground = pa.sector_spectrum(block)[0]
    reference = exact_ground_energy_sector(poly, num_sites=n_spatial, num_particles=particles, indexing="blocked")
    assert abs(ground - reference) < 1.0e-12


def _contract_context(tmp_path):
    mapping = synthetic_qdynamics_mapping("contract4")
    bridge_problem = bridge_problem_from_qdynamics(qdynamics_matrix_elements_from_mapping(mapping, source={"payload_kind": "synthetic_witness"}))
    manifest = write_bridge_problem_json(bridge_problem, tmp_path / "problem.json")
    request = ProblemRequest(problem_key="molecular_restricted_closed_shell", num_sites=2, t=0.0, u=0.0, dv=0.0, omega0=0.0, g_ep=0.0, n_ph_max=0, boson_encoding="binary", ordering="blocked", boundary="open", molecular_problem_json=manifest["problem_json_path"])
    return resolve_molecular_diagnostic_intake(request, expected_raw_sha256=manifest["raw_sha256"], declaration=declaration_from_mapping(manifest["declaration"]))


def test_contract_fixture_reproduces_analytic_sector_roots(tmp_path):
    context = _contract_context(tmp_path)
    bound = pa.BridgePolynomialAction.from_polynomial(context.hamiltonian).bind(ab.NumpyBackend())
    block, leakage = pa.sector_matrix(bound, pa.fixed_sector_basis_indices(2, (1, 1)))
    assert leakage == 0.0
    np.testing.assert_allclose(pa.sector_spectrum(block), np.sort(np.asarray(CONTRACT_FOUR_QUBIT_ROOTS)), atol=1.0e-12, rtol=0.0)


def test_out_of_sector_norm():
    psi = np.zeros(16, dtype=complex)
    psi[5] = 0.6
    psi[0] = 0.8
    assert abs(pa.out_of_sector_norm(psi, np.array([5, 6, 9, 10])) - 0.8) < 1.0e-15


def test_parse_backend_spec():
    assert ab.parse_backend_spec("numpy") == ("numpy", "cpu", "float64")
    assert ab.parse_backend_spec("numpy:float32") == ("numpy", "cpu", "float32")
    assert ab.parse_backend_spec("torch:mps") == ("torch", "mps", "float64")
    assert ab.parse_backend_spec("torch:cuda:float32") == ("torch", "cuda", "float32")
    for bad in ("", "cupy", "torch", "torch:tpu", "numpy:mps", "torch:cpu:int8", "numpy:float64:extra"):
        with pytest.raises(ValueError):
            ab.parse_backend_spec(bad)


def test_availability_report_names_the_obstacle():
    report = ab.backend_availability("torch:cuda:float64")
    assert report.requested_device == "cuda"
    if report.available:
        assert report.exercised_device == "cuda" and report.is_gpu
    else:
        assert report.exercised_device is None and not report.is_gpu and report.reason
    malformed = ab.backend_availability("not-a-backend")
    assert not malformed.available and "malformed" in str(malformed.reason)


def test_status_never_counts_a_cpu_exercise_as_gpu():
    def report(requested, exercised, available=True):
        return ab.BackendReport(spec="x", library="torch", library_version=None, requested_device=requested, exercised_device=exercised, real_dtype="float32", complex_dtype="complex64", is_gpu=exercised in {"cuda", "mps"}, available=available, reason=None)

    assert verified_status(report("mps", "mps"), checks_passed=True) == STATUS_GPU_VERIFIED
    assert verified_status(report("cuda", "cuda"), checks_passed=True) == STATUS_GPU_VERIFIED
    assert verified_status(report("cpu", "cpu"), checks_passed=True) == STATUS_CPU_VERIFIED
    assert verified_status(report("mps", "cpu"), checks_passed=True) == STATUS_FAILED
    assert verified_status(report("mps", None, available=False), checks_passed=True) == STATUS_SKIPPED_UNSUPPORTED
    assert verified_status(report("mps", "mps"), checks_passed=False) == STATUS_FAILED


def test_torch_backend_refuses_silent_fallback_or_reports_missing_torch():
    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        with pytest.raises(ab.BackendUnavailable, match="torch is not importable"):
            ab.resolve_backend("torch:cpu:float64")
        return
    if torch.cuda.is_available():
        pytest.skip("cuda is available here; the no-fallback probe needs an absent device")
    with pytest.raises(ab.BackendUnavailable):
        ab.resolve_backend("torch:cuda:float64")
