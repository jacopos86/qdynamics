"""Independent mathematical and snapshot witnesses for opt-in molecular intake."""

from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.contracts.problem import ProblemRequest
from pipelines.static_adapt.builders import molecular_diagnostic_intake as intake
from pipelines.static_adapt.builders import problem_registry as registry
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


def _payload() -> dict:
    eri = np.zeros((2, 2, 2, 2))
    for indices, value in [
        ((0, 0, 0, 0), 0.4), ((1, 1, 1, 1), 0.2),
        ((0, 0, 1, 1), 0.1), ((1, 1, 0, 0), 0.1),
        ((0, 1, 0, 1), 0.05), ((1, 0, 0, 1), 0.05),
        ((0, 1, 1, 0), 0.05), ((1, 0, 1, 0), 0.05),
    ]:
        eri[indices] = value
    return {
        "geometry_spec": "synthetic mathematical fixture",
        "basis": "synthetic orthonormal spatial basis",
        "charge": 0, "multiplicity": 1, "reference": "rhf",
        "n_spatial_orbitals": 2, "n_spin_orbitals": 4,
        "n_alpha": 1, "n_beta": 1,
        "hf_energy": -1.3, "nuclear_repulsion_energy": 0.3,
        "one_body_integrals_mo": [[-1.0, 0.0], [0.0, 0.5]],
        "two_body_integrals_mo": eri.tolist(),
    }


def _declaration(**changes):
    declaration = intake.MolecularDiagnosticDeclaration(
        energy_units="hartree", eri_convention="real_spatial_chemists",
        orbital_basis="orthonormal_spatial_mo", occupied_order="occupied_prefix",
        scalar_semantics="nuclear_only", nuclear_scalar_hartree=0.3,
        frozen_core_scalar_hartree=0.0, one_body_core_policy="none",
    )
    return replace(declaration, **changes)


def _request(path: Path, **changes) -> ProblemRequest:
    request = ProblemRequest(
        problem_key="molecular_restricted_closed_shell", num_sites=2,
        t=0.0, u=0.0, dv=0.0, omega0=0.0, g_ep=0.0,
        n_ph_max=0, boson_encoding="binary", ordering="blocked",
        boundary="open", molecular_problem_json=str(path),
    )
    return replace(request, **changes)


def _write(path: Path, payload: dict, **json_options) -> bytes:
    raw = json.dumps(payload, **json_options).encode("utf-8")
    path.write_bytes(raw)
    return raw


def _resolve(path: Path, raw: bytes, *, declaration=None, request=None, **kwargs):
    return intake.resolve_molecular_diagnostic_intake(
        _request(path) if request is None else request,
        expected_raw_sha256=hashlib.sha256(raw).hexdigest(),
        declaration=_declaration() if declaration is None else declaration,
        **kwargs,
    )


def _receipt(context):
    return context.runtime_data["molecular_diagnostic_intake_receipt"]


def _dense_pauli_sum(polynomial) -> np.ndarray:
    # Independent Pauli matrices; no production dense-builder implementation.
    matrices = {
        "e": np.eye(2), "x": np.array([[0, 1], [1, 0]]),
        "y": np.array([[0, -1j], [1j, 0]]), "z": np.diag([1, -1]),
    }
    dense = np.zeros((16, 16), dtype=complex)
    for term in polynomial.return_polynomial():
        word = str(term.pw2strng())
        assert len(word) == 4
        matrix = np.ones((1, 1), dtype=complex)
        for letter in word:
            matrix = np.kron(matrix, matrices[letter])
        dense += complex(term.p_coeff) * matrix
    return dense


def test_analytic_sector_block(tmp_path, monkeypatch):
    def no_exact_solve(*args, **kwargs):
        raise AssertionError("intake must leave the exact reference lazy")

    monkeypatch.setattr(registry, "_exact_gs_energy_for_problem", no_exact_solve)
    path = tmp_path / "synthetic.json"
    raw = _write(path, _payload())
    context = _resolve(path, raw)
    assert context.family_key == "molecular_restricted_closed_shell"
    assert context.layout.total_qubits == 4
    assert context.default_num_particles == (1, 1)
    assert context.runtime_data["molecular_problem_json"] == str(path)
    dense = _dense_pauli_sum(context.hamiltonian)
    indices = [5, 9, 6, 10]
    block = dense[np.ix_(indices, indices)]
    analytic = np.array([
        [-1.3, 0, 0, 0.05], [0, -0.1, 0.05, 0],
        [0, 0.05, -0.1, 0], [0.05, 0, 0, 1.5],
    ])
    np.testing.assert_allclose(block, analytic, atol=1e-12, rtol=0)
    roots = [0.1 - np.sqrt(785) / 20, -0.15, -0.05,
             0.1 + np.sqrt(785) / 20]
    np.testing.assert_allclose(np.linalg.eigvalsh(block), roots, atol=1e-12, rtol=0)
    outside = [index for index in range(16) if index not in indices]
    np.testing.assert_allclose(dense[np.ix_(outside, indices)], 0, atol=1e-12, rtol=0)
    receipt = _receipt(context)
    assert receipt["schema"] == "molecular_diagnostic_intake_v1"
    assert receipt["raw_sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["expected_raw_sha256"] == receipt["raw_sha256"]
    assert receipt["checks"]["input_tensor_determinant_energy"] == pytest.approx(-1.3)
    assert receipt["checks"]["hf_witness_abs_defect"] < 1e-12
    assert receipt["builder"]["fixed_pruning_tolerance"] == 1e-12


def test_legacy_h2_fixture_rejected(tmp_path, monkeypatch):
    source = (REPO_ROOT / "test/test_static_adapt_builders.py").read_text()
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "_write_molecular_problem_json")
    assignment = next(node for node in function.body
                      if isinstance(node, ast.AnnAssign)
                      and isinstance(node.target, ast.Name)
                      and node.target.id == "payload")
    payload = ast.literal_eval(assignment.value)
    path = tmp_path / "legacy.json"
    raw = _write(path, payload)
    monkeypatch.setattr(registry, "build_restricted_closed_shell_molecular_hamiltonian",
                        lambda *a, **k: pytest.fail("invalid fixture reached JW"))
    with pytest.raises(ValueError):
        _resolve(path, raw, declaration=_declaration(
            nuclear_scalar_hartree=payload["nuclear_repulsion_energy"]))


@pytest.mark.parametrize("field,value", [
    ("nuclear_repulsion_energy", None), ("hf_energy", None),
    ("hf_energy", -1.2), ("hf_energy", True), ("hf_energy", "-1.3"),
    ("nuclear_repulsion_energy", 0.4), ("nuclear_repulsion_energy", float("nan")),
    ("hf_energy", float("inf")), ("n_spatial_orbitals", True),
    ("n_spatial_orbitals", 2.5), ("n_spatial_orbitals", "2"),
    ("n_spatial_orbitals", 0), ("n_spatial_orbitals", 7),
    ("n_spin_orbitals", 5), ("n_spin_orbitals", 4.5),
    ("n_alpha", True), ("n_alpha", 1.5), ("n_alpha", "1"),
    ("n_alpha", -1), ("n_alpha", 3), ("n_beta", 0),
    ("multiplicity", 3), ("multiplicity", True), ("charge", 0.5),
    ("reference", "uhf"),
    ("one_body_integrals_mo", [[-1, 0.1], [0, 0.5]]),
    ("one_body_integrals_mo", [[-1, 0], [0, float("nan")]]),
    ("one_body_integrals_mo", [[-1, 0], [0, "0.5"]]),
    ("one_body_integrals_mo", [[-1, 0], [0, "1+2j"]]),
    ("one_body_integrals_mo", [[-1, 0], [0, 2**53 + 1]]),
    ("one_body_integrals_mo", [[-1]]),
    ("two_body_integrals_mo", [[[[0]]]]),
])
def test_invalid_payload_fails_before_jw(tmp_path, monkeypatch, field, value):
    payload = _payload()
    if value is None:
        del payload[field]
    else:
        payload[field] = value
    path = tmp_path / "invalid.json"
    raw = _write(path, payload)
    monkeypatch.setattr(registry, "build_restricted_closed_shell_molecular_hamiltonian",
                        lambda *a, **k: pytest.fail("invalid payload reached JW"))
    with pytest.raises(ValueError):
        _resolve(path, raw)


@pytest.mark.parametrize("location", [(0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 0)])
def test_broken_eri_symmetry_fails_before_jw(tmp_path, monkeypatch, location):
    payload = _payload()
    eri = np.array(payload["two_body_integrals_mo"])
    eri[location] += 0.03
    payload["two_body_integrals_mo"] = eri.tolist()
    path = tmp_path / "invalid_eri.json"
    raw = _write(path, payload)
    monkeypatch.setattr(registry, "build_restricted_closed_shell_molecular_hamiltonian",
                        lambda *a, **k: pytest.fail("broken ERI reached JW"))
    with pytest.raises(ValueError):
        _resolve(path, raw)


@pytest.mark.parametrize("changes", [
    {"energy_units": "eV"}, {"eri_convention": "antisymmetrized"},
    {"orbital_basis": "ao"}, {"occupied_order": "explicit"},
    {"scalar_semantics": "unknown"}, {"nuclear_scalar_hartree": 0.4},
    {"frozen_core_scalar_hartree": 0.1}, {"one_body_core_policy": "contract"},
    {"nuclear_scalar_hartree": float("nan")},
])
def test_invalid_declaration_fails_before_jw(tmp_path, monkeypatch, changes):
    path = tmp_path / "declaration.json"
    raw = _write(path, _payload())
    monkeypatch.setattr(registry, "build_restricted_closed_shell_molecular_hamiltonian",
                        lambda *a, **k: pytest.fail("invalid declaration reached JW"))
    with pytest.raises(ValueError):
        _resolve(path, raw, declaration=_declaration(**changes))


@pytest.mark.parametrize("changes", [
    {"problem_key": "hh"}, {"ordering": "interleaved"}, {"num_sites": 3},
])
def test_invalid_request_fails_before_jw(tmp_path, monkeypatch, changes):
    path = tmp_path / "request.json"
    raw = _write(path, _payload())
    monkeypatch.setattr(registry, "build_restricted_closed_shell_molecular_hamiltonian",
                        lambda *a, **k: pytest.fail("invalid request reached JW"))
    with pytest.raises(ValueError):
        _resolve(path, raw, request=_request(path, **changes))


@pytest.mark.parametrize("mode", ["duplicate", "nested_duplicate", "ambiguous", "malformed_hash", "wrong_hash"])
def test_snapshot_syntax_and_identity_fail_before_jw(tmp_path, monkeypatch, mode):
    payload = _payload()
    raw = json.dumps(payload).encode()
    if mode == "duplicate":
        raw = b'{"hf_energy":-1.3,' + raw[1:]
    elif mode == "nested_duplicate":
        raw = b'{"problem":{"hf_energy":-1.3,' + raw[1:] + b'}'
    elif mode == "ambiguous":
        raw = json.dumps({"problem": payload, "chemistry": payload}).encode()
    path = tmp_path / "syntax.json"
    path.write_bytes(raw)
    expected = hashlib.sha256(raw).hexdigest()
    if mode == "malformed_hash":
        expected = "not-a-sha256"
    elif mode == "wrong_hash":
        expected = "0" * 64
    monkeypatch.setattr(registry, "build_restricted_closed_shell_molecular_hamiltonian",
                        lambda *a, **k: pytest.fail("invalid snapshot reached JW"))
    with pytest.raises(ValueError):
        intake.resolve_molecular_diagnostic_intake(
            _request(path), expected_raw_sha256=expected, declaration=_declaration())


@pytest.mark.parametrize("atol", [-1, float("nan"), float("inf")])
def test_invalid_tolerance_fails_before_jw(tmp_path, monkeypatch, atol):
    path = tmp_path / "tolerance.json"
    raw = _write(path, _payload())
    monkeypatch.setattr(registry, "build_restricted_closed_shell_molecular_hamiltonian",
                        lambda *a, **k: pytest.fail("invalid tolerance reached JW"))
    with pytest.raises(ValueError):
        _resolve(path, raw, atol=atol)


@pytest.mark.parametrize("wrapper,location", [
    ("root", "$"), ("problem", "$.problem"),
    ("chemistry", "$.chemistry"), ("chemistry.problem", "$.chemistry.problem"),
])
def test_supported_payload_locations(tmp_path, wrapper, location):
    payload = _payload()
    if wrapper == "problem": payload = {"problem": payload}
    elif wrapper == "chemistry": payload = {"chemistry": payload}
    elif wrapper == "chemistry.problem": payload = {"chemistry": {"problem": payload}}
    path = tmp_path / "wrapped.json"
    raw = _write(path, payload)
    assert _receipt(_resolve(path, raw))["payload_location"] == location


def test_snapshot_survives_path_change(tmp_path, monkeypatch):
    path = tmp_path / "snapshot.json"
    raw = _write(path, _payload())
    original_read = Path.read_bytes
    reads = []

    def replace_after_read(self):
        snapshot = original_read(self)
        if self == path:
            reads.append(self)
            self.write_bytes(b'{"replacement":true}')
        return snapshot

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    context = _resolve(path, raw)
    assert reads == [path]
    assert _receipt(context)["raw_sha256"] == hashlib.sha256(raw).hexdigest()
    assert context.runtime_data["molecular_problem"].hf_energy == pytest.approx(-1.3)
    assert original_read(path) == b'{"replacement":true}'


def test_raw_and_tensor_hashes_have_distinct_semantics(tmp_path):
    path = tmp_path / "format.json"
    first = _receipt(_resolve(path, _write(path, _payload(), separators=(",", ":"))))
    second = _receipt(_resolve(path, _write(path, _payload(), indent=2)))
    assert first["raw_sha256"] != second["raw_sha256"]
    for key in ("tensor_sha256", "semantic_sha256", "operator_sha256"):
        assert first[key] == second[key]
    changed = _payload()
    changed["one_body_integrals_mo"][1][1] = 0.6
    third = _receipt(_resolve(path, _write(path, changed)))
    for key in ("tensor_sha256", "semantic_sha256", "operator_sha256"):
        assert third[key] != first[key]


def test_signed_zero_has_canonical_content_binding(tmp_path):
    path = tmp_path / "zeros.json"
    first = _receipt(_resolve(path, _write(path, _payload())))
    payload = _payload()
    payload["one_body_integrals_mo"][0][1] = -0.0
    second = _receipt(_resolve(path, _write(path, payload)))
    assert first["raw_sha256"] != second["raw_sha256"]
    assert first["tensor_sha256"] == second["tensor_sha256"]
    assert first["operator_sha256"] == second["operator_sha256"]
    assert second["encoding"]["signed_zero"] == "canonical_positive_zero"


def test_operator_hash_binds_actual_builder_output(tmp_path, monkeypatch):
    path = tmp_path / "operator.json"
    raw = _write(path, _payload())
    first = _receipt(_resolve(path, raw))
    original = registry.build_restricted_closed_shell_molecular_hamiltonian

    def shifted_builder(*args, **kwargs):
        polynomial = original(*args, **kwargs)
        polynomial.add_term(PauliTerm(4, ps="eeee", pc=0.125))
        return polynomial

    monkeypatch.setattr(registry, "build_restricted_closed_shell_molecular_hamiltonian", shifted_builder)
    second = _receipt(_resolve(path, raw))
    assert first["raw_sha256"] == second["raw_sha256"]
    assert first["tensor_sha256"] == second["tensor_sha256"]
    assert first["operator_sha256"] != second["operator_sha256"]


def test_operator_hash_binds_actual_returned_term_order(tmp_path, monkeypatch):
    path = tmp_path / "operator_order.json"
    raw = _write(path, _payload())
    first_context = _resolve(path, raw)
    original = registry.build_restricted_closed_shell_molecular_hamiltonian

    def reversed_builder(*args, **kwargs):
        polynomial = original(*args, **kwargs)
        assert len(polynomial.return_polynomial()) > 1
        polynomial.return_polynomial().reverse()
        return polynomial

    monkeypatch.setattr(registry, "build_restricted_closed_shell_molecular_hamiltonian", reversed_builder)
    second_context = _resolve(path, raw)
    np.testing.assert_allclose(_dense_pauli_sum(first_context.hamiltonian),
                               _dense_pauli_sum(second_context.hamiltonian), atol=1e-12, rtol=0)
    assert _receipt(first_context)["tensor_sha256"] == _receipt(second_context)["tensor_sha256"]
    assert _receipt(first_context)["operator_sha256"] != _receipt(second_context)["operator_sha256"]
    assert _receipt(second_context)["encoding"]["operator_order"] == "actual_return_polynomial_order"


def test_contracted_scalar_is_declared_once(tmp_path):
    path = tmp_path / "core.json"
    raw = _write(path, _payload())
    context = _resolve(path, raw, declaration=_declaration(
        scalar_semantics="nuclear_plus_frozen_core", nuclear_scalar_hartree=0.2,
        frozen_core_scalar_hartree=0.1, one_body_core_policy="already_contracted"))
    assert _dense_pauli_sum(context.hamiltonian)[5, 5] == pytest.approx(-1.3)


def test_diagnostic_tolerance_does_not_change_fixed_builder_pruning(tmp_path):
    payload = _payload()
    tiny_integral = 7.5e-13
    payload["two_body_integrals_mo"][0][0][0][0] = tiny_integral
    payload["hf_energy"] = -1.7 + tiny_integral
    path = tmp_path / "near_pruning.json"
    context = _resolve(path, _write(path, payload), atol=1e-14)
    receipt = _receipt(context)
    assert receipt["diagnostic_atol"] == 1e-14
    assert receipt["builder"]["fixed_pruning_tolerance"] == 1e-12
    assert receipt["checks"]["hf_witness_abs_defect"] <= 1e-14
    # The input determinant witness includes this ERI; the fixed mapper prunes it.
    mapped_hf = _dense_pauli_sum(context.hamiltonian)[5, 5].real
    assert mapped_hf == pytest.approx(-1.7, abs=1e-14)
    assert abs(mapped_hf - payload["hf_energy"]) > receipt["diagnostic_atol"]


@pytest.mark.parametrize("scalar", [0.0, 0.5e-12])
def test_zero_and_near_pruned_scalar_follow_input_tensor_witness(tmp_path, scalar):
    payload = _payload()
    payload["nuclear_repulsion_energy"] = scalar
    payload["hf_energy"] = -1.6 + scalar
    path = tmp_path / "small_scalar.json"
    context = _resolve(path, _write(path, payload),
                       declaration=_declaration(nuclear_scalar_hartree=scalar), atol=1e-15)
    receipt = _receipt(context)
    assert receipt["checks"]["input_tensor_determinant_energy"] == pytest.approx(payload["hf_energy"], abs=1e-15)
    assert receipt["checks"]["hf_witness_abs_defect"] <= 1e-15
    assert receipt["builder"]["fixed_pruning_tolerance"] == 1e-12
    mapped_hf = _dense_pauli_sum(context.hamiltonian)[5, 5].real
    assert mapped_hf == pytest.approx(-1.6, abs=1e-15)
    if scalar:
        assert abs(mapped_hf - payload["hf_energy"]) > receipt["diagnostic_atol"]


@pytest.mark.parametrize("mode", ["width", "fractional_width", "nonfinite", "imaginary", "empty", "representation"])
def test_actual_mapped_operator_is_validated(tmp_path, monkeypatch, mode):
    path = tmp_path / "mapped.json"
    raw = _write(path, _payload())

    def invalid_builder(*args, **kwargs):
        polynomial = PauliPolynomial("PW" if mode == "representation" else "JW")
        if mode != "empty":
            width = 3 if mode == "width" else 4
            coefficient = float("inf") if mode == "nonfinite" else 1j if mode == "imaginary" else 1
            term = PauliTerm(width, ps="e" * width, pc=coefficient)
            if mode == "fractional_width":
                term.nqubit = lambda: 4.5
            polynomial.add_term(term)
        return polynomial

    monkeypatch.setattr(registry, "build_restricted_closed_shell_molecular_hamiltonian", invalid_builder)
    with pytest.raises(ValueError):
        _resolve(path, raw)


def test_ordinary_ordering_error_precedes_path_read(tmp_path):
    with pytest.raises(ValueError, match="ordering='blocked'"):
        registry.resolve_problem_context(_request(tmp_path / "absent.json", ordering="interleaved"))
    with pytest.raises(FileNotFoundError):
        registry.resolve_problem_context(_request(tmp_path / "absent.json"))


def test_ordinary_override_and_exact_closure_remain_lazy(tmp_path, monkeypatch):
    path = tmp_path / "ordinary.json"
    _write(path, _payload())
    override = PauliPolynomial("JW")
    override.add_term(PauliTerm(4, ps="eeee", pc=0.7))
    calls = []

    def exact_callback(hamiltonian, **kwargs):
        calls.append((hamiltonian, kwargs))
        return 12.5

    monkeypatch.setattr(registry, "build_restricted_closed_shell_molecular_hamiltonian",
                        lambda *a, **k: pytest.fail("override unexpectedly rebuilt"))
    context = registry.resolve_problem_context(
        _request(path), hamiltonian=override, exact_energy_impl=exact_callback)
    assert context.hamiltonian is override
    assert calls == []
    assert context.exact_target.resolve_energy(ai_log=None) == 12.5
    assert calls[0][0] is override
    assert calls[0][1]["num_particles"] == (1, 1)
    assert calls[0][1]["problem"] == "molecular_restricted_closed_shell"
    with pytest.raises(ValueError, match="request.num_sites"):
        registry.resolve_problem_context(_request(path, num_sites=3), hamiltonian=override)
