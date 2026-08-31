"""Contract tests for the immutable SIAM prepared-state seed.

These tests exercise only the public seed/runtime-input seams.  Digest
expectations are recomputed here from the public model and pool objects rather
than by importing private serialization helpers from ``siam_seed``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import pipelines.time_dynamics.siam_seed as siam_seed_module
from pipelines.contracts.scaffold import ScaffoldRuntimeInput
from pipelines.time_dynamics.ap_mclachlan.state import (
    state_from_scaffold_runtime_input,
)
from pipelines.time_dynamics.siam_screening import SIAMScreeningConfig
from pipelines.time_dynamics.siam_runtime import (
    build_siam_runtime_input,
    pool_ansatz_terms,
)
from pipelines.time_dynamics.siam_seed import (
    SIAM_SEED_SCHEMA_V1,
    SIAM_SEED_SCHEMA_V2,
    SIAMSeedManifest,
    build_siam_seed_manifest,
    load_siam_runtime_input,
    load_siam_seed_manifest,
    write_siam_seed_manifest,
)
from src.quantum.siam_model import build_initial_state, build_siam_hamiltonian
from src.quantum.siam_pool import build_siam_pool


def _config(**overrides: Any) -> SIAMScreeningConfig:
    values: dict[str, Any] = {
        "n_bath": 2,
        "u": 1.375,
        "v_final": 0.625,
        "hopping": 0.875,
        "t_final": 5.0,
        "num_times": 51,
        "label": "siam_seed_contract",
    }
    values.update(overrides)
    return SIAMScreeningConfig(**values)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _canonical_float(value: float) -> str:
    number = float(value)
    if number == 0.0:
        number = 0.0
    return number.hex()


def _polynomial_payload(polynomial: Any) -> dict[str, Any]:
    terms = []
    for term in polynomial.return_polynomial():
        coefficient = complex(term.p_coeff)
        terms.append(
            {
                "word": str(term.pw2strng()),
                "coefficient_real_hex": _canonical_float(coefficient.real),
                "coefficient_imag_hex": _canonical_float(coefficient.imag),
            }
        )
    terms.sort(key=lambda row: row["word"])
    return {"n_qubits": int(polynomial.get_nq()), "terms": terms}


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _expected_digests(config: SIAMScreeningConfig) -> dict[str, str]:
    initial_state = build_initial_state(
        config.n_bath,
        hopping=config.hopping,
    )
    pre_quench = build_siam_hamiltonian(
        config.n_bath,
        u=config.u,
        v=0.0,
        hopping=config.hopping,
    )
    post_quench = build_siam_hamiltonian(
        config.n_bath,
        u=config.u,
        v=config.v_final,
        hopping=config.hopping,
    )
    ordered_pool = [
        {
            "label": str(block.label),
            "family": str(block.family),
            "operator": _polynomial_payload(block.operator),
        }
        for block in build_siam_pool(config.n_bath)
    ]
    return {
        "initial_state_semantic_sha256": (
            siam_seed_module.portable_statevector_sha256(initial_state)
        ),
        "pre_quench_hamiltonian_sha256": _json_sha256(
            _polynomial_payload(pre_quench)
        ),
        "post_quench_hamiltonian_sha256": _json_sha256(
            _polynomial_payload(post_quench)
        ),
        "ordered_logical_pool_sha256": _json_sha256(ordered_pool),
    }


def test_seed_manifest_identity_ignores_run_horizon_and_label() -> None:
    short = build_siam_seed_manifest(
        _config(t_final=1.0, num_times=3, label="short_diagnostic")
    )
    long = build_siam_seed_manifest(
        _config(t_final=20.0, num_times=2001, label="production_horizon")
    )

    assert short == long
    payload = short.to_json_dict()
    encoded = json.dumps(payload, sort_keys=True)
    for forbidden_key in ("label", "t_final", "num_times"):
        assert f'"{forbidden_key}"' not in encoded


def test_seed_manifest_round_trip_is_byte_deterministic(tmp_path: Path) -> None:
    first_path = tmp_path / "first" / "siam_seed.json"
    second_path = tmp_path / "second" / "siam_seed.json"

    written = write_siam_seed_manifest(first_path, _config())
    loaded = load_siam_seed_manifest(first_path)
    rewritten = write_siam_seed_manifest(second_path, loaded.physical_config())

    assert isinstance(loaded, SIAMSeedManifest)
    assert loaded.schema == SIAM_SEED_SCHEMA_V2
    assert loaded == written == rewritten
    assert first_path.read_bytes() == second_path.read_bytes()


def test_relocated_seed_accepts_phase_and_roundoff_equivalent_prepared_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source-locked seed must survive another NumPy/LAPACK realization."""

    authored = tmp_path / "authored" / "siam_seed.json"
    relocated = tmp_path / "package" / "input" / "seed.json"
    write_siam_seed_manifest(authored, _config())
    relocated.parent.mkdir(parents=True)
    relocated.write_bytes(authored.read_bytes())

    original = build_initial_state(2, hopping=0.875)
    equivalent = np.asarray(original * np.exp(0.731j), dtype=complex)
    occupied = np.flatnonzero(np.abs(equivalent) > 1.0e-10)
    equivalent[int(occupied[-1])] += 2.0e-15 - 1.0e-15j
    equivalent /= np.linalg.norm(equivalent)
    assert not np.array_equal(original, equivalent)
    assert abs(np.vdot(original, equivalent)) == pytest.approx(1.0, abs=1.0e-14)

    def _equivalent_initial_state(n_bath: int, *, hopping: float) -> np.ndarray:
        assert int(n_bath) == 2
        assert float(hopping) == pytest.approx(0.875)
        return equivalent.copy()

    monkeypatch.setattr(
        siam_seed_module,
        "build_initial_state",
        _equivalent_initial_state,
    )

    loaded = load_siam_runtime_input(relocated)

    assert loaded.provenance["artifact_json"] == str(relocated.resolve())
    assert loaded.provenance["seed_schema"] == "siam_screening_seed_v2"


def test_portable_state_identity_ignores_roundoff_but_detects_state_change() -> None:
    original = build_initial_state(2, hopping=0.875)
    equivalent = np.asarray(original * np.exp(-0.413j), dtype=complex)
    occupied = np.flatnonzero(np.abs(equivalent) > 1.0e-10)
    equivalent[int(occupied[-1])] += 1.0e-15 + 2.0e-15j
    equivalent /= np.linalg.norm(equivalent)

    zero_support = np.flatnonzero(np.abs(original) < 1.0e-15)
    orthogonal = np.zeros_like(original)
    orthogonal[int(zero_support[0])] = 1.0
    assert abs(np.vdot(original, orthogonal)) < 1.0e-15
    changed = original + 1.0e-8 * orthogonal
    changed /= np.linalg.norm(changed)

    identity = siam_seed_module.portable_statevector_sha256(original)
    assert siam_seed_module.portable_statevector_sha256(equivalent) == identity
    assert siam_seed_module.portable_statevector_sha256(changed) != identity


def test_seed_write_is_immutable_and_identical_bytes_are_a_noop(
    tmp_path: Path,
) -> None:
    path = tmp_path / "siam_seed.json"
    original = write_siam_seed_manifest(path, _config())
    original_bytes = path.read_bytes()

    repeated = write_siam_seed_manifest(path, _config())
    assert repeated == original
    assert path.read_bytes() == original_bytes

    with pytest.raises(FileExistsError, match="Refusing to replace immutable"):
        write_siam_seed_manifest(path, _config(u=2.0))
    assert path.read_bytes() == original_bytes


def test_seed_manifest_binds_preparation_hamiltonians_and_ordered_pool() -> None:
    config = _config()
    manifest = build_siam_seed_manifest(config)

    assert manifest.digests == _expected_digests(config)
    assert set(manifest.digests) == {
        "initial_state_semantic_sha256",
        "pre_quench_hamiltonian_sha256",
        "post_quench_hamiltonian_sha256",
        "ordered_logical_pool_sha256",
    }
    assert all(
        len(value) == 64 and set(value) <= set("0123456789abcdef")
        for value in manifest.digests.values()
    )
    assert manifest.pool["ordered_labels"] == [
        block.label for block in build_siam_pool(config.n_bath)
    ]
    assert manifest.preparation["state_identity"]["schema"] == (
        "phase_canonical_quantized_complex_v1"
    )


def test_schema_v1_seed_remains_strictly_loadable_on_its_authoring_stack(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy_v1_seed.json"
    migrated_path = tmp_path / "portable_v2_seed.json"
    written = write_siam_seed_manifest(
        path,
        _config(),
        schema=SIAM_SEED_SCHEMA_V1,
    )

    loaded = load_siam_seed_manifest(path)
    runtime = load_siam_runtime_input(path)
    migrated = write_siam_seed_manifest(
        migrated_path,
        loaded.physical_config(),
    )

    assert loaded == written
    assert loaded.schema == SIAM_SEED_SCHEMA_V1
    assert "initial_state_sha256" in loaded.digests
    assert "initial_state_semantic_sha256" not in loaded.digests
    assert runtime.provenance["seed_schema"] == SIAM_SEED_SCHEMA_V1
    assert migrated.schema == SIAM_SEED_SCHEMA_V2
    assert migrated.physical == loaded.physical
    assert migrated.layout == loaded.layout
    assert migrated.pool == loaded.pool
    for stable_digest in (
        "pre_quench_hamiltonian_sha256",
        "post_quench_hamiltonian_sha256",
        "ordered_logical_pool_sha256",
    ):
        assert migrated.digests[stable_digest] == loaded.digests[stable_digest]


def _tamper_physics(payload: dict[str, Any]) -> None:
    payload["physical"]["u"] = float(payload["physical"]["u"]) + 0.25


def _tamper_layout(payload: dict[str, Any]) -> None:
    payload["layout"]["n_qubits"] = int(payload["layout"]["n_qubits"]) + 1


def _tamper_preparation(payload: dict[str, Any]) -> None:
    payload["preparation"]["rng_seed"] = 17


def _tamper_pool(payload: dict[str, Any]) -> None:
    payload["pool"]["ordered_labels"] = list(
        reversed(payload["pool"]["ordered_labels"])
    )


def _tamper_digest(payload: dict[str, Any]) -> None:
    payload["digests"]["post_quench_hamiltonian_sha256"] = "0" * 64


def _tamper_state_digest(payload: dict[str, Any]) -> None:
    payload["digests"]["initial_state_semantic_sha256"] = "0" * 64


@pytest.mark.parametrize(
    "tamper",
    (
        _tamper_physics,
        _tamper_layout,
        _tamper_preparation,
        _tamper_pool,
        _tamper_digest,
        _tamper_state_digest,
    ),
    ids=(
        "physics",
        "layout",
        "preparation",
        "pool",
        "hamiltonian_digest",
        "state_digest",
    ),
)
def test_seed_loader_rejects_tampering(tmp_path: Path, tamper) -> None:
    path = tmp_path / "siam_seed.json"
    write_siam_seed_manifest(path, _config())
    payload = json.loads(path.read_text(encoding="utf-8"))
    tamper(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="failed deterministic regeneration"):
        load_siam_seed_manifest(path)


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    (
        ("n_bath", 2.0, TypeError, "n_bath must be an integer"),
        ("u", True, TypeError, "u must be a real number"),
        ("v_post_quench", "0.5", TypeError, "v_final must be a real number"),
        ("v_pre_quench", False, ValueError, "declared decoupled V_pre=0"),
        ("hopping", False, TypeError, "hopping must be a real number"),
        ("hopping", float("inf"), ValueError, "hopping must be finite"),
    ),
)
def test_seed_loader_rejects_raw_physical_type_and_finite_violations(
    tmp_path: Path,
    field: str,
    value,
    error: type[Exception],
    message: str,
) -> None:
    path = tmp_path / "siam_seed.json"
    write_siam_seed_manifest(path, _config())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["physical"][field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(error, match=message):
        load_siam_seed_manifest(path)


def test_seed_loads_empty_support_complete_logical_runtime_input(
    tmp_path: Path,
) -> None:
    path = tmp_path / "siam_seed.json"
    manifest = write_siam_seed_manifest(path, _config())

    runtime_input = load_siam_runtime_input(path)

    assert isinstance(runtime_input, ScaffoldRuntimeInput)
    assert runtime_input.resolved_problem.family_key == "siam"
    assert runtime_input.selected_terms == ()
    assert runtime_input.base_layout.logical_parameter_count == 0
    assert runtime_input.base_layout.runtime_parameter_count == 0
    assert np.asarray(runtime_input.theta_logical).shape == (0,)
    assert np.asarray(runtime_input.theta_runtime).shape == (0,)
    assert runtime_input.structure_locked is False
    assert runtime_input.exact_energy is None
    assert runtime_input.extensions["parameterization_mode"] == "logical_shared"
    assert runtime_input.extensions["seed_manifest_bound"] is True
    assert runtime_input.candidate_pool_source.source_kind == "resolved_pool"
    assert runtime_input.candidate_pool_source.completeness == "complete"
    assert runtime_input.can_structural_edit
    assert [term.label for term in runtime_input.candidate_pool_terms] == list(
        manifest.pool["ordered_labels"]
    )
    assert runtime_input.provenance["artifact_json"] == str(path.resolve())
    assert runtime_input.provenance["verified_seed_sha256"] == hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def test_logical_shared_state_loaded_from_seed_reproduces_prepared_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "siam_seed.json"
    write_siam_seed_manifest(path, _config())
    runtime_input = load_siam_runtime_input(path)

    state = state_from_scaffold_runtime_input(
        runtime_input,
        parameterization_mode="logical_shared",
    )
    expected = build_initial_state(2, hopping=0.875)

    assert state.parameterization_mode == "logical_shared"
    assert state.terms == ()
    assert state.runtime_parameter_count == 0
    assert state.can_structural_edit
    np.testing.assert_allclose(state.psi_ref, expected, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        state.psi_initial,
        expected,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        state.prepare_state(),
        expected,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_runtime_input_cannot_bind_a_seed_path_without_verified_manifest() -> None:
    with pytest.raises(ValueError, match="must be supplied together"):
        build_siam_runtime_input(_config(), artifact_json="unverified.json")


def test_manifest_backed_runtime_input_forbids_unbound_seed_support(
    tmp_path: Path,
) -> None:
    path = tmp_path / "siam_seed.json"
    manifest = write_siam_seed_manifest(path, _config())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="require empty seed_terms"):
        build_siam_runtime_input(
            _config(),
            artifact_json=str(path),
            seed_manifest=manifest,
            verified_seed_sha256=digest,
            seed_terms=(pool_ansatz_terms(2)[0],),
        )
