from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.scaffold.handoff_state_bundle import build_statevector_manifest
from pipelines.scaffold.runtime_contract import ScaffoldRuntimeInput
from pipelines.scaffold import runtime_loader
from pipelines.scaffold.hh_vqe_from_adapt_family import (
    _replay_terms_from_serialized_selected_pauli_terms,
)
from pipelines.static_adapt.resume_scaffold import (
    ResumeScaffoldSource,
    match_resume_scaffold_to_pool,
)
from pipelines.static_adapt.selector_exact_query_geometry import (
    candidate_generator_fingerprint,
)
from src.quantum.ansatz_parameterization import (
    build_parameter_layout,
    deserialize_layout,
    expand_legacy_logical_theta,
    serialize_layout,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.vqe_latex_python_pairs import (
    AnsatzTerm,
    PauliPolynomial,
    PauliTerm,
)


def test_resume_checkpoint_accepts_exact_logical_shared_alias() -> None:
    polynomial = PauliPolynomial(
        "JW",
        [
            PauliTerm(1, ps="x", pc=1.0),
            PauliTerm(1, ps="z", pc=1.0),
        ],
    )
    term = AnsatzTerm(label="logical_shared_two_factor", polynomial=polynomial)
    layout = build_parameter_layout(
        [term],
        ignore_identity=True,
        coefficient_tolerance=1.0e-12,
        sort_terms=True,
    )
    theta_logical = np.asarray([0.2], dtype=float)
    theta_runtime = np.asarray([0.6, -0.2], dtype=float)
    theta_alias = expand_legacy_logical_theta(theta_logical, layout)
    psi_ref = np.asarray([1.0 + 0.0j, 0.0 + 0.0j])
    executor = CompiledAnsatzExecutor(
        [term],
        coefficient_tolerance=float(layout.coefficient_tolerance),
        ignore_identity=bool(layout.ignore_identity),
        sort_terms=True,
        parameterization_mode="per_pauli_term",
        parameterization_layout=layout,
    )
    psi_initial = executor.prepare_state(theta_alias, psi_ref)
    payload = {
        "initial_state": build_statevector_manifest(
            psi_state=psi_initial,
            source="checkpoint",
            handoff_state_kind="prepared_state",
        )
    }
    replay_context = SimpleNamespace(
        base_layout=layout,
        replay_terms=(term,),
        adapt_theta_runtime=theta_runtime,
        adapt_theta_logical=theta_logical,
        psi_ref=psi_ref,
    )

    loaded_state, source, error = runtime_loader._resolve_prepared_state(
        payload,
        replay_context,
    )

    assert source == "payload_logical_shared_alias"
    assert error < 1.0e-12
    assert np.allclose(loaded_state, psi_initial)


def test_resume_checkpoint_logical_alias_preserves_grouped_exact_execution() -> None:
    polynomial = PauliPolynomial(
        "JW",
        [
            PauliTerm(1, ps="x", pc=1.0),
            PauliTerm(1, ps="z", pc=0.5),
        ],
    )
    term = AnsatzTerm(
        label="logical_shared_grouped_exact",
        polynomial=polynomial,
        execution_mode="grouped_exact",
    )
    layout = build_parameter_layout(
        [term],
        ignore_identity=True,
        coefficient_tolerance=1.0e-12,
        sort_terms=True,
    )
    theta_logical = np.asarray([0.2], dtype=float)
    theta_runtime = expand_legacy_logical_theta(theta_logical, layout)
    psi_ref = np.asarray([1.0 + 0.0j, 0.0 + 0.0j])
    psi_initial = CompiledAnsatzExecutor(
        [term],
        coefficient_tolerance=float(layout.coefficient_tolerance),
        ignore_identity=bool(layout.ignore_identity),
        sort_terms=True,
        parameterization_mode="logical_shared",
        parameterization_layout=layout,
    ).prepare_state(theta_logical, psi_ref)
    payload = {
        "initial_state": build_statevector_manifest(
            psi_state=psi_initial,
            source="checkpoint",
            handoff_state_kind="prepared_state",
        )
    }
    replay_context = SimpleNamespace(
        base_layout=layout,
        replay_terms=(term,),
        adapt_theta_runtime=theta_runtime,
        adapt_theta_logical=theta_logical,
        psi_ref=psi_ref,
    )

    loaded_state, source, error = runtime_loader._resolve_prepared_state(
        payload,
        replay_context,
    )

    assert source == "payload_logical_shared_alias"
    assert error < 1.0e-12
    assert np.allclose(loaded_state, psi_initial)


def test_replay_selected_support_prefers_serialized_pauli_terms_and_execution_mode() -> None:
    payload = {
        "adapt_vqe": {
            "operators": ["serialized_grouped"],
            "selected_operator_execution_modes": ["grouped_exact"],
            "selected_operator_pauli_terms": [
                [
                    {"pauli_exyz": "x", "coeff_re": 1.0, "coeff_im": 0.0},
                    {"pauli_exyz": "z", "coeff_re": 0.5, "coeff_im": 0.0},
                ]
            ],
        }
    }

    replay_terms = _replay_terms_from_serialized_selected_pauli_terms(payload)

    assert replay_terms is not None
    assert len(replay_terms) == 1
    assert replay_terms[0].label == "serialized_grouped"
    assert replay_terms[0].execution_mode == "grouped_exact"
    coeffs = {
        term.pw2strng(): float(complex(term.p_coeff).real)
        for term in replay_terms[0].polynomial.return_polynomial()
    }
    assert coeffs == {"x": 1.0, "z": 0.5}


def test_generic_runtime_reconstruction_respects_logical_shared_mode() -> None:
    polynomial = PauliPolynomial(
        "JW",
        [
            PauliTerm(1, ps="x", pc=1.0),
            PauliTerm(1, ps="z", pc=0.5),
        ],
    )
    term = AnsatzTerm(label="logical_shared_runtime", polynomial=polynomial)
    layout = build_parameter_layout(
        [term],
        ignore_identity=True,
        coefficient_tolerance=1.0e-12,
        sort_terms=True,
    )
    theta_logical = np.asarray([0.2], dtype=float)
    theta_runtime = expand_legacy_logical_theta(theta_logical, layout)
    psi_ref = np.asarray([1.0 + 0.0j, 0.0 + 0.0j])
    expected = CompiledAnsatzExecutor(
        [term],
        coefficient_tolerance=float(layout.coefficient_tolerance),
        ignore_identity=bool(layout.ignore_identity),
        sort_terms=True,
        parameterization_mode="logical_shared",
        parameterization_layout=layout,
    ).prepare_state(theta_logical, psi_ref)

    reconstructed = runtime_loader._reconstruct_prepared_state_from_runtime_input(
        selected_terms=[term],
        layout=layout,
        theta_runtime=theta_runtime,
        theta_logical=theta_logical,
        parameterization_mode="logical_shared",
        psi_ref=psi_ref,
    )

    assert np.allclose(reconstructed, expected, atol=1.0e-12, rtol=0.0)


def test_legacy_generic_layout_recovers_grouped_execution_from_exact_model_pool() -> None:
    pool_term = AnsatzTerm(
        label="coupled::mode::dH_dQ_times_p",
        polynomial=PauliPolynomial(
            "JW",
            [PauliTerm(2, ps="xx", pc=0.25), PauliTerm(2, ps="yy", pc=-0.75)],
        ),
        execution_mode="grouped_exact",
    )
    parameterization = serialize_layout(build_parameter_layout([pool_term]))
    parameterization["blocks"][0].pop("execution_mode")
    legacy_layout = deserialize_layout(parameterization)
    legacy_terms = runtime_loader._selected_terms_from_layout(legacy_layout)
    resolved_problem = SimpleNamespace(
        runtime_data={
            "vibronic_h2o_linear_fd_model": SimpleNamespace(pool=[pool_term]),
        }
    )

    recovered_layout, recovered_terms, metadata = (
        runtime_loader._recover_legacy_layout_execution_modes(
            parameterization=parameterization,
            layout=legacy_layout,
            selected_terms=legacy_terms,
            resolved_problem=resolved_problem,
        )
    )

    assert recovered_terms == (pool_term,)
    assert recovered_layout.blocks[0].execution_mode == "grouped_exact"
    assert metadata["legacy_missing_execution_mode"] is True
    assert metadata["exact_pool_rebind_count"] == 1
    assert metadata["execution_mode_changed_count"] == 1
    assert metadata["unresolved_block_count"] == 0


def _logical_shared_resume_source(
    *,
    mode_fields: dict[str, str] | None = None,
    theta_runtime: np.ndarray | None = None,
    theta_logical: np.ndarray | None = None,
    checkpoint_phase: float = 0.0,
    corrupt_checkpoint: bool = False,
) -> tuple[ResumeScaffoldSource, AnsatzTerm]:
    polynomial = PauliPolynomial(
        "JW",
        [
            PauliTerm(1, ps="x", pc=1.0),
            PauliTerm(1, ps="z", pc=0.5),
        ],
    )
    term = AnsatzTerm(label="resume_logical_shared", polynomial=polynomial)
    layout = build_parameter_layout(
        [term],
        ignore_identity=True,
        coefficient_tolerance=1.0e-12,
        sort_terms=True,
    )
    logical = (
        np.asarray([0.2], dtype=float)
        if theta_logical is None
        else np.asarray(theta_logical, dtype=float).reshape(-1)
    )
    runtime = (
        expand_legacy_logical_theta(logical, layout)
        if theta_runtime is None
        else np.asarray(theta_runtime, dtype=float).reshape(-1)
    )
    psi_ref = np.asarray([1.0 + 0.0j, 0.0 + 0.0j])
    prepared = CompiledAnsatzExecutor(
        [term],
        coefficient_tolerance=float(layout.coefficient_tolerance),
        ignore_identity=bool(layout.ignore_identity),
        sort_terms=True,
        parameterization_mode="logical_shared",
        parameterization_layout=layout,
    ).prepare_state(logical, psi_ref)
    psi_initial = np.exp(1.0j * float(checkpoint_phase)) * prepared
    if corrupt_checkpoint:
        psi_initial = np.asarray([0.0 + 0.0j, 1.0 + 0.0j])

    adapt_vqe: dict[str, object] = {}
    checkpoint: dict[str, object] = {}
    for field, value in (mode_fields or {}).items():
        if field.startswith("adapt_vqe."):
            adapt_vqe[field.split(".", 1)[1]] = value
        elif field.startswith("checkpoint."):
            checkpoint[field.split(".", 1)[1]] = value
        else:  # pragma: no cover - test-helper misuse
            raise AssertionError(f"unsupported test mode field: {field}")
    payload: dict[str, object] = {"adapt_vqe": adapt_vqe}
    if checkpoint:
        payload["checkpoint"] = checkpoint
    source = ResumeScaffoldSource(
        artifact_json=Path("logical_shared_resume.json"),
        artifact_sha256="b" * 64,
        payload=payload,
        runtime_input=ScaffoldRuntimeInput(
            resolved_problem=SimpleNamespace(),
            psi_ref=psi_ref,
            psi_initial=psi_initial,
            base_layout=layout,
            theta_runtime=runtime,
            theta_logical=logical,
            structure_locked=False,
            exact_energy=None,
            selected_terms=(term,),
        ),
        import_summary={},
    )
    return source, term


def _match_logical_shared(
    source: ResumeScaffoldSource,
    term: AnsatzTerm,
):
    return match_resume_scaffold_to_pool(
        source,
        pool=[term],
        build_selected_layout=lambda ops: build_parameter_layout(
            ops,
            ignore_identity=True,
            coefficient_tolerance=1.0e-12,
            sort_terms=True,
        ),
        expected_parameterization_mode="logical_shared",
    )


def test_resume_match_replays_matching_explicit_mode_up_to_global_phase() -> None:
    source, term = _logical_shared_resume_source(
        mode_fields={
            "adapt_vqe.parameterization_execution_mode": "logical_shared",
            "adapt_vqe.parameterization_mode": "logical_shared_v1",
            "checkpoint.parameterization_execution_mode": "logical_shared",
            "checkpoint.parameterization_mode": "logical-shared",
        },
        checkpoint_phase=0.37,
    )

    match = _match_logical_shared(source, term)

    replay = match.validation["strict_expected_mode_replay"]
    assert replay["passed"] is True
    assert replay["global_phase_invariant"] is True
    assert replay["l2_error_up_to_global_phase"] < 1.0e-12
    assert match.validation["source_parameterization_mode_inferred"] is False


def test_resume_match_rejects_conflicting_explicit_mode_aliases() -> None:
    source, term = _logical_shared_resume_source(
        mode_fields={
            "adapt_vqe.parameterization_execution_mode": "logical_shared",
            "checkpoint.parameterization_mode": "per_pauli_term",
        }
    )

    with pytest.raises(ValueError, match="conflicting explicit parameterization"):
        _match_logical_shared(source, term)


def test_resume_match_rejects_conflicting_checkpoint_execution_alias() -> None:
    source, term = _logical_shared_resume_source(
        mode_fields={
            "adapt_vqe.parameterization_execution_mode": "logical_shared",
            "checkpoint.parameterization_execution_mode": "per_pauli_term",
        }
    )

    with pytest.raises(ValueError, match="conflicting explicit parameterization"):
        _match_logical_shared(source, term)


def test_resume_match_rejects_explicit_mode_mismatch() -> None:
    source, term = _logical_shared_resume_source(
        mode_fields={"adapt_vqe.parameterization_mode": "per_pauli_term"}
    )

    with pytest.raises(ValueError, match="does not match the current route"):
        _match_logical_shared(source, term)


def test_logical_shared_resume_rejects_nonalias_runtime_block() -> None:
    source, term = _logical_shared_resume_source(
        mode_fields={"adapt_vqe.parameterization_execution_mode": "logical_shared"},
        theta_runtime=np.asarray([0.6, -0.2]),
    )

    with pytest.raises(ValueError, match="blockwise alias"):
        _match_logical_shared(source, term)


def test_logical_shared_resume_requires_explicit_logical_vector() -> None:
    source, term = _logical_shared_resume_source(
        mode_fields={"adapt_vqe.parameterization_execution_mode": "logical_shared"}
    )
    source = replace(
        source,
        runtime_input=replace(source.runtime_input, theta_logical=None),
    )

    with pytest.raises(ValueError, match="requires an explicit logical_optimal_point"):
        _match_logical_shared(source, term)


def test_mode_missing_legacy_is_inferred_only_after_strict_replay() -> None:
    source, term = _logical_shared_resume_source(mode_fields={})

    match = _match_logical_shared(source, term)

    assert match.validation["source_parameterization_mode_inferred"] is True
    assert match.validation["source_parameterization_mode_resolution"] == (
        "expected_mode_strict_replay_inference"
    )
    assert match.validation["strict_expected_mode_replay"]["passed"] is True


def test_mode_missing_legacy_fails_closed_when_expected_mode_replay_fails() -> None:
    source, term = _logical_shared_resume_source(
        mode_fields={},
        corrupt_checkpoint=True,
    )

    with pytest.raises(ValueError, match="strict expected-mode replay failed"):
        _match_logical_shared(source, term)


def test_resume_match_preserves_serialized_generator_when_label_semantics_differ() -> None:
    source, serialized_term = _logical_shared_resume_source(
        mode_fields={"adapt_vqe.parameterization_execution_mode": "logical_shared"}
    )
    pool_term = AnsatzTerm(
        label=serialized_term.label,
        polynomial=PauliPolynomial(
            "JW",
            [
                PauliTerm(1, ps="y", pc=1.0),
                PauliTerm(1, ps="z", pc=0.5),
            ],
        ),
    )

    match = match_resume_scaffold_to_pool(
        source,
        pool=[pool_term],
        build_selected_layout=lambda ops: build_parameter_layout(
            ops,
            ignore_identity=True,
            coefficient_tolerance=1.0e-12,
            sort_terms=True,
        ),
        expected_parameterization_mode="logical_shared",
    )

    assert match.selected_ops == (serialized_term,)
    assert match.selected_pool_indices == ()
    assert match.validation["selected_terms_outside_pool_reason"] == (
        "serialized_selected_generator_semantics_preserved_v1"
    )
    record = match.validation["selected_terms_outside_pool_records"][0]
    assert record["label"] == serialized_term.label
    assert record["reason"] == "same_label_pool_semantics_mismatch"

def test_resume_match_uses_canonical_polynomial_semantics_not_component_order() -> None:
    source, serialized_term = _logical_shared_resume_source(
        mode_fields={"adapt_vqe.parameterization_execution_mode": "logical_shared"}
    )
    serialized_components = list(serialized_term.polynomial.return_polynomial())
    pool_term = AnsatzTerm(
        label=serialized_term.label,
        polynomial=SimpleNamespace(
            return_polynomial=lambda: tuple(reversed(serialized_components))
        ),
        execution_mode=serialized_term.execution_mode,
    )
    assert candidate_generator_fingerprint(pool_term) != candidate_generator_fingerprint(
        serialized_term
    )

    match = match_resume_scaffold_to_pool(
        source,
        pool=[pool_term],
        build_selected_layout=lambda ops: build_parameter_layout(
            ops,
            ignore_identity=True,
            coefficient_tolerance=1.0e-12,
            sort_terms=True,
        ),
        expected_parameterization_mode="logical_shared",
    )

    assert match.selected_ops == (pool_term,)
    assert match.selected_pool_indices == (0,)
    assert match.validation["selected_terms_outside_pool_count"] == 0
