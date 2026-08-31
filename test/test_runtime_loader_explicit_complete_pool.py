"""Regression contracts for explicit generic replay-family candidate pools."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from pipelines.scaffold.runtime_loader import load_scaffold_runtime_input_from_payload
from pipelines.time_dynamics.pure_hubbard_fixed_vqe import (
    PureHubbardFixedVQEConfig,
    build_pure_hubbard_fixed_vqe_problem,
)
from src.quantum.ansatz_parameterization import expand_legacy_logical_theta, serialize_layout


LOCKED_FILTER = "fixed_spin_sector_logical_generators_v1"
LOCKED_POOL = "uccsd_qeb_hva_blocks"
LOCKED_ORDER = "repository_deduplicated_pool_order_v1"
LOCKED_EXECUTION = "grouped_exact_logical_shared_v1"
LOCKED_UNIQUE_GENERATOR_COUNT = 113
LOCKED_GENERATOR_POOL_SHA256 = (
    "a01e952a7496faa48ce10bcba64a1c90b5cced229f7c3ae24c87dfd2d1fbf1bb"
)


def _independent_pool_digest(terms: object) -> str:
    payload = []
    for term in terms:
        polynomial = sorted(
            (
                {
                    "pauli_exyz": str(child.pw2strng()),
                    "coefficient_real": float(complex(child.p_coeff).real),
                    "coefficient_imag": float(complex(child.p_coeff).imag),
                }
                for child in term.polynomial.return_polynomial()
            ),
            key=lambda row: row["pauli_exyz"],
        )
        payload.append(
            {
                "label": str(term.label),
                "execution_mode": str(term.execution_mode),
                "polynomial": polynomial,
            }
        )
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _state_payload(state: np.ndarray, *, kind: str) -> dict[str, object]:
    vector = np.asarray(state, dtype=complex).reshape(-1)
    nq = int(np.log2(vector.size))
    amplitudes = {
        format(index, f"0{nq}b"): {
            "re": float(np.real(amplitude)),
            "im": float(np.imag(amplitude)),
        }
        for index, amplitude in enumerate(vector)
        if abs(complex(amplitude)) > 1.0e-14
    }
    return {
        "source": "test",
        "handoff_state_kind": kind,
        "nq_total": nq,
        "amplitudes_qn_to_q0": amplitudes,
    }


def _pure_hubbard_payload() -> tuple[dict[str, object], int]:
    config = PureHubbardFixedVQEConfig(
        num_sites=4,
        n_up=2,
        n_down=2,
        reps=2,
        restarts=1,
        maxiter=1,
    )
    problem = build_pure_hubbard_fixed_vqe_problem(config)
    theta_logical = np.zeros(problem.ansatz.num_parameters, dtype=float)
    theta_runtime = expand_legacy_logical_theta(theta_logical, problem.ansatz.layout)
    settings = config.to_json_dict()
    settings.update(
        {
            "adapt_pool": LOCKED_POOL,
            "dv": 0.0,
            "sector_n_dn": int(config.n_down),
            "include_zero_point": False,
        }
    )
    payload: dict[str, object] = {
        "schema": "paper_ii_vqe_seed_artifact_v1",
        "lane_id": "paper_ii_hubbard_l4_scale_v1",
        "model": "hubbard",
        "settings": settings,
        "ansatz_input_state": _state_payload(
            problem.reference_state, kind="reference_state"
        ),
        "initial_state": _state_payload(
            problem.reference_state, kind="prepared_state"
        ),
        "adapt_vqe": {
            "pool_type": LOCKED_POOL,
            "structure_locked": True,
            "parameterization": serialize_layout(problem.ansatz.layout),
            "parameterization_mode": "logical_shared",
            "parameterization_execution_mode": "logical_shared",
            "operators": [str(term.label) for term in problem.generator_terms],
            "optimal_point": [float(value) for value in theta_runtime],
            "logical_optimal_point": [float(value) for value in theta_logical],
            "fixed_scaffold_metadata": {
                "generator_filter": LOCKED_FILTER,
                "term_order_basis": LOCKED_ORDER,
                "generator_execution": LOCKED_EXECUTION,
                "generator_pool_sha256": LOCKED_GENERATOR_POOL_SHA256,
                "operator_count": len(problem.generator_terms),
            },
        },
        "architecture_binding": {
            "policy": {
                "generator_pool": LOCKED_POOL,
                "generator_filter": LOCKED_FILTER,
                "generator_order": LOCKED_ORDER,
                "generator_execution": LOCKED_EXECUTION,
                "expected_unique_generator_count": LOCKED_UNIQUE_GENERATOR_COUNT,
            },
            "generator_pool_sha256": LOCKED_GENERATOR_POOL_SHA256,
            "unique_generator_count": LOCKED_UNIQUE_GENERATOR_COUNT,
        },
        "exact_energy": float(problem.exact_energy),
    }
    return payload, LOCKED_UNIQUE_GENERATOR_COUNT


def test_pure_hubbard_complete_pool_override_materializes_admissible_terms() -> None:
    payload, expected_pool_count = _pure_hubbard_payload()

    legacy = load_scaffold_runtime_input_from_payload(payload)
    assert legacy.candidate_pool_source.completeness == "selected_only"
    assert legacy.candidate_pool_terms == ()

    explicit_payload = dict(payload)
    explicit_payload["replay_candidate_pool_mode"] = "replay_family_pool"
    complete = load_scaffold_runtime_input_from_payload(explicit_payload)

    assert complete.candidate_pool_source.completeness == "complete"
    assert complete.candidate_pool_source.source_kind == "resolved_pool"
    assert len(complete.candidate_pool_terms) == expected_pool_count
    assert _independent_pool_digest(complete.candidate_pool_terms) == (
        LOCKED_GENERATOR_POOL_SHA256
    )
    assert len(complete.candidate_pool_terms) > 0
    assert all(
        str(term.execution_mode) == "grouped_exact"
        for term in complete.candidate_pool_terms
    )
    assert all(
        not str(term.label).startswith("layer::")
        for term in complete.candidate_pool_terms
    )
    override = complete.candidate_pool_source.filter_payload[
        "explicit_complete_pool_override"
    ]
    assert override["requested_mode"] == "replay_family_pool"
    assert override["candidate_pool_build_executed"] is True
    assert override["resolved_candidate_pool_term_count"] == expected_pool_count
    assert override["declared_generator_filter"] == LOCKED_FILTER
    assert complete.extensions["generic_pool_meta"]["dedup_total_after_declared_filter"] == (
        expected_pool_count
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_architecture", "architecture binding"),
        ("missing_filter", "generator filter"),
        ("unknown_filter", "generator filter"),
        ("wrong_count", "generator count"),
        ("wrong_order", "generator order"),
        ("wrong_digest", "generator pool digest"),
    ),
)
def test_pure_hubbard_complete_pool_override_fails_closed_on_seed_declaration_drift(
    mutation: str,
    message: str,
) -> None:
    payload, _ = _pure_hubbard_payload()
    fixed = payload["adapt_vqe"]["fixed_scaffold_metadata"]
    architecture = payload["architecture_binding"]
    policy = architecture["policy"]
    if mutation == "missing_architecture":
        payload.pop("architecture_binding")
    elif mutation == "missing_filter":
        fixed.pop("generator_filter")
    elif mutation == "unknown_filter":
        fixed["generator_filter"] = "unknown_filter_v9"
    elif mutation == "wrong_count":
        architecture["unique_generator_count"] = 112
    elif mutation == "wrong_order":
        policy["generator_order"] = "reverse_order_v9"
    elif mutation == "wrong_digest":
        architecture["generator_pool_sha256"] = "0" * 64
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(mutation)
    payload["replay_candidate_pool_mode"] = "replay_family_pool"

    with pytest.raises(ValueError, match=message):
        load_scaffold_runtime_input_from_payload(payload)


@pytest.mark.parametrize("legacy_filter", (None, "unknown_legacy_filter_v1"))
def test_nonprospective_generic_replay_preserves_legacy_filter_fallback(
    legacy_filter: str | None,
) -> None:
    payload, _ = _pure_hubbard_payload()
    payload.pop("schema")
    payload.pop("architecture_binding")
    fixed = payload["adapt_vqe"]["fixed_scaffold_metadata"]
    if legacy_filter is None:
        fixed.pop("generator_filter")
    else:
        fixed["generator_filter"] = legacy_filter
    payload["replay_candidate_pool_mode"] = "replay_family_pool"

    loaded = load_scaffold_runtime_input_from_payload(payload)

    assert loaded.candidate_pool_source.completeness == "complete"
    assert len(loaded.candidate_pool_terms) >= LOCKED_UNIQUE_GENERATOR_COUNT
    assert (
        loaded.extensions["generic_pool_meta"].get(
            "prospective_seed_declarations_verified"
        )
        is not True
    )
