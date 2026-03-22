from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from src.quantum.ansatz_parameterization import build_parameter_layout, serialize_layout
from pipelines.hardcoded.hh_vqe_from_adapt_family import (
    PoolTermwiseAnsatz,
    REPLAY_CONTRACT_VERSION,
    REPLAY_SEED_POLICIES,
    _build_full_meta_replay_terms_sparse,
    _build_replay_seed_theta,
    _build_replay_seed_theta_policy,
    _build_replay_terms_from_adapt_labels,
    _extract_adapt_operator_theta_sequence,
    _infer_handoff_state_kind,
)
from src.quantum.hubbard_latex_python_pairs import build_hubbard_holstein_hamiltonian
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, PauliPolynomial, PauliTerm


@dataclass(frozen=True)
class _DummyTerm:
    label: str


def test_extract_adapt_operator_theta_sequence_valid() -> None:
    payload = {
        "adapt_vqe": {
            "operators": ["g0", "g1", "g2"],
            "optimal_point": [0.1, -0.2, 0.3],
        }
    }
    labels, theta = _extract_adapt_operator_theta_sequence(payload)
    assert labels == ["g0", "g1", "g2"]
    assert np.allclose(theta, np.array([0.1, -0.2, 0.3], dtype=float))


def test_extract_adapt_operator_theta_sequence_rejects_missing_block() -> None:
    with pytest.raises(ValueError, match="missing object key 'adapt_vqe'"):
        _extract_adapt_operator_theta_sequence({})


def test_extract_adapt_operator_theta_sequence_rejects_missing_operators() -> None:
    payload = {"adapt_vqe": {"optimal_point": [0.1]}}
    with pytest.raises(ValueError, match="adapt_vqe\\.operators"):
        _extract_adapt_operator_theta_sequence(payload)


def test_extract_adapt_operator_theta_sequence_rejects_missing_optimal_point() -> None:
    payload = {"adapt_vqe": {"operators": ["g0"]}}
    with pytest.raises(ValueError, match="adapt_vqe\\.optimal_point"):
        _extract_adapt_operator_theta_sequence(payload)


def test_extract_adapt_operator_theta_sequence_rejects_length_mismatch() -> None:
    payload = {
        "adapt_vqe": {
            "operators": ["g0", "g1"],
            "optimal_point": [0.1],
        }
    }
    with pytest.raises(ValueError, match="Length mismatch"):
        _extract_adapt_operator_theta_sequence(payload)


def test_extract_adapt_operator_theta_sequence_rejects_nonfinite_theta() -> None:
    payload = {
        "adapt_vqe": {
            "operators": ["g0"],
            "optimal_point": [float("nan")],
        }
    }
    with pytest.raises(ValueError, match="Non-finite theta value"):
        _extract_adapt_operator_theta_sequence(payload)


def test_extract_adapt_operator_theta_sequence_accepts_parameterized_runtime_vector() -> None:
    terms = [
        AnsatzTerm(
            label="g0",
            polynomial=PauliPolynomial("JW", [PauliTerm(2, ps="xx", pc=1.0), PauliTerm(2, ps="zz", pc=0.5)]),
        ),
        AnsatzTerm(
            label="g1",
            polynomial=PauliPolynomial("JW", [PauliTerm(2, ps="xy", pc=1.0)]),
        ),
    ]
    layout = build_parameter_layout(terms)
    payload = {
        "adapt_vqe": {
            "operators": ["g0", "g1"],
            "optimal_point": [0.1, -0.2, 0.3],
            "parameterization": serialize_layout(layout),
        }
    }
    labels, theta = _extract_adapt_operator_theta_sequence(payload)
    assert labels == ["g0", "g1"]
    assert np.allclose(theta, [0.1, -0.2, 0.3])


def test_build_replay_terms_preserves_operator_order_and_duplicates() -> None:
    pool = [_DummyTerm("a"), _DummyTerm("b"), _DummyTerm("c")]
    replay = _build_replay_terms_from_adapt_labels(pool, ["b", "a", "b", "c"])
    assert [str(t.label) for t in replay] == ["b", "a", "b", "c"]
    assert replay[0] is pool[1]
    assert replay[2] is pool[1]


def test_build_replay_terms_rejects_unknown_label() -> None:
    pool = [_DummyTerm("a"), _DummyTerm("b")]
    with pytest.raises(ValueError, match="not present"):
        _build_replay_terms_from_adapt_labels(pool, ["a", "missing"])


def test_build_replay_terms_reconstructs_runtime_split_children_from_payload() -> None:
    child_label = "macro::split[0]::eyeexy"
    replay = _build_replay_terms_from_adapt_labels(
        [],
        [child_label],
        payload={
            "continuation": {
                "selected_generator_metadata": [
                    {
                        "candidate_label": child_label,
                        "compile_metadata": {
                            "serialized_terms_exyz": [
                                {
                                    "pauli_exyz": "eyeexy",
                                    "coeff_re": 1.0,
                                    "coeff_im": 0.0,
                                    "nq": 6,
                                }
                            ]
                        },
                    }
                ]
            }
        },
    )
    assert len(replay) == 1
    assert str(replay[0].label) == child_label
    terms = list(replay[0].polynomial.return_polynomial())
    assert len(terms) == 1
    assert str(terms[0].pw2strng()) == "eyeexy"
    assert complex(terms[0].p_coeff) == pytest.approx(1.0 + 0.0j)


def test_sparse_full_meta_replay_terms_reconstruct_runtime_split_children_from_payload() -> None:
    child_label = "macro::split[0]::eeeeeexy"
    cfg = SimpleNamespace(
        L=2,
        t=1.0,
        u=4.0,
        dv=0.0,
        omega0=1.0,
        g_ep=0.5,
        n_ph_max=2,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        sector_n_up=1,
        sector_n_dn=1,
        paop_r=1,
        paop_split_paulis=False,
        paop_prune_eps=0.0,
        paop_normalization="none",
    )
    h_poly = build_hubbard_holstein_hamiltonian(
        dims=2,
        J=1.0,
        U=4.0,
        omega0=1.0,
        g=0.5,
        n_ph_max=2,
        boson_encoding="binary",
        v_t=None,
        v0=0.0,
        t_eval=None,
        repr_mode="JW",
        indexing="blocked",
        pbc=False,
        include_zero_point=True,
    )
    replay_terms, meta = _build_full_meta_replay_terms_sparse(
        cfg,
        h_poly=h_poly,
        adapt_labels=[child_label],
        payload={
            "continuation": {
                "selected_generator_metadata": [
                    {
                        "candidate_label": child_label,
                        "compile_metadata": {
                            "serialized_terms_exyz": [
                                {
                                    "pauli_exyz": "eeeeeexy",
                                    "coeff_re": 1.0,
                                    "coeff_im": 0.0,
                                    "nq": 8,
                                }
                            ]
                        },
                    }
                ]
            }
        },
    )
    assert meta["family"] == "full_meta"
    assert [str(term.label) for term in replay_terms] == [child_label]
    terms = list(replay_terms[0].polynomial.return_polynomial())
    assert len(terms) == 1
    assert str(terms[0].pw2strng()) == "eeeeeexy"


def test_build_replay_seed_theta_tiled_and_npar_matches_runtime_parameter_count_times_reps() -> None:
    replay_terms = [
        AnsatzTerm(
            label="g0",
            polynomial=PauliPolynomial("JW", [PauliTerm(2, ps="xx", pc=1.0), PauliTerm(2, ps="zz", pc=0.5)]),
        ),
        AnsatzTerm(
            label="g1",
            polynomial=PauliPolynomial("JW", [PauliTerm(2, ps="xy", pc=1.0)]),
        ),
    ]
    layout = build_parameter_layout(replay_terms)
    adapt_theta = np.array([0.2, -0.1, 0.05], dtype=float)
    reps = 4
    seed = _build_replay_seed_theta(adapt_theta, reps=reps)
    assert np.allclose(seed, np.tile(adapt_theta, reps))

    ansatz = PoolTermwiseAnsatz(
        terms=replay_terms,
        reps=reps,
        nq=2,
        parameterization_layout=layout,
        parameterization_mode="per_pauli_term",
    )
    assert int(seed.size) == int(ansatz.num_parameters)


# ── Replay contract version / constant tests ────────────────────────────

def test_replay_contract_version_is_2() -> None:
    assert REPLAY_CONTRACT_VERSION == 2


def test_replay_seed_policies_set() -> None:
    assert REPLAY_SEED_POLICIES == {"auto", "scaffold_plus_zero", "residual_only", "tile_adapt"}


# ── Provenance inference tests ───────────────────────────────────────────

class TestInferHandoffStateKind:
    """Test _infer_handoff_state_kind legacy inference and explicit lookup."""

    def test_explicit_prepared_state(self) -> None:
        payload = {"initial_state": {"handoff_state_kind": "prepared_state"}}
        kind, src = _infer_handoff_state_kind(payload)
        assert kind == "prepared_state"
        assert src == "explicit"

    def test_explicit_reference_state(self) -> None:
        payload = {"initial_state": {"handoff_state_kind": "reference_state"}}
        kind, src = _infer_handoff_state_kind(payload)
        assert kind == "reference_state"
        assert src == "explicit"

    def test_infer_hf_as_reference(self) -> None:
        payload = {"initial_state": {"source": "hf"}}
        kind, src = _infer_handoff_state_kind(payload)
        assert kind == "reference_state"
        assert src == "inferred_source"

    def test_infer_exact_as_reference(self) -> None:
        payload = {"initial_state": {"source": "exact"}}
        kind, src = _infer_handoff_state_kind(payload)
        assert kind == "reference_state"
        assert src == "inferred_source"

    def test_infer_adapt_vqe_as_prepared(self) -> None:
        payload = {"initial_state": {"source": "adapt_vqe"}}
        kind, src = _infer_handoff_state_kind(payload)
        assert kind == "prepared_state"
        assert src == "inferred_source"

    def test_infer_a_probe_final_as_prepared(self) -> None:
        payload = {"initial_state": {"source": "A_probe_final"}}
        kind, src = _infer_handoff_state_kind(payload)
        assert kind == "prepared_state"
        assert src == "inferred_source"

    def test_infer_b_medium_final_as_prepared(self) -> None:
        payload = {"initial_state": {"source": "B_medium_final"}}
        kind, src = _infer_handoff_state_kind(payload)
        assert kind == "prepared_state"
        assert src == "inferred_source"

    def test_infer_warm_start_hva_as_prepared(self) -> None:
        payload = {"initial_state": {"source": "warm_start_hva"}}
        kind, src = _infer_handoff_state_kind(payload)
        assert kind == "prepared_state"
        assert src == "inferred_source"

    def test_ambiguous_when_no_initial_state(self) -> None:
        kind, src = _infer_handoff_state_kind({})
        assert kind == "ambiguous"
        assert src == "ambiguous"

    def test_ambiguous_for_unknown_source(self) -> None:
        payload = {"initial_state": {"source": "unknown_thing"}}
        kind, src = _infer_handoff_state_kind(payload)
        assert kind == "ambiguous"
        assert src == "ambiguous"

    def test_explicit_overrides_source(self) -> None:
        """Explicit field takes precedence over source inference."""
        payload = {
            "initial_state": {
                "handoff_state_kind": "reference_state",
                "source": "adapt_vqe",
            }
        }
        kind, src = _infer_handoff_state_kind(payload)
        assert kind == "reference_state"
        assert src == "explicit"


# ── Policy-aware seed builder tests ──────────────────────────────────────

class TestBuildReplaySeedThetaPolicy:
    """Test _build_replay_seed_theta_policy for each policy."""

    def test_tile_adapt(self) -> None:
        theta = np.array([0.1, -0.2, 0.3])
        seed, resolved = _build_replay_seed_theta_policy(
            theta, reps=3, policy="tile_adapt", handoff_state_kind="prepared_state",
        )
        assert resolved == "tile_adapt"
        assert np.allclose(seed, np.tile(theta, 3))

    def test_scaffold_plus_zero(self) -> None:
        theta = np.array([0.1, -0.2, 0.3])
        seed, resolved = _build_replay_seed_theta_policy(
            theta, reps=3, policy="scaffold_plus_zero", handoff_state_kind="reference_state",
        )
        assert resolved == "scaffold_plus_zero"
        expected = np.zeros(9, dtype=float)
        expected[:3] = theta
        assert np.allclose(seed, expected)

    def test_residual_only(self) -> None:
        theta = np.array([0.1, -0.2, 0.3])
        seed, resolved = _build_replay_seed_theta_policy(
            theta, reps=3, policy="residual_only", handoff_state_kind="prepared_state",
        )
        assert resolved == "residual_only"
        assert np.allclose(seed, np.zeros(9))

    def test_auto_prepared_gives_residual_only(self) -> None:
        theta = np.array([0.5, -0.5])
        seed, resolved = _build_replay_seed_theta_policy(
            theta, reps=2, policy="auto", handoff_state_kind="prepared_state",
        )
        assert resolved == "residual_only"
        assert np.allclose(seed, np.zeros(4))

    def test_auto_reference_gives_scaffold_plus_zero(self) -> None:
        theta = np.array([0.5, -0.5])
        seed, resolved = _build_replay_seed_theta_policy(
            theta, reps=2, policy="auto", handoff_state_kind="reference_state",
        )
        assert resolved == "scaffold_plus_zero"
        expected = np.zeros(4, dtype=float)
        expected[:2] = theta
        assert np.allclose(seed, expected)

    def test_auto_seed_semantics_are_mode_independent(self) -> None:
        theta = np.array([0.5, -0.5])
        seed_prepared, resolved_prepared = _build_replay_seed_theta_policy(
            theta, reps=2, policy="auto", handoff_state_kind="prepared_state",
        )
        seed_reference, resolved_reference = _build_replay_seed_theta_policy(
            theta, reps=2, policy="auto", handoff_state_kind="reference_state",
        )
        assert resolved_prepared == "residual_only"
        assert np.allclose(seed_prepared, np.zeros(4))
        assert resolved_reference == "scaffold_plus_zero"
        assert np.allclose(seed_reference[:2], theta)

    def test_auto_ambiguous_raises(self) -> None:
        theta = np.array([0.1])
        with pytest.raises(ValueError, match="Cannot resolve replay seed policy 'auto'"):
            _build_replay_seed_theta_policy(
                theta, reps=1, policy="auto", handoff_state_kind="ambiguous",
            )

    def test_seed_length_matches_adapt_depth_times_reps(self) -> None:
        """Verify seed length equals adapt_depth * reps for all policies."""
        theta = np.array([0.1, 0.2, 0.3])
        for policy in ["tile_adapt", "scaffold_plus_zero", "residual_only"]:
            seed, _ = _build_replay_seed_theta_policy(
                theta, reps=4, policy=policy, handoff_state_kind="reference_state",
            )
            assert int(seed.size) == 12, f"Policy {policy}: expected 12, got {seed.size}"

    def test_scaffold_plus_zero_reps_1_equals_adapt_theta(self) -> None:
        theta = np.array([0.1, -0.2, 0.3])
        seed, resolved = _build_replay_seed_theta_policy(
            theta, reps=1, policy="scaffold_plus_zero", handoff_state_kind="reference_state",
        )
        assert resolved == "scaffold_plus_zero"
        assert np.allclose(seed, theta)

    def test_tile_adapt_reps_1_equals_adapt_theta(self) -> None:
        theta = np.array([0.1, -0.2, 0.3])
        seed, resolved = _build_replay_seed_theta_policy(
            theta, reps=1, policy="tile_adapt", handoff_state_kind="reference_state",
        )
        assert resolved == "tile_adapt"
        assert np.allclose(seed, theta)

    def test_residual_only_reps_1_is_all_zeros(self) -> None:
        theta = np.array([0.1, -0.2, 0.3])
        seed, resolved = _build_replay_seed_theta_policy(
            theta, reps=1, policy="residual_only", handoff_state_kind="prepared_state",
        )
        assert resolved == "residual_only"
        assert np.allclose(seed, np.zeros(3))
