import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "agent_guidance"
    / "skills"
    / "paper-i-results"
    / "scripts"
    / "compute_paper_i_main_fidelities.py"
)
ACTUAL_SHAPE_DUPLICATE_TERMINAL_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "paper_i_main_fidelity_duplicate_terminal_checkpoint_actual_shape.json"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("paper_i_main_fidelity_audit", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


audit = _load_module()


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _rehash_checkpoint(checkpoint: dict) -> None:
    payload = dict(checkpoint)
    payload.pop("checkpoint_sha256", None)
    checkpoint["checkpoint_sha256"] = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _tables_i_ii_payload(source_json: Path) -> dict:
    return {
        "schema": "paper_i_tables_i_ii_spsa_optuna_current_best_promotion_v1",
        "source_rows": [
            {
                "table_label": "tab:fixed_accuracy_claims",
                "family": "hubbard_family",
                "method_id": "static_full_meta_append_adapt_vqe",
                "method_label": "append ADAPT",
                "cases": [
                    {
                        "case_id": "hubbard_L2_three_model_weak",
                        "generic_static_single_json": str(source_json),
                        "generic_static_single_sha256": "abc",
                        "one_minus_F_display": "--",
                    }
                ],
            }
        ],
    }


def _hh_prefix_payload(source_json: Path) -> dict:
    return {
        "schema": "paper_i_hh_tableiii_first_effective_plateau_prefix_cost_audit_v1",
        "rows": [
            {
                "table_label": "tab:hh_first_plateau_prefix_costs",
                "regime": "weak_weak",
                "method": "Geo-ADAPT",
                "algorithm_id": "static_geo_adapt_vqe",
                "source_json": str(source_json),
                "source_sha256": "def",
                "accepted_operator_groups": 6,
                "plateau_iteration": 6,
                "n_ph_work": 2,
                "status": "ok",
            }
        ],
    }


def test_extracts_tables_i_ii_and_hh_visible_specs(tmp_path: Path):
    source = tmp_path / "source" / "generic_static_single.json"
    tables = _write_json(tmp_path / "tables.json", _tables_i_ii_payload(source))
    hh = _write_json(tmp_path / "hh_prefix.json", _hh_prefix_payload(source))

    table_specs, table_input = audit._tables_i_ii_specs(tables)
    hh_specs, hh_input = audit._hh_table_iii_specs(hh)

    assert table_input["schema"] == "paper_i_tables_i_ii_spsa_optuna_current_best_promotion_v1"
    assert table_specs == [
        {
            "table_label": "tab:fixed_accuracy_claims",
            "table_surface": "tables_i_ii",
            "family": "hubbard",
            "case_id": "hubbard_L2_three_model_weak",
            "regime": "weak",
            "method": "append ADAPT",
            "algorithm_id": "static_full_meta_append_adapt_vqe",
            "source_json": str(source),
            "source_sha256": "abc",
            "visible_one_minus_F": "--",
            "prefix_operator_count": None,
            "source_map_json": str(tables),
        }
    ]
    assert hh_input["schema"] == "paper_i_hh_tableiii_first_effective_plateau_prefix_cost_audit_v1"
    assert hh_specs[0]["case_id"] == "hh_L2_nph2_three_model_sym_weak_weak"
    assert hh_specs[0]["prefix_operator_count"] == 6


def test_one_minus_fidelity_display_threshold():
    assert audit.format_one_minus_fidelity(None) == "--"
    assert audit.format_one_minus_fidelity(9.9e-6) == "0"
    assert audit.format_one_minus_fidelity(4.37e-5) == "4.37e-05"


def test_build_audit_computed_path_is_audit_only(tmp_path: Path, monkeypatch):
    source = tmp_path / "source" / "generic_static_single.json"
    _write_json(source, {"result": {"case_id": "hubbard_L2_three_model_weak"}})
    tables = _write_json(tmp_path / "tables.json", _tables_i_ii_payload(source))
    hh = _write_json(tmp_path / "hh_prefix.json", {"schema": "empty_hh", "rows": []})

    monkeypatch.setattr(
        audit,
        "compute_row_fidelity",
        lambda _row: {
            "one_minus_fidelity": 0.001,
            "fidelity": 0.999,
            "infidelity_source_key": "infidelity_same",
            "reference_kind": "same_cutoff_ed_state",
            "metric_statuses": {"infidelity_same": "ok"},
            "state_replay_source": "unit_test",
        },
    )

    payload = audit.build_audit(
        tables_i_ii_promotion=tables,
        hh_table_iii_prefix_audit=hh,
        hh_table_iii_source_map=None,
        hubbard_snake_audit=None,
        spin_boson_snake_audit=None,
    )

    assert payload["manuscript_edited"] is False
    assert payload["status_counts"] == {"computed": 1}
    assert payload["rows"][0]["one_minus_F_display"] == "0.001"
    assert payload["rows"][0]["safe_for_manuscript_transfer"] is False


def test_blocks_missing_source_json():
    row = audit._audit_one(
        {
            "table_label": "tab:fixed_accuracy_claims",
            "family": "hubbard",
            "case_id": "hubbard_L2_three_model_weak",
            "method": "append ADAPT",
            "algorithm_id": "static_full_meta_append_adapt_vqe",
            "source_json": "does/not/exist.json",
        }
    )

    assert row["status"] == "blocked"
    assert row["blocker"] == "source_json_missing"
    assert row["one_minus_F_display"] == "--"


def test_prefix_replay_blocks_parameter_count_mismatch():
    with pytest.raises(audit.FidelityBlocked) as excinfo:
        audit._prefix_static_row(
            {"selected_operators": ["a", "b"], "theta": [0.1]},
            prefix_count=1,
        )

    assert excinfo.value.status == "not_reconstructable_parameter_count_mismatch"


def test_compute_row_fidelity_uses_locked_projector_inputs(tmp_path: Path):
    source = _write_json(
        tmp_path / "source" / "generic_static_single.json",
        {
            "result": {
                "selected_operators": ["unit_test_generator"],
                "theta": [0.0],
                "ground_space_fidelity_inputs": {
                    "hamiltonian": [[0.0, 0.0], [0.0, 1.0]],
                    "variational_state": [1.0, 0.0],
                    "working_cutoff": 2,
                    "reference_cutoff": 2,
                    "fixed_sector_basis_indices": [0, 1],
                    "legal_binary_basis_indices": [0, 1],
                    "fixed_sector_label": "unit_test_sector",
                    "legal_binary_basis_label": "unit_test_legal_basis",
                },
            }
        },
    )

    result = audit.compute_row_fidelity(
        {
            "source_json": str(source),
            "source_sha256": "not-a-lock-in-this-fixture",
            "prefix_operator_count": None,
        }
    )

    assert result["fidelity"] == pytest.approx(1.0)
    assert result["one_minus_fidelity"] == pytest.approx(0.0)
    assert result["ground_space_fidelity"]["ground_space_multiplicity"] == 1
    assert result["ground_space_fidelity"]["usage_scope"] == "post_run_reporting_only"


def test_prefix_replay_requires_exact_saved_checkpoint_instead_of_truncating_terminal():
    with pytest.raises(audit.FidelityBlocked) as excinfo:
        audit._prefix_static_row(
            {
                "selected_operators": ["a", "b"],
                "theta": [0.1, -0.2],
            },
            prefix_count=1,
        )

    assert excinfo.value.status == "not_reconstructable_missing_exact_prefix_checkpoint"


def test_terminal_replay_prepares_nontrivial_saved_ansatz_not_input_state():
    from pipelines.static_adapt.estimator_call_ledger import (
        projective_state_fingerprint,
    )
    from src.quantum.ansatz_parameterization import build_parameter_layout
    from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
    from src.quantum.pauli_polynomial_class import PauliPolynomial
    from src.quantum.qubitization_module import PauliTerm
    from src.quantum.vqe_latex_python_pairs import AnsatzTerm

    generator = AnsatzTerm(
        label="x",
        polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="x", pc=1.0)]),
    )
    layout = build_parameter_layout([generator], sort_terms=True)
    psi_ref = np.asarray([1.0, 0.0], dtype=complex)
    theta = np.asarray([0.37], dtype=float)
    expected = CompiledAnsatzExecutor(
        [generator],
        parameterization_mode="logical_shared",
        parameterization_layout=layout,
    ).prepare_state(theta, psi_ref)
    runtime_input = SimpleNamespace(
        selected_terms=(generator,),
        base_layout=layout,
        theta_runtime=theta.copy(),
        theta_logical=theta.copy(),
        psi_ref=psi_ref.copy(),
        # Regression guard: the old bug returned this field directly.
        psi_initial=psi_ref.copy(),
    )

    replayed = audit._replay_terminal_saved_ansatz(
        runtime_input,
        {
            "selected_operators": ["x"],
            "theta": theta.tolist(),
            "parameterization_mode": "logical_shared",
            "projective_state_fingerprint": projective_state_fingerprint(expected),
        },
    )

    assert not np.allclose(replayed, runtime_input.psi_initial)
    assert projective_state_fingerprint(replayed) == projective_state_fingerprint(
        expected
    )


def test_generic_checkpoint_terminal_discovery_and_strict_replay():
    from pipelines.exact_bench import generic_static_adapt_variants as variants
    from src.quantum.ansatz_parameterization import build_parameter_layout
    from src.quantum.pauli_polynomial_class import PauliPolynomial
    from src.quantum.qubitization_module import PauliTerm

    candidate = variants._PoolCandidate(
        label="x",
        polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="x", pc=1.0)]),
        support=(0,),
        pauli_labels_exyz=("x",),
        construction="unit_test",
    )
    candidate_z = variants._PoolCandidate(
        label="z",
        polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="z", pc=1.0)]),
        support=(0,),
        pauli_labels_exyz=("z",),
        construction="unit_test",
    )
    theta = np.asarray([0.23], dtype=float)
    theta_terminal = np.asarray([0.23, -0.17], dtype=float)
    psi_ref = np.asarray([1.0, 0.0], dtype=complex)
    layout = build_parameter_layout([candidate], sort_terms=True)
    terminal_layout = build_parameter_layout([candidate, candidate_z], sort_terms=True)
    prepared = variants._prepare_selected_state(
        selected=[candidate],
        theta=theta,
        psi_ref=psi_ref,
        pauli_action_cache={},
        parameterization_mode="logical_shared",
        parameterization_layout=layout,
    )
    checkpoint = variants._paper_i_comparator_active_prefix_checkpoint(
        algorithm_id="static_full_meta_append_adapt_vqe",
        iteration=0,
        selected=[candidate],
        theta=theta,
        parameterization_mode="logical_shared",
        parameterization_layout=layout,
        psi_ref=psi_ref,
        psi_prepared=prepared,
        sector_padding_audit={},
    )
    assert checkpoint["iteration"] == 0
    assert checkpoint["outer_iteration"] == 1
    assert checkpoint["active_ansatz_depth"] == 1
    prepared_terminal = variants._prepare_selected_state(
        selected=[candidate, candidate_z],
        theta=theta_terminal,
        psi_ref=psi_ref,
        pauli_action_cache={},
        parameterization_mode="logical_shared",
        parameterization_layout=terminal_layout,
    )
    terminal_checkpoint = variants._paper_i_comparator_active_prefix_checkpoint(
        algorithm_id="static_full_meta_append_adapt_vqe",
        iteration=1,
        selected=[candidate, candidate_z],
        theta=theta_terminal,
        parameterization_mode="logical_shared",
        parameterization_layout=terminal_layout,
        psi_ref=psi_ref,
        psi_prepared=prepared_terminal,
        sector_padding_audit={},
    )
    assert terminal_checkpoint["iteration"] == 1
    assert terminal_checkpoint["outer_iteration"] == 2
    assert terminal_checkpoint["active_ansatz_depth"] == 2
    result = {
        "selected_operators": ["x", "z"],
        "theta": theta_terminal.tolist(),
        "adapt_history": [
            {"active_prefix_checkpoint": checkpoint},
            {"active_prefix_checkpoint": terminal_checkpoint},
        ],
    }

    terminal = audit._prefix_static_row(result, prefix_count=None)
    selected_prefix = audit._prefix_static_row(result, prefix_count=1)
    assert terminal["_active_prefix_checkpoint"]["checkpoint_sha256"] == (
        terminal_checkpoint["checkpoint_sha256"]
    )
    assert selected_prefix["_active_prefix_checkpoint"]["checkpoint_sha256"] == checkpoint[
        "checkpoint_sha256"
    ]
    replayed = audit._replay_generic_checkpoint(
        SimpleNamespace(psi_ref=psi_ref), checkpoint
    )
    assert np.allclose(replayed, prepared)


def test_prefix_resolution_prioritizes_exact_active_depth_over_round_number():
    depth_match = {
        "outer_iteration": 7,
        "active_ansatz_depth": 1,
        "selected_operators": ["depth_match"],
        "theta": [0.1],
    }
    round_match = {
        "outer_iteration": 1,
        "active_ansatz_depth": 2,
        "selected_operators": ["round_match", "other"],
        "theta": [0.2, 0.3],
    }
    selected = audit._prefix_static_row(
        {
            "selected_operators": ["terminal", "other"],
            "theta": [0.4, 0.5],
            "active_prefix_checkpoints": [round_match, depth_match],
        },
        prefix_count=1,
    )

    assert selected["selected_operators"] == ["depth_match"]
    assert selected["theta"] == [0.1]


def test_actual_shape_terminal_copy_is_collapsed_by_canonical_replay_identity():
    payload = json.loads(
        ACTUAL_SHAPE_DUPLICATE_TERMINAL_FIXTURE.read_text(encoding="utf-8")
    )
    result = audit._result_mapping(payload)
    candidates = audit._checkpoint_candidates(result)
    complete_hashes = {
        audit._checkpoint_identity(checkpoint) for checkpoint in candidates
    }
    canonical_hashes = {
        audit._checkpoint_canonical_copy_identity(checkpoint)
        for checkpoint in candidates
    }

    assert len(complete_hashes) == 2
    assert canonical_hashes == {
        "cdd7f9f25759a98db4cf1201e0979be83213b95106dca5ea004a380a6bd6c1ce"
    }
    selected = audit._prefix_static_row(result, prefix_count=2)
    checkpoint = selected["_active_prefix_checkpoint"]
    assert checkpoint["checkpoint_kind"] == "terminal_post_final_refit_and_prune"
    assert checkpoint["checkpoint_sha256"] == (
        "23fc988713f244f64e6dae7e6561a1be39cdfc084e8496d7faab16535d226cd9"
    )
    assert selected["state_replay_source"] == (
        "exact_terminal_active_prefix_checkpoint"
    )


def test_actual_shape_genuinely_conflicting_terminal_checkpoint_fails_closed():
    payload = json.loads(
        ACTUAL_SHAPE_DUPLICATE_TERMINAL_FIXTURE.read_text(encoding="utf-8")
    )
    conflicting = payload["adapt_vqe"]["terminal_active_prefix_checkpoint"]
    conflicting["projective_state_fingerprint"] = "e" * 64
    _rehash_checkpoint(conflicting)

    with pytest.raises(audit.FidelityBlocked) as excinfo:
        audit._prefix_static_row(audit._result_mapping(payload), prefix_count=2)

    assert excinfo.value.status == "not_reconstructable_ambiguous_terminal_checkpoint"


def test_actual_shape_invalid_duplicate_checkpoint_hash_fails_closed():
    payload = json.loads(
        ACTUAL_SHAPE_DUPLICATE_TERMINAL_FIXTURE.read_text(encoding="utf-8")
    )
    payload["adapt_vqe"]["terminal_active_prefix_checkpoint"][
        "checkpoint_sha256"
    ] = "0" * 64

    with pytest.raises(audit.FidelityBlocked) as excinfo:
        audit._prefix_static_row(audit._result_mapping(payload), prefix_count=2)

    assert excinfo.value.status == "not_reconstructable_checkpoint_sha256_mismatch"


def test_signed_checkpoint_fidelity_replay_repairs_only_execution_order(
    monkeypatch: pytest.MonkeyPatch,
):
    from pipelines.exact_bench.paper_i_hh_recovery_prefix_qiskit_sidecar import (
        CHECKPOINT_SCHEMA,
        _canonical_sha256,
    )
    from pipelines.static_adapt.estimator_call_ledger import (
        projective_state_fingerprint,
    )
    from src.quantum.ansatz_parameterization import deserialize_layout
    from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
    from src.quantum.pauli_polynomial_class import PauliPolynomial
    from src.quantum.qubitization_module import PauliTerm
    from src.quantum.vqe_latex_python_pairs import AnsatzTerm

    expected_terms = [
        {"pauli_exyz": "xx", "coeff_re": 0.5, "coeff_im": 0.0, "nq": 2},
        {"pauli_exyz": "zz", "coeff_re": 0.75, "coeff_im": 0.0, "nq": 2},
    ]
    parameterization = {
        "mode": "per_pauli_term_v1",
        "term_order": "sorted",
        "ignore_identity": True,
        "coefficient_tolerance": 1.0e-12,
        "logical_operator_count": 1,
        "runtime_parameter_count": 2,
        "blocks": [
            {
                "candidate_label": "macro",
                "logical_index": 0,
                "runtime_start": 0,
                "runtime_count": 2,
                "runtime_terms_exyz": expected_terms,
            }
        ],
    }
    layout = deserialize_layout(parameterization)
    generator = AnsatzTerm(
        label="macro",
        polynomial=PauliPolynomial(
            "JW",
            [
                PauliTerm(2, ps=row["pauli_exyz"], pc=row["coeff_re"])
                for row in expected_terms
            ],
        ),
        execution_mode="termwise_product",
    )
    psi_ref = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=complex)
    theta = np.asarray([0.17], dtype=float)
    expected_state = CompiledAnsatzExecutor(
        [generator],
        parameterization_mode="logical_shared",
        parameterization_layout=layout,
    ).prepare_state(theta, psi_ref)
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "checkpoint_kind": "post_admission_prune",
        "outer_iteration": 1,
        "active_ansatz_depth": 1,
        "ordered_active_operator_labels": ["macro"],
        "ordered_active_operators": [
            {
                "active_position": 0,
                "label": "macro",
                "generator_id": "gen:macro",
                "parent_generator_id": None,
                "execution_mode": "termwise_product",
                # Native macro order differs only by permutation from the
                # sorted execution order recorded in the parameter layout.
                "serialized_terms_exyz_in_execution_order": list(
                    reversed(expected_terms)
                ),
                "runtime_split": None,
                "route_a_child_padding_lineage": None,
            }
        ],
        "signed_unwrapped_runtime_parameters": [0.17, 0.17],
        "signed_unwrapped_logical_parameters": [0.17],
        "parameterization_mode": "logical_shared",
        "parameterization": parameterization,
        "projective_state_fingerprint": projective_state_fingerprint(expected_state),
    }
    checkpoint["checkpoint_sha256"] = _canonical_sha256(checkpoint)

    runtime_input = SimpleNamespace(psi_ref=psi_ref)
    monkeypatch.setattr(
        "pipelines.scaffold.runtime_loader.load_scaffold_runtime_input_from_payload",
        lambda *_args, **_kwargs: runtime_input,
    )
    monkeypatch.setattr(
        "pipelines.exact_bench.paper_i_hh_recovery_prefix_qiskit_sidecar.reconstruct_reference_state",
        lambda *_args, **_kwargs: (psi_ref.copy(), {"source": "unit_test"}),
    )
    _loaded, replayed, replay_source = audit._runtime_replayed_state(
        {},
        source_path=Path("unused.json"),
        static_row={"_active_prefix_checkpoint": checkpoint},
    )

    assert replay_source == (
        "exact_signed_prefix_checkpoint_permutation_repaired_replay"
    )
    assert projective_state_fingerprint(replayed) == projective_state_fingerprint(
        expected_state
    )
    assert checkpoint["ordered_active_operators"][0][
        "serialized_terms_exyz_in_execution_order"
    ] == list(reversed(expected_terms))
