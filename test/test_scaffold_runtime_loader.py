from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipelines.scaffold.hh_fixed_manifold_loader as scaffold_fixed_loader
import pipelines.scaffold.hh_vqe_from_adapt_family as hh_replay
import pipelines.scaffold.runtime_loader as runtime_loader
import pipelines.time_dynamics.fixed_manifold.mclachlan as fixed_runner
from pipelines.scaffold.hh_vqe_from_adapt_family import (
    ReplayScaffoldContext,
    RunConfig as ReplayRunConfig,
    build_replay_scaffold_context,
)
from pipelines.time_dynamics.fixed_manifold.mclachlan import (
    FixedManifoldRunSpec,
    LoadedRunContext,
)
from src.quantum.ansatz_parameterization import (
    build_parameter_layout,
    serialize_layout,
)
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, PauliPolynomial, PauliTerm


def test_fixed_manifold_loader_names_reexport_from_scaffold_owner() -> None:
    assert fixed_runner.FixedManifoldRunSpec is scaffold_fixed_loader.FixedManifoldRunSpec
    assert fixed_runner.LoadedRunContext is scaffold_fixed_loader.LoadedRunContext
    assert fixed_runner.normalize_replay_payload is scaffold_fixed_loader.normalize_replay_payload
    assert (
        fixed_runner.build_fixed_scaffold_context_from_payload
        is scaffold_fixed_loader.build_fixed_scaffold_context_from_payload
    )


def _toy_fixed_scaffold_payload() -> dict:
    return {
        "pipeline": "toy_fixed_scaffold_export",
        "settings": {
            "L": 1,
            "problem": "hh",
            "t": 1.0,
            "u": 4.0,
            "dv": 0.0,
            "omega0": 1.0,
            "g_ep": 0.5,
            "n_ph_max": 1,
            "boson_encoding": "binary",
            "ordering": "blocked",
            "boundary": "open",
            "adapt_pool": "fixed_scaffold_locked",
        },
        "adapt_vqe": {
            "success": True,
            "pool_type": "fixed_scaffold_locked",
            "method": "toy_fixed_scaffold",
            "num_particles": {"n_up": 1, "n_dn": 0},
            "operators": ["toy_x"],
            "optimal_point": [0.0],
            "logical_optimal_point": [0.0],
            "parameterization": {
                "mode": "per_pauli_term_v1",
                "term_order": "native",
                "ignore_identity": True,
                "coefficient_tolerance": 1.0e-12,
                "logical_operator_count": 1,
                "runtime_parameter_count": 1,
                "blocks": [
                    {
                        "candidate_label": "toy_x",
                        "logical_index": 0,
                        "runtime_start": 0,
                        "runtime_count": 1,
                        "runtime_terms_exyz": [
                            {
                                "pauli_exyz": "eex",
                                "coeff_re": 1.0,
                                "coeff_im": 0.0,
                                "nq": 3,
                            }
                        ],
                    }
                ],
            },
            "structure_locked": True,
            "fixed_scaffold_kind": "toy_locked_v1",
            "fixed_scaffold_metadata": {
                "route_family": "locked_imported_scaffold_v1",
                "subject_kind": "toy_locked_v1",
                "source_artifact_json": "toy_source.json",
            },
        },
        "ansatz_input_state": {
            "source": "hf",
            "nq_total": 3,
            "amplitudes_qn_to_q0": {
                "001": {"re": 1.0, "im": 0.0},
            },
            "handoff_state_kind": "reference_state",
        },
        "initial_state": {
            "source": "fixed_scaffold_vqe",
            "nq_total": 3,
            "amplitudes_qn_to_q0": {
                "001": {"re": 1.0, "im": 0.0},
            },
            "handoff_state_kind": "prepared_state",
        },
    }


def _toy_replay_payload() -> dict:
    return {
        "settings": {
            "L": 1,
            "problem": "hh",
            "t": 1.0,
            "u": 4.0,
            "dv": 0.0,
            "omega0": 1.0,
            "g_ep": 0.5,
            "n_ph_max": 1,
            "boson_encoding": "binary",
            "ordering": "blocked",
            "boundary": "open",
            "adapt_pool": "paop_full",
        },
        "adapt_vqe": {
            "pool_type": "paop_full",
            "operators": ["toy_x"],
            "optimal_point": [0.0],
        },
        "ansatz_input_state": {
            "source": "hf",
            "nq_total": 3,
            "amplitudes_qn_to_q0": {"001": {"re": 1.0, "im": 0.0}},
            "handoff_state_kind": "reference_state",
        },
        "initial_state": {
            "source": "adapt_vqe",
            "nq_total": 3,
            "amplitudes_qn_to_q0": {"001": {"re": 1.0, "im": 0.0}},
            "handoff_state_kind": "prepared_state",
        },
    }


def _toy_cfg(tmp_path: Path) -> ReplayRunConfig:
    scratch = tmp_path / "scratch"
    return ReplayRunConfig(
        adapt_input_json=tmp_path / "input.json",
        output_json=scratch / "out.json",
        output_csv=scratch / "out.csv",
        output_md=scratch / "out.md",
        output_log=scratch / "out.log",
        tag="toy_runtime_loader",
        generator_family="match_adapt",
        fallback_family="full_meta",
        legacy_paop_key="paop_lf_full",
        replay_seed_policy="auto",
        replay_continuation_mode="phase3_v1",
        L=1,
        t=1.0,
        u=4.0,
        dv=0.0,
        omega0=1.0,
        g_ep=0.5,
        n_ph_max=1,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        sector_n_up=1,
        sector_n_dn=0,
        reps=1,
        restarts=1,
        maxiter=1,
        method="POWELL",
        seed=7,
        energy_backend="dense",
        progress_every_s=30.0,
        wallclock_cap_s=3600,
        paop_r=1,
        paop_split_paulis=False,
        paop_prune_eps=0.0,
        paop_normalization="none",
        spsa_a=0.1,
        spsa_c=0.1,
        spsa_alpha=0.602,
        spsa_gamma=0.101,
        spsa_A=10.0,
        spsa_avg_last=1,
        spsa_eval_repeats=1,
        spsa_eval_agg="mean",
        replay_freeze_fraction=0.0,
        replay_unfreeze_fraction=0.0,
        replay_full_fraction=1.0,
        replay_qn_spsa_refresh_every=0,
        replay_qn_spsa_refresh_mode="never",
        phase3_symmetry_mitigation_mode="none",
    )


def _toy_replay_context(
    tmp_path: Path,
    *,
    resolved_family: str,
    candidate_pool_complete: bool,
    selection_mode: str | None = None,
    family_terms_count: int = 1,
) -> ReplayScaffoldContext:
    x_term = AnsatzTerm(
        label="toy_x",
        polynomial=PauliPolynomial("JW", [PauliTerm(3, ps="eex", pc=1.0)]),
    )
    family_pool = [x_term]
    for idx in range(1, int(family_terms_count)):
        family_pool.append(
            AnsatzTerm(
                label=f"toy_extra_{idx}",
                polynomial=PauliPolynomial("JW", [PauliTerm(3, ps="eey", pc=1.0)]),
            )
        )
    layout = build_parameter_layout(
        [x_term],
        ignore_identity=True,
        coefficient_tolerance=1.0e-12,
        sort_terms=True,
    )
    pool_meta = {"candidate_pool_complete": candidate_pool_complete}
    if selection_mode is not None:
        pool_meta["selection_mode"] = selection_mode
    return ReplayScaffoldContext(
        cfg=_toy_cfg(tmp_path),
        h_poly=PauliPolynomial("JW", [PauliTerm(3, ps="ezz", pc=1.0)]),
        psi_ref=np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=complex),
        payload_in=_toy_replay_payload(),
        family_info={
            "requested": resolved_family,
            "resolved": resolved_family,
            "resolution_source": "settings.adapt_pool",
            "fallback_family": "full_meta",
            "fallback_used": False,
            "warning": None,
        },
        family_pool=tuple(family_pool),
        pool_meta=pool_meta,
        replay_terms=(x_term,),
        base_layout=layout,
        adapt_theta_runtime=np.array([0.0]),
        adapt_theta_logical=np.array([0.0]),
        adapt_depth=1,
        handoff_state_kind="prepared_state",
        provenance_source="explicit",
        family_terms_count=len(family_pool),
    )


def _spin_boson_fixture_path() -> Path:
    return REPO_ROOT / "test_support" / "fixtures" / "spin_boson_realtime_seed.json"


def test_load_scaffold_runtime_input_fixed_scaffold_normalizes_locked_payload(
    tmp_path: Path,
) -> None:
    artifact_json = tmp_path / "toy_fixed_scaffold.json"
    artifact_json.write_text(json.dumps(_toy_fixed_scaffold_payload()), encoding="utf-8")

    runtime_input = runtime_loader.load_scaffold_runtime_input(artifact_json)

    assert runtime_input.provenance["loader_mode"] == "fixed_scaffold"
    assert runtime_input.resolved_problem.family_key == "hh"
    assert runtime_input.candidate_pool_source.source_kind == "selected_terms_only"
    assert runtime_input.candidate_pool_source.completeness == "selected_only"
    assert runtime_input.structure_locked is True
    assert runtime_input.can_structural_edit is False
    assert np.allclose(runtime_input.theta_runtime, np.array([0.0]))
    assert np.allclose(runtime_input.theta_logical, np.array([0.0]))
    assert int(runtime_input.base_layout.logical_parameter_count) == 1
    assert int(runtime_input.base_layout.runtime_parameter_count) == 1
    assert len(runtime_input.selected_terms) == 1
    assert len(runtime_input.candidate_pool_terms) == 1


def test_load_scaffold_runtime_input_replay_family_maps_complete_pool(
    monkeypatch,
    tmp_path: Path,
) -> None:
    payload = _toy_replay_payload()
    artifact_json = tmp_path / "toy_replay.json"
    artifact_json.write_text(json.dumps(payload), encoding="utf-8")

    replay_context = _toy_replay_context(
        tmp_path,
        resolved_family="paop_full",
        candidate_pool_complete=True,
    )

    def _fake_load_legacy_loaded_context(
        payload_in,
        *,
        artifact_json: Path,
        loader_mode: str,
        tag: str,
        generator_family: str,
        fallback_family: str,
    ):
        assert loader_mode == "replay_family"
        return LoadedRunContext(
            spec=FixedManifoldRunSpec(
                name=str(Path(artifact_json).stem),
                artifact_json=Path(artifact_json),
                loader_mode=str(loader_mode),
                generator_family=str(generator_family),
                fallback_family=str(fallback_family),
            ),
            cfg=replay_context.cfg,
            payload=dict(payload_in),
            replay_context=replay_context,
            psi_initial=np.asarray(replay_context.psi_ref, dtype=complex).reshape(-1),
            loader_summary={"loader_mode": "replay_family"},
        )

    monkeypatch.setattr(
        runtime_loader,
        "_load_legacy_loaded_context",
        _fake_load_legacy_loaded_context,
    )

    runtime_input = runtime_loader.load_scaffold_runtime_input(artifact_json)

    assert runtime_input.provenance["loader_mode"] == "replay_family"
    assert runtime_input.provenance["resolved_family"] == "paop_full"
    assert runtime_input.candidate_pool_source.source_kind == "resolved_pool"
    assert runtime_input.candidate_pool_source.completeness == "complete"
    assert runtime_input.structure_locked is False
    assert runtime_input.can_structural_edit is True
    assert runtime_input.resolved_problem.family_key == "hh"
    assert len(runtime_input.selected_terms) == 1
    assert len(runtime_input.candidate_pool_terms) == 1


def test_load_scaffold_runtime_input_replay_family_maps_sparse_pool_to_selected_only(
    monkeypatch,
    tmp_path: Path,
) -> None:
    payload = _toy_replay_payload()
    artifact_json = tmp_path / "toy_sparse_replay.json"
    artifact_json.write_text(json.dumps(payload), encoding="utf-8")

    replay_context = _toy_replay_context(
        tmp_path,
        resolved_family="full_meta",
        candidate_pool_complete=False,
        selection_mode="sparse_label_lookup",
        family_terms_count=5,
    )

    def _fake_load_legacy_loaded_context(
        payload_in,
        *,
        artifact_json: Path,
        loader_mode: str,
        tag: str,
        generator_family: str,
        fallback_family: str,
    ):
        return LoadedRunContext(
            spec=FixedManifoldRunSpec(
                name=str(Path(artifact_json).stem),
                artifact_json=Path(artifact_json),
                loader_mode=str(loader_mode),
                generator_family=str(generator_family),
                fallback_family=str(fallback_family),
            ),
            cfg=replay_context.cfg,
            payload=dict(payload_in),
            replay_context=replay_context,
            psi_initial=np.asarray(replay_context.psi_ref, dtype=complex).reshape(-1),
            loader_summary={"loader_mode": "replay_family"},
        )

    monkeypatch.setattr(
        runtime_loader,
        "_load_legacy_loaded_context",
        _fake_load_legacy_loaded_context,
    )

    runtime_input = runtime_loader.load_scaffold_runtime_input(artifact_json)

    assert runtime_input.candidate_pool_source.source_kind == "selected_terms_only"
    assert runtime_input.candidate_pool_source.completeness == "selected_only"
    assert runtime_input.can_structural_edit is False
    assert len(runtime_input.selected_terms) == 1
    assert len(runtime_input.candidate_pool_terms) == 1


def test_replay_context_prefers_serialized_parameterization_for_selected_support(
    monkeypatch,
    tmp_path: Path,
) -> None:
    child_term = AnsatzTerm(
        label="missing_parent::child_set[0]",
        polynomial=PauliPolynomial("JW", [PauliTerm(3, ps="eex", pc=0.5)]),
    )
    layout = build_parameter_layout(
        [child_term],
        ignore_identity=True,
        coefficient_tolerance=1.0e-12,
        sort_terms=True,
    )
    payload = _toy_replay_payload()
    payload["settings"]["adapt_pool"] = "full_meta"
    payload["settings"]["n_ph_max"] = 2
    payload["adapt_vqe"]["pool_type"] = "full_meta"
    payload["adapt_vqe"]["operators"] = ["missing_parent::child_set[0]"]
    payload["adapt_vqe"]["optimal_point"] = [0.25]
    payload["adapt_vqe"]["logical_optimal_point"] = [0.25]
    payload["adapt_vqe"]["parameterization"] = serialize_layout(layout)

    def _fail_sparse_lookup(*args, **kwargs):
        raise AssertionError("serialized layout selected support should not use label lookup")

    monkeypatch.setattr(
        hh_replay,
        "_build_full_meta_replay_terms_sparse",
        _fail_sparse_lookup,
    )
    cfg = replace(
        _toy_cfg(tmp_path),
        generator_family="full_meta",
        n_ph_max=2,
    )

    context = build_replay_scaffold_context(
        cfg,
        h_poly=PauliPolynomial("JW", [PauliTerm(3, ps="ezz", pc=1.0)]),
        psi_ref=np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=complex),
        payload_in=payload,
    )

    assert [term.label for term in context.replay_terms] == ["missing_parent::child_set[0]"]
    assert int(context.base_layout.runtime_parameter_count) == 1
    assert np.allclose(context.adapt_theta_runtime, np.array([0.25]))
    assert context.pool_meta["selection_mode"] == "serialized_parameterization"
    assert context.pool_meta["candidate_pool_complete"] is False


def test_replay_child_set_fallback_preserves_parent_coefficients() -> None:
    parent = AnsatzTerm(
        label="parent",
        polynomial=PauliPolynomial(
            "JW",
            [
                PauliTerm(3, ps="eex", pc=0.25),
                PauliTerm(3, ps="eey", pc=-0.75),
            ],
        ),
    )
    payload = {
        "result": {
            "pool_pauli_labels_exyz": {
                "parent::child_set[0]": ["eey"],
            }
        }
    }

    terms = hh_replay._build_replay_terms_from_adapt_labels(
        [parent],
        ["parent::child_set[0]"],
        payload=payload,
    )

    assert [term.label for term in terms] == ["parent::child_set[0]"]
    poly_terms = terms[0].polynomial.return_polynomial()
    assert [term.pw2strng() for term in poly_terms] == ["eey"]
    assert np.allclose([complex(term.p_coeff) for term in poly_terms], [-0.75])


def test_load_scaffold_runtime_input_replay_family_diagnostic_pool_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    payload = _toy_replay_payload()
    payload["replay_candidate_pool_mode"] = "diagnostic_replay_family_pool"
    artifact_json = tmp_path / "toy_sparse_replay_diagnostic_pool.json"
    artifact_json.write_text(json.dumps(payload), encoding="utf-8")

    replay_context = _toy_replay_context(
        tmp_path,
        resolved_family="full_meta",
        candidate_pool_complete=False,
        selection_mode="sparse_label_lookup",
        family_terms_count=3,
    )

    def _fake_load_legacy_loaded_context(
        payload_in,
        *,
        artifact_json: Path,
        loader_mode: str,
        tag: str,
        generator_family: str,
        fallback_family: str,
    ):
        return LoadedRunContext(
            spec=FixedManifoldRunSpec(
                name=str(Path(artifact_json).stem),
                artifact_json=Path(artifact_json),
                loader_mode=str(loader_mode),
                generator_family=str(generator_family),
                fallback_family=str(fallback_family),
            ),
            cfg=replay_context.cfg,
            payload=dict(payload_in),
            replay_context=replay_context,
            psi_initial=np.asarray(replay_context.psi_ref, dtype=complex).reshape(-1),
            loader_summary={"loader_mode": "replay_family"},
        )

    monkeypatch.setattr(
        runtime_loader,
        "_load_legacy_loaded_context",
        _fake_load_legacy_loaded_context,
    )

    runtime_input = runtime_loader.load_scaffold_runtime_input(artifact_json)

    assert runtime_input.candidate_pool_source.source_kind == "resolved_pool"
    assert runtime_input.candidate_pool_source.completeness == "complete"
    assert runtime_input.can_structural_edit is True
    assert len(runtime_input.selected_terms) == 1
    assert [term.label for term in runtime_input.candidate_pool_terms] == [
        "toy_x",
        "toy_extra_1",
        "toy_extra_2",
    ]
    override = runtime_input.candidate_pool_source.filter_payload[
        "diagnostic_append_pool_override"
    ]
    assert override["enabled"] is True
    assert override["source"] == "replay_context.family_pool"
    assert override["original_candidate_pool_complete"] is False
    assert override["original_selection_mode"] == "sparse_label_lookup"
    assert override["selected_term_count"] == 1
    assert override["family_pool_term_count"] == 3


def test_diagnostic_replay_family_pool_applies_legal_subspace_hard_guard(
    monkeypatch,
    tmp_path: Path,
) -> None:
    payload = _toy_replay_payload()
    payload["replay_candidate_pool_mode"] = "diagnostic_replay_family_pool"
    payload["adapt_vqe"]["adapt_pool_legal_subspace_filter"] = {
        "active": True,
        "offender_labels": [
            {"label": "toy_extra_1", "action": "dropped"},
            {"label": "toy_extra_2", "action": "kept_with_component_risk"},
            {"label": "toy_extra_3", "action": "dropped"},
        ],
        "component_risk_labels": [
            {
                "label": "toy_extra_2",
                "action": "kept_with_component_risk",
                "termwise_component_leaking_term_count": 1,
            },
        ],
    }
    artifact_json = tmp_path / "toy_sparse_replay_diagnostic_pool_guarded.json"
    artifact_json.write_text(json.dumps(payload), encoding="utf-8")

    replay_context = _toy_replay_context(
        tmp_path,
        resolved_family="full_meta",
        candidate_pool_complete=False,
        selection_mode="sparse_label_lookup",
        family_terms_count=4,
    )

    def _fake_load_legacy_loaded_context(
        payload_in,
        *,
        artifact_json: Path,
        loader_mode: str,
        tag: str,
        generator_family: str,
        fallback_family: str,
    ):
        return LoadedRunContext(
            spec=FixedManifoldRunSpec(
                name=str(Path(artifact_json).stem),
                artifact_json=Path(artifact_json),
                loader_mode=str(loader_mode),
                generator_family=str(generator_family),
                fallback_family=str(fallback_family),
            ),
            cfg=replay_context.cfg,
            payload=dict(payload_in),
            replay_context=replay_context,
            psi_initial=np.asarray(replay_context.psi_ref, dtype=complex).reshape(-1),
            loader_summary={"loader_mode": "replay_family"},
        )

    monkeypatch.setattr(
        runtime_loader,
        "_load_legacy_loaded_context",
        _fake_load_legacy_loaded_context,
    )

    runtime_input = runtime_loader.load_scaffold_runtime_input(artifact_json)

    assert runtime_input.candidate_pool_source.source_kind == "resolved_pool"
    assert runtime_input.candidate_pool_source.completeness == "complete"
    assert [term.label for term in runtime_input.candidate_pool_terms] == [
        "toy_x",
        "toy_extra_2",
    ]
    guard = runtime_input.candidate_pool_source.filter_payload[
        "legal_subspace_append_guard"
    ]
    assert guard["enabled"] is True
    assert guard["dropped_candidate_count"] == 2
    assert guard["candidate_pool_count_before"] == 4
    assert guard["candidate_pool_count_after"] == 2
    assert guard["dropped_labels_sample"] == ["toy_extra_1", "toy_extra_3"]
    assert guard["no_pauli_split_parent_labels"] == ["toy_extra_2"]


def test_load_scaffold_runtime_input_spin_boson_fixture_maps_generic_runtime_contract() -> None:
    runtime_input = runtime_loader.load_scaffold_runtime_input(_spin_boson_fixture_path())

    assert runtime_input.provenance["loader_mode"] == "replay_family"
    assert runtime_input.provenance["resolved_family"] == "full_meta"
    assert runtime_input.resolved_problem.family_key == "spin_boson"
    assert runtime_input.candidate_pool_source.source_kind == "resolved_pool"
    assert runtime_input.candidate_pool_source.completeness == "complete"
    assert runtime_input.structure_locked is False
    assert runtime_input.can_structural_edit is True
    assert int(runtime_input.base_layout.logical_parameter_count) == 2
    assert int(runtime_input.base_layout.runtime_parameter_count) == 3
    assert [term.label for term in runtime_input.selected_terms] == [
        "full_meta::boson_displacement",
        "full_meta::transverse_x",
    ]
    assert len(runtime_input.candidate_pool_terms) >= 2
    assert np.allclose(runtime_input.theta_runtime, np.array([0.05, -0.11, 0.07]))
    assert np.allclose(runtime_input.theta_logical, np.array([0.05, -0.02]))
    assert runtime_input.extensions["generic_loader_summary"]["initial_state_source"] == "payload"


def test_generic_runtime_loader_rejects_prepared_state_parity_mismatch(tmp_path: Path) -> None:
    payload = json.loads(_spin_boson_fixture_path().read_text(encoding="utf-8"))
    payload["initial_state"]["amplitudes_qn_to_q0"] = {
        "000": {"re": 1.0, "im": 0.0},
    }
    artifact_json = tmp_path / "spin_boson_bad_initial_state.json"
    artifact_json.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Prepared-state parity check failed"):
        runtime_loader.load_scaffold_runtime_input(artifact_json)


def test_legacy_runtime_loader_rejects_prepared_state_parity_mismatch(tmp_path: Path) -> None:
    replay_context = _toy_replay_context(
        tmp_path,
        resolved_family="paop_full",
        candidate_pool_complete=True,
    )
    payload = _toy_replay_payload()
    payload["initial_state"]["amplitudes_qn_to_q0"] = {
        "000": {"re": 1.0, "im": 0.0},
    }

    with pytest.raises(ValueError, match="fallback to reconstructed"):
        runtime_loader._resolve_prepared_state(payload, replay_context)
