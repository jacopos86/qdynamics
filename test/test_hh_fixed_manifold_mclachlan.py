from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipelines.hardcoded.hh_fixed_manifold_mclachlan as fmm
from pipelines.hardcoded.hh_fixed_manifold_mclachlan import (
    FixedManifoldRunSpec,
    build_fixed_scaffold_context_from_payload,
    load_run_context,
    normalize_replay_payload,
    run_fixed_manifold_exact,
)
from pipelines.hardcoded.hh_vqe_from_adapt_family import RunConfig as ReplayRunConfig
from src.quantum.ansatz_parameterization import build_parameter_layout
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, PauliPolynomial, PauliTerm


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


def _toy_cfg(tmp_path: Path) -> ReplayRunConfig:
    scratch = tmp_path / "scratch"
    return ReplayRunConfig(
        adapt_input_json=tmp_path / "input.json",
        output_json=scratch / "out.json",
        output_csv=scratch / "out.csv",
        output_md=scratch / "out.md",
        output_log=scratch / "out.log",
        tag="toy_fixed",
        generator_family="fixed_scaffold_locked",
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


def test_normalize_replay_payload_lifts_nested_continuation() -> None:
    payload = {
        "adapt_vqe": {
            "continuation": {
                "selected_generator_metadata": [
                    {"candidate_label": "toy_op"},
                ]
            }
        }
    }

    normalized = normalize_replay_payload(payload)

    assert "continuation" not in payload
    assert isinstance(normalized.get("continuation"), dict)
    assert normalized["continuation"]["selected_generator_metadata"][0]["candidate_label"] == "toy_op"


def test_build_fixed_scaffold_context_reconstructs_runtime_terms(tmp_path: Path) -> None:
    payload = _toy_fixed_scaffold_payload()
    cfg = _toy_cfg(tmp_path)

    context = build_fixed_scaffold_context_from_payload(payload, cfg=cfg)

    assert context.family_info["resolved"] == "fixed_scaffold_locked"
    assert int(context.base_layout.logical_parameter_count) == 1
    assert int(context.base_layout.runtime_parameter_count) == 1
    assert len(context.replay_terms) == 1
    assert str(context.replay_terms[0].label) == "toy_x"
    assert np.allclose(context.adapt_theta_runtime, np.array([0.0]))
    assert np.allclose(context.adapt_theta_logical, np.array([0.0]))


def test_load_run_context_replay_family_lifts_continuation_end_to_end(
    monkeypatch,
    tmp_path: Path,
) -> None:
    payload = {
        "settings": {
            "L": 1,
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
            "pool_type": "phase3_v1",
            "operators": ["toy_x"],
            "optimal_point": [0.0],
            "continuation": {
                "selected_generator_metadata": [
                    {"candidate_label": "toy_x", "compile_metadata": {"serialized_terms_exyz": []}}
                ]
            },
        },
        "ansatz_input_state": {
            "source": "hf",
            "nq_total": 1,
            "amplitudes_qn_to_q0": {"0": {"re": 1.0, "im": 0.0}},
            "handoff_state_kind": "reference_state",
        },
        "initial_state": {
            "source": "adapt_vqe",
            "nq_total": 1,
            "amplitudes_qn_to_q0": {"0": {"re": 1.0, "im": 0.0}},
            "handoff_state_kind": "prepared_state",
        },
    }
    artifact_json = tmp_path / "replay.json"
    artifact_json.write_text(json.dumps(payload), encoding="utf-8")

    x_term = AnsatzTerm(
        label="toy_x",
        polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="x", pc=1.0)]),
    )
    layout = build_parameter_layout([x_term], ignore_identity=True, coefficient_tolerance=1.0e-12, sort_terms=True)
    observed: dict[str, object] = {}

    def _fake_build_replay_scaffold_context(cfg, *, h_poly=None, psi_ref=None, payload_in=None):
        observed["payload_in"] = payload_in
        return fmm.ReplayScaffoldContext(
            cfg=cfg,
            h_poly=h_poly,
            psi_ref=np.asarray(psi_ref, dtype=complex),
            payload_in=dict(payload_in),
            family_info={"resolved": "paop_full", "resolution_source": "settings.adapt_pool"},
            family_pool=(x_term,),
            pool_meta={"candidate_pool_complete": True},
            replay_terms=(x_term,),
            base_layout=layout,
            adapt_theta_runtime=np.array([0.0]),
            adapt_theta_logical=np.array([0.0]),
            adapt_depth=1,
            handoff_state_kind="prepared_state",
            provenance_source="explicit",
            family_terms_count=1,
        )

    monkeypatch.setattr(fmm, "_build_hh_hamiltonian", lambda cfg: object())
    monkeypatch.setattr(fmm, "build_replay_scaffold_context", _fake_build_replay_scaffold_context)

    loaded = load_run_context(
        FixedManifoldRunSpec(
            name="toy_replay",
            artifact_json=artifact_json,
            loader_mode="replay_family",
        ),
        tag="pytest_replay",
    )

    assert isinstance(observed["payload_in"], dict)
    assert "continuation" in observed["payload_in"]
    assert loaded.loader_summary["normalized_continuation_lifted"] is True
    assert loaded.loader_summary["fixed_manifold_locked"] is True
    assert len(loaded.replay_context.family_pool) == len(loaded.replay_context.replay_terms) == 1


def test_run_fixed_manifold_exact_stays_for_locked_toy_payload(tmp_path: Path) -> None:
    payload = _toy_fixed_scaffold_payload()
    artifact_json = tmp_path / "toy_locked.json"
    artifact_json.write_text(json.dumps(payload), encoding="utf-8")

    record = run_fixed_manifold_exact(
        FixedManifoldRunSpec(
            name="toy_locked",
            artifact_json=artifact_json,
            loader_mode="fixed_scaffold",
        ),
        tag="pytest_fixed_manifold",
        output_dir=tmp_path / "out",
        t_final=0.2,
        num_times=3,
        miss_threshold=1.0e9,
        gain_ratio_threshold=1.0e-9,
        append_margin_abs=1.0e-12,
    )

    assert record["status"] == "completed"
    assert int(record["summary"]["append_count"]) == 0
    assert int(record["summary"]["stay_count"]) == 3
    assert record["loader"]["fixed_manifold_locked"] is True

    output_json = Path(record["output_json"])
    written = json.loads(output_json.read_text(encoding="utf-8"))
    assert written["pipeline"] == "hh_fixed_manifold_exact_mclachlan_v1"
    assert all(str(row["action_kind"]) == "stay" for row in written["trajectory"])
    assert written["run_config"]["structure_policy"] == "fixed_manifold_locked_pool"


def test_run_fixed_manifold_exact_drive_schema_on_toy_payload(tmp_path: Path) -> None:
    payload = _toy_fixed_scaffold_payload()
    artifact_json = tmp_path / "toy_locked_drive.json"
    artifact_json.write_text(json.dumps(payload), encoding="utf-8")

    record = run_fixed_manifold_exact(
        FixedManifoldRunSpec(
            name="toy_locked_drive",
            artifact_json=artifact_json,
            loader_mode="fixed_scaffold",
        ),
        tag="pytest_fixed_manifold_drive",
        output_dir=tmp_path / "out",
        t_final=0.2,
        num_times=3,
        miss_threshold=1.0e9,
        gain_ratio_threshold=1.0e-9,
        append_margin_abs=1.0e-12,
        enable_drive=True,
        drive_A=0.25,
        drive_omega=1.1,
        drive_tbar=0.8,
        drive_phi=0.4,
        drive_time_sampling="midpoint",
        drive_t0=0.2,
        exact_steps_multiplier=2,
    )

    written = json.loads(Path(record["output_json"]).read_text(encoding="utf-8"))
    assert written["manifest"]["drive_enabled"] is True
    assert written["drive_profile"]["A"] == pytest.approx(0.25)
    assert written["reference"]["kind"] == "driven_piecewise_constant_reference_from_replay_seed"
    assert written["reference"]["reference_method"] == "exponential_midpoint_magnus2_order2"
    assert written["reference"]["projection_time_sampling"] == "midpoint"
    assert written["reference"]["geometry_sample_time_policy"] == "interval_midpoint_plus_t0_with_final_endpoint_fallback"
    assert written["trajectory"][0]["physical_time"] == pytest.approx(0.25)
    assert written["trajectory"][1]["physical_time"] == pytest.approx(0.35)
    assert any(int(row.get("drive_term_count", 0)) >= 1 for row in written["trajectory"])


def test_run_fixed_manifold_exact_drive_a0_matches_static_on_toy_payload(tmp_path: Path) -> None:
    payload = _toy_fixed_scaffold_payload()
    artifact_json = tmp_path / "toy_locked_a0.json"
    artifact_json.write_text(json.dumps(payload), encoding="utf-8")

    static_record = run_fixed_manifold_exact(
        FixedManifoldRunSpec(
            name="toy_locked_static",
            artifact_json=artifact_json,
            loader_mode="fixed_scaffold",
        ),
        tag="pytest_fixed_manifold_static",
        output_dir=tmp_path / "out_static",
        t_final=0.2,
        num_times=3,
        miss_threshold=1.0e9,
        gain_ratio_threshold=1.0e-9,
        append_margin_abs=1.0e-12,
    )
    drive_record = run_fixed_manifold_exact(
        FixedManifoldRunSpec(
            name="toy_locked_drive_a0",
            artifact_json=artifact_json,
            loader_mode="fixed_scaffold",
        ),
        tag="pytest_fixed_manifold_drive_a0",
        output_dir=tmp_path / "out_drive",
        t_final=0.2,
        num_times=3,
        miss_threshold=1.0e9,
        gain_ratio_threshold=1.0e-9,
        append_margin_abs=1.0e-12,
        enable_drive=True,
        drive_A=0.0,
        drive_omega=1.1,
        drive_tbar=0.8,
        drive_phi=0.3,
        drive_time_sampling="midpoint",
        exact_steps_multiplier=2,
    )

    static_payload = json.loads(Path(static_record["output_json"]).read_text(encoding="utf-8"))
    drive_payload = json.loads(Path(drive_record["output_json"]).read_text(encoding="utf-8"))
    assert drive_payload["manifest"]["drive_enabled"] is True
    for row_static, row_drive in zip(static_payload["trajectory"], drive_payload["trajectory"]):
        assert abs(float(row_static["rho_miss"]) - float(row_drive["rho_miss"])) < 1.0e-10
        assert (
            abs(
                float(row_static["baseline_geometry"]["condition_number"])
                - float(row_drive["baseline_geometry"]["condition_number"])
            )
            < 1.0e-10
        )
        assert abs(float(row_static["fidelity_exact"]) - float(row_drive["fidelity_exact"])) < 1.0e-10
        assert abs(float(row_static["abs_energy_total_error"]) - float(row_drive["abs_energy_total_error"])) < 1.0e-10
        assert abs(float(row_static["energy_total_controller"]) - float(row_drive["energy_total_controller"])) < 1.0e-10
        assert abs(float(row_static["energy_total_exact"]) - float(row_drive["energy_total_exact"])) < 1.0e-10


def test_run_fixed_manifold_exact_rejects_corrupted_prepared_state(tmp_path: Path) -> None:
    payload = _toy_fixed_scaffold_payload()
    payload["adapt_vqe"]["optimal_point"] = [0.2]
    payload["adapt_vqe"]["logical_optimal_point"] = [0.2]
    payload["initial_state"]["amplitudes_qn_to_q0"] = {
        "000": {"re": 1.0, "im": 0.0},
    }
    artifact_json = tmp_path / "toy_locked_bad_state.json"
    artifact_json.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Prepared-state reconstruction mismatch"):
        run_fixed_manifold_exact(
            FixedManifoldRunSpec(
                name="toy_locked_bad_state",
                artifact_json=artifact_json,
                loader_mode="fixed_scaffold",
            ),
            tag="pytest_fixed_manifold_bad_state",
            output_dir=tmp_path / "out",
            t_final=0.2,
            num_times=3,
            miss_threshold=1.0e9,
            gain_ratio_threshold=1.0e-9,
            append_margin_abs=1.0e-12,
        )
