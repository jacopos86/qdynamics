from __future__ import annotations

from pathlib import Path
import sys

import json

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs.reports import pdf_utils
from plots.adapt_scaffold.artifact_reader import load_adapt_plot_artifact
from plots.adapt_scaffold.landscape import (
    build_scaffold_trajectory_dataset,
    trajectory_prerequisite_failure_reason,
)
from plots.adapt_scaffold.main import main
from plots.adapt_scaffold.orqviz_adapter import try_write_orqviz_overlay
from plots.adapt_scaffold.renderers import (
    build_generator_usefulness_rows,
    write_energy_vs_iteration_outputs,
    write_generator_usefulness_outputs,
)


def _base_exported_payload() -> dict[str, object]:
    return {
        "generated_utc": "2026-04-16T00:00:00+00:00",
        "pipeline": "adapt_plot_fixture",
        "settings": {
            "L": 1,
            "t": 1.0,
            "u": 2.0,
            "dv": 0.0,
            "omega0": 1.0,
            "g_ep": 0.5,
            "n_ph_max": 1,
            "boson_encoding": "binary",
            "ordering": "blocked",
            "boundary": "open",
        },
        "ground_state": {"exact_energy": -1.6},
        "ansatz_input_state": {
            "source": "hf",
            "handoff_state_kind": "reference_state",
            "nq_total": 3,
            "amplitudes_qn_to_q0": {"000": {"re": 1.0, "im": 0.0}},
        },
        "adapt_vqe": {
            "exact_gs_energy": -1.6,
            "operators": ["g0", "g1"],
            "optimal_point": [0.25, -0.2],
            "parameterization": {
                "mode": "per_pauli_term_v1",
                "term_order": "sorted",
                "ignore_identity": True,
                "coefficient_tolerance": 1e-12,
                "logical_operator_count": 2,
                "runtime_parameter_count": 3,
                "blocks": [
                    {
                        "candidate_label": "g0",
                        "logical_index": 0,
                        "runtime_start": 0,
                        "runtime_count": 1,
                        "runtime_terms_exyz": [
                            {"pauli_exyz": "xee", "coeff_re": 1.0, "coeff_im": 0.0, "nq": 3}
                        ],
                    },
                    {
                        "candidate_label": "g1",
                        "logical_index": 1,
                        "runtime_start": 1,
                        "runtime_count": 2,
                        "runtime_terms_exyz": [
                            {"pauli_exyz": "yee", "coeff_re": 0.5, "coeff_im": 0.0, "nq": 3},
                            {"pauli_exyz": "zee", "coeff_re": -0.5, "coeff_im": 0.0, "nq": 3},
                        ],
                    },
                ],
            },
            "history": [
                {
                    "depth": 1,
                    "selected_op": "g0",
                    "selected_ops": ["g0"],
                    "selection_mode": "singleton",
                    "energy_before_opt": -1.0,
                    "energy_after_opt": -1.3,
                    "delta_energy": -0.3,
                    "delta_abs_drop_from_prev": 0.3,
                    "selected_feature_rows": [
                        {
                            "generator_id": "gen0",
                            "template_id": "tmpl0",
                            "candidate_family": "family_a",
                            "runtime_split_mode": "off",
                            "generator_metadata": {
                                "family_id": "family_a",
                                "support_sites": [0],
                                "support_qubits": [0],
                                "compile_metadata": {
                                    "num_polynomial_terms": 1,
                                    "has_boson_support": False,
                                    "has_fermion_support": True,
                                },
                            },
                        }
                    ],
                },
                {
                    "depth": 2,
                    "selected_ops": ["g1", "g2"],
                    "selection_mode": "batch",
                    "energy_before_opt": -1.3,
                    "energy_after_opt": -1.5,
                    "delta_energy": -0.2,
                    "delta_abs_drop_from_prev": 0.1,
                    "selected_feature_rows": [
                        {
                            "generator_id": "gen1",
                            "template_id": "tmpl1",
                            "candidate_family": "family_b",
                            "runtime_split_mode": "off",
                            "generator_metadata": {
                                "family_id": "family_b",
                                "support_sites": [0],
                                "support_qubits": [1],
                                "compile_metadata": {
                                    "num_polynomial_terms": 2,
                                    "has_boson_support": True,
                                    "has_fermion_support": True,
                                },
                            },
                        },
                        {
                            "generator_id": "gen2",
                            "template_id": "tmpl2",
                            "candidate_family": "family_c",
                            "runtime_split_mode": "off",
                            "generator_metadata": {
                                "family_id": "family_c",
                                "support_sites": [0],
                                "support_qubits": [2],
                                "compile_metadata": {
                                    "num_polynomial_terms": 1,
                                    "has_boson_support": False,
                                    "has_fermion_support": True,
                                },
                            },
                        },
                    ],
                },
            ],
            "continuation": {
                "selected_scaffold_record_chain": [
                    {
                        "step_index": 1,
                        "record_index": 1,
                        "generator_label": "g0",
                        "generator_id": "gen0",
                        "template_id": "tmpl0",
                        "runtime_split_mode": "off",
                    },
                    {
                        "step_index": 2,
                        "record_index": 1,
                        "generator_label": "g1",
                        "generator_id": "gen1",
                        "template_id": "tmpl1",
                        "runtime_split_mode": "off",
                    },
                    {
                        "step_index": 2,
                        "record_index": 2,
                        "generator_label": "g2",
                        "generator_id": "gen2",
                        "template_id": "tmpl2",
                        "runtime_split_mode": "off",
                    },
                ],
                "selected_generator_metadata": [
                    {
                        "generator_id": "gen0",
                        "family_id": "family_a",
                        "template_id": "tmpl0",
                        "candidate_label": "g0",
                        "support_sites": [0],
                        "support_qubits": [0],
                        "compile_metadata": {
                            "num_polynomial_terms": 1,
                            "has_boson_support": False,
                            "has_fermion_support": True,
                            "serialized_terms_exyz": [
                                {"pauli_exyz": "xee", "coeff_re": 1.0, "coeff_im": 0.0, "nq": 3}
                            ],
                        },
                    },
                    {
                        "generator_id": "gen1",
                        "family_id": "family_b",
                        "template_id": "tmpl1",
                        "candidate_label": "g1",
                        "support_sites": [0],
                        "support_qubits": [1],
                        "compile_metadata": {
                            "num_polynomial_terms": 2,
                            "has_boson_support": True,
                            "has_fermion_support": True,
                            "serialized_terms_exyz": [
                                {"pauli_exyz": "yee", "coeff_re": 0.5, "coeff_im": 0.0, "nq": 3},
                                {"pauli_exyz": "zee", "coeff_re": -0.5, "coeff_im": 0.0, "nq": 3},
                            ],
                        },
                    },
                ],
            },
        },
    }


def _write_payload(tmp_path: Path, payload: dict[str, object], *, name: str = "artifact.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_load_exported_artifact_nested_path(tmp_path: Path) -> None:
    artifact = load_adapt_plot_artifact(_write_payload(tmp_path, _base_exported_payload()))
    assert len(artifact.history_steps) == 2
    assert len(artifact.admission_records) == 3
    assert len(artifact.final_scaffold_records) == 2
    assert np.allclose(artifact.final_theta_runtime, [0.25, -0.2, -0.2])
    assert artifact.final_scaffold_records[1].admission_order_index == 2


def test_load_raw_payload_top_level_history_fallback(tmp_path: Path) -> None:
    raw = {
        "settings": {"L": 1},
        "exact_gs_energy": -1.2,
        "operators": ["g0"],
        "optimal_point": [0.1],
        "history": [
            {
                "depth": 1,
                "selected_op": "g0",
                "energy_before_opt": -1.0,
                "energy_after_opt": -1.1,
            }
        ],
    }
    artifact = load_adapt_plot_artifact(_write_payload(tmp_path, raw, name="raw.json"))
    assert artifact.exact_gs_energy == -1.2
    assert artifact.history_steps[0].selected_labels == ("g0",)


def test_equal_split_batch_usefulness_allocation(tmp_path: Path) -> None:
    artifact = load_adapt_plot_artifact(_write_payload(tmp_path, _base_exported_payload()))
    rows = build_generator_usefulness_rows(artifact)
    batch_rows = [row for row in rows if int(row["step_index"]) == 2]
    assert len(batch_rows) == 2
    assert batch_rows[0]["allocated_usefulness"] == pytest.approx(0.05)
    assert batch_rows[1]["allocated_usefulness"] == pytest.approx(0.05)


def test_usefulness_falls_back_to_raw_energy_drop(tmp_path: Path) -> None:
    payload = _base_exported_payload()
    del payload["adapt_vqe"]["history"][1]["delta_abs_drop_from_prev"]  # type: ignore[index]
    del payload["adapt_vqe"]["exact_gs_energy"]  # type: ignore[index]
    del payload["ground_state"]  # type: ignore[arg-type]
    artifact = load_adapt_plot_artifact(_write_payload(tmp_path, payload, name="fallback.json"))
    rows = build_generator_usefulness_rows(artifact)
    batch_rows = [row for row in rows if int(row["step_index"]) == 2]
    assert batch_rows[0]["usefulness_metric"] == "raw_energy_drop"
    assert batch_rows[0]["step_usefulness"] == pytest.approx(0.2)


def test_survivor_flag_marks_pruned_records(tmp_path: Path) -> None:
    artifact = load_adapt_plot_artifact(_write_payload(tmp_path, _base_exported_payload()))
    rows = build_generator_usefulness_rows(artifact)
    survivor_map = {str(row["generator_label"]): row["survives_in_final_scaffold"] for row in rows}
    assert survivor_map["g0"] is True
    assert survivor_map["g1"] is True
    assert survivor_map["g2"] is False


def test_final_scaffold_trajectory_dataset_uses_prefix_runtime_path(tmp_path: Path) -> None:
    artifact = load_adapt_plot_artifact(_write_payload(tmp_path, _base_exported_payload()))
    dataset = build_scaffold_trajectory_dataset(artifact)
    assert dataset.prefix_runtime_points.shape == (3, 3)
    assert np.allclose(dataset.prefix_runtime_points[0], [0.0, 0.0, 0.0])
    assert np.allclose(dataset.prefix_runtime_points[1], [0.25, 0.0, 0.0])
    assert np.allclose(dataset.prefix_runtime_points[2], [0.25, -0.2, -0.2])


def test_landscape_skip_reason_when_parameterization_missing(tmp_path: Path) -> None:
    payload = _base_exported_payload()
    del payload["adapt_vqe"]["parameterization"]  # type: ignore[index]
    artifact = load_adapt_plot_artifact(_write_payload(tmp_path, payload, name="no_layout.json"))
    assert trajectory_prerequisite_failure_reason(artifact) == "missing_parameterization_layout"


def test_selected_scaffold_history_fallback_keeps_per_step_record_indices(tmp_path: Path) -> None:
    payload = _base_exported_payload()
    del payload["adapt_vqe"]["continuation"]["selected_scaffold_record_chain"]  # type: ignore[index]
    payload["adapt_vqe"]["continuation"]["selected_scaffold_history"] = [  # type: ignore[index]
        {
            "step_index": 1,
            "selected_records": [{"generator_label": "g0"}],
        },
        {
            "step_index": 2,
            "selected_records": [{"generator_label": "g1"}, {"generator_label": "g2"}],
        },
    ]
    artifact = load_adapt_plot_artifact(_write_payload(tmp_path, payload, name="history_only.json"))
    assert artifact.admission_records[1].record_index == 1
    assert artifact.admission_records[2].record_index == 2
    assert artifact.admission_records[2].template_id == "tmpl2"


def test_incomplete_matching_marks_survivor_unknown(tmp_path: Path) -> None:
    payload = _base_exported_payload()
    del payload["adapt_vqe"]["parameterization"]  # type: ignore[index]
    payload["adapt_vqe"]["operators"] = ["seed0", "g0", "g1"]  # type: ignore[index]
    payload["adapt_vqe"]["continuation"]["selected_generator_metadata"] = [  # type: ignore[index]
        {
            "generator_id": "seed_gen",
            "family_id": "seed_family",
            "template_id": "seed_tmpl",
            "candidate_label": "seed0",
            "support_sites": [0],
            "support_qubits": [0],
            "compile_metadata": {
                "num_polynomial_terms": 1,
                "has_boson_support": False,
                "has_fermion_support": True,
                "serialized_terms_exyz": [
                    {"pauli_exyz": "xee", "coeff_re": 1.0, "coeff_im": 0.0, "nq": 3}
                ],
            },
        },
        *payload["adapt_vqe"]["continuation"]["selected_generator_metadata"],  # type: ignore[index]
    ]
    artifact = load_adapt_plot_artifact(_write_payload(tmp_path, payload, name="partial_match.json"))
    assert artifact.final_matching_complete is False
    rows = build_generator_usefulness_rows(artifact)
    unknown_row = next(row for row in rows if str(row["generator_label"]) == "g2")
    assert unknown_row["survives_in_final_scaffold"] is None


def test_optional_orqviz_skip_when_dependency_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import importlib

    artifact = load_adapt_plot_artifact(_write_payload(tmp_path, _base_exported_payload()))
    dataset = build_scaffold_trajectory_dataset(artifact)

    class DummyEnergyModel:
        def energy(self, theta_runtime: np.ndarray) -> float:
            return float(np.sum(theta_runtime**2))

    real_import_module = importlib.import_module

    def _fake_import_module(name: str, package: str | None = None):
        if name.startswith("orqviz"):
            raise ModuleNotFoundError("missing test dep")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _fake_import_module)
    result = try_write_orqviz_overlay(
        energy_model=DummyEnergyModel(),  # type: ignore[arg-type]
        dataset=dataset,
        output_png=tmp_path / "scan.png",
        output_json=tmp_path / "scan.json",
    )
    assert result["status"] == "skipped_optional_dependency"
    assert result["reason"] == "orqviz_not_installed"


@pytest.mark.skipif(not pdf_utils.HAS_MATPLOTLIB, reason="matplotlib not available")
def test_renderers_write_outputs(tmp_path: Path) -> None:
    artifact = load_adapt_plot_artifact(_write_payload(tmp_path, _base_exported_payload()))
    out_dir = tmp_path / "out"
    energy_result = write_energy_vs_iteration_outputs(
        artifact,
        output_png=out_dir / "energy.png",
        output_csv=out_dir / "energy.csv",
    )
    ranking_result = write_generator_usefulness_outputs(
        artifact,
        output_png=out_dir / "ranking.png",
        output_csv=out_dir / "ranking.csv",
        top_k=5,
    )
    assert energy_result["status"] == "written"
    assert ranking_result["status"] == "written"
    assert (out_dir / "energy.png").exists()
    assert (out_dir / "energy.csv").exists()
    assert (out_dir / "ranking.png").exists()
    assert (out_dir / "ranking.csv").exists()


@pytest.mark.skipif(not pdf_utils.HAS_MATPLOTLIB, reason="matplotlib not available")
def test_cli_end_to_end(tmp_path: Path) -> None:
    input_json = _write_payload(tmp_path, _base_exported_payload())
    output_dir = tmp_path / "cli"
    rc = main(
        [
            "--input-json",
            str(input_json),
            "--output-dir",
            str(output_dir),
            "--include-landscape",
        ]
    )
    assert rc == 0
    manifest = json.loads((output_dir / "adapt_scaffold_plot_manifest.json").read_text(encoding="utf-8"))
    assert manifest["views"]["energy_vs_iteration"]["status"] == "written"
    assert manifest["views"]["generator_usefulness"]["status"] == "written"
    assert manifest["views"]["final_scaffold_trajectory"]["status"] == "written"
