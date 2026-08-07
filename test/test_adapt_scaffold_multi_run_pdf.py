from __future__ import annotations

from pathlib import Path
import sys

import json

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs.reports import pdf_utils
import plots.adapt_scaffold.multi_run_pdf as multi_run_pdf
from plots.adapt_scaffold.multi_run_pdf import ReportRunTarget, build_multi_run_pdf, load_report_runs

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency in test env
    PdfReader = None


def _payload(label: str, energy_after: float, gap: float, *, with_parameterization: bool = True) -> dict[str, object]:
    payload: dict[str, object] = {
        "generated_utc": "2026-04-16T00:00:00+00:00",
        "pipeline": "adapt_pdf_fixture",
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
        "ground_state": {"exact_energy": energy_after - gap},
        "ansatz_input_state": {
            "source": "hf",
            "handoff_state_kind": "reference_state",
            "nq_total": 3,
            "amplitudes_qn_to_q0": {"000": {"re": 1.0, "im": 0.0}},
        },
        "adapt_vqe": {
            "energy": energy_after,
            "abs_delta_e": abs(gap),
            "ansatz_depth": 2,
            "num_parameters": 3,
            "logical_num_parameters": 2,
            "operators": [f"{label}_g0", f"{label}_g1"],
            "optimal_point": [0.25, -0.2, -0.2],
            "parameterization": {
                "mode": "per_pauli_term_v1",
                "term_order": "sorted",
                "ignore_identity": True,
                "coefficient_tolerance": 1e-12,
                "logical_operator_count": 2,
                "runtime_parameter_count": 3,
                "blocks": [
                    {
                        "candidate_label": f"{label}_g0",
                        "logical_index": 0,
                        "runtime_start": 0,
                        "runtime_count": 1,
                        "runtime_terms_exyz": [
                            {"pauli_exyz": "xee", "coeff_re": 1.0, "coeff_im": 0.0, "nq": 3}
                        ],
                    },
                    {
                        "candidate_label": f"{label}_g1",
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
                    "selected_op": f"{label}_g0",
                    "selected_ops": [f"{label}_g0"],
                    "selection_mode": "singleton",
                    "energy_before_opt": -1.0,
                    "energy_after_opt": -1.3,
                    "delta_abs_drop_from_prev": 0.3,
                    "selected_feature_rows": [
                        {
                            "generator_id": f"{label}_gen0",
                            "template_id": "tmpl0",
                            "candidate_family": "family_a",
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
                    "selected_ops": [f"{label}_g1", f"{label}_g2"],
                    "selection_mode": "batch",
                    "energy_before_opt": -1.3,
                    "energy_after_opt": energy_after,
                    "delta_abs_drop_from_prev": 0.2,
                    "selected_feature_rows": [
                        {
                            "generator_id": f"{label}_gen1",
                            "template_id": "tmpl1",
                            "candidate_family": "family_b",
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
                            "generator_id": f"{label}_gen2",
                            "template_id": "tmpl2",
                            "candidate_family": "family_c",
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
                        "generator_label": f"{label}_g0",
                        "generator_id": f"{label}_gen0",
                        "template_id": "tmpl0",
                        "runtime_split_mode": "off",
                    },
                    {
                        "step_index": 2,
                        "record_index": 1,
                        "generator_label": f"{label}_g1",
                        "generator_id": f"{label}_gen1",
                        "template_id": "tmpl1",
                        "runtime_split_mode": "off",
                    },
                    {
                        "step_index": 2,
                        "record_index": 2,
                        "generator_label": f"{label}_g2",
                        "generator_id": f"{label}_gen2",
                        "template_id": "tmpl2",
                        "runtime_split_mode": "off",
                    },
                ],
                "selected_generator_metadata": [
                    {
                        "generator_id": f"{label}_gen0",
                        "family_id": "family_a",
                        "template_id": "tmpl0",
                        "candidate_label": f"{label}_g0",
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
                        "generator_id": f"{label}_gen1",
                        "family_id": "family_b",
                        "template_id": "tmpl1",
                        "candidate_label": f"{label}_g1",
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
                    {
                        "generator_id": f"{label}_gen2",
                        "family_id": "family_c",
                        "template_id": "tmpl2",
                        "candidate_label": f"{label}_g2",
                        "support_sites": [0],
                        "support_qubits": [2],
                        "compile_metadata": {
                            "num_polynomial_terms": 1,
                            "has_boson_support": False,
                            "has_fermion_support": True,
                            "serialized_terms_exyz": [
                                {"pauli_exyz": "eex", "coeff_re": 1.0, "coeff_im": 0.0, "nq": 3}
                            ],
                        },
                    },
                ],
            },
        },
    }
    if not with_parameterization:
        del payload["adapt_vqe"]["parameterization"]  # type: ignore[index]
        del payload["adapt_vqe"]["optimal_point"]  # type: ignore[index]
    return payload


@pytest.mark.skipif(not pdf_utils.HAS_MATPLOTLIB, reason="matplotlib not available")
def test_multi_run_pdf_smoke_and_page_count(tmp_path: Path) -> None:
    p1 = tmp_path / "run1.json"
    p2 = tmp_path / "run2.json"
    p1.write_text(json.dumps(_payload("r1", -1.5, 0.01, with_parameterization=True), indent=2), encoding="utf-8")
    p2.write_text(json.dumps(_payload("r2", -1.45, 0.02, with_parameterization=False), indent=2), encoding="utf-8")

    runs = load_report_runs(
        [
            ReportRunTarget(label="run1", json_path=str(p1), math_reference="test"),
            ReportRunTarget(label="run2", json_path=str(p2), math_reference="test"),
        ]
    )
    out = build_multi_run_pdf(runs=runs, output_pdf=tmp_path / "report.pdf")
    assert out.exists()
    assert out.stat().st_size > 0
    if PdfReader is not None:
        assert len(PdfReader(str(out)).pages) == 4 + 2 * len(runs)


def test_shared_step_drop_and_symbolic_catalog(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text(json.dumps(_payload("sym", -1.5, 0.01, with_parameterization=True), indent=2), encoding="utf-8")
    run = load_report_runs([ReportRunTarget(label="sym", json_path=str(path), math_reference="test")])[0]

    step_rows = multi_run_pdf._build_step_rows(run.artifact)
    assert len(step_rows) == 2
    assert step_rows[0].shared_step_drop == pytest.approx(0.3)
    assert step_rows[1].shared_step_drop == pytest.approx(0.2)

    context = multi_run_pdf._build_run_symbolic_context(run.artifact)
    batch_points = [point for point in context.admission_points if point.step_index == 2]
    assert len(batch_points) == 2
    assert all(point.shared_step_drop == pytest.approx(0.2) for point in batch_points)
    assert any("Y_{2}" in entry.display_math_body for entry in context.type_entries)
    assert any("X_{0}" in entry.display_math_body for entry in context.type_entries)
    assert all("sym_g" not in entry.display_math_body for entry in context.type_entries)


def test_repo_relative_resolution_uses_repo_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    rel1 = Path("fixtures") / "run1.json"
    rel2 = Path("fixtures") / "run2.json"
    (fixture_dir / "run1.json").write_text(json.dumps(_payload("rr1", -1.5, 0.01), indent=2), encoding="utf-8")
    (fixture_dir / "run2.json").write_text(json.dumps(_payload("rr2", -1.4, 0.02), indent=2), encoding="utf-8")

    monkeypatch.setattr(multi_run_pdf, "REPO_ROOT", tmp_path)
    runs = load_report_runs(
        [
            ReportRunTarget(label="rr1", json_path=str(rel1), math_reference="test"),
            ReportRunTarget(label="rr2", json_path=str(rel2), math_reference="test"),
        ]
    )
    assert len(runs) == 2
    assert runs[0].resolved_json_path == (tmp_path / rel1).resolve()
