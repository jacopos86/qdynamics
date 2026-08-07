from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.promote_locked_imported_scaffold import build_locked_imported_scaffold_payload


def test_build_locked_imported_scaffold_payload_marks_locked_metadata() -> None:
    source_payload = {
        "generated_utc": "2026-03-27T00:00:00+00:00",
        "pipeline": "adapt_vqe",
        "settings": {
            "adapt_pool": "full_meta",
            "L": 2,
            "t": 1.0,
            "u": 0.5,
            "g_ep": 0.2,
            "omega0": 1.0,
            "n_ph_max": 1,
        },
        "adapt_vqe": {
            "pool_type": "phase3_v1",
            "operators": [
                "uccsd_ferm_lifted::uccsd_sing(alpha:0->1)::split[0]::eeeeyx",
                "paop_full:paop_cloud_p(site=1->phonon=0)",
            ],
            "optimal_point": [0.1, -0.2, 0.3],
            "parameterization": {
                "mode": "per_pauli_term_v1",
                "term_order": "sorted",
                "ignore_identity": True,
                "coefficient_tolerance": 1e-12,
                "logical_operator_count": 2,
                "runtime_parameter_count": 3,
                "blocks": [
                    {
                        "candidate_label": "uccsd_ferm_lifted::uccsd_sing(alpha:0->1)::split[0]::eeeeyx",
                        "logical_index": 0,
                        "runtime_start": 0,
                        "runtime_count": 1,
                        "runtime_terms_exyz": [
                            {"pauli_exyz": "eeeeyx", "coeff_re": 0.5, "coeff_im": 0.0, "nq": 6}
                        ],
                    },
                    {
                        "candidate_label": "paop_full:paop_cloud_p(site=1->phonon=0)",
                        "logical_index": 1,
                        "runtime_start": 1,
                        "runtime_count": 2,
                        "runtime_terms_exyz": [
                            {"pauli_exyz": "eyeeze", "coeff_re": -0.5, "coeff_im": 0.0, "nq": 6},
                            {"pauli_exyz": "eyzeee", "coeff_re": -0.5, "coeff_im": 0.0, "nq": 6},
                        ],
                    },
                ],
            },
        },
        "ansatz_input_state": {
            "source": "hf",
            "handoff_state_kind": "reference_state",
            "nq_total": 6,
            "amplitudes_qn_to_q0": {"000000": {"re": 1.0, "im": 0.0}},
            "amplitude_cutoff": 1e-12,
            "norm": 1.0,
        },
    }

    promoted = build_locked_imported_scaffold_payload(
        source_payload,
        source_artifact_json="artifacts/json/source.json",
        subject_kind="hh_promoted_locked_scaffold_v1",
    )

    adapt_vqe = promoted["adapt_vqe"]
    assert promoted["pipeline"] == "hh_promote_locked_imported_scaffold_v1"
    assert promoted["source_artifact_json"] == "artifacts/json/source.json"
    assert promoted["settings"]["adapt_pool"] == "fixed_scaffold_locked"
    assert adapt_vqe["pool_type"] == "fixed_scaffold_locked"
    assert adapt_vqe["structure_locked"] is True
    assert adapt_vqe["fixed_scaffold_kind"] == "hh_promoted_locked_scaffold_v1"
    assert len(adapt_vqe["logical_optimal_point"]) == 2
    assert abs(adapt_vqe["logical_optimal_point"][0] - 0.1) < 1.0e-12
    assert abs(adapt_vqe["logical_optimal_point"][1] - 0.05) < 1.0e-12
    assert adapt_vqe["fixed_scaffold_metadata"]["route_family"] == "locked_imported_scaffold_v1"
    assert adapt_vqe["fixed_scaffold_metadata"]["runtime_term_count"] == 3
    assert adapt_vqe["fixed_scaffold_metadata"]["source_pool_type"] == "full_meta"
    assert adapt_vqe["fixed_scaffold_metadata"]["runtime_term_labels_exyz"] == [
        "eeeeyx",
        "eyeeze",
        "eyzeee",
    ]
    assert adapt_vqe["fixed_scaffold_metadata"]["source_order_runtime_indices"] == [0, 1, 2]
