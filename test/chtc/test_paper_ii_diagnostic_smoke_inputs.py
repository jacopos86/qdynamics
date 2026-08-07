from __future__ import annotations

import csv
import json
from pathlib import Path

from chtc.generic_time_dynamics_table.build_paper_ii_diagnostic_smoke_inputs import (
    CASE_MANIFEST,
    DEFAULT_ALGORITHMS,
    RECORDS_TSV,
    SOURCE_CASE_MANIFEST,
    build_inputs,
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def test_paper_ii_diagnostic_smoke_builder_derives_short_weak_weak_snake_case(tmp_path: Path) -> None:
    source = tmp_path / SOURCE_CASE_MANIFEST
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        json.dumps(
            {
                "schema": "unit_source_manifest",
                "cases": [
                    {
                        "case_id": "table1_hh_weak_weak_snake_A0p2_t8_dt321_seedtracks_v1",
                        "family": "hh",
                        "table_class": "hubbard_holstein",
                        "tuning_class": "hybrid",
                        "artifact_json": "seed_artifacts/hh_weak_weak_snake_seed.json",
                        "t_final": 8.0,
                        "num_times": 321,
                        "metadata": {
                            "same_seed_comparator_group_id": "hh_weak_weak_snake_A0p2_t8_dt321_same_seed_v1",
                            "drive": {"A": 0.2, "enable_drive": True, "time_sampling": "midpoint"},
                            "seed_lock": {
                                "hh_regime_id": "weak_weak",
                                "seed_track": "snake",
                                "same_seed_comparator_group_id": "hh_weak_weak_snake_A0p2_t8_dt321_same_seed_v1",
                                "seed_artifact_sha256": "seed-sha",
                                "latest_phase3_source_artifact_missing_locally": False,
                            },
                        },
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = build_inputs(root=tmp_path)

    assert manifest["schema"] == "paper_ii_diagnostic_smoke_cases_v1"
    assert manifest["paper_facing"] is False
    assert manifest["case_count"] == 1
    assert manifest["record_count"] == len(DEFAULT_ALGORITHMS)
    assert manifest["paper_ii_calibration_gate_status"] == "not_run_diagnostic_only"
    case = manifest["cases"][0]
    metadata = case["metadata"]
    seed_lock = metadata["seed_lock"]
    assert case["case_id"].startswith("diag_hh_weak_weak_snake_A0p2_t0p1_n3")
    assert case["t_final"] == 0.1
    assert case["num_times"] == 3
    assert metadata["diagnostic_only_not_paper_evidence"] is True
    assert metadata["smoke_only_not_paper_evidence"] is True
    assert metadata["diagnostic_source_case_id"] == "table1_hh_weak_weak_snake_A0p2_t8_dt321_seedtracks_v1"
    assert seed_lock["same_seed_comparator_group_id"].startswith("diag_hh_weak_weak_snake_A0p2")
    assert metadata["diagnostic_source_same_seed_comparator_group_id"] != seed_lock["same_seed_comparator_group_id"]
    assert metadata["qiskit_community_dynamics"]["varqrte_max_runtime_parameters"] == 64

    rows = _rows(tmp_path / RECORDS_TSV)
    assert (tmp_path / CASE_MANIFEST).exists()
    assert {row["algorithm_id"] for row in rows} == set(DEFAULT_ALGORITHMS)
    assert {row["visible_table_method"] for row in rows} == {"0"}
    assert {row["diagnostic_only_not_paper_evidence"] for row in rows} == {"1"}
    assert all(row["case_manifest"] == str(CASE_MANIFEST) for row in rows)
