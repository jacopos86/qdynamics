from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chtc.phase3_optuna import generate_paper_i_comparator_spsa_calibration_records as generator
from chtc.phase3_optuna import preflight_submit, run_task
from pipelines.exact_bench.paper_i_comparator_spsa_calibration import (
    PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID,
)

SMOKE_SUBMIT = REPO_ROOT / "chtc" / "phase3_optuna" / "submit_paper_i_comparator_spsa_calibration_v1_smoke.sub"
FULL_SUBMIT = REPO_ROOT / "chtc" / "phase3_optuna" / "submit_paper_i_comparator_spsa_calibration_v1_full.sub"
REPAIR_FULL_SUBMIT = REPO_ROOT / "chtc" / "phase3_optuna" / "submit_paper_i_hh_geo_qeb_spsa_repair_v1_full.sub"
SMOKE_RECORDS = (
    REPO_ROOT
    / "chtc"
    / "phase3_optuna"
    / "input"
    / "paper_i_comparator_spsa_calibration_v1_smoke"
    / generator.SMOKE_RECORDS_TSV
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{str(k): "" if v is None else str(v) for k, v in row.items()} for row in csv.DictReader(handle, delimiter="\t")]


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _tmp_repo_with_one_record(
    tmp_path: Path,
    *,
    mutate: Callable[[dict[str, str]], None] | None = None,
    submit_name: str = "submit_smoke.sub",
) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    config = root / "chtc" / "phase3_optuna" / "config" / "paper_i_comparator_spsa_calibration_v1_smoke.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        (REPO_ROOT / "chtc" / "phase3_optuna" / "config" / "paper_i_comparator_spsa_calibration_v1_smoke.json").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    plan = root / "docs" / "plans" / "paper-i-comparator-spsa-optuna-calibration-2026-05-31.md"
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text("# plan fixture\n", encoding="utf-8")
    phase3 = root / "chtc" / "phase3_optuna"
    for name in (
        "requirements-chtc.txt",
        "run_paper_i_comparator_spsa_calibration_task.sh",
        "run_paper_i_comparator_spsa_calibration_task_apptainer.sh",
    ):
        (phase3 / name).write_text((REPO_ROOT / "chtc" / "phase3_optuna" / name).read_text(encoding="utf-8"), encoding="utf-8")
    input_dir = phase3 / "input" / "paper_i_comparator_spsa_calibration_v1_smoke"
    generator.generate_records(output_dir=input_dir, config_path=config, generation_mode="smoke")
    records = input_dir / generator.SMOKE_RECORDS_TSV
    ids = input_dir / generator.SMOKE_RECORD_IDS_TXT
    rows = [_rows(records)[0]]
    if mutate is not None:
        mutate(rows[0])
    _write_rows(records, rows)
    ids.write_text(rows[0]["record_id"] + "\n", encoding="utf-8")
    submit = phase3 / submit_name
    records_rel = records.relative_to(root)
    ids_rel = ids.relative_to(root)
    submit.write_text(
        "\n".join(
            [
                "universe = vanilla",
                "executable = chtc/phase3_optuna/run_paper_i_comparator_spsa_calibration_task_apptainer.sh",
                f"arguments = $(record_id) {records_rel} raw_outputs/paper_i_comparator_spsa_calibration_v1/records/$(record_id) {rows[0]['config_path']}",
                "transfer_input_files = pipelines, src, docs, test_support, chtc/phase3_optuna",
                "requirements = TARGET.HasSIF",
                f"queue record_id from {ids_rel}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return root, submit


def test_calibration_submit_templates_are_manual_one_record_surfaces() -> None:
    smoke = SMOKE_SUBMIT.read_text(encoding="utf-8")
    full = FULL_SUBMIT.read_text(encoding="utf-8")
    shell = (REPO_ROOT / "chtc" / "phase3_optuna" / "run_paper_i_comparator_spsa_calibration_task.sh").read_text(
        encoding="utf-8"
    )
    apptainer = (
        REPO_ROOT / "chtc" / "phase3_optuna" / "run_paper_i_comparator_spsa_calibration_task_apptainer.sh"
    ).read_text(encoding="utf-8")

    for text in (smoke, full):
        assert "run_paper_i_comparator_spsa_calibration_task_apptainer.sh" in text
        assert "paper_i_comparator_spsa_calibration" in text
        assert "queue record_id from" in text
        assert "condor_submit" not in text
    assert generator.SMOKE_RECORD_IDS_TXT in smoke
    assert generator.RECORD_IDS_TXT in full
    assert "paper_i_comparator_spsa_calibration_runner" in shell
    assert "optuna" in shell
    assert "qiskit_algorithms" in shell
    assert "run_paper_i_comparator_spsa_calibration_task.sh" in apptainer


def test_smoke_submit_preflight_validates_generated_calibration_records() -> None:
    payload = preflight_submit.build_preflight_bundle(submit_path=SMOKE_SUBMIT, repo_root=REPO_ROOT)

    assert payload["status"] == "pass", payload
    assert payload["record_count"] == 6
    assert payload["failed_record_count"] == 0
    assert all(record["schema"] == "paper_i_comparator_spsa_calibration_chtc_preflight_manifest_v1" for record in payload["records"])
    assert all(record["evidence_role"] == "calibration_only_not_manuscript_table_evidence" for record in payload["records"])
    first = payload["records"][0]
    assert first["calibration_record"]["n_jobs"] == "1"
    assert first["current_best_expectation"]["progress_current_best_json"].endswith("/progress/current_best.json")
    assert all(check["ok"] for check in first["dependency_checks"])


def test_preflight_accepts_visible_warm_start_schedule_lock(tmp_path: Path) -> None:
    root, submit = _tmp_repo_with_one_record(tmp_path)
    records = (
        root
        / "chtc"
        / "phase3_optuna"
        / "input"
        / "paper_i_comparator_spsa_calibration_v1_smoke"
        / generator.SMOKE_RECORDS_TSV
    )
    rows = _rows(records)
    row = rows[0]
    config = json.loads(Path(row["config_path"]).read_text(encoding="utf-8"))
    method_space = config["per_method_search_spaces"][row["method_id"]]
    schedule = {
        str(name): spec["choices"][0] if spec["type"] == "choice" else spec["low"]
        for name, spec in method_space.items()
    }
    lock_rel = "chtc/phase3_optuna/input/paper_i_comparator_spsa_calibration_v1_smoke/warm_start_schedule_lock.json"
    lock_path = root / lock_rel
    key = f"{row['method_id']}::{row['target_id']}"
    lock_path.write_text(
        json.dumps(
            {
                "schema": "paper_i_comparator_spsa_schedule_lock_candidate_v1",
                "method_target_schedules": {
                    key: {"method_id": row["method_id"], "target_id": row["target_id"], "schedule": schedule}
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    row["warm_start_schedule_lock_json"] = lock_rel
    row["warm_start_schedule_key"] = key
    _write_rows(records, rows)

    payload = preflight_submit.build_preflight_bundle(submit_path=submit, repo_root=root)
    first = payload["records"][0]
    warm_artifact = next(item for item in first["source_artifacts"] if item["field"] == "warm_start_schedule_lock_json")

    assert payload["status"] == "pass", payload
    assert first["calibration_record"]["warm_start_schedule_key"] == key
    assert first["calibration_record"]["warm_start_schedule_fields"] == sorted(schedule)
    assert warm_artifact["exists"] is True
    assert warm_artifact["sandbox_visible"] is True


def test_hh_geo_qeb_repair_full_submit_preflight_validates_exact_eight_row_matrix() -> None:
    payload = preflight_submit.build_preflight_bundle(submit_path=REPAIR_FULL_SUBMIT, repo_root=REPO_ROOT)

    assert payload["status"] == "pass", payload
    assert payload["record_count"] == 8
    assert payload["failed_record_count"] == 0
    repair_records = [record["calibration_record"] for record in payload["records"]]
    assert {record["method_id"] for record in repair_records} == set(generator.HH_TABLEIII_REPAIR_METHOD_IDS)
    assert {record["target_id"] for record in repair_records} == set(generator.HH_TABLEIII_REPAIR_TARGET_IDS)
    assert {record["repair_scope"] for record in repair_records} == {generator.HH_GEO_QEB_TABLEIII_REPAIR_SCOPE}
    assert {record["table_label"] for record in repair_records} == {"tab:fixed_accuracy_hh_cartesian"}
    assert {record["calibration_usable_status_policy"] for record in repair_records} == {
        "finite_metrics_allow_quality_nonpassing_v1"
    }


def test_preflight_rejects_bad_n_jobs_and_config_hash(tmp_path: Path) -> None:
    def mutate(row: dict[str, str]) -> None:
        row["n_jobs"] = "2"
        row["config_sha256"] = "0" * 64

    root, submit = _tmp_repo_with_one_record(tmp_path, mutate=mutate)
    payload = preflight_submit.build_preflight_bundle(submit_path=submit, repo_root=root)
    reasons = "\n".join(payload["blocking_reasons"])

    assert payload["status"] == "fail"
    assert "calibration_n_jobs_must_be_1" in reasons
    assert "config_sha256_mismatch" in reasons


def test_preflight_rejects_unapproved_smoke_config_for_full_queue(tmp_path: Path) -> None:
    def mutate(row: dict[str, str]) -> None:
        row["run_class"] = "calibration_candidate_not_table_evidence"

    root, submit = _tmp_repo_with_one_record(tmp_path, mutate=mutate, submit_name="submit_full.sub")
    payload = preflight_submit.build_preflight_bundle(submit_path=submit, repo_root=root)
    reasons = "\n".join(payload["blocking_reasons"])

    assert payload["status"] == "fail"
    assert "full_queue_requires_approved_full_config" in reasons


def test_preflight_rejects_mismatched_submit_output_root(tmp_path: Path) -> None:
    root, submit = _tmp_repo_with_one_record(tmp_path)
    text = submit.read_text(encoding="utf-8")
    submit.write_text(text.replace("raw_outputs/paper_i_comparator_spsa_calibration_v1/records/$(record_id)", "raw_outputs/wrong/$(record_id)"), encoding="utf-8")

    payload = preflight_submit.build_preflight_bundle(submit_path=submit, repo_root=root)
    reasons = "\n".join(payload["blocking_reasons"])

    assert payload["status"] == "fail"
    assert "calibration_output_root_mismatch" in reasons


def test_submit_contract_parser_exposes_calibration_executable_and_records_path() -> None:
    contract = run_task.parse_submit_contract(SMOKE_SUBMIT)

    assert contract["executable"] == "chtc/phase3_optuna/run_paper_i_comparator_spsa_calibration_task_apptainer.sh"
    assert contract["argument_records_path"].endswith(generator.SMOKE_RECORDS_TSV)
    assert contract["queue_record_id_file"].endswith(generator.SMOKE_RECORD_IDS_TXT)
    assert contract["requirements"] == "TARGET.HasSIF"
    assert contract["max_runtime"] == "7200"
    assert PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID in SMOKE_RECORDS.read_text(encoding="utf-8")
