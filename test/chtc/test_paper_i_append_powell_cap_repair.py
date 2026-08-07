from __future__ import annotations

import csv
import json
from pathlib import Path

from chtc.phase3_optuna import prepare_paper_i_append_powell_cap_repair as repair
from chtc.phase3_optuna import preflight_submit
from chtc.phase3_optuna import run_paper_i_scaling_matrix_cell as runner


def test_single_append_powell_cap_repair_is_audited_and_prepared_only(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / repair.REPAIR_BATCH_ID
    submit_path = tmp_path / f"submit_{repair.REPAIR_BATCH_ID}.sub"
    manifest = repair.prepare(
        output_dir=output_dir,
        submit_path=submit_path,
    )

    with (output_dir / "paper_i_scaling_matrix_records.tsv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 1
    row = rows[0]
    runner.validate_record(row)
    assert row["record_id"] == repair.REPAIR_RECORD_ID
    assert row["family"] == "hubbard"
    assert row["case_id"] == "hubbard_L4_scaling_strong"
    assert row["method_key"] == "append"
    assert row["adapt_optimizer_kind"] == "powell"
    assert row["budget"] == "200"
    assert row["expected_horizon"] == "30"
    assert row["powell_maxiter_cap_policy"] == "accept_finite_nonincreasing_v1"

    repair_audit = json.loads(
        (output_dir / "implementation_repair_audit.json").read_text(encoding="utf-8")
    )
    assert repair_audit["status"] == "approved_implementation_repair_prepared"
    assert repair_audit["classification"] == "implementation_repair_not_sensitivity_sweep"
    assert repair_audit["source_locked_sensitivity_claim"] is False
    assert repair_audit["source"]["cluster"] == 8772847
    assert repair_audit["source"]["process"] == 53
    assert repair_audit["prepared_rows"][0]["changed_fields_vs_source"] == [
        "powell_maxiter_cap_policy"
    ]
    assert repair_audit["prepared_rows"][0]["declared_record_non_changed_fields_diff"] == []
    assert repair_audit["source_value_anchor"]["status"] == "not_claimed"
    assert repair_audit["source_evidence_fetch_status"] == (
        "local_bytes_verified_and_packaged_for_transfer"
    )
    assert repair_audit["repair"]["unresolved_source_fields"] == []
    assert (output_dir / "source_evidence" / "proc53_generic_static_single.json").is_file()
    assert (output_dir / "source_evidence" / "proc53_cell_manifest.json").is_file()
    assert preflight_submit._paper_i_scaling_append_powell_cap_repair_audit_blockers(
        row,
        repo_root=repair.ROOT,
    ) == []

    settings_diff = json.loads(
        (output_dir / "settings_diff.json").read_text(encoding="utf-8")
    )
    assert settings_diff["declared_record_science_settings_diff"] == {
        "powell_maxiter_cap_policy": {
            "source": "strict_failure_v1",
            "repair": "accept_finite_nonincreasing_v1",
        }
    }
    assert settings_diff["source_lock_status"] == "not_claimed_not_evaluated"
    assert settings_diff["status"] == "pass_declared_record_diff_not_source_lock"
    assert manifest["status"] == "prepared_not_submitted"
    assert manifest["submission_authority"] == "prepared_only_not_submitted"
    assert submit_path.is_file()

    tampered_hash = dict(row)
    tampered_hash["source_result_json_sha256"] = "0" * 64
    assert any(
        "source_result_json_sha256_mismatch" in blocker
        for blocker in preflight_submit._paper_i_scaling_matrix_contract_blockers(tampered_hash)
    )

    repair_audit["source_locked_sensitivity_claim"] = True
    (output_dir / "implementation_repair_audit.json").write_text(
        json.dumps(repair_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert any(
        "false_source_lock_claim" in blocker
        for blocker in preflight_submit._paper_i_scaling_append_powell_cap_repair_audit_blockers(
            row,
            repo_root=repair.ROOT,
        )
    )
