from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chtc.paper_iii_excited_dynamics import generate_compatibility_matrix
from chtc.paper_iii_excited_dynamics import preflight_inputs, validate_outputs


def _generate(tmp_path: Path) -> tuple[Path, dict]:
    out = tmp_path / "compatibility_matrix_nph1"
    result = generate_compatibility_matrix.generate_compatibility_matrix(
        output_dir=out,
        repo_root=REPO_ROOT,
        profile=generate_compatibility_matrix.PROFILE,
        no_submit=True,
    )
    return out, result


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh, delimiter="\t")]


def _split_codes(value: str) -> set[str]:
    return {item.strip() for item in str(value or "").replace(",", ";").split(";") if item.strip()}


def test_generate_matrix_writes_required_local_artifacts_and_counts(tmp_path: Path) -> None:
    out, result = _generate(tmp_path)

    assert result["ok"] is True
    assert result["record_count"] == 289
    assert result["blocker_count"] > 0
    assert result["seed_repair_unique_slot_count"] == 36
    assert result["seed_repair_expected_unique_slot_count"] == 36
    assert result["seed_repair_scope_decision_row_count"] == 1
    for name in (
        "manifest.json",
        "records.tsv",
        "full_record_ids.txt",
        "smoke_record_ids.txt",
        "blockers.tsv",
        "blockers.json",
        "compatibility_audit_summary.md",
        "seed_repair_inventory.tsv",
        "seed_repair_inventory.json",
        "seed_repair_summary.md",
    ):
        assert (out / name).exists(), name

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    expected = manifest["expected_counts"]
    actual = manifest["actual_counts"]
    assert manifest["schema_version"] == "paper_iii_compatibility_matrix_manifest_v1"
    assert manifest["profile"] == "paper_iii_nph1_compatibility_v1"
    assert manifest["no_submit"] is True
    assert manifest["submits_chtc"] is False
    assert manifest["runs_realtime_dynamics"] is False
    assert expected["table_i_declared_families"] == 10
    assert expected["table_i_declared_cases"] == 18
    assert expected["paper_ii_source_cases"] == 20
    assert expected["dynamics_classes"] == 3
    assert expected["comparators"] == 8
    assert expected["comparator_rows"] == 288
    assert expected["records_total"] == 289
    assert actual["record_count"] == 289
    assert actual["scope_decision_row_count"] == 1
    assert manifest["seed_repair_inventory_schema_version"] == "paper_iii_seed_repair_inventory_v1"
    assert manifest["seed_repair_unique_slot_count"] == 36
    assert manifest["seed_repair_expected_unique_slot_count"] == 36
    assert manifest["seed_repair_scope_decision_row_count"] == 1

    records = _read_tsv(out / "records.tsv")
    comparator_rows = [row for row in records if row["row_kind"] == "comparator_row"]
    scope_rows = [row for row in records if row["row_kind"] == "scope_decision"]
    assert len(comparator_rows) == 288
    assert len(scope_rows) == 1
    assert {row["family"] for row in comparator_rows} == set(manifest["declared_table_i_families"])
    assert len({(row["family"], row["paper_i_case_id"]) for row in comparator_rows}) == 18
    assert {row["comparator_id"] for row in comparator_rows} == {
        spec["comparator_id"] for spec in manifest["comparator_specs"]
    }
    assert all(row["mode"] == preflight_inputs.MODE_COMPATIBILITY_AUDIT_ONLY for row in records)
    assert all(row["no_submit"] == "true" for row in records)
    assert all(row["target_excited_roots"] == "6" for row in comparator_rows)
    assert all(row["n_ph_max"] == "1" for row in comparator_rows)
    assert all(row["strict_policy_status"] == "pass" for row in comparator_rows)

    full_ids = preflight_inputs.load_record_ids(out / "full_record_ids.txt")
    smoke_ids = preflight_inputs.load_record_ids(out / "smoke_record_ids.txt")
    assert full_ids == [row["record_id"] for row in records]
    assert len(smoke_ids) == 18
    assert set(smoke_ids) <= set(full_ids)


def test_seed_repair_inventory_collapses_comparator_rows_to_unique_slots(tmp_path: Path) -> None:
    out, _result = _generate(tmp_path)

    records = _read_tsv(out / "records.tsv")
    seed_tsv = _read_tsv(out / "seed_repair_inventory.tsv")
    payload = json.loads((out / "seed_repair_inventory.json").read_text(encoding="utf-8"))
    summary = (out / "seed_repair_summary.md").read_text(encoding="utf-8").lower()

    comparator_rows = [row for row in records if row["row_kind"] == "comparator_row"]
    expected_keys = {
        (row["family"], row["paper_i_case_id"], row["drive_amplitude"])
        for row in comparator_rows
    }
    json_slots = payload["slots"]
    assert payload["no_submit"] is True
    assert payload["repairs_seed_artifacts"] is False
    assert payload["changes_seed_paths"] is False
    assert "no submit" in summary or "no-submit" in summary
    assert "no seed repair" in summary
    assert len(expected_keys) == 36
    assert len(seed_tsv) == 36
    assert len(json_slots) == 36
    assert payload["unique_seed_slot_count"] == 36
    assert payload["expected_unique_seed_slot_count"] == 36
    assert payload["scope_decision_row_count"] == 1
    assert payload["scope_decision_rows"][0]["family"] == "molecular_restricted_closed_shell"
    assert not any(slot["family"] == "molecular_restricted_closed_shell" for slot in json_slots)

    slot_keys = {(slot["family"], slot["paper_i_case_id"], slot["drive_amplitude"]) for slot in json_slots}
    assert slot_keys == expected_keys
    assert len({slot["slot_id"] for slot in json_slots}) == 36
    assert {row["slot_id"] for row in seed_tsv} == {slot["slot_id"] for slot in json_slots}

    expected_comparators = {spec.comparator_id for spec in generate_compatibility_matrix.COMPARATOR_SPECS}
    seed_blockers_by_key: dict[tuple[str, str, str], set[str]] = {}
    for row in comparator_rows:
        key = (row["family"], row["paper_i_case_id"], row["drive_amplitude"])
        seed_blockers_by_key.setdefault(key, set()).update(
            code for code in _split_codes(row["blocker_codes"]) if code.startswith("seed.")
        )
    allowed_statuses = set(generate_compatibility_matrix.SEED_REPAIR_ALLOWED_STATUSES)
    status_counts: dict[str, int] = {}
    for slot in json_slots:
        key = (slot["family"], slot["paper_i_case_id"], slot["drive_amplitude"])
        assert int(slot["row_count"]) == 8
        assert _split_codes(slot["comparator_ids"]) == expected_comparators
        inherited = _split_codes(slot["inherited_seed_blocker_codes"])
        assert all(code.startswith("seed.") for code in inherited)
        assert inherited == seed_blockers_by_key[key]
        assert slot["repair_status"] in allowed_statuses
        status_counts[slot["repair_status"]] = status_counts.get(slot["repair_status"], 0) + 1
    assert payload["repair_status_counts"] == status_counts


def test_matrix_blockers_make_missing_support_explicit(tmp_path: Path) -> None:
    out, _result = _generate(tmp_path)

    records = _read_tsv(out / "records.tsv")
    blockers_payload = json.loads((out / "blockers.json").read_text(encoding="utf-8"))
    blockers = blockers_payload["blockers"]
    codes = {row["code"] for row in blockers}

    assert blockers_payload["no_submit"] is True
    assert "seed.staged_or_recovery_policy" in codes
    assert "seed.source_points_to_staged_dynamics_path" in codes
    assert "comparator.krylov_hamiltonian_power_qse_not_implemented" in codes
    assert "comparator.qeom_qsc_eom_not_implemented" in codes
    assert "qse.target_six_excited_roots_not_validated_for_matrix" in codes
    assert "static.table_i_case_deferred_at_head" in codes
    assert "scope.registry_only_family_not_table_i_declared" in codes

    molecular_scope = [row for row in records if row["family"] == "molecular_restricted_closed_shell"]
    assert len(molecular_scope) == 1
    assert molecular_scope[0]["row_kind"] == "scope_decision"
    assert molecular_scope[0]["expected_status"] == "blocked"
    assert "scope.requires_user_decision_for_paper_iii" in molecular_scope[0]["blocker_codes"]

    for row in records:
        parsed_codes = [code for code in row["blocker_codes"].split(";") if code]
        assert int(row["blocker_count"]) == len(parsed_codes)
        if row["expected_status"] == "blocked":
            assert parsed_codes


def test_preflight_accepts_audit_only_matrix_without_staging(tmp_path: Path) -> None:
    out, _result = _generate(tmp_path)

    report = preflight_inputs.preflight_records(
        records_path=out / "records.tsv",
        record_list=out / "full_record_ids.txt",
        repo_root=REPO_ROOT,
        stage=False,
        write_report=False,
    )

    assert report["ok"] is True, report
    assert report["record_count"] == 289
    assert report["stage"] is False
    assert report["removed_stale_artifacts"] == []
    assert {row["mode"] for row in report["records"]} == {preflight_inputs.MODE_COMPATIBILITY_AUDIT_ONLY}
    assert all(row["staged_artifact_exists"] is False for row in report["records"])


def test_validate_compatibility_matrix_dir_accepts_generated_matrix(tmp_path: Path) -> None:
    out, _result = _generate(tmp_path)

    report = validate_outputs.validate_compatibility_matrix_dir(out)

    assert report["ok"] is True, report
    assert report["record_count"] == 289
    assert report["comparator_row_count"] == 288
    assert report["family_count"] == 10
    assert report["case_count"] == 18
    assert report["comparator_count"] == 8
    assert report["scope_decision_row_count"] == 1
    assert report["seed_repair_unique_slot_count"] == 36
    assert report["seed_repair_expected_unique_slot_count"] == 36
    assert report["seed_repair_scope_decision_row_count"] == 1
    assert report["seed_repair_status_counts"]
    assert report["no_submit"] is True


def test_generate_and_validate_nph2_compatibility_matrix(tmp_path: Path) -> None:
    out = tmp_path / "compatibility_matrix_nph2"
    result = generate_compatibility_matrix.generate_compatibility_matrix(
        output_dir=out,
        repo_root=REPO_ROOT,
        profile=generate_compatibility_matrix._profile_for_n_ph(2),
        no_submit=True,
        n_ph_max=2,
    )

    assert result["ok"] is True
    assert result["profile"] == "paper_iii_nph2_compatibility_v1"
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["profile"] == "paper_iii_nph2_compatibility_v1"
    assert manifest["matrix_batch_id"] == "paper_iii_nph2_compatibility_v1_local_audit"
    assert manifest["n_ph_max"] == 2
    records = _read_tsv(out / "records.tsv")
    assert all(row["profile"] == "paper_iii_nph2_compatibility_v1" for row in records)
    assert all(row["n_ph_max"] == "2" for row in records)
    assert all(row["record_id"].startswith("paper_iii_nph2_") for row in records)

    report = validate_outputs.validate_compatibility_matrix_dir(out)

    assert report["ok"] is True, report
    assert report["record_count"] == 289
    assert report["comparator_row_count"] == 288


def test_preflight_accepts_nph2_audit_only_matrix_without_staging(tmp_path: Path) -> None:
    out = tmp_path / "compatibility_matrix_nph2"
    generate_compatibility_matrix.generate_compatibility_matrix(
        output_dir=out,
        repo_root=REPO_ROOT,
        profile=generate_compatibility_matrix._profile_for_n_ph(2),
        no_submit=True,
        n_ph_max=2,
    )

    report = preflight_inputs.preflight_records(
        records_path=out / "records.tsv",
        record_list=out / "full_record_ids.txt",
        repo_root=REPO_ROOT,
        stage=False,
        write_report=False,
    )

    assert report["ok"] is True, report
    assert report["record_count"] == 289
    comparator_rows = [row for row in report["records"] if row["row_kind"] == "comparator_row"]
    assert comparator_rows
    assert all(row["profile"] == "paper_iii_nph2_compatibility_v1" for row in comparator_rows)
    assert all(row["t_final"] is not None for row in comparator_rows)
    assert all(row["num_times"] is not None for row in comparator_rows)


def test_validate_compatibility_matrix_dir_rejects_seed_inventory_status_regression(tmp_path: Path) -> None:
    out, _result = _generate(tmp_path)
    inventory_path = out / "seed_repair_inventory.json"
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    payload["slots"][0]["repair_status"] = "ready"
    inventory_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = validate_outputs.validate_compatibility_matrix_dir(out)

    assert report["ok"] is False
    assert any("repair_status" in error for error in report["errors"])


def test_validate_compatibility_matrix_dir_rejects_submit_policy_regression(tmp_path: Path) -> None:
    out, _result = _generate(tmp_path)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["no_submit"] = False
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = validate_outputs.validate_compatibility_matrix_dir(out)

    assert report["ok"] is False
    assert any("no_submit" in error for error in report["errors"])
