from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chtc.paper_iii_excited_dynamics import generate_compatibility_matrix, production_gate


def _generate_matrix(tmp_path: Path, *, n_ph_max: int) -> Path:
    out = tmp_path / f"compatibility_matrix_nph{n_ph_max}"
    generate_compatibility_matrix.generate_compatibility_matrix(
        output_dir=out,
        repo_root=REPO_ROOT,
        profile=generate_compatibility_matrix._profile_for_n_ph(n_ph_max),
        no_submit=True,
        n_ph_max=n_ph_max,
    )
    return out


def _generate_pair(tmp_path: Path) -> tuple[Path, Path]:
    return _generate_matrix(tmp_path, n_ph_max=1), _generate_matrix(tmp_path, n_ph_max=2)


def _snapshot_tree(root: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        snapshot[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def _rewrite_records(path: Path, mutate_first_comparator: dict[str, str]) -> None:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    for row in rows:
        if row.get("row_kind") == "comparator_row":
            row.update(mutate_first_comparator)
            break
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_production_gate_summarizes_generated_compatibility_pair_read_only(tmp_path: Path) -> None:
    nph1_dir, nph2_dir = _generate_pair(tmp_path)
    before = {
        "nph1": _snapshot_tree(nph1_dir),
        "nph2": _snapshot_tree(nph2_dir),
    }

    report = production_gate.validate_production_gate(nph1_dir=nph1_dir, nph2_dir=nph2_dir)

    assert report["schema_version"] == production_gate.PRODUCTION_GATE_SCHEMA_VERSION
    assert report["read_only"] is True
    assert report["mutates_chtc_inputs"] is False
    assert report["mutates_generated_artifacts"] is False
    assert report["first_pass_ready"] is True
    assert report["n_ph1_first_pass_status"] == "available_with_blockers"
    assert report["production_ready"] is False
    assert report["ok"] is False
    assert report["n_ph2_production_readiness"] == "blocked_by_compatibility"
    assert set(report["consumer_fail_closed"]) == set(production_gate.CONSUMER_IDS)
    assert all(report["consumer_fail_closed"].values())

    nph2_target = report["target_excited_root_count"]["n_ph2"]
    assert nph2_target["status"] == "pass"
    assert nph2_target["required"] == 6
    assert nph2_target["manifest"] == 6
    assert nph2_target["comparator_row_values"] == [6]

    missing = report["missing_comparator_blockers"]
    assert missing["status"] == "blocked"
    assert "comparator.fixed_alphabet_registry_missing" in missing["codes"]
    assert "comparator.krylov_hamiltonian_power_qse_not_implemented" in missing["codes"]
    assert "comparator.qeom_qsc_eom_not_implemented" in missing["codes"]
    assert missing["by_comparator"]["fixed_alphabet_qse"] > 0

    assert report["exact_reference_boundary_status"]["status"] == "pass"
    assert report["compatibility_tiers"]["n_ph_1"]["validation_ok"] is True
    assert report["compatibility_tiers"]["n_ph_2"]["validation_ok"] is True
    with pytest.raises(production_gate.ProductionGateError, match="optuna_production_mode fail-closed"):
        production_gate.require_production_ready(report, consumer_id="optuna_production_mode")

    after = {
        "nph1": _snapshot_tree(nph1_dir),
        "nph2": _snapshot_tree(nph2_dir),
    }
    assert after == before


def test_production_gate_fails_closed_when_nph2_evidence_is_missing(tmp_path: Path) -> None:
    nph1_dir = _generate_matrix(tmp_path, n_ph_max=1)
    missing_nph2_dir = tmp_path / "compatibility_matrix_nph2_missing"

    report = production_gate.validate_production_gate(nph1_dir=nph1_dir, nph2_dir=missing_nph2_dir)

    assert report["first_pass_ready"] is True
    assert report["n_ph1_first_pass_status"] == "available_with_blockers"
    assert report["production_ready"] is False
    assert report["n_ph2_production_readiness"] == "missing"
    assert report["compatibility_tiers"]["n_ph_2"]["exists"] is False
    assert any("n_ph=2 compatibility matrix directory is missing" in error for error in report["errors"])
    with pytest.raises(production_gate.ProductionGateError, match="source_map_generation fail-closed"):
        production_gate.require_production_ready(report, consumer_id="source_map_generation")


def test_production_gate_reports_exact_reference_boundary_violation(tmp_path: Path) -> None:
    nph1_dir, nph2_dir = _generate_pair(tmp_path)
    _rewrite_records(nph2_dir / "records.tsv", {"uses_reference_for_decision": "true"})

    report = production_gate.validate_production_gate(nph1_dir=nph1_dir, nph2_dir=nph2_dir)

    assert report["production_ready"] is False
    assert report["exact_reference_boundary_status"]["status"] == "fail"
    nph2_boundary = report["compatibility_tiers"]["n_ph_2"]["exact_reference_boundary"]
    assert nph2_boundary["status"] == "fail"
    assert any("uses_reference_for_decision" in violation for violation in nph2_boundary["violations"])


def test_production_gate_reports_target_root_count_regression(tmp_path: Path) -> None:
    nph1_dir, nph2_dir = _generate_pair(tmp_path)
    manifest_path = nph2_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["target_excited_roots"] = 4
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = production_gate.validate_production_gate(nph1_dir=nph1_dir, nph2_dir=nph2_dir)

    nph2_target = report["target_excited_root_count"]["n_ph2"]
    assert report["production_ready"] is False
    assert report["n_ph2_production_readiness"] == "blocked"
    assert nph2_target["status"] == "fail"
    assert nph2_target["manifest"] == 4
    assert any("manifest.target_excited_roots expected 6" in item for item in nph2_target["violations"])
