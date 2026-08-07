#!/usr/bin/env python3
"""Build the source-value anchor for no-overlap SR trust calibration.

The anchor retains the validated support-projected Phase-III route and changes
only the executable source archive.  That archive contains the candidate
no-overlap trust controller, but the anchor still requests the predecessor
displacement-calibrated policy.  The six candidate rows remain fail-closed
until the anchor reproduces the locked projected weak-weak result.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "chtc/phase3_optuna/input"
BASE_ID = (
    "paper_i_hh_sr_snake_phase3_projected_generalized_all_six_"
    "r50_20260720_v1_chtc"
)
ANCHOR_ID = (
    "paper_i_hh_sr_snake_no_overlap_trust_parent_anchor_weak_weak_"
    "r50_20260720_v2_chtc"
)
BASE = INPUT / BASE_ID
ANCHOR = INPUT / ANCHOR_ID
BASE_BATCH = "paper-i-hh-sr-phase3-projected-generalized-six-r50-20260720-v1"
ANCHOR_BATCH = "paper-i-hh-sr-no-overlap-trust-parent-anchor-ww-r50-20260720-v2"
PARENT_ALIAS = "sr_snake_no_prune_symmetric_cost_projected_phase3_v1"
PARENT_PROFILE = (
    "supported_projected_generalized_adaptive_trust_full_response_"
    "symmetric_cost_no_prune_v1"
)
PARENT_DIGEST = "3ff2abb1455cda3cf8cc2de0cf739172f8cdcfe6b1c9436e1afdd40076cd3ce8"
CHILD_ALIAS = (
    "sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1"
)
CHILD_PROFILE = (
    "supported_projected_generalized_source_metric_no_overlap_trust_"
    "full_response_symmetric_cost_no_prune_v1"
)
CHILD_DIGEST = "fd5ec3fa2c98b2a9d1cbcc304241d723f57dbd6210f4ea2daf30753603a146c2"
IMAGE_SHA256 = "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
SOURCE_RESULT = ROOT / (
    "raw_outputs/chtc_fetch_paper_i_hh_sr_20260720/projected_8908614_core/"
    "raw_outputs/paper_i_hh_sr_snake_phase3_projected_generalized_all_six_"
    "r50_20260720_v1_chtc/weak_weak/json/result.json"
)
SOURCE_TRANSFER = ROOT / (
    "raw_outputs/chtc_fetch_paper_i_hh_sr_20260720/"
    "clusters_8908613_8908614_8908617/8908614.0__weak_weak_transfer.tar.gz"
)
OVERLAY_FILES = (
    "pipelines/static_adapt/route_a_schur_selector.py",
    "pipelines/static_adapt/route_a_trust_region.py",
    "pipelines/static_adapt/adapt_pipeline.py",
    "pipelines/static_adapt/cli_config.py",
    "pipelines/static_adapt/sr_snake_route_profile.py",
    "test/test_static_adapt_route_a_trust_region.py",
    "test/test_static_adapt_projected_phase3_route_profile.py",
    "test/test_static_adapt_accepted_joint_coordinate_step.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def replace_tree(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [replace_tree(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: replace_tree(item, replacements) for key, item in value.items()}
    return value


def deterministic_archive(source: Path, output: Path) -> None:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source).as_posix()
            data = path.read_bytes()
            info = tarfile.TarInfo(relative)
            info.size = len(data)
            info.mode = path.stat().st_mode & 0o777
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))
    with output.open("wb") as handle:
        with gzip.GzipFile(fileobj=handle, mode="wb", filename="", mtime=0) as zipped:
            zipped.write(raw.getvalue())


def inventory(source: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(source).as_posix(): {
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(source.rglob("*"))
        if path.is_file()
    }


def patch_projected_evidence_validator(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    old = '''        for key in (
            "joint_solve_policy",
            "joint_linear_solve_policy_requested",
            "joint_linear_solve_policy_effective",
        ):
            if summary.get(key) != PROJECTED_GENERALIZED_POLICY:
                raise ValueError(
                    f"round {expected_round} projected solver policy drift: {key}"
                )
        if (
            summary.get("supported_metric_projection_active") is not True
            or summary.get("supported_metric_whitening_active") is not False
            or summary.get("supported_metric_inverse_sqrt_constructed") is not False
            or summary.get("supported_metric_inverse_constructed") is not False
            or summary.get("metric_regularization_applied") is not False
            or int(summary.get("classical_quantum_query_charge", -1)) != 0
        ):
            raise ValueError(
                f"round {expected_round} projected/no-whitening receipt drift"
            )
        provenance = str(
            summary.get("supported_metric_projection_provenance_id") or ""
        )
        if len(provenance) != 64:
            raise ValueError(
                f"round {expected_round} projection provenance is unresolved"
            )
        provenance_ids.append(provenance)
        if bool(summary.get("feasible", False)):
            residual = float(summary.get("supported_generalized_kkt_residual"))
            if not math.isfinite(residual):
                raise ValueError(
                    f"round {expected_round} generalized KKT residual is nonfinite"
                )
            feasible_count += 1
        else:
            fallback_count += 1
'''
    new = '''        fallback_fired = bool(
            row.get("all_energy_models_infeasible_novelty_fallback_fired", False)
        )
        if fallback_fired:
            if (
                row.get("all_energy_models_infeasible_novelty_fallback_enabled")
                is not True
                or int(
                    row.get(
                        "all_energy_models_infeasible_novelty_fallback_query_charge",
                        -1,
                    )
                )
                != 0
            ):
                raise ValueError(
                    f"round {expected_round} fallback receipt is incomplete"
                )
            fallback_count += 1
        else:
            for key in (
                "joint_solve_policy",
                "joint_linear_solve_policy_requested",
                "joint_linear_solve_policy_effective",
            ):
                if summary.get(key) != PROJECTED_GENERALIZED_POLICY:
                    raise ValueError(
                        f"round {expected_round} projected solver policy drift: {key}"
                    )
            if (
                summary.get("supported_metric_projection_active") is not True
                or summary.get("supported_metric_whitening_active") is not False
                or summary.get("supported_metric_inverse_sqrt_constructed") is not False
                or summary.get("supported_metric_inverse_constructed") is not False
                or summary.get("metric_regularization_applied") is not False
                or int(summary.get("classical_quantum_query_charge", -1)) != 0
            ):
                raise ValueError(
                    f"round {expected_round} projected/no-whitening receipt drift"
                )
            provenance = str(
                summary.get("supported_metric_projection_provenance_id") or ""
            )
            if len(provenance) != 64:
                raise ValueError(
                    f"round {expected_round} projection provenance is unresolved"
                )
            provenance_ids.append(provenance)
            residual = float(summary.get("supported_generalized_kkt_residual"))
            if not math.isfinite(residual):
                raise ValueError(
                    f"round {expected_round} generalized KKT residual is nonfinite"
                )
            feasible_count += 1
'''
    if old not in text:
        raise ValueError("projected evidence validator patch anchor not found")
    path.write_text(text.replace(old, new), encoding="utf-8")


def isolated_route_digests(source: Path) -> dict[str, str]:
    code = (
        "import json\n"
        "from pipelines.static_adapt.sr_snake_route_profile import "
        "canonical_sr_snake_contract_sha256\n"
        f"print(json.dumps({{{PARENT_ALIAS!r}: canonical_sr_snake_contract_sha256({PARENT_ALIAS!r}), "
        f"{CHILD_ALIAS!r}: canonical_sr_snake_contract_sha256({CHILD_ALIAS!r})}}, sort_keys=True))\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source)
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=source, env=env,
        check=True, capture_output=True, text=True,
    )
    return json.loads(completed.stdout)


def build() -> dict[str, Any]:
    if ANCHOR.exists():
        raise FileExistsError(ANCHOR)
    if not SOURCE_RESULT.is_file() or not SOURCE_TRANSFER.is_file():
        raise FileNotFoundError("locked projected weak-weak source evidence is missing")
    source_result = load(SOURCE_RESULT)
    base_job = load(BASE / "jobs/weak_weak.json")
    old_archive_sha = sha256(BASE / "source_locked.tar.gz")

    with tempfile.TemporaryDirectory(prefix="sr-no-overlap-anchor-") as raw:
        temp = Path(raw)
        source = temp / "source"
        source.mkdir()
        with tarfile.open(BASE / "source_locked.tar.gz", "r:gz") as archive:
            archive.extractall(source, filter="data")
        overlays: dict[str, Any] = {}
        replacements = {BASE_ID: ANCHOR_ID, BASE_BATCH: ANCHOR_BATCH}
        for relative in OVERLAY_FILES:
            live = ROOT / relative
            target = source / relative
            before = sha256(target) if target.is_file() else None
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(live, target)
            after = sha256(target)
            if before:
                replacements[before] = after
            overlays[relative] = {
                "parent_sha256": before,
                "overlay_sha256": after,
                "size_bytes": target.stat().st_size,
                "classification": "no_overlap_trust_controller_or_focused_test_v1",
            }
        digests = isolated_route_digests(source)
        if digests != {PARENT_ALIAS: PARENT_DIGEST, CHILD_ALIAS: CHILD_DIGEST}:
            raise ValueError(f"isolated route digest drift: {digests}")
        new_archive = temp / "source_locked.tar.gz"
        deterministic_archive(source, new_archive)
        new_archive_sha = sha256(new_archive)
        replacements[old_archive_sha] = new_archive_sha
        files = inventory(source)
        shutil.copytree(BASE, ANCHOR, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copy2(new_archive, ANCHOR / "source_locked.tar.gz")

    archive_manifest = replace_tree(
        load(BASE / "source_archive_manifest.json"), replacements
    )
    no_overlap_overlay = {
        "schema": "paper_i_sr_no_overlap_trust_source_overlay_v1",
        "parent_source_archive_sha256": old_archive_sha,
        "source_archive_sha256": new_archive_sha,
        "parent_route_contract_sha256": PARENT_DIGEST,
        "candidate_route_contract_sha256": CHILD_DIGEST,
        "overlay_files": overlays,
    }
    archive_manifest.update({
        "archive": f"chtc/phase3_optuna/input/{ANCHOR_ID}/source_locked.tar.gz",
        "archive_sha256": new_archive_sha,
        "archive_size_bytes": (ANCHOR / "source_locked.tar.gz").stat().st_size,
        "file_count": len(files),
        "files": files,
        "no_overlap_trust_source_overlay": no_overlap_overlay,
    })
    dump(ANCHOR / "source_archive_manifest.json", archive_manifest)
    archive_manifest_sha = sha256(ANCHOR / "source_archive_manifest.json")

    revision = replace_tree(load(BASE / "source_revision_manifest.json"), replacements)
    revision["no_overlap_trust_source_overlay"] = no_overlap_overlay
    dump(ANCHOR / "source_revision_manifest.json", revision)
    revision_sha = sha256(ANCHOR / "source_revision_manifest.json")

    physics = replace_tree(load(BASE / "physics_and_exact_reference_lock.json"), replacements)
    dump(ANCHOR / "physics_and_exact_reference_lock.json", physics)
    physics_sha = sha256(ANCHOR / "physics_and_exact_reference_lock.json")

    job = replace_tree(copy.deepcopy(base_job), replacements)
    job["bundle_id"] = ANCHOR_ID
    job["batch_name"] = ANCHOR_BATCH
    job["source_lock"].update({
        "source_archive": f"chtc/phase3_optuna/input/{ANCHOR_ID}/source_locked.tar.gz",
        "source_archive_sha256": new_archive_sha,
        "source_archive_manifest": f"chtc/phase3_optuna/input/{ANCHOR_ID}/source_archive_manifest.json",
        "source_archive_manifest_sha256": archive_manifest_sha,
        "source_revision_manifest": f"chtc/phase3_optuna/input/{ANCHOR_ID}/source_revision_manifest.json",
        "source_revision_manifest_sha256": revision_sha,
        "physics_reference_lock": f"chtc/phase3_optuna/input/{ANCHOR_ID}/physics_and_exact_reference_lock.json",
        "physics_reference_lock_sha256": physics_sha,
        "no_overlap_trust_source_overlay": no_overlap_overlay,
    })
    job["source_value_anchor"] = {
        "schema": "source_locked_sensitivity_anchor_plan_v1",
        "source_result": str(SOURCE_RESULT.relative_to(ROOT)),
        "source_result_sha256": sha256(SOURCE_RESULT),
        "source_transfer_archive": str(SOURCE_TRANSFER.relative_to(ROOT)),
        "source_transfer_archive_sha256": sha256(SOURCE_TRANSFER),
        "swept_field": "historical_singleton_trust_region_update_policy",
        "source_value": "displacement_calibrated_unbounded_v2",
        "candidate_value": "source_metric_inverse_sqrt_no_overlap_v1",
        "fanout_allowed_before_anchor_pass": False,
    }
    for key, value in list(job["paths"].items()):
        job["paths"][key] = str(value).replace(BASE_ID, ANCHOR_ID)
    dump(ANCHOR / "jobs/weak_weak.json", job)

    normalized = replace_tree(
        load(BASE / "normalized_manifests/weak_weak.json"), replacements
    )
    normalized["bundle_id"] = ANCHOR_ID
    normalized["batch_name"] = ANCHOR_BATCH
    normalized["source_lock"] = copy.deepcopy(job["source_lock"])
    normalized["source_value_anchor"] = copy.deepcopy(job["source_value_anchor"])
    dump(ANCHOR / "normalized_manifests/weak_weak.json", normalized)
    for folder in (ANCHOR / "jobs", ANCHOR / "normalized_manifests"):
        for path in folder.glob("*.json"):
            if path.name != "weak_weak.json":
                path.unlink()

    for relative in (
        "run_job.py", "evidence_validation.py", "validate_fetched.py",
        "execute_source_locked_job.sh",
    ):
        path = ANCHOR / relative
        text = path.read_text(encoding="utf-8")
        for old, new in replacements.items():
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")
    patch_projected_evidence_validator(ANCHOR / "evidence_validation.py")

    (ANCHOR / "queue.tsv").write_text(
        f"weak_weak\tchtc/phase3_optuna/input/{ANCHOR_ID}/jobs/weak_weak.json\t"
        f"chtc/phase3_optuna/input/{ANCHOR_ID}/normalized_manifests/weak_weak.json\t"
        "40960\t61440\n",
        encoding="utf-8",
    )
    submit = (BASE / "submit.sub").read_text(encoding="utf-8")
    for old, new in replacements.items():
        submit = submit.replace(old, new)
    submit = "\n".join(
        line for line in submit.splitlines()
        if not line.startswith("queue regime_slug")
    ) + (
        "\nqueue regime_slug, job_manifest, normalized_manifest, memory_mb, disk_mb "
        f"from chtc/phase3_optuna/input/{ANCHOR_ID}/queue.tsv\n"
    )
    (ANCHOR / "submit.sub").write_text(submit, encoding="utf-8")

    audit = {
        "schema": "source_locked_sensitivity_audit_v1",
        "source": {
            "method": "SR-SNAKE projected Phase III",
            "regime_or_case": "weak_weak",
            "source_json": str(SOURCE_RESULT.relative_to(ROOT)),
            "source_sha256": sha256(SOURCE_RESULT),
            "source_transfer_archive": str(SOURCE_TRANSFER.relative_to(ROOT)),
            "source_transfer_archive_sha256": sha256(SOURCE_TRANSFER),
            "route_or_profile_id": PARENT_ALIAS,
            "route_contract_sha256": PARENT_DIGEST,
            "source_variable_value": "displacement_calibrated_unbounded_v2",
        },
        "sweep": {
            "run_class": "candidate",
            "variable": "historical_singleton_trust_region_update_policy",
            "grid": [
                "displacement_calibrated_unbounded_v2",
                "source_metric_inverse_sqrt_no_overlap_v1",
            ],
            "runner_mode": "direct_source_locked_replay",
            "baseline_materialization_status": "complete",
            "unresolved_source_fields": [],
            "fields_added_by_current_defaults": [],
        },
        "planned_rows": [{
            "value": "displacement_calibrated_unbounded_v2",
            "changed_fields_vs_source": [],
            "non_swept_settings_diff": [],
            "bundle": ANCHOR_ID,
        }],
        "anchor": {
            "value": "displacement_calibrated_unbounded_v2",
            "anchor_result_json": None,
            "anchor_reproduces_source": False,
            "metric_abs_diff": None,
            "operator_sequence_match": None,
            "non_swept_settings_diff": [],
        },
        "fanout_authorized": False,
        "status": "anchor_pending",
    }
    dump(ANCHOR / "source_locked_sensitivity_audit.json", audit)
    receipt = {
        "schema": "paper_i_sr_no_overlap_trust_parent_anchor_bundle_v1",
        "bundle_id": ANCHOR_ID,
        "batch_name": ANCHOR_BATCH,
        "source_archive_sha256": new_archive_sha,
        "source_archive_manifest_sha256": archive_manifest_sha,
        "source_revision_manifest_sha256": revision_sha,
        "physics_reference_lock_sha256": physics_sha,
        "parent_route_profile": PARENT_PROFILE,
        "parent_route_contract_sha256": PARENT_DIGEST,
        "candidate_route_profile": CHILD_PROFILE,
        "candidate_route_contract_sha256": CHILD_DIGEST,
        "job_count": 1,
        "fanout_authorized": False,
        "submission_performed": False,
    }
    dump(ANCHOR / "anchor_bundle_receipt.json", receipt)
    dump(ANCHOR / "bundle_manifest.json", receipt)
    dump(ANCHOR / "route_parity.json", {
        "schema": "paper_i_sr_no_overlap_trust_anchor_route_parity_v1",
        "status": "pass",
        "anchor_changed_scientific_fields_vs_source": [],
        "candidate_swept_field": "historical_singleton_trust_region_update_policy",
        "parent_route_contract_sha256": PARENT_DIGEST,
        "candidate_route_contract_sha256": CHILD_DIGEST,
    })
    dump(ANCHOR / "scientific_settings_audit.json", {
        "schema": "paper_i_sr_no_overlap_trust_anchor_scientific_settings_audit_v1",
        "status": "pass",
        "anchor_changed_scientific_fields_vs_source": [],
        "candidate_not_executed_in_anchor": True,
    })
    dump(ANCHOR / "preflight.json", {
        "schema": "paper_i_sr_no_overlap_trust_anchor_preflight_v1",
        "status": "pending_archive_only_validation",
        "checks": {
            "one_parent_anchor_record": True,
            "same_cutoff_n_ph_3": True,
            "exact_round_50_horizon": True,
            "candidate_not_executed": True,
            "archive_only_worker_validation": False,
            "archive_focused_tests": False,
        },
    })
    (ANCHOR / "README.md").write_text(
        "# No-overlap trust source-value anchor\n\n"
        "This one-row bundle replays the locked projected Phase-III weak-weak "
        "source under the candidate implementation archive. The six candidate "
        "rows remain fail-closed until exact source-value reproduction.\n",
        encoding="utf-8",
    )
    verifier = f'''#!/usr/bin/env python3
import hashlib, json
from pathlib import Path
B=Path(__file__).resolve().parent
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
    r=json.loads((B/"anchor_bundle_receipt.json").read_text())
    assert h(B/"source_locked.tar.gz")==r["source_archive_sha256"]
    jobs=list((B/"jobs").glob("*.json")); assert [p.name for p in jobs]==["weak_weak.json"]
    j=json.loads(jobs[0].read_text())
    assert j["route_identity"]["profile_request"]=={PARENT_ALIAS!r}
    assert j["route_identity"]["profile_contract_sha256"]=={PARENT_DIGEST!r}
    assert j["route_identity"]["profile_contract"]["execution_settings"]["historical_singleton_trust_region_update_policy"]=="displacement_calibrated_unbounded_v2"
    assert int(j["segment"]["target_controller_round"])==50
    assert j["physics"]["n_ph_work"]==j["physics"]["n_ph_reference"]==3
    assert json.loads((B/"source_locked_sensitivity_audit.json").read_text())["fanout_authorized"] is False
    assert "requirements = False" not in (B/"submit.sub").read_text()
    return True
if __name__=="__main__": verify(); print("no-overlap trust anchor bundle verified")
'''
    (ANCHOR / "build_bundle.py").write_text(verifier, encoding="utf-8")
    (ANCHOR / "test_bundle.py").write_text(
        "import build_bundle\ndef test_bundle(): assert build_bundle.verify()\n",
        encoding="utf-8",
    )
    return receipt


def archive_preflight() -> None:
    with tempfile.TemporaryDirectory(prefix="sr-no-overlap-anchor-preflight-") as raw:
        root = Path(raw)
        with tarfile.open(ANCHOR / "source_locked.tar.gz", "r:gz") as archive:
            archive.extractall(root, filter="data")
        target = root / "chtc/phase3_optuna/input" / ANCHOR_ID
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ANCHOR, target)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root)
        subprocess.run(
            [sys.executable, str(target / "run_job.py"), "--validate-only", str(target / "jobs/weak_weak.json")],
            cwd=root, env=env, check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q",
             "test/test_static_adapt_route_a_trust_region.py",
             "test/test_static_adapt_projected_phase3_route_profile.py",
             "test/test_static_adapt_accepted_joint_coordinate_step.py"],
            cwd=root, env=env, check=True,
        )


def main() -> int:
    receipt = build()
    subprocess.run([sys.executable, str(ANCHOR / "build_bundle.py")], check=True)
    subprocess.run([sys.executable, "-m", "pytest", "-q", str(ANCHOR / "test_bundle.py")], check=True)
    archive_preflight()
    preflight = load(ANCHOR / "preflight.json")
    preflight["status"] = "pass"
    preflight["checks"]["archive_only_worker_validation"] = True
    preflight["checks"]["archive_focused_tests"] = True
    dump(ANCHOR / "preflight.json", preflight)
    dump(ANCHOR / "submission_artifact_hashes.json", {
        "schema": "paper_i_sr_no_overlap_trust_anchor_submission_artifacts_v1",
        "files": {
            path.relative_to(ANCHOR).as_posix(): {
                "sha256": sha256(path), "size_bytes": path.stat().st_size,
            }
            for path in sorted(ANCHOR.rglob("*"))
            if path.is_file() and path.name != "submission_artifact_hashes.json"
        },
    })
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
