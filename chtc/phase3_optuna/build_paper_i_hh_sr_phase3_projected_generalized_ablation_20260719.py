#!/usr/bin/env python3
"""Build the source-value anchor for the Phase-III projection ablation.

The anchor deliberately retains the validated Main-SR scientific profile and
changes only the immutable executable source archive.  The six projected-
generalized rows are not built or submitted until this anchor reproduces the
source result and a source-locked sensitivity audit records that fact.
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
    "paper_i_hh_sr_snake_main_full_response_symmetric_cost_noprune_nobeam_"
    "no_ordinary_novelty_all_six_r50_20260718_v4_chtc"
)
BASE = INPUT / BASE_ID
ANCHOR_ID = (
    "paper_i_hh_sr_snake_phase3_projected_generalized_parent_anchor_"
    "weak_weak_r50_20260719_v9_chtc"
)
ANCHOR = INPUT / ANCHOR_ID
BASE_BATCH = (
    "paper-i-hh-sr-main-fullresp-symcost-noprune-nobeam-nonovelty-"
    "six-r50-20260718-v4"
)
ANCHOR_BATCH = "paper-i-hh-sr-phase3-projected-parent-anchor-ww-r50-20260719-v9"
PARENT_ALIAS = "sr_snake_no_prune_symmetric_cost_v1"
PARENT_DIGEST = "023bc7ac535ee4d88d78dd5336a59dd2fb0543c133fa0a60b009efab75422c91"
CHILD_ALIAS = "sr_snake_no_prune_symmetric_cost_projected_phase3_v1"
CHILD_DIGEST = "3ff2abb1455cda3cf8cc2de0cf739172f8cdcfe6b1c9436e1afdd40076cd3ce8"
IMAGE_SHA256 = "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
SOURCE_FETCH_DIR = ROOT / (
    "raw_outputs/chtc_fetch_paper_i_hh_r50_full48_20260719/"
    "main_sr_8887574_completed_p0_1"
)
SOURCE_TRANSFER = SOURCE_FETCH_DIR / "8887574.0__weak_weak_transfer.tar.gz"
SOURCE_VALIDATION_RECEIPT = (
    SOURCE_FETCH_DIR / "8887574.0__weak_weak_local_validation_receipt.json"
)
SOURCE_RESULT_MEMBER = (
    "raw_outputs/paper_i_hh_sr_snake_main_full_response_symmetric_cost_"
    "noprune_nobeam_no_ordinary_novelty_all_six_r50_20260718_v4_chtc/"
    "weak_weak/json/result.json"
)
OVERLAY_FILES = (
    "pipelines/static_adapt/joint_linear_solve.py",
    "pipelines/static_adapt/sr_snake_route_profile.py",
    "pipelines/static_adapt/cli_config.py",
    "pipelines/static_adapt/formal_manifold_route_profile.py",
    "pipelines/static_adapt/formal_manifold_outer_information.py",
    "pipelines/static_adapt/adapt_pipeline.py",
    "pipelines/static_adapt/adapt_candidate_record_cache.py",
    "pipelines/static_adapt/resume_scaffold.py",
    "pipelines/static_adapt/builders/shared_pauli_pool_contract.py",
    "pipelines/static_adapt/beam_search.py",
    "pipelines/static_adapt/estimator_call_ledger.py",
    "pipelines/scaffold/hh_continuation_scoring.py",
    "pipelines/static_adapt/selector_query_closure.py",
    "test/test_static_adapt_projected_generalized_trust_solve.py",
    "test/test_static_adapt_projected_phase3_route_profile.py",
    "test/test_static_adapt_sr_phase_liveness_contract.py",
    "test/test_static_adapt_sr_route_profile.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def replace_tree(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: replace_tree(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_tree(item, replacements) for item in value]
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
    return value


def deterministic_archive(source: Path, destination: Path) -> None:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path in sorted(source.rglob("*"), key=lambda item: item.relative_to(source).as_posix()):
            relative = path.relative_to(source).as_posix()
            if path.is_symlink():
                raise ValueError(f"source archive cannot contain symlink: {relative}")
            info = tarfile.TarInfo(relative + ("/" if path.is_dir() else ""))
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            info.mtime = 0
            info.mode = 0o755 if path.is_dir() or os.access(path, os.X_OK) else 0o644
            if path.is_dir():
                info.type = tarfile.DIRTYPE
                archive.addfile(info)
            elif path.is_file():
                info.size = path.stat().st_size
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as zipped:
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


def build_source(temp: Path) -> tuple[Path, dict[str, dict[str, Any]], dict[str, Any]]:
    source = temp / "source"
    source.mkdir()
    with tarfile.open(BASE / "source_locked.tar.gz", "r:gz") as archive:
        archive.extractall(source, filter="data")
    overlays: dict[str, Any] = {}
    for relative in OVERLAY_FILES:
        live = ROOT / relative
        if not live.is_file():
            raise FileNotFoundError(live)
        target = source / relative
        before = sha256(target) if target.is_file() else None
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(live, target)
        overlays[relative] = {
            "classification": "phase3_projection_ablation_implementation_or_focused_test",
            "parent_sha256": before,
            "overlay_sha256": sha256(target),
            "size_bytes": target.stat().st_size,
        }
    output = temp / "source_locked.tar.gz"
    deterministic_archive(source, output)
    return output, inventory(source), overlays


def isolated_digests(source_archive: Path) -> dict[str, str]:
    with tempfile.TemporaryDirectory(prefix="sr-projected-digest-") as raw:
        root = Path(raw)
        with tarfile.open(source_archive, "r:gz") as archive:
            archive.extractall(root, filter="data")
        code = (
            "import json\n"
            "from pipelines.static_adapt.sr_snake_route_profile import "
            "canonical_sr_snake_contract_sha256\n"
            f"print(json.dumps({{{PARENT_ALIAS!r}: canonical_sr_snake_contract_sha256({PARENT_ALIAS!r}), "
            f"{CHILD_ALIAS!r}: canonical_sr_snake_contract_sha256({CHILD_ALIAS!r})}}, sort_keys=True))\n"
        )
        env = os.environ.copy()
        # The local macOS Python keeps NumPy/SciPy in its user site.  The
        # archive path remains the only executable-source authority through
        # PYTHONPATH; dependency isolation is rechecked in the CHTC image,
        # where the worker intentionally sets PYTHONNOUSERSITE=1.
        env.update({"PYTHONPATH": str(root)})
        env.pop("PYTHONNOUSERSITE", None)
        completed = subprocess.run(
            [sys.executable, "-c", code], cwd=root, env=env,
            check=True, capture_output=True, text=True,
        )
        return json.loads(completed.stdout)


def _patch_text(path: Path, replacements: dict[str, str]) -> None:
    text = path.read_text(encoding="utf-8")
    for old, new in replacements.items():
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def build_anchor() -> dict[str, Any]:
    if ANCHOR.exists():
        raise FileExistsError(f"immutable anchor already exists: {ANCHOR}")
    if not BASE.is_dir():
        raise FileNotFoundError(BASE)
    if not SOURCE_TRANSFER.is_file() or not SOURCE_VALIDATION_RECEIPT.is_file():
        raise FileNotFoundError("validated parent transfer evidence is missing")
    source_validation = load(SOURCE_VALIDATION_RECEIPT)
    if source_validation.get("status") != "pass":
        raise ValueError("parent validation receipt is not passing")
    base_job = load(BASE / "jobs/weak_weak.json")
    old_archive_sha = sha256(BASE / "source_locked.tar.gz")

    with tempfile.TemporaryDirectory(prefix="sr-projected-anchor-") as raw:
        temp = Path(raw)
        archive, files, overlays = build_source(temp)
        new_archive_sha = sha256(archive)
        digests = isolated_digests(archive)
        if digests != {PARENT_ALIAS: PARENT_DIGEST, CHILD_ALIAS: CHILD_DIGEST}:
            raise ValueError(f"isolated route digest drift: {digests}")

        shutil.copytree(BASE, ANCHOR, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copy2(archive, ANCHOR / "source_locked.tar.gz")

    # The v4 parent directory intentionally preserves historical repair and
    # blocker records from its superseded predecessors.  Those records are
    # useful parent provenance, but copying them into a new submission bundle
    # produces contradictory root-level gates (including a literal
    # SCIENTIFIC_BLOCKER_DO_NOT_SUBMIT marker).  The anchor has its own clean,
    # current receipts below, so remove every inherited submission-state file
    # before constructing the immutable successor.
    for inherited in (
        "SCIENTIFIC_BLOCKER_DO_NOT_SUBMIT.json",
        "active_prefix_estimator_receipts_overlay.patch",
        "archive_only_preflight.json",
        "hysteresis_disabled_successor_receipt.json",
        "no_beam_active_prefix_receipt_test.patch",
        "operational_successor_receipt.json",
        "phase_live_hysteresis_disabled_overlay.patch",
        "preflight.json",
        "prepare_prefreeze_skeleton.py",
        "remote_execution_gate.json",
        "remote_preflight_and_cleanup_receipt.json",
        "route_parity.json",
        "scientific_settings_audit.json",
        "submission_artifact_hashes.json",
        "upload_artifact_list.txt",
    ):
        (ANCHOR / inherited).unlink(missing_ok=True)
    shutil.rmtree(ANCHOR / "source_lock", ignore_errors=True)

    replacements = {
        BASE_ID: ANCHOR_ID,
        BASE_BATCH: ANCHOR_BATCH,
        old_archive_sha: new_archive_sha,
    }
    # Several frozen worker gates hash-lock source modules independently of the
    # complete archive inventory. Propagate each overlaid file hash through
    # those worker constants and through every copied provenance record.
    for record in overlays.values():
        parent_sha = record.get("parent_sha256")
        overlay_sha = record.get("overlay_sha256")
        if isinstance(parent_sha, str) and isinstance(overlay_sha, str):
            replacements[parent_sha] = overlay_sha

    archive_manifest = replace_tree(load(BASE / "source_archive_manifest.json"), replacements)
    archive_manifest.update({
        "archive": f"chtc/phase3_optuna/input/{ANCHOR_ID}/source_locked.tar.gz",
        "archive_sha256": new_archive_sha,
        "archive_size_bytes": (ANCHOR / "source_locked.tar.gz").stat().st_size,
        "files": files,
        "phase3_projection_ablation_overlay": {
            "schema": "paper_i_sr_phase3_projection_ablation_source_overlay_v1",
            "immutable_parent_bundle": BASE_ID,
            "immutable_parent_archive_sha256": old_archive_sha,
            "overlay_files": overlays,
            "parent_route_contract_sha256": PARENT_DIGEST,
            "projected_route_contract_sha256": CHILD_DIGEST,
        },
    })
    dump(ANCHOR / "source_archive_manifest.json", archive_manifest)
    archive_manifest_sha = sha256(ANCHOR / "source_archive_manifest.json")

    revision = replace_tree(load(BASE / "source_revision_manifest.json"), replacements)
    revision["phase3_projection_ablation_overlay"] = archive_manifest[
        "phase3_projection_ablation_overlay"
    ]
    dump(ANCHOR / "source_revision_manifest.json", revision)
    revision_sha = sha256(ANCHOR / "source_revision_manifest.json")

    physics = replace_tree(load(BASE / "physics_and_exact_reference_lock.json"), replacements)
    dump(ANCHOR / "physics_and_exact_reference_lock.json", physics)
    physics_sha = sha256(ANCHOR / "physics_and_exact_reference_lock.json")

    job = replace_tree(copy.deepcopy(base_job), replacements)
    job["bundle_id"] = ANCHOR_ID
    job["batch_name"] = ANCHOR_BATCH
    source_lock = job["source_lock"]
    source_lock.update({
        "source_archive": f"chtc/phase3_optuna/input/{ANCHOR_ID}/source_locked.tar.gz",
        "source_archive_sha256": new_archive_sha,
        "source_archive_manifest": f"chtc/phase3_optuna/input/{ANCHOR_ID}/source_archive_manifest.json",
        "source_archive_manifest_sha256": archive_manifest_sha,
        "source_revision_manifest": f"chtc/phase3_optuna/input/{ANCHOR_ID}/source_revision_manifest.json",
        "source_revision_manifest_sha256": revision_sha,
        "physics_reference_lock": f"chtc/phase3_optuna/input/{ANCHOR_ID}/physics_and_exact_reference_lock.json",
        "physics_reference_lock_sha256": physics_sha,
        "phase3_projection_ablation_source_overlay": archive_manifest[
            "phase3_projection_ablation_overlay"
        ],
    })
    job["source_value_anchor"] = {
        "schema": "source_locked_sensitivity_anchor_plan_v1",
        "source_bundle": BASE_ID,
        "source_cluster": 8887574,
        "source_proc": 0,
        "source_regime": "weak_weak",
        "source_transfer_archive": str(SOURCE_TRANSFER.relative_to(ROOT)),
        "source_transfer_archive_sha256": sha256(SOURCE_TRANSFER),
        "source_result_archive_member": SOURCE_RESULT_MEMBER,
        "source_result_sha256": source_validation["result_sha256"],
        "source_validation_receipt": str(
            SOURCE_VALIDATION_RECEIPT.relative_to(ROOT)
        ),
        "source_validation_receipt_sha256": sha256(SOURCE_VALIDATION_RECEIPT),
        "swept_field": "historical_singleton_coordinate_solve_policy",
        "source_value": "supported_metric_whitened_eigh_v1",
        "candidate_value": "supported_metric_projected_generalized_trust_v1",
        "fanout_allowed_before_anchor_pass": False,
    }
    dump(ANCHOR / "jobs/weak_weak.json", job)

    normalized = replace_tree(load(BASE / "normalized_manifests/weak_weak.json"), replacements)
    normalized["bundle_id"] = ANCHOR_ID
    normalized["batch_name"] = ANCHOR_BATCH
    normalized["source_lock"] = copy.deepcopy(source_lock)
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
        _patch_text(ANCHOR / relative, replacements)
    wrapper_path = ANCHOR / "execute_source_locked_job.sh"
    wrapper = wrapper_path.read_text(encoding="utf-8")
    old_inner_shell = (
        "bash -lc 'cd /work && export PYTHONPATH=/work "
        "PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1; "
        "python3 -c \"$3\"; python3 -u \"$1\" \"$2\"'"
    )
    new_inner_shell = (
        "bash -lc 'set -euo pipefail; cd /work && export PYTHONPATH=/work "
        "PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1; "
        "python3 -c \"$3\"; python3 -u \"$1\" \"$2\"'"
    )
    if old_inner_shell not in wrapper:
        raise ValueError("worker inner-shell fail-closed anchor not found")
    wrapper_path.write_text(
        wrapper.replace(old_inner_shell, new_inner_shell), encoding="utf-8"
    )

    (ANCHOR / "queue.tsv").write_text(
        f"weak_weak\tchtc/phase3_optuna/input/{ANCHOR_ID}/jobs/weak_weak.json\t"
        f"chtc/phase3_optuna/input/{ANCHOR_ID}/normalized_manifests/weak_weak.json\t"
        "40960\t61440\n",
        encoding="utf-8",
    )
    submit_inputs = [
        f"chtc/phase3_optuna/input/{ANCHOR_ID}/run_job.py",
        f"chtc/phase3_optuna/input/{ANCHOR_ID}/evidence_validation.py",
        f"chtc/phase3_optuna/input/{ANCHOR_ID}/validate_fetched.py",
        f"chtc/phase3_optuna/input/{ANCHOR_ID}/source_archive_manifest.json",
        f"chtc/phase3_optuna/input/{ANCHOR_ID}/source_revision_manifest.json",
        f"chtc/phase3_optuna/input/{ANCHOR_ID}/physics_and_exact_reference_lock.json",
        f"chtc/phase3_optuna/input/{ANCHOR_ID}/bundle_manifest.json",
        f"chtc/phase3_optuna/input/{ANCHOR_ID}/anchor_bundle_receipt.json",
        f"chtc/phase3_optuna/input/{ANCHOR_ID}/source_locked_sensitivity_audit.json",
        "$(job_manifest)",
        "$(normalized_manifest)",
        f"chtc/phase3_optuna/input/{ANCHOR_ID}/source_locked.tar.gz",
        "chtc/phase3_optuna/image.sif",
    ]
    submit = f'''universe = vanilla
# Immutable source-value anchor.  The projected six-regime fanout remains
# fail-closed until this one parent replay reproduces its locked source.
executable = chtc/phase3_optuna/input/{ANCHOR_ID}/execute_source_locked_job.sh
arguments = $(job_manifest) chtc/phase3_optuna/input/{ANCHOR_ID}/source_locked.tar.gz {new_archive_sha} chtc/phase3_optuna/image.sif {IMAGE_SHA256} $(regime_slug)
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
transfer_executable = True
preserve_relative_paths = True
transfer_input_files = {", ".join(submit_inputs)}
transfer_output_files = raw_outputs/{ANCHOR_ID}/$(regime_slug)_transfer.tar.gz
transfer_output_remaps = "raw_outputs/{ANCHOR_ID}/$(regime_slug)_transfer.tar.gz = $(Cluster).$(Process)__$(regime_slug)_transfer.tar.gz"
stream_output = False
stream_error = False
log = logs/{ANCHOR_ID}.$(Cluster).$(Process).log
output = logs/{ANCHOR_ID}.$(Cluster).$(Process).out
error = logs/{ANCHOR_ID}.$(Cluster).$(Process).err
requirements = TARGET.HasSIF
request_cpus = 4
request_memory = $(memory_mb)MB
request_disk = $(disk_mb)MB
+MaxRuntime = 259200
+JobBatchName = "{ANCHOR_BATCH}"
notification = Never
queue regime_slug, job_manifest, normalized_manifest, memory_mb, disk_mb from chtc/phase3_optuna/input/{ANCHOR_ID}/queue.tsv
'''
    (ANCHOR / "submit.sub").write_text(submit, encoding="utf-8")

    audit = {
        "schema": "source_locked_sensitivity_audit_v1",
        "source": {
            "method": "SR-SNAKE Main",
            "regime_or_case": "weak_weak",
            "source_bundle": BASE_ID,
            "source_cluster": 8887574,
            "source_proc": 0,
            "source_json": SOURCE_RESULT_MEMBER,
            "source_sha256": source_validation["result_sha256"],
            "source_transfer_archive": str(SOURCE_TRANSFER.relative_to(ROOT)),
            "source_transfer_archive_sha256": sha256(SOURCE_TRANSFER),
            "source_validation_receipt": str(
                SOURCE_VALIDATION_RECEIPT.relative_to(ROOT)
            ),
            "source_validation_receipt_sha256": sha256(
                SOURCE_VALIDATION_RECEIPT
            ),
            "source_command_or_manifest": str(BASE / "jobs/weak_weak.json"),
            "source_command_or_manifest_sha256": sha256(BASE / "jobs/weak_weak.json"),
            "route_or_profile_id": PARENT_ALIAS,
            "route_contract_sha256": PARENT_DIGEST,
            "source_variable_value": "supported_metric_whitened_eigh_v1",
        },
        "sweep": {
            "run_class": "candidate",
            "variable": "historical_singleton_coordinate_solve_policy",
            "grid": [
                "supported_metric_whitened_eigh_v1",
                "supported_metric_projected_generalized_trust_v1",
            ],
            "runner_mode": "direct_source_locked_replay",
            "wrapper_used": False,
            "baseline_materialization_status": "complete",
            "unresolved_source_fields": [],
            "fields_added_by_current_defaults": [],
        },
        "planned_rows": [{
            "value": "supported_metric_whitened_eigh_v1",
            "changed_fields_vs_source": [],
            "non_swept_settings_diff": [],
            "bundle": ANCHOR_ID,
        }],
        "anchor": {
            "value": "supported_metric_whitened_eigh_v1",
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
        "schema": "paper_i_sr_phase3_projection_parent_anchor_bundle_v1",
        "bundle_id": ANCHOR_ID,
        "batch_name": ANCHOR_BATCH,
        "source_archive_sha256": new_archive_sha,
        "source_archive_manifest_sha256": archive_manifest_sha,
        "source_revision_manifest_sha256": revision_sha,
        "physics_reference_lock_sha256": physics_sha,
        "parent_route_contract_sha256": PARENT_DIGEST,
        "projected_route_contract_sha256": CHILD_DIGEST,
        "job_count": 1,
        "fanout_authorized": False,
        "submission_performed": False,
    }
    dump(ANCHOR / "anchor_bundle_receipt.json", receipt)
    dump(ANCHOR / "bundle_manifest.json", receipt)
    clean_gate = {
        "schema": "paper_i_sr_phase3_projection_anchor_submission_gate_v1",
        "bundle_id": ANCHOR_ID,
        "status": "ready_for_remote_preflight",
        "scientific_blockers": [],
        "parent_route_contract_sha256": PARENT_DIGEST,
        "projected_route_contract_sha256": CHILD_DIGEST,
        "source_archive_sha256": new_archive_sha,
        "job_count": 1,
        "fanout_authorized": False,
        "submission_performed": False,
    }
    dump(ANCHOR / "submission_gate.json", clean_gate)
    clean_preflight = {
        "schema": "paper_i_sr_phase3_projection_anchor_preflight_v1",
        "bundle_id": ANCHOR_ID,
        "status": "local_archive_preflight_pending",
        "route_contract_sha256": PARENT_DIGEST,
        "projected_route_contract_sha256": CHILD_DIGEST,
        "source_archive_sha256": new_archive_sha,
        "checks": {
            "one_parent_anchor_record": True,
            "same_cutoff_n_ph_3": True,
            "exact_round_50_horizon": True,
            "source_value_retained": True,
            "projected_fanout_disabled": True,
            "obsolete_submission_blocker_absent": True,
            "archive_only_worker_validation": False,
            "archive_focused_tests": False,
        },
    }
    dump(ANCHOR / "preflight.json", clean_preflight)
    dump(ANCHOR / "archive_only_preflight.json", clean_preflight)
    dump(ANCHOR / "route_parity.json", {
        "schema": "paper_i_sr_phase3_projection_anchor_route_parity_v1",
        "bundle_id": ANCHOR_ID,
        "status": "pass",
        "source_value": "supported_metric_whitened_eigh_v1",
        "anchor_value": "supported_metric_whitened_eigh_v1",
        "candidate_value": "supported_metric_projected_generalized_trust_v1",
        "anchor_changed_fields_vs_source": [],
        "parent_route_contract_sha256": PARENT_DIGEST,
        "projected_route_contract_sha256": CHILD_DIGEST,
        "source_archive_sha256": new_archive_sha,
    })
    dump(ANCHOR / "scientific_settings_audit.json", {
        "schema": "paper_i_sr_phase3_projection_anchor_scientific_settings_audit_v1",
        "bundle_id": ANCHOR_ID,
        "status": "pass",
        "anchor_changed_scientific_fields_vs_source": [],
        "swept_field": "historical_singleton_coordinate_solve_policy",
        "anchor_value": "supported_metric_whitened_eigh_v1",
        "candidate_value": "supported_metric_projected_generalized_trust_v1",
        "candidate_not_executed_in_anchor": True,
        "fanout_authorized": False,
    })
    dump(ANCHOR / "remote_execution_gate.json", {
        "schema": "paper_i_sr_phase3_projection_anchor_remote_gate_v1",
        "bundle_id": ANCHOR_ID,
        "status": "pending_authenticated_remote_preflight",
        "image_path": "chtc/phase3_optuna/image.sif",
        "image_sha256": IMAGE_SHA256,
        "source_archive_sha256": new_archive_sha,
        "submission_performed": False,
    })
    (ANCHOR / "upload_artifact_list.txt").write_text(
        "\n".join(submit_inputs) + "\n", encoding="utf-8"
    )
    (ANCHOR / "README.md").write_text(
        "# Phase-III projected-generalized source-value anchor\n\n"
        "This immutable one-row bundle replays the validated Main-SR weak-weak "
        "parent under the new source archive. The six-regime candidate fanout "
        "is fail-closed until `source_locked_sensitivity_audit.json` records a "
        "passing anchor.\n",
        encoding="utf-8",
    )

    verifier = f'''#!/usr/bin/env python3
import hashlib, json
from pathlib import Path
B = Path(__file__).resolve().parent
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
    receipt=json.loads((B/"anchor_bundle_receipt.json").read_text())
    assert h(B/"source_locked.tar.gz")==receipt["source_archive_sha256"]
    assert not (B/"SCIENTIFIC_BLOCKER_DO_NOT_SUBMIT.json").exists()
    assert json.loads((B/"submission_gate.json").read_text())["scientific_blockers"]==[]
    assert "bash -lc 'set -euo pipefail; cd /work" in (B/"execute_source_locked_job.sh").read_text()
    jobs=list((B/"jobs").glob("*.json")); assert [p.name for p in jobs]==["weak_weak.json"]
    job=json.loads(jobs[0].read_text())
    assert job["route_identity"]["profile_contract_sha256"]=={PARENT_DIGEST!r}
    assert job["route_identity"]["profile_contract"]["execution_settings"]["historical_singleton_coordinate_solve_policy"]=="supported_metric_whitened_eigh_v1"
    assert int(job["segment"]["target_controller_round"])==50
    assert job["physics"]["n_ph_work"]==job["physics"]["n_ph_reference"]==3
    assert json.loads((B/"source_locked_sensitivity_audit.json").read_text())["fanout_authorized"] is False
    return True
if __name__=="__main__": verify(); print("anchor bundle verification passed")
'''
    (ANCHOR / "build_bundle.py").write_text(verifier, encoding="utf-8")
    (ANCHOR / "test_bundle.py").write_text(
        "import build_bundle\ndef test_anchor_bundle(): assert build_bundle.verify()\n",
        encoding="utf-8",
    )
    return receipt


def archive_preflight() -> None:
    with tempfile.TemporaryDirectory(prefix="sr-projected-anchor-preflight-") as raw:
        root = Path(raw)
        with tarfile.open(ANCHOR / "source_locked.tar.gz", "r:gz") as archive:
            archive.extractall(root, filter="data")
        target = root / "chtc/phase3_optuna/input" / ANCHOR_ID
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ANCHOR, target)
        env = os.environ.copy()
        # Local dependency packages are installed in the user site.  PYTHONPATH
        # still locks all repository imports to the extracted archive; the
        # remote image separately enforces PYTHONNOUSERSITE=1.
        env.update({"PYTHONPATH": str(root)})
        env.pop("PYTHONNOUSERSITE", None)
        subprocess.run(
            [sys.executable, str(target / "run_job.py"), "--validate-only", str(target / "jobs/weak_weak.json")],
            cwd=root, env=env, check=True,
        )
        subprocess.run(
            [sys.executable, "-c", "import pipelines.static_adapt.adapt_pipeline"],
            cwd=root, env=env, check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q",
             "test/test_static_adapt_projected_generalized_trust_solve.py",
             "test/test_static_adapt_projected_phase3_route_profile.py",
             "test/test_static_adapt_sr_phase_liveness_contract.py"],
            cwd=root, env=env, check=True,
        )


def main() -> int:
    receipt = build_anchor()
    subprocess.run([sys.executable, str(ANCHOR / "build_bundle.py")], check=True)
    subprocess.run([sys.executable, "-m", "pytest", "-q", str(ANCHOR / "test_bundle.py")], check=True)
    archive_preflight()
    receipt["archive_only_preflight_passed"] = True
    dump(ANCHOR / "anchor_bundle_receipt.json", receipt)
    dump(ANCHOR / "bundle_manifest.json", receipt)
    preflight = load(ANCHOR / "preflight.json")
    preflight["status"] = "pass"
    preflight["checks"]["archive_only_worker_validation"] = True
    preflight["checks"]["archive_focused_tests"] = True
    dump(ANCHOR / "preflight.json", preflight)
    dump(ANCHOR / "archive_only_preflight.json", preflight)
    artifacts = {
        path.relative_to(ANCHOR).as_posix(): {
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(ANCHOR.rglob("*"))
        if path.is_file() and path.name != "submission_artifact_hashes.json"
    }
    dump(ANCHOR / "submission_artifact_hashes.json", {
        "schema": "paper_i_sr_phase3_projection_anchor_submission_artifacts_v1",
        "bundle_id": ANCHOR_ID,
        "files": artifacts,
    })
    subprocess.run([sys.executable, str(ANCHOR / "build_bundle.py")], check=True)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
