#!/usr/bin/env python3
"""Derive the fixed-source greedy Phase-III batch-3 CHTC bundle from v6.

The immutable v6 combinatorial bundle is the executable parent.  This builder
changes only the Phase-III post-shortlist batch selector from exhaustive
combinatorial reduced-plane selection to fixed-source marginal greedy
reduced-plane selection.  Phase II remains singleton, the batch cap/target
remain three, and the selected batch still receives one joint supported-FS
trust solve followed by one full accepted-ansatz supported-FS-whitened refit.
"""

from __future__ import annotations

import copy
import difflib
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


REPO = Path(__file__).resolve().parents[2]
INPUT = REPO / "chtc/phase3_optuna/input"
PARENT_ID = (
    "paper_i_hh_sr_snake_appendix_phase3_batch3_full_response_symmetric_cost_"
    "noprune_nobeam_no_ordinary_novelty_all_six_r50_20260718_v6_chtc"
)
OUTPUT_ID = (
    "paper_i_hh_sr_snake_appendix_phase3_greedy_batch3_full_response_"
    "symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_r50_"
    "20260720_v5_chtc"
)
PARENT = INPUT / PARENT_ID
OUTPUT = INPUT / OUTPUT_ID
PARENT_BATCH = (
    "paper-i-hh-sr-appendix-phase3-batch3-fullresp-symcost-noprune-"
    "nobeam-nonovelty-six-r50-20260718-v6"
)
OUTPUT_BATCH = (
    "paper-i-hh-sr-appendix-phase3-greedy-batch3-fullresp-symcost-"
    "noprune-nobeam-nonovelty-six-r50-20260720-v5"
)
PARENT_PROFILE_REQUEST = "sr_snake_no_prune_symmetric_cost_phase3_batch_v1"
OUTPUT_PROFILE_REQUEST = (
    "sr_snake_no_prune_symmetric_cost_phase3_greedy_batch_v1"
)
PARENT_PROFILE = (
    "supported_whitened_adaptive_trust_full_response_symmetric_cost_"
    "no_prune_phase3_batch_v1"
)
OUTPUT_PROFILE = (
    "supported_whitened_adaptive_trust_full_response_symmetric_cost_"
    "no_prune_phase3_greedy_batch_v1"
)
PARENT_ROUTE = "27df701ab280c02422e7030ec60a77d37ff20b73132ae4824cc41017f93fa050"
PARENT_SOURCE = "f11607321e426d73627910a1da76a22a96f4d4bd82f66708b5b202b2e5a61453"
PARENT_STATE = (
    "frozen_phase3_batch3_hysteresis_disabled_v4_plus_serialized_"
    "zero_extent_matrix_receipt_repair_v5_plus_batch_selector_workspace_"
    "receipt_repair_v6"
)
OUTPUT_STATE = (
    "frozen_phase3_batch3_v6_plus_fixed_source_greedy_selection_v1"
)
OLD_MODE = "combinatorial_reduced_plane"
NEW_MODE = "greedy_reduced_plane"
SOURCE_AUTHORITY = (
    "v6_archive_sha256_plus_exact_fixed_source_greedy_mode_derivation_v1"
)
DERIVATION_SCHEMA = "paper_i_hh_sr_phase3_greedy_batch_derivation_v1"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            name = PurePosixPath(member.name)
            if (
                name.is_absolute()
                or ".." in name.parts
                or member.issym()
                or member.islnk()
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"unsafe source member: {member.name}")
        handle.extractall(destination, filter="data")


def deterministic_archive(root: Path) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as handle:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            relative = path.relative_to(root).as_posix()
            data = path.read_bytes()
            info = tarfile.TarInfo(relative)
            info.size = len(data)
            info.mode = path.stat().st_mode & 0o777
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            handle.addfile(info, io.BytesIO(data))
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb", filename="", mtime=0) as gz:
        gz.write(raw.getvalue())
    return compressed.getvalue()


def source_mode_transform(text: str) -> str:
    replacements = (
        (PARENT_PROFILE_REQUEST, OUTPUT_PROFILE_REQUEST),
        (PARENT_PROFILE, OUTPUT_PROFILE),
        ("no_prune_phase3_batch_v1", "no_prune_phase3_greedy_batch_v1"),
        ('"combinatorial_reduced_plane"', '"greedy_reduced_plane"'),
        ("'combinatorial_reduced_plane'", "'greedy_reduced_plane'"),
        ("Phase-III-only combinatorial batching", "Phase-III-only fixed-source greedy batching"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def transform_string(value: str, route: str, source: str) -> str:
    replacements = (
        (PARENT_ID, OUTPUT_ID),
        (PARENT_BATCH, OUTPUT_BATCH),
        (PARENT_PROFILE_REQUEST, OUTPUT_PROFILE_REQUEST),
        (PARENT_PROFILE, OUTPUT_PROFILE),
        (PARENT_ROUTE, route),
        (PARENT_SOURCE, source),
        (PARENT_STATE, OUTPUT_STATE),
    )
    for old, new in replacements:
        value = value.replace(old, new)
    if value == OLD_MODE:
        return NEW_MODE
    return value


def transform_json(value: Any, route: str, source: str) -> Any:
    if isinstance(value, dict):
        return {
            transform_string(str(key), route, source): transform_json(item, route, source)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [transform_json(item, route, source) for item in value]
    if isinstance(value, str):
        return transform_string(value, route, source)
    return value


def restore_inherited_receipts(value: Any, parent_revision: dict[str, Any]) -> None:
    if isinstance(value, dict):
        if "empty_gram_receipt_repair" in value:
            value["empty_gram_receipt_repair"] = copy.deepcopy(
                parent_revision["empty_gram_receipt_repair"]
            )
        if "batch_selector_workspace_receipt_repair" in value:
            value["batch_selector_workspace_receipt_repair"] = copy.deepcopy(
                parent_revision["batch_selector_workspace_receipt_repair"]
            )
        if "phase3_batch_overlay" in value:
            value["phase3_batch_overlay"] = copy.deepcopy(
                parent_revision["phase3_batch_overlay"]
            )
        for item in value.values():
            restore_inherited_receipts(item, parent_revision)
    elif isinstance(value, list):
        for item in value:
            restore_inherited_receipts(item, parent_revision)


def route_probe(source_root: Path) -> tuple[str, dict[str, Any]]:
    code = (
        "import json; from pipelines.static_adapt.sr_snake_route_profile import "
        "canonical_sr_snake_contract,canonical_sr_snake_contract_sha256; "
        f"p={OUTPUT_PROFILE_REQUEST!r}; print(json.dumps({{'digest':"
        "canonical_sr_snake_contract_sha256(p),'contract':canonical_sr_snake_contract(p)},"
        "sort_keys=True,allow_nan=False))"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source_root)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=source_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr)
    payload = json.loads(completed.stdout)
    return str(payload["digest"]), payload["contract"]


def patch_runtime_validator(path: Path, *, fetched: bool = False) -> None:
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r'PROFILE = \(\n\s*"supported_whitened_adaptive_trust_full_response_symmetric_cost_"\n\s*"no_prune_phase3_batch_v1"\n\)',
        f'PROFILE = {OUTPUT_PROFILE!r}',
        text,
        count=1,
    )
    text = re.sub(
        r'SOURCE_LOCK_STATE = \(\n\s*"frozen_phase3_batch3_hysteresis_disabled_v4_plus_serialized_"\n\s*"zero_extent_matrix_receipt_repair_v5_plus_batch_selector_workspace_"\n\s*"receipt_repair_v6"\n\)',
        f'SOURCE_LOCK_STATE = {OUTPUT_STATE!r}',
        text,
        count=1,
    )
    anchor = "BATCH_OVERLAY_SHA256 = ("
    insertion = (
        f'COMBINATORIAL_PARENT_DIGEST = "{PARENT_ROUTE}"\n'
        f'GREEDY_PARENT_SOURCE_ARCHIVE_SHA256 = "{PARENT_SOURCE}"\n'
        "GREEDY_DERIVATION_SCHEMA = \"paper_i_hh_sr_phase3_greedy_batch_derivation_v1\"\n"
    )
    if insertion not in text:
        text = text.replace(anchor, insertion + anchor, 1)
    digest_name = "PROFILE_DIGEST" if fetched else "DIGEST"
    text = text.replace(
        'revision_empty_gram_repair.get("route_contract_sha256")\n        != '
        + digest_name,
        'revision_empty_gram_repair.get("route_contract_sha256")\n        != COMBINATORIAL_PARENT_DIGEST',
    )
    text = text.replace(
        f'revision_empty_gram_repair.get("route_contract_sha256") != {digest_name}',
        'revision_empty_gram_repair.get("route_contract_sha256") != COMBINATORIAL_PARENT_DIGEST',
    )
    text = text.replace(
        'revision_workspace_repair.get("route_contract_sha256")\n        != '
        + digest_name,
        'revision_workspace_repair.get("route_contract_sha256")\n        != COMBINATORIAL_PARENT_DIGEST',
    )
    text = text.replace(
        f'revision_workspace_repair.get("route_contract_sha256") != {digest_name}',
        'revision_workspace_repair.get("route_contract_sha256") != COMBINATORIAL_PARENT_DIGEST',
    )
    if fetched:
        text = text.replace(
            'revision_workspace_repair.get("successor_source_archive_sha256")\n        != source_archive_sha256',
            'revision_workspace_repair.get("successor_source_archive_sha256")\n        != GREEDY_PARENT_SOURCE_ARCHIVE_SHA256',
        )
    else:
        text = text.replace(
            'revision_workspace_repair.get("successor_source_archive_sha256")\n        != source_archive_sha256',
            'revision_workspace_repair.get("successor_source_archive_sha256")\n        != GREEDY_PARENT_SOURCE_ARCHIVE_SHA256',
        )
    old_authority = (
        '"v5_archive_sha256_plus_exact_batch_selector_workspace_receipt_"\n'
        '        "repair_inventory_v6"'
    )
    text = text.replace(old_authority, repr(SOURCE_AUTHORITY))
    path.write_text(text, encoding="utf-8")


def write_verifier(route: str, source: str, patch_sha: str) -> None:
    verifier = f'''#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, tarfile, tempfile, subprocess, sys, os
from pathlib import Path
BUNDLE = Path(__file__).resolve().parent
ROUTE = {route!r}
SOURCE = {source!r}
PARENT_ROUTE = {PARENT_ROUTE!r}
PARENT_SOURCE = {PARENT_SOURCE!r}
PROFILE = {OUTPUT_PROFILE_REQUEST!r}
MODE = {NEW_MODE!r}
PATCH_SHA = {patch_sha!r}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify():
    assert sha(BUNDLE / "source_locked.tar.gz") == SOURCE
    assert sha(BUNDLE / "phase3_greedy_batch_mode_overlay.patch") == PATCH_SHA
    manifest = json.loads((BUNDLE / "source_archive_manifest.json").read_text())
    assert manifest["archive_sha256"] == SOURCE
    assert manifest["greedy_batch_derivation"]["parent_source_archive_sha256"] == PARENT_SOURCE
    with tarfile.open(BUNDLE / "source_locked.tar.gz", "r:gz") as handle:
        files = {{m.name: handle.extractfile(m).read() for m in handle if m.isfile()}}
    assert set(files) == set(manifest["files"])
    for name, data in files.items():
        assert hashlib.sha256(data).hexdigest() == manifest["files"][name]["sha256"]
    jobs = sorted((BUNDLE / "jobs").glob("*.json"))
    assert len(jobs) == 6
    for path in jobs:
        job = json.loads(path.read_text())
        assert job["bundle_id"] == {OUTPUT_ID!r}
        route = job["route_identity"]
        assert route["profile_request"] == PROFILE
        assert route["profile_contract_sha256"] == ROUTE
        settings = route["profile_contract"]["execution_settings"]
        assert settings["phase2_enable_batching"] is False
        assert settings["phase3_enable_batching"] is True
        assert settings["phase2_batch_selection_mode"] == MODE
        assert settings["phase3_batch_selection_mode"] == MODE
        assert settings["phase3_batch_target_size"] == 3
        assert settings["phase3_batch_size_cap"] == 3
        assert int(job["segment"]["target_controller_round"]) == 50
        assert job["source_lock"]["source_archive_sha256"] == SOURCE
    audit = json.loads((BUNDLE / "source_locked_sensitivity_audit.json").read_text())
    assert audit["status"] == "pass_exact_one_mechanism_change"
    return True
if __name__ == "__main__":
    verify(); print("immutable fixed-source greedy batch-3 verification passed")
'''
    (OUTPUT / "build_bundle.py").write_text(verifier, encoding="utf-8")
    test = '''#!/usr/bin/env python3
from __future__ import annotations
import json, unittest
from pathlib import Path
import build_bundle
class GreedyBundleTest(unittest.TestCase):
    def test_bundle(self): self.assertTrue(build_bundle.verify())
    def test_exact_six_rows_and_cutoffs(self):
        for path in sorted((build_bundle.BUNDLE / "jobs").glob("*.json")):
            job=json.loads(path.read_text())
            strong = job["regime_slug"] in {"weak_strong","intermediate_strong","strong_strong_u8"}
            self.assertEqual(job["physics"]["n_ph_work"], 7 if strong else 3)
            self.assertEqual(job["physics"]["n_ph_reference"], 7 if strong else 3)
            self.assertEqual(job["segment"]["target_controller_round"], 50)
    def test_fixed_source_greedy_contract(self):
        job=json.loads(next((build_bundle.BUNDLE / "jobs").glob("*.json")).read_text())
        settings=job["route_identity"]["profile_contract"]["execution_settings"]
        self.assertFalse(settings["phase2_enable_batching"])
        self.assertTrue(settings["phase3_enable_batching"])
        self.assertEqual(settings["phase3_batch_selection_mode"], "greedy_reduced_plane")
        self.assertEqual(settings["phase3_batch_target_size"], 3)
        self.assertEqual(settings["phase3_batch_size_cap"], 3)
if __name__ == "__main__": unittest.main()
'''
    (OUTPUT / "test_bundle.py").write_text(test, encoding="utf-8")


def main() -> int:
    if OUTPUT.exists():
        # A failed first-pass probe may leave only the byte-for-byte template
        # copy.  That state is safe to resume; any derived artifact remains
        # immutable and fail-closed.
        if (
            not (OUTPUT / "source_locked.tar.gz").is_file()
            or (OUTPUT / "phase3_greedy_batch_mode_overlay.patch").exists()
            or (OUTPUT / "source_locked_sensitivity_audit.json").exists()
        ):
            raise FileExistsError(f"immutable output already exists: {OUTPUT}")
        resume_template_copy = True
    else:
        resume_template_copy = False
    if sha(PARENT / "source_locked.tar.gz") != PARENT_SOURCE:
        raise ValueError("parent v6 source archive drift")
    parent_revision = json.loads((PARENT / "source_revision_manifest.json").read_text())
    if not resume_template_copy:
        shutil.copytree(PARENT, OUTPUT)

    with tempfile.TemporaryDirectory(prefix="sr_greedy_batch3_") as tmp:
        root = Path(tmp)
        safe_extract(PARENT / "source_locked.tar.gz", root)
        changed: dict[str, tuple[str, str]] = {}
        for relative in (
            "pipelines/static_adapt/sr_snake_route_profile.py",
            "test/test_static_adapt_sr_phase3_batch_appendix_profile.py",
        ):
            path = root / relative
            old = path.read_text(encoding="utf-8")
            new = source_mode_transform(old)
            if old == new:
                raise ValueError(f"greedy source transformation was inert: {relative}")
            path.write_text(new, encoding="utf-8")
            changed[relative] = (old, new)
        route, contract = route_probe(root)
        archive_bytes = deterministic_archive(root)
        (OUTPUT / "source_locked.tar.gz").write_bytes(archive_bytes)
        source = sha_bytes(archive_bytes)
        patch_text = "".join(
            "".join(
                difflib.unified_diff(
                    old.splitlines(keepends=True),
                    new.splitlines(keepends=True),
                    fromfile=f"a/{relative}",
                    tofile=f"b/{relative}",
                )
            )
            for relative, (old, new) in changed.items()
        )
        (OUTPUT / "phase3_greedy_batch_mode_overlay.patch").write_text(
            patch_text, encoding="utf-8"
        )
        patch_sha = sha(OUTPUT / "phase3_greedy_batch_mode_overlay.patch")
        file_records = {
            path.relative_to(root).as_posix(): {
                "sha256": sha(path),
                "size_bytes": path.stat().st_size,
                "mode": format(path.stat().st_mode & 0o777, "04o"),
            }
            for path in sorted(p for p in root.rglob("*") if p.is_file())
        }

    derivation = {
        "schema": DERIVATION_SCHEMA,
        "classification": "scientific_one_mechanism_phase3_batch_selector_ablation",
        "parent_source_archive_sha256": PARENT_SOURCE,
        "parent_route_contract_sha256": PARENT_ROUTE,
        "child_source_archive_sha256": source,
        "child_route_contract_sha256": route,
        "overlay_path": "phase3_greedy_batch_mode_overlay.patch",
        "overlay_sha256": patch_sha,
        "changed_source_files": {
            relative: file_records[relative]["sha256"] for relative in changed
        },
        "only_scientific_setting_changed": "phase3_batch_selection_mode",
        "parent_value": OLD_MODE,
        "child_value": NEW_MODE,
        "phase2_batching_unchanged_off": True,
        "batch_target_and_cap_unchanged_three": True,
        "fixed_source_marginal_greedy": True,
        "single_joint_refit_after_selection": True,
    }

    # Transform executable wrappers, validators, Condor paths, and prose.
    for path in sorted(p for p in OUTPUT.rglob("*") if p.is_file()):
        if path.name in {
            "source_locked.tar.gz",
            "build_bundle.py",
            "test_bundle.py",
            "submission_artifact_hashes.json",
        } or path.suffix == ".patch" or path.suffix == ".json":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        text = transform_string(text, route, source)
        text = text.replace(f'"{OLD_MODE}"', f'"{NEW_MODE}"')
        text = text.replace(f"'{OLD_MODE}'", f"'{NEW_MODE}'")
        path.write_text(text, encoding="utf-8")

    # Transform all JSON contracts while retaining inherited repair receipts.
    for path in sorted(OUTPUT.rglob("*.json")):
        if path.name == "submission_artifact_hashes.json":
            continue
        payload = transform_json(json.loads(path.read_text()), route, source)
        restore_inherited_receipts(payload, parent_revision)
        dump(path, payload)

    # Rebuild the two source authority manifests around the derived archive.
    archive_manifest = transform_json(
        json.loads((PARENT / "source_archive_manifest.json").read_text()),
        route,
        source,
    )
    restore_inherited_receipts(archive_manifest, parent_revision)
    archive_manifest.update(
        {
            "archive": (OUTPUT / "source_locked.tar.gz").relative_to(REPO).as_posix(),
            "archive_sha256": source,
            "archive_size_bytes": (OUTPUT / "source_locked.tar.gz").stat().st_size,
            "file_count": len(file_records),
            "files": file_records,
            "worker_source_mode": OUTPUT_STATE,
            "greedy_batch_derivation": derivation,
        }
    )
    dump(OUTPUT / "source_archive_manifest.json", archive_manifest)

    revision = transform_json(parent_revision, route, source)
    restore_inherited_receipts(revision, parent_revision)
    revision.update(
        {
            "profile_request": OUTPUT_PROFILE_REQUEST,
            "profile_resolved": OUTPUT_PROFILE,
            "profile_contract_sha256": route,
            "worker_source_mode": OUTPUT_STATE,
            "executable_source_authority": SOURCE_AUTHORITY,
            "greedy_batch_derivation": derivation,
        }
    )
    critical = revision.get("critical_source_sha256", {})
    for relative in changed:
        if relative in critical:
            critical[relative] = file_records[relative]["sha256"]
    dump(OUTPUT / "source_revision_manifest.json", revision)

    archive_manifest_sha = sha(OUTPUT / "source_archive_manifest.json")
    revision_sha = sha(OUTPUT / "source_revision_manifest.json")
    physics_sha = sha(OUTPUT / "physics_and_exact_reference_lock.json")
    for directory in (OUTPUT / "jobs", OUTPUT / "normalized_manifests"):
        for path in sorted(directory.glob("*.json")):
            payload = json.loads(path.read_text())
            source_lock = payload["source_lock"]
            source_lock.update(
                {
                    "source_archive_sha256": source,
                    "source_archive_manifest_sha256": archive_manifest_sha,
                    "source_revision_manifest_sha256": revision_sha,
                    "physics_reference_lock_sha256": physics_sha,
                    "worker_source_mode": OUTPUT_STATE,
                    "greedy_batch_derivation": derivation,
                }
            )
            source_lock["empty_gram_receipt_repair"] = copy.deepcopy(
                parent_revision["empty_gram_receipt_repair"]
            )
            source_lock["batch_selector_workspace_receipt_repair"] = copy.deepcopy(
                parent_revision["batch_selector_workspace_receipt_repair"]
            )
            dump(path, payload)

    # Fail-closed runtime validators retain the inherited operational receipts.
    patch_runtime_validator(OUTPUT / "run_job.py")
    patch_runtime_validator(OUTPUT / "validate_fetched.py", fetched=True)

    audit = {
        "schema": "paper_i_hh_sr_source_locked_sensitivity_audit_v1",
        "created_utc": utc_now(),
        "status": "pass_exact_one_mechanism_change",
        "parent_bundle": PARENT.relative_to(REPO).as_posix(),
        "child_bundle": OUTPUT.relative_to(REPO).as_posix(),
        "parent_source_archive_sha256": PARENT_SOURCE,
        "child_source_archive_sha256": source,
        "parent_route_contract_sha256": PARENT_ROUTE,
        "child_route_contract_sha256": route,
        "allowed_execution_setting_diff": {
            "phase2_batch_selection_mode": {"parent": OLD_MODE, "child": NEW_MODE},
            "phase3_batch_selection_mode": {"parent": OLD_MODE, "child": NEW_MODE},
        },
        "conceptual_change": "phase3_fixed_source_combinatorial_to_marginal_greedy",
        "unchanged": {
            "phase2_batching": "off",
            "phase3_batch_target": 3,
            "phase3_batch_cap": 3,
            "phase3_response_scope": "full_active_plus_batch_v1",
            "accepted_refit": "full_supported_fs_whitened",
            "round_horizon": 50,
            "cutoffs": "n_ph3_weak_holstein_n_ph7_strong_holstein_same_cutoff",
        },
        "derivation": derivation,
    }
    dump(OUTPUT / "source_locked_sensitivity_audit.json", audit)

    # Refresh the bundle manifest's artifact hashes after all dependent files.
    bundle_manifest = json.loads((OUTPUT / "bundle_manifest.json").read_text())
    bundle_manifest.update(
        {
            "bundle_id": OUTPUT_ID,
            "batch_name": OUTPUT_BATCH,
            "source_archive_sha256": source,
            "source_lock_state": OUTPUT_STATE,
            "greedy_batch_derivation": derivation,
            "source_locked_sensitivity_audit": {
                "path": (OUTPUT / "source_locked_sensitivity_audit.json").relative_to(REPO).as_posix(),
                "sha256": sha(OUTPUT / "source_locked_sensitivity_audit.json"),
            },
        }
    )
    for key, filename in (
        ("source_archive_manifest", "source_archive_manifest.json"),
        ("source_revision_manifest", "source_revision_manifest.json"),
        ("physics_reference_lock", "physics_and_exact_reference_lock.json"),
        ("scientific_settings_audit", "scientific_settings_audit.json"),
        ("route_parity", "route_parity.json"),
        ("archive_only_preflight", "archive_only_preflight.json"),
        ("remote_preflight_and_cleanup_receipt", "remote_preflight_and_cleanup_receipt.json"),
    ):
        record = bundle_manifest.get(key)
        if isinstance(record, dict):
            record["path"] = (OUTPUT / filename).relative_to(REPO).as_posix()
            record["sha256"] = sha(OUTPUT / filename)
    dump(OUTPUT / "bundle_manifest.json", bundle_manifest)

    readme = f"""# Fixed-source greedy Phase-III batch-3 appendix bundle

This immutable six-regime bundle derives from the repaired v6 combinatorial
bundle.  The only scientific mechanism changed is the Phase-III post-shortlist
selector: `{OLD_MODE}` becomes `{NEW_MODE}`.  The greedy selector adds up to
three candidates using fixed-source marginal gains, then performs one joint
supported-FS trust solve and one full accepted-ansatz supported-FS-whitened
Powell refit.  Phase II remains singleton.

- batch: `{OUTPUT_BATCH}`
- source archive SHA-256: `{source}`
- route-contract SHA-256: `{route}`
- parent source SHA-256: `{PARENT_SOURCE}`
- parent route SHA-256: `{PARENT_ROUTE}`
- horizon: 50 controller rounds in all six regimes
- cutoff: n_ph=3 for weak-Holstein, n_ph=7 for strong-Holstein, same-cutoff references
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")
    write_verifier(route, source, patch_sha)

    upload = (OUTPUT / "upload_artifact_list.txt").read_text(encoding="utf-8")
    extra = (OUTPUT / "source_locked_sensitivity_audit.json").relative_to(REPO).as_posix()
    if extra not in upload.splitlines():
        (OUTPUT / "upload_artifact_list.txt").write_text(upload + extra + "\n", encoding="utf-8")

    inventory = {}
    for path in sorted(p for p in OUTPUT.rglob("*") if p.is_file()):
        if path.name == "submission_artifact_hashes.json" or "__pycache__" in path.parts:
            continue
        inventory[path.relative_to(REPO).as_posix()] = {
            "sha256": sha(path), "size_bytes": path.stat().st_size
        }
    dump(
        OUTPUT / "submission_artifact_hashes.json",
        {"schema": "paper_i_hh_sr_submission_artifact_hashes_v1", "artifacts": inventory},
    )

    subprocess.run([sys.executable, str(OUTPUT / "build_bundle.py")], check=True)
    subprocess.run([sys.executable, str(OUTPUT / "test_bundle.py")], check=True)
    print(json.dumps({
        "status": "pass_submission_ready_not_yet_submitted",
        "bundle": OUTPUT.relative_to(REPO).as_posix(),
        "batch": OUTPUT_BATCH,
        "source_archive_sha256": source,
        "route_contract_sha256": route,
        "jobs": 6,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
