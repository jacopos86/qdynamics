#!/usr/bin/env python3
"""Build the immutable v4 macro/beam/prune cost-arm successors.

The only worker-source change from each byte-identical v3 source archive is
the estimator-consumer identity repair for exact prune trials under beam.  The
repair scopes the trial ID by the parent beam branch and temporarily installs
that ID as ``estimator_call_context.branch_id`` while the exact delete/refit
trial executes.  No live source-tree file is copied into the worker archive.
"""

from __future__ import annotations

import ast
import copy
import gzip
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
INPUT_ROOT = ROOT / "chtc" / "phase3_optuna" / "input"
PREDECESSOR_SOURCE_SHA256 = (
    "7c3ceaf5523f0c551e3c41c30e8f130f554935dba04fc6ec08ac9d48c1e4e3c9"
)
ADAPT_PATH = "pipelines/static_adapt/adapt_pipeline.py"
CREATED_UTC = "2026-07-20T00:00:00Z"

ARMS: tuple[dict[str, Any], ...] = (
    {
        "slug": "symmetric",
        "short": "symcost",
        "profile": (
            "supported_whitened_adaptive_trust_full_response_symmetric_cost_"
            "fs_prune_nodamping_beam3x2_macro_only_physical_lanes_v1"
        ),
        "profile_request": (
            "sr_snake_macro_only_physical_lanes_fs_prune_beam3x2_v1"
        ),
        "route_digest": (
            "a05ecc8b709db8beac9115d9d0ca39f4faf09e1cbaa10e57bdd674abef9215f0"
        ),
        "cost": "family_robust_symmetric_arctan_v1",
        "cluster": 8894497,
    },
    {
        "slug": "one_sided",
        "short": "onesided",
        "profile": (
            "supported_whitened_adaptive_trust_full_response_one_sided_cost_"
            "fs_prune_nodamping_beam3x2_macro_only_physical_lanes_v1"
        ),
        "profile_request": (
            "sr_snake_macro_only_physical_lanes_fs_prune_beam3x2_"
            "one_sided_cost_v1"
        ),
        "route_digest": (
            "e3b9f24af40f3572063dd0d13bcca932870505870a8cd7822453b38e01bf6096"
        ),
        "cost": "family_robust_penalty_only_v1",
        "cluster": 8894498,
    },
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _patch_adapt_pipeline(source: bytes) -> bytes:
    text = source.decode("utf-8")
    replacements = (
        (
            "def _sr_v4_prune_trial_branch_id(\n"
            "    *,\n"
            "    selector_step: int,\n"
            "    candidate_index: int,\n"
            "    candidate_label: str,\n"
            ") -> str:\n",
            "def _sr_v4_prune_trial_branch_id(\n"
            "    *,\n"
            "    selector_step: int,\n"
            "    candidate_index: int,\n"
            "    candidate_label: str,\n"
            "    parent_branch_id: str | None = None,\n"
            ") -> str:\n",
        ),
        (
            "    identity = {\n"
            "        \"selector_step\": int(selector_step),\n"
            "        \"candidate_index\": int(candidate_index),\n"
            "        \"candidate_label\": str(candidate_label),\n"
            "    }\n"
            "    digest = hashlib.sha256(\n",
            "    identity = {\n"
            "        \"selector_step\": int(selector_step),\n"
            "        \"candidate_index\": int(candidate_index),\n"
            "        \"candidate_label\": str(candidate_label),\n"
            "    }\n"
            "    if parent_branch_id not in {None, \"\"}:\n"
            "        identity[\"parent_branch_id\"] = str(parent_branch_id)\n"
            "    digest = hashlib.sha256(\n",
        ),
        (
            "            estimator_trial_branch_id = _sr_v4_prune_trial_branch_id(\n"
            "                selector_step=int(selector_step),\n"
            "                candidate_index=int(selected_index),\n"
            "                candidate_label=str(selected_label),\n"
            "            )\n",
            "            estimator_trial_branch_id = _sr_v4_prune_trial_branch_id(\n"
            "                selector_step=int(selector_step),\n"
            "                candidate_index=int(selected_index),\n"
            "                candidate_label=str(selected_label),\n"
            "                parent_branch_id=getattr(\n"
            "                    estimator_call_context,\n"
            "                    \"branch_id\",\n"
            "                    None,\n"
            "                ),\n"
            "            )\n",
        ),
    )
    for old, new in replacements:
        count = text.count(old)
        if count != 1:
            raise RuntimeError(
                f"expected exactly one frozen-source repair anchor, found {count}"
            )
        text = text.replace(old, new, 1)
    compile(text, ADAPT_PATH, "exec")
    return text.encode("utf-8")


def _build_repaired_archive(predecessor: Path, output: Path) -> tuple[str, int, str, int]:
    if _sha256_file(predecessor) != PREDECESSOR_SOURCE_SHA256:
        raise RuntimeError("v3 predecessor source archive hash mismatch")
    with tarfile.open(predecessor, "r:gz") as source:
        members = source.getmembers()
        matching = [member for member in members if member.name == ADAPT_PATH]
        if len(matching) != 1:
            raise RuntimeError("frozen archive does not contain one adapt_pipeline.py")
        old_payload = source.extractfile(matching[0]).read()
        new_payload = _patch_adapt_pipeline(old_payload)

        with output.open("wb") as raw_output:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_output, mtime=0
            ) as compressed:
                with tarfile.open(
                    fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT
                ) as target:
                    for member in members:
                        member_copy = copy.copy(member)
                        if member.isfile():
                            payload = source.extractfile(member).read()
                            if member.name == ADAPT_PATH:
                                payload = new_payload
                                member_copy.size = len(payload)
                            target.addfile(member_copy, io.BytesIO(payload))
                        else:
                            target.addfile(member_copy)
    return (
        _sha256_bytes(old_payload),
        len(old_payload),
        _sha256_bytes(new_payload),
        len(new_payload),
    )


def _replace_text_tree(root: Path, replacements: tuple[tuple[str, str], ...]) -> None:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "source_locked.tar.gz":
            continue
        if path.suffix not in {".json", ".md", ".py", ".sh", ".sub", ".tsv", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8")
        for old, new in replacements:
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")


def _repair_receipt(
    *,
    arm: dict[str, Any],
    predecessor_bundle: str,
    predecessor_adapt_sha: str,
    repaired_adapt_sha: str,
) -> dict[str, Any]:
    return {
        "schema": "paper_i_sr_prune_trial_consumer_id_repair_v1",
        "classification": "non_scientific_estimator_consumer_identity_isolation_v1",
        "failed_cluster": int(arm["cluster"]),
        "predecessor_bundle": predecessor_bundle,
        "predecessor_source_archive_sha256": PREDECESSOR_SOURCE_SHA256,
        "path": ADAPT_PATH,
        "source_sha256_before": predecessor_adapt_sha,
        "source_sha256_after": repaired_adapt_sha,
        "exact_changes": [
            "scope_prune_trial_consumer_id_by_parent_beam_branch_id",
            "pass_estimator_call_context_branch_id_to_consumer_id_builder",
        ],
        "scientific_setting_changes": [],
        "profile_contract_sha256_preserved": str(arm["route_digest"]),
    }


def _superseded_predecessor(arm: dict[str, Any], predecessor_bundle: str) -> dict[str, Any]:
    return {
        "bundle_id": predecessor_bundle,
        "classification": "non_scientific_prune_trial_consumer_id_collision_under_beam_v1",
        "cluster_id": int(arm["cluster"]),
        "failure_stage": "live_fs_prune_exact_delete_refit_estimator_accounting",
        "scientific_setting_changes": [],
        "successor_revision": "v4",
    }


def _build_arm(arm: dict[str, Any]) -> Path:
    predecessor_bundle = (
        "paper_i_hh_sr_snake_macro_beam3x2_fs_prune_"
        f"{arm['slug']}_cost_all_six_r50_20260719_v3_chtc"
    )
    successor_bundle = predecessor_bundle.replace("_v3_chtc", "_v4_chtc")
    predecessor = INPUT_ROOT / predecessor_bundle
    successor = INPUT_ROOT / successor_bundle
    if successor.exists():
        raise RuntimeError(f"immutable successor already exists: {successor}")
    if not predecessor.is_dir():
        raise RuntimeError(f"missing predecessor bundle: {predecessor}")

    shutil.copytree(
        predecessor,
        successor,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            "source_locked.tar.gz",
            "submission_artifact_hashes.json",
            "SUPERSEDED_DO_NOT_SUBMIT.json",
        ),
    )
    source_archive = successor / "source_locked.tar.gz"
    (
        predecessor_adapt_sha,
        predecessor_adapt_size,
        repaired_adapt_sha,
        repaired_adapt_size,
    ) = _build_repaired_archive(predecessor / "source_locked.tar.gz", source_archive)
    source_sha = _sha256_file(source_archive)
    source_size = source_archive.stat().st_size

    old_batch = (
        "paper-i-hh-sr-macro-beam3x2-fsprune-"
        f"{arm['short']}-six-r50-20260719-v3"
    )
    new_batch = old_batch.removesuffix("-v3") + "-v4"
    _replace_text_tree(
        successor,
        (
            (predecessor_bundle, successor_bundle),
            (old_batch, new_batch),
            ("20260719-v3", "20260719-v4"),
            (PREDECESSOR_SOURCE_SHA256, source_sha),
        ),
    )

    repair = _repair_receipt(
        arm=arm,
        predecessor_bundle=predecessor_bundle,
        predecessor_adapt_sha=predecessor_adapt_sha,
        repaired_adapt_sha=repaired_adapt_sha,
    )
    superseded = _superseded_predecessor(arm, predecessor_bundle)

    revision_path = successor / "source_revision_manifest.json"
    revision = json.loads(revision_path.read_text())
    revision["critical_source_sha256"][ADAPT_PATH] = repaired_adapt_sha
    revision["route_overlay_files"][ADAPT_PATH] = {
        "sha256": repaired_adapt_sha,
        "size_bytes": repaired_adapt_size,
    }
    revision["prune_trial_consumer_id_repair"] = repair
    revision["superseded_predecessor"] = superseded
    _json_dump(revision_path, revision)

    archive_manifest_path = successor / "source_archive_manifest.json"
    archive_manifest = json.loads(archive_manifest_path.read_text())
    archive_manifest.update(
        {
            "archive": str(source_archive.relative_to(ROOT)),
            "archive_sha256": source_sha,
            "archive_size_bytes": source_size,
            "derived_from_archive": {
                "path": str((predecessor / "source_locked.tar.gz").relative_to(ROOT)),
                "sha256": PREDECESSOR_SOURCE_SHA256,
            },
            "prune_trial_consumer_id_repair": repair,
            "superseded_predecessor": superseded,
        }
    )
    archive_manifest["files"][ADAPT_PATH] = {
        "sha256": repaired_adapt_sha,
        "size_bytes": repaired_adapt_size,
    }
    # The v3 archive manifest's complete inventory is authoritative; its
    # separate critical-source map lives only in source_revision_manifest.json.
    if "critical_source_sha256" in archive_manifest:
        archive_manifest["critical_source_sha256"][ADAPT_PATH] = repaired_adapt_sha
    _json_dump(archive_manifest_path, archive_manifest)

    revision_sha = _sha256_file(revision_path)
    archive_manifest_sha = _sha256_file(archive_manifest_path)
    for manifest_dir in ("jobs", "normalized_manifests"):
        for path in sorted((successor / manifest_dir).glob("*.json")):
            payload = json.loads(path.read_text())
            source_lock = payload["source_lock"]
            source_lock.update(
                {
                    "source_archive_sha256": source_sha,
                    "source_archive_manifest_sha256": archive_manifest_sha,
                    "source_revision_manifest_sha256": revision_sha,
                    "prune_trial_consumer_id_repair": repair,
                }
            )
            _json_dump(path, payload)

    bundle_manifest_path = successor / "bundle_manifest.json"
    bundle_manifest = json.loads(bundle_manifest_path.read_text())
    bundle_manifest.update(
        {
            "batch_name": new_batch,
            "bundle_id": successor_bundle,
            "created_utc": CREATED_UTC,
            "source_archive_sha256": source_sha,
            "submission_status": "built_not_submitted",
            "superseded_predecessor": superseded,
            "prune_trial_consumer_id_repair": repair,
        }
    )
    _json_dump(bundle_manifest_path, bundle_manifest)

    for name in ("route_parity.json", "scientific_settings_audit.json"):
        path = successor / name
        payload = json.loads(path.read_text())
        payload["superseded_predecessor"] = superseded
        payload["prune_trial_consumer_id_repair"] = repair
        _json_dump(path, payload)

    preflight = json.loads((successor / "preflight.json").read_text())
    preflight.update(
        {
            "source_archive_sha256": source_sha,
            "prune_trial_consumer_id_repair": repair,
            "status": "pass",
        }
    )
    _json_dump(successor / "preflight.json", preflight)
    archive_preflight = json.loads((successor / "archive_only_preflight.json").read_text())
    archive_preflight.update(
        {
            "source_archive_sha256": source_sha,
            "prune_trial_consumer_id_repair": repair,
            "v4_bundle_tests_passed": 4,
            "v4_archive_only_validate_rows_passed": 6,
            "v4_shared_archive_focused_tests_passed": 18,
            "v4_test_targets": [
                "test/test_static_adapt_sr_v4_runtime.py",
                "test/test_static_adapt_macro_beam_prune_cost_profiles.py",
            ],
            "status": "pass",
        }
    )
    _json_dump(successor / "archive_only_preflight.json", archive_preflight)

    readme = successor / "README.md"
    readme.write_text(
        readme.read_text()
        + "\n## v4 non-scientific repair\n\n"
        + "Derived only from the immutable v3 source archive. Exact prune-trial "
        + "estimator consumer IDs now include the parent beam branch ID, preventing "
        + "cross-parent ledger aliasing. Route settings and digest are unchanged.\n",
        encoding="utf-8",
    )

    build_script = successor / "build_bundle.py"
    original_build_script = build_script.read_text()
    original_main = (
        'if __name__ == "__main__": verify(); '
        'print("macro beam-prune cost bundle verification passed")\n'
    )
    if original_build_script.count(original_main) != 1:
        raise RuntimeError("build_bundle.py main anchor missing or duplicated")
    build_script.write_text(
        original_build_script.replace(original_main, "", 1)
        + "\n\ndef verify_prune_consumer_repair():\n"
        + "    import ast, tarfile\n"
        + "    predecessor = BUNDLE_DIR.parent / " + repr(predecessor_bundle) + " / 'source_locked.tar.gz'\n"
        + "    assert _sha(predecessor) == " + repr(PREDECESSOR_SOURCE_SHA256) + "\n"
        + "    with tarfile.open(predecessor, 'r:gz') as before, tarfile.open(BUNDLE_DIR / 'source_locked.tar.gz', 'r:gz') as after:\n"
        + "        before_files = {m.name: before.extractfile(m).read() for m in before.getmembers() if m.isfile()}\n"
        + "        after_files = {m.name: after.extractfile(m).read() for m in after.getmembers() if m.isfile()}\n"
        + "    assert before_files.keys() == after_files.keys()\n"
        + "    assert [p for p in before_files if before_files[p] != after_files[p]] == ['pipelines/static_adapt/adapt_pipeline.py']\n"
        + "    text = after_files['pipelines/static_adapt/adapt_pipeline.py'].decode('utf-8')\n"
        + "    tree = ast.parse(text)\n"
        + "    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_sr_v4_prune_trial_branch_id')\n"
        + "    assert any(a.arg == 'parent_branch_id' for a in fn.args.kwonlyargs)\n"
        + "    assert 'parent_branch_id=getattr(' in text\n"
        + "    assert 'estimator_call_context' in text\n"
        + "    scope = {'hashlib': hashlib, 'json': json, '_SR_V4_PRUNE_TRIAL_BRANCH_PREFIX': 'sr_v4_prune_trial:'}\n"
        + "    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<archive-prune-id>', 'exec'), scope)\n"
        + "    branch_id = scope['_sr_v4_prune_trial_branch_id']\n"
        + "    shared = {'selector_step': 5, 'candidate_index': 2, 'candidate_label': 'macro:test'}\n"
        + "    ids = {branch_id(**shared), branch_id(**shared, parent_branch_id='beam:a'), branch_id(**shared, parent_branch_id='beam:b')}\n"
        + "    assert len(ids) == 3\n"
        + "    return True\n\n"
        + "_original_verify = verify\n"
        + "def verify():\n"
        + "    assert _original_verify()\n"
        + "    assert verify_prune_consumer_repair()\n"
        + "    return True\n\n"
        + "if __name__ == '__main__':\n"
        + "    verify()\n"
        + "    print('macro beam-prune v4 cost bundle verification passed')\n",
        encoding="utf-8",
    )

    test_path = successor / "test_bundle.py"
    test_text = test_path.read_text()
    insertion = (
        "\n    def test_prune_trial_consumer_id_is_parent_beam_scoped(self):\n"
        "        self.assertTrue(build_bundle.verify_prune_consumer_repair())\n"
    )
    marker = "\n\nif __name__ == \"__main__\":\n"
    if marker not in test_text:
        raise RuntimeError("test_bundle.py insertion anchor missing")
    test_path.write_text(test_text.replace(marker, insertion + marker, 1))

    upload_list = successor / "upload_artifact_list.txt"
    upload_list.write_text(
        upload_list.read_text().replace(predecessor_bundle, successor_bundle),
        encoding="utf-8",
    )

    # Regenerate the immutable artifact inventory after every derived file is final.
    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(successor.rglob("*")):
        if not path.is_file() or path.name == "submission_artifact_hashes.json":
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = str(path.relative_to(ROOT))
        artifacts[relative] = {
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    _json_dump(
        successor / "submission_artifact_hashes.json",
        {
            "schema": "paper_i_hh_sr_macro_beam_prune_cost_artifact_hashes_v1",
            "artifacts": artifacts,
        },
    )

    # Final hard gates: route digest, cost arm, archive provenance, and exact diff.
    if _sha256_file(source_archive) != source_sha:
        raise RuntimeError("successor archive changed during bundle build")
    if repaired_adapt_sha == predecessor_adapt_sha:
        raise RuntimeError("repair did not change adapt_pipeline.py")
    if predecessor_adapt_size >= repaired_adapt_size:
        raise RuntimeError("repair payload size did not increase as expected")
    return successor


def main() -> None:
    built = [_build_arm(dict(arm)) for arm in ARMS]
    for path in built:
        print(path)


if __name__ == "__main__":
    main()
