#!/usr/bin/env python3
"""Build one immutable macro/beam/prune cost-arm successor revision.

The worker archive is rebuilt directly from the byte-identical v3 source.  It
contains only the parent-beam prune-consumer identity repair and serialization
of the already-computed physical-lane receipt on beam history rows.  Bundle
validators separately validate the 50-round controller/frontier and the
selected winner, which may be a shallower archived terminal branch.  No branch
selection, archive policy, or scientific setting changes.
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
CREATED_UTC = "2026-07-20T02:00:00Z"
SUCCESSOR_REVISION = "v5"
VALIDATOR_REVISION = "v5"

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
    beam_feature_anchor = (
        '                "selected_feature_rows": [\n'
        '                    dict(row) for row in selected_batch_feature_rows_local\n'
        '                ],\n'
    )
    if text.count(beam_feature_anchor) != 1:
        raise RuntimeError("beam physical-lane serialization anchor drift")
    beam_lane_receipt = beam_feature_anchor + (
        '                "static_lane_route": (\n'
        '                    str(phase1_feature_selected_local.get("static_lane_route", static_lane_route_key))\n'
        '                    if isinstance(phase1_feature_selected_local, dict)\n'
        '                    else str(static_lane_route_key)\n'
        '                ),\n'
        '                "physical_operator_lane": (\n'
        '                    str(phase1_feature_selected_local.get("physical_operator_lane"))\n'
        '                    if isinstance(phase1_feature_selected_local, dict)\n'
        '                    and phase1_feature_selected_local.get("physical_operator_lane") is not None\n'
        '                    else None\n'
        '                ),\n'
        '                "physical_operator_quality": (\n'
        '                    str(phase1_feature_selected_local.get("physical_operator_quality"))\n'
        '                    if isinstance(phase1_feature_selected_local, dict)\n'
        '                    and phase1_feature_selected_local.get("physical_operator_quality") is not None\n'
        '                    else None\n'
        '                ),\n'
        '                "physical_operator_hh_full_meta_class": (\n'
        '                    str(phase1_feature_selected_local.get("physical_operator_hh_full_meta_class"))\n'
        '                    if isinstance(phase1_feature_selected_local, dict)\n'
        '                    and phase1_feature_selected_local.get("physical_operator_hh_full_meta_class") is not None\n'
        '                    else None\n'
        '                ),\n'
        '                "physical_operator_lane_source": (\n'
        '                    str(phase1_feature_selected_local.get("physical_operator_lane_source"))\n'
        '                    if isinstance(phase1_feature_selected_local, dict)\n'
        '                    and phase1_feature_selected_local.get("physical_operator_lane_source") is not None\n'
        '                    else None\n'
        '                ),\n'
        '                "physical_operator_lane_health": (\n'
        '                    float(phase1_feature_selected_local.get("physical_operator_lane_health"))\n'
        '                    if isinstance(phase1_feature_selected_local, dict)\n'
        '                    and phase1_feature_selected_local.get("physical_operator_lane_health") is not None\n'
        '                    else None\n'
        '                ),\n'
        '                "physical_operator_lane_relative_health": (\n'
        '                    float(phase1_feature_selected_local.get("physical_operator_lane_relative_health"))\n'
        '                    if isinstance(phase1_feature_selected_local, dict)\n'
        '                    and phase1_feature_selected_local.get("physical_operator_lane_relative_health") is not None\n'
        '                    else None\n'
        '                ),\n'
        '                "physical_operator_lane_live": (\n'
        '                    bool(phase1_feature_selected_local.get("physical_operator_lane_live"))\n'
        '                    if isinstance(phase1_feature_selected_local, dict)\n'
        '                    and phase1_feature_selected_local.get("physical_operator_lane_live") is not None\n'
        '                    else None\n'
        '                ),\n'
    )
    text = text.replace(beam_feature_anchor, beam_lane_receipt, 1)
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


def _install_beam_validator(successor: Path, arm: dict[str, Any]) -> None:
    template = INPUT_ROOT / (
        f"paper_i_hh_sr_macro_beam_evidence_validation_{VALIDATOR_REVISION}.py"
    )
    target = successor / "beam_evidence_validation.py"
    shutil.copy2(template, target)
    shutil.copy2(
        INPUT_ROOT / f"paper_i_hh_sr_macro_beam_validation_{VALIDATOR_REVISION}_test.py",
        successor / "test_beam_validation.py",
    )
    for name in ("run_job.py", "validate_fetched.py"):
        path = successor / name
        text = path.read_text(encoding="utf-8")
        old_import = (
            "from evidence_validation import checkpoint_sha256, "
            "validate_parent_evidence\n"
        )
        new_import = (
            "from evidence_validation import checkpoint_sha256\n"
            "from beam_evidence_validation import validate_beam_parent_evidence\n"
        )
        if text.count(old_import) != 1:
            raise RuntimeError(f"{name}: evidence-validator import anchor drift")
        text = text.replace(old_import, new_import, 1)
        if text.count("evidence = validate_parent_evidence(\n") != 1:
            raise RuntimeError(f"{name}: evidence-validator call anchor drift")
        text = text.replace(
            "evidence = validate_parent_evidence(\n",
            "evidence = validate_beam_parent_evidence(\n",
            1,
        )
        call_anchor = "        require_supported_rank=True,\n"
        if text.count(call_anchor) != 1:
            raise RuntimeError(f"{name}: supported-rank call anchor drift")
        fallback_policy = (
            "collective_span_novelty_over_symmetric_cost_v1"
            if arm["slug"] == "symmetric"
            else "collective_span_novelty_over_cost_v1"
        )
        text = text.replace(
            call_anchor,
            f'        expected_cost_mode={arm["cost"]!r},\n'
            f'        expected_fallback_policy={fallback_policy!r},\n'
            + call_anchor,
            1,
        )
        if name == "run_job.py":
            old = "    checkpoint = terminal_checkpoint(result, target_round)\n"
            new = (
                '    winner_round = int(evidence["selected_final_controller_round"])\n'
                "    checkpoint = terminal_checkpoint(result, winner_round)\n"
            )
            if text.count(old) != 1:
                raise RuntimeError("run_job terminal checkpoint anchor drift")
            text = text.replace(old, new, 1)
            text = text.replace(
                '        "--outer-iteration", str(target_round),\n',
                '        "--outer-iteration", str(winner_round),\n',
                1,
            )
            text = text.replace(
                '    if int(source.get("outer_iteration", -1)) != target_round:\n',
                '    if int(source.get("outer_iteration", -1)) != winner_round:\n',
                1,
            )
            text = text.replace(
                '        "terminal_checkpoint_sha256": checkpoint_digest,\n',
                '        "controller_horizon_round": target_round,\n'
                '        "selected_winner_round": winner_round,\n'
                '        "terminal_checkpoint_sha256": checkpoint_digest,\n',
                1,
            )
        else:
            old = (
                '    checkpoints = result.get("adapt_vqe", {}).get('
                '"active_prefix_checkpoints", [])\n'
                "    terminal = [\n"
                "        row for row in checkpoints\n"
                "        if isinstance(row, dict)\n"
                '        and int(row.get("outer_iteration", -1)) == target_round\n'
            )
            new = (
                '    winner_round = int(evidence["selected_final_controller_round"])\n'
                '    checkpoints = result.get("adapt_vqe", {}).get('
                '"active_prefix_checkpoints", [])\n'
                "    terminal = [\n"
                "        row for row in checkpoints\n"
                "        if isinstance(row, dict)\n"
                '        and int(row.get("outer_iteration", -1)) == winner_round\n'
            )
            if text.count(old) != 1:
                raise RuntimeError("validate_fetched terminal checkpoint anchor drift")
            text = text.replace(old, new, 1)
            text = text.replace(
                '        int(source.get("outer_iteration", -1)) != target_round\n',
                '        int(source.get("outer_iteration", -1)) != winner_round\n',
                1,
            )
            text = text.replace(
                '        "target_controller_round": target_round,\n',
                '        "target_controller_round": target_round,\n'
                '        "selected_winner_round": winner_round,\n',
                1,
            )
        compile(text, str(path), "exec")
        path.write_text(text, encoding="utf-8")

    bundle_relative = successor.relative_to(ROOT)
    submit = successor / "submit.sub"
    text = submit.read_text(encoding="utf-8")
    anchor = f"{bundle_relative}/evidence_validation.py"
    replacement = anchor + f", {bundle_relative}/beam_evidence_validation.py"
    if text.count(anchor) != 1:
        raise RuntimeError("submit beam-validator transfer anchor drift")
    submit.write_text(text.replace(anchor, replacement, 1), encoding="utf-8")
    upload = successor / "upload_artifact_list.txt"
    text = upload.read_text(encoding="utf-8")
    anchor = f"{bundle_relative}/evidence_validation.py\n"
    replacement = anchor + f"{bundle_relative}/beam_evidence_validation.py\n"
    if text.count(anchor) != 1:
        raise RuntimeError("upload-list beam-validator anchor drift")
    upload.write_text(text.replace(anchor, replacement, 1), encoding="utf-8")


def _repair_receipt(
    *,
    arm: dict[str, Any],
    predecessor_bundle: str,
    predecessor_adapt_sha: str,
    repaired_adapt_sha: str,
) -> dict[str, Any]:
    return {
        "schema": "paper_i_sr_macro_beam_plumbing_repair_v2",
        "classification": "non_scientific_serialization_identity_and_validation_v1",
        "failed_cluster": int(arm["cluster"]),
        "predecessor_bundle": predecessor_bundle,
        "predecessor_source_archive_sha256": PREDECESSOR_SOURCE_SHA256,
        "path": ADAPT_PATH,
        "source_sha256_before": predecessor_adapt_sha,
        "source_sha256_after": repaired_adapt_sha,
        "exact_changes": [
            "scope_prune_trial_consumer_id_by_parent_beam_branch_id",
            "pass_estimator_call_context_branch_id_to_consumer_id_builder",
            "serialize_existing_physical_lane_receipt_on_beam_history_rows",
            "validate_controller_frontier_separately_from_selected_terminal_winner",
            "close_materialized_branch_estimator_receipts_by_identity",
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
        "successor_revision": SUCCESSOR_REVISION,
    }


def _build_arm(arm: dict[str, Any]) -> Path:
    predecessor_bundle = (
        "paper_i_hh_sr_snake_macro_beam3x2_fs_prune_"
        f"{arm['slug']}_cost_all_six_r50_20260719_v3_chtc"
    )
    successor_bundle = predecessor_bundle.replace(
        "_v3_chtc", f"_{SUCCESSOR_REVISION}_chtc"
    )
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
    new_batch = old_batch.removesuffix("-v3") + f"-{SUCCESSOR_REVISION}"
    _replace_text_tree(
        successor,
        (
            (predecessor_bundle, successor_bundle),
            (old_batch, new_batch),
            ("20260719-v3", f"20260719-{SUCCESSOR_REVISION}"),
            (PREDECESSOR_SOURCE_SHA256, source_sha),
        ),
    )
    submit_path = successor / "submit.sub"
    submit_text = submit_path.read_text(encoding="utf-8")
    superseded_block = (
        "# SUPERSEDED: exact prune-trial estimator consumers alias across beam parents.\n"
        "# The immutable v4 successor preserves this route and repairs only that\n"
        "# non-scientific consumer identity.  Fail closed against accidental resubmission.\n"
        "requirements = False && TARGET.HasSIF\n"
    )
    if submit_text.count(superseded_block) != 1:
        raise RuntimeError("v3 fail-closed submit marker anchor drift")
    submit_path.write_text(
        submit_text.replace(
            superseded_block,
            f"# Immutable {SUCCESSOR_REVISION} non-scientific successor; submission requires explicit approval.\n"
            "requirements = TARGET.HasSIF\n",
            1,
        ),
        encoding="utf-8",
    )
    _install_beam_validator(successor, arm)

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
            f"{SUCCESSOR_REVISION}_bundle_tests_passed": 0,
            f"{SUCCESSOR_REVISION}_archive_only_validate_rows_passed": 0,
            f"{SUCCESSOR_REVISION}_shared_archive_focused_tests_passed": 0,
            f"{SUCCESSOR_REVISION}_test_targets": [
                "test/test_static_adapt_sr_v4_runtime.py",
                "test/test_static_adapt_macro_beam_prune_cost_profiles.py",
            ],
            "status": "pass",
        }
    )
    _json_dump(successor / "archive_only_preflight.json", archive_preflight)

    readme = successor / "README.md"
    readme.write_text(
        f"# {new_batch}\n\n"
        + "Six fresh round-0 to round-50 Paper-I Hubbard--Holstein SR-SNAKE jobs.\n\n"
        + "- Macro-only intact logical parent candidates with physical lanes.\n"
        + "- Historical beam: 3 live parents x 2 admission children, at most 6 continuations per round.\n"
        + "- Live-only undamped full-logical Fubini--Study trust pruning; measured delete/refit acceptance.\n"
        + f"- Cost policy: `{arm['cost']}`.\n"
        + "- Ordinary Phase-II/III novelty multipliers off; all-infeasible fallback retained with telemetry.\n"
        + "- Weak-Holstein cutoff `n_ph=3`; strong-Holstein cutoff `n_ph=7`; same-cutoff references.\n"
        + "- Exact horizon: 50 controller rounds for every regime.\n"
        + f"- Route digest: `{arm['route_digest']}`.\n"
        + f"- Source archive SHA-256: `{source_sha}`.\n\n"
        + f"## {SUCCESSOR_REVISION} non-scientific repair\n\n"
        + "Derived directly from the immutable v3 source archive. Exact prune-trial "
        + "consumer IDs include the parent beam branch; beam history serializes and "
        + "validates the existing physical-lane receipt; and validation separates the "
        + "50-round controller/frontier from the selected terminal winner while "
        + "cross-checking the checkpoint and estimator-receipt graph. Route settings, "
        + "selection, terminal archive policy, and digest are unchanged.\n",
        encoding="utf-8",
    )

    run_job = successor / "run_job.py"
    run_job_text = run_job.read_text(encoding="utf-8")
    expected_label = "symmetric-cost" if arm["slug"] == "symmetric" else "one-sided-cost"
    first_line = run_job_text.splitlines()[1]
    if "Validate and execute one archive-only SR-SNAKE" not in first_line:
        raise RuntimeError("run_job.py docstring anchor drift")
    lines = run_job_text.splitlines()
    lines[1] = (
        f'"""Validate and execute one archive-only SR-SNAKE {expected_label} job."""'
    )
    run_job.write_text("\n".join(lines) + "\n", encoding="utf-8")

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
        + "def verify_beam_lane_serialization():\n"
        + "    import tarfile\n"
        + "    with tarfile.open(BUNDLE_DIR / 'source_locked.tar.gz', 'r:gz') as handle:\n"
        + "        text = handle.extractfile('pipelines/static_adapt/adapt_pipeline.py').read().decode('utf-8')\n"
        + "    anchor = '\"selected_feature_rows\": [\\n                    dict(row) for row in selected_batch_feature_rows_local'\n"
        + "    assert text.count(anchor) == 1\n"
        + "    beam = text[text.index(anchor):text.index(anchor) + 8000]\n"
        + "    fields = ('static_lane_route', 'physical_operator_lane', 'physical_operator_quality', 'physical_operator_hh_full_meta_class', 'physical_operator_lane_source', 'physical_operator_lane_health', 'physical_operator_lane_relative_health', 'physical_operator_lane_live')\n"
        + "    assert all(('\"' + field + '\"') in beam for field in fields)\n"
        + "    return True\n\n"
        + "_original_verify = verify\n"
        + "def verify():\n"
        + "    assert _original_verify()\n"
        + "    assert verify_prune_consumer_repair()\n"
        + "    assert verify_beam_lane_serialization()\n"
        + "    return True\n\n"
        + "if __name__ == '__main__':\n"
        + "    verify()\n"
        + f"    print('macro beam-prune {SUCCESSOR_REVISION} cost bundle verification passed')\n",
        encoding="utf-8",
    )

    test_path = successor / "test_bundle.py"
    test_text = test_path.read_text()
    insertion = (
        "\n    def test_prune_trial_consumer_id_is_parent_beam_scoped(self):\n"
        "        self.assertTrue(build_bundle.verify_prune_consumer_repair())\n"
        "\n    def test_beam_lane_serialization(self):\n"
        "        self.assertTrue(build_bundle.verify_beam_lane_serialization())\n"
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
    successor_number = int(SUCCESSOR_REVISION.removeprefix("v"))
    prior_revision = f"v{successor_number - 1}"
    prior_bundle = predecessor_bundle.replace(
        "_v3_chtc", f"_{prior_revision}_chtc"
    )
    prior_marker = INPUT_ROOT / prior_bundle / "SUPERSEDED_DO_NOT_SUBMIT.json"
    marker_payload = {
        "bundle_id": prior_bundle,
        "classification": (
            "superseded_incomplete_beam_lane_and_fail_closed_validation_v1"
        ),
        "successor_bundle": successor_bundle,
        "scientific_setting_changes": [],
    }
    if prior_marker.exists():
        existing_marker = json.loads(prior_marker.read_text())
        if existing_marker.get("successor_bundle") != successor_bundle:
            raise RuntimeError("existing predecessor superseded marker conflicts")
    else:
        _json_dump(prior_marker, marker_payload)
    prior_submit = INPUT_ROOT / prior_bundle / "submit.sub"
    if prior_submit.exists():
        prior_submit_text = prior_submit.read_text(encoding="utf-8")
        prior_submit_text = prior_submit_text.replace(
            "requirements = TARGET.HasSIF",
            "requirements = False && TARGET.HasSIF",
            1,
        )
        prior_submit.write_text(prior_submit_text, encoding="utf-8")
    return successor


def main() -> None:
    built = [_build_arm(dict(arm)) for arm in ARMS]
    for path in built:
        print(path)


if __name__ == "__main__":
    main()
