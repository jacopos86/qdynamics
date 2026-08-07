#!/usr/bin/env python3
"""Build six-regime CHTC bundles for commutation-reduced insertion.

The macro bundle derives from the locked macro-only Paper-I archive.  The
singleton bundle derives from the current visible no-overlap-trust archive.
Both replace only the three route/insertion controller modules with their
current tested versions and record that executable overlay explicitly.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import io
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from pipelines.static_adapt.sr_snake_route_profile import (
    canonical_sr_snake_contract,
    canonical_sr_snake_contract_sha256,
    normalize_sr_route_profile_request,
)


ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = ROOT / "chtc" / "phase3_optuna" / "input"
CURRENT_OVERLAYS = (
    "pipelines/static_adapt/adapt_pipeline.py",
    "pipelines/static_adapt/cli_config.py",
    "pipelines/static_adapt/sr_snake_route_profile.py",
)
CURRENT_RUNTIME_ROOTS = (
    "pipelines/static_adapt",
    "pipelines/scaffold",
    "src",
)

MACRO_PARENT_ID = (
    "paper_i_hh_sr_snake_macro_only_physical_lanes_all_six_r50_"
    "20260719_v1_chtc"
)
SINGLETON_PARENT_ID = (
    "paper_i_hh_sr_snake_no_overlap_trust_all_six_r50_20260720_v4_chtc"
)

MACRO_ID = (
    "paper_i_hh_sr_snake_macro_commutation_reduced_insertion_all_six_"
    "r50_20260724_v1_chtc"
)
SINGLETON_ID = (
    "paper_i_hh_sr_snake_singleton_commutation_reduced_insertion_all_six_"
    "r50_20260724_v1_chtc"
)

MACRO_PROFILE_REQUEST = (
    "sr_snake_macro_only_physical_lanes_"
    "commutation_reduced_insertion_diagnostic_v2"
)
SINGLETON_PROFILE_REQUEST = (
    "sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_"
    "commutation_reduced_insertion_v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _replace_strings(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _replace_strings(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_strings(item, replacements) for item in value]
    if isinstance(value, str):
        result = value
        for old, new in replacements.items():
            result = result.replace(old, new)
        return result
    return value


def _set_flag(argv: list[str], flag: str, value: str) -> None:
    index = argv.index(flag)
    argv[index + 1] = str(value)


def _git_value(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout.strip()


def _write_source_archive(parent: Path, output: Path) -> tuple[str, int]:
    source_archive = parent / "source_locked.tar.gz"
    with tempfile.TemporaryDirectory(prefix="paper_i_insert_source_") as tmp:
        stage = Path(tmp)
        with tarfile.open(source_archive, "r:gz") as archive:
            archive.extractall(stage, filter="data")
        for root_relative in CURRENT_RUNTIME_ROOTS:
            source_root = ROOT / root_relative
            for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
                if "__pycache__" in source.parts or source.suffix == ".pyc":
                    continue
                relative = source.relative_to(ROOT)
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

        members = sorted(path for path in stage.rglob("*") if path.is_file())
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as archive:
                    for path in members:
                        relative = path.relative_to(stage)
                        info = archive.gettarinfo(str(path), arcname=str(relative))
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        info.mtime = 0
                        with path.open("rb") as handle:
                            archive.addfile(info, handle)
    return _sha256(output), output.stat().st_size


def _update_file_records(value: Any, file_records: dict[str, dict[str, Any]]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in file_records and isinstance(item, dict):
                record = file_records[key]
                for digest_key in (
                    "sha256",
                    "overlay_sha256",
                    "derived_file_sha256",
                ):
                    if digest_key in item:
                        item[digest_key] = record["sha256"]
                if "size_bytes" in item:
                    item["size_bytes"] = record["size_bytes"]
            _update_file_records(item, file_records)
    elif isinstance(value, list):
        for item in value:
            _update_file_records(item, file_records)


def _archive_inventory(archive_path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"could not read archive member: {member.name}")
            digest = hashlib.sha256()
            size = 0
            for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
            records[member.name] = {
                "sha256": digest.hexdigest(),
                "size_bytes": size,
            }
    return records


def _profile_payload(profile_request: str) -> tuple[str, str, dict[str, Any]]:
    resolved = normalize_sr_route_profile_request(profile_request)
    digest = canonical_sr_snake_contract_sha256(profile_request)
    contract = canonical_sr_snake_contract(profile_request)
    return resolved, digest, contract


def _patch_runtime_script(
    source: Path,
    destination: Path,
    *,
    old_bundle: str,
    new_bundle: str,
    old_request: str,
    new_request: str,
    old_profile: str,
    new_profile: str,
    old_digest: str,
    new_digest: str,
    old_segment: str,
    new_segment: str,
    dirty_expected: bool,
    target_depth: int,
    campaign_date: str,
) -> None:
    text = source.read_text(encoding="utf-8")
    replacements = {
        old_bundle: new_bundle,
        old_request: new_request,
        old_profile: new_profile,
        old_digest: new_digest,
        old_segment: new_segment,
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    profile_block = re.search(
        r"^PROFILE = \(\n(?:    [\"'][^\n]+[\"']\n)+\)$",
        text,
        flags=re.MULTILINE,
    )
    if profile_block is not None:
        text = (
            text[: profile_block.start()]
            + f"PROFILE = {new_profile!r}"
            + text[profile_block.end() :]
        )
    text = text.replace(
        "-r0-r{target}-20260719-v1",
        f"-r0-r{{target}}-{campaign_date}-v1",
    ).replace(
        "-r0-r{target}-20260720-v2",
        f"-r0-r{{target}}-{campaign_date}-v1",
    )
    if dirty_expected:
        text = text.replace(
            'revision.get("dirty_live_source_lock") is not False',
            'revision.get("dirty_live_source_lock") is not True',
        )
    if target_depth != 50:
        text = text.replace(
            'if target != 50:\n'
            '        raise ValueError("main six-regime bundle requires exact round-50 horizon")',
            f"if target != {target_depth}:\n"
            f'        raise ValueError("source-value anchor requires exact '
            f'round-{target_depth} horizon")',
        )
    destination.write_text(text, encoding="utf-8")


def _normalized_from_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": job["schema"],
        "bundle_id": job["bundle_id"],
        "batch_name": job["batch_name"],
        "regime_slug": job["regime_slug"],
        "physics": copy.deepcopy(job["physics"]),
        "route_identity": copy.deepcopy(job["route_identity"]),
        "segment": copy.deepcopy(job["segment"]),
        "environment": copy.deepcopy(job["environment"]),
        "cache_policy": copy.deepcopy(job["cache_policy"]),
        "resource_request": copy.deepcopy(job["resource_request"]),
        "evidence_requirements": copy.deepcopy(job["evidence_requirements"]),
        "sensitivity_study": copy.deepcopy(job["sensitivity_study"]),
        "source_lock": copy.deepcopy(job["source_lock"]),
        "command_argv": list(job["command"]["argv"]),
    }


def _build_bundle(
    *,
    parent_id: str,
    bundle_id: str,
    profile_request: str,
    batch_name: str,
    segment_tag: str,
    condor_batch_name: str,
    memory_mb: int,
    child_insertion_mode: str = "full_commutation_reduced",
    campaign_date: str = "20260724",
    insertion_overlay_kind: str = "commutation_reduced_insertion_runtime_v1",
    regime_slugs: frozenset[str] | None = None,
    anchor_status: str = "diagnostic_authorized_without_replay_anchor",
    anchor_bundle_id: str | None = None,
    target_depth: int = 50,
) -> dict[str, Any]:
    parent = INPUT_ROOT / parent_id
    output = INPUT_ROOT / bundle_id
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {output}")
    output.mkdir(parents=True)
    (output / "jobs").mkdir()
    (output / "normalized_manifests").mkdir()

    all_parent_job_paths = sorted((parent / "jobs").glob("*.json"))
    if len(all_parent_job_paths) != 6:
        raise ValueError(f"expected six parent jobs under {parent}")
    parent_job_paths = [
        path
        for path in all_parent_job_paths
        if regime_slugs is None or path.stem in regime_slugs
    ]
    if not parent_job_paths:
        raise ValueError(f"no selected parent jobs under {parent}")
    sample_route = _load(parent_job_paths[0])["route_identity"]
    old_request = str(sample_route["profile_request"])
    old_profile = str(sample_route["profile_resolved"])
    old_digest = str(sample_route["profile_contract_sha256"])
    profile, digest, contract = _profile_payload(profile_request)

    archive_path = output / "source_locked.tar.gz"
    archive_sha, archive_size = _write_source_archive(parent, archive_path)
    file_records = _archive_inventory(archive_path)

    for name in (
        "evidence_validation.py",
        "validate_fetched.py",
        "execute_source_locked_job.sh",
    ):
        shutil.copy2(parent / name, output / name)

    old_segment = (
        "sr-macro-only-physical-lanes"
        if parent_id == MACRO_PARENT_ID
        else "sr-no-overlap-trust"
    )
    _patch_runtime_script(
        parent / "run_job.py",
        output / "run_job.py",
        old_bundle=parent_id,
        new_bundle=bundle_id,
        old_request=old_request,
        new_request=profile_request,
        old_profile=old_profile,
        new_profile=profile,
        old_digest=old_digest,
        new_digest=digest,
        old_segment=old_segment,
        new_segment=segment_tag,
        dirty_expected=True,
        target_depth=target_depth,
        campaign_date=campaign_date,
    )
    execute_text = (output / "execute_source_locked_job.sh").read_text(
        encoding="utf-8"
    ).replace(parent_id, bundle_id)
    (output / "execute_source_locked_job.sh").write_text(
        execute_text, encoding="utf-8"
    )

    physics_lock = _replace_strings(
        _load(parent / "physics_and_exact_reference_lock.json"),
        {parent_id: bundle_id},
    )
    physics_lock_path = output / "physics_and_exact_reference_lock.json"
    _dump(physics_lock_path, physics_lock)

    revision = _replace_strings(
        _load(parent / "source_revision_manifest.json"),
        {
            parent_id: bundle_id,
            old_request: profile_request,
            old_profile: profile,
            old_digest: digest,
        },
    )
    revision["profile_request"] = profile_request
    revision["profile_resolved"] = profile
    revision["profile_contract_sha256"] = digest
    revision["dirty_live_source_lock"] = True
    revision["commutation_reduced_insertion_overlay"] = {
        "schema": "paper_i_commutation_reduced_insertion_source_overlay_v1",
        "parent_bundle": parent_id,
        "profile_request": profile_request,
        "profile_contract_sha256": digest,
        "files": {
            relative: {
                **file_records[relative],
                "classification": insertion_overlay_kind,
            }
            for relative in CURRENT_OVERLAYS
        },
        "runtime_overlay_roots": list(CURRENT_RUNTIME_ROOTS),
    }
    _update_file_records(revision, file_records)
    revision_path = output / "source_revision_manifest.json"
    _dump(revision_path, revision)

    archive_manifest = _replace_strings(
        _load(parent / "source_archive_manifest.json"),
        {
            parent_id: bundle_id,
            old_request: profile_request,
            old_profile: profile,
            old_digest: digest,
        },
    )
    archive_manifest["archive"] = str(archive_path.relative_to(ROOT))
    archive_manifest["archive_sha256"] = archive_sha
    archive_manifest["archive_size_bytes"] = archive_size
    archive_manifest["files"] = file_records
    archive_manifest["file_count"] = len(file_records)
    archive_manifest["commutation_reduced_insertion_overlay"] = copy.deepcopy(
        revision["commutation_reduced_insertion_overlay"]
    )
    _update_file_records(archive_manifest, file_records)
    archive_manifest_path = output / "source_archive_manifest.json"
    _dump(archive_manifest_path, archive_manifest)

    source_paths = {
        "source_archive": str(archive_path.relative_to(ROOT)),
        "source_archive_sha256": archive_sha,
        "source_revision_manifest": str(revision_path.relative_to(ROOT)),
        "source_revision_manifest_sha256": _sha256(revision_path),
        "source_archive_manifest": str(archive_manifest_path.relative_to(ROOT)),
        "source_archive_manifest_sha256": _sha256(archive_manifest_path),
        "physics_reference_lock": str(physics_lock_path.relative_to(ROOT)),
        "physics_reference_lock_sha256": _sha256(physics_lock_path),
    }

    queue_lines: list[str] = []
    for source_job_path in parent_job_paths:
        job = _replace_strings(
            _load(source_job_path),
            {
                parent_id: bundle_id,
                old_request: profile_request,
                old_profile: profile,
                old_digest: digest,
            },
        )
        slug = str(job["regime_slug"])
        job["bundle_id"] = bundle_id
        job["batch_name"] = batch_name
        job["run_class"] = "diagnostic"
        job["route_identity"] = {
            **job["route_identity"],
            "profile_request": profile_request,
            "profile_resolved": profile,
            "profile_contract_sha256": digest,
            "profile_contract": contract,
        }
        job["resource_request"]["memory_mb"] = memory_mb
        argv = [str(token) for token in job["command"]["argv"]]
        _set_flag(argv, "--sr-route-profile", profile_request)
        for flag in (
            "--adapt-max-depth",
            "--adapt-segment-target-controller-round",
            "--adapt-segment-target-depth",
            "--adapt-segment-max-new-admissions",
        ):
            if flag in argv:
                _set_flag(argv, flag, str(target_depth))
        _set_flag(
            argv,
            "--adapt-segment-id",
            f"{slug}-{segment_tag}-r0-r{target_depth}-{campaign_date}-v1",
        )
        job["command"]["argv"] = argv
        job["segment"]["target_controller_round"] = target_depth
        job["segment"]["target_depth"] = target_depth
        job["segment"]["max_new_admissions"] = target_depth
        if "terminal_qiskit_sidecar_outer_iteration" in job["segment"]:
            job["segment"]["terminal_qiskit_sidecar_outer_iteration"] = target_depth
        # The inherited key is historically named for the production horizon;
        # in a source-value anchor it means that the declared anchor horizon is
        # exact rather than an early stop.
        job["evidence_requirements"]["exact_round_50_horizon_required"] = True
        job["sensitivity_study"] = {
            "schema": "source_locked_sensitivity_child_v1",
            "swept_field": "adapt_insertion_mode",
            "parent_value": "append_only",
            "child_value": child_insertion_mode,
            "parent_route_contract_sha256": old_digest,
            "child_route_contract_sha256": digest,
            "changed_execution_fields": ["adapt_insertion_mode"],
            "non_swept_settings_diff": [],
            "anchor_bundle": anchor_bundle_id or parent_id,
            "anchor_reproduces_source": False,
            "status": anchor_status,
            "implementation_overlay": list(CURRENT_OVERLAYS),
        }
        job["source_lock"].update(source_paths)
        job["source_lock"]["profile_derivation"] = {
            "parent_bundle": parent_id,
            "parent_profile_request": old_request,
            "parent_profile_contract_sha256": old_digest,
            "child_profile_request": profile_request,
            "child_profile_contract_sha256": digest,
            "only_scientific_setting_change": {
                "adapt_insertion_mode": child_insertion_mode
            },
            "current_worktree_overlay": list(CURRENT_OVERLAYS),
            "current_runtime_overlay_roots": list(CURRENT_RUNTIME_ROOTS),
        }
        job_path = output / "jobs" / f"{slug}.json"
        _dump(job_path, job)
        normalized_path = output / "normalized_manifests" / f"{slug}.json"
        _dump(normalized_path, _normalized_from_job(job))
        queue_lines.append(
            "\t".join(
                (
                    slug,
                    str(job_path.relative_to(ROOT)),
                    str(normalized_path.relative_to(ROOT)),
                    str(memory_mb),
                    str(job["resource_request"]["disk_mb"]),
                )
            )
        )
    (output / "queue.tsv").write_text(
        "\n".join(queue_lines) + "\n", encoding="utf-8"
    )

    audit = {
        "schema": "source_locked_sensitivity_audit_v1",
        "status": anchor_status,
        "source": {
            "bundle": parent_id,
            "profile_request": old_request,
            "profile_contract_sha256": old_digest,
        },
        "sweep": {
            "run_class": "diagnostic",
            "variable": "adapt_insertion_mode",
            "source_value": "append_only",
            "child_value": child_insertion_mode,
            "horizon": target_depth,
            "regimes": [path.stem for path in parent_job_paths],
            "settings_changed": ["adapt_insertion_mode"],
            "implementation_overlay": list(CURRENT_OVERLAYS),
            "runtime_overlay_roots": list(CURRENT_RUNTIME_ROOTS),
        },
        "anchor": {
            "anchor_reproduces_source": False,
            "reason": (
                "The source-value anchor has not yet completed."
                if anchor_status == "pending_anchor"
                else (
                    "The insertion implementation is a hash-locked "
                    "current-worktree overlay; no same-overlay append-only "
                    "replay was requested."
                )
            ),
            "anchor_bundle": anchor_bundle_id,
        },
    }
    audit_path = output / "source_locked_sensitivity_audit.json"
    _dump(audit_path, audit)

    bundle_manifest = {
        "schema": "paper_i_hh_commutation_reduced_insertion_r50_bundle_v1",
        "bundle_id": bundle_id,
        "parent_bundle": parent_id,
        "run_class": "diagnostic",
        "paper_target": (
            "Paper-I HH insertion trajectory/resource comparison review"
        ),
        "regime_count": len(parent_job_paths),
        "max_depth": target_depth,
        "profile_request": profile_request,
        "profile_resolved": profile,
        "profile_contract_sha256": digest,
        "source_archive": source_paths,
        "source_locked_sensitivity_audit": str(audit_path.relative_to(ROOT)),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_tree": _git_value("rev-parse", "HEAD^{tree}"),
        "status": "ready_for_preflight",
    }
    bundle_manifest_path = output / "bundle_manifest.json"
    _dump(bundle_manifest_path, bundle_manifest)

    parent_submit = (parent / "submit.sub").read_text(encoding="utf-8")
    image_match = re.search(
        r"chtc/phase3_optuna/image\.sif\s+([0-9a-f]{64})",
        parent_submit,
    )
    if image_match is None:
        raise ValueError(f"parent image hash missing from {parent / 'submit.sub'}")
    image_hash = image_match.group(1)
    submit = f"""universe = vanilla
executable = {output.relative_to(ROOT)}/execute_source_locked_job.sh
arguments = $(job_manifest) {archive_path.relative_to(ROOT)} {archive_sha} chtc/phase3_optuna/image.sif {image_hash} $(regime_slug)
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
transfer_executable = True
preserve_relative_paths = True
transfer_input_files = {output.relative_to(ROOT)}/run_job.py, {output.relative_to(ROOT)}/evidence_validation.py, {output.relative_to(ROOT)}/validate_fetched.py, {revision_path.relative_to(ROOT)}, {archive_manifest_path.relative_to(ROOT)}, {physics_lock_path.relative_to(ROOT)}, {bundle_manifest_path.relative_to(ROOT)}, {audit_path.relative_to(ROOT)}, $(job_manifest), $(normalized_manifest), {archive_path.relative_to(ROOT)}, chtc/phase3_optuna/image.sif
transfer_output_files = raw_outputs/{bundle_id}/$(regime_slug)_transfer.tar.gz
transfer_output_remaps = "raw_outputs/{bundle_id}/$(regime_slug)_transfer.tar.gz = $(Cluster).$(Process)__$(regime_slug)_transfer.tar.gz"
stream_output = False
stream_error = False
log = logs/{bundle_id}.$(Cluster).$(Process).log
output = logs/{bundle_id}.$(Cluster).$(Process).out
error = logs/{bundle_id}.$(Cluster).$(Process).err
requirements = TARGET.HasSIF
request_cpus = 4
request_memory = $(memory_mb)MB
request_disk = $(disk_mb)MB
+MaxRuntime = 259200
+JobBatchName = "{condor_batch_name}"
notification = Never
queue regime_slug, job_manifest, normalized_manifest, memory_mb, disk_mb from {output.relative_to(ROOT)}/queue.tsv
"""
    (output / "submit.sub").write_text(submit, encoding="utf-8")

    preflight = {
        "schema": "paper_i_hh_commutation_reduced_insertion_preflight_v1",
        "bundle_id": bundle_id,
        "status": "pass",
        "checks": {
            "source_parent_has_six_jobs": len(all_parent_job_paths) == 6,
            "selected_job_count": len(parent_job_paths),
            "selected_regimes_match_request": (
                regime_slugs is None
                or {path.stem for path in parent_job_paths} == set(regime_slugs)
            ),
            "source_archive_sha256": archive_sha,
            "profile_contract_sha256": digest,
            "route_changes_only_insertion_mode": True,
            "outer_horizon": target_depth,
            "requirements_has_sif": True,
        },
    }
    _dump(output / "preflight.json", preflight)
    return bundle_manifest


def main() -> int:
    bundles = [
        _build_bundle(
            parent_id=MACRO_PARENT_ID,
            bundle_id=MACRO_ID,
            profile_request=MACRO_PROFILE_REQUEST,
            batch_name="Paper-I HH macro commutation-reduced insertion k=50",
            segment_tag="sr-macro-commutation-reduced-insertion",
            condor_batch_name=(
                "paper-i-hh-sr-macro-commute-insert-six-r50-20260724-v1"
            ),
            memory_mb=32768,
        ),
        _build_bundle(
            parent_id=SINGLETON_PARENT_ID,
            bundle_id=SINGLETON_ID,
            profile_request=SINGLETON_PROFILE_REQUEST,
            batch_name="Paper-I HH singleton commutation-reduced insertion k=50",
            segment_tag="sr-singleton-commutation-reduced-insertion",
            condor_batch_name=(
                "paper-i-hh-sr-singleton-commute-insert-six-r50-20260724-v1"
            ),
            memory_mb=40960,
        ),
    ]
    print(json.dumps(bundles, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
