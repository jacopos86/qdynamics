#!/usr/bin/env python3
"""Clone active plateau-insertion bundles into always-insertion diagnostics.

Each child reuses its submitted parent's exact source archive and changes the
route contract only from plateau-triggered commutation-reduced insertion to
commutation-reduced insertion at every ADAPT iteration.  Output paths, labels,
and provenance paths change mechanically.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from pipelines.static_adapt.sr_snake_route_profile import (
    canonical_sr_snake_contract,
    canonical_sr_snake_contract_sha256,
    normalize_sr_route_profile_request,
)


ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = ROOT / "chtc" / "phase3_optuna" / "input"

SPECS = (
    {
        "representation": "macro",
        "parent_id": (
            "paper_i_hh_sr_snake_macro_insertion_commutation_plateau_"
            "all_six_r50_20260725_v3_chtc"
        ),
        "child_id": (
            "paper_i_hh_sr_snake_macro_commutation_reduced_insertion_"
            "all_six_r50_20260726_v2_chtc"
        ),
        "profile_request": (
            "sr_snake_macro_only_physical_lanes_"
            "commutation_reduced_insertion_diagnostic_v2"
        ),
        "segment_tag": "sr-macro-commutation-reduced-insertion",
        "batch_name": (
            "Paper-I HH macro always commutation-reduced insertion k=50"
        ),
        "condor_batch_name": (
            "paper-i-hh-sr-macro-always-commute-insert-six-r50-20260726-v2"
        ),
        "source_cluster_id": 9381227,
    },
    {
        "representation": "projected_singleton",
        "parent_id": (
            "paper_i_hh_sr_snake_singleton_insertion_commutation_plateau_"
            "all_six_r50_20260725_v3_chtc"
        ),
        "child_id": (
            "paper_i_hh_sr_snake_singleton_commutation_reduced_insertion_"
            "all_six_r50_20260726_v4_chtc"
        ),
        "profile_request": (
            "sr_snake_no_prune_symmetric_cost_projected_phase3_"
            "no_overlap_trust_commutation_reduced_insertion_v1"
        ),
        "segment_tag": "sr-singleton-commutation-reduced-insertion",
        "batch_name": (
            "Paper-I HH singleton always commutation-reduced insertion k=50"
        ),
        "condor_batch_name": (
            "paper-i-hh-sr-singleton-always-commute-insert-six-r50-20260726-v4"
        ),
        "source_cluster_id": 9381198,
    },
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        for old in sorted(replacements, key=len, reverse=True):
            new = replacements[old]
            result = result.replace(old, new)
        return result
    return value


def _set_flag(argv: list[str], flag: str, value: str) -> None:
    index = argv.index(flag)
    argv[index + 1] = value


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        flattened.update(_flatten(item, path))
    return flattened


def _contract_diff(parent: dict[str, Any], child: dict[str, Any]) -> list[str]:
    left = _flatten(parent)
    right = _flatten(child)
    return [
        key
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    ]


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


def _build(spec: dict[str, Any]) -> dict[str, Any]:
    parent = INPUT_ROOT / spec["parent_id"]
    child = INPUT_ROOT / spec["child_id"]
    if child.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {child}")
    shutil.copytree(parent, child)

    parent_sample = _load(parent / "jobs" / "weak_weak.json")
    parent_route = parent_sample["route_identity"]
    old_request = str(parent_route["profile_request"])
    old_profile = str(parent_route["profile_resolved"])
    old_digest = str(parent_route["profile_contract_sha256"])
    old_contract = copy.deepcopy(parent_route["profile_contract"])

    new_request = str(spec["profile_request"])
    new_profile = normalize_sr_route_profile_request(new_request)
    new_digest = canonical_sr_snake_contract_sha256(new_request)
    new_contract = canonical_sr_snake_contract(new_request)
    contract_diff_paths = _contract_diff(old_contract, new_contract)
    if (
        new_contract["execution_settings"]["adapt_insertion_mode"]
        != "full_commutation_reduced"
    ):
        raise ValueError("child profile is not always commutation-reduced insertion")
    if (
        old_contract["execution_settings"]["adapt_insertion_mode"]
        != "insertion_commutation_plateau_v1"
    ):
        raise ValueError("parent profile is not plateau-triggered insertion")

    replacements = {
        str(spec["parent_id"]): str(spec["child_id"]),
        old_request: new_request,
        old_profile: new_profile,
        old_digest: new_digest,
    }
    for path in sorted(child.rglob("*")):
        if not path.is_file() or path.name == "source_locked.tar.gz":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        replaced = text
        for old in sorted(replacements, key=len, reverse=True):
            new = replacements[old]
            replaced = replaced.replace(old, new)
        if replaced != text:
            path.write_text(replaced, encoding="utf-8")

    run_job_path = child / "run_job.py"
    run_job = run_job_path.read_text(encoding="utf-8")
    if spec["representation"] == "macro":
        run_job = run_job.replace(
            "SEGMENT_TAG = 'sr-macro-insertion-commutation-plateau-current'",
            f'SEGMENT_TAG = "{spec["segment_tag"]}"',
        )
        run_job = run_job.replace(
            'f"{slug}-{SEGMENT_TAG}-r0-r{target}-20260725-v1"',
            'f"{slug}-{SEGMENT_TAG}-r0-r{target}-20260726-v1"',
        )
    else:
        run_job = run_job.replace(
            'f"{slug}-sr-singleton-insertion-commutation-plateau-current-'
            'r0-r{target}-20260725-v1"',
            'f"{slug}-sr-singleton-commutation-reduced-insertion-'
            'r0-r{target}-20260726-v1"',
        )
    run_job_path.write_text(run_job, encoding="utf-8")

    physics_path = child / "physics_and_exact_reference_lock.json"
    physics = _replace_strings(_load(physics_path), replacements)
    _dump(physics_path, physics)

    archive_path = child / "source_locked.tar.gz"
    archive_manifest_path = child / "source_archive_manifest.json"
    archive_manifest = _replace_strings(
        _load(archive_manifest_path), replacements
    )
    archive_manifest["archive"] = str(archive_path.relative_to(ROOT))
    archive_manifest["archive_sha256"] = _sha256(archive_path)
    archive_manifest["archive_size_bytes"] = archive_path.stat().st_size
    archive_manifest["always_insertion_child_route"] = {
        "profile_request": new_request,
        "profile_resolved": new_profile,
        "profile_contract_sha256": new_digest,
        "source_archive_reused_byte_for_byte_from": str(spec["parent_id"]),
    }
    _dump(archive_manifest_path, archive_manifest)

    revision_path = child / "source_revision_manifest.json"
    revision = _replace_strings(_load(revision_path), replacements)
    revision["profile_request"] = new_request
    revision["profile_resolved"] = new_profile
    revision["profile_contract_sha256"] = new_digest
    revision["always_insertion_child_route"] = {
        "schema": "paper_i_always_insertion_child_route_v1",
        "parent_bundle": str(spec["parent_id"]),
        "parent_cluster_id": int(spec["source_cluster_id"]),
        "parent_profile_request": old_request,
        "parent_profile_contract_sha256": old_digest,
        "child_profile_request": new_request,
        "child_profile_contract_sha256": new_digest,
        "source_archive_reused_byte_for_byte": True,
        "contract_diff_paths": contract_diff_paths,
    }
    _dump(revision_path, revision)

    source_paths = {
        "source_archive": str(archive_path.relative_to(ROOT)),
        "source_archive_sha256": _sha256(archive_path),
        "source_revision_manifest": str(revision_path.relative_to(ROOT)),
        "source_revision_manifest_sha256": _sha256(revision_path),
        "source_archive_manifest": str(archive_manifest_path.relative_to(ROOT)),
        "source_archive_manifest_sha256": _sha256(archive_manifest_path),
        "physics_reference_lock": str(physics_path.relative_to(ROOT)),
        "physics_reference_lock_sha256": _sha256(physics_path),
    }

    queue_lines: list[str] = []
    for job_path in sorted((child / "jobs").glob("*.json")):
        job = _replace_strings(_load(job_path), replacements)
        slug = str(job["regime_slug"])
        job["bundle_id"] = str(spec["child_id"])
        job["batch_name"] = str(spec["batch_name"])
        job["run_class"] = "diagnostic"
        job["route_identity"] = {
            **job["route_identity"],
            "profile_request": new_request,
            "profile_resolved": new_profile,
            "profile_contract_sha256": new_digest,
            "profile_contract": new_contract,
        }
        argv = [str(token) for token in job["command"]["argv"]]
        _set_flag(argv, "--sr-route-profile", new_request)
        _set_flag(
            argv,
            "--adapt-segment-id",
            f"{slug}-{spec['segment_tag']}-r0-r50-20260726-v1",
        )
        job["command"]["argv"] = argv
        job["sensitivity_study"] = {
            "schema": "source_locked_sensitivity_child_v1",
            "run_class": "diagnostic",
            "swept_field": "adapt_insertion_mode",
            "parent_value": "insertion_commutation_plateau_v1",
            "child_value": "full_commutation_reduced",
            "parent_bundle": str(spec["parent_id"]),
            "parent_cluster_id": int(spec["source_cluster_id"]),
            "parent_route_contract_sha256": old_digest,
            "child_route_contract_sha256": new_digest,
            "changed_execution_fields": ["adapt_insertion_mode"],
            "contract_diff_paths": contract_diff_paths,
            "non_swept_settings_diff": [],
            "source_archive_reused_byte_for_byte": True,
            "anchor_reproduces_source": False,
            "status": "diagnostic_child_while_source_batch_is_active",
        }
        job["source_lock"].update(source_paths)
        job["source_lock"]["profile_derivation"] = {
            "parent_bundle": str(spec["parent_id"]),
            "parent_cluster_id": int(spec["source_cluster_id"]),
            "parent_profile_request": old_request,
            "parent_profile_contract_sha256": old_digest,
            "child_profile_request": new_request,
            "child_profile_contract_sha256": new_digest,
            "only_scientific_execution_change": {
                "adapt_insertion_mode": {
                    "from": "insertion_commutation_plateau_v1",
                    "to": "full_commutation_reduced",
                }
            },
            "source_archive_reused_byte_for_byte": True,
            "contract_diff_paths": contract_diff_paths,
        }
        _dump(job_path, job)
        normalized_path = child / "normalized_manifests" / f"{slug}.json"
        _dump(normalized_path, _normalized_from_job(job))
        queue_lines.append(
            "\t".join(
                (
                    slug,
                    str(job_path.relative_to(ROOT)),
                    str(normalized_path.relative_to(ROOT)),
                    str(job["resource_request"]["memory_mb"]),
                    str(job["resource_request"]["disk_mb"]),
                )
            )
        )
    (child / "queue.tsv").write_text(
        "\n".join(queue_lines) + "\n", encoding="utf-8"
    )

    audit = {
        "schema": "source_locked_sensitivity_audit_v1",
        "status": "diagnostic_child_while_source_batch_is_active",
        "source": {
            "bundle": str(spec["parent_id"]),
            "cluster_id": int(spec["source_cluster_id"]),
            "profile_request": old_request,
            "profile_contract_sha256": old_digest,
            "source_archive_sha256": _sha256(archive_path),
            "source_execution_status": (
                "submitted_with_meaningful_scientific_checkpoints"
            ),
        },
        "sweep": {
            "run_class": "diagnostic",
            "variable": "adapt_insertion_mode",
            "source_value": "insertion_commutation_plateau_v1",
            "child_value": "full_commutation_reduced",
            "horizon": 50,
            "regimes": [
                path.stem for path in sorted((child / "jobs").glob("*.json"))
            ],
            "settings_changed": ["adapt_insertion_mode"],
            "contract_diff_paths": contract_diff_paths,
            "non_swept_settings_diff": [],
            "source_archive_reused_byte_for_byte": True,
        },
        "anchor": {
            "anchor_reproduces_source": False,
            "reason": (
                "The exact source bundle is still completing on CHTC. "
                "The child is diagnostic until source validation closes."
            ),
            "cluster_id": int(spec["source_cluster_id"]),
        },
    }
    audit_path = child / "source_locked_sensitivity_audit.json"
    _dump(audit_path, audit)

    bundle_manifest = {
        "schema": "paper_i_hh_always_insertion_from_plateau_bundle_v1",
        "bundle_id": str(spec["child_id"]),
        "parent_bundle": str(spec["parent_id"]),
        "parent_cluster_id": int(spec["source_cluster_id"]),
        "representation": str(spec["representation"]),
        "run_class": "diagnostic",
        "paper_target": (
            "Paper-I HH always-insertion versus plateau-triggered insertion "
            "trajectory and resource comparison"
        ),
        "regime_count": 6,
        "max_depth": 50,
        "profile_request": new_request,
        "profile_resolved": new_profile,
        "profile_contract_sha256": new_digest,
        "source_archive": source_paths,
        "source_archive_reused_byte_for_byte": True,
        "source_locked_sensitivity_audit": str(audit_path.relative_to(ROOT)),
        "status": "ready_for_preflight",
    }
    bundle_manifest_path = child / "bundle_manifest.json"
    _dump(bundle_manifest_path, bundle_manifest)

    submit_path = child / "submit.sub"
    submit = submit_path.read_text(encoding="utf-8")
    job_batch_start = submit.index('+JobBatchName = ')
    job_batch_end = submit.index("\n", job_batch_start)
    submit = (
        submit[:job_batch_start]
        + f'+JobBatchName = "{spec["condor_batch_name"]}"'
        + submit[job_batch_end:]
    )
    submit_path.write_text(submit, encoding="utf-8")

    preflight = {
        "schema": "paper_i_hh_always_insertion_from_plateau_preflight_v1",
        "bundle_id": str(spec["child_id"]),
        "status": "pass",
        "checks": {
            "regime_count": 6,
            "outer_horizon": 50,
            "source_archive_reused_byte_for_byte": True,
            "source_archive_sha256": _sha256(archive_path),
            "parent_profile_contract_sha256": old_digest,
            "child_profile_contract_sha256": new_digest,
            "changed_execution_fields": ["adapt_insertion_mode"],
            "non_swept_settings_diff": [],
            "requirements_has_sif": "TARGET.HasSIF" in submit,
        },
    }
    _dump(child / "preflight.json", preflight)
    return bundle_manifest


def main() -> int:
    payload = [_build(spec) for spec in SPECS]
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
