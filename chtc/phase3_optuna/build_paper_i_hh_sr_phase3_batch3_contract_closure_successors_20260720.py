#!/usr/bin/env python3
"""Build immutable bundle-only successors for the batch-3 source-lock gate.

The accepted-coordinate repair correctly updated the source-lock state in the
JSON manifests, but the worker validator expresses that state as a multiline
Python constant and therefore retained the predecessor value.  This builder
changes only that bundle-side validator constant and bundle identifiers.  The
scientific source archives and route contracts remain byte-identical.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "chtc/phase3_optuna/input"
SCHEMA = "paper_i_sr_batch3_bundle_contract_closure_repair_v1"

FAMILIES = (
    {
        "name": "combinatorial",
        "parent": (
            "paper_i_hh_sr_snake_appendix_phase3_batch3_full_response_"
            "symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_"
            "r50_20260720_v10_chtc"
        ),
        "output": (
            "paper_i_hh_sr_snake_appendix_phase3_batch3_full_response_"
            "symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_"
            "r50_20260720_v11_chtc"
        ),
        "parent_batch": (
            "paper-i-hh-sr-appendix-phase3-batch3-fullresp-symcost-"
            "noprune-nobeam-nonovelty-six-r50-20260720-v10"
        ),
        "output_batch": (
            "paper-i-hh-sr-appendix-phase3-batch3-fullresp-symcost-"
            "noprune-nobeam-nonovelty-six-r50-20260720-v11"
        ),
    },
    {
        "name": "greedy",
        "parent": (
            "paper_i_hh_sr_snake_appendix_phase3_greedy_batch3_full_response_"
            "symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_"
            "r50_20260720_v9_chtc"
        ),
        "output": (
            "paper_i_hh_sr_snake_appendix_phase3_greedy_batch3_full_response_"
            "symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_"
            "r50_20260720_v10_chtc"
        ),
        "parent_batch": (
            "paper-i-hh-sr-appendix-phase3-greedy-batch3-fullresp-symcost-"
            "noprune-nobeam-nonovelty-six-r50-20260720-v9"
        ),
        "output_batch": (
            "paper-i-hh-sr-appendix-phase3-greedy-batch3-fullresp-symcost-"
            "noprune-nobeam-nonovelty-six-r50-20260720-v10"
        ),
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, value: Any) -> None:
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
        return {
            replace_tree(str(key), replacements): replace_tree(item, replacements)
            for key, item in value.items()
        }
    return value


def source_lock_state(bundle: Path) -> str:
    manifest = json.loads((bundle / "source_archive_manifest.json").read_text())
    value = str(manifest.get("worker_source_mode", ""))
    if not value.endswith("+accepted_batch_coordinate_receipt_repair_v1"):
        raise ValueError("parent bundle is missing the accepted-coordinate repair")
    return value


def patch_validator_constant(text: str, expected_state: str) -> str:
    start = text.index("SOURCE_LOCK_STATE = ")
    end = text.index("\nPREDECESSOR_V4_ARCHIVE_SHA256", start)
    return text[:start] + f"SOURCE_LOCK_STATE = {expected_state!r}\n" + text[end + 1 :]


def patch_validator_repair_chain(
    text: str,
    *,
    expected_state: str,
    current_adapt_sha256: str,
    predecessor_source_sha256: str,
    current_source_sha256: str,
) -> str:
    text = patch_validator_constant(text, expected_state)
    marker = "BATCH_SELECTOR_WORKSPACE_OVERLAY_SHA256 = ("
    insert = (
        "ACCEPTED_BATCH_COORDINATE_REPAIRED_SOURCE_SHA256 = {\n"
        f"    'pipelines/static_adapt/adapt_pipeline.py': {current_adapt_sha256!r},\n"
        "}\n"
        f"ACCEPTED_BATCH_COORDINATE_PREDECESSOR_SOURCE_SHA256 = {predecessor_source_sha256!r}\n"
        f"ACCEPTED_BATCH_COORDINATE_SOURCE_SHA256 = {current_source_sha256!r}\n"
    )
    if "ACCEPTED_BATCH_COORDINATE_REPAIRED_SOURCE_SHA256" not in text:
        text = text.replace(marker, insert + marker, 1)

    loop_start = text.index(
        "    for relative, expected_hash in EMPTY_GRAM_REPAIRED_SOURCE_SHA256.items():"
    )
    loop_end = text.index("    revision_workspace_repair = revision.get(", loop_start)
    replacement = '''    for relative, original_repair_hash in EMPTY_GRAM_REPAIRED_SOURCE_SHA256.items():
        expected_hash = ACCEPTED_BATCH_COORDINATE_REPAIRED_SOURCE_SHA256.get(
            relative, original_repair_hash
        )
        path = Path(relative)
        if (
            not path.is_file()
            or sha256(path) != expected_hash
            or archive_files.get(relative, {}).get("sha256") != expected_hash
        ):
            raise ValueError(
                f"serialized matrix repair source missing/drifted: {relative}"
            )
    revision_coordinate_repair = revision.get(
        "accepted_batch_coordinate_receipt_repair"
    )
    archive_coordinate_repair = archive_manifest.get(
        "accepted_batch_coordinate_receipt_repair"
    )
    source_coordinate_repair = source.get(
        "accepted_batch_coordinate_receipt_repair"
    )
    if (
        not isinstance(revision_coordinate_repair, dict)
        or revision_coordinate_repair != archive_coordinate_repair
        or archive_coordinate_repair != source_coordinate_repair
        or revision_coordinate_repair.get("schema")
        != "paper_i_sr_batch3_accepted_coordinate_receipt_repair_v1"
        or revision_coordinate_repair.get("predecessor_source_archive_sha256")
        != ACCEPTED_BATCH_COORDINATE_PREDECESSOR_SOURCE_SHA256
        or revision_coordinate_repair.get("successor_source_archive_sha256")
        != ACCEPTED_BATCH_COORDINATE_SOURCE_SHA256
        or revision_coordinate_repair.get("route_contract_sha256") != DIGEST
        or revision_coordinate_repair.get("scientific_settings_changed") is not False
        or revision_coordinate_repair.get("selector_or_model_inputs_changed") is not False
        or revision_coordinate_repair.get("accepted_subset_changed") is not False
        or revision_coordinate_repair.get("successor_adapt_pipeline_sha256")
        != ACCEPTED_BATCH_COORDINATE_REPAIRED_SOURCE_SHA256[
            "pipelines/static_adapt/adapt_pipeline.py"
        ]
    ):
        raise ValueError("accepted batch-coordinate repair provenance drift")
'''
    return text[:loop_start] + replacement + text[loop_end:]


def build_family(spec: dict[str, str]) -> dict[str, Any]:
    parent = INPUT / spec["parent"]
    output = INPUT / spec["output"]
    if output.exists():
        raise FileExistsError(output)
    state = source_lock_state(parent)
    source_sha = sha256(parent / "source_locked.tar.gz")
    parent_job = json.loads((parent / "jobs/weak_weak.json").read_text())
    route = parent_job[
        "route_identity"
    ]["profile_contract_sha256"]
    coordinate_repair = parent_job["source_lock"][
        "accepted_batch_coordinate_receipt_repair"
    ]
    predecessor_source_sha = coordinate_repair[
        "predecessor_source_archive_sha256"
    ]
    archive_manifest = json.loads(
        (parent / "source_archive_manifest.json").read_text()
    )
    current_adapt_sha = archive_manifest["files"][
        "pipelines/static_adapt/adapt_pipeline.py"
    ]["sha256"]
    shutil.copytree(parent, output, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    replacements = {
        spec["parent"]: spec["output"],
        spec["parent_batch"]: spec["output_batch"],
    }
    for relative in (
        "execute_source_locked_job.sh",
        "run_job.py",
        "evidence_validation.py",
        "validate_fetched.py",
        "submit.sub",
        "README.md",
        "queue.tsv",
    ):
        path = output / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for old, new in replacements.items():
            text = text.replace(old, new)
        if relative == "run_job.py":
            text = patch_validator_repair_chain(
                text,
                expected_state=state,
                current_adapt_sha256=current_adapt_sha,
                predecessor_source_sha256=predecessor_source_sha,
                current_source_sha256=source_sha,
            )
        path.write_text(text, encoding="utf-8")

    repair = {
        "schema": SCHEMA,
        "predecessor_bundle": spec["parent"],
        "successor_bundle": spec["output"],
        "source_archive_sha256": source_sha,
        "route_contract_sha256": route,
        "scientific_source_archive_changed": False,
        "scientific_settings_changed": False,
        "repair": (
            "align the bundle-side worker validator SOURCE_LOCK_STATE constant "
            "with the already serialized accepted-coordinate successor manifests"
        ),
    }

    for path in sorted(output.rglob("*.json")):
        if path.name == "submission_artifact_hashes.json":
            continue
        value = replace_tree(json.loads(path.read_text()), replacements)
        dump(path, value)
    archive_manifest_sha = sha256(output / "source_archive_manifest.json")
    revision_manifest_sha = sha256(output / "source_revision_manifest.json")
    for folder in (output / "jobs", output / "normalized_manifests"):
        for path in sorted(folder.glob("*.json")):
            value = json.loads(path.read_text())
            lock = value.get("source_lock", {})
            lock["source_archive_manifest_sha256"] = archive_manifest_sha
            lock["source_revision_manifest_sha256"] = revision_manifest_sha
            value["source_lock"] = lock
            dump(path, value)
    dump(output / "bundle_contract_closure_repair.json", repair)

    verifier = f'''#!/usr/bin/env python3
import hashlib, json, re
from pathlib import Path
B=Path(__file__).resolve().parent
SOURCE={source_sha!r}
ROUTE={route!r}
STATE={state!r}
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
    assert h(B/"source_locked.tar.gz")==SOURCE
    jobs=sorted((B/"jobs").glob("*.json")); assert len(jobs)==6
    for path in jobs:
        job=json.loads(path.read_text())
        assert job["bundle_id"]=={spec['output']!r}
        assert job["batch_name"]=={spec['output_batch']!r}
        assert job["source_lock"]["source_archive_sha256"]==SOURCE
        assert job["source_lock"]["worker_source_mode"]==STATE
        assert job["route_identity"]["profile_contract_sha256"]==ROUTE
    text=(B/"run_job.py").read_text()
    match=re.search(r"SOURCE_LOCK_STATE = ([^\\n]+)", text); assert match
    assert eval(match.group(1), {{}})==STATE
    assert "requirements = False" not in (B/"submit.sub").read_text()
    return True
if __name__=="__main__": verify(); print("batch3 contract-closure successor verified")
'''
    (output / "build_bundle.py").write_text(verifier, encoding="utf-8")
    (output / "test_bundle.py").write_text(
        "import build_bundle\ndef test_bundle(): assert build_bundle.verify()\n",
        encoding="utf-8",
    )
    subprocess.run([sys.executable, str(output / "build_bundle.py")], check=True)
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(output / "test_bundle.py")],
        check=True,
    )
    dump(
        output / "submission_artifact_hashes.json",
        {
            "schema": "paper_i_sr_batch3_contract_closure_artifacts_v1",
            "bundle_id": spec["output"],
            "files": {
                path.relative_to(output).as_posix(): {
                    "sha256": sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "submission_artifact_hashes.json"
            },
        },
    )
    return {
        "family": spec["name"],
        "bundle": spec["output"],
        "batch": spec["output_batch"],
        "source_archive_sha256": source_sha,
        "route_contract_sha256": route,
        "jobs": 6,
    }


def main() -> int:
    print(json.dumps([build_family(spec) for spec in FAMILIES], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
