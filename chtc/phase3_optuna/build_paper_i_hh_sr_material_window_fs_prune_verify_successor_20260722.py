#!/usr/bin/env python3
"""Build the immutable Test-2 successor after the v1 pre-science scope bug.

The v1 source archive is the sole parent.  This builder changes one execution
seam: normalize the already configured prune-prefilter policy before the route
invariant receipt reads it.  No route contract or scientific setting changes.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import build_paper_i_hh_sr_material_window_anchor_20260721 as common


ROOT = common.ROOT
INPUT = common.INPUT
BASE_ID = (
    "paper_i_hh_sr_snake_no_overlap_trust_material_window_fs_prune_verify_"
    "all_six_r50_20260722_v1_chtc"
)
BASE_BATCH = "paper-i-hh-sr-material-window-fsprune-verify-six-r50-20260722-v1"
BASE = INPUT / BASE_ID
OUTPUT_ID = (
    "paper_i_hh_sr_snake_no_overlap_trust_material_window_fs_prune_verify_"
    "all_six_r50_20260722_v2_chtc"
)
OUTPUT_BATCH = "paper-i-hh-sr-material-window-fsprune-verify-six-r50-20260722-v2"
OUTPUT = INPUT / OUTPUT_ID

ROUTE_DIGEST = "b43b23181ab1d93294fd2fb4ab96b32f7669c82db38082c86af39636cdf05201"
BASE_SOURCE_SHA256 = "b7eff9260bc304fc56279e17f993c9465d51f30bd66f193c1afa9232f28a3b39"
BASE_ADAPT_SHA256 = "1703de58fd95400114ae94b8ecb88ce4a16aa7d1a748ea4bed9c1c3d9cd67bc2"

OLD_SEAM = '''    phase1_prune_endpoint_overlap_policy_key = str(
        phase1_prune_endpoint_overlap_policy or "off"
    ).strip().lower()
    if (
        sr_controller_ablation_contract_key
'''
NEW_SEAM = '''    phase1_prune_endpoint_overlap_policy_key = str(
        phase1_prune_endpoint_overlap_policy or "off"
    ).strip().lower()
    phase1_prune_prefilter_policy_key = str(
        phase1_prune_prefilter_policy or PRUNE_PREFILTER_OFF
    ).strip().lower()
    if (
        sr_controller_ablation_contract_key
'''


def _load(path: Path) -> dict[str, Any]:
    return common.load(path)


def _dump(path: Path, value: Any) -> None:
    common.dump(path, value)


def _extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(destination, filter="data")


def _build_source(temp: Path) -> tuple[Path, dict[str, Any]]:
    archive = BASE / "source_locked.tar.gz"
    if common.sha256(archive) != BASE_SOURCE_SHA256:
        raise ValueError("Test-2 v1 source archive hash drift")
    source = temp / "source"
    _extract(archive, source)
    adapt = source / "pipelines/static_adapt/adapt_pipeline.py"
    if common.sha256(adapt) != BASE_ADAPT_SHA256:
        raise ValueError("Test-2 v1 adapt source hash drift")
    text = adapt.read_text(encoding="utf-8")
    if text.count(OLD_SEAM) != 1:
        raise ValueError("prune-prefilter normalization repair seam drift")
    text = text.replace(OLD_SEAM, NEW_SEAM, 1)
    ast.parse(text)
    adapt.write_text(text, encoding="utf-8")

    # A dependency-free regression gate for the exact lexical-order failure.
    regression = source / "test/test_static_adapt_prune_prefilter_scope_order.py"
    regression.write_text(
        '''from __future__ import annotations

import ast
from pathlib import Path


def test_prune_prefilter_policy_is_bound_before_first_route_receipt_read() -> None:
    path = Path("pipelines/static_adapt/adapt_pipeline.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_hardcoded_adapt_vqe"
    )
    events: list[tuple[int, str]] = []
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and node.id == "phase1_prune_prefilter_policy_key":
            events.append((int(node.lineno), type(node.ctx).__name__))
    events.sort()
    assert events
    assert events[0][1] == "Store"
    assert any(kind == "Load" for _, kind in events[1:])
''',
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", regression.relative_to(source).as_posix()],
        cwd=source,
        env=env,
        check=True,
    )
    common.strip_bytecode(source)
    successor = temp / "source_locked.tar.gz"
    common.deterministic_archive(source, successor)
    repair = {
        "schema": "paper_i_sr_test2_prune_prefilter_scope_repair_v1",
        "predecessor_bundle": BASE_ID,
        "predecessor_source_archive_sha256": BASE_SOURCE_SHA256,
        "predecessor_cluster": 9308673,
        "failure_class": "pre_science_unbound_normalized_prune_prefilter_policy_key",
        "changed_paths": [
            "pipelines/static_adapt/adapt_pipeline.py",
            "test/test_static_adapt_prune_prefilter_scope_order.py",
        ],
        "route_contract_sha256_unchanged": ROUTE_DIGEST,
        "scientific_settings_changed": False,
        "algorithmic_query_delta": 0,
    }
    return successor, repair


def _patch_bundle_text(replacements: Mapping[str, str]) -> None:
    for path in sorted(OUTPUT.rglob("*")):
        if not path.is_file() or path.name == "source_locked.tar.gz":
            continue
        if path.suffix == ".json":
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            _dump(path, common.replace_tree(value, replacements))
        elif path.suffix in {".py", ".sh", ".sub", ".tsv", ".md", ".txt"}:
            common.patch_text(path, replacements)


def _source_inventory(source_archive: Path) -> tuple[int, dict[str, dict[str, Any]]]:
    with tempfile.TemporaryDirectory(prefix="paper-i-test2-v2-inventory-") as raw:
        root = Path(raw)
        _extract(source_archive, root)
        inventory = common.inventory(root)
    return len(inventory), inventory


def build() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(f"immutable successor already exists: {OUTPUT}")
    if not BASE.is_dir():
        raise FileNotFoundError(BASE)
    with tempfile.TemporaryDirectory(prefix="paper-i-test2-v2-source-") as raw:
        successor, repair = _build_source(Path(raw))
        successor_sha = common.sha256(successor)
        shutil.copytree(BASE, OUTPUT, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        common.clean_inherited_bundle_state(OUTPUT)
        for name in (
            "route_parity.json",
            "scientific_settings_audit.json",
            "source_locked_sensitivity_audit.json",
            "material_window_threshold_source_audit.json",
            "fanout_bundle_receipt.json",
        ):
            shutil.copy2(BASE / name, OUTPUT / name)
        for name in (
            "remote_execution_gate.json",
            "submission_artifact_hashes.json",
            "submission_receipt.json",
        ):
            (OUTPUT / name).unlink(missing_ok=True)
        shutil.copy2(successor, OUTPUT / "source_locked.tar.gz")

    replacements = {
        BASE_ID: OUTPUT_ID,
        BASE_BATCH: OUTPUT_BATCH,
        BASE_SOURCE_SHA256: successor_sha,
        "sr-material-window-fsprune-verify-r0-r50-20260722-v1": (
            "sr-material-window-fsprune-verify-r0-r50-20260722-v2"
        ),
        "sr-material-window-fsprune-verify-r0-r{target}-20260722-v1": (
            "sr-material-window-fsprune-verify-r0-r{target}-20260722-v2"
        ),
    }
    _patch_bundle_text(replacements)

    count, inventory = _source_inventory(OUTPUT / "source_locked.tar.gz")
    source_manifest = _load(OUTPUT / "source_archive_manifest.json")
    source_manifest.update(
        {
            "archive": f"chtc/phase3_optuna/input/{OUTPUT_ID}/source_locked.tar.gz",
            "archive_sha256": successor_sha,
            "archive_size_bytes": (OUTPUT / "source_locked.tar.gz").stat().st_size,
            "file_count": count,
            "files": inventory,
            "operational_repair": copy.deepcopy(repair),
        }
    )
    _dump(OUTPUT / "source_archive_manifest.json", source_manifest)
    source_manifest_sha = common.sha256(OUTPUT / "source_archive_manifest.json")

    revision = _load(OUTPUT / "source_revision_manifest.json")
    revision["operational_repair"] = copy.deepcopy(repair)
    _dump(OUTPUT / "source_revision_manifest.json", revision)
    revision_sha = common.sha256(OUTPUT / "source_revision_manifest.json")
    physics_sha = common.sha256(OUTPUT / "physics_and_exact_reference_lock.json")

    for job_path in sorted((OUTPUT / "jobs").glob("*.json")):
        job = _load(job_path)
        if job["route_identity"]["profile_contract_sha256"] != ROUTE_DIGEST:
            raise ValueError(f"route digest drift in {job_path.name}")
        job["source_lock"].update(
            {
                "source_archive_sha256": successor_sha,
                "source_archive_manifest_sha256": source_manifest_sha,
                "source_revision_manifest_sha256": revision_sha,
                "physics_reference_lock_sha256": physics_sha,
                "operational_repair": copy.deepcopy(repair),
            }
        )
        job["source_locked_sensitivity"]["non_swept_settings_diff"] = []
        _dump(job_path, job)
        normalized_path = OUTPUT / "normalized_manifests" / job_path.name
        normalized = _load(normalized_path)
        normalized["source_lock"] = copy.deepcopy(job["source_lock"])
        normalized["source_locked_sensitivity"] = copy.deepcopy(
            job["source_locked_sensitivity"]
        )
        _dump(normalized_path, normalized)

    sensitivity = _load(OUTPUT / "source_locked_sensitivity_audit.json")
    sensitivity.update(
        {
            "candidate_bundle": OUTPUT_ID,
            "candidate_archive_sha256": successor_sha,
            "operational_repair": copy.deepcopy(repair),
            "non_swept_settings_diff": [],
        }
    )
    _dump(OUTPUT / "source_locked_sensitivity_audit.json", sensitivity)
    _dump(OUTPUT / "operational_repair.json", repair)

    for receipt_name in ("fanout_bundle_receipt.json", "bundle_manifest.json"):
        receipt = _load(OUTPUT / receipt_name)
        receipt.update(
            {
                "bundle_id": OUTPUT_ID,
                "batch_name": OUTPUT_BATCH,
                "source_archive_sha256": successor_sha,
                "source_archive_manifest_sha256": source_manifest_sha,
                "source_revision_manifest_sha256": revision_sha,
                "physics_reference_lock_sha256": physics_sha,
                "operational_repair": copy.deepcopy(repair),
                "submission_performed": False,
            }
        )
        _dump(OUTPUT / receipt_name, receipt)

    queue_rel = f"chtc/phase3_optuna/input/{OUTPUT_ID}/queue.tsv"
    (OUTPUT / "submit.sub").write_text(
        common.submit_text(OUTPUT_ID, OUTPUT_BATCH, successor_sha, queue_rel),
        encoding="utf-8",
    )
    verifier = f'''#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
 r=json.loads((B/"fanout_bundle_receipt.json").read_text())
 assert r["bundle_id"]=={OUTPUT_ID!r}
 assert r["batch_name"]=={OUTPUT_BATCH!r}
 assert h(B/"source_locked.tar.gz")==r["source_archive_sha256"]
 repair=json.loads((B/"operational_repair.json").read_text())
 assert repair["scientific_settings_changed"] is False
 assert repair["route_contract_sha256_unchanged"]=={ROUTE_DIGEST!r}
 jobs=sorted((B/"jobs").glob("*.json")); assert len(jobs)==6
 for p in jobs:
  j=json.loads(p.read_text())
  assert j["route_identity"]["profile_contract_sha256"]=={ROUTE_DIGEST!r}
  assert int(j["segment"]["target_controller_round"])==50
  assert j["physics"]["same_cutoff_reference"] is True
  assert j["source_lock"]["source_archive_sha256"]==r["source_archive_sha256"]
  c=j["route_identity"]["profile_contract"]
  assert c["semantic_invariants"]["prune_verification_beam"]=="minimal_immutable_keep_vs_one_delete_refit_sibling_v1"
  assert c["semantic_invariants"]["historical_admission_beam_active"] is False
 assert "requirements = False" not in (B/"submit.sub").read_text()
 return True
if __name__=="__main__": verify(); print("Test-2 v2 successor verified")
'''
    (OUTPUT / "build_bundle.py").write_text(verifier, encoding="utf-8")
    (OUTPUT / "test_bundle.py").write_text(
        "import build_bundle\ndef test_bundle(): assert build_bundle.verify()\n",
        encoding="utf-8",
    )
    return _load(OUTPUT / "fanout_bundle_receipt.json")


def archive_preflight() -> None:
    with tempfile.TemporaryDirectory(prefix="paper-i-test2-v2-preflight-") as raw:
        root = Path(raw)
        _extract(OUTPUT / "source_locked.tar.gz", root)
        target = root / "chtc/phase3_optuna/input" / OUTPUT_ID
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(OUTPUT, target)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        for job in sorted((target / "jobs").glob("*.json")):
            subprocess.run(
                [sys.executable, str(target / "run_job.py"), "--validate-only", str(job)],
                cwd=root,
                env=env,
                check=True,
            )
        subprocess.run([sys.executable, str(target / "build_bundle.py")], check=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-preflight", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = build()
    if args.archive_preflight:
        archive_preflight()
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
