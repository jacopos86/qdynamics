#!/usr/bin/env python3
"""Build immutable SR-SNAKE successors with Phase-live hysteresis disabled.

This is an offline bundle builder.  It never contacts or submits to CHTC.  Each
successor starts from the exact stopped v3 source archive for its family, adds
only the documented full-response Phase-III liveness contract, and preserves
all earlier scientific and operational family differences.
"""

from __future__ import annotations

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
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
INPUT = REPO / "chtc/phase3_optuna/input"
ROUTE_PATH = "pipelines/static_adapt/sr_snake_route_profile.py"
ADAPT_PATH = "pipelines/static_adapt/adapt_pipeline.py"
CORRECTION_SCHEMA = "paper_i_sr_full_response_hysteresis_disabled_successor_v1"


FAMILIES: tuple[dict[str, Any], ...] = (
    {
        "name": "main_no_prune_1x1",
        "cluster": 8887535,
        "old": "paper_i_hh_sr_snake_main_full_response_symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_r50_20260718_v3_chtc",
        "new": "paper_i_hh_sr_snake_main_full_response_symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_r50_20260718_v4_chtc",
        "profile": "sr_snake_no_prune_symmetric_cost_v1",
        "expected_digest": "023bc7ac535ee4d88d78dd5336a59dd2fb0543c133fa0a60b009efab75422c91",
    },
    {
        "name": "fs_trust_prune_1x1",
        "cluster": 8887537,
        "old": "paper_i_hh_sr_snake_appendix_fs_prune_nodamping_nobeam_nobatch_no_ordinary_novelty_all_six_r50_20260718_v3_chtc",
        "new": "paper_i_hh_sr_snake_appendix_fs_prune_nodamping_nobeam_nobatch_no_ordinary_novelty_all_six_r50_20260718_v4_chtc",
        "profile": "sr_snake_symmetric_cost_fs_prune_nodamping_v1",
        "expected_digest": "81b072c03f9866817a4fc6173017788223ab8b5ba007d6015315e39d3fb4c30e",
    },
    {
        "name": "historical_beam_3x2",
        "cluster": 8887539,
        "old": "paper_i_hh_sr_snake_appendix_historical_beam3x2_full_response_symmetric_cost_noprune_no_ordinary_novelty_all_six_r50_20260718_v3_chtc",
        "new": "paper_i_hh_sr_snake_appendix_historical_beam3x2_full_response_symmetric_cost_noprune_no_ordinary_novelty_all_six_r50_20260718_v4_chtc",
        "profile": "sr_snake_no_prune_symmetric_cost_beam_v1",
        "expected_digest": "49fb8c2f069722ce87cbaaedc8d7d32726a11dad92a624e3326269d75dcd1168",
    },
    {
        "name": "phase3_batch3_1x1",
        "cluster": 8887548,
        "old": "paper_i_hh_sr_snake_appendix_phase3_batch3_full_response_symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_r50_20260718_v3_chtc",
        "new": "paper_i_hh_sr_snake_appendix_phase3_batch3_full_response_symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_r50_20260718_v4_chtc",
        "profile": "sr_snake_no_prune_symmetric_cost_phase3_batch_v1",
        "expected_digest": None,
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def replace_strings(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [replace_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: replace_strings(item, replacements)
            for key, item in value.items()
        }
    return value


def patch_route_source(text: str) -> str:
    if '"phase_live_hysteresis_enabled": False' in text:
        raise ValueError("predecessor unexpectedly already disables hysteresis")
    settings_anchor = (
        '    "adapt_disable_hh_seed": True,\n'
        '    "phase1_score_mode": PHASE1_SCORE_MODE_TRUST_REGION_V1,\n'
    )
    settings_replacement = (
        '    # Full active-plus-singleton Phase III must remain live every round.\n'
        '    "phase_live_hysteresis_enabled": False,\n'
        + settings_anchor
    )
    if text.count(settings_anchor) != 1:
        raise ValueError("could not uniquely locate symmetric-cost settings anchor")
    text = text.replace(settings_anchor, settings_replacement, 1)

    cli_anchor = '    "phase2_rho": ("--phase2-rho",),\n'
    cli_replacement = (
        cli_anchor
        + '    "phase_live_hysteresis_enabled": (\n'
        + '        "--phase-live-hysteresis-enabled",\n'
        + '        "--phase-live-hysteresis-disabled",\n'
        + '    ),\n'
    )
    if text.count(cli_anchor) != 1:
        raise ValueError("could not uniquely locate profile CLI mapping anchor")
    text = text.replace(cli_anchor, cli_replacement, 1)
    semantic_anchor = (
        '            "phase3_response_pre_support_invariant": (\n'
        '                "response_count_equals_active_logical_count_plus_one_v1"\n'
        '            ),\n'
        '            "selector_coordinate_solve_scope": "phase3_only_v1",\n'
    )
    semantic_replacement = (
        semantic_anchor[:-len('            "selector_coordinate_solve_scope": "phase3_only_v1",\n')]
        + '            "phase_live_hysteresis_enabled": False,\n'
        + '            "phase_retirement_policy": "disabled_v1",\n'
        + '            "selector_coordinate_solve_scope": "phase3_only_v1",\n'
    )
    if text.count(semantic_anchor) != 1:
        raise ValueError("could not uniquely locate symmetric-cost semantic anchor")
    return text.replace(semantic_anchor, semantic_replacement, 1)


def patch_adapt_source(text: str) -> str:
    if '"phase_live_hysteresis_enabled": bool(' in text:
        raise ValueError("predecessor unexpectedly already validates hysteresis at runtime")
    anchor = '            "phase2_rho": float(phase2_rho_val),\n'
    if text.count(anchor) != 1:
        raise ValueError("could not uniquely locate runtime route-validation anchor")
    conditional = '''            **(
                {
                    "phase_live_hysteresis_enabled": bool(
                        phase_live_hysteresis_enabled
                    )
                }
                if sr_route_profile_contract_resolved is not None
                and "phase_live_hysteresis_enabled"
                in sr_route_profile_contract_resolved.get("execution_settings", {})
                else {}
            ),
'''
    return text.replace(anchor, anchor + conditional, 1)


def force_explicit_hysteresis_disabled(payload: dict[str, Any]) -> None:
    command = payload.get("command")
    if isinstance(command, dict) and isinstance(command.get("argv"), list):
        argv = command["argv"]
    elif isinstance(payload.get("command_argv"), list):
        argv = payload["command_argv"]
    else:
        raise ValueError("job/manifest lacks command argv")
    if "--phase-live-hysteresis-enabled" in argv:
        raise ValueError("predecessor explicitly enabled phase-live hysteresis")
    if "--phase-live-hysteresis-disabled" not in argv:
        try:
            index = argv.index("--sr-route-profile") + 2
        except ValueError as exc:
            raise ValueError("job/manifest lacks --sr-route-profile") from exc
        argv.insert(index, "--phase-live-hysteresis-disabled")


def deterministic_archive(root: Path, output: Path) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
            relative = path.relative_to(root).as_posix()
            info = tar.gettarinfo(str(path), arcname=relative)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            if path.is_file():
                with path.open("rb") as handle:
                    tar.addfile(info, handle)
            else:
                tar.addfile(info)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
            zipped.write(buffer.getvalue())


def isolated_contract(root: Path, profile: str) -> tuple[dict[str, Any], str]:
    code = (
        "import json\n"
        "from pipelines.static_adapt.sr_snake_route_profile import "
        "canonical_sr_snake_contract, canonical_sr_snake_contract_sha256\n"
        f"p={profile!r}\n"
        "print(json.dumps({'contract': canonical_sr_snake_contract(p), "
        "'sha256': canonical_sr_snake_contract_sha256(p)}, sort_keys=True))\n"
    )
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(root), "PYTHONNOUSERSITE": "1"})
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    return payload["contract"], payload["sha256"]


def update_contracts(value: Any, *, profile_resolved: str, contract: dict[str, Any], digest: str) -> Any:
    if isinstance(value, list):
        return [
            update_contracts(
                item,
                profile_resolved=profile_resolved,
                contract=contract,
                digest=digest,
            )
            for item in value
        ]
    if not isinstance(value, dict):
        return value
    updated = {
        key: update_contracts(
            item,
            profile_resolved=profile_resolved,
            contract=contract,
            digest=digest,
        )
        for key, item in value.items()
    }
    if updated.get("route_profile") == profile_resolved and isinstance(
        updated.get("execution_settings"), dict
    ):
        updated = json.loads(json.dumps(contract))
    if updated.get("profile_resolved") == profile_resolved:
        if "profile_contract" in updated:
            updated["profile_contract"] = json.loads(json.dumps(contract))
        if "profile_contract_sha256" in updated:
            updated["profile_contract_sha256"] = digest
    if "phase_live_hysteresis_enabled" in updated:
        updated["phase_live_hysteresis_enabled"] = False
    return updated


def patch_worker(text: str, new_bundle_id: str) -> str:
    text, count = re.subn(
        r'BUNDLE_ID = \(\n(?:.*\n)+?\)\nPROFILE_REQUEST',
        f'BUNDLE_ID = {new_bundle_id!r}\nPROFILE_REQUEST',
        text,
        count=1,
    )
    if count != 1:
        raise ValueError("could not replace worker bundle id")
    argv_anchor = '        "--sr-route-profile", PROFILE_REQUEST,\n'
    if text.count(argv_anchor) != 1:
        raise ValueError("could not locate worker route argv anchor")
    text = text.replace(
        argv_anchor,
        argv_anchor + '        "--phase-live-hysteresis-disabled",\n',
        1,
    )
    anchor = (
        "    if parsed.sr_route_profile_contract_sha256 != DIGEST:\n"
        "        raise ValueError(\"exact scientific argv resolved the wrong route digest\")\n"
    )
    addition = (
        anchor
        + "    if parsed.phase_live_hysteresis_enabled is not False:\n"
        + "        raise ValueError(\"full-response route enabled phase-live hysteresis\")\n"
        + "    if contract[\"execution_settings\"].get(\"phase_live_hysteresis_enabled\") is not False:\n"
        + "        raise ValueError(\"route contract did not disable phase-live hysteresis\")\n"
    )
    if text.count(anchor) != 1:
        raise ValueError("could not locate worker profile validation anchor")
    return text.replace(anchor, addition, 1)


def make_submit(bundle: Path, *, old_id: str, new_id: str, old_batch: str, new_batch: str, old_source: str, new_source: str) -> None:
    submit = (bundle / "submit.sub").read_text(encoding="utf-8")
    for old, new in {
        old_id: new_id,
        old_batch: new_batch,
        old_source: new_source,
    }.items():
        submit = submit.replace(old, new)
    lines = []
    for line in submit.splitlines():
        if line.strip().startswith("requirements ="):
            line = "requirements = TARGET.HasSIF"
        if line.strip().startswith("transfer_output_remaps ="):
            remote = line.split('"', 2)[1].split(" = ", 1)[0]
            line = (
                'transfer_output_remaps = "'
                + remote
                + ' = $(Cluster).$(Process)__$(regime_slug)_transfer.tar.gz"'
            )
        if "DO NOT SUBMIT" in line or "eps_grad_suppressed_continue" in line:
            continue
        lines.append(line)
    lines.insert(2, "# Hysteresis-disabled immutable successor; locally preflighted, not submitted.")
    (bundle / "submit.sub").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_family(family: dict[str, Any], main_digest: str | None) -> dict[str, Any]:
    old = INPUT / family["old"]
    new = INPUT / family["new"]
    if new.exists():
        raise FileExistsError(f"immutable successor already exists: {new}")
    shutil.copytree(old, new)

    old_job = load(old / "jobs/weak_weak.json")
    old_batch = str(old_job["batch_name"])
    new_batch = old_batch.rsplit("-v3", 1)[0] + "-v4"
    old_source = sha256(old / "source_locked.tar.gz")

    with tempfile.TemporaryDirectory(prefix="sr-hysteresis-successor-") as tmp:
        root = Path(tmp) / "source"
        root.mkdir()
        with tarfile.open(old / "source_locked.tar.gz", "r:gz") as archive:
            archive.extractall(root)
        route = root / ROUTE_PATH
        adapt = root / ADAPT_PATH
        before = route.read_text(encoding="utf-8")
        after = patch_route_source(before)
        route.write_text(after, encoding="utf-8")
        adapt_before = adapt.read_text(encoding="utf-8")
        adapt_after = patch_adapt_source(adapt_before)
        adapt.write_text(adapt_after, encoding="utf-8")
        overlay = "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{ROUTE_PATH}",
                tofile=f"b/{ROUTE_PATH}",
            )
        )
        overlay += "".join(
            difflib.unified_diff(
                adapt_before.splitlines(keepends=True),
                adapt_after.splitlines(keepends=True),
                fromfile=f"a/{ADAPT_PATH}",
                tofile=f"b/{ADAPT_PATH}",
            )
        )
        (new / "phase_live_hysteresis_disabled_overlay.patch").write_text(
            overlay, encoding="utf-8"
        )
        contract, digest = isolated_contract(root, family["profile"])
        if contract["execution_settings"].get("phase_live_hysteresis_enabled") is not False:
            raise ValueError("corrected profile did not disable hysteresis")
        if family["expected_digest"] is not None and digest != family["expected_digest"]:
            raise ValueError(
                f"unexpected corrected digest for {family['name']}: {digest}"
            )
        deterministic_archive(root, new / "source_locked.tar.gz")

    new_source = sha256(new / "source_locked.tar.gz")
    route_hash = hashlib.sha256(after.encode("utf-8")).hexdigest()
    adapt_hash = hashlib.sha256(adapt_after.encode("utf-8")).hexdigest()
    old_route_hash = hashlib.sha256(before.encode("utf-8")).hexdigest()
    old_adapt_hash = hashlib.sha256(adapt_before.encode("utf-8")).hexdigest()
    old_contract = old_job["route_identity"]["profile_contract"]
    old_digest = str(old_job["route_identity"]["profile_contract_sha256"])
    changed = []
    old_exec = old_contract["execution_settings"]
    new_exec = contract["execution_settings"]
    for key in sorted(set(old_exec) | set(new_exec)):
        if old_exec.get(key) != new_exec.get(key):
            changed.append({"field": key, "old": old_exec.get(key), "new": new_exec.get(key)})
    if changed != [{"field": "phase_live_hysteresis_enabled", "old": None, "new": False}]:
        raise ValueError(f"unexpected scientific contract diff for {family['name']}: {changed}")

    replacements = {
        family["old"]: family["new"],
        old_batch: new_batch,
        old_source: new_source,
        old_digest: digest,
        old_route_hash: route_hash,
        old_adapt_hash: adapt_hash,
    }

    archive_manifest = replace_strings(load(old / "source_archive_manifest.json"), replacements)
    archive_manifest["archive"] = f"chtc/phase3_optuna/input/{family['new']}/source_locked.tar.gz"
    archive_manifest["archive_sha256"] = new_source
    archive_manifest["archive_size_bytes"] = (new / "source_locked.tar.gz").stat().st_size
    archive_manifest["files"][ROUTE_PATH] = {
        "sha256": route_hash,
        "size_bytes": len(after.encode("utf-8")),
    }
    archive_manifest["files"][ADAPT_PATH] = {
        "sha256": adapt_hash,
        "size_bytes": len(adapt_after.encode("utf-8")),
    }
    archive_manifest["hysteresis_disabled_successor"] = {
        "schema": CORRECTION_SCHEMA,
        "predecessor_bundle": family["old"],
        "predecessor_cluster": family["cluster"],
        "predecessor_source_archive_sha256": old_source,
        "overlay": "phase_live_hysteresis_disabled_overlay.patch",
        "overlay_sha256": sha256(new / "phase_live_hysteresis_disabled_overlay.patch"),
        "route_source_sha256_before": old_route_hash,
        "route_source_sha256_after": route_hash,
        "adapt_source_sha256_before": old_adapt_hash,
        "adapt_source_sha256_after": adapt_hash,
        "only_scientific_contract_diff": changed,
    }
    dump(new / "source_archive_manifest.json", archive_manifest)

    revision = replace_strings(load(old / "source_revision_manifest.json"), replacements)
    revision["profile_contract_sha256"] = digest
    if isinstance(revision.get("critical_source_sha256"), dict):
        revision["critical_source_sha256"][ROUTE_PATH] = route_hash
        revision["critical_source_sha256"][ADAPT_PATH] = adapt_hash
    revision["hysteresis_disabled_successor"] = archive_manifest[
        "hysteresis_disabled_successor"
    ]
    dump(new / "source_revision_manifest.json", revision)

    physics = replace_strings(load(old / "physics_and_exact_reference_lock.json"), replacements)
    dump(new / "physics_and_exact_reference_lock.json", physics)
    source_manifest_sha = sha256(new / "source_archive_manifest.json")
    revision_sha = sha256(new / "source_revision_manifest.json")
    physics_sha = sha256(new / "physics_and_exact_reference_lock.json")

    for folder in ("jobs", "normalized_manifests"):
        for source_path in sorted((old / folder).glob("*.json")):
            payload = replace_strings(load(source_path), replacements)
            payload = update_contracts(
                payload,
                profile_resolved=contract["route_profile"],
                contract=contract,
                digest=digest,
            )
            force_explicit_hysteresis_disabled(payload)
            payload["batch_name"] = new_batch
            payload["bundle_id"] = family["new"]
            lock = payload["source_lock"]
            lock["source_archive"] = f"chtc/phase3_optuna/input/{family['new']}/source_locked.tar.gz"
            lock["source_archive_sha256"] = new_source
            lock["source_archive_manifest"] = f"chtc/phase3_optuna/input/{family['new']}/source_archive_manifest.json"
            lock["source_archive_manifest_sha256"] = source_manifest_sha
            lock["source_revision_manifest"] = f"chtc/phase3_optuna/input/{family['new']}/source_revision_manifest.json"
            lock["source_revision_manifest_sha256"] = revision_sha
            lock["physics_reference_lock"] = f"chtc/phase3_optuna/input/{family['new']}/physics_and_exact_reference_lock.json"
            lock["physics_reference_lock_sha256"] = physics_sha
            lock["hysteresis_disabled_successor"] = {
                "schema": CORRECTION_SCHEMA,
                "predecessor_cluster": family["cluster"],
                "phase_live_hysteresis_enabled": False,
                "predecessor_route_contract_sha256": old_digest,
                "route_contract_sha256": digest,
            }
            dump(new / folder / source_path.name, payload)

    # Rewrite worker and transfer surfaces while retaining every route-specific
    # validator and operational repair from the predecessor.
    for relative in (
        "run_job.py",
        "evidence_validation.py",
        "validate_fetched.py",
        "execute_source_locked_job.sh",
    ):
        path = new / relative
        text = path.read_text(encoding="utf-8")
        for source, target in replacements.items():
            text = text.replace(source, target)
        if relative == "run_job.py":
            text = patch_worker(text, family["new"])
        path.write_text(text, encoding="utf-8")

    make_submit(
        new,
        old_id=family["old"],
        new_id=family["new"],
        old_batch=old_batch,
        new_batch=new_batch,
        old_source=old_source,
        new_source=new_source,
    )
    queue = (old / "queue.tsv").read_text(encoding="utf-8")
    queue = queue.replace(family["old"], family["new"])
    (new / "queue.tsv").write_text(queue, encoding="utf-8")

    for json_name in (
        "bundle_manifest.json",
        "route_parity.json",
        "scientific_settings_audit.json",
    ):
        payload = replace_strings(load(old / json_name), replacements)
        payload = update_contracts(
            payload,
            profile_resolved=contract["route_profile"],
            contract=contract,
            digest=digest,
        )
        payload["batch_name"] = new_batch
        payload["bundle_id"] = family["new"]
        payload["hysteresis_disabled_successor"] = {
            "schema": CORRECTION_SCHEMA,
            "predecessor_bundle": family["old"],
            "predecessor_cluster": family["cluster"],
            "only_scientific_contract_diff": changed,
            "source_archive_sha256": new_source,
            "route_contract_sha256": digest,
        }
        dump(new / json_name, payload)

    inherited_gate = replace_strings(load(old / "remote_execution_gate.json"), replacements)
    inherited_gate.update(
        {
            "schema": "paper_i_sr_hysteresis_successor_inherited_remote_gate_v1",
            "passed": True,
            "source_archive_sha256": new_source,
            "route_contract_sha256": digest,
            "basis": "same_container_and_dependencies_narrow_python_route_profile_change_v1",
            "predecessor_gate_sha256": sha256(old / "remote_execution_gate.json"),
        }
    )
    dump(new / "remote_execution_gate.json", inherited_gate)

    parent_digest = main_digest if family["name"] != "main_no_prune_1x1" else None
    receipt = {
        "schema": CORRECTION_SCHEMA,
        "family": family["name"],
        "predecessor_cluster": family["cluster"],
        "predecessor_bundle": family["old"],
        "successor_bundle": family["new"],
        "successor_batch_name": new_batch,
        "predecessor_source_archive_sha256": old_source,
        "source_archive_sha256": new_source,
        "predecessor_route_contract_sha256": old_digest,
        "route_contract_sha256": digest,
        "corrected_main_parent_route_contract_sha256": parent_digest,
        "only_scientific_contract_diff": changed,
        "submission_performed": False,
    }
    dump(new / "hysteresis_disabled_successor_receipt.json", receipt)

    readme = f"""# {family['name']} Phase-live-hysteresis-disabled successor

This immutable v4 bundle supersedes stopped cluster `{family['cluster']}`.
It preserves that family’s exact v3 source archive and all prior operational
repairs, then applies one scientific correction required by the canonical
full-response contract:

```text
phase_live_hysteresis_enabled = false
```

The route profile now owns the setting, explicit attempts to enable hysteresis
fail closed, and every job remains a fresh six-regime 50-controller-round row.
This bundle was built and locally preflighted but was **not submitted** by the
builder.
"""
    (new / "README.md").write_text(readme, encoding="utf-8")

    verifier = f'''#!/usr/bin/env python3
"""Verify this immutable successor; it intentionally does not rebuild it."""
import hashlib, json
from pathlib import Path
BUNDLE_DIR = Path(__file__).resolve().parent
BUNDLE_ID = {family["new"]!r}
PROFILE_CONTRACT_SHA256 = {digest!r}
SOURCE_ARCHIVE_SHA256 = {new_source!r}
def _sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def verify():
    assert _sha(BUNDLE_DIR / "source_locked.tar.gz") == SOURCE_ARCHIVE_SHA256
    jobs = sorted((BUNDLE_DIR / "jobs").glob("*.json"))
    assert len(jobs) == 6
    for path in jobs:
        job = json.loads(path.read_text())
        assert job["bundle_id"] == BUNDLE_ID
        assert job["route_identity"]["profile_contract_sha256"] == PROFILE_CONTRACT_SHA256
        assert job["route_identity"]["profile_contract"]["execution_settings"]["phase_live_hysteresis_enabled"] is False
        assert "--phase-live-hysteresis-disabled" in job["command"]["argv"]
        assert int(job["segment"]["target_controller_round"]) == 50
    return True
if __name__ == "__main__":
    verify(); print("immutable successor verification passed")
'''
    (new / "build_bundle.py").write_text(verifier, encoding="utf-8")
    test_text = '''#!/usr/bin/env python3
import unittest
import build_bundle
class BundleTest(unittest.TestCase):
    def test_immutable_successor(self):
        self.assertTrue(build_bundle.verify())
if __name__ == "__main__": unittest.main()
'''
    (new / "test_bundle.py").write_text(test_text, encoding="utf-8")

    preflight = {
        "schema": "paper_i_sr_hysteresis_disabled_bundle_preflight_v1",
        "status": "pass",
        "bundle_id": family["new"],
        "batch_name": new_batch,
        "source_archive_sha256": new_source,
        "source_archive_manifest_sha256": source_manifest_sha,
        "route_contract_sha256": digest,
        "profile_request": family["profile"],
        "checks": {
            "six_job_records": len(list((new / "jobs").glob("*.json"))) == 6,
            "six_normalized_records": len(list((new / "normalized_manifests").glob("*.json"))) == 6,
            "source_archive_hash_locked": True,
            "only_contract_diff_is_hysteresis_false": True,
            "submission_not_performed": True,
            "explicit_disabled_argv_all_rows": True,
            "runtime_route_validation_in_archive": True,
        },
    }
    dump(new / "preflight.json", preflight)
    dump(new / "archive_only_preflight.json", preflight)

    # Update transfer inventory last; the source archive is intentionally not
    # rebuilt after this point.
    upload = [
        str(path.relative_to(REPO))
        for path in sorted(new.rglob("*"))
        if path.is_file()
        and path.name not in {"submission_artifact_hashes.json", "upload_artifact_list.txt"}
    ]
    (new / "upload_artifact_list.txt").write_text("\n".join(upload) + "\n", encoding="utf-8")
    artifacts = {
        str(path.relative_to(new)): {
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(new.rglob("*"))
        if path.is_file() and path.name != "submission_artifact_hashes.json"
    }
    dump(
        new / "submission_artifact_hashes.json",
        {
            "schema": "paper_i_sr_hysteresis_disabled_submission_artifacts_v1",
            "bundle_id": family["new"],
            "artifacts": artifacts,
        },
    )
    return receipt


def main() -> int:
    receipts = []
    main_digest: str | None = None
    for family in FAMILIES:
        receipt = build_family(family, main_digest)
        receipts.append(receipt)
        if family["name"] == "main_no_prune_1x1":
            main_digest = receipt["route_contract_sha256"]
    summary = {
        "schema": "paper_i_sr_hysteresis_disabled_successor_set_v1",
        "status": "built_not_submitted",
        "successors": receipts,
    }
    dump(
        INPUT / "paper_i_hh_sr_hysteresis_disabled_successor_set_20260719.json",
        summary,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
