#!/usr/bin/env python3
"""Recover reporting artifacts for macro-only CHTC rows 8890778.0-.2.

This utility is deliberately post-run only.  It extracts only the immutable
science/result records from each original transfer archive, runs the original
bundle's postprocessor inside the repaired source archive, proves that the
execution/science/ledger records remain byte-identical, and writes a compact
immutable recovery archive plus hash receipts.  It never invokes the ADAPT
pipeline, controller, or optimizer.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


REPO = Path(__file__).resolve().parents[2]
ORIGINAL_BUNDLE_RELATIVE = Path(
    "chtc/phase3_optuna/input/"
    "paper_i_hh_sr_snake_macro_only_physical_lanes_all_six_r50_20260719_v1_chtc"
)
REPORTING_SOURCE_BUNDLE_RELATIVE = Path(
    "chtc/phase3_optuna/input/"
    "paper_i_hh_sr_snake_macro_only_physical_lanes_"
    "fidelity_repair_remaining2_r50_20260719_v2_chtc"
)
FETCH_ROOT_RELATIVE = Path(
    "raw_outputs/chtc_fetch_paper_i_hh_sr_pool_complements_20260719/"
    "heartbeat_20260719T2159Z"
)
RECOVERY_ROOT_RELATIVE = Path(
    "raw_outputs/chtc_fetch_paper_i_hh_sr_pool_complements_20260719/"
    "macro_8890778_post_run_recovery_20260719"
)
CLUSTER_ID = "8890778"
ROUTE_DIGEST = "d14d582e532ee41500cd7d3ebaa21b83da91bb3fcf014be53ab8d1049d1452fa"
ORIGINAL_SOURCE_ARCHIVE_SHA256 = (
    "3a5ed36ebdf260357aa86b3a5ab3c7d8372072329a8fec2e1043e90b6f7c34c7"
)
REPORTING_SOURCE_ARCHIVE_SHA256 = (
    "a3a80c964ba2925de1c14843c75dd2c9b8440f4e27f3f713341c96c95e40e2fc"
)
FIDELITY_REPORTER_SHA256 = (
    "91edb3bba6e20d7f84e331d8428fd422c7b5ae272b63fd3f99ebb4c9be5f2dce"
)
EXPECTED_FAILURE = (
    "ValueError: Active operator 0 execution terms disagree with parameterization."
)
ORIGINAL_RUN_JOB_SHA256 = (
    "9bdedc80fa4fba4edecaecab053344e19fcc7f33d7cdd39913b66cc4382614ce"
)
ORIGINAL_EVIDENCE_VALIDATION_SHA256 = (
    "567273813b5c89c94a0f2b459c0c16762a95bd537db4df426799110e5f37c3e3"
)
ORIGINAL_ARCHIVE_SHA256 = {
    0: "a6e29aa540ac552f39080940fe38a18547b51fe8a878527b692da0a7583e988d",
    1: "52f2018f6ce34563d3153d32d88329f582900579b3bf340ba236fd5270819953",
    2: "5aa6178f222aba2db59c6d69647a52605525eacead1da82214e372604d6f14d7",
}
ORIGINAL_EXECUTION_SHA256 = {
    0: "cd0e16ba59655a40db2c5a8c256bec8afbf8d261f05d4a1902711431537a6c85",
    1: "e4b9cbd979d34ef7fd19879337f2703366d8f3eef57e333e5b4f4bcc73a81489",
    2: "50fc609726a460f808eca29642c9e1572f3993ef329ef663884db4df5fadd65f",
}
ORIGINAL_JOB_MANIFEST_SHA256 = {
    0: "d1b9e91594944fedea6129268d51368b52f876fb78df7850192d3699c8ac83e1",
    1: "602ef4fc139a3c20c83b9d45a539943364f826c4f26c6d2930968e5a0c1408f8",
    2: "6ac4e06975d5da40dd3360b52d5a691c07a48b9061fb8157f5a179a2395a5983",
}
RECOVERY_SCHEMA = "paper_i_hh_sr_macro_post_run_recovery_v1"
RECOVERY_ARCHIVE_SCHEMA = "paper_i_hh_sr_macro_post_run_recovery_archive_v1"
AGGREGATE_SCHEMA = "paper_i_hh_sr_macro_post_run_recovery_aggregate_v1"

ROWS = {
    0: "weak_weak",
    1: "intermediate_weak",
    2: "strong_weak_u8",
}

SELECTED_SUFFIXES = (
    "execution.json",
    "normalized_run_manifest.json",
    "json/result.json",
    "json/current.json",
    "json/estimator_call_ledger.json",
)

ORIGINAL_ARTIFACTS = {
    "result_json": "json/result.json",
    "current_json": "json/current.json",
    "ledger_json": "json/estimator_call_ledger.json",
    "normalized_runtime_manifest_json": "normalized_run_manifest.json",
}

RECOVERED_ARTIFACTS = {
    "validation_json": "validation.json",
    "qiskit_cost_sidecar_json": "qiskit_cost_sidecar.json",
    "repaired_terminal_checkpoint_json": (
        "terminal_checkpoint.execution_order_repaired.json"
    ),
    "ground_space_fidelity_json": "ground_space_projector_fidelity.json",
}


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_member(member: tarfile.TarInfo) -> PurePosixPath:
    name = PurePosixPath(member.name)
    if (
        name.is_absolute()
        or ".." in name.parts
        or member.issym()
        or member.islnk()
        or not (member.isfile() or member.isdir())
        or any(
            part in {".DS_Store", "__MACOSX"} or part.startswith("._")
            for part in name.parts
        )
    ):
        raise ValueError(f"unsafe archive member: {member.name}")
    return name


def extract_source_archive(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            _safe_member(member)
        handle.extractall(destination, members=members, filter="data")


def extract_selected_transfer(
    archive: Path, destination: Path, regime: str
) -> tuple[Path, list[str]]:
    expected_tail = PurePosixPath(regime) / "execution.json"
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        selected: list[tarfile.TarInfo] = []
        execution_names: list[PurePosixPath] = []
        for member in members:
            name = _safe_member(member)
            if member.isfile() and name.parts[-2:] == expected_tail.parts:
                execution_names.append(name)
            if member.isfile() and any(
                str(name).endswith(f"/{regime}/{suffix}")
                for suffix in SELECTED_SUFFIXES
            ):
                selected.append(member)
        if len(execution_names) != 1:
            raise ValueError(
                f"expected one {regime} execution record; got {execution_names}"
            )
        output_root_name = execution_names[0].parent
        expected_names = {
            output_root_name / suffix for suffix in SELECTED_SUFFIXES
        }
        selected_names = {PurePosixPath(member.name) for member in selected}
        if selected_names != expected_names:
            missing = sorted(str(name) for name in expected_names - selected_names)
            extra = sorted(str(name) for name in selected_names - expected_names)
            raise ValueError(
                f"selected transfer inventory mismatch; missing={missing}, extra={extra}"
            )
        for member in selected:
            handle.extract(member, destination, filter="data")
    return destination / Path(str(output_root_name)), sorted(
        str(name) for name in selected_names
    )


def file_record(path: Path) -> dict[str, Any]:
    return {
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def tar_member_sha256(archive: Path, basename: str) -> str:
    with tarfile.open(archive, "r:gz") as handle:
        matches = []
        for member in handle.getmembers():
            name = _safe_member(member)
            if member.isfile() and name.name == basename:
                matches.append(member)
        if len(matches) != 1:
            raise ValueError(f"expected one {basename}; got {len(matches)}")
        extracted = handle.extractfile(matches[0])
        if extracted is None:
            raise ValueError(f"could not read {matches[0].name}")
        digest = hashlib.sha256()
        for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def deterministic_tar_gz(source_root: Path, archive: Path) -> None:
    files = sorted(path for path in source_root.rglob("*") if path.is_file())
    if not files:
        raise ValueError("refusing to create an empty recovery archive")
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for path in files:
                    relative = path.relative_to(source_root)
                    info = tar.gettarinfo(str(path), arcname=relative.as_posix())
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)
    temporary.replace(archive)


def rewritten_manifest(
    *, source_manifest: Path, output_root: Path, runtime_bundle: Path
) -> dict[str, Any]:
    manifest = load(source_manifest)
    for key, relative in list(manifest["paths"].items()):
        if key == "output_root":
            manifest["paths"][key] = str(output_root.resolve())
            continue
        manifest["paths"][key] = str(
            (output_root / Path(relative).name).resolve()
        )
    manifest["paths"].update({
        "result_json": str((output_root / "json/result.json").resolve()),
        "current_json": str((output_root / "json/current.json").resolve()),
        "ledger_json": str(
            (output_root / "json/estimator_call_ledger.json").resolve()
        ),
        "normalized_runtime_manifest_json": str(
            (output_root / "normalized_run_manifest.json").resolve()
        ),
    })
    for key in (
        "source_archive",
        "physics_reference_lock",
        "source_revision_manifest",
        "source_archive_manifest",
    ):
        manifest["source_lock"][key] = str(
            (runtime_bundle / Path(manifest["source_lock"][key]).name).resolve()
        )
    return manifest


def worker(manifest_path: Path, runtime_bundle: Path) -> int:
    sys.path.insert(0, str(runtime_bundle))
    run_job_path = runtime_bundle / "run_job.py"
    spec = importlib.util.spec_from_file_location(
        "macro_only_original_run_job", run_job_path
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import source-locked worker: {run_job_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = load(manifest_path)
    validation = module.validate_result_and_compile(manifest)
    module.dump(Path(manifest["paths"]["validation_json"]), validation)
    print(json.dumps({"status": "pass", "validation": validation}, sort_keys=True))
    return 0


def run_postprocessor(
    *, runtime_root: Path, runtime_bundle: Path, manifest_path: Path
) -> None:
    env = os.environ.copy()
    # The CHTC image has NumPy in its system environment; this Mac keeps NumPy
    # in the user site.  Source imports remain pinned to runtime_root below.
    env.pop("PYTHONNOUSERSITE", None)
    env.update({
        "PYTHONPATH": str(runtime_root),
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLCONFIGDIR": str(runtime_root / ".mpl-cache"),
    })
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-manifest",
        str(manifest_path),
        "--runtime-bundle",
        str(runtime_bundle),
    ]
    completed = subprocess.run(
        command,
        cwd=runtime_root,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"source-locked post-run reporting worker exited {completed.returncode}"
        )


def recover_row(proc_id: int, recovery_root: Path) -> dict[str, Any]:
    regime = ROWS[proc_id]
    fetch_root = REPO / FETCH_ROOT_RELATIVE
    original_archive = fetch_root / (
        f"{CLUSTER_ID}.{proc_id}__{regime}_transfer.tar.gz"
    )
    original_bundle = REPO / ORIGINAL_BUNDLE_RELATIVE
    reporting_bundle = REPO / REPORTING_SOURCE_BUNDLE_RELATIVE
    reporting_source_archive = reporting_bundle / "source_locked.tar.gz"
    if not original_archive.is_file():
        raise FileNotFoundError(original_archive)
    if sha256(original_archive) != ORIGINAL_ARCHIVE_SHA256[proc_id]:
        raise ValueError("original transfer archive hash drift")
    if sha256(original_bundle / "source_locked.tar.gz") != ORIGINAL_SOURCE_ARCHIVE_SHA256:
        raise ValueError("original macro source archive hash drift")
    if sha256(reporting_source_archive) != REPORTING_SOURCE_ARCHIVE_SHA256:
        raise ValueError("repaired reporting source archive hash drift")
    reporter = (
        REPO
        / "agent_guidance/skills/paper-i-results/scripts/"
        "compute_paper_i_main_fidelities.py"
    )
    if sha256(reporter) != FIDELITY_REPORTER_SHA256:
        raise ValueError("repaired fidelity reporter hash drift")
    run_job_path = original_bundle / "run_job.py"
    evidence_validation_path = original_bundle / "evidence_validation.py"
    original_job_manifest = original_bundle / "jobs" / f"{regime}.json"
    if sha256(run_job_path) != ORIGINAL_RUN_JOB_SHA256:
        raise ValueError("original post-run control source hash drift")
    if sha256(evidence_validation_path) != ORIGINAL_EVIDENCE_VALIDATION_SHA256:
        raise ValueError("original evidence validator source hash drift")
    if sha256(original_job_manifest) != ORIGINAL_JOB_MANIFEST_SHA256[proc_id]:
        raise ValueError("original job manifest hash drift")

    recovered_archive = recovery_root / (
        f"{CLUSTER_ID}.{proc_id}__{regime}_reporting_recovered_v1.tar.gz"
    )
    validation_receipt = recovery_root / (
        f"{CLUSTER_ID}.{proc_id}__{regime}_local_validation.json"
    )
    archive_receipt = recovery_root / (
        f"{CLUSTER_ID}.{proc_id}__{regime}_archive_receipt.json"
    )
    for path in (recovered_archive, validation_receipt, archive_receipt):
        if path.exists():
            raise FileExistsError(
                f"refusing to overwrite immutable recovery output: {path}"
            )

    with tempfile.TemporaryDirectory(prefix=f"macro_8890778_{proc_id}_") as tmp:
        workspace = Path(tmp)
        transfer_root = workspace / "transfer"
        runtime_root = workspace / "runtime_source"
        transfer_root.mkdir()
        runtime_root.mkdir()
        output_root, selected_inventory = extract_selected_transfer(
            original_archive, transfer_root, regime
        )
        original_execution_path = output_root / "execution.json"
        original_execution = load(original_execution_path)
        original_execution_digest = sha256(original_execution_path)
        if (
            original_execution_digest != ORIGINAL_EXECUTION_SHA256[proc_id]
            or original_execution.get("status") != "failed"
            or int(original_execution.get("exit_code", -1)) != 70
            or original_execution.get("validation_error") != EXPECTED_FAILURE
        ):
            raise ValueError("original execution is not the authorized failure")
        execution_artifacts = original_execution.get("artifacts", {})
        for key in RECOVERED_ARTIFACTS:
            record = execution_artifacts.get(key, {})
            if record.get("exists") is not False or record.get("sha256") is not None:
                raise ValueError(
                    f"original execution unexpectedly contained recovered artifact: {key}"
                )
        original_records = {
            key: file_record(output_root / relative)
            for key, relative in ORIGINAL_ARTIFACTS.items()
        }
        ledger_digest_before = original_records["ledger_json"]["sha256"]

        extract_source_archive(reporting_source_archive, runtime_root)
        runtime_bundle = runtime_root / ORIGINAL_BUNDLE_RELATIVE
        shutil.copytree(original_bundle, runtime_bundle)
        runtime_reporter = (
            runtime_root
            / "agent_guidance/skills/paper-i-results/scripts/"
            "compute_paper_i_main_fidelities.py"
        )
        if sha256(runtime_reporter) != FIDELITY_REPORTER_SHA256:
            raise ValueError("runtime source did not contain repaired reporter")
        source_manifest = runtime_bundle / "jobs" / f"{regime}.json"
        staged_manifest = rewritten_manifest(
            source_manifest=source_manifest,
            output_root=output_root,
            runtime_bundle=runtime_bundle,
        )
        staged_manifest_path = workspace / f"{regime}.postprocess.json"
        dump(staged_manifest_path, staged_manifest)
        run_postprocessor(
            runtime_root=runtime_root,
            runtime_bundle=runtime_bundle,
            manifest_path=staged_manifest_path,
        )

        if sha256(original_execution_path) != original_execution_digest:
            raise ValueError("postprocessor modified original execution.json")
        for key, relative in ORIGINAL_ARTIFACTS.items():
            if file_record(output_root / relative) != original_records[key]:
                raise ValueError(f"postprocessor modified original artifact: {key}")
        ledger_digest_after = sha256(
            output_root / "json/estimator_call_ledger.json"
        )
        if ledger_digest_after != ledger_digest_before:
            raise ValueError("postprocessor changed the estimator ledger")

        recovered_records = {
            key: file_record(output_root / relative)
            for key, relative in RECOVERED_ARTIFACTS.items()
        }
        repaired = load(
            output_root / "terminal_checkpoint.execution_order_repaired.json"
        )
        fidelity = load(output_root / "ground_space_projector_fidelity.json")
        repair_summary = repaired.get("repair", {})
        repaired_checkpoint = repaired.get("repaired_checkpoint", {})
        source_checkpoint = repaired.get("source", {}).get("checkpoint_sha256")
        if (
            repair_summary.get("substantive_term_changes") is not False
            or fidelity.get("state_replay_source")
            != "exact_signed_prefix_checkpoint_permutation_repaired_replay"
        ):
            raise ValueError("postprocessor did not prove permutation-only replay")
        recovery_receipt = {
            "schema": RECOVERY_SCHEMA,
            "status": "pass",
            "classification": "post_run_reporting_only_permutation_repair",
            "regime_slug": regime,
            "source": {
                "cluster_id": CLUSTER_ID,
                "proc_id": proc_id,
                "bundle_id": original_bundle.name,
                "profile_contract_sha256": ROUTE_DIGEST,
                "original_transfer_archive": {
                    "name": original_archive.name,
                    **file_record(original_archive),
                    "untouched": True,
                },
                "original_job_manifest": {
                    "path": str(original_job_manifest.relative_to(REPO)),
                    **file_record(original_job_manifest),
                },
                "original_execution_sha256": original_execution_digest,
                "original_execution_size_bytes": original_execution_path.stat().st_size,
                "original_execution_status": original_execution["status"],
                "original_execution_exit_code": original_execution["exit_code"],
                "original_validation_error": original_execution["validation_error"],
                "selected_original_member_inventory": selected_inventory,
            },
            "reporting_runtime": {
                "source_archive": str(REPORTING_SOURCE_BUNDLE_RELATIVE / "source_locked.tar.gz"),
                "source_archive_sha256": REPORTING_SOURCE_ARCHIVE_SHA256,
                "parent_scientific_source_archive_sha256": ORIGINAL_SOURCE_ARCHIVE_SHA256,
                "fidelity_reporter": (
                    "agent_guidance/skills/paper-i-results/scripts/"
                    "compute_paper_i_main_fidelities.py"
                ),
                "fidelity_reporter_sha256": FIDELITY_REPORTER_SHA256,
                "original_run_job_sha256": ORIGINAL_RUN_JOB_SHA256,
                "original_evidence_validation_sha256": (
                    ORIGINAL_EVIDENCE_VALIDATION_SHA256
                ),
                "fidelity_state_replay_source": fidelity["state_replay_source"],
                "recovery_utility": str(Path(__file__).resolve().relative_to(REPO)),
                "recovery_utility_sha256": sha256(Path(__file__).resolve()),
            },
            "safeguards": {
                "scientific_execution_replayed": False,
                "controller_replayed": False,
                "optimizer_replayed": False,
                "scientific_settings_changed": False,
                "original_execution_json_modified": False,
                "original_archive_untouched": True,
                "estimator_ledger_sha256_before": ledger_digest_before,
                "estimator_ledger_sha256_after": ledger_digest_after,
                "estimator_query_delta": 0,
            },
            "checkpoint_repair": {
                "status": "repaired_permutation_only",
                "source_checkpoint_sha256": source_checkpoint,
                "repaired_checkpoint_sha256": repaired_checkpoint.get(
                    "checkpoint_sha256"
                ),
                "substantive_term_changes": False,
            },
            "original_science_artifacts": original_records,
            "recovered_reporting_artifacts": recovered_records,
        }
        inner_receipt = output_root / "post_run_recovery_receipt.json"
        dump(inner_receipt, recovery_receipt)
        deterministic_tar_gz(transfer_root, recovered_archive)

    validator = original_bundle / "validate_fetched.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(validator),
            str(recovered_archive),
            "--output-json",
            str(validation_receipt),
        ],
        cwd=REPO,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"recovered archive validation failed with exit {completed.returncode}"
        )
    validation = load(validation_receipt)
    if validation.get("status") != "pass":
        raise ValueError("recovered archive validator did not return pass")
    archive_record = {
        "schema": RECOVERY_ARCHIVE_SCHEMA,
        "status": "pass",
        "cluster_id": CLUSTER_ID,
        "proc_id": proc_id,
        "regime_slug": regime,
        "original_transfer_archive": {
            "path": str(original_archive.relative_to(REPO)),
            **file_record(original_archive),
        },
        "recovered_archive": {
            "path": str(recovered_archive.relative_to(REPO)),
            **file_record(recovered_archive),
        },
        "local_validation": {
            "path": str(validation_receipt.relative_to(REPO)),
            **file_record(validation_receipt),
        },
        "inner_post_run_recovery_receipt_sha256": tar_member_sha256(
            recovered_archive, "post_run_recovery_receipt.json"
        ),
    }
    dump(archive_receipt, archive_record)
    return {
        **archive_record,
        "archive_receipt": {
            "path": str(archive_receipt.relative_to(REPO)),
            **file_record(archive_receipt),
        },
        "fidelity": validation.get("post_run_projector_fidelity", {}).get(
            "fidelity"
        ),
        "same_cutoff_abs_error": validation.get(
            "scientific_evidence_validation", {}
        ).get("same_cutoff_abs_error"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc", type=int, action="append", choices=sorted(ROWS))
    parser.add_argument("--worker-manifest", type=Path)
    parser.add_argument("--runtime-bundle", type=Path)
    args = parser.parse_args(argv)
    if args.worker_manifest is not None:
        if args.runtime_bundle is None:
            parser.error("--worker-manifest requires --runtime-bundle")
        return worker(args.worker_manifest, args.runtime_bundle)
    if args.runtime_bundle is not None:
        parser.error("--runtime-bundle is worker-only")
    proc_ids = args.proc or sorted(ROWS)
    recovery_root = REPO / RECOVERY_ROOT_RELATIVE
    recovery_root.mkdir(parents=True, exist_ok=True)
    rows = [recover_row(proc_id, recovery_root) for proc_id in proc_ids]
    aggregate = recovery_root / "recovery_aggregate.json"
    if aggregate.exists():
        raise FileExistsError(f"refusing to overwrite immutable output: {aggregate}")
    dump(aggregate, {
        "schema": AGGREGATE_SCHEMA,
        "status": "pass",
        "cluster_id": CLUSTER_ID,
        "route_digest": ROUTE_DIGEST,
        "row_count": len(rows),
        "rows": rows,
    })
    print(json.dumps({
        "status": "pass",
        "aggregate": str(aggregate),
        "rows": rows,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
