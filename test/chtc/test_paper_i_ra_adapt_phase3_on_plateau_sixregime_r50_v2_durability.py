from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
REPAIR_ROOT = REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727"
V1_PACKAGE_DIR = REPAIR_ROOT / (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_"
    "r50_20260803_v1_chtc"
)
V2_PACKAGE_DIR = REPAIR_ROOT / (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_"
    "r50_20260803_v2_chtc"
)
V2_ACTIVATION_DIR = REPAIR_ROOT / (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_"
    "r50_20260803_v2_chtc_activation_ordinary_v1"
)
PACKAGE_ID = (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_"
    "r50_20260803_v2_chtc"
)
CAMPAIGN_ID = (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_r50_v1"
)
PROTOCOL_SHA256 = "5" * 64
PROTOCOL_FILE_SHA256 = "6" * 64
PACKAGE_MANIFEST_SHA256 = "3" * 64
SOURCE_ARCHIVE_SHA256 = "1" * 64
IMAGE_SHA256 = "2" * 64


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    prior_contract = sys.modules.pop("package_contract", None)
    sys.path.insert(0, path.parent.as_posix())
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
        sys.path.remove(path.parent.as_posix())
        sys.modules.pop("package_contract", None)
        if prior_contract is not None:
            sys.modules["package_contract"] = prior_contract
    return module


def _load_run_cell(package_dir: Path, name: str) -> ModuleType:
    return _load_module(package_dir / "run_cell.py", name)


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digested(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("sha256", None)
    result["sha256"] = hashlib.sha256(
        _canonical_json_bytes(result)
    ).hexdigest()
    return result


def _write_dynamic_checkpoint_payloads(
    root: Path,
    *,
    bad_pointer: bool = False,
) -> tuple[str, str]:
    estimator_payload = {
        "schema": "paper_i_estimator_call_ledger_checkpoint_sidecar_v2",
        "checkpoint": {"depth": 50, "current_round_finalized": True},
        "ledger": {},
        "no_credentials_serialized": True,
    }
    estimator_bytes = _json_bytes(estimator_payload)
    estimator_sha = hashlib.sha256(estimator_bytes).hexdigest()
    estimator_name = (
        "checkpoint.estimator_call_ledger_checkpoint."
        f"{estimator_sha[:16]}.json"
    )
    (root / estimator_name).write_bytes(estimator_bytes)

    resume_payload = {
        "schema": "static_adapt_signed_active_prefix_resume_sidecar_v2",
        "source": {"checkpoint_depth": 50},
        "no_credentials_serialized": True,
    }
    resume_bytes = _json_bytes(resume_payload)
    resume_sha = hashlib.sha256(resume_bytes).hexdigest()
    resume_name = (
        "checkpoint.verified_singleton_resume."
        f"{resume_sha[:16]}.json"
    )
    (root / resume_name).write_bytes(resume_bytes)

    checkpoint = {
        "schema_version": "static_adapt_current_checkpoint_v1",
        "checkpoint": {"depth": 50},
        "adapt_vqe": {
            "estimator_call_ledger_checkpoint": {
                "path": estimator_name,
                "sha256": ("0" * 64 if bad_pointer else estimator_sha),
            },
            "verified_singleton_resume_sidecar": {
                "path": resume_name,
                "sha256": resume_sha,
            },
        },
    }
    (root / "checkpoint.json").write_bytes(_json_bytes(checkpoint))
    return estimator_name, resume_name


def _closed_ledger() -> dict[str, Any]:
    components = {
        "N_H_outer": 50,
        "N_H_refit": 20,
        "N_grad": 30,
        "N_metric": 40,
    }
    return {
        "schema": "paper_i_estimator_call_ledger_sidecar_v2",
        "accounting": {
            "complete": True,
            "exact_blockers": [],
            "components": components,
            "S_alg": sum(components.values()),
        },
        "ledger": {},
        "adapt_success": True,
        "adapt_error": None,
    }


def _write_terminal_science_payloads(
    root: Path,
    *,
    args: argparse.Namespace,
    bad_pointer: bool = False,
    include_execution_manifest: bool = True,
) -> tuple[str, str]:
    root.mkdir(parents=True)
    sidecars = _write_dynamic_checkpoint_payloads(
        root,
        bad_pointer=bad_pointer,
    )
    (root / "estimator_ledger.json").write_bytes(
        _json_bytes(_closed_ledger())
    )
    (root / "paper_i_summary.json").write_bytes(
        _json_bytes({"schema": "paper_i_run_summary_v1"})
    )
    (root / "result.json").write_bytes(
        _json_bytes({"schema": "ra_adapt_result_v1"})
    )
    if include_execution_manifest:
        job = json.loads(args.job.read_text(encoding="utf-8"))
        authorization = json.loads(
            args.authorization.read_text(encoding="utf-8")
        )
        output_payloads = {
            path.name: {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(root.iterdir())
        }
        execution_manifest = _digested({
            "schema": (
                "paper_i_ra_adapt_singleton_phase3_on_plateau_"
                "sixregime_r50_execution_manifest_v2"
            ),
            "status": "passed",
            "package_id": job["package_id"],
            "campaign_id": job["campaign_id"],
            "execution_id": job["execution_id"],
            "job_spec_sha256": job["sha256"],
            "authorization_sha256": authorization["sha256"],
            "protocol_sha256": job["protocol_sha256"],
            "target_horizon": job["target_horizon"],
            "controller_rounds_completed": job["target_horizon"],
            "fresh_start": True,
            "source_checkpoint_consumed": False,
            "worker_owned_live_progress": True,
            "same_filesystem_atomic_success_publication": True,
            "output_payloads": output_payloads,
        })
        (root / "execution_manifest.json").write_bytes(
            _json_bytes(execution_manifest)
        )
    return sidecars


def _write_worker_receipt(
    worker_root: Path,
    *,
    args: argparse.Namespace,
) -> Path:
    artifacts_root = worker_root / "artifacts"
    job = json.loads(args.job.read_text(encoding="utf-8"))
    authorization = json.loads(
        args.authorization.read_text(encoding="utf-8")
    )
    execution_manifest = json.loads(
        (artifacts_root / "execution_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    receipt = _digested(
        {
            "schema": (
                "paper_i_ra_adapt_singleton_phase3_on_plateau_"
                "sixregime_r50_worker_receipt_v2"
            ),
            "status": "passed",
            "package_id": job["package_id"],
            "campaign_id": job["campaign_id"],
            "execution_id": job["execution_id"],
            "job_spec_sha256": job["sha256"],
            "authorization_sha256": authorization["sha256"],
            "execution_manifest_sha256": execution_manifest["sha256"],
            "controller_rounds_completed": job["target_horizon"],
            "fresh_start": True,
            "artifacts": [
                {
                    "path": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted(artifacts_root.iterdir())
            ],
        }
    )
    path = worker_root / "worker_receipt.json"
    path.write_bytes(_json_bytes(receipt))
    return path


def _attempt_args(
    *,
    execution_id: str,
    worker_root: Path,
    worker_exit_status: int,
) -> argparse.Namespace:
    job = Path(f"{execution_id}.json")
    authorization = Path("execution_authorization.json")
    activation_manifest = Path("activation_manifest.json")
    job_payload = _digested(
        {
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "execution_id": execution_id,
            "protocol_sha256": PROTOCOL_SHA256,
            "protocol_file_sha256": PROTOCOL_FILE_SHA256,
            "target_horizon": 50,
        }
    )
    authorization_payload = _digested(
        {
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "execution_id": execution_id,
            "job_spec_sha256": job_payload["sha256"],
            "protocol_sha256": PROTOCOL_SHA256,
            "protocol_file_sha256": PROTOCOL_FILE_SHA256,
            "package_manifest_sha256": PACKAGE_MANIFEST_SHA256,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "remote_image_sha256": IMAGE_SHA256,
        }
    )
    activation_payload = _digested(
        {
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "executions": [
                {
                    "execution_id": execution_id,
                    "job": {"canonical_sha256": job_payload["sha256"]},
                }
            ],
            "execution_authorizations": [
                {
                    "execution_id": execution_id,
                    "canonical_sha256": authorization_payload["sha256"],
                }
            ],
            "sealed_package": {
                "manifest": {
                    "canonical_sha256": PACKAGE_MANIFEST_SHA256,
                },
                "source_archive": {"sha256": SOURCE_ARCHIVE_SHA256},
            },
            "remote_image": {"sha256": IMAGE_SHA256},
        }
    )
    job.write_bytes(_json_bytes(job_payload))
    authorization.write_bytes(_json_bytes(authorization_payload))
    activation_manifest.write_bytes(_json_bytes(activation_payload))
    Path("transfer").mkdir(exist_ok=True)
    return argparse.Namespace(
        worker_root=worker_root,
        job=job,
        authorization=authorization,
        activation_manifest=activation_manifest,
        output_archive=Path("transfer/attempt.tar.gz"),
        execution_id=execution_id,
        cluster_id=123,
        proc_id=4,
        attempt_ordinal=1,
        worker_exit_status=worker_exit_status,
        source_archive_sha256=SOURCE_ARCHIVE_SHA256,
        image_sha256=IMAGE_SHA256,
    )


class _TemporarySource:
    def __init__(self, parent: Path) -> None:
        self._temporary = tempfile.TemporaryDirectory(dir=parent)
        self.name = self._temporary.name
        (Path(self.name) / "source").mkdir()

    def cleanup(self) -> None:
        self._temporary.cleanup()


def _exercise_run_cell(
    module: ModuleType,
    *,
    tmp_path: Path,
    monkeypatch: Any,
    bad_pointer: bool = False,
) -> tuple[Path, Path, tuple[str, str]]:
    source = _TemporarySource(tmp_path)
    protocol = SimpleNamespace(sha256="1" * 64)
    summary = SimpleNamespace(to_dict=lambda: {"schema": "paper_i_run_summary_v1"})
    result = SimpleNamespace(
        protocol=protocol,
        run=SimpleNamespace(paper_i_summary=summary),
        to_dict=lambda: {"schema": "ra_adapt_result_v1"},
    )
    job = {"execution_id": "durability_probe", "sha256": "2" * 64}
    manifest = {"sha256": "3" * 64}
    sidecar_names: list[str] = []

    def fake_prepare(_job_path: Path):
        return job, manifest, protocol, object(), source

    def fake_execute(*, protocol: Any, problem: Any, staging: Path, maximum_rounds: int):
        assert staging.parent == tmp_path / "worker_outputs"
        assert staging.name == "artifacts.in_progress"
        names = _write_dynamic_checkpoint_payloads(
            staging,
            bad_pointer=bad_pointer,
        )
        sidecar_names.extend(names)
        (staging / "estimator_ledger.json").write_bytes(
            _json_bytes(_closed_ledger())
        )
        return result, 50

    monkeypatch.setattr(module, "_prepare", fake_prepare)
    monkeypatch.setattr(
        module,
        "_validate_authorization",
        lambda path, job, manifest: {"sha256": "4" * 64},
    )
    monkeypatch.setattr(module, "_execute", fake_execute)

    worker_root = tmp_path / "worker_outputs"
    worker_root.mkdir()
    output = worker_root / "artifacts"
    receipt = worker_root / "worker_receipt.json"
    module.run_cell(
        job_path=tmp_path / "durability_probe.json",
        authorization_path=tmp_path / "authorization.json",
        output_dir=output,
        receipt_path=receipt,
    )
    return output, receipt, tuple(sidecar_names)


def test_v1_production_finalizer_rejects_legitimate_checkpoint_sidecars(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Production witness: v1 rejects sidecars and deletes its private staging."""

    module = _load_run_cell(V1_PACKAGE_DIR, "phase3_plateau_v1_durability_witness")
    source = _TemporarySource(tmp_path)
    source_root = Path(source.name)
    protocol = SimpleNamespace(sha256="1" * 64)
    result = SimpleNamespace(
        protocol=protocol,
        run=SimpleNamespace(
            paper_i_summary=SimpleNamespace(to_dict=lambda: {"summary": True})
        ),
        to_dict=lambda: {"result": True},
    )

    monkeypatch.setattr(
        module,
        "_prepare",
        lambda _path: (
            {"execution_id": "v1_probe", "sha256": "2" * 64},
            {"sha256": "3" * 64},
            protocol,
            object(),
            source,
        ),
    )
    monkeypatch.setattr(
        module,
        "_validate_authorization",
        lambda path, job, manifest: {"sha256": "4" * 64},
    )

    def fake_execute(*, protocol: Any, problem: Any, staging: Path, maximum_rounds: int):
        _write_dynamic_checkpoint_payloads(staging)
        (staging / "estimator_ledger.json").write_bytes(
            _json_bytes(_closed_ledger())
        )
        return result, 50

    monkeypatch.setattr(module, "_execute", fake_execute)
    worker_root = tmp_path / "worker_outputs"
    worker_root.mkdir()
    with pytest.raises(
        module.PackageContractError,
        match="Successful execution payload closure is incomplete",
    ):
        module.run_cell(
            job_path=tmp_path / "v1_probe.json",
            authorization_path=tmp_path / "authorization.json",
            output_dir=worker_root / "artifacts",
            receipt_path=worker_root / "worker_receipt.json",
        )
    assert not source_root.exists()
    assert list(worker_root.iterdir()) == []


def test_v2_source_archive_is_exact_v1_plus_reviewed_engine_patch() -> None:
    contract = _load_module(
        V2_PACKAGE_DIR / "package_contract.py",
        "phase3_plateau_v2_source_contract",
    )
    before_manifest = json.loads(
        (V1_PACKAGE_DIR / "source/source_archive_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    after_manifest = json.loads(
        (V2_PACKAGE_DIR / "source/source_archive_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    before = {row["path"]: row for row in before_manifest["members"]}
    after = {row["path"]: row for row in after_manifest["members"]}
    expected_patch = {
        path: (before_sha, after_sha)
        for path, before_sha, after_sha in contract.SOURCE_PATCH_BINDINGS
    }

    assert set(expected_patch) == {
        "pipelines/static_adapt/ra_adapt/engine.py"
    }
    assert set(after) == set(before)
    assert {
        path for path in after if after[path] != before[path]
    } == set(expected_patch)
    for path, (before_sha, after_sha) in expected_patch.items():
        assert before[path]["sha256"] == before_sha
        assert after[path]["sha256"] == after_sha
    assert after_manifest["member_count"] == before_manifest["member_count"]
    assert after_manifest["implementation_source_inventory_sha256"] == (
        contract.TARGET_IMPLEMENTATION_INVENTORY_SHA256
    )
    locks = json.loads(
        (V2_PACKAGE_DIR / "source_locks_snapshot.json").read_text(
            encoding="utf-8"
        )
    )
    assert locks["sha256"] == contract.TARGET_SOURCE_LOCKS_CANONICAL_SHA256
    assert locks["implementation_sources"]["sha256"] == (
        contract.TARGET_IMPLEMENTATION_INVENTORY_SHA256
    )


def test_v2_protocols_preserve_v1_science_exactly() -> None:
    contract = _load_module(
        V2_PACKAGE_DIR / "package_contract.py",
        "phase3_plateau_v2_protocol_contract",
    )
    expected_paths = set(contract.SOURCE_TO_TARGET_DIFFERENCE_PATHS)
    for after_path in sorted((V2_PACKAGE_DIR / "protocols").glob("*.json")):
        before = json.loads(
            (V1_PACKAGE_DIR / "protocols" / after_path.name).read_text(
                encoding="utf-8"
            )
        )
        after = json.loads(after_path.read_text(encoding="utf-8"))
        differences = contract.scalar_differences(before, after)
        assert {path for path, _before, _after in differences} == expected_paths
        for key in (
            "problem",
            "parent_inventory",
            "executable_pool",
            "optimizer",
            "optimizer_maxiter",
            "seeds",
            "candidate_representation",
            "active_gradient_policy",
            "resource_weighting_scope",
            "route_contract",
            "request",
        ):
            assert after[key] == before[key]


def test_v2_production_finalizer_accepts_and_binds_dynamic_sidecars(
    tmp_path: Path, monkeypatch: Any
) -> None:
    module = _load_run_cell(V2_PACKAGE_DIR, "phase3_plateau_v2_durability_success")
    output, receipt, sidecars = _exercise_run_cell(
        module,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )

    assert output.is_dir()
    assert receipt.is_file()
    assert not (output.parent / "artifacts.in_progress").exists()
    assert set(path.name for path in output.iterdir()) == {
        "checkpoint.json",
        "estimator_ledger.json",
        "execution_manifest.json",
        "paper_i_summary.json",
        "result.json",
        *sidecars,
    }
    execution_manifest = json.loads(
        (output / "execution_manifest.json").read_text(encoding="utf-8")
    )
    assert set(execution_manifest["output_payloads"]) == {
        "checkpoint.json",
        "estimator_ledger.json",
        "paper_i_summary.json",
        "result.json",
        *sidecars,
    }
    for name in sidecars:
        binding = execution_manifest["output_payloads"][name]
        assert binding["sha256"] == hashlib.sha256(
            (output / name).read_bytes()
        ).hexdigest()


def test_v2_post_run_validation_failure_preserves_in_progress_science(
    tmp_path: Path, monkeypatch: Any
) -> None:
    module = _load_run_cell(V2_PACKAGE_DIR, "phase3_plateau_v2_durability_failure")
    with pytest.raises(
        module.PackageContractError,
        match="(?i)checkpoint sidecar binding",
    ):
        _exercise_run_cell(
            module,
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            bad_pointer=True,
        )

    progress = tmp_path / "worker_outputs/artifacts.in_progress"
    assert (progress / "checkpoint.json").is_file()
    assert (progress / "estimator_ledger.json").is_file()
    assert (progress / "result.json").is_file()
    assert (progress / "paper_i_summary.json").is_file()
    assert not (tmp_path / "worker_outputs/artifacts").exists()
    assert not (tmp_path / "worker_outputs/worker_receipt.json").exists()


def test_v2_success_attempt_archive_requires_both_hash_bound_sidecars(
    tmp_path: Path, monkeypatch: Any
) -> None:
    builder = _load_module(
        V2_ACTIVATION_DIR / "build_attempt_archive.py",
        "phase3_plateau_v2_attempt_success",
    )
    monkeypatch.chdir(tmp_path)
    execution_id = "durability_probe"
    worker_root = Path("worker_outputs")
    args = _attempt_args(
        execution_id=execution_id,
        worker_root=worker_root,
        worker_exit_status=0,
    )
    _write_terminal_science_payloads(
        worker_root / "artifacts",
        args=args,
    )
    (worker_root / "attempt_identity.tsv").write_text(
        f"{execution_id}\t123\t4\t1\n", encoding="utf-8"
    )
    (worker_root / "worker_exit_status.txt").write_text(
        "0\n", encoding="utf-8"
    )
    _write_worker_receipt(worker_root, args=args)

    result = builder.build_archive(args)
    assert result["status"] == "passed"
    with tarfile.open(args.output_archive, "r:gz") as archive:
        receipt = json.load(archive.extractfile("worker_attempt_receipt.json"))
        names = set(archive.getnames())
    assert receipt["science_evidence_state"] == "success_payload_closed_v2"
    assert any("estimator_call_ledger_checkpoint" in name for name in names)
    assert any("verified_singleton_resume" in name for name in names)


def test_v2_success_attempt_archive_rejects_unbound_sidecar(
    tmp_path: Path, monkeypatch: Any
) -> None:
    builder = _load_module(
        V2_ACTIVATION_DIR / "build_attempt_archive.py",
        "phase3_plateau_v2_attempt_bad_pointer",
    )
    monkeypatch.chdir(tmp_path)
    execution_id = "durability_probe"
    worker_root = Path("worker_outputs")
    args = _attempt_args(
        execution_id=execution_id,
        worker_root=worker_root,
        worker_exit_status=0,
    )
    _write_terminal_science_payloads(
        worker_root / "artifacts",
        args=args,
        bad_pointer=True,
    )
    _write_worker_receipt(worker_root, args=args)
    with pytest.raises(
        builder.AttemptArchiveError,
        match="(?i)checkpoint sidecar binding",
    ):
        builder.build_archive(args)


def test_v2_success_attempt_archive_rejects_missing_worker_receipt(
    tmp_path: Path, monkeypatch: Any
) -> None:
    builder = _load_module(
        V2_ACTIVATION_DIR / "build_attempt_archive.py",
        "phase3_plateau_v2_attempt_missing_receipt",
    )
    monkeypatch.chdir(tmp_path)
    execution_id = "durability_probe"
    worker_root = Path("worker_outputs")
    args = _attempt_args(
        execution_id=execution_id,
        worker_root=worker_root,
        worker_exit_status=0,
    )
    _write_terminal_science_payloads(
        worker_root / "artifacts",
        args=args,
    )
    with pytest.raises(
        builder.AttemptArchiveError,
        match="Successful worker artifact closure is incomplete",
    ):
        builder.build_archive(args)


def test_v2_success_attempt_archive_rejects_tampered_worker_receipt(
    tmp_path: Path, monkeypatch: Any
) -> None:
    builder = _load_module(
        V2_ACTIVATION_DIR / "build_attempt_archive.py",
        "phase3_plateau_v2_attempt_tampered_receipt",
    )
    monkeypatch.chdir(tmp_path)
    execution_id = "durability_probe"
    worker_root = Path("worker_outputs")
    args = _attempt_args(
        execution_id=execution_id,
        worker_root=worker_root,
        worker_exit_status=0,
    )
    _write_terminal_science_payloads(
        worker_root / "artifacts",
        args=args,
    )
    receipt_path = _write_worker_receipt(worker_root, args=args)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["artifacts"] = receipt["artifacts"][:-1]
    receipt_path.write_bytes(_json_bytes(_digested(receipt)))
    with pytest.raises(
        builder.AttemptArchiveError,
        match="Successful worker receipt artifact closure drifted",
    ):
        builder.build_archive(args)


def test_v2_failure_attempt_archive_preserves_unvalidated_progress(
    tmp_path: Path, monkeypatch: Any
) -> None:
    builder = _load_module(
        V2_ACTIVATION_DIR / "build_attempt_archive.py",
        "phase3_plateau_v2_attempt_progress",
    )
    monkeypatch.chdir(tmp_path)
    execution_id = "durability_probe"
    worker_root = Path("worker_outputs")
    args = _attempt_args(
        execution_id=execution_id,
        worker_root=worker_root,
        worker_exit_status=2,
    )
    progress = worker_root / "artifacts.in_progress"
    _write_terminal_science_payloads(
        progress,
        args=args,
        bad_pointer=True,
        include_execution_manifest=False,
    )
    (worker_root / "worker_exit_status.txt").write_text(
        "2\n", encoding="utf-8"
    )
    builder.build_archive(args)
    with tarfile.open(args.output_archive, "r:gz") as archive:
        receipt = json.load(archive.extractfile("worker_attempt_receipt.json"))
        names = set(archive.getnames())
    assert receipt["science_evidence_state"] == (
        "in_progress_science_preserved_unvalidated_v2"
    )
    assert "worker_outputs/artifacts.in_progress/checkpoint.json" in names
    assert any("estimator_call_ledger_checkpoint" in name for name in names)
    assert any("verified_singleton_resume" in name for name in names)
