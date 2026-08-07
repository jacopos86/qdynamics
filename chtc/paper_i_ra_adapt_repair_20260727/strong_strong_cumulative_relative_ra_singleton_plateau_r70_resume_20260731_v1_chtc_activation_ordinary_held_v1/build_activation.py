#!/usr/bin/env python3
"""Build the inert one-row ordinary-held CHTC activation overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping


ACTIVATION_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True

PACKAGE_ID = (
    "paper_i_ra_adapt_cumulative_relative_ss_singleton_plateau_"
    "r70_resume_20260731_v1_chtc"
)
ACTIVATION_ID = f"{PACKAGE_ID}_activation_ordinary_held_v1"
PACKAGE_DIRNAME = (
    "strong_strong_cumulative_relative_ra_singleton_plateau_"
    "r70_resume_20260731_v1_chtc"
)
ACTIVATION_DIRNAME = f"{PACKAGE_DIRNAME}_activation_ordinary_held_v1"
CAMPAIGN_ID = (
    "paper_i_ra_adapt_cumulative_relative_ss_singleton_plateau_"
    "r70_resume_v1"
)
EXECUTION_ID = (
    "core__strong_strong_u8__nph7__ra_singleton_plateau__r70"
)
PACKAGE_REL = Path(
    "chtc/paper_i_ra_adapt_repair_20260727"
) / PACKAGE_DIRNAME
ACTIVATION_REL = Path(
    "chtc/paper_i_ra_adapt_repair_20260727"
) / ACTIVATION_DIRNAME
RUNTIME_REL = Path(
    "chtc/paper_i_ra_adapt_repair_20260727"
) / f"{PACKAGE_DIRNAME}_runtime"
BATCH_NAME = (
    "paper-i-ra-adapt-cumulative-relative-ss-singleton-plateau-"
    "r70-resume-20260731-v1-held"
)
IMAGE_REL = Path("chtc/phase3_optuna/image.sif")
IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)
IMAGE_SIZE_BYTES = 216_371_200
SOURCE_PROTOCOL_PATH = (
    "protocols/core__strong_strong_u8__nph7__ra_singleton_plateau.json"
)
SOURCE_PROTOCOL_FILE_SHA256 = (
    "38dfb80bbef62ecc3e148f7bc429fe0ee5ca615b9b1c99e52e6108153cbe7687"
)
SOURCE_PROTOCOL_CANONICAL_SHA256 = (
    "ee0c304698ba4a6532e4271d57db2bd06f091f3e7f7b2e9947c7d01b0e2f2ae0"
)
R50_TERMINAL_SHA256 = (
    "446999c1d184defdcd246387ca2dc74ae311230a35d7a39252f47f3e6d224754"
)
R50_REPAIR_SHA256 = (
    "c4d1fb06a3b08fc1b974d8bd020e1322077a7c5b04efa8df3b6606e70a7c9d22"
)
CONTROL_FILES = (
    "build_activation.py",
    "validate_activation.py",
    "build_attempt_archive.py",
    "execute_authorized_job.sh.in",
    "submit.sub.in",
    "README.md",
)
GENERATED_FILES = (
    "execution_authorization.json",
    "execute_authorized_job.sh",
    "submit.sub",
    "activation_manifest.json",
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ActivationBuildError(ValueError):
    """Raised when the activation cannot be sealed safely."""


def _repo_root() -> Path:
    for parent in ACTIVATION_DIR.parents:
        if (parent / "AGENTS.md").is_file() and (parent / "pipelines").is_dir():
            return parent
    raise ActivationBuildError("Active repository root was not found.")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digested(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    if "sha256" in result:
        raise ActivationBuildError("Digest input already contains sha256.")
    result["sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ActivationBuildError(f"Unsafe {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ActivationBuildError(f"{label} must be a JSON object.")
    return value


def _verify_digest(value: Mapping[str, Any], *, label: str) -> str:
    expected = value.get("sha256")
    body = dict(value)
    body.pop("sha256", None)
    observed = hashlib.sha256(_canonical(body)).hexdigest()
    if (
        not isinstance(expected, str)
        or not _HEX64.fullmatch(expected)
        or observed != expected
    ):
        raise ActivationBuildError(f"{label} self-digest drifted.")
    return expected


def _binding(path: Path, *, relative_to: Path, canonical: bool = False) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ActivationBuildError(f"Unsafe bound file: {path}")
    result: dict[str, Any] = {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "executable": bool(path.stat().st_mode & 0o111),
    }
    if canonical:
        result["canonical_sha256"] = _verify_digest(
            _load_json(path, label=path.name), label=path.name
        )
    return result


def _exclusive_write(path: Path, value: bytes, *, executable: bool = False) -> None:
    if path.exists() or path.is_symlink():
        raise ActivationBuildError(f"Refusing to overwrite: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        if executable:
            temporary.chmod(0o755)
        os.link(temporary, path)
        temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _exclusive_write(path, _canonical(value) + b"\n")


def _render(path: Path, replacements: Mapping[str, str]) -> str:
    text = path.read_text(encoding="utf-8")
    for token, replacement in replacements.items():
        text = text.replace(token, replacement)
    leftovers = sorted(set(re.findall(r"__[A-Z][A-Z0-9_]*__", text)))
    if leftovers:
        raise ActivationBuildError(f"Unrendered template tokens: {leftovers}")
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorized-utc", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", args.authorized_utc):
        raise ActivationBuildError("authorized UTC must use second-resolution RFC3339 Z form.")

    repo_root = _repo_root()
    if ACTIVATION_DIR != repo_root / ACTIVATION_REL:
        raise ActivationBuildError("Activation directory identity drifted.")
    package_dir = repo_root / PACKAGE_REL
    for root, label in (
        (package_dir, "package"),
        (ACTIVATION_DIR, "activation"),
    ):
        if any(
            path.name == "__pycache__" or path.suffix == ".pyc"
            for path in root.rglob("*")
        ):
            raise ActivationBuildError(
                f"Unbound Python bytecode is forbidden in the {label}."
            )
    for name in CONTROL_FILES:
        if not (ACTIVATION_DIR / name).is_file() or (ACTIVATION_DIR / name).is_symlink():
            raise ActivationBuildError(f"Missing activation control: {name}")
    for name in GENERATED_FILES:
        if (ACTIVATION_DIR / name).exists() or (ACTIVATION_DIR / name).is_symlink():
            raise ActivationBuildError(f"Refusing to overwrite: {name}")

    completed = subprocess.run(
        [sys.executable, "-B", str(package_dir / "validate_package.py")],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ActivationBuildError(
            "Sealed package validation failed: " + completed.stderr.strip()
        )

    manifest_path = package_dir / "package_manifest.json"
    job_path = package_dir / "job.json"
    audit_path = package_dir / "source_lock_audit.json"
    plan_path = package_dir / "execution_plan.json"
    manifest = _load_json(manifest_path, label="package manifest")
    job = _load_json(job_path, label="job")
    audit = _load_json(audit_path, label="source-lock audit")
    plan = _load_json(plan_path, label="execution plan")
    for label, value in (
        ("package manifest", manifest),
        ("job", job),
        ("source-lock audit", audit),
        ("execution plan", plan),
    ):
        _verify_digest(value, label=label)
    if (
        manifest.get("status") != "passed_inert_one_row"
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("execution_id") != EXECUTION_ID
        or manifest.get("row_count") != 1
        or manifest.get("execution_authorized") is not False
        or manifest.get("submission_authorized") is not False
        or manifest.get("submitted") is not False
        or job.get("execution_id") != EXECUTION_ID
        or job.get("run_class") != "diagnostic_continuation"
        or job.get("source_horizon") != 50
        or job.get("target_horizon") != 70
        or job.get("only_scientific_change") != "maximum_controller_rounds_50_to_70"
        or job.get("non_swept_settings_diff") != []
        or job.get("active_gradient_policy") != "stationary_source_response_v1"
        or job.get("insertion_policy") != "plateau_commutation"
        or job.get("candidate_representation") != "single_pauli_word_v1"
        or audit.get("r50_terminal_sha256") != R50_TERMINAL_SHA256
        or audit.get("r50_repair_sha256") != R50_REPAIR_SHA256
    ):
        raise ActivationBuildError("Sealed one-row scientific identity drifted.")
    resources = job.get("resources")
    if resources != {
        "request_cpus": 4,
        "request_memory_mb": 90_112,
        "request_disk_mb": 98_304,
        "max_runtime_seconds": 259_200,
    }:
        raise ActivationBuildError("Resource request drifted.")
    source_binding = job.get("source_archive")
    resume_binding = job.get("resume_input", {}).get("archive")
    derived_binding = job.get("derived_protocol")
    protocol_binding = job.get("source_protocol")
    if not all(
        isinstance(row, Mapping)
        for row in (
            source_binding,
            resume_binding,
            derived_binding,
            protocol_binding,
        )
    ):
        raise ActivationBuildError("Job input bindings are malformed.")
    source_path = package_dir / str(source_binding["path"])
    resume_path = package_dir / str(resume_binding["path"])
    protocol_path = package_dir / str(protocol_binding["path"])
    if (
        _sha256_file(source_path) != source_binding.get("sha256")
        or _sha256_file(resume_path) != resume_binding.get("sha256")
        or protocol_binding.get("path") != SOURCE_PROTOCOL_PATH
        or protocol_binding.get("sha256")
        != SOURCE_PROTOCOL_FILE_SHA256
        or protocol_binding.get("canonical_sha256")
        != SOURCE_PROTOCOL_CANONICAL_SHA256
        or _sha256_file(protocol_path) != SOURCE_PROTOCOL_FILE_SHA256
        or manifest.get("source_protocol") != protocol_binding
        or audit.get("source_protocol") != protocol_binding
    ):
        raise ActivationBuildError("Job input binding drifted.")

    controls = [
        _binding(ACTIVATION_DIR / name, relative_to=ACTIVATION_DIR)
        for name in CONTROL_FILES
    ]
    control_plane_sha256 = hashlib.sha256(_canonical(controls)).hexdigest()
    authorization = _digested(
        {
            "schema": "paper_i_ra_adapt_cumulative_relative_r70_execution_authorization_v1",
            "status": "passed",
            "authorization_id": f"{ACTIVATION_ID}__{EXECUTION_ID}",
            "authorized_utc": args.authorized_utc,
            "authorization_source": "explicit_user_request_2026-07-31",
            "authorization_kind": "explicit_user_execution_authority",
            "scope": "single_cell_execution_only",
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "execution_id": EXECUTION_ID,
            "job_spec_sha256": job["sha256"],
            "package_manifest_sha256": manifest["sha256"],
            "execution_plan_sha256": plan["sha256"],
            "source_lock_audit_sha256": audit["sha256"],
            "source_protocol_sha256": (
                SOURCE_PROTOCOL_CANONICAL_SHA256
            ),
            "derived_protocol_sha256": derived_binding["canonical_sha256"],
            "source_archive_sha256": source_binding["sha256"],
            "resume_archive_sha256": resume_binding["sha256"],
            "r50_terminal_sha256": R50_TERMINAL_SHA256,
            "r50_repair_sha256": R50_REPAIR_SHA256,
            "activation_control_plane_sha256": control_plane_sha256,
            "execution_authorized": True,
            "submission_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    authorization_path = ACTIVATION_DIR / "execution_authorization.json"
    _write_json(authorization_path, authorization)
    authorization_file_sha256 = _sha256_file(authorization_path)

    wrapper = _render(
        ACTIVATION_DIR / "execute_authorized_job.sh.in",
        {
            "__JOB_FILE_SHA__": _sha256_file(job_path),
            "__AUTH_FILE_SHA__": authorization_file_sha256,
            "__RESUME_SHA__": str(resume_binding["sha256"]),
            "__SOURCE_SHA__": str(source_binding["sha256"]),
        },
    )
    wrapper_path = ACTIVATION_DIR / "execute_authorized_job.sh"
    _exclusive_write(wrapper_path, wrapper.encode("utf-8"), executable=True)
    submit = _render(
        ACTIVATION_DIR / "submit.sub.in",
        {
            "__ACTIVATION_REL__": ACTIVATION_REL.as_posix(),
            "__PACKAGE_REL__": PACKAGE_REL.as_posix(),
            "__RUNTIME_REL__": RUNTIME_REL.as_posix(),
            "__EXECUTION_ID__": EXECUTION_ID,
            "__BATCH_NAME__": BATCH_NAME,
        },
    )
    submit_path = ACTIVATION_DIR / "submit.sub"
    _exclusive_write(submit_path, submit.encode("utf-8"))

    generated = {
        "execution_authorization": _binding(
            authorization_path, relative_to=ACTIVATION_DIR, canonical=True
        ),
        "worker_wrapper": _binding(wrapper_path, relative_to=ACTIVATION_DIR),
        "submit_descriptor": _binding(submit_path, relative_to=ACTIVATION_DIR),
    }
    activation = _digested(
        {
            "schema": "paper_i_ra_adapt_cumulative_relative_r70_held_activation_v1",
            "status": "authorized_not_submitted_image_reverification_required",
            "activation_id": ACTIVATION_ID,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "execution_id": EXECUTION_ID,
            "authorized_utc": args.authorized_utc,
            "run_class": "diagnostic_continuation",
            "row_count": 1,
            "matrix_scope": {
                "strong_strong": ["ra_singleton_plateau"],
                "strong_weak": [],
            },
            "sealed_package": {
                "path": PACKAGE_REL.as_posix(),
                "manifest": _binding(
                    manifest_path, relative_to=repo_root, canonical=True
                ),
                "job": _binding(job_path, relative_to=repo_root, canonical=True),
                "execution_plan": _binding(
                    plan_path, relative_to=repo_root, canonical=True
                ),
                "source_lock_audit": _binding(
                    audit_path, relative_to=repo_root, canonical=True
                ),
                "source_archive_sha256": source_binding["sha256"],
                "resume_archive_sha256": resume_binding["sha256"],
                "source_protocol": dict(protocol_binding),
            },
            "control_plane": controls,
            "activation_control_plane_sha256": control_plane_sha256,
            "generated": generated,
            "image": {
                "path": IMAGE_REL.as_posix(),
                "sha256": IMAGE_SHA256,
                "size_bytes": IMAGE_SIZE_BYTES,
                "remote_byte_reverification_required_before_release": True,
            },
            "resources": resources,
            "lifecycle": {
                "mode": "ordinary_held_exact_proc_release_v1",
                "initial_hold": True,
                "periodic_release": False,
                "row_count": 1,
                "release_scope": "exact_cluster_proc_after_remote_byte_validation",
                "successful_jobs_retained": False,
                "failed_jobs_retained": True,
            },
            "execution_authorized": True,
            "submission_authorized": True,
            "submission_state": "authorized_not_submitted",
            "submitted": False,
            "paper_evidence_adopted": False,
        }
    )
    activation_path = ACTIVATION_DIR / "activation_manifest.json"
    _write_json(activation_path, activation)
    print(
        _canonical(
            {
                "status": activation["status"],
                "activation_manifest_sha256": activation["sha256"],
                "authorization_sha256": authorization["sha256"],
                "source_archive_sha256": source_binding["sha256"],
                "resume_archive_sha256": resume_binding["sha256"],
                "row_count": 1,
            }
        ).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
