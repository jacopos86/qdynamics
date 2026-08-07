#!/usr/bin/env python3
"""Read-only validation for the one-row ordinary-held activation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
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
PACKAGE_REL = PurePosixPath(
    "chtc/paper_i_ra_adapt_repair_20260727"
) / PACKAGE_DIRNAME
ACTIVATION_REL = PurePosixPath(
    "chtc/paper_i_ra_adapt_repair_20260727"
) / ACTIVATION_DIRNAME
RUNTIME_REL = PurePosixPath(
    "chtc/paper_i_ra_adapt_repair_20260727"
) / f"{PACKAGE_DIRNAME}_runtime"
IMAGE_REL = PurePosixPath("chtc/phase3_optuna/image.sif")
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
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ActivationValidationError(ValueError):
    """Raised when a sealed activation binding drifts."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ActivationValidationError(f"Unsafe {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ActivationValidationError(f"{label} must be a JSON object.")
    return value


def _verify_digest(value: Mapping[str, Any], *, label: str) -> str:
    expected = value.get("sha256")
    body = dict(value)
    body.pop("sha256", None)
    if (
        not isinstance(expected, str)
        or not _HEX64.fullmatch(expected)
        or hashlib.sha256(_canonical(body)).hexdigest() != expected
    ):
        raise ActivationValidationError(f"{label} self-digest drifted.")
    return expected


def _safe_path(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str):
        raise ActivationValidationError(f"{label} must be a path string.")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ActivationValidationError(f"Unsafe {label}: {value}")
    path = root / relative
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ActivationValidationError(f"{label} escaped its root.") from exc
    return path


def _verify_binding(
    root: Path,
    raw: Any,
    *,
    label: str,
    canonical: bool = False,
) -> tuple[Path, dict[str, Any] | None]:
    if not isinstance(raw, Mapping):
        raise ActivationValidationError(f"{label} binding is malformed.")
    path = _safe_path(root, raw.get("path"), label=f"{label} path")
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != raw.get("size_bytes")
        or _sha256_file(path) != raw.get("sha256")
        or bool(path.stat().st_mode & 0o111) != raw.get("executable", False)
    ):
        raise ActivationValidationError(f"{label} byte binding drifted.")
    if not canonical:
        return path, None
    value = _load_json(path, label=label)
    digest = _verify_digest(value, label=label)
    if digest != raw.get("canonical_sha256"):
        raise ActivationValidationError(f"{label} canonical binding drifted.")
    return path, value


def _assignment(text: str, name: str) -> list[str]:
    pattern = re.compile(
        rf"^[ \t]*{re.escape(name)}[ \t]*=[ \t]*(.*?)[ \t]*$",
        re.IGNORECASE | re.MULTILINE,
    )
    return pattern.findall(text)


def validate(repo_root: Path, *, require_image: bool) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    expected_activation = repo_root / Path(ACTIVATION_REL.as_posix())
    package_dir = repo_root / Path(PACKAGE_REL.as_posix())
    if ACTIVATION_DIR.resolve() != expected_activation.resolve():
        raise ActivationValidationError("Activation directory identity drifted.")
    for root, label in (
        (package_dir, "package"),
        (ACTIVATION_DIR, "activation"),
    ):
        if any(
            path.name == "__pycache__" or path.suffix == ".pyc"
            for path in root.rglob("*")
        ):
            raise ActivationValidationError(
                f"Unbound Python bytecode is forbidden in the {label}."
            )

    activation = _load_json(
        ACTIVATION_DIR / "activation_manifest.json",
        label="activation manifest",
    )
    _verify_digest(activation, label="activation manifest")
    if (
        activation.get("schema")
        != "paper_i_ra_adapt_cumulative_relative_r70_held_activation_v1"
        or activation.get("status")
        != "authorized_not_submitted_image_reverification_required"
        or activation.get("activation_id") != ACTIVATION_ID
        or activation.get("package_id") != PACKAGE_ID
        or activation.get("campaign_id") != CAMPAIGN_ID
        or activation.get("execution_id") != EXECUTION_ID
        or activation.get("row_count") != 1
        or activation.get("run_class") != "diagnostic_continuation"
        or activation.get("matrix_scope")
        != {
            "strong_strong": ["ra_singleton_plateau"],
            "strong_weak": [],
        }
        or activation.get("execution_authorized") is not True
        or activation.get("submission_authorized") is not True
        or activation.get("submission_state") != "authorized_not_submitted"
        or activation.get("submitted") is not False
        or activation.get("paper_evidence_adopted") is not False
    ):
        raise ActivationValidationError("Activation authority drifted.")

    controls = activation.get("control_plane")
    if not isinstance(controls, list) or len(controls) != len(CONTROL_FILES):
        raise ActivationValidationError("Control-plane closure drifted.")
    # CHTC login nodes may provide Python 3.9, where ``zip(strict=...)`` is
    # unavailable.  The exact-length check above supplies the same guard.
    for expected_name, raw in zip(CONTROL_FILES, controls):
        path, _ = _verify_binding(
            ACTIVATION_DIR, raw, label=f"control {expected_name}"
        )
        if path.name != expected_name:
            raise ActivationValidationError("Control ordering drifted.")
    if hashlib.sha256(_canonical(controls)).hexdigest() != activation.get(
        "activation_control_plane_sha256"
    ):
        raise ActivationValidationError("Control-plane digest drifted.")

    generated = activation.get("generated")
    if not isinstance(generated, Mapping) or set(generated) != {
        "execution_authorization",
        "worker_wrapper",
        "submit_descriptor",
    }:
        raise ActivationValidationError("Generated-file closure drifted.")
    authorization_path, authorization = _verify_binding(
        ACTIVATION_DIR,
        generated["execution_authorization"],
        label="execution authorization",
        canonical=True,
    )
    wrapper_path, _ = _verify_binding(
        ACTIVATION_DIR,
        generated["worker_wrapper"],
        label="worker wrapper",
    )
    submit_path, _ = _verify_binding(
        ACTIVATION_DIR,
        generated["submit_descriptor"],
        label="submit descriptor",
    )
    assert authorization is not None
    if (
        authorization_path.name != "execution_authorization.json"
        or wrapper_path.name != "execute_authorized_job.sh"
        or not bool(wrapper_path.stat().st_mode & 0o111)
        or submit_path.name != "submit.sub"
    ):
        raise ActivationValidationError("Generated-file identity drifted.")

    sealed = activation.get("sealed_package")
    if not isinstance(sealed, Mapping) or sealed.get("path") != PACKAGE_REL.as_posix():
        raise ActivationValidationError("Sealed-package root drifted.")
    _, package_manifest = _verify_binding(
        repo_root,
        sealed.get("manifest"),
        label="package manifest",
        canonical=True,
    )
    job_path, job = _verify_binding(
        repo_root, sealed.get("job"), label="job", canonical=True
    )
    _, plan = _verify_binding(
        repo_root,
        sealed.get("execution_plan"),
        label="execution plan",
        canonical=True,
    )
    _, audit = _verify_binding(
        repo_root,
        sealed.get("source_lock_audit"),
        label="source-lock audit",
        canonical=True,
    )
    assert package_manifest is not None and job is not None
    assert plan is not None and audit is not None
    if (
        job_path != package_dir / "job.json"
        or package_manifest.get("status") != "passed_inert_one_row"
        or package_manifest.get("package_id") != PACKAGE_ID
        or package_manifest.get("row_count") != 1
        or package_manifest.get("execution_authorized") is not False
        or package_manifest.get("submission_authorized") is not False
        or package_manifest.get("submitted") is not False
        or job.get("execution_id") != EXECUTION_ID
        or job.get("source_horizon") != 50
        or job.get("target_horizon") != 70
        or job.get("active_gradient_policy")
        != "stationary_source_response_v1"
        or job.get("insertion_policy") != "plateau_commutation"
        or job.get("candidate_representation") != "single_pauli_word_v1"
        or job.get("only_scientific_change")
        != "maximum_controller_rounds_50_to_70"
        or job.get("non_swept_settings_diff") != []
        or job.get("resources") != activation.get("resources")
        or audit.get("r50_terminal_sha256") != R50_TERMINAL_SHA256
        or audit.get("r50_repair_sha256") != R50_REPAIR_SHA256
    ):
        raise ActivationValidationError("Sealed-package identity drifted.")

    source = job.get("source_archive")
    resume = job.get("resume_input", {}).get("archive")
    derived = job.get("derived_protocol")
    protocol = job.get("source_protocol")
    if not all(
        isinstance(item, Mapping)
        for item in (source, resume, derived, protocol)
    ):
        raise ActivationValidationError("Job input binding is malformed.")
    source_path = _safe_path(package_dir, source.get("path"), label="source archive")
    resume_path = _safe_path(package_dir, resume.get("path"), label="resume archive")
    protocol_path = _safe_path(
        package_dir, protocol.get("path"), label="source protocol"
    )
    if (
        not source_path.is_file()
        or source_path.is_symlink()
        or source_path.stat().st_size != source.get("size_bytes")
        or _sha256_file(source_path) != source.get("sha256")
        or not resume_path.is_file()
        or resume_path.is_symlink()
        or resume_path.stat().st_size != resume.get("size_bytes")
        or _sha256_file(resume_path) != resume.get("sha256")
        or sealed.get("source_archive_sha256") != source.get("sha256")
        or sealed.get("resume_archive_sha256") != resume.get("sha256")
        or protocol.get("path") != SOURCE_PROTOCOL_PATH
        or protocol.get("sha256") != SOURCE_PROTOCOL_FILE_SHA256
        or protocol.get("canonical_sha256")
        != SOURCE_PROTOCOL_CANONICAL_SHA256
        or not protocol_path.is_file()
        or protocol_path.is_symlink()
        or protocol_path.stat().st_size != protocol.get("size_bytes")
        or _sha256_file(protocol_path) != SOURCE_PROTOCOL_FILE_SHA256
        or package_manifest.get("source_protocol") != protocol
        or audit.get("source_protocol") != protocol
        or sealed.get("source_protocol") != protocol
    ):
        raise ActivationValidationError("Transferred input binding drifted.")

    if (
        authorization.get("schema")
        != "paper_i_ra_adapt_cumulative_relative_r70_execution_authorization_v1"
        or authorization.get("status") != "passed"
        or authorization.get("package_id") != PACKAGE_ID
        or authorization.get("campaign_id") != CAMPAIGN_ID
        or authorization.get("execution_id") != EXECUTION_ID
        or authorization.get("job_spec_sha256") != job.get("sha256")
        or authorization.get("package_manifest_sha256")
        != package_manifest.get("sha256")
        or authorization.get("execution_plan_sha256") != plan.get("sha256")
        or authorization.get("source_lock_audit_sha256") != audit.get("sha256")
        or authorization.get("source_protocol_sha256")
        != SOURCE_PROTOCOL_CANONICAL_SHA256
        or authorization.get("derived_protocol_sha256")
        != derived.get("canonical_sha256")
        or authorization.get("source_archive_sha256") != source.get("sha256")
        or authorization.get("resume_archive_sha256") != resume.get("sha256")
        or authorization.get("r50_terminal_sha256") != R50_TERMINAL_SHA256
        or authorization.get("r50_repair_sha256") != R50_REPAIR_SHA256
        or authorization.get("activation_control_plane_sha256")
        != activation.get("activation_control_plane_sha256")
        or authorization.get("scope") != "single_cell_execution_only"
        or authorization.get("authorization_kind")
        != "explicit_user_execution_authority"
        or authorization.get("execution_authorized") is not True
        or authorization.get("submission_authorized") is not False
        or authorization.get("paper_evidence_adoption_authorized") is not False
    ):
        raise ActivationValidationError("Execution authorization drifted.")

    submit = submit_path.read_text(encoding="utf-8")
    expected_archive = f"transfer/{EXECUTION_ID}__$(ClusterId)__$(ProcId).tar.gz"
    expected_remap = (
        f"{expected_archive}={RUNTIME_REL.as_posix()}/fetched/"
        f"{EXECUTION_ID}__cluster_$(ClusterId)__proc_$(ProcId).tar.gz"
    )
    queue_lines = [
        line.strip()
        for line in submit.splitlines()
        if line.strip().lower().startswith("queue")
    ]
    if (
        _assignment(submit, "universe") != ["vanilla"]
        or _assignment(submit, "executable") != ["/bin/bash"]
        or _assignment(submit, "hold") != ["True"]
        or _assignment(submit, "periodic_release") != ["False"]
        or _assignment(submit, "+HolsteinLifecycleMode")
        != ['"ordinary_held_exact_proc_release_v1"']
        or _assignment(submit, "request_cpus") != ["4"]
        or _assignment(submit, "request_memory") != ["90112MB"]
        or _assignment(submit, "request_disk") != ["98304MB"]
        or _assignment(submit, "+MaxRuntime") != ["259200"]
        or _assignment(submit, "when_to_transfer_output") != ["ON_EXIT"]
        or _assignment(submit, "transfer_output_files") != [expected_archive]
        or _assignment(submit, "transfer_output_remaps") != [f'"{expected_remap}"']
        or queue_lines != ["queue 1"]
        or "max_materialize" in submit.lower()
        or "max_idle" in submit.lower()
        or "strong_weak" in submit.lower()
        or "strong-weak" in submit.lower()
        or submit.count(EXECUTION_ID) < 6
    ):
        raise ActivationValidationError("One-row held submit contract drifted.")
    wrapper = wrapper_path.read_text(encoding="utf-8")
    if (
        f'expected_job_sha="{_sha256_file(job_path)}"' not in wrapper
        or f'expected_authorization_sha="{_sha256_file(authorization_path)}"'
        not in wrapper
        or f'expected_source_sha="{source["sha256"]}"' not in wrapper
        or f'expected_resume_sha="{resume["sha256"]}"' not in wrapper
        or 'expected_output="transfer/${execution_id}__${cluster_id}__${proc_id}.tar.gz"'
        not in wrapper
    ):
        raise ActivationValidationError("Worker wrapper authority drifted.")

    lifecycle = activation.get("lifecycle")
    if lifecycle != {
        "mode": "ordinary_held_exact_proc_release_v1",
        "initial_hold": True,
        "periodic_release": False,
        "row_count": 1,
        "release_scope": "exact_cluster_proc_after_remote_byte_validation",
        "successful_jobs_retained": False,
        "failed_jobs_retained": True,
    }:
        raise ActivationValidationError("Lifecycle receipt drifted.")

    image = activation.get("image")
    if image != {
        "path": IMAGE_REL.as_posix(),
        "sha256": IMAGE_SHA256,
        "size_bytes": IMAGE_SIZE_BYTES,
        "remote_byte_reverification_required_before_release": True,
    }:
        raise ActivationValidationError("Image expectation drifted.")
    image_path = repo_root / Path(IMAGE_REL.as_posix())
    image_verified = False
    if image_path.exists() or image_path.is_symlink():
        if (
            not image_path.is_file()
            or image_path.is_symlink()
            or image_path.stat().st_size != IMAGE_SIZE_BYTES
            or _sha256_file(image_path) != IMAGE_SHA256
        ):
            raise ActivationValidationError("Image byte binding drifted.")
        image_verified = True
    elif require_image:
        raise ActivationValidationError("Bound image is absent.")

    return {
        "status": "passed",
        "activation_manifest_sha256": activation["sha256"],
        "execution_authorization_sha256": authorization["sha256"],
        "execution_id": EXECUTION_ID,
        "row_count": 1,
        "initial_hold": True,
        "image_verified": image_verified,
    }


def _default_repo_root() -> Path:
    for parent in ACTIVATION_DIR.parents:
        if (parent / "AGENTS.md").is_file() and (parent / "pipelines").is_dir():
            return parent
    raise ActivationValidationError("Active repository root was not found.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--require-image", action="store_true")
    args = parser.parse_args()
    try:
        result = validate(
            args.repo_root if args.repo_root is not None else _default_repo_root(),
            require_image=args.require_image,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(_canonical(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
