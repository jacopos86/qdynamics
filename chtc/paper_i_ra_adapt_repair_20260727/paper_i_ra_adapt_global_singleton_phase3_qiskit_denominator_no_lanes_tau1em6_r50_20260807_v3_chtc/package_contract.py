#!/usr/bin/env python3
"""Closed contract for the six-regime Phase-III Qiskit-denominator run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


PACKAGE_ID = (
    "paper_i_ra_adapt_global_singleton_phase3_qiskit_denominator_no_lanes_"
    "tau1em6_r50_20260807_v3_chtc"
)
CAMPAIGN_ID = (
    "paper_i_ra_adapt_global_singleton_phase3_qiskit_denominator_no_lanes_"
    "tau1em6_r50_v3"
)
BUNDLE_ID = (
    "ra_adapt_global_singleton_phase3_qiskit_denominator_no_lanes_"
    "tau1em6_r50_v3"
)
RUN_CLASS = "candidate"
EXECUTION_TARGET = "chtc"
ALGORITHM_ID = (
    "paper_i_ra_adapt_global_singleton_plateau_commutation_"
    "qiskit_phase3_denominator_no_lanes_tau1em6_v1"
)
SOURCE_ALGORITHM_ID = (
    "paper_i_ra_adapt_global_singleton_plateau_commutation_"
    "qiskit_phase3_only_v1"
)
ROUTE_ID = "ra_global_singleton_plateau_commutation"
SOURCE_ROUTE_PROFILE = (
    "paper_i_ra_adapt__single_pauli_word_v1__"
    "insertion_commutation_plateau_v2__global_guarded_singleton_phase_i__"
    "identity_phase_ii__stationary_source_response_v1__"
    "all_phase_resource_weighting_v1__"
    "qiskit_full_ansatz_transpile_cost_phase3_only_v1"
)
SOURCE_ROUTE_CONTRACT_SHA256 = (
    "13fd10645f88ed7c32883ebee31dc2a6cc2c5a8325d6d946afe7cc71614ea839"
)
TARGET_ROUTE_SUFFIX = (
    "qiskit_full_ansatz_positive_marginal_denominator_phase3_only_"
    "no_lanes_tau1em6_v1"
)
TARGET_ROUTE_PROFILE = (
    "paper_i_ra_adapt__single_pauli_word_v1__"
    "insertion_commutation_plateau_v2__global_guarded_singleton_phase_i__"
    "identity_phase_ii__stationary_source_response_v1__"
    "all_phase_resource_weighting_v1__"
    f"{TARGET_ROUTE_SUFFIX}"
)
BACKEND_COMPILE_SCOPE = (
    "phase_i_phase_ii_marrakesh_graph_span_phase_iii_qiskit_transpile_v1"
)
SELECTOR_COMPILE_COST_POLICY = (
    "qiskit_positive_marginal_family_robust_denominator_phase3_only_v1"
)
SELECTOR_COMPILE_COST_PHASE_REUSE = (
    "phase_i_phase_ii_graph_span_then_phase_iii_recompile_population_v1"
)
REMOTE_IMAGE_PATH = "chtc/phase3_optuna/image.sif"
REMOTE_IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)
ACTIVE_GRADIENT_POLICY = "stationary_source_response_v1"
RESOURCE_WEIGHTING_SCOPE = "all_phase_resource_weighting_v1"
CANDIDATE_REPRESENTATION = "single_pauli_word_v1"
CANDIDATE_ADAPTER_ID = (
    "paper_i_ra_adapt_global_single_pauli_word_candidate_adapter_v1"
)

WEAK_HORIZON = 50
STRONG_HORIZON = 50
REGIME_ROWS: tuple[tuple[str, int, int], ...] = (
    ("weak_weak", 3, WEAK_HORIZON),
    ("intermediate_weak", 3, WEAK_HORIZON),
    ("strong_weak_u8", 3, WEAK_HORIZON),
    ("weak_strong", 7, STRONG_HORIZON),
    ("intermediate_strong", 7, STRONG_HORIZON),
    ("strong_strong_u8", 7, STRONG_HORIZON),
)

WEAK_SOURCE_PACKAGE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph3_"
    "r50_20260802_v3_chtc"
)
WEAK_SOURCE_PACKAGE_MANIFEST_FILE_SHA256 = (
    "908b47396ddef396a76270d69bfd0fc493564b106888dc26ce9cca132130aeae"
)
WEAK_SOURCE_PACKAGE_MANIFEST_CANONICAL_SHA256 = (
    "dd756ffa8fa0b1d9b21f906d2587a664ff49743f4eb80c4f1c787c0989cf4f23"
)
STRONG_SOURCE_PACKAGE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_"
    "r70_20260804_v1_resume256gb_loaderfix_v2_chtc"
)
STRONG_SOURCE_PACKAGE_MANIFEST_FILE_SHA256 = (
    "b7f307bbbe3bc2b406094c777e07a57ecaf8e1f3f77a058a53ae36cdf84e000a"
)
STRONG_SOURCE_PACKAGE_MANIFEST_CANONICAL_SHA256 = (
    "6dba640133bbdfffc7e9be14af54b36781d3a3c684856a6c7c5fea80fb20e9bd"
)
PREDECESSOR_SOURCE_LOCKS = WEAK_SOURCE_PACKAGE / "source_locks_snapshot.json"
PREDECESSOR_SOURCE_LOCKS_FILE_SHA256 = (
    "0e5043a15193163a6c5807fd23f451c85a2eeb3b0bf18c885b2721dc8e207384"
)
PREDECESSOR_SOURCE_LOCKS_CANONICAL_SHA256 = (
    "6dea118449d22ed0f11accbbaaaf4a0bed64bf1462bb2b0175495613ab7ff2d9"
)
PROBLEM_BASELINES = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/bundles/materializations/"
    "ra_adapt_stationary_late_core_v13/source_materialization/"
    "problem_baselines.json"
)
PROBLEM_BASELINES_FILE_SHA256 = (
    "a12a36c3f2c8bfe74e4c8a0c9db1d1baecf3b100b00480c5386e903d973c4015"
)

SOURCE_PROTOCOLS: tuple[dict[str, Any], ...] = (
    {
        "regime_id": "weak_weak",
        "nph": 3,
        "horizon": 50,
        "path": WEAK_SOURCE_PACKAGE / "protocols/"
        "historical_mean_global_singleton_v3_nph3_r50__weak_weak__nph3__"
        "ra_global_singleton_plateau.json",
        "file_sha256": (
            "6197d380f32f0e342a8021c4dcb1e6e1c8376c95b9092907c3fb175769443e42"
        ),
        "canonical_sha256": (
            "07c28a38be30f7d9361fe3b7538a774d1edb74a35e1e0c39e93fef4f16543edc"
        ),
    },
    {
        "regime_id": "intermediate_weak",
        "nph": 3,
        "horizon": 50,
        "path": WEAK_SOURCE_PACKAGE / "protocols/"
        "historical_mean_global_singleton_v3_nph3_r50__intermediate_weak__"
        "nph3__ra_global_singleton_plateau.json",
        "file_sha256": (
            "5d6784023860160557ccdf9e8713492c1093f3e3b773b3c52625743d7d2a019e"
        ),
        "canonical_sha256": (
            "353326c43364b4663035b030105468b4659dc38c10289d6377a5fb0a9254f64a"
        ),
    },
    {
        "regime_id": "strong_weak_u8",
        "nph": 3,
        "horizon": 50,
        "path": WEAK_SOURCE_PACKAGE / "protocols/"
        "historical_mean_global_singleton_v3_nph3_r50__strong_weak_u8__"
        "nph3__ra_global_singleton_plateau.json",
        "file_sha256": (
            "52d924fd56be70eeb8d01757f6fafb590c78d64f63141c315e8ba304b9916764"
        ),
        "canonical_sha256": (
            "81217b17f91e0b2f4595ae7c9614a8e94d6764cdebdb1bef90cb95c74169241c"
        ),
    },
    {
        "regime_id": "weak_strong",
        "nph": 7,
        "horizon": 70,
        "path": STRONG_SOURCE_PACKAGE / "protocols/"
        "historical_mean_global_singleton_v2_nph7_r70__weak_strong__nph7__"
        "ra_global_singleton_plateau__resume_from_d49_to_r70_256gb_"
        "loaderfix_v2.json",
        "file_sha256": (
            "e171df35b55294ef19b7438a08321b66709ade3544f90be81278400d6bedd0ff"
        ),
        "canonical_sha256": (
            "287e4a7f1df12fba087fca900fc413fbe5f1ad54b62e53908a13862aeb89e7a4"
        ),
    },
    {
        "regime_id": "intermediate_strong",
        "nph": 7,
        "horizon": 70,
        "path": STRONG_SOURCE_PACKAGE / "protocols/"
        "historical_mean_global_singleton_v2_nph7_r70__intermediate_strong__"
        "nph7__ra_global_singleton_plateau__resume_from_d45_to_r70_256gb_"
        "loaderfix_v2.json",
        "file_sha256": (
            "cdba290a8dfc28792d66ea005535673f4a242ef52e92047b2e06541c56d0c7f9"
        ),
        "canonical_sha256": (
            "a5dffdda32d974f0bcc30007d1b63b64957334b166af2dc2261e8ef11a4aa481"
        ),
    },
    {
        "regime_id": "strong_strong_u8",
        "nph": 7,
        "horizon": 70,
        "path": STRONG_SOURCE_PACKAGE / "protocols/"
        "historical_mean_global_singleton_v2_nph7_r70__strong_strong_u8__"
        "nph7__ra_global_singleton_plateau__resume_from_d31_to_r70_256gb_"
        "loaderfix_v2.json",
        "file_sha256": (
            "132e7f9bdc797760fa87a3b371b8d956f20a94dfedd741b5f064166d536366d1"
        ),
        "canonical_sha256": (
            "cef991acbf1b855173a9b89250cb363c6e6f0113814826b0cb41f9788856d11e"
        ),
    },
)

RESOURCE_ENVELOPES = {
    3: {
        "request_cpus": 4,
        "request_memory_mb": 49_152,
        "request_disk_mb": 61_440,
        "max_runtime_seconds": 259_200,
        "basis": "page7_nph3_r50_global_singleton_envelope_v1",
    },
    7: {
        "request_cpus": 4,
        "request_memory_mb": 262_144,
        "request_disk_mb": 102_400,
        "max_runtime_seconds": 259_200,
        "basis": "page7_nph7_r70_256gib_loaderfix_envelope_v1",
    },
}
REQUIRED_PHASE3_QISKIT_SOURCE_PATHS = (
    "pipelines/scaffold/hh_continuation_scoring.py",
    "pipelines/static_adapt/adapt_pipeline.py",
    "pipelines/static_adapt/hh_backend_compile_oracle.py",
    "pipelines/static_adapt/ra_adapt/bundles.py",
    "pipelines/static_adapt/ra_adapt/engine.py",
)

PACKAGE_MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_global_singleton_phase3_qiskit_mixed_horizon_"
    "package_manifest_v1"
)
JOB_SCHEMA = (
    "paper_i_ra_adapt_global_singleton_phase3_qiskit_mixed_horizon_job_v1"
)
AUTHORIZATION_SCHEMA = (
    "paper_i_ra_adapt_global_singleton_phase3_qiskit_mixed_horizon_"
    "execution_authorization_v1"
)
ACTIVATION_REQUEST_SCHEMA = (
    "paper_i_ra_adapt_global_singleton_phase3_qiskit_mixed_horizon_"
    "activation_request_v1"
)
ACTIVATION_MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_global_singleton_phase3_qiskit_mixed_horizon_"
    "activation_manifest_v1"
)
EXECUTION_PLAN_SCHEMA = (
    "paper_i_ra_adapt_global_singleton_phase3_qiskit_mixed_horizon_"
    "execution_plan_v1"
)
SOURCE_AUTHORITY_SCHEMA = (
    "paper_i_ra_adapt_global_singleton_phase3_qiskit_page7_"
    "source_authority_v1"
)
SOURCE_LOCK_AUDIT_SCHEMA = (
    "paper_i_ra_adapt_global_singleton_phase3_qiskit_source_lock_audit_v1"
)
SOURCE_ARCHIVE_MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_global_singleton_phase3_qiskit_source_archive_"
    "manifest_v1"
)

CONTROL_FILES = (
    "package_contract.py",
    "build_package.py",
    "activate_package.py",
    "run_cell.py",
    "validate_package.py",
    "probe_image_runtime.py",
    "execute_authorized_job.sh",
    "submit.sub.in",
)
GENERATED_PATHS = (
    "source_authority",
    "source_lock_derivation_receipt.json",
    "bundle_materialization",
    "source",
    "jobs",
    "queue.tsv",
    "execution_plan.json",
    "source_lock_audit.json",
    "package_manifest.json",
)


class PackageContractError(RuntimeError):
    """Fail-closed package or worker-contract violation."""


def source_lock_id(regime_id: str, nph: int) -> str:
    return f"{regime_id}__nph{int(nph)}__{ROUTE_ID}"


def execution_id(regime_id: str, nph: int) -> str:
    return (
        "phase3_qiskit_denominator_no_lanes__"
        f"{regime_id}__nph{int(nph)}__{ROUTE_ID}"
    )


def expected_execution_ids() -> tuple[str, ...]:
    return tuple(execution_id(regime, nph) for regime, nph, _ in REGIME_ROWS)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def digested(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload.pop("sha256", None)
    payload["sha256"] = canonical_sha256(payload)
    return payload


def verify_self_digest(value: Mapping[str, Any], *, label: str) -> str:
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    observed = canonical_sha256(unsigned)
    if value.get("sha256") != observed:
        raise PackageContractError(f"{label} self-digest drifted.")
    return observed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackageContractError(f"Cannot load {label}: {path}") from exc
    if not isinstance(value, dict):
        raise PackageContractError(f"{label} must be a JSON object.")
    return value


def safe_relative_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PackageContractError(f"{label} must be a nonempty path.")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise PackageContractError(f"{label} is unsafe: {value!r}.")
    return Path(*pure.parts)


def binding(path: Path, *, root: Path, canonical: bool = False) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise PackageContractError(f"Missing or unsafe binding target: {path}")
    try:
        display = resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise PackageContractError(f"Binding target escaped package: {path}") from exc
    result: dict[str, Any] = {
        "path": display,
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }
    if canonical:
        payload = load_json(resolved, label=display)
        result["canonical_sha256"] = verify_self_digest(payload, label=display)
    return result


def repo_root_from_script(script: str | Path) -> Path:
    current = Path(script).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "AGENTS.md").is_file() and (
            candidate / "pipelines/static_adapt"
        ).is_dir():
            return candidate
    raise PackageContractError("Could not resolve the active repository root.")
