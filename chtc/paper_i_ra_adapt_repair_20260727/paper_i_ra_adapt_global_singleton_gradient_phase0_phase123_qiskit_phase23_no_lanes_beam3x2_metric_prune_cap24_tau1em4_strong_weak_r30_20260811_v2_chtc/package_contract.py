#!/usr/bin/env python3
"""Closed contract for one Page-12 singleton beam/metric-pruning cell."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


PACKAGE_ID = (
    "paper_i_ra_adapt_global_singleton_gradient_phase0_phase123_qiskit_"
    "phase23_no_lanes_beam3x2_metric_prune_cap24_tau1em4_strong_weak_"
    "r30_20260811_v2_chtc"
)
CAMPAIGN_ID = (
    "paper_i_ra_adapt_global_singleton_gradient_phase0_phase123_qiskit_"
    "phase23_no_lanes_beam3x2_metric_prune_cap24_tau1em4_strong_weak_r30_v2"
)
BUNDLE_ID = (
    "ra_adapt_global_singleton_gradient_phase0_phase123_qiskit_phase23_"
    "no_lanes_beam3x2_metric_prune_cap24_tau1em4_strong_weak_r30_v2"
)
BATCH_NAME = "paper-i-page12-sw-singleton-beam-metric-r30-20260811-v2"
RUN_CLASS = "candidate"
EXECUTION_TARGET = "chtc"
ALGORITHM_ID = (
    "paper_i_ra_adapt_global_singleton_gradient_phase0_phase1_phase2_"
    "phase3_qiskit_phase2_phase3_plateau_no_lanes_v1"
)
SOURCE_ALGORITHM_ID = (
    "paper_i_ra_adapt_singleton_plateau_insertion_repair_v1"
)
ROUTE_ID = (
    "ra_global_singleton_gradient_phase0_phase123_qiskit_phase23_plateau_"
    "beam3x2_metric_prune"
)
INHERITED_SOURCE_LOCK_ROUTE_ID = "ra_global_singleton_plateau_commutation"
SOURCE_ROUTE_PROFILE = (
    "supported_projected_generalized_source_metric_no_overlap_trust_"
    "full_response_symmetric_cost_no_prune_insertion_commutation_plateau_v2"
)
SOURCE_ROUTE_CONTRACT_SHA256 = (
    "aa669d7f0c3621d9ddf7f8595f96333c56b536c8fc79547607e76d8d91d4b6ff"
)
TARGET_PARENT_ROUTE_PROFILE = (
    "paper_i_canonical_sr_snake__admission-singleton__"
    "insertion-plateau_commutation__pruning-metric__beam-fork_local_v1"
)
TARGET_ROUTE_SUFFIX = (
    "global_singleton_abs_gradient_phase0_then_singleton_phase1_then_"
    "qiskit_phase2_phase3_no_lanes_v1"
)
TARGET_ROUTE_PROFILE = (
    "paper_i_ra_adapt__single_pauli_word_v1__"
    "insertion_commutation_plateau_v2__global_guarded_singleton_phase_i__"
    "identity_phase_ii__stationary_source_response_v1__"
    "all_phase_resource_weighting_v1__pruning-metric__beam-fork_local__"
    f"{TARGET_ROUTE_SUFFIX}"
)
TARGET_PARENT_ROUTE_CONTRACT_SHA256 = (
    "14e26e20d3ca4ad5fcc2e1697fa445ed83de4d19fc257b48d056c9e1b4ffaac2"
)
TARGET_ROUTE_CONTRACT_SHA256 = (
    "d545fd25a162a85c4dabf09ae3fccd8ba6e095c9b794fec0fcf9a096f655c0e7"
)
PRUNING_POLICY = "metric"
BEAM_LIVE_BRANCHES = 3
BEAM_CHILDREN_PER_PARENT = 2
BEAM_MAXIMUM_CHILDREN_PER_ROUND = 6
BEAM_S_ALG_WEIGHT = 0.005
BACKEND_COMPILE_SCOPE = (
    "phase_i_proxy_phase_ii_phase_iii_qiskit_transpile_v1"
)
SELECTOR_COMPILE_COST_POLICY = (
    "qiskit_full_trial_ansatz_signed_marginal_phase2_phase3_v1"
)
SELECTOR_COMPILE_COST_PHASE_REUSE = (
    "phase_ii_phase_iii_shared_oracle_snapshot_and_cache_v1"
)
REMOTE_IMAGE_PATH = "chtc/phase3_optuna/image.sif"
REMOTE_IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)
ACTIVE_GRADIENT_POLICY = "stationary_source_response_v1"
RESOURCE_WEIGHTING_SCOPE = "all_phase_resource_weighting_v1"
CANDIDATE_REPRESENTATION = "single_pauli_word_v1"
CANDIDATE_ADAPTER_ID = (
    "paper_i_ra_adapt_global_singleton_gradient_phase0_candidate_adapter_v1"
)
PHASE0_VARIANT = "global_singleton"
PHASE0_POLICY = "global_singleton_absolute_gradient_shortlist_v1"
PHASE0_SHORTLIST_SIZE = 24
EXPECTED_CANDIDATE_FUNNEL = (
    "global_singleton_gradient_phase0_shortlist_then_singleton_phase1_"
    "shortlist_then_singleton_phase2_then_singleton_phase3_v1"
)
EXECUTION_ID_PREFIX = (
    "global_singleton_gradient_phase0_phase23_qiskit_no_lanes_beam3x2_metric"
)
STAGE_ID = (
    "global_singleton_gradient_phase0_phase23_qiskit_no_lanes_beam3x2_"
    "metric_candidate"
)

WEAK_HORIZON = 30
STRONG_HORIZON = 30
REGIME_ROWS: tuple[tuple[str, int, int], ...] = (
    ("strong_weak_u8", 3, WEAK_HORIZON),
)

BASELINE_PACKAGE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_macro_then_singleton_phase123_qiskit_phase23_no_lanes_"
    "tau1em4_r50_20260807_v1_chtc"
)
BASELINE_BUNDLE_ID = (
    "ra_adapt_macro_then_singleton_phase123_qiskit_phase23_no_lanes_"
    "tau1em4_r50_v1"
)
BASELINE_PACKAGE_MANIFEST_FILE_SHA256 = (
    "a649c99703bcc433b96c0e8d6316ce8e1cc37fd0bb811a5864b8eef4c70379af"
)
BASELINE_PACKAGE_MANIFEST_CANONICAL_SHA256 = (
    "260bb30c731a3dfc68f8c8a23e91d55f869cb69f740880118f4093f75b925b0e"
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
    "strong_weak_u8": {
        "request_cpus": 4,
        "request_memory_mb": 65_536,
        "request_disk_mb": 61_440,
        "max_runtime_seconds": 259_200,
        "basis": "page12_singleton_beam_metric_nph3_r30_v1",
    },
}
REQUIRED_PHASE3_QISKIT_SOURCE_PATHS = (
    "pipelines/scaffold/hh_continuation_scoring.py",
    "pipelines/static_adapt/adapt_pipeline.py",
    "pipelines/static_adapt/hh_backend_compile_oracle.py",
    "pipelines/static_adapt/ra_adapt/adapters.py",
    "pipelines/static_adapt/ra_adapt/bundles.py",
    "pipelines/static_adapt/ra_adapt/contracts.py",
    "pipelines/static_adapt/ra_adapt/engine.py",
    "pipelines/static_adapt/ra_adapt/phase0.py",
    "pipelines/static_adapt/sr_snake/_selection.py",
)

PACKAGE_MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_gradient_phase0_phase23_qiskit_package_manifest_v1"
)
JOB_SCHEMA = (
    "paper_i_ra_adapt_gradient_phase0_phase23_qiskit_job_v1"
)
AUTHORIZATION_SCHEMA = (
    "paper_i_ra_adapt_gradient_phase0_phase23_qiskit_"
    "execution_authorization_v1"
)
ACTIVATION_REQUEST_SCHEMA = (
    "paper_i_ra_adapt_gradient_phase0_phase23_qiskit_"
    "activation_request_v1"
)
ACTIVATION_MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_gradient_phase0_phase23_qiskit_"
    "activation_manifest_v1"
)
EXECUTION_PLAN_SCHEMA = (
    "paper_i_ra_adapt_gradient_phase0_phase23_qiskit_"
    "execution_plan_v1"
)
SOURCE_AUTHORITY_SCHEMA = (
    "paper_i_ra_adapt_gradient_phase0_phase23_qiskit_"
    "source_authority_v1"
)
SOURCE_LOCK_AUDIT_SCHEMA = (
    "paper_i_ra_adapt_gradient_phase0_phase23_qiskit_source_lock_audit_v1"
)
SOURCE_ARCHIVE_MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_gradient_phase0_phase23_qiskit_source_archive_"
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
    return (
        f"{regime_id}__nph{int(nph)}__"
        f"{INHERITED_SOURCE_LOCK_ROUTE_ID}"
    )


def execution_id(regime_id: str, nph: int) -> str:
    return (
        f"{EXECUTION_ID_PREFIX}__"
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
