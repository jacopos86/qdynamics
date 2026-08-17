#!/usr/bin/env python3
"""Closed constants and JSON helpers for the local matched singleton-12 suite."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
REPAIR_ROOT = PACKAGE_DIR.parent
REPO_ROOT = PACKAGE_DIR.parents[2]

PACKAGE_ID = "paper_i_page12_matched_singleton12_r50_20260815_v1_local"
CAMPAIGN_ID = "paper_i_page12_matched_singleton12_r50_20260815_v1"
BUNDLE_ID = "paper_i_page12_matched_singleton12_r50_v1"
RUN_CLASS = "candidate"
TARGET_HORIZON = 50

PARENT_PACKAGE_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_global_singleton_gradient_phase0_phase123_qiskit_"
    "phase23_no_lanes_cap24_tau1em4_r50_20260807_v1_chtc"
)
PARENT_PACKAGE = REPO_ROOT / PARENT_PACKAGE_RELATIVE
PARENT_BUNDLE_ID = (
    "ra_adapt_global_singleton_gradient_phase0_phase123_qiskit_phase23_"
    "no_lanes_cap24_tau1em4_r50_v1"
)
PARENT_BUNDLE = PARENT_PACKAGE / "bundle_materialization" / PARENT_BUNDLE_ID
PARENT_PACKAGE_MANIFEST_FILE_SHA256 = (
    "ae96ea800ac108b207e4ccdac148584f1bc5dd6082dd23da893b9f958c1a1896"
)
PARENT_PACKAGE_MANIFEST_CANONICAL_SHA256 = (
    "a0930b878087799aa81b37b5dcaf8a66859aebf871f4e4e02054fa58a82f6731"
)
PARENT_BUNDLE_MANIFEST_FILE_SHA256 = (
    "09ba3ae2dfb71dfb57a114cbaa6bacf92999ce8a793e0091a3793dfa239dd774"
)
PARENT_BUNDLE_MANIFEST_CANONICAL_SHA256 = (
    "6cbf40ed19d96f73151f27854a81690538708cb39f16468d2c9e8e5c8016d2d4"
)
PARENT_SOURCE_LOCKS_FILE_SHA256 = (
    "1dae0adc61161be5e3fbda22c9d0c035f4da2f4a946c011cfaf13ed5d7b2ab99"
)
PARENT_SOURCE_LOCKS_CANONICAL_SHA256 = (
    "e8da64fc347cba75ba733434c2c8cc46142875ad46a77fc91fdc9500c5ab2ae6"
)
PARENT_IMPLEMENTATION_SOURCE_INVENTORY_SHA256 = (
    "deb669c8d9e10eabf8a916586c7727b624d107565ba987a9ae027f7a919ac7b9"
)
PARENT_SOURCE_ARCHIVE_SHA256 = (
    "690d54dbf5bafcaaf974dc11339ed927cb7f5d117265ed51adbb811785740762"
)
PARENT_SOURCE_ARCHIVE_SIZE_BYTES = 2_068_501
PARENT_SOURCE_MANIFEST_FILE_SHA256 = (
    "5aedd26f1578ca56e214ee210cc2e8a3e6eab9b40f3bb9d787359438b692e1f8"
)
PARENT_SOURCE_MANIFEST_CANONICAL_SHA256 = (
    "0470584463090ffa732b9ddbd4dd016781a0cbf1b8c31cec120acdc7afd8cddf"
)

APPEND_RUNTIME_DEPENDENCY = Path(
    "pipelines/exact_bench/generic_static_adapt_variants.py"
)
APPEND_RUNTIME_DEPENDENCY_SHA256 = (
    "1a82945bfcc8e4273c09e2c4f24fb7c1f85df71bb1b952163afe8f349d4262e1"
)
APPEND_RUNTIME_DEPENDENCY_SIZE_BYTES = 490_408

SEALED_CHECKPOINT_SHA256 = (
    "87e032010e009261de415101b717ff38fdb3d9b894b18d1939e6b219d94219f3"
)
OPERATIONAL_CHECKPOINT_OVERLAY = Path(
    "pipelines/static_adapt/current_checkpoint.py"
)
OPERATIONAL_CHECKPOINT_OVERLAY_SHA256 = (
    "b6a0913ae2ee5f3dfd51ab99577980888a77a0cc01fd76bf5fe8437eab801535"
)
SEALED_RESUME_READER = Path("pipelines/static_adapt/sr_snake/_resume.py")
SEALED_RESUME_READER_SHA256 = (
    "173fcbc219453b4a90d604afdfe117718a34318bc621a11ab178a63304e72032"
)
SEALED_RESUME_READER_SIZE_BYTES = 196_544
EXECUTION_SOURCE_POLICY = (
    "sealed_archive_plus_single_authorized_post_extraction_overlay"
)
CHECKPOINT_USAGE = "compact_observation_only"
PARITY_CANARY_SCOPE = "one_round_scientific_and_ledger_equivalence"
STRONG5_PARITY_RECEIPT = REPAIR_ROOT / (
    "paper_i_page12_strong_holstein_sector5_local_repair_20260814_v1_"
    "activation/scientific_parity_canary.json"
)
STRONG5_PARITY_RECEIPT_FILE_SHA256 = (
    "ecd8eec182cc9110f35f6ffb8417d3c9d3c97a4d3b07184046b56396ecc1c6ee"
)
STRONG5_PARITY_RECEIPT_CANONICAL_SHA256 = (
    "ad870ca15fd75b31400986c71245a56283532a5b5714b1c456185ce87ad0ceaa"
)

RA_ALGORITHM_ID = (
    "paper_i_ra_adapt_global_singleton_gradient_phase0_phase1_phase2_"
    "phase3_qiskit_phase2_phase3_plateau_no_lanes_v1"
)
RA_ROUTE_ID = (
    "ra_global_singleton_gradient_phase0_phase123_qiskit_phase23_plateau"
)
RA_ROUTE_CONTRACT_SHA256 = (
    "9811652b332b592bee048a8e5f3048972256abae186921ed7efea52bfd5f3dd8"
)
RA_ADAPTER_ID = (
    "paper_i_ra_adapt_global_singleton_gradient_phase0_candidate_adapter_v1"
)
APPEND_ALGORITHM_ID = "paper_i_append_adapt_v1"
APPEND_ROUTE_ID = "append_singleton"
APPEND_ADAPTER_ID = "paper_i_ra_adapt_single_pauli_word_candidate_adapter_v1"
CANDIDATE_REPRESENTATION = "single_pauli_word_v1"
ACTIVE_GRADIENT_POLICY = "stationary_source_response_v1"
RESOURCE_WEIGHTING_SCOPE = "all_phase_resource_weighting_v1"
APPEND_SELECTOR_ID = "append_adapt_largest_absolute_commutator_gradient_v1"
APPEND_SELECTOR_SCOPE = "conventional_append_no_phase3_no_trust_v1"

REGIME_ROWS: tuple[tuple[str, int], ...] = (
    ("weak_weak", 3),
    ("intermediate_weak", 3),
    ("strong_weak_u8", 3),
    ("weak_strong", 7),
    ("intermediate_strong", 7),
    ("strong_strong_u8", 7),
)
PAIR_EXECUTION_ORDER: tuple[tuple[str, int], ...] = (
    ("strong_strong_u8", 7),
    ("intermediate_strong", 7),
    ("weak_strong", 7),
    ("strong_weak_u8", 3),
    ("intermediate_weak", 3),
    ("weak_weak", 3),
)
METHODS = ("ra_singleton_plateau", "append_singleton")

RA_PROTOCOL_SHA256_BY_REGIME: Mapping[str, str] = {
    "weak_weak": "5b47ffde287d3d82eaaa59720346dead093094c92cab8267449d2ee3f6b304a3",
    "intermediate_weak": "b815cabd3a82272213ec0fe2773a592dcd51608e72949f2b5f3e398e26e59faa",
    "strong_weak_u8": "1c4af4946574d528fedfb6d3ae19210fb4d41f24641447806b56eb278641419a",
    "weak_strong": "90d0a55461335ce897c9eaebb02e8c54b34c3c100d18c3a5c4f66e0d44f79225",
    "intermediate_strong": "6a325b38caa74ffd484d08a804996c317968900739e7d4ca64493cb3531b78d0",
    "strong_strong_u8": "ffe0d3128448c71666f44b7ccd0abe48673e513ce30c703dbf194331d84e8849",
}

PACKAGE_MANIFEST_SCHEMA = "paper_i_page12_matched_singleton12_package_manifest_v1"
BUNDLE_MANIFEST_SCHEMA = "paper_i_page12_matched_singleton12_bundle_manifest_v1"
SOURCE_ARCHIVE_MANIFEST_SCHEMA = (
    "paper_i_page12_matched_singleton12_source_archive_manifest_v1"
)
EXPECTED_ARTIFACTS_SCHEMA = (
    "paper_i_page12_matched_singleton12_expected_artifacts_v1"
)
VALIDATION_REPORT_SCHEMA = "paper_i_page12_matched_singleton12_validation_v1"
EXECUTION_PLAN_SCHEMA = "paper_i_page12_matched_singleton12_execution_plan_v1"
JOB_SCHEMA = "paper_i_page12_matched_singleton12_job_v1"


class PackageContractError(RuntimeError):
    """Raised when immutable package input or output drifts."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def digested(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    unsigned.pop("sha256", None)
    return {**unsigned, "sha256": canonical_sha256(unsigned)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PackageContractError(f"{label} must be a JSON object.")
    return value


def verify_self_digest(value: Mapping[str, Any], *, label: str) -> str:
    supplied = value.get("sha256")
    expected = canonical_sha256(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    if supplied != expected:
        raise PackageContractError(f"{label} canonical digest drifted.")
    return expected


def safe_relative_path(value: Any, *, label: str) -> Path:
    text = str(value or "")
    pure = PurePosixPath(text)
    if (
        not text
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise PackageContractError(f"{label} is not a safe relative path.")
    return Path(*pure.parts)


def binding(path: Path, *, root: Path, canonical: bool = False) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise PackageContractError(f"Binding escapes package root: {path}") from exc
    result: dict[str, Any] = {
        "path": relative.as_posix(),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }
    if canonical:
        payload = load_json(resolved, label=relative.as_posix())
        result["canonical_sha256"] = verify_self_digest(
            payload, label=relative.as_posix()
        )
    return result


def ra_execution_id(regime: str, nph: int) -> str:
    return (
        "global_singleton_gradient_phase0_phase23_qiskit_no_lanes__"
        f"{regime}__nph{nph}__{RA_ROUTE_ID}"
    )


def append_execution_id(regime: str, nph: int) -> str:
    return f"matched_singleton12__{regime}__nph{nph}__append_conventional_unwhitened"


def execution_id(regime: str, nph: int, method: str) -> str:
    if method == "ra_singleton_plateau":
        return ra_execution_id(regime, nph)
    if method == "append_singleton":
        return append_execution_id(regime, nph)
    raise PackageContractError(f"Unknown matched method: {method}")


def source_lock_id(regime: str, nph: int) -> str:
    return f"{regime}__nph{nph}__ra_global_singleton_plateau_commutation"


def expected_execution_ids() -> tuple[str, ...]:
    return tuple(
        execution_id(regime, nph, method)
        for regime, nph in REGIME_ROWS
        for method in METHODS
    )


def expected_run_artifacts(execution: str) -> dict[str, Any]:
    root = f"runs/{execution}"
    return {
        "execution_manifest": {
            "path": f"{root}/execution_manifest.json",
            "required": True,
            "direct_file_required": True,
            "reference_receipt_required": False,
            "fulfillment_kind": "direct_execution_v1",
        },
        "checkpoint": {
            "path": f"{root}/checkpoints/current.json",
            "required": True,
            "direct_file_required": True,
            "reference_receipt_required": False,
            "fulfillment_kind": "direct_execution_v1",
        },
        "estimator_ledger": {
            "path": f"{root}/result/estimator_ledger.json",
            "required": True,
            "direct_file_required": True,
            "reference_receipt_required": False,
            "fulfillment_kind": "direct_execution_v1",
        },
        "result": {
            "path": f"{root}/result/result.json",
            "required": True,
            "direct_file_required": True,
            "reference_receipt_required": False,
            "fulfillment_kind": "direct_execution_v1",
        },
        "summary": {
            "path": f"{root}/summary/summary.json",
            "required": True,
            "direct_file_required": True,
            "reference_receipt_required": False,
            "fulfillment_kind": "direct_execution_v1",
        },
    }


__all__ = [name for name in globals() if name.isupper()] + [
    "PackageContractError",
    "append_execution_id",
    "binding",
    "canonical_json_bytes",
    "canonical_sha256",
    "digested",
    "execution_id",
    "expected_execution_ids",
    "expected_run_artifacts",
    "load_json",
    "ra_execution_id",
    "safe_relative_path",
    "sha256_file",
    "source_lock_id",
    "verify_self_digest",
]
