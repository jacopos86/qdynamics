#!/usr/bin/env python3
"""Authenticated archive rotation for the local Paper-I singleton-12 run.

This module is deliberately campaign-local but runner-agnostic.  The caller
supplies the exact authority and cell metadata that must remain bound through
the archive, closure, rotation intent, cleanup receipt, and terminal
validation.  Nothing in this module starts science, decides that a scientific
cell is complete, or discovers files outside the exact paths represented by
``CellArchivePaths``.

The destructive half of the protocol is intentionally narrow:

1. build and fully stream-validate a deterministic gzip/PAX archive;
2. publish an immutable archive manifest and archive-closure receipt;
3. publish an independently authorized rotation intent;
4. atomically rename exactly ``runs/<execution_id>`` to
   ``retiring/<execution_id>``;
5. revalidate the renamed tree against the authenticated archive before
   removing it; and
6. publish an immutable cleanup receipt.

Every receipt is canonical JSON with a self digest and is created with a
same-directory temporary file plus a no-replace hard-link publication.  A
restart may finish any authenticated state after the intent is present, but
it may never infer deletion authority from an archive alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import stat
import tarfile
from typing import Any, BinaryIO, Mapping, Sequence
import zlib


ARCHIVE_SCHEMA = "paper_i_matched_singleton12_cell_archive_v1"
VALIDATION_SCHEMA = "paper_i_matched_singleton12_archive_validation_v1"
CLOSURE_SCHEMA = "paper_i_matched_singleton12_archive_closure_v1"
ROTATION_INTENT_SCHEMA = (
    "paper_i_matched_singleton12_archive_rotation_intent_v1"
)
CLEANUP_SCHEMA = "paper_i_matched_singleton12_archive_cleanup_v1"
ARCHIVE_BACKED_CLOSURE_SCHEMA = (
    "paper_i_matched_singleton12_archive_backed_closure_validation_v1"
)
MANIFEST_MEMBER = "cell_archive_manifest.json"
BLOCK_SIZE = 8 * 1024 * 1024
MAX_JSON_BYTES = 64 * 1024 * 1024
GZIP_COMPRESSLEVEL = 6
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
GIB_BYTES = 1024**3
PAPER_I_MATCHED_SINGLETON12_CAPACITY_FLOOR_GIB = 31
PAPER_I_MATCHED_SINGLETON12_CAPACITY_FLOOR_BYTES = (
    PAPER_I_MATCHED_SINGLETON12_CAPACITY_FLOOR_GIB * GIB_BYTES
)
PAPER_I_MATCHED_SINGLETON12_ARCHIVE_MAX_COMPRESSED_BYTES = 4 * GIB_BYTES
PAPER_I_MATCHED_SINGLETON12_ARCHIVE_POST_WRITE_RESERVE_BYTES = 6 * GIB_BYTES
PAPER_I_MATCHED_SINGLETON12_ARCHIVE_START_FLOOR_BYTES = (
    PAPER_I_MATCHED_SINGLETON12_ARCHIVE_MAX_COMPRESSED_BYTES
    + PAPER_I_MATCHED_SINGLETON12_ARCHIVE_POST_WRITE_RESERVE_BYTES
)
PAPER_I_MATCHED_SINGLETON12_ARCHIVE_MAX_MEMBER_PAYLOAD_BYTES = 32 * GIB_BYTES
PAPER_I_MATCHED_SINGLETON12_ARCHIVE_MAX_TOTAL_PAYLOAD_BYTES = 32 * GIB_BYTES
PAPER_I_MATCHED_SINGLETON12_ARCHIVE_MAX_DECOMPRESSED_BYTES = 33 * GIB_BYTES
PAPER_I_MATCHED_SINGLETON12_REGIME_NPH = {
    "strong_strong_u8": 7,
    "intermediate_strong": 7,
    "weak_strong": 7,
    "strong_weak_u8": 3,
    "intermediate_weak": 3,
    "weak_weak": 3,
}
PAPER_I_MATCHED_SINGLETON12_CAPACITY_EVIDENCE_SOURCE = (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "run_local_page12_insertion_comparators_20260812.py"
)
PAPER_I_MATCHED_SINGLETON12_CAPACITY_EVIDENCE_MAPPING = (
    "PRIOR_RESOURCE_EVIDENCE"
)
PAPER_I_MATCHED_SINGLETON12_WORKING_SAFETY_NUMERATOR = 5
PAPER_I_MATCHED_SINGLETON12_WORKING_SAFETY_DENOMINATOR = 4
PAPER_I_MATCHED_SINGLETON12_REGIME_OBSERVED_WORKING_DISK_KIB = {
    "strong_strong_u8": 17_500_000,
    "intermediate_strong": 12_500_000,
    "weak_strong": 10_000_000,
    "strong_weak_u8": 3_750_000,
    "intermediate_weak": 3_750_000,
    "weak_weak": 3_500_000,
}
PAPER_I_MATCHED_SINGLETON12_REGIME_CAPACITY_EVIDENCE_CLUSTER_PROC = {
    "strong_strong_u8": "9605157.5",
    "intermediate_strong": "9605157.4",
    "weak_strong": "9605157.3",
    "strong_weak_u8": "9605157.2",
    "intermediate_weak": "9605157.1",
    "weak_weak": "9605157.0",
}
# These are the already-published three-decimal display values.  They remain
# a no-lowering compatibility envelope, but no safety calculation converts
# them back into bytes.  Exact floors are derived directly from the integer
# KiB evidence above.
PAPER_I_MATCHED_SINGLETON12_PRIOR_DISPLAY_FLOOR_MILLIGIB = {
    "strong_strong_u8": 30_862,
    "intermediate_strong": 24_901,
    "weak_strong": 21_921,
    "strong_weak_u8": 14_470,
    "intermediate_weak": 14_470,
    "weak_weak": 14_172,
}
PAPER_I_MATCHED_SINGLETON12_MAX_OBSERVED_RAW_KIB = max(
    PAPER_I_MATCHED_SINGLETON12_REGIME_OBSERVED_WORKING_DISK_KIB.values()
)
PAPER_I_MATCHED_SINGLETON12_MAX_OBSERVED_RAW_BYTES = (
    PAPER_I_MATCHED_SINGLETON12_MAX_OBSERVED_RAW_KIB * 1024
)


class Singleton12ArchiveError(ValueError):
    """Raised when archive or rotation state is not exactly authenticated."""


def campaign_capacity_floor() -> dict[str, Any]:
    """Return the fixed evidence-backed launch floor used by this campaign."""

    largest_exact_formula = _exact_regime_formula_floor_bytes(
        "strong_strong_u8"
    )

    return {
        "schema": "paper_i_matched_singleton12_capacity_floor_v1",
        "campaign_minimum_free_bytes": (
            PAPER_I_MATCHED_SINGLETON12_CAPACITY_FLOOR_BYTES
        ),
        "campaign_minimum_free_gib": (
            PAPER_I_MATCHED_SINGLETON12_CAPACITY_FLOOR_GIB
        ),
        "largest_observed_cell_raw_bytes": (
            PAPER_I_MATCHED_SINGLETON12_MAX_OBSERVED_RAW_BYTES
        ),
        "largest_observed_cell_raw_kib": (
            PAPER_I_MATCHED_SINGLETON12_MAX_OBSERVED_RAW_KIB
        ),
        "working_space_safety_factor": {
            "numerator": (
                PAPER_I_MATCHED_SINGLETON12_WORKING_SAFETY_NUMERATOR
            ),
            "denominator": (
                PAPER_I_MATCHED_SINGLETON12_WORKING_SAFETY_DENOMINATOR
            ),
        },
        "archive_start_free_floor_bytes": (
            PAPER_I_MATCHED_SINGLETON12_ARCHIVE_START_FLOOR_BYTES
        ),
        "largest_regime_exact_formula_floor_bytes": largest_exact_formula,
        "capacity_evidence": _regime_capacity_evidence(
            "strong_strong_u8"
        ),
        "comparison": "free_bytes_greater_than_or_equal_to_floor",
    }


def campaign_archive_capacity_contract() -> dict[str, Any]:
    """Return the distinct archive-temp cap, reserve, and start floor."""

    return {
        "schema": "paper_i_matched_singleton12_archive_capacity_v1",
        "max_compressed_archive_bytes": (
            PAPER_I_MATCHED_SINGLETON12_ARCHIVE_MAX_COMPRESSED_BYTES
        ),
        "post_write_free_reserve_bytes": (
            PAPER_I_MATCHED_SINGLETON12_ARCHIVE_POST_WRITE_RESERVE_BYTES
        ),
        "archive_start_free_floor_bytes": (
            PAPER_I_MATCHED_SINGLETON12_ARCHIVE_START_FLOOR_BYTES
        ),
        "max_member_payload_bytes": (
            PAPER_I_MATCHED_SINGLETON12_ARCHIVE_MAX_MEMBER_PAYLOAD_BYTES
        ),
        "max_total_payload_bytes": (
            PAPER_I_MATCHED_SINGLETON12_ARCHIVE_MAX_TOTAL_PAYLOAD_BYTES
        ),
        "max_decompressed_bytes": (
            PAPER_I_MATCHED_SINGLETON12_ARCHIVE_MAX_DECOMPRESSED_BYTES
        ),
        "start_floor_relation": "compressed_cap_plus_post_write_reserve",
    }


def campaign_default_archive_limits() -> ArchiveLimits:
    """Return the bounded archive limits fixed for the matched-12 campaign."""

    return ArchiveLimits(
        max_member_payload_bytes=(
            PAPER_I_MATCHED_SINGLETON12_ARCHIVE_MAX_MEMBER_PAYLOAD_BYTES
        ),
        max_total_payload_bytes=(
            PAPER_I_MATCHED_SINGLETON12_ARCHIVE_MAX_TOTAL_PAYLOAD_BYTES
        ),
        max_decompressed_bytes=(
            PAPER_I_MATCHED_SINGLETON12_ARCHIVE_MAX_DECOMPRESSED_BYTES
        ),
        max_compressed_bytes=(
            PAPER_I_MATCHED_SINGLETON12_ARCHIVE_MAX_COMPRESSED_BYTES
        ),
        min_free_disk_bytes=(
            PAPER_I_MATCHED_SINGLETON12_ARCHIVE_POST_WRITE_RESERVE_BYTES
        ),
    )


def regime_launch_capacity_floor(
    *, regime_id: str, nph: int
) -> dict[str, Any]:
    """Return the descending per-regime floor, including accumulated archives."""

    if regime_id not in PAPER_I_MATCHED_SINGLETON12_REGIME_NPH:
        raise Singleton12ArchiveError(
            f"Unknown matched-singleton12 regime: {regime_id}"
        )
    expected_nph = PAPER_I_MATCHED_SINGLETON12_REGIME_NPH[regime_id]
    if isinstance(nph, bool) or not isinstance(nph, int) or nph != expected_nph:
        raise Singleton12ArchiveError(
            f"Regime/nph relation drifted for {regime_id}: {nph}"
        )
    raw_kib = PAPER_I_MATCHED_SINGLETON12_REGIME_OBSERVED_WORKING_DISK_KIB[
        regime_id
    ]
    raw_bytes = raw_kib * 1024
    exact_formula_bytes = _exact_regime_formula_floor_bytes(regime_id)
    prior_display_milligib = (
        PAPER_I_MATCHED_SINGLETON12_PRIOR_DISPLAY_FLOOR_MILLIGIB[regime_id]
    )
    prior_enforced_bytes = (
        prior_display_milligib * GIB_BYTES + 999
    ) // 1000
    minimum_bytes = max(exact_formula_bytes, prior_enforced_bytes)
    exact_formula_milligib_ceil = (
        exact_formula_bytes * 1000 + GIB_BYTES - 1
    ) // GIB_BYTES
    minimum_milligib = max(
        prior_display_milligib, exact_formula_milligib_ceil
    )
    return {
        "schema": "paper_i_matched_singleton12_regime_capacity_floor_v1",
        "regime_id": regime_id,
        "nph": nph,
        "observed_working_disk_kib": raw_kib,
        "observed_working_disk_bytes": raw_bytes,
        "working_space_safety_factor": {
            "numerator": (
                PAPER_I_MATCHED_SINGLETON12_WORKING_SAFETY_NUMERATOR
            ),
            "denominator": (
                PAPER_I_MATCHED_SINGLETON12_WORKING_SAFETY_DENOMINATOR
            ),
        },
        "archive_start_free_floor_bytes": (
            PAPER_I_MATCHED_SINGLETON12_ARCHIVE_START_FLOOR_BYTES
        ),
        "exact_formula_floor_bytes": exact_formula_bytes,
        "exact_formula_floor_milligib_ceil": (
            exact_formula_milligib_ceil
        ),
        "prior_display_floor_milligib": prior_display_milligib,
        "prior_display_derived_floor_bytes": prior_enforced_bytes,
        "minimum_free_bytes": minimum_bytes,
        "minimum_free_milligib": minimum_milligib,
        "comparison": "free_bytes_greater_than_or_equal_to_floor",
        "capacity_model": (
            "max_of_exact_5_over_4_working_anchor_plus_archive_start_"
            "and_prior_display_floor_no_lowering"
        ),
        "accumulated_archive_accounting": (
            "already_reflected_in_live_free_space_observation"
        ),
        "capacity_evidence": _regime_capacity_evidence(regime_id),
    }


def _regime_capacity_evidence(regime_id: str) -> dict[str, Any]:
    return {
        "source_path": PAPER_I_MATCHED_SINGLETON12_CAPACITY_EVIDENCE_SOURCE,
        "source_mapping": PAPER_I_MATCHED_SINGLETON12_CAPACITY_EVIDENCE_MAPPING,
        "cluster_proc": (
            PAPER_I_MATCHED_SINGLETON12_REGIME_CAPACITY_EVIDENCE_CLUSTER_PROC[
                regime_id
            ]
        ),
    }


def _exact_regime_formula_floor_bytes(regime_id: str) -> int:
    raw_bytes = (
        PAPER_I_MATCHED_SINGLETON12_REGIME_OBSERVED_WORKING_DISK_KIB[
            regime_id
        ]
        * 1024
    )
    numerator = PAPER_I_MATCHED_SINGLETON12_WORKING_SAFETY_NUMERATOR
    denominator = PAPER_I_MATCHED_SINGLETON12_WORKING_SAFETY_DENOMINATOR
    guarded_working_bytes = (
        numerator * raw_bytes + denominator - 1
    ) // denominator
    return (
        guarded_working_bytes
        + PAPER_I_MATCHED_SINGLETON12_ARCHIVE_START_FLOOR_BYTES
    )


def require_regime_launch_capacity(
    path: Path, *, regime_id: str, nph: int
) -> dict[str, Any]:
    """Fail closed unless the next cell meets its exact regime/nph floor."""

    candidate = Path(path)
    if not candidate.exists() or candidate.is_symlink():
        raise Singleton12ArchiveError(
            f"Capacity-probe path is absent or a symlink: {candidate}"
        )
    floor = regime_launch_capacity_floor(regime_id=regime_id, nph=nph)
    free = int(shutil.disk_usage(candidate).free)
    required = int(floor["minimum_free_bytes"])
    if free < required:
        raise Singleton12ArchiveError(
            f"Free space is below the {regime_id}/nph{nph} launch floor."
        )
    return {
        **floor,
        "status": "passed_regime_launch_capacity_floor",
        "observed_free_bytes": free,
        "headroom_bytes": free - required,
    }


def require_campaign_capacity(path: Path) -> dict[str, Any]:
    """Check the one-time initial campaign gate, not every later cell."""

    candidate = Path(path)
    if not candidate.exists() or candidate.is_symlink():
        raise Singleton12ArchiveError(
            f"Capacity-probe path is absent or a symlink: {candidate}"
        )
    observation = shutil.disk_usage(candidate)
    floor = campaign_capacity_floor()
    free = int(observation.free)
    required = int(floor["campaign_minimum_free_bytes"])
    if free < required:
        raise Singleton12ArchiveError(
            "Filesystem free space is below the fixed 31-GiB campaign floor."
        )
    return {
        **floor,
        "status": "passed_campaign_capacity_floor",
        "observed_free_bytes": free,
        "headroom_bytes": free - required,
    }


def canonical_json_bytes(payload: Any) -> bytes:
    """Return the repository's compact, deterministic JSON representation."""

    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Singleton12ArchiveError(
            "Payload is not finite canonical JSON."
        ) from exc


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def digested(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    if "sha256" in result:
        raise Singleton12ArchiveError(
            "Self-digest input already contains sha256."
        )
    result["sha256"] = canonical_sha256(result)
    return result


def verify_self_digest(payload: Mapping[str, Any], *, label: str) -> None:
    observed = payload.get("sha256")
    unsigned = dict(payload)
    unsigned.pop("sha256", None)
    if not isinstance(observed, str) or observed != canonical_sha256(unsigned):
        raise Singleton12ArchiveError(f"{label} self digest drifted.")


def _require_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Singleton12ArchiveError(f"{label} is not a valid integer.")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise Singleton12ArchiveError(
            f"{label} is not a lowercase SHA-256 digest."
        )
    return value


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Singleton12ArchiveError(
                f"JSON object contains duplicate key: {key}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise Singleton12ArchiveError(f"JSON contains non-finite value: {value}")


def _load_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(
            payload,
            object_pairs_hook=_json_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Singleton12ArchiveError(f"Malformed {label} JSON.") from exc
    if not isinstance(parsed, dict):
        raise Singleton12ArchiveError(f"{label} is not a JSON object.")
    return parsed


def _normalized_mapping(
    payload: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise Singleton12ArchiveError(f"{label} is not a mapping.")
    normalized = _load_json_bytes(
        canonical_json_bytes(dict(payload)), label=label
    )
    return normalized


def _safe_component(value: str, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise Singleton12ArchiveError(f"Unsafe {label}: {value!r}")
    return value


def _safe_member_name(value: str, *, label: str = "archive member") -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise Singleton12ArchiveError(f"Unsafe {label}: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or "." in path.parts
        or ".." in path.parts
        or any(not part for part in path.parts)
        or path.as_posix() != value
    ):
        raise Singleton12ArchiveError(f"Unsafe {label}: {value!r}")
    return value


def _safe_relative(value: str, *, label: str) -> str:
    return _safe_member_name(value, label=label)


def _utc_timestamp(value: str | None, *, label: str) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ).replace("+00:00", "Z")
    if not isinstance(value, str) or not value.endswith("Z"):
        raise Singleton12ArchiveError(f"{label} must be RFC-3339 UTC with Z.")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise Singleton12ArchiveError(f"{label} is not RFC-3339.") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise Singleton12ArchiveError(f"{label} is not UTC.")
    return value


@dataclass(frozen=True)
class ArchiveLimits:
    """Hard archive and local-reserve limits supplied by the runner."""

    max_member_payload_bytes: int
    max_total_payload_bytes: int
    max_decompressed_bytes: int
    max_compressed_bytes: int
    min_free_disk_bytes: int = 0

    def __post_init__(self) -> None:
        member = _require_int(
            self.max_member_payload_bytes,
            label="maximum member payload bytes",
            minimum=1,
        )
        total = _require_int(
            self.max_total_payload_bytes,
            label="maximum total payload bytes",
            minimum=1,
        )
        decompressed = _require_int(
            self.max_decompressed_bytes,
            label="maximum decompressed bytes",
            minimum=1,
        )
        _require_int(
            self.max_compressed_bytes,
            label="maximum compressed bytes",
            minimum=1,
        )
        _require_int(
            self.min_free_disk_bytes,
            label="minimum free disk bytes",
        )
        if member > total:
            raise Singleton12ArchiveError(
                "Per-member limit exceeds total-payload limit."
            )
        if total > decompressed:
            raise Singleton12ArchiveError(
                "Total-payload limit exceeds decompressed-byte limit."
            )

    def as_dict(self) -> dict[str, int]:
        return {
            "max_compressed_bytes": self.max_compressed_bytes,
            "max_decompressed_bytes": self.max_decompressed_bytes,
            "max_member_payload_bytes": self.max_member_payload_bytes,
            "max_total_payload_bytes": self.max_total_payload_bytes,
            "min_free_disk_bytes": self.min_free_disk_bytes,
        }

    @property
    def archive_start_free_floor_bytes(self) -> int:
        """Free bytes needed to write the full cap and retain the reserve."""

        return self.max_compressed_bytes + self.min_free_disk_bytes


@dataclass(frozen=True)
class CellArchivePaths:
    """The only filesystem locations one cell's rotation may affect."""

    runtime_root: Path
    execution_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime_root", Path(self.runtime_root))
        _safe_component(self.execution_id, label="execution ID")

    @property
    def source_root(self) -> Path:
        return self.runtime_root / "runs" / self.execution_id

    @property
    def retiring_root(self) -> Path:
        return self.runtime_root / "retiring" / self.execution_id

    @property
    def archive_path(self) -> Path:
        return self.runtime_root / "archives" / f"{self.execution_id}.tar.gz"

    @property
    def archive_manifest_path(self) -> Path:
        return (
            self.runtime_root
            / "archive_manifests"
            / f"{self.execution_id}.json"
        )

    @property
    def archive_closure_path(self) -> Path:
        return (
            self.runtime_root
            / "archive_closure_receipts"
            / f"{self.execution_id}.json"
        )

    @property
    def rotation_intent_path(self) -> Path:
        return (
            self.runtime_root
            / "rotation_intents"
            / f"{self.execution_id}.json"
        )

    @property
    def cleanup_receipt_path(self) -> Path:
        return (
            self.runtime_root
            / "rotation_cleanup_receipts"
            / f"{self.execution_id}.json"
        )


@dataclass(frozen=True)
class _SourceFile:
    path: Path
    relative_path: str
    archive_path: str
    sha256: str
    size_bytes: int
    mode: int

    def tree_row(self) -> dict[str, Any]:
        return {
            "archive_path": self.archive_path,
            "mode": self.mode,
            "path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    def payload_row(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "path": self.archive_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class _ArchiveScan:
    manifest: dict[str, Any]
    manifest_bytes: bytes
    validation: dict[str, Any]


def _require_runtime(paths: CellArchivePaths) -> Path:
    root = paths.runtime_root.absolute()
    try:
        observed = root.lstat()
    except FileNotFoundError as exc:
        raise Singleton12ArchiveError(
            f"Runtime root is absent: {root}"
        ) from exc
    if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
        raise Singleton12ArchiveError(
            f"Runtime root is not a plain directory: {root}"
        )
    return root


def _ensure_operational_directory(path: Path, *, runtime_root: Path) -> None:
    if path.parent != runtime_root:
        raise Singleton12ArchiveError(
            f"Operational directory escaped runtime root: {path}"
        )
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    observed = path.lstat()
    if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
        raise Singleton12ArchiveError(
            f"Operational path is not a plain directory: {path}"
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_plain_file(path: Path) -> tuple[BinaryIO, os.stat_result]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Singleton12ArchiveError(
            f"Unable to open plain file safely: {path}"
        ) from exc
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode):
            raise Singleton12ArchiveError(f"Not a regular file: {path}")
        return os.fdopen(descriptor, "rb"), observed
    except Exception:
        os.close(descriptor)
        raise


def _sha256_plain_file(path: Path) -> tuple[str, int, int]:
    stream, before = _open_plain_file(path)
    digest = hashlib.sha256()
    try:
        while block := stream.read(BLOCK_SIZE):
            digest.update(block)
        after = os.fstat(stream.fileno())
    finally:
        stream.close()
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, key) != getattr(after, key) for key in stable_fields):
        raise Singleton12ArchiveError(f"File changed while hashing: {path}")
    mode = 0o755 if before.st_mode & 0o111 else 0o644
    return digest.hexdigest(), before.st_size, mode


def sha256_file(path: Path) -> str:
    digest, _size, _mode = _sha256_plain_file(path)
    return digest


def _scan_source_tree(
    root: Path, *, source_member_prefix: str, require_files: bool = True
) -> tuple[list[dict[str, Any]], list[_SourceFile]]:
    prefix = _safe_member_name(
        source_member_prefix, label="source member prefix"
    )
    try:
        root_stat = root.lstat()
    except FileNotFoundError as exc:
        raise Singleton12ArchiveError(f"Source tree is absent: {root}") from exc
    if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
        raise Singleton12ArchiveError(
            f"Source tree is not a plain directory: {root}"
        )

    directories: list[dict[str, Any]] = []
    files: list[_SourceFile] = []

    def visit(directory: Path, relative_parts: tuple[str, ...]) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda row: row.name)
        except OSError as exc:
            raise Singleton12ArchiveError(
                f"Unable to scan source directory: {directory}"
            ) from exc
        for entry in entries:
            component = _safe_component(entry.name, label="source component")
            child_parts = (*relative_parts, component)
            relative = PurePosixPath(*child_parts).as_posix()
            _safe_relative(relative, label="source relative path")
            try:
                observed = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise Singleton12ArchiveError(
                    f"Unable to inspect source member: {entry.path}"
                ) from exc
            if stat.S_ISLNK(observed.st_mode):
                raise Singleton12ArchiveError(
                    f"Source-tree symlink is forbidden: {entry.path}"
                )
            path = Path(entry.path)
            if stat.S_ISDIR(observed.st_mode):
                directories.append({"mode": 0o755, "path": relative})
                visit(path, child_parts)
            elif stat.S_ISREG(observed.st_mode):
                digest, size, mode = _sha256_plain_file(path)
                archive_path = f"{prefix}/{relative}"
                _safe_member_name(archive_path)
                files.append(
                    _SourceFile(
                        path=path,
                        relative_path=relative,
                        archive_path=archive_path,
                        sha256=digest,
                        size_bytes=size,
                        mode=mode,
                    )
                )
            else:
                raise Singleton12ArchiveError(
                    f"Source-tree special file is forbidden: {entry.path}"
                )

    visit(root, ())
    directories.sort(key=lambda row: str(row["path"]))
    files.sort(key=lambda row: row.relative_path)
    if require_files and not files:
        raise Singleton12ArchiveError("Source tree contains no regular files.")
    return directories, files


def _tree_payload(
    directories: Sequence[Mapping[str, Any]],
    files: Sequence[_SourceFile],
) -> dict[str, Any]:
    result = {
        "directories": [dict(row) for row in directories],
        "files": [row.tree_row() for row in files],
    }
    result["sha256"] = canonical_sha256(result)
    return result


def _tree_summary(tree: Mapping[str, Any]) -> dict[str, Any]:
    directories = tree.get("directories")
    files = tree.get("files")
    if not isinstance(directories, list) or not isinstance(files, list):
        raise Singleton12ArchiveError("Source-tree inventory is malformed.")
    total = 0
    for row in files:
        if not isinstance(row, Mapping):
            raise Singleton12ArchiveError("Source-tree file row is malformed.")
        total += _require_int(
            row.get("size_bytes"), label="source-tree file size"
        )
    return {
        "directory_count": len(directories),
        "file_count": len(files),
        "sha256": _require_sha256(
            tree.get("sha256"), label="source-tree inventory digest"
        ),
        "total_file_bytes": total,
    }


def _external_rows(
    external_members: Mapping[str, Path],
    *,
    source_member_prefix: str,
) -> tuple[list[dict[str, Any]], list[tuple[str, Path]]]:
    if not isinstance(external_members, Mapping):
        raise Singleton12ArchiveError("External members are not a mapping.")
    rows: list[dict[str, Any]] = []
    sources: list[tuple[str, Path]] = []
    for raw_name, raw_path in external_members.items():
        name = _safe_member_name(raw_name, label="external member")
        if (
            name == MANIFEST_MEMBER
            or name == source_member_prefix
            or name.startswith(f"{source_member_prefix}/")
        ):
            raise Singleton12ArchiveError(
                f"External archive member collides with reserved scope: {name}"
            )
        path = Path(raw_path)
        digest, size, mode = _sha256_plain_file(path)
        rows.append(
            {
                "mode": mode,
                "path": name,
                "sha256": digest,
                "size_bytes": size,
            }
        )
        sources.append((name, path))
    rows.sort(key=lambda row: str(row["path"]))
    sources.sort(key=lambda row: row[0])
    return rows, sources


def _manifest_payload(
    *,
    paths: CellArchivePaths,
    source_member_prefix: str,
    directories: Sequence[Mapping[str, Any]],
    source_files: Sequence[_SourceFile],
    external_rows: Sequence[Mapping[str, Any]],
    authority_metadata: Mapping[str, Any],
    cell_metadata: Mapping[str, Any],
    limits: ArchiveLimits,
) -> dict[str, Any]:
    source_tree = _tree_payload(directories, source_files)
    payload_rows = [row.payload_row() for row in source_files]
    payload_rows.extend(dict(row) for row in external_rows)
    payload_rows.sort(key=lambda row: str(row["path"]))
    if len({str(row["path"]) for row in payload_rows}) != len(payload_rows):
        raise Singleton12ArchiveError("Archive payload paths are not unique.")
    total_payload = sum(int(row["size_bytes"]) for row in payload_rows)
    if any(
        int(row["size_bytes"]) > limits.max_member_payload_bytes
        for row in payload_rows
    ):
        raise Singleton12ArchiveError(
            "Archive member exceeds the caller-supplied payload limit."
        )
    if total_payload > limits.max_total_payload_bytes:
        raise Singleton12ArchiveError(
            "Archive payload exceeds the caller-supplied total limit."
        )
    payload_inventory = {"files": payload_rows}
    return digested(
        {
            "schema": ARCHIVE_SCHEMA,
            "status": "passed_deterministic_authenticated_archive_manifest",
            "execution_id": paths.execution_id,
            "source_member_prefix": source_member_prefix,
            "authority_metadata": dict(authority_metadata),
            "cell_metadata": dict(cell_metadata),
            "compression": {
                "format": "gzip_pax_tar",
                "gzip_compresslevel": GZIP_COMPRESSLEVEL,
                "gzip_filename": "",
                "gzip_mtime": 0,
                "member_gid": 0,
                "member_gname": "",
                "member_mtime": 0,
                "member_uid": 0,
                "member_uname": "",
                "regular_members_only": True,
            },
            "limits": limits.as_dict(),
            "source_tree": source_tree,
            "external_members": [dict(row) for row in external_rows],
            "payload_files": payload_rows,
            "payload_inventory_sha256": canonical_sha256(payload_inventory),
            "total_payload_bytes": total_payload,
        }
    )


def _tar_info(*, name: str, size: int, mode: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = tarfile.REGTYPE
    info.size = size
    info.mode = mode
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.linkname = ""
    info.pax_headers = {}
    return info


class _BoundedCompressedWriter:
    """Enforce compressed size and free-space reserve while gzip writes."""

    def __init__(
        self,
        stream: BinaryIO,
        *,
        parent: Path,
        max_bytes: int,
        min_free_bytes: int,
    ) -> None:
        self._stream = stream
        self._parent = parent
        self._max_bytes = max_bytes
        self._min_free_bytes = min_free_bytes
        self.bytes_written = 0

    def write(self, payload: bytes) -> int:
        size = len(payload)
        if size > self._max_bytes - self.bytes_written:
            raise Singleton12ArchiveError(
                "Compressed archive exceeds its fixed byte budget."
            )
        if (
            self._min_free_bytes
            and shutil.disk_usage(self._parent).free - size
            < self._min_free_bytes
        ):
            raise Singleton12ArchiveError(
                "Archive write would cross the fixed free-space reserve."
            )
        written = self._stream.write(payload)
        if written != size:
            raise Singleton12ArchiveError("Short write while creating archive.")
        self.bytes_written += written
        return written

    def flush(self) -> None:
        self._stream.flush()

    def tell(self) -> int:
        return self.bytes_written


def _write_bound_file(
    archive: tarfile.TarFile,
    *,
    name: str,
    path: Path,
    expected_sha256: str,
    expected_size: int,
    mode: int,
) -> None:
    stream, observed = _open_plain_file(path)
    try:
        if observed.st_size != expected_size:
            raise Singleton12ArchiveError(
                f"Archive source size drifted before write: {path}"
            )
        archive.addfile(
            _tar_info(name=name, size=expected_size, mode=mode), stream
        )
    finally:
        stream.close()
    digest, size, observed_mode = _sha256_plain_file(path)
    if (
        digest != expected_sha256
        or size != expected_size
        or observed_mode != mode
    ):
        raise Singleton12ArchiveError(
            f"Archive source drifted while being written: {path}"
        )


def _temporary_archive_path(paths: CellArchivePaths) -> Path:
    token = secrets.token_hex(16)
    return paths.archive_path.with_name(
        f".{paths.archive_path.name}.tmp.{token}"
    )


def _temporary_json_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp.{secrets.token_hex(16)}")


def _publish_bytes_atomic_noreplace(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(errno.EEXIST, "Destination already exists", path)
    temporary = _temporary_json_path(path)
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        _fsync_directory(path.parent)


def write_json_atomic_noreplace(
    path: Path, payload: Mapping[str, Any]
) -> None:
    """Publish canonical JSON without ever replacing an existing inode."""

    _publish_bytes_atomic_noreplace(
        Path(path), canonical_json_bytes(dict(payload)) + b"\n"
    )


def _load_canonical_json_file(
    path: Path,
    *,
    label: str,
    require_digest: bool = True,
) -> tuple[bytes, dict[str, Any]]:
    stream, observed = _open_plain_file(path)
    try:
        if observed.st_size > MAX_JSON_BYTES:
            raise Singleton12ArchiveError(f"{label} is unexpectedly large.")
        payload = stream.read(MAX_JSON_BYTES + 1)
    finally:
        stream.close()
    if len(payload) != observed.st_size:
        raise Singleton12ArchiveError(f"{label} changed while being read.")
    parsed = _load_json_bytes(payload, label=label)
    if payload != canonical_json_bytes(parsed) + b"\n":
        raise Singleton12ArchiveError(f"{label} is not canonical JSON bytes.")
    if require_digest:
        verify_self_digest(parsed, label=label)
    return payload, parsed


def _publish_json_idempotent(
    path: Path,
    payload: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    expected = dict(payload)
    try:
        write_json_atomic_noreplace(path, expected)
        return expected
    except FileExistsError:
        _raw, observed = _load_canonical_json_file(path, label=label)
        if observed != expected:
            raise Singleton12ArchiveError(
                f"Existing {label} differs from the requested receipt."
            )
        return observed


class _HashingReader:
    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        block = self._stream.read(size)
        self.digest.update(block)
        self.bytes_read += len(block)
        return block


class _StrictSingleGzipReader:
    """Stream one and only one gzip member under a decompressed-byte cap."""

    def __init__(self, stream: _HashingReader, *, limit_bytes: int) -> None:
        self._stream = stream
        self._inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
        self._compressed_pending = b""
        self._output = bytearray()
        self._finished = False
        self.limit_bytes = _require_int(
            limit_bytes, label="decompressed-byte limit", minimum=1
        )
        self.bytes_read = 0
        self.bytes_produced = 0

    def _pump(self) -> None:
        if self._finished:
            return
        if not self._compressed_pending:
            self._compressed_pending = self._stream.read(BLOCK_SIZE)
            if not self._compressed_pending:
                if not self._inflater.eof:
                    raise Singleton12ArchiveError(
                        "Compressed archive ended before the gzip trailer."
                    )
                self._finished = True
                return
        remaining = self.limit_bytes - self.bytes_produced
        max_output = min(BLOCK_SIZE, remaining + 1)
        try:
            block = self._inflater.decompress(
                self._compressed_pending, max_output
            )
        except zlib.error as exc:
            raise Singleton12ArchiveError("Malformed gzip stream.") from exc
        self._compressed_pending = self._inflater.unconsumed_tail
        self._output.extend(block)
        self.bytes_produced += len(block)
        if self.bytes_produced > self.limit_bytes:
            raise Singleton12ArchiveError(
                "Archive exceeds the decompressed-byte limit."
            )
        if self._inflater.eof:
            if self._inflater.unused_data or self._compressed_pending:
                raise Singleton12ArchiveError(
                    "Archive contains a second gzip member or trailing bytes."
                )
            if self._stream.read(1):
                raise Singleton12ArchiveError(
                    "Archive contains a second gzip member or trailing bytes."
                )
            self._finished = True

    def read(self, size: int = -1) -> bytes:
        requested = BLOCK_SIZE if size is None or size < 0 else size
        if requested == 0:
            return b""
        while len(self._output) < requested and not self._finished:
            before = (len(self._output), len(self._compressed_pending))
            self._pump()
            after = (len(self._output), len(self._compressed_pending))
            if before == after and not self._finished:
                raise Singleton12ArchiveError("Gzip decoder made no progress.")
        result = bytes(self._output[:requested])
        del self._output[:requested]
        self.bytes_read += len(result)
        return result

    def readinto(self, buffer: bytearray) -> int:
        block = self.read(len(buffer))
        buffer[: len(block)] = block
        return len(block)

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.bytes_read


def _consume_member(
    stream: BinaryIO,
    *,
    expected_size: int,
    capture: bool,
    label: str,
) -> tuple[str, int, bytes | None]:
    if capture and expected_size > MAX_JSON_BYTES:
        raise Singleton12ArchiveError(f"{label} is unexpectedly large.")
    digest = hashlib.sha256()
    size = 0
    chunks: list[bytes] | None = [] if capture else None
    while block := stream.read(BLOCK_SIZE):
        digest.update(block)
        size += len(block)
        if chunks is not None:
            chunks.append(block)
    if size != expected_size:
        raise Singleton12ArchiveError(
            f"{label} size differs from its tar header."
        )
    return digest.hexdigest(), size, b"".join(chunks) if chunks is not None else None


def _parse_payload_rows(
    rows: Any, *, label: str
) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise Singleton12ArchiveError(f"{label} is not a list.")
    parsed: dict[str, dict[str, Any]] = {}
    ordered: list[str] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or set(raw) != {
            "mode",
            "path",
            "sha256",
            "size_bytes",
        }:
            raise Singleton12ArchiveError(f"{label} row {index} is malformed.")
        path = _safe_member_name(str(raw.get("path")), label=f"{label} path")
        if path in parsed:
            raise Singleton12ArchiveError(f"Duplicate {label} path: {path}")
        mode = _require_int(raw.get("mode"), label=f"{label} mode")
        if mode not in {0o644, 0o755}:
            raise Singleton12ArchiveError(f"Unsupported mode in {label}: {path}")
        parsed[path] = {
            "mode": mode,
            "path": path,
            "sha256": _require_sha256(
                raw.get("sha256"), label=f"{label} digest for {path}"
            ),
            "size_bytes": _require_int(
                raw.get("size_bytes"), label=f"{label} size for {path}"
            ),
        }
        ordered.append(path)
    if ordered != sorted(ordered):
        raise Singleton12ArchiveError(f"{label} is not canonically ordered.")
    return parsed


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_execution_id: str,
    expected_source_member_prefix: str,
    expected_authority_metadata: Mapping[str, Any],
    expected_cell_metadata: Mapping[str, Any],
    limits: ArchiveLimits,
) -> dict[str, dict[str, Any]]:
    expected_keys = {
        "authority_metadata",
        "cell_metadata",
        "compression",
        "execution_id",
        "external_members",
        "limits",
        "payload_files",
        "payload_inventory_sha256",
        "schema",
        "sha256",
        "source_member_prefix",
        "source_tree",
        "status",
        "total_payload_bytes",
    }
    if set(manifest) != expected_keys:
        raise Singleton12ArchiveError("Archive manifest field closure drifted.")
    verify_self_digest(manifest, label="archive manifest")
    expected_compression = {
        "format": "gzip_pax_tar",
        "gzip_compresslevel": GZIP_COMPRESSLEVEL,
        "gzip_filename": "",
        "gzip_mtime": 0,
        "member_gid": 0,
        "member_gname": "",
        "member_mtime": 0,
        "member_uid": 0,
        "member_uname": "",
        "regular_members_only": True,
    }
    if (
        manifest.get("schema") != ARCHIVE_SCHEMA
        or manifest.get("status")
        != "passed_deterministic_authenticated_archive_manifest"
        or manifest.get("execution_id") != expected_execution_id
        or manifest.get("source_member_prefix")
        != expected_source_member_prefix
        or manifest.get("authority_metadata")
        != dict(expected_authority_metadata)
        or manifest.get("cell_metadata") != dict(expected_cell_metadata)
        or manifest.get("compression") != expected_compression
        or manifest.get("limits") != limits.as_dict()
    ):
        raise Singleton12ArchiveError(
            "Archive manifest authority or compression contract drifted."
        )
    payload_rows = _parse_payload_rows(
        manifest.get("payload_files"), label="payload inventory"
    )
    if MANIFEST_MEMBER in payload_rows:
        raise Singleton12ArchiveError(
            "Manifest member may not self-appear in payload inventory."
        )
    if manifest.get("payload_inventory_sha256") != canonical_sha256(
        {"files": list(payload_rows.values())}
    ):
        raise Singleton12ArchiveError("Payload inventory digest drifted.")
    total_payload = sum(row["size_bytes"] for row in payload_rows.values())
    if manifest.get("total_payload_bytes") != total_payload:
        raise Singleton12ArchiveError("Total archive payload size drifted.")
    if total_payload > limits.max_total_payload_bytes:
        raise Singleton12ArchiveError("Archive payload exceeds its total limit.")
    if any(
        row["size_bytes"] > limits.max_member_payload_bytes
        for row in payload_rows.values()
    ):
        raise Singleton12ArchiveError("Archive member exceeds its size limit.")

    tree = manifest.get("source_tree")
    if not isinstance(tree, Mapping) or set(tree) != {
        "directories",
        "files",
        "sha256",
    }:
        raise Singleton12ArchiveError("Source-tree manifest is malformed.")
    tree_unsigned = dict(tree)
    tree_digest = tree_unsigned.pop("sha256", None)
    if tree_digest != canonical_sha256(tree_unsigned):
        raise Singleton12ArchiveError("Source-tree inventory digest drifted.")
    directories = tree.get("directories")
    tree_files = tree.get("files")
    if not isinstance(directories, list) or not isinstance(tree_files, list):
        raise Singleton12ArchiveError("Source-tree rows are malformed.")
    directory_names: list[str] = []
    for index, row in enumerate(directories):
        if not isinstance(row, Mapping) or set(row) != {"mode", "path"}:
            raise Singleton12ArchiveError(
                f"Source-tree directory row {index} is malformed."
            )
        name = _safe_relative(
            str(row.get("path")), label="source-tree directory"
        )
        if row.get("mode") != 0o755:
            raise Singleton12ArchiveError(
                f"Source-tree directory mode drifted: {name}"
            )
        directory_names.append(name)
    if directory_names != sorted(set(directory_names)):
        raise Singleton12ArchiveError(
            "Source-tree directories are duplicate or unordered."
        )

    source_names: list[str] = []
    source_payload_names: set[str] = set()
    for index, row in enumerate(tree_files):
        if not isinstance(row, Mapping) or set(row) != {
            "archive_path",
            "mode",
            "path",
            "sha256",
            "size_bytes",
        }:
            raise Singleton12ArchiveError(
                f"Source-tree file row {index} is malformed."
            )
        relative = _safe_relative(
            str(row.get("path")), label="source-tree file"
        )
        archive_path = _safe_member_name(
            str(row.get("archive_path")), label="source archive member"
        )
        if archive_path != f"{expected_source_member_prefix}/{relative}":
            raise Singleton12ArchiveError(
                f"Source archive path drifted: {relative}"
            )
        expected_row = payload_rows.get(archive_path)
        if expected_row is None or expected_row != {
            "mode": row.get("mode"),
            "path": archive_path,
            "sha256": row.get("sha256"),
            "size_bytes": row.get("size_bytes"),
        }:
            raise Singleton12ArchiveError(
                f"Source-to-payload binding drifted: {relative}"
            )
        source_names.append(relative)
        source_payload_names.add(archive_path)
    if source_names != sorted(set(source_names)) or not source_names:
        raise Singleton12ArchiveError(
            "Source-tree files are absent, duplicate, or unordered."
        )
    directory_set = set(directory_names)
    for name in [*directory_names, *source_names]:
        parts = PurePosixPath(name).parts
        for depth in range(1, len(parts)):
            parent = PurePosixPath(*parts[:depth]).as_posix()
            if parent not in directory_set:
                raise Singleton12ArchiveError(
                    f"Source-tree parent directory is undeclared: {parent}"
                )

    external_rows = _parse_payload_rows(
        manifest.get("external_members"), label="external-member inventory"
    )
    if set(external_rows) & source_payload_names:
        raise Singleton12ArchiveError(
            "External and source archive inventories overlap."
        )
    if set(payload_rows) != source_payload_names | set(external_rows):
        raise Singleton12ArchiveError(
            "Source and external inventories do not close payload members."
        )
    for name, row in external_rows.items():
        if payload_rows.get(name) != row:
            raise Singleton12ArchiveError(
                f"External-to-payload binding drifted: {name}"
            )
    return payload_rows


def _validate_cell_archive_details(
    archive_path: Path,
    *,
    expected_execution_id: str,
    expected_source_member_prefix: str,
    expected_authority_metadata: Mapping[str, Any],
    expected_cell_metadata: Mapping[str, Any],
    limits: ArchiveLimits,
) -> _ArchiveScan:
    _safe_component(expected_execution_id, label="execution ID")
    prefix = _safe_member_name(
        expected_source_member_prefix, label="source member prefix"
    )
    if PurePosixPath(prefix).parts[-1] != expected_execution_id:
        raise Singleton12ArchiveError(
            "Source member prefix is not scoped to the execution ID."
        )
    authority = _normalized_mapping(
        expected_authority_metadata, label="expected authority metadata"
    )
    cell = _normalized_mapping(
        expected_cell_metadata, label="expected cell metadata"
    )
    archive_file = Path(archive_path)
    stream, initial_stat = _open_plain_file(archive_file)
    if initial_stat.st_size < 1:
        stream.close()
        raise Singleton12ArchiveError("Archive is empty.")
    if initial_stat.st_size > limits.max_compressed_bytes:
        stream.close()
        raise Singleton12ArchiveError("Archive exceeds compressed-byte limit.")

    hashing = _HashingReader(stream)
    decompressed = _StrictSingleGzipReader(
        hashing, limit_bytes=limits.max_decompressed_bytes
    )
    observed_members: dict[str, dict[str, Any]] = {}
    manifest_bytes: bytes | None = None
    total_payload = 0
    try:
        try:
            with tarfile.open(
                fileobj=decompressed, mode="r|", format=tarfile.PAX_FORMAT
            ) as archive:
                for member in archive:
                    name = _safe_member_name(member.name)
                    if name in observed_members:
                        raise Singleton12ArchiveError(
                            f"Duplicate archive member: {name}"
                        )
                    if member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE}:
                        raise Singleton12ArchiveError(
                            f"Non-regular archive member: {name}"
                        )
                    size = _require_int(
                        member.size, label=f"archive member size for {name}"
                    )
                    if size > limits.max_member_payload_bytes:
                        raise Singleton12ArchiveError(
                            f"Archive member exceeds its byte limit: {name}"
                        )
                    if size > limits.max_total_payload_bytes - total_payload:
                        raise Singleton12ArchiveError(
                            "Archive exceeds the total member-payload limit."
                        )
                    total_payload += size
                    mode = member.mode & 0o7777
                    if (
                        mode not in {0o644, 0o755}
                        or member.uid != 0
                        or member.gid != 0
                        or member.uname != ""
                        or member.gname != ""
                        or member.mtime != 0
                        or member.linkname != ""
                    ):
                        raise Singleton12ArchiveError(
                            f"Archive header normalization drifted: {name}"
                        )
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise Singleton12ArchiveError(
                            f"Archive member is unreadable: {name}"
                        )
                    digest, consumed, captured = _consume_member(
                        extracted,
                        expected_size=size,
                        capture=name == MANIFEST_MEMBER,
                        label=f"archive member {name}",
                    )
                    observed_members[name] = {
                        "mode": mode,
                        "path": name,
                        "sha256": digest,
                        "size_bytes": consumed,
                    }
                    if name == MANIFEST_MEMBER:
                        assert captured is not None
                        manifest_bytes = captured
            while trailing := decompressed.read(BLOCK_SIZE):
                if trailing.strip(b"\0"):
                    raise Singleton12ArchiveError(
                        "Archive contains non-zero trailing tar payload."
                    )
        except (OSError, EOFError, tarfile.TarError) as exc:
            raise Singleton12ArchiveError(
                f"Archive is not a complete gzip/PAX tar: {archive_file}"
            ) from exc
        final_stat = os.fstat(stream.fileno())
    finally:
        stream.close()

    if hashing.bytes_read != initial_stat.st_size:
        raise Singleton12ArchiveError(
            "Archive compressed bytes were not consumed exactly once."
        )
    if (
        final_stat.st_dev != initial_stat.st_dev
        or final_stat.st_ino != initial_stat.st_ino
        or final_stat.st_size != initial_stat.st_size
        or final_stat.st_mtime_ns != initial_stat.st_mtime_ns
        or final_stat.st_ctime_ns != initial_stat.st_ctime_ns
    ):
        raise Singleton12ArchiveError("Archive changed during validation.")
    try:
        path_stat = archive_file.lstat()
    except FileNotFoundError as exc:
        raise Singleton12ArchiveError(
            "Archive pathname disappeared during validation."
        ) from exc
    if (
        not stat.S_ISREG(path_stat.st_mode)
        or stat.S_ISLNK(path_stat.st_mode)
        or path_stat.st_dev != initial_stat.st_dev
        or path_stat.st_ino != initial_stat.st_ino
    ):
        raise Singleton12ArchiveError(
            "Archive pathname identity changed during validation."
        )
    if manifest_bytes is None:
        raise Singleton12ArchiveError("Archive manifest member is missing.")
    manifest = _load_json_bytes(manifest_bytes, label="archive manifest")
    if manifest_bytes != canonical_json_bytes(manifest) + b"\n":
        raise Singleton12ArchiveError(
            "Archive manifest is not canonical JSON bytes."
        )
    payload_rows = _validate_manifest(
        manifest,
        expected_execution_id=expected_execution_id,
        expected_source_member_prefix=prefix,
        expected_authority_metadata=authority,
        expected_cell_metadata=cell,
        limits=limits,
    )
    expected_names = set(payload_rows) | {MANIFEST_MEMBER}
    if set(observed_members) != expected_names:
        raise Singleton12ArchiveError("Archive member-name closure drifted.")
    for name, expected in payload_rows.items():
        if observed_members[name] != expected:
            raise Singleton12ArchiveError(
                f"Archived member digest/size/mode drifted: {name}"
            )
    manifest_observed = observed_members[MANIFEST_MEMBER]
    if manifest_observed["mode"] != 0o644:
        raise Singleton12ArchiveError("Archive manifest mode drifted.")
    archive_sha = hashing.digest.hexdigest()
    validation = digested(
        {
            "schema": VALIDATION_SCHEMA,
            "status": "passed_full_bounded_streaming_validation",
            "execution_id": expected_execution_id,
            "authority_metadata": authority,
            "cell_metadata": cell,
            "source_member_prefix": prefix,
            "archive": {
                "sha256": archive_sha,
                "size_bytes": initial_stat.st_size,
            },
            "archive_manifest": {
                "canonical_sha256": manifest["sha256"],
                "file_sha256": manifest_observed["sha256"],
                "size_bytes": manifest_observed["size_bytes"],
            },
            "member_validation": {
                "decompressed_bytes": decompressed.bytes_produced,
                "member_count": len(observed_members),
                "regular_unique_safe_members_only": True,
                "single_gzip_member_only": True,
                "total_member_payload_bytes": total_payload,
            },
            "payload_inventory_sha256": manifest[
                "payload_inventory_sha256"
            ],
            "source_tree": _tree_summary(manifest["source_tree"]),
        }
    )
    return _ArchiveScan(
        manifest=dict(manifest),
        manifest_bytes=manifest_bytes,
        validation=validation,
    )


def validate_cell_archive(
    archive_path: Path,
    *,
    expected_execution_id: str,
    expected_source_member_prefix: str,
    expected_authority_metadata: Mapping[str, Any],
    expected_cell_metadata: Mapping[str, Any],
    limits: ArchiveLimits,
) -> dict[str, Any]:
    """Fully stream-validate an archive without extracting any member."""

    return _validate_cell_archive_details(
        Path(archive_path),
        expected_execution_id=expected_execution_id,
        expected_source_member_prefix=expected_source_member_prefix,
        expected_authority_metadata=expected_authority_metadata,
        expected_cell_metadata=expected_cell_metadata,
        limits=limits,
    ).validation


def _ensure_external_manifest(
    paths: CellArchivePaths, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    path = paths.archive_manifest_path
    try:
        write_json_atomic_noreplace(path, manifest)
        return dict(manifest)
    except FileExistsError:
        _raw, observed = _load_canonical_json_file(
            path, label="external archive manifest"
        )
        if observed != dict(manifest):
            raise Singleton12ArchiveError(
                "External archive manifest differs from archived manifest."
            )
        return observed


def _write_deterministic_archive(
    *,
    temporary: Path,
    archive_parent: Path,
    source_files: Sequence[_SourceFile],
    external_rows: Sequence[Mapping[str, Any]],
    external_sources: Sequence[tuple[str, Path]],
    manifest: Mapping[str, Any],
    limits: ArchiveLimits,
) -> None:
    manifest_bytes = canonical_json_bytes(manifest) + b"\n"
    total = int(manifest["total_payload_bytes"]) + len(manifest_bytes)
    if len(manifest_bytes) > limits.max_member_payload_bytes:
        raise Singleton12ArchiveError("Archive manifest exceeds member limit.")
    if total > limits.max_total_payload_bytes:
        raise Singleton12ArchiveError(
            "Archive including its manifest exceeds total payload limit."
        )
    external_by_name = {name: path for name, path in external_sources}
    source_by_name = {row.archive_path: row for row in source_files}
    payload_rows = _parse_payload_rows(
        manifest["payload_files"], label="builder payload inventory"
    )
    external_inventory = {str(row["path"]): dict(row) for row in external_rows}
    with temporary.open("xb") as raw:
        bounded = _BoundedCompressedWriter(
            raw,
            parent=archive_parent,
            max_bytes=limits.max_compressed_bytes,
            min_free_bytes=limits.min_free_disk_bytes,
        )
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=GZIP_COMPRESSLEVEL,
            fileobj=bounded,
            mtime=0,
        ) as compressed:
            with tarfile.open(
                mode="w|",
                fileobj=compressed,
                format=tarfile.PAX_FORMAT,
            ) as archive:
                for name, row in payload_rows.items():
                    source = source_by_name.get(name)
                    if source is not None:
                        path = source.path
                    else:
                        path = external_by_name.get(name)
                        if path is None or external_inventory.get(name) != row:
                            raise Singleton12ArchiveError(
                                f"Builder inventory does not resolve: {name}"
                            )
                    _write_bound_file(
                        archive,
                        name=name,
                        path=path,
                        expected_sha256=str(row["sha256"]),
                        expected_size=int(row["size_bytes"]),
                        mode=int(row["mode"]),
                    )
                archive.addfile(
                    _tar_info(
                        name=MANIFEST_MEMBER,
                        size=len(manifest_bytes),
                        mode=0o644,
                    ),
                    io.BytesIO(manifest_bytes),
                )
        bounded.flush()
        raw.flush()
        os.fsync(raw.fileno())


def build_cell_archive(
    *,
    paths: CellArchivePaths,
    source_member_prefix: str,
    external_members: Mapping[str, Path],
    authority_metadata: Mapping[str, Any],
    cell_metadata: Mapping[str, Any],
    limits: ArchiveLimits,
) -> dict[str, Any]:
    """Build, validate, and no-replace publish one cell archive.

    If the final archive already exists, this acts as a strict restart: it
    validates that archive and idempotently publishes only the matching
    external manifest.  The source tree is never removed here.
    """

    runtime = _require_runtime(paths)
    prefix = _safe_member_name(
        source_member_prefix, label="source member prefix"
    )
    if PurePosixPath(prefix).parts[-1] != paths.execution_id:
        raise Singleton12ArchiveError(
            "Source member prefix is not scoped to the execution ID."
        )
    authority = _normalized_mapping(authority_metadata, label="authority metadata")
    cell = _normalized_mapping(cell_metadata, label="cell metadata")
    for directory in (
        paths.archive_path.parent,
        paths.archive_manifest_path.parent,
    ):
        _ensure_operational_directory(directory, runtime_root=runtime)

    if paths.archive_path.exists() or paths.archive_path.is_symlink():
        scan = _validate_cell_archive_details(
            paths.archive_path,
            expected_execution_id=paths.execution_id,
            expected_source_member_prefix=prefix,
            expected_authority_metadata=authority,
            expected_cell_metadata=cell,
            limits=limits,
        )
        _ensure_external_manifest(paths, scan.manifest)
        return scan.validation

    if (
        shutil.disk_usage(paths.archive_path.parent).free
        < limits.archive_start_free_floor_bytes
    ):
        raise Singleton12ArchiveError(
            "Free disk is below compressed-cap plus post-write reserve."
        )
    directories, source_files = _scan_source_tree(
        paths.source_root, source_member_prefix=prefix
    )
    external_rows, external_sources = _external_rows(
        external_members, source_member_prefix=prefix
    )
    manifest = _manifest_payload(
        paths=paths,
        source_member_prefix=prefix,
        directories=directories,
        source_files=source_files,
        external_rows=external_rows,
        authority_metadata=authority,
        cell_metadata=cell,
        limits=limits,
    )
    temporary = _temporary_archive_path(paths)
    linked = False
    try:
        _write_deterministic_archive(
            temporary=temporary,
            archive_parent=paths.archive_path.parent,
            source_files=source_files,
            external_rows=external_rows,
            external_sources=external_sources,
            manifest=manifest,
            limits=limits,
        )
        temporary_scan = _validate_cell_archive_details(
            temporary,
            expected_execution_id=paths.execution_id,
            expected_source_member_prefix=prefix,
            expected_authority_metadata=authority,
            expected_cell_metadata=cell,
            limits=limits,
        )
        if temporary_scan.manifest != manifest:
            raise Singleton12ArchiveError(
                "Temporary archive manifest drifted before publication."
            )
        os.link(temporary, paths.archive_path)
        linked = True
        _fsync_directory(paths.archive_path.parent)
    except FileExistsError as exc:
        raise Singleton12ArchiveError(
            "Archive destination was concurrently published; restart validation."
        ) from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        _fsync_directory(paths.archive_path.parent)
    if not linked:
        raise Singleton12ArchiveError("Archive publication did not complete.")
    final_scan = _validate_cell_archive_details(
        paths.archive_path,
        expected_execution_id=paths.execution_id,
        expected_source_member_prefix=prefix,
        expected_authority_metadata=authority,
        expected_cell_metadata=cell,
        limits=limits,
    )
    _ensure_external_manifest(paths, final_scan.manifest)
    return final_scan.validation


def _relative_binding(
    paths: CellArchivePaths,
    path: Path,
    *,
    canonical_digest: str | None = None,
) -> dict[str, Any]:
    runtime = _require_runtime(paths)
    absolute = path.absolute()
    try:
        relative = absolute.relative_to(runtime).as_posix()
    except ValueError as exc:
        raise Singleton12ArchiveError(
            f"Receipt binding escaped runtime root: {path}"
        ) from exc
    digest, size, _mode = _sha256_plain_file(path)
    result: dict[str, Any] = {
        "path": relative,
        "sha256": digest,
        "size_bytes": size,
    }
    if canonical_digest is not None:
        result["canonical_sha256"] = canonical_digest
    return result


def _expected_archive_binding(
    paths: CellArchivePaths, validation: Mapping[str, Any]
) -> dict[str, Any]:
    archive = validation.get("archive")
    if not isinstance(archive, Mapping):
        raise Singleton12ArchiveError("Archive validation binding is malformed.")
    binding = _relative_binding(paths, paths.archive_path)
    if (
        binding["sha256"] != archive.get("sha256")
        or binding["size_bytes"] != archive.get("size_bytes")
    ):
        raise Singleton12ArchiveError(
            "Archive validation and final-path binding differ."
        )
    return binding


def _tree_matches_manifest(
    paths: CellArchivePaths,
    *,
    root: Path,
    source_member_prefix: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    directories, files = _scan_source_tree(
        root, source_member_prefix=source_member_prefix
    )
    observed = _tree_payload(directories, files)
    expected = manifest.get("source_tree")
    if not isinstance(expected, Mapping) or observed != dict(expected):
        raise Singleton12ArchiveError(
            f"Safe-tree inventory differs from authenticated archive: {root}"
        )
    return observed


def _remaining_tree_is_manifest_subset(
    *,
    root: Path,
    source_member_prefix: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate every member left after an interrupted safe removal.

    Missing authenticated members are allowed because ``shutil.rmtree`` may
    have removed them before the process crashed.  Every member that remains
    must still be a plain directory or byte-identical regular file at its
    original relative path; nothing outside the archived source inventory is
    accepted.
    """

    tree = manifest.get("source_tree")
    if not isinstance(tree, Mapping):
        raise Singleton12ArchiveError(
            "Authenticated source-tree manifest is malformed."
        )
    expected_directories_raw = tree.get("directories")
    expected_files_raw = tree.get("files")
    if not isinstance(expected_directories_raw, list) or not isinstance(
        expected_files_raw, list
    ):
        raise Singleton12ArchiveError(
            "Authenticated source-tree rows are malformed."
        )
    expected_directories = {
        str(row.get("path")): dict(row)
        for row in expected_directories_raw
        if isinstance(row, Mapping)
    }
    expected_files = {
        str(row.get("path")): dict(row)
        for row in expected_files_raw
        if isinstance(row, Mapping)
    }
    if (
        len(expected_directories) != len(expected_directories_raw)
        or len(expected_files) != len(expected_files_raw)
    ):
        raise Singleton12ArchiveError(
            "Authenticated source-tree inventory is not unique."
        )

    directories, files = _scan_source_tree(
        root,
        source_member_prefix=source_member_prefix,
        require_files=False,
    )
    for row in directories:
        expected = expected_directories.get(str(row["path"]))
        if expected is None:
            raise Singleton12ArchiveError(
                "Retiring tree contains an extra or wrong-type directory: "
                f"{row['path']}"
            )
        if row != expected:
            raise Singleton12ArchiveError(
                f"Retiring directory metadata drifted: {row['path']}"
            )
    for source_file in files:
        observed = source_file.tree_row()
        expected = expected_files.get(source_file.relative_path)
        if expected is None:
            raise Singleton12ArchiveError(
                "Retiring tree contains an extra or wrong-type file: "
                f"{source_file.relative_path}"
            )
        if observed != expected:
            raise Singleton12ArchiveError(
                "Retiring file digest, size, or mode drifted: "
                f"{source_file.relative_path}"
            )
    return _tree_payload(directories, files)


def _require_no_archive_temporaries(paths: CellArchivePaths) -> None:
    if _stale_archive_temporaries(paths):
        raise Singleton12ArchiveError(
            "Stale archive temporary files require explicit safe disposal."
        )


def _stale_archive_temporaries(paths: CellArchivePaths) -> list[Path]:
    parent = paths.archive_path.parent
    if not parent.is_dir() or parent.is_symlink():
        return []
    pattern = re.compile(
        rf"\.{re.escape(paths.archive_path.name)}\.tmp\.[0-9a-f]{{32}}\Z"
    )
    return sorted(
        (
            row
            for row in parent.iterdir()
            if pattern.fullmatch(row.name) is not None
        ),
        key=lambda row: row.name,
    )


def discard_stale_archive_temporaries(paths: CellArchivePaths) -> list[str]:
    """Remove only module-named build temporaries while source is preserved."""

    _require_runtime(paths)
    if (
        not paths.source_root.is_dir()
        or paths.source_root.is_symlink()
        or paths.retiring_root.exists()
        or paths.retiring_root.is_symlink()
        or paths.archive_closure_path.exists()
        or paths.archive_closure_path.is_symlink()
        or paths.rotation_intent_path.exists()
        or paths.rotation_intent_path.is_symlink()
        or paths.cleanup_receipt_path.exists()
        or paths.cleanup_receipt_path.is_symlink()
    ):
        raise Singleton12ArchiveError(
            "Stale archive temporaries are not disposable in this state."
        )
    removed: list[str] = []
    for temporary in _stale_archive_temporaries(paths):
        observed = temporary.lstat()
        if not stat.S_ISREG(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
            raise Singleton12ArchiveError(
                f"Archive temporary is not a plain file: {temporary}"
            )
        temporary.unlink()
        removed.append(temporary.name)
    if removed:
        _fsync_directory(paths.archive_path.parent)
    return removed


def _validate_external_manifest(
    paths: CellArchivePaths, scan: _ArchiveScan
) -> dict[str, Any]:
    raw, manifest = _load_canonical_json_file(
        paths.archive_manifest_path, label="external archive manifest"
    )
    if manifest != scan.manifest or raw != scan.manifest_bytes:
        raise Singleton12ArchiveError(
            "External and archived manifests are not byte-identical."
        )
    return manifest


def publish_archive_closure(
    *,
    paths: CellArchivePaths,
    source_member_prefix: str,
    authority_metadata: Mapping[str, Any],
    cell_metadata: Mapping[str, Any],
    limits: ArchiveLimits,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Publish the no-replace receipt proving archive/source closure."""

    runtime = _require_runtime(paths)
    _require_no_archive_temporaries(paths)
    authority = _normalized_mapping(authority_metadata, label="authority metadata")
    cell = _normalized_mapping(cell_metadata, label="cell metadata")
    scan = _validate_cell_archive_details(
        paths.archive_path,
        expected_execution_id=paths.execution_id,
        expected_source_member_prefix=source_member_prefix,
        expected_authority_metadata=authority,
        expected_cell_metadata=cell,
        limits=limits,
    )
    manifest = _validate_external_manifest(paths, scan)
    observed_tree = _tree_matches_manifest(
        paths,
        root=paths.source_root,
        source_member_prefix=source_member_prefix,
        manifest=manifest,
    )
    if paths.retiring_root.exists() or paths.retiring_root.is_symlink():
        raise Singleton12ArchiveError(
            "Retiring tree exists before archive closure."
        )
    _ensure_operational_directory(
        paths.archive_closure_path.parent, runtime_root=runtime
    )
    if paths.archive_closure_path.exists() or paths.archive_closure_path.is_symlink():
        _raw, existing = _load_canonical_json_file(
            paths.archive_closure_path, label="archive closure"
        )
        _validate_closure_payload(
            paths=paths,
            closure=existing,
            scan=scan,
            authority_metadata=authority,
            cell_metadata=cell,
            source_member_prefix=source_member_prefix,
        )
        return existing
    closure = digested(
        {
            "schema": CLOSURE_SCHEMA,
            "status": "passed_archive_and_direct_tree_byte_closure",
            "created_at_utc": _utc_timestamp(
                created_at_utc, label="archive closure timestamp"
            ),
            "execution_id": paths.execution_id,
            "authority_metadata": authority,
            "cell_metadata": cell,
            "source_member_prefix": source_member_prefix,
            "archive": _expected_archive_binding(paths, scan.validation),
            "archive_manifest": _relative_binding(
                paths,
                paths.archive_manifest_path,
                canonical_digest=str(manifest["sha256"]),
            ),
            "archive_validation": scan.validation,
            "source_tree": _tree_summary(observed_tree),
            "direct_source_present_at_closure": True,
            "retiring_source_absent_at_closure": True,
        }
    )
    return _publish_json_idempotent(
        paths.archive_closure_path, closure, label="archive closure"
    )


def _validate_closure_payload(
    *,
    paths: CellArchivePaths,
    closure: Mapping[str, Any],
    scan: _ArchiveScan,
    authority_metadata: Mapping[str, Any],
    cell_metadata: Mapping[str, Any],
    source_member_prefix: str,
) -> None:
    verify_self_digest(closure, label="archive closure")
    _utc_timestamp(str(closure.get("created_at_utc")), label="closure timestamp")
    expected_archive = _expected_archive_binding(paths, scan.validation)
    expected_manifest = _relative_binding(
        paths,
        paths.archive_manifest_path,
        canonical_digest=str(scan.manifest["sha256"]),
    )
    if (
        closure.get("schema") != CLOSURE_SCHEMA
        or closure.get("status")
        != "passed_archive_and_direct_tree_byte_closure"
        or closure.get("execution_id") != paths.execution_id
        or closure.get("authority_metadata") != dict(authority_metadata)
        or closure.get("cell_metadata") != dict(cell_metadata)
        or closure.get("source_member_prefix") != source_member_prefix
        or closure.get("archive") != expected_archive
        or closure.get("archive_manifest") != expected_manifest
        or closure.get("archive_validation") != scan.validation
        or closure.get("source_tree")
        != _tree_summary(scan.manifest["source_tree"])
        or closure.get("direct_source_present_at_closure") is not True
        or closure.get("retiring_source_absent_at_closure") is not True
    ):
        raise Singleton12ArchiveError("Archive closure binding drifted.")


def _load_valid_closure(
    *,
    paths: CellArchivePaths,
    scan: _ArchiveScan,
    authority_metadata: Mapping[str, Any],
    cell_metadata: Mapping[str, Any],
    source_member_prefix: str,
) -> dict[str, Any]:
    _raw, closure = _load_canonical_json_file(
        paths.archive_closure_path, label="archive closure"
    )
    _validate_closure_payload(
        paths=paths,
        closure=closure,
        scan=scan,
        authority_metadata=authority_metadata,
        cell_metadata=cell_metadata,
        source_member_prefix=source_member_prefix,
    )
    return closure


def _validate_rotation_authority(
    rotation_authority: Mapping[str, Any],
) -> dict[str, Any]:
    authority = _normalized_mapping(
        rotation_authority, label="rotation authority"
    )
    if (
        authority.get("execution_authorized") is not True
        or authority.get("archive_rotation_authorized") is not True
    ):
        raise Singleton12ArchiveError(
            "Rotation authority must explicitly authorize execution and archive rotation."
        )
    return authority


def publish_rotation_intent(
    *,
    paths: CellArchivePaths,
    source_member_prefix: str,
    authority_metadata: Mapping[str, Any],
    cell_metadata: Mapping[str, Any],
    rotation_authority: Mapping[str, Any],
    limits: ArchiveLimits,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Publish independent deletion authority after archive closure passes."""

    runtime = _require_runtime(paths)
    _require_no_archive_temporaries(paths)
    authority = _normalized_mapping(authority_metadata, label="authority metadata")
    cell = _normalized_mapping(cell_metadata, label="cell metadata")
    rotation = _validate_rotation_authority(rotation_authority)
    scan = _validate_cell_archive_details(
        paths.archive_path,
        expected_execution_id=paths.execution_id,
        expected_source_member_prefix=source_member_prefix,
        expected_authority_metadata=authority,
        expected_cell_metadata=cell,
        limits=limits,
    )
    _validate_external_manifest(paths, scan)
    closure = _load_valid_closure(
        paths=paths,
        scan=scan,
        authority_metadata=authority,
        cell_metadata=cell,
        source_member_prefix=source_member_prefix,
    )
    _tree_matches_manifest(
        paths,
        root=paths.source_root,
        source_member_prefix=source_member_prefix,
        manifest=scan.manifest,
    )
    if paths.retiring_root.exists() or paths.retiring_root.is_symlink():
        raise Singleton12ArchiveError(
            "Retiring tree exists before rotation intent."
        )
    _ensure_operational_directory(
        paths.rotation_intent_path.parent, runtime_root=runtime
    )
    if paths.rotation_intent_path.exists() or paths.rotation_intent_path.is_symlink():
        _raw, existing = _load_canonical_json_file(
            paths.rotation_intent_path, label="rotation intent"
        )
        _validate_intent_payload(
            paths=paths,
            intent=existing,
            closure=closure,
            scan=scan,
            authority_metadata=authority,
            cell_metadata=cell,
            source_member_prefix=source_member_prefix,
            rotation_authority=rotation,
        )
        return existing
    intent = digested(
        {
            "schema": ROTATION_INTENT_SCHEMA,
            "status": "authorized_exact_safe_tree_rotation",
            "created_at_utc": _utc_timestamp(
                created_at_utc, label="rotation intent timestamp"
            ),
            "execution_id": paths.execution_id,
            "authority_metadata": authority,
            "cell_metadata": cell,
            "rotation_authority": rotation,
            "source_member_prefix": source_member_prefix,
            "archive": _expected_archive_binding(paths, scan.validation),
            "archive_closure": _relative_binding(
                paths,
                paths.archive_closure_path,
                canonical_digest=str(closure["sha256"]),
            ),
            "source_tree": _tree_summary(scan.manifest["source_tree"]),
            "source_path": paths.source_root.relative_to(runtime).as_posix(),
            "retiring_path": paths.retiring_root.relative_to(runtime).as_posix(),
            "rotation_scope": "rename_then_revalidate_then_remove_exact_tree",
        }
    )
    return _publish_json_idempotent(
        paths.rotation_intent_path, intent, label="rotation intent"
    )


def _validate_intent_payload(
    *,
    paths: CellArchivePaths,
    intent: Mapping[str, Any],
    closure: Mapping[str, Any],
    scan: _ArchiveScan,
    authority_metadata: Mapping[str, Any],
    cell_metadata: Mapping[str, Any],
    source_member_prefix: str,
    rotation_authority: Mapping[str, Any] | None,
) -> None:
    verify_self_digest(intent, label="rotation intent")
    _utc_timestamp(str(intent.get("created_at_utc")), label="intent timestamp")
    runtime = _require_runtime(paths)
    observed_rotation = intent.get("rotation_authority")
    if not isinstance(observed_rotation, Mapping):
        raise Singleton12ArchiveError("Rotation authority is absent from intent.")
    _validate_rotation_authority(observed_rotation)
    if rotation_authority is not None and dict(observed_rotation) != dict(
        rotation_authority
    ):
        raise Singleton12ArchiveError("Rotation authority drifted.")
    if (
        intent.get("schema") != ROTATION_INTENT_SCHEMA
        or intent.get("status") != "authorized_exact_safe_tree_rotation"
        or intent.get("execution_id") != paths.execution_id
        or intent.get("authority_metadata") != dict(authority_metadata)
        or intent.get("cell_metadata") != dict(cell_metadata)
        or intent.get("source_member_prefix") != source_member_prefix
        or intent.get("archive")
        != _expected_archive_binding(paths, scan.validation)
        or intent.get("archive_closure")
        != _relative_binding(
            paths,
            paths.archive_closure_path,
            canonical_digest=str(closure["sha256"]),
        )
        or intent.get("source_tree")
        != _tree_summary(scan.manifest["source_tree"])
        or intent.get("source_path")
        != paths.source_root.relative_to(runtime).as_posix()
        or intent.get("retiring_path")
        != paths.retiring_root.relative_to(runtime).as_posix()
        or intent.get("rotation_scope")
        != "rename_then_revalidate_then_remove_exact_tree"
    ):
        raise Singleton12ArchiveError("Rotation-intent binding drifted.")


def _load_valid_intent(
    *,
    paths: CellArchivePaths,
    closure: Mapping[str, Any],
    scan: _ArchiveScan,
    authority_metadata: Mapping[str, Any],
    cell_metadata: Mapping[str, Any],
    source_member_prefix: str,
    rotation_authority: Mapping[str, Any] | None,
) -> dict[str, Any]:
    _raw, intent = _load_canonical_json_file(
        paths.rotation_intent_path, label="rotation intent"
    )
    _validate_intent_payload(
        paths=paths,
        intent=intent,
        closure=closure,
        scan=scan,
        authority_metadata=authority_metadata,
        cell_metadata=cell_metadata,
        source_member_prefix=source_member_prefix,
        rotation_authority=rotation_authority,
    )
    return intent


def _validate_cleanup_payload(
    *,
    paths: CellArchivePaths,
    cleanup: Mapping[str, Any],
    intent: Mapping[str, Any],
    closure: Mapping[str, Any],
    scan: _ArchiveScan,
    authority_metadata: Mapping[str, Any],
    cell_metadata: Mapping[str, Any],
    source_member_prefix: str,
) -> None:
    verify_self_digest(cleanup, label="cleanup receipt")
    _utc_timestamp(str(cleanup.get("completed_at_utc")), label="cleanup timestamp")
    if (
        cleanup.get("schema") != CLEANUP_SCHEMA
        or cleanup.get("status")
        != "passed_exact_safe_tree_removed_archive_retained"
        or cleanup.get("execution_id") != paths.execution_id
        or cleanup.get("authority_metadata") != dict(authority_metadata)
        or cleanup.get("cell_metadata") != dict(cell_metadata)
        or cleanup.get("source_member_prefix") != source_member_prefix
        or cleanup.get("archive")
        != _expected_archive_binding(paths, scan.validation)
        or cleanup.get("archive_closure")
        != _relative_binding(
            paths,
            paths.archive_closure_path,
            canonical_digest=str(closure["sha256"]),
        )
        or cleanup.get("rotation_intent")
        != _relative_binding(
            paths,
            paths.rotation_intent_path,
            canonical_digest=str(intent["sha256"]),
        )
        or cleanup.get("removed_source_tree")
        != _tree_summary(scan.manifest["source_tree"])
        or cleanup.get("direct_source_absent") is not True
        or cleanup.get("retiring_source_absent") is not True
        or cleanup.get("archive_retained") is not True
    ):
        raise Singleton12ArchiveError("Cleanup-receipt binding drifted.")


def _load_valid_cleanup(
    *,
    paths: CellArchivePaths,
    intent: Mapping[str, Any],
    closure: Mapping[str, Any],
    scan: _ArchiveScan,
    authority_metadata: Mapping[str, Any],
    cell_metadata: Mapping[str, Any],
    source_member_prefix: str,
) -> dict[str, Any]:
    _raw, cleanup = _load_canonical_json_file(
        paths.cleanup_receipt_path, label="cleanup receipt"
    )
    _validate_cleanup_payload(
        paths=paths,
        cleanup=cleanup,
        intent=intent,
        closure=closure,
        scan=scan,
        authority_metadata=authority_metadata,
        cell_metadata=cell_metadata,
        source_member_prefix=source_member_prefix,
    )
    return cleanup


def _path_presence(path: Path, *, expected: str) -> bool:
    try:
        observed = path.lstat()
    except FileNotFoundError:
        return False
    if expected == "directory":
        valid = stat.S_ISDIR(observed.st_mode) and not stat.S_ISLNK(
            observed.st_mode
        )
    else:
        valid = stat.S_ISREG(observed.st_mode) and not stat.S_ISLNK(
            observed.st_mode
        )
    if not valid:
        raise Singleton12ArchiveError(
            f"Rotation artifact has unsafe type: {path}"
        )
    return True


def inspect_rotation_state(paths: CellArchivePaths) -> dict[str, Any]:
    """Classify the exact durable state; reject every impossible combination."""

    _require_runtime(paths)
    source = _path_presence(paths.source_root, expected="directory")
    retiring = _path_presence(paths.retiring_root, expected="directory")
    archive = _path_presence(paths.archive_path, expected="file")
    manifest = _path_presence(paths.archive_manifest_path, expected="file")
    closure = _path_presence(paths.archive_closure_path, expected="file")
    intent = _path_presence(paths.rotation_intent_path, expected="file")
    cleanup = _path_presence(paths.cleanup_receipt_path, expected="file")
    temporaries = _stale_archive_temporaries(paths)
    for temporary in temporaries:
        _path_presence(temporary, expected="file")

    durable = (archive, manifest, closure, intent, cleanup)
    state: str | None = None
    if source and retiring:
        state = None
    elif source:
        mapping = {
            (False, False, False, False, False): "direct_unarchived",
            (True, False, False, False, False): (
                "archive_published_pending_manifest"
            ),
            (True, True, False, False, False): (
                "manifest_published_pending_closure"
            ),
            (True, True, True, False, False): (
                "closure_published_pending_intent"
            ),
            (True, True, True, True, False): (
                "intent_published_pending_rename"
            ),
        }
        state = mapping.get(durable)
    elif retiring:
        if durable == (True, True, True, True, False):
            state = "retiring_pending_removal"
    else:
        if durable == (False, False, False, False, False):
            state = "empty"
        elif durable == (True, True, True, True, False):
            state = "cleanup_receipt_pending"
        elif durable == (True, True, True, True, True):
            state = "archived_closed"
    if state is None:
        raise Singleton12ArchiveError(
            "Rotation artifacts form an impossible durable state."
        )
    if cleanup and temporaries:
        raise Singleton12ArchiveError(
            "Archived-closed state contains stale archive temporaries."
        )
    return {
        "execution_id": paths.execution_id,
        "state": state,
        "source_present": source,
        "retiring_present": retiring,
        "archive_present": archive,
        "archive_manifest_present": manifest,
        "archive_closure_present": closure,
        "rotation_intent_present": intent,
        "cleanup_receipt_present": cleanup,
        "stale_archive_temporaries": [row.name for row in temporaries],
    }


def complete_safe_tree_rotation(
    *,
    paths: CellArchivePaths,
    source_member_prefix: str,
    authority_metadata: Mapping[str, Any],
    cell_metadata: Mapping[str, Any],
    rotation_authority: Mapping[str, Any],
    limits: ArchiveLimits,
    completed_at_utc: str | None = None,
) -> dict[str, Any]:
    """Resume and complete only the authenticated exact-tree rotation."""

    runtime = _require_runtime(paths)
    _require_no_archive_temporaries(paths)
    authority = _normalized_mapping(authority_metadata, label="authority metadata")
    cell = _normalized_mapping(cell_metadata, label="cell metadata")
    rotation = _validate_rotation_authority(rotation_authority)
    scan = _validate_cell_archive_details(
        paths.archive_path,
        expected_execution_id=paths.execution_id,
        expected_source_member_prefix=source_member_prefix,
        expected_authority_metadata=authority,
        expected_cell_metadata=cell,
        limits=limits,
    )
    _validate_external_manifest(paths, scan)
    closure = _load_valid_closure(
        paths=paths,
        scan=scan,
        authority_metadata=authority,
        cell_metadata=cell,
        source_member_prefix=source_member_prefix,
    )
    intent = _load_valid_intent(
        paths=paths,
        closure=closure,
        scan=scan,
        authority_metadata=authority,
        cell_metadata=cell,
        source_member_prefix=source_member_prefix,
        rotation_authority=rotation,
    )
    _ensure_operational_directory(
        paths.cleanup_receipt_path.parent, runtime_root=runtime
    )
    _ensure_operational_directory(paths.retiring_root.parent, runtime_root=runtime)

    with paths.rotation_intent_path.open("rb") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        if paths.cleanup_receipt_path.exists() or paths.cleanup_receipt_path.is_symlink():
            cleanup = _load_valid_cleanup(
                paths=paths,
                intent=intent,
                closure=closure,
                scan=scan,
                authority_metadata=authority,
                cell_metadata=cell,
                source_member_prefix=source_member_prefix,
            )
            if paths.source_root.exists() or paths.retiring_root.exists():
                raise Singleton12ArchiveError(
                    "Cleanup receipt exists while a direct tree remains."
                )
            return cleanup

        source_present = _path_presence(paths.source_root, expected="directory")
        retiring_present = _path_presence(
            paths.retiring_root, expected="directory"
        )
        resumed_retiring_removal = retiring_present and not source_present
        if source_present and retiring_present:
            raise Singleton12ArchiveError(
                "Both source and retiring trees exist; rotation is ambiguous."
            )
        if source_present:
            _tree_matches_manifest(
                paths,
                root=paths.source_root,
                source_member_prefix=source_member_prefix,
                manifest=scan.manifest,
            )
            if paths.retiring_root.exists() or paths.retiring_root.is_symlink():
                raise Singleton12ArchiveError(
                    "Retiring destination appeared before rename."
                )
            os.rename(paths.source_root, paths.retiring_root)
            _fsync_directory(paths.source_root.parent)
            _fsync_directory(paths.retiring_root.parent)
            retiring_present = True
        if retiring_present:
            if resumed_retiring_removal:
                _remaining_tree_is_manifest_subset(
                    root=paths.retiring_root,
                    source_member_prefix=source_member_prefix,
                    manifest=scan.manifest,
                )
            else:
                _tree_matches_manifest(
                    paths,
                    root=paths.retiring_root,
                    source_member_prefix=source_member_prefix,
                    manifest=scan.manifest,
                )
            if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
                raise Singleton12ArchiveError(
                    "Platform lacks symlink-resistant safe-tree removal."
                )
            shutil.rmtree(paths.retiring_root)
            _fsync_directory(paths.retiring_root.parent)
        if paths.source_root.exists() or paths.source_root.is_symlink():
            raise Singleton12ArchiveError("Direct source tree still exists.")
        if paths.retiring_root.exists() or paths.retiring_root.is_symlink():
            raise Singleton12ArchiveError("Retiring source tree still exists.")

        cleanup = digested(
            {
                "schema": CLEANUP_SCHEMA,
                "status": "passed_exact_safe_tree_removed_archive_retained",
                "completed_at_utc": _utc_timestamp(
                    completed_at_utc, label="cleanup timestamp"
                ),
                "execution_id": paths.execution_id,
                "authority_metadata": authority,
                "cell_metadata": cell,
                "source_member_prefix": source_member_prefix,
                "archive": _expected_archive_binding(paths, scan.validation),
                "archive_closure": _relative_binding(
                    paths,
                    paths.archive_closure_path,
                    canonical_digest=str(closure["sha256"]),
                ),
                "rotation_intent": _relative_binding(
                    paths,
                    paths.rotation_intent_path,
                    canonical_digest=str(intent["sha256"]),
                ),
                "removed_source_tree": _tree_summary(
                    scan.manifest["source_tree"]
                ),
                "direct_source_absent": True,
                "retiring_source_absent": True,
                "archive_retained": True,
            }
        )
        return _publish_json_idempotent(
            paths.cleanup_receipt_path, cleanup, label="cleanup receipt"
        )


def validate_archive_backed_closure(
    *,
    paths: CellArchivePaths,
    source_member_prefix: str,
    expected_authority_metadata: Mapping[str, Any],
    expected_cell_metadata: Mapping[str, Any],
    limits: ArchiveLimits,
    expected_rotation_authority: Mapping[str, Any] | None = None,
    require_cleanup: bool = True,
) -> dict[str, Any]:
    """Return terminal-ready bindings after a full archive/receipt rescan."""

    _require_runtime(paths)
    authority = _normalized_mapping(
        expected_authority_metadata, label="expected authority metadata"
    )
    cell = _normalized_mapping(
        expected_cell_metadata, label="expected cell metadata"
    )
    rotation = (
        _validate_rotation_authority(expected_rotation_authority)
        if expected_rotation_authority is not None
        else None
    )
    scan = _validate_cell_archive_details(
        paths.archive_path,
        expected_execution_id=paths.execution_id,
        expected_source_member_prefix=source_member_prefix,
        expected_authority_metadata=authority,
        expected_cell_metadata=cell,
        limits=limits,
    )
    manifest = _validate_external_manifest(paths, scan)
    closure = _load_valid_closure(
        paths=paths,
        scan=scan,
        authority_metadata=authority,
        cell_metadata=cell,
        source_member_prefix=source_member_prefix,
    )
    intent: dict[str, Any] | None = None
    cleanup: dict[str, Any] | None = None
    if paths.rotation_intent_path.exists() or paths.rotation_intent_path.is_symlink():
        intent = _load_valid_intent(
            paths=paths,
            closure=closure,
            scan=scan,
            authority_metadata=authority,
            cell_metadata=cell,
            source_member_prefix=source_member_prefix,
            rotation_authority=rotation,
        )
    elif require_cleanup:
        raise Singleton12ArchiveError("Rotation intent is absent.")
    if require_cleanup:
        if intent is None:
            raise Singleton12ArchiveError("Rotation intent is absent.")
        cleanup = _load_valid_cleanup(
            paths=paths,
            intent=intent,
            closure=closure,
            scan=scan,
            authority_metadata=authority,
            cell_metadata=cell,
            source_member_prefix=source_member_prefix,
        )
        state = inspect_rotation_state(paths)
        if state["state"] != "archived_closed":
            raise Singleton12ArchiveError(
                "Cleanup receipt does not yield archived-closed state."
            )
        _require_no_archive_temporaries(paths)
    elif paths.source_root.exists() and not paths.source_root.is_symlink():
        _tree_matches_manifest(
            paths,
            root=paths.source_root,
            source_member_prefix=source_member_prefix,
            manifest=manifest,
        )

    result: dict[str, Any] = {
        "schema": ARCHIVE_BACKED_CLOSURE_SCHEMA,
        "status": (
            "passed_archive_backed_terminal_closure"
            if require_cleanup
            else "passed_archive_closure_cleanup_not_required"
        ),
        "execution_id": paths.execution_id,
        "authority_metadata": authority,
        "cell_metadata": cell,
        "source_member_prefix": source_member_prefix,
        "archive_validation": scan.validation,
        "archive": _expected_archive_binding(paths, scan.validation),
        "archive_manifest": _relative_binding(
            paths,
            paths.archive_manifest_path,
            canonical_digest=str(manifest["sha256"]),
        ),
        "archive_closure": _relative_binding(
            paths,
            paths.archive_closure_path,
            canonical_digest=str(closure["sha256"]),
        ),
        "source_tree": _tree_summary(manifest["source_tree"]),
        "direct_source_absent": not (
            paths.source_root.exists() or paths.source_root.is_symlink()
        ),
        "retiring_source_absent": not (
            paths.retiring_root.exists() or paths.retiring_root.is_symlink()
        ),
        "cleanup_required": require_cleanup,
    }
    if intent is not None:
        result["rotation_intent"] = _relative_binding(
            paths,
            paths.rotation_intent_path,
            canonical_digest=str(intent["sha256"]),
        )
    if cleanup is not None:
        result["cleanup_receipt"] = _relative_binding(
            paths,
            paths.cleanup_receipt_path,
            canonical_digest=str(cleanup["sha256"]),
        )
    return digested(result)
