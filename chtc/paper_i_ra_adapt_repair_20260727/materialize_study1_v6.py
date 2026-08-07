"""Atomically materialize and audit the immutable Paper-I Study-1 v6 bundles.

This command has no execution or scheduler seam.  It:

* anchors and snapshots immutable v1-v5 materializations;
* byte-copies the v5 source-materialization tree without rebasing the
  historical v5 paths embedded in those provenance records;
* materializes both v6 Study-1 bundles into a private sibling staging tree;
* validates the staged bundles, all 116 protocol loader bindings, and an
  allowlisted v5-to-v6 semantic diff;
* atomically renames the complete staged tree to the v6 destination;
* repeats all 116 loader checks from the final paths; and
* writes self-digested cross-loader and final receipts.

Any failed check raises before a final passed receipt exists.  The command
never overwrites v6, deletes a failed staging tree, executes a scientific
cell, submits a scheduler job, or authorizes execution.
"""

from __future__ import annotations

import copy
import ctypes
import errno
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pipelines.contracts.problem import ProblemRequest
from pipelines.static_adapt.builders.problem_registry import (
    resolve_problem_context,
)
from pipelines.static_adapt.ra_adapt.adapters import (
    SinglePauliWordCandidateAdapter,
)
from pipelines.static_adapt.ra_adapt.bundles import (
    MEASURED_BUNDLE_ID,
    STATIONARY_BUNDLE_ID,
    STUDY1_BUNDLE_POLICIES,
    _implementation_source_inventory,
    load_validated_bundle_protocol,
    materialize_study1_bundles,
    preservation_execution_gate_contract,
    study1_shared_execution_dedupe_contract,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    APPEND_CONVENTIONAL_SELECTOR_SCOPE,
    LOGICAL_SHARED_REDUCED_REFIT_BASE_CHART,
    NATIVE_REFIT_CHART,
    canonical_json_bytes,
    canonical_sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_ROOT = REPO_ROOT / "chtc" / "paper_i_ra_adapt_repair_20260727"
MATERIALIZATIONS_ROOT = CAMPAIGN_ROOT / "bundles" / "materializations"
V2_ROOT = MATERIALIZATIONS_ROOT / "ra_adapt_unification_post_refactor_v2"
V3_ROOT = MATERIALIZATIONS_ROOT / "ra_adapt_unification_post_refactor_v3"
V4_ROOT = MATERIALIZATIONS_ROOT / "ra_adapt_unification_post_refactor_v4"
V5_ROOT = MATERIALIZATIONS_ROOT / "ra_adapt_unification_post_refactor_v5"
V6_ROOT = MATERIALIZATIONS_ROOT / "ra_adapt_unification_post_refactor_v6"
V5_SOURCE_ROOT = V5_ROOT / "source_materialization"
SOURCE_LOCKS_INPUT = V5_SOURCE_ROOT / "source_locks_input.json"
PROBLEM_BASELINES = V5_SOURCE_ROOT / "problem_baselines.json"
V5_FINAL_RECEIPT = V5_ROOT / "final_materialization_receipt.json"

V1_ROOTS = (
    CAMPAIGN_ROOT / "bundles" / "source_materialization",
    CAMPAIGN_ROOT / "bundles" / STATIONARY_BUNDLE_ID,
    CAMPAIGN_ROOT / "bundles" / MEASURED_BUNDLE_ID,
)
HISTORICAL_ROOTS: dict[str, tuple[Path, ...]] = {
    "v1": V1_ROOTS,
    "v2": (V2_ROOT,),
    "v3": (V3_ROOT,),
    "v4": (V4_ROOT,),
    "v5": (V5_ROOT,),
}

# These hashes anchor the exact immutable inputs audited before this script was
# introduced.  The current implementation inventory is intentionally *not*
# pinned: it is computed at runtime and must remain identical at every gate.
EXPECTED_HISTORICAL_TREE_SHA256 = {
    "v1": "a3026399cc25f25cd41c61aa43744fa2233364ae883cdd298fbbb3fbcf08342d",
    "v2": "294497dbb505a1e7949a41ad55b742e82869c7a2e8134bdf353c2ca044dd6f83",
    "v3": "80c1e4012f2c9f1b3dea0272df0adfa9ab8b039fee55d082b7b279b8b1b88777",
    "v4": "c944c40ed009e308956e71563ec1cf4eaf129018d7ce9cc0ad2c36276f44f307",
    "v5": "19790d2ac3fc1ebaef1b4e0a75097e1b0d791d7c3df2a96f28d618fe8b849d1a",
}
EXPECTED_HISTORICAL_FILE_COUNTS = {
    "v1": 364,
    "v2": 123,
    "v3": 369,
    "v4": 127,
    "v5": 369,
}
EXPECTED_V5_SOURCE_RELATIVE_TREE_SHA256 = (
    "08ac8303870ba206493859e2037e53f0e32f758f235c06084681b149073e7c26"
)
EXPECTED_V5_SOURCE_FILE_COUNT = 126
EXPECTED_V5_FINAL_RECEIPT_FILE_SHA256 = (
    "10c6fb37c540efbea5ece2a42e638a38106843710409867fe1644300a71fcda8"
)
EXPECTED_SOURCE_LOCKS_INPUT_FILE_SHA256 = (
    "bee791d98a008e604e053ff07a6ce55117c448d32ffec6c747b1771c2e7c4fba"
)
EXPECTED_PROBLEM_BASELINES_FILE_SHA256 = (
    "a12a36c3f2c8bfe74e4c8a0c9db1d1baecf3b100b00480c5386e903d973c4015"
)

EXPECTED_BUNDLE_CELL_COUNT = 58
EXPECTED_VALIDATION_CELL_COUNT = 10
EXPECTED_FULL_CELL_COUNT = 48
EXPECTED_SOURCE_LOCK_COUNT = 50
EXPECTED_GLOBAL_SOURCE_COUNT = 4
EXPECTED_CROSS_LOADER_COUNT = 116
DARWIN_RENAME_EXCL = 0x00000004
DARWIN_O_NOFOLLOW_ANY = 0x20000000
EXPECTED_VALIDATION_CHECK_IDS = {
    "bundle_schema_and_digest",
    "finite_cell_matrix",
    "source_locks_exact_bytes",
    "validation_horizon",
    "resolved_protocol_contracts",
    "macro_pool_hash_equality",
    "singleton_pool_exposure_contracts",
    "study1_append_shared_execution_dedupe",
    "protocol_execution_separation",
    "paper_i_run_materialization_gate",
}
PRESERVATION_GATE_ID = "preservation_policy_semantics"

OLD_APPEND_REFIT_CHART = "supported_fs_whitened_fixed_v1"
OLD_APPEND_BASE_CHART = "expanded_runtime_projected_logical_v1"
OLD_ALWAYS_INSERTION_KIND = "plateau_commutation"
NEW_ALWAYS_INSERTION_KIND = "full_commutation"
EXPECTED_V5_APPEND_MODULE_SHA256 = (
    "9b263b890724d5270252da2fc9da6f746e62947677409ba5e9991b70df3fe641"
)
EXPECTED_V5_POOL_MODULE_SHA256 = (
    "9314321c157de412a18bdfdba6589d3cc632d3554cbcb978fb629e405e8a9cd3"
)

# The repaired global hard-guard child identities are scientific invariants,
# not provenance ripple.  A runtime pool smoke must reproduce them exactly.
EXPECTED_SINGLETON_CHILD_IDENTITIES = {
    "strong_weak_u8": {
        "parent_count": 123,
        "parent_ordered_labels_sha256": (
            "17cc97b744f8e6b50b686b24edd28426ca2c055bc2c31054fd353ddfa10efbe3"
        ),
        "parent_ordered_pool_sha256": (
            "468cb94dacac1d4986f3700910f216f1d4db64d14da7560ea64c7aee18f2406b"
        ),
        "child_count": 948,
        "child_ordered_labels_sha256": (
            "02995a2c570d4322e46e55e3a532381ff7eff85dc3c2de8cb2b30ed888b76906"
        ),
        "child_ordered_pool_sha256": (
            "62a24f68adc8a71f78fa5d3afb28356d15b988a2003e1c97e69871a65726e90c"
        ),
        "child_receipt_sha256": (
            "8defe6db83a18a4ee2aba57bd44193a954a7661065ce0366485dae274a854a24"
        ),
        "shared_contract_ordered_pool_sha256": (
            "3c6f9cc30ae31e1aee418a47f0c9bf5d06c4d7c7a0f7ee3a60837de2e176cb5b"
        ),
        "historical_source_runtime_ordered_pool_sha256": (
            "442c7bdd582f84378ce051cb526f0ff2264e8e43a1cd0903793350eaf6248618"
        ),
    },
    "strong_strong_u8": {
        "parent_count": 123,
        "parent_ordered_labels_sha256": (
            "17cc97b744f8e6b50b686b24edd28426ca2c055bc2c31054fd353ddfa10efbe3"
        ),
        "parent_ordered_pool_sha256": (
            "45cb63e861747c84f67d50bde328f106e2f79e484cc4e14f6ad25ef519ce7e3b"
        ),
        "child_count": 948,
        "child_ordered_labels_sha256": (
            "02995a2c570d4322e46e55e3a532381ff7eff85dc3c2de8cb2b30ed888b76906"
        ),
        "child_ordered_pool_sha256": (
            "dfba967f333bb8356d9a2e018745f5297e13cba25d7a8e127ab10d189178b654"
        ),
        "child_receipt_sha256": (
            "36060620c8822efd8059afc92ec8a47ebc945eaef046b6c029539dca8fd288af"
        ),
        "shared_contract_ordered_pool_sha256": (
            "5d6d58c663bd997960a9ff4cf802679b47aab2c0a5feb548576c3bc4612cb9e1"
        ),
        "historical_source_runtime_ordered_pool_sha256": (
            "b80bc6eee2bf63449f2ece1e2f27ed24a54fd54188d78ece7b8365a443bef77f"
        ),
    },
}


class MaterializationAuditError(RuntimeError):
    """Raised when the immutable v6 materialization contract is not met."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaterializationAuditError(
            f"Could not load required JSON object: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise MaterializationAuditError(
            f"Expected a JSON object at {path}."
        )
    return payload


def _verify_self_digest(payload: Mapping[str, Any], *, label: str) -> None:
    observed = str(payload.get("sha256", ""))
    unsigned = dict(payload)
    unsigned.pop("sha256", None)
    expected = canonical_sha256(unsigned)
    if observed != expected:
        raise MaterializationAuditError(
            f"{label} self-digest mismatch: {observed} != {expected}."
        )


def _load_canonical_digested(path: Path, *, label: str) -> dict[str, Any]:
    payload = _load_mapping(path)
    if path.read_bytes() != canonical_json_bytes(payload) + b"\n":
        raise MaterializationAuditError(
            f"{label} is not canonical JSON with one trailing newline: {path}"
        )
    _verify_self_digest(payload, label=label)
    return payload


def _digested(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("sha256", None)
    result["sha256"] = canonical_sha256(result)
    return result


def _darwin_renameatx_np() -> Any:
    if sys.platform != "darwin":
        raise MaterializationAuditError(
            "Atomic no-replace publication requires macOS renameatx_np with "
            "RENAME_EXCL; unsupported platforms fail closed."
        )
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameatx_np = libc.renameatx_np
    except (AttributeError, OSError) as exc:
        raise MaterializationAuditError(
            "This macOS runtime does not expose renameatx_np; refusing a "
            "non-exclusive publication fallback."
        ) from exc
    renameatx_np.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameatx_np.restype = ctypes.c_int
    return renameatx_np


def _atomic_rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename one same-directory entry without replacement."""

    source_parent = source.parent.resolve()
    destination_parent = destination.parent.resolve()
    if source_parent != destination_parent:
        raise MaterializationAuditError(
            "Atomic no-replace rename requires source and destination to "
            f"share one parent: {source} -> {destination}."
        )
    renameatx_np = _darwin_renameatx_np()
    open_flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_CLOEXEC
        | DARWIN_O_NOFOLLOW_ANY
    )
    parent_descriptor = os.open(source_parent, open_flags)
    try:
        ctypes.set_errno(0)
        result = renameatx_np(
            parent_descriptor,
            os.fsencode(source.name),
            parent_descriptor,
            os.fsencode(destination.name),
            DARWIN_RENAME_EXCL,
        )
        if result != 0:
            error_number = ctypes.get_errno()
            if error_number == errno.EEXIST:
                raise FileExistsError(
                    error_number,
                    f"Refusing to replace existing path: {destination}",
                    str(destination),
                )
            if error_number in {
                errno.ENOTSUP,
                errno.EOPNOTSUPP,
                errno.ENOSYS,
                errno.EXDEV,
            }:
                raise MaterializationAuditError(
                    "Atomic directory-relative RENAME_EXCL is unavailable "
                    "for this runtime or filesystem; refusing a "
                    "non-exclusive fallback."
                )
            raise OSError(
                error_number,
                (
                    "Atomic no-replace rename failed for "
                    f"{source} -> {destination}: "
                    f"{os.strerror(error_number)}"
                ),
                str(destination),
            )
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _write_bytes_atomic_no_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _atomic_rename_no_replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    result = _digested(payload)
    _write_bytes_atomic_no_replace(
        path, canonical_json_bytes(result) + b"\n"
    )
    persisted = _load_canonical_digested(
        path, label=f"new receipt {path.name}"
    )
    if persisted != result:
        raise MaterializationAuditError(
            f"New receipt changed during publication: {path}"
        )
    return result


def _git_output(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository_state() -> dict[str, Any]:
    return {
        "git_commit": _git_output("rev-parse", "HEAD"),
        "dirty_working_tree": bool(
            _git_output("status", "--porcelain", "--untracked-files=normal")
        ),
        "cwd": str(REPO_ROOT),
    }


def _iter_regular_files(roots: Sequence[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            raise MaterializationAuditError(
                f"Required immutable tree is missing: {root}"
            )
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise MaterializationAuditError(
                    f"Immutable tree contains a symlink: {path}"
                )
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen:
                raise MaterializationAuditError(
                    f"Immutable snapshot contains a duplicate file: {path}"
                )
            seen.add(resolved)
            yield path


def _snapshot_roots(
    roots: Sequence[Path],
    *,
    relative_to: Path = REPO_ROOT,
) -> dict[str, Any]:
    files = [
        {
            "path": path.relative_to(relative_to).as_posix(),
            "sha256": _hash_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in _iter_regular_files(roots)
    ]
    return {
        "file_count": len(files),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in files),
        "tree_sha256": canonical_sha256(files),
        "files": files,
    }


def _snapshot_historical_materializations() -> dict[str, dict[str, Any]]:
    return {
        revision: _snapshot_roots(roots)
        for revision, roots in HISTORICAL_ROOTS.items()
    }


def _snapshot_summary(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "file_count": int(snapshot["file_count"]),
        "total_size_bytes": int(snapshot["total_size_bytes"]),
        "tree_sha256": str(snapshot["tree_sha256"]),
    }


def _assert_historical_anchors(
    snapshots: Mapping[str, Mapping[str, Any]],
) -> None:
    for revision in HISTORICAL_ROOTS:
        snapshot = snapshots.get(revision)
        if not isinstance(snapshot, Mapping):
            raise MaterializationAuditError(
                f"Missing immutable snapshot for {revision}."
            )
        if int(snapshot["file_count"]) != EXPECTED_HISTORICAL_FILE_COUNTS[
            revision
        ]:
            raise MaterializationAuditError(
                f"{revision} immutable file count drifted."
            )
        if str(snapshot["tree_sha256"]) != (
            EXPECTED_HISTORICAL_TREE_SHA256[revision]
        ):
            raise MaterializationAuditError(
                f"{revision} immutable tree hash drifted."
            )
    pinned_files = {
        V5_FINAL_RECEIPT: EXPECTED_V5_FINAL_RECEIPT_FILE_SHA256,
        SOURCE_LOCKS_INPUT: EXPECTED_SOURCE_LOCKS_INPUT_FILE_SHA256,
        PROBLEM_BASELINES: EXPECTED_PROBLEM_BASELINES_FILE_SHA256,
    }
    for path, expected in pinned_files.items():
        observed = _hash_file(path)
        if observed != expected:
            raise MaterializationAuditError(
                f"Pinned v5 input drifted: {path} ({observed} != {expected})."
            )


def _assert_snapshots_equal(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
    *,
    label: str,
) -> None:
    if before != after:
        changed = [
            revision
            for revision in HISTORICAL_ROOTS
            if before.get(revision) != after.get(revision)
        ]
        raise MaterializationAuditError(
            f"{label} modified immutable materializations: {changed}."
        )


def _copy_and_verify_source_materialization(
    staging_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_before = _snapshot_roots(
        (V5_SOURCE_ROOT,), relative_to=V5_SOURCE_ROOT
    )
    if (
        int(source_before["file_count"]) != EXPECTED_V5_SOURCE_FILE_COUNT
        or str(source_before["tree_sha256"])
        != EXPECTED_V5_SOURCE_RELATIVE_TREE_SHA256
    ):
        raise MaterializationAuditError(
            "The v5 source-materialization tree no longer matches its "
            "audited byte identity."
        )
    destination = staging_root / "source_materialization"
    shutil.copytree(V5_SOURCE_ROOT, destination, copy_function=shutil.copy2)
    source_after = _snapshot_roots((destination,), relative_to=destination)
    if source_before != source_after:
        raise MaterializationAuditError(
            "The inherited source-materialization copy is not byte-identical."
        )
    return source_before, source_after


def _snapshot_inherited_source(materialization_root: Path) -> dict[str, Any]:
    source_root = materialization_root / "source_materialization"
    return _snapshot_roots((source_root,), relative_to=source_root)


def _assert_inherited_source_unchanged(
    observed: Mapping[str, Any],
    pinned_v5: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if observed != pinned_v5:
        raise MaterializationAuditError(
            f"{label} inherited source tree drifted: "
            f"{_snapshot_summary(observed)!r} != "
            f"{_snapshot_summary(pinned_v5)!r}."
        )


def _problem_resolver_from(baselines: Mapping[str, Any]):
    regimes = baselines.get("regimes")
    if not isinstance(regimes, Mapping):
        raise MaterializationAuditError(
            "Problem baselines have no regime mapping."
        )

    def resolve(regime_id: str, nph: int):
        regime = regimes.get(regime_id)
        if not isinstance(regime, Mapping):
            raise KeyError(f"Unknown Paper-I regime {regime_id!r}.")
        physics = regime.get("physics")
        if not isinstance(physics, Mapping):
            raise MaterializationAuditError(
                f"Regime {regime_id!r} has no physics baseline."
            )
        return resolve_problem_context(
            ProblemRequest(
                problem_key="hh",
                num_sites=int(physics["L"]),
                t=float(physics["t"]),
                u=float(physics["u"]),
                dv=float(physics["dv"]),
                omega0=float(physics["omega0"]),
                g_ep=float(physics["g_ep"]),
                n_ph_max=int(nph),
                boson_encoding=str(physics["boson_encoding"]),
                ordering=str(physics["ordering"]),
                boundary=str(physics["boundary"]),
                include_zero_point=bool(
                    physics["include_zero_point"]
                ),
                v_nn=0.0,
                t_prime=0.0,
                n_fermions=None,
            )
        )

    return resolve


def _assert_equal(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected:
        raise MaterializationAuditError(
            f"{label} drifted: {actual!r} != {expected!r}."
        )


def _assert_unsubmitted(payload: Mapping[str, Any], *, label: str) -> None:
    _assert_equal(
        payload.get("execution_authorized"), False, label=f"{label}.execution"
    )
    _assert_equal(
        payload.get("submission_state"),
        "not_submitted",
        label=f"{label}.submission_state",
    )
    _assert_equal(
        payload.get("submitted"), False, label=f"{label}.submitted"
    )


def _validate_bundle_surface(
    bundle_root: Path,
    *,
    expected_implementation_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = _load_canonical_digested(
        bundle_root / "bundle_manifest.json", label="bundle manifest"
    )
    source_locks = _load_canonical_digested(
        bundle_root / "source_locks.json", label="source-lock manifest"
    )
    expected = _load_canonical_digested(
        bundle_root / "expected_artifacts.json",
        label="expected-artifact index",
    )
    validation = _load_canonical_digested(
        bundle_root / "validation_report.json",
        label="bundle validation report",
    )
    bundle_id = bundle_root.name
    _assert_equal(manifest.get("bundle_id"), bundle_id, label="bundle id")
    _assert_equal(
        int(manifest.get("cell_count", -1)),
        EXPECTED_BUNDLE_CELL_COUNT,
        label=f"{bundle_id}.cell_count",
    )
    _assert_equal(
        int(manifest.get("validation_cell_count", -1)),
        EXPECTED_VALIDATION_CELL_COUNT,
        label=f"{bundle_id}.validation_cell_count",
    )
    _assert_equal(
        int(manifest.get("full_cell_count", -1)),
        EXPECTED_FULL_CELL_COUNT,
        label=f"{bundle_id}.full_cell_count",
    )
    cells = manifest.get("cells")
    if not isinstance(cells, list):
        raise MaterializationAuditError(f"{bundle_id} has no cell list.")
    _assert_equal(len(cells), EXPECTED_BUNDLE_CELL_COUNT, label="cell list")
    stage_counts = {
        "validation": sum(row.get("stage") == "validation" for row in cells),
        "full": sum(row.get("stage") == "full" for row in cells),
    }
    _assert_equal(
        stage_counts,
        {
            "validation": EXPECTED_VALIDATION_CELL_COUNT,
            "full": EXPECTED_FULL_CELL_COUNT,
        },
        label=f"{bundle_id}.stage_counts",
    )
    cell_ids = [str(row.get("cell_id", "")) for row in cells]
    _assert_equal(
        len(set(cell_ids)), EXPECTED_BUNDLE_CELL_COUNT, label="unique cell ids"
    )
    protocol_files = sorted((bundle_root / "protocols").glob("*.json"))
    template_files = sorted(
        (bundle_root / "execution_templates").glob("*.json")
    )
    _assert_equal(
        len(protocol_files), EXPECTED_BUNDLE_CELL_COUNT, label="protocol count"
    )
    _assert_equal(
        len(template_files), EXPECTED_BUNDLE_CELL_COUNT, label="template count"
    )
    _assert_equal(
        {path.stem for path in protocol_files}, set(cell_ids), label="protocol set"
    )
    _assert_equal(
        {path.stem for path in template_files}, set(cell_ids), label="template set"
    )
    _assert_unsubmitted(manifest, label=f"{bundle_id}.manifest")
    _assert_unsubmitted(validation, label=f"{bundle_id}.validation")
    _assert_equal(
        validation.get("materialization_status"),
        "passed",
        label=f"{bundle_id}.materialization_status",
    )
    raw_checks = validation.get("checks")
    if not isinstance(raw_checks, list):
        raise MaterializationAuditError(
            f"{bundle_id} validation report has no checks."
        )
    checks = {
        str(row.get("id")): row
        for row in raw_checks
        if isinstance(row, Mapping)
    }
    _assert_equal(
        set(checks), EXPECTED_VALIDATION_CHECK_IDS, label="validation check set"
    )
    if any(row.get("status") != "passed" for row in checks.values()):
        raise MaterializationAuditError(
            f"{bundle_id} has a non-passed materialization check."
        )
    dedupe = manifest.get("study1_shared_execution_dedupe")
    _assert_equal(
        dedupe,
        study1_shared_execution_dedupe_contract(),
        label=f"{bundle_id}.Study-1 dedupe contract",
    )
    _assert_equal(
        source_locks.get("all_required_files_verified"),
        True,
        label=f"{bundle_id}.source verification",
    )
    _assert_equal(
        int(source_locks.get("required_cell_lock_count", -1)),
        EXPECTED_SOURCE_LOCK_COUNT,
        label=f"{bundle_id}.required source-lock count",
    )
    _assert_equal(
        len(source_locks.get("cell_locks", {})),
        EXPECTED_SOURCE_LOCK_COUNT,
        label=f"{bundle_id}.cell source-lock count",
    )
    _assert_equal(
        len(source_locks.get("global_sources", {})),
        EXPECTED_GLOBAL_SOURCE_COUNT,
        label=f"{bundle_id}.global source count",
    )
    _assert_equal(
        source_locks.get("implementation_sources"),
        expected_implementation_inventory,
        label=f"{bundle_id}.implementation inventory",
    )
    _assert_equal(
        int(expected.get("cell_count", -1)),
        EXPECTED_BUNDLE_CELL_COUNT,
        label=f"{bundle_id}.expected-artifact count",
    )
    _assert_equal(
        set(expected.get("cells", {})), set(cell_ids), label="expected cell set"
    )
    return {
        "bundle_id": bundle_id,
        "bundle_manifest_sha256": manifest["sha256"],
        "source_locks_sha256": source_locks["sha256"],
        "expected_artifacts_sha256": expected["sha256"],
        "validation_report_sha256": validation["sha256"],
        "cell_count": EXPECTED_BUNDLE_CELL_COUNT,
        "validation_cell_count": EXPECTED_VALIDATION_CELL_COUNT,
        "full_cell_count": EXPECTED_FULL_CELL_COUNT,
        "protocol_count": len(protocol_files),
        "execution_template_count": len(template_files),
        "materialization_status": "passed",
        "execution_authorized": False,
        "submission_state": "not_submitted",
        "submitted": False,
    }


def _deep_difference_paths(left: Any, right: Any, path: str = "$") -> list[str]:
    if type(left) is not type(right):
        return [path]
    if isinstance(left, Mapping):
        paths: list[str] = []
        for key in sorted(set(left) | set(right), key=str):
            child = f"{path}.{key}"
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(_deep_difference_paths(left[key], right[key], child))
        return paths
    if isinstance(left, list):
        if len(left) != len(right):
            return [f"{path}.length"]
        paths = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            paths.extend(
                _deep_difference_paths(
                    left_item, right_item, f"{path}[{index}]"
                )
            )
        return paths
    return [] if left == right else [path]


def _require_normalized_equal(
    old: Any,
    new: Any,
    *,
    label: str,
) -> None:
    if old != new:
        paths = _deep_difference_paths(old, new)[:25]
        raise MaterializationAuditError(
            f"Unexpected normalized v5-to-v6 drift in {label}: {paths}."
        )


def _normalize_protocol_pair(
    old_payload: Mapping[str, Any],
    new_payload: Mapping[str, Any],
    *,
    cell_id: str,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    old = copy.deepcopy(dict(old_payload))
    new = copy.deepcopy(dict(new_payload))
    allowed: list[str] = []
    for payload in (old, new):
        payload.pop("sha256", None)
        payload.pop("bundle_manifest_sha256", None)
        materialization = payload.get("bundle_materialization")
        if not isinstance(materialization, dict):
            raise MaterializationAuditError(
                f"{cell_id} has no bundle-materialization receipt."
            )
        for key in (
            "sha256",
            "bundle_manifest_sha256",
            "source_locks_sha256",
            "source_lock_refs_sha256",
        ):
            materialization.pop(key, None)
        locks = payload.get("source_locks")
        if not isinstance(locks, dict):
            raise MaterializationAuditError(
                f"{cell_id} has no protocol source-lock mapping."
            )
        for key in (
            "implementation_source_inventory_sha256",
            "source_locks_manifest_sha256",
        ):
            locks[key] = "<materialization-provenance>"
    allowed.extend(
        [
            "$.sha256",
            "$.bundle_manifest_sha256",
            "$.bundle_materialization.{sha256,bundle_manifest_sha256,"
            "source_locks_sha256,source_lock_refs_sha256}",
            "$.source_locks.{implementation_source_inventory_sha256,"
            "source_locks_manifest_sha256}",
        ]
    )
    if cell_id.endswith("__append_macro"):
        old_locks = old["source_locks"]
        new_locks = new["source_locks"]
        implementation_module_repairs = {
            "append_module_sha256": (
                EXPECTED_V5_APPEND_MODULE_SHA256,
                REPO_ROOT
                / "pipelines"
                / "static_adapt"
                / "ra_adapt"
                / "append.py",
            ),
            "pool_module_sha256": (
                EXPECTED_V5_POOL_MODULE_SHA256,
                REPO_ROOT
                / "pipelines"
                / "static_adapt"
                / "ra_adapt"
                / "pools.py",
            ),
        }
        for key, (expected_old, current_path) in (
            implementation_module_repairs.items()
        ):
            _assert_equal(
                old_locks.get(key),
                expected_old,
                label=f"{cell_id}.v5 {key}",
            )
            expected_new = _hash_file(current_path)
            _assert_equal(
                new_locks.get(key),
                expected_new,
                label=f"{cell_id}.v6 {key}",
            )
            if expected_new == expected_old:
                raise MaterializationAuditError(
                    f"{cell_id} did not carry the expected implementation "
                    f"provenance repair for {key}."
                )
            old_locks[key] = "<implementation-module-repair>"
            new_locks[key] = "<implementation-module-repair>"
            allowed.append(f"$.source_locks.{key}")
        _assert_equal(
            old.get("accepted_refit_coordinate_chart"),
            OLD_APPEND_REFIT_CHART,
            label=f"{cell_id}.v5 Append refit chart",
        )
        _assert_equal(
            new.get("accepted_refit_coordinate_chart"),
            NATIVE_REFIT_CHART,
            label=f"{cell_id}.v6 Append refit chart",
        )
        _assert_equal(
            old.get("accepted_refit_base_chart_policy"),
            OLD_APPEND_BASE_CHART,
            label=f"{cell_id}.v5 Append base chart",
        )
        _assert_equal(
            new.get("accepted_refit_base_chart_policy"),
            LOGICAL_SHARED_REDUCED_REFIT_BASE_CHART,
            label=f"{cell_id}.v6 Append base chart",
        )
        if "selector_scope" in old:
            raise MaterializationAuditError(
                f"{cell_id} unexpectedly had selector_scope in v5."
            )
        _assert_equal(
            new.get("selector_scope"),
            APPEND_CONVENTIONAL_SELECTOR_SCOPE,
            label=f"{cell_id}.v6 selector_scope",
        )
        for payload in (old, new):
            payload.pop("accepted_refit_coordinate_chart", None)
            payload.pop("accepted_refit_base_chart_policy", None)
            payload.pop("selector_scope", None)
        old_lineage = old.get("lineage_authority")
        new_lineage = new.get("lineage_authority")
        if not isinstance(old_lineage, dict) or not isinstance(
            new_lineage, dict
        ):
            raise MaterializationAuditError(
                f"{cell_id} has no Append lineage authority."
            )
        if "selector_scope" in old_lineage:
            raise MaterializationAuditError(
                f"{cell_id} unexpectedly had lineage selector_scope in v5."
            )
        _assert_equal(
            new_lineage.pop("selector_scope", None),
            APPEND_CONVENTIONAL_SELECTOR_SCOPE,
            label=f"{cell_id}.lineage_authority.selector_scope",
        )
        old_route = old.get("route_contract")
        new_route = new.get("route_contract")
        if not isinstance(old_route, dict) or not isinstance(new_route, dict):
            raise MaterializationAuditError(
                f"{cell_id} has no Append route contract."
            )
        old_route.pop("sha256", None)
        new_route.pop("sha256", None)
        old_route_lineage = old_route.get("lineage_authority")
        new_route_lineage = new_route.get("lineage_authority")
        if not isinstance(old_route_lineage, dict) or not isinstance(
            new_route_lineage, dict
        ):
            raise MaterializationAuditError(
                f"{cell_id} has no Append route lineage authority."
            )
        if "selector_scope" in old_route_lineage:
            raise MaterializationAuditError(
                f"{cell_id} unexpectedly had route-lineage selector_scope "
                "in v5."
            )
        _assert_equal(
            new_route_lineage.pop("selector_scope", None),
            APPEND_CONVENTIONAL_SELECTOR_SCOPE,
            label=f"{cell_id}.route lineage selector_scope",
        )
        old_semantics = old_route.get("semantic_invariants")
        new_semantics = new_route.get("semantic_invariants")
        if not isinstance(old_semantics, dict) or not isinstance(
            new_semantics, dict
        ):
            raise MaterializationAuditError(
                f"{cell_id} has malformed Append semantic invariants."
            )
        additions = {
            "accepted_refit_coordinate_chart": NATIVE_REFIT_CHART,
            "accepted_refit_base_chart_policy": (
                LOGICAL_SHARED_REDUCED_REFIT_BASE_CHART
            ),
            "selector_scope": APPEND_CONVENTIONAL_SELECTOR_SCOPE,
        }
        for key, expected in additions.items():
            if key in old_semantics:
                raise MaterializationAuditError(
                    f"{cell_id} unexpectedly had route field {key} in v5."
                )
            _assert_equal(
                new_semantics.get(key),
                expected,
                label=f"{cell_id}.route_contract.{key}",
            )
            new_semantics.pop(key)
        allowed.extend(
            [
                "$.accepted_refit_coordinate_chart",
                "$.accepted_refit_base_chart_policy",
                "$.selector_scope",
                "$.lineage_authority.selector_scope",
                "$.route_contract.sha256",
                "$.route_contract.lineage_authority.selector_scope",
                "$.route_contract.semantic_invariants."
                "{accepted_refit_coordinate_chart,"
                "accepted_refit_base_chart_policy,selector_scope}",
            ]
        )

    if cell_id.endswith("__ra_macro_always"):
        try:
            old_insertion = old["request"]["method"]["insertion"]
            new_insertion = new["request"]["method"]["insertion"]
        except (KeyError, TypeError) as exc:
            raise MaterializationAuditError(
                f"{cell_id} has no typed insertion request."
            ) from exc
        _assert_equal(
            old_insertion.get("kind"),
            OLD_ALWAYS_INSERTION_KIND,
            label=f"{cell_id}.v5 insertion kind",
        )
        _assert_equal(
            new_insertion.get("kind"),
            NEW_ALWAYS_INSERTION_KIND,
            label=f"{cell_id}.v6 insertion kind",
        )
        old_insertion["kind"] = "<typed-full-insertion-repair>"
        new_insertion["kind"] = "<typed-full-insertion-repair>"
        allowed.append("$.request.method.insertion.kind")
    return old, new, allowed


def _normalize_manifest_pair(
    old_manifest: Mapping[str, Any],
    new_manifest: Mapping[str, Any],
    *,
    bundle_id: str,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    old = copy.deepcopy(dict(old_manifest))
    new = copy.deepcopy(dict(new_manifest))
    for payload in (old, new):
        payload.pop("sha256", None)
        payload.pop("materialization_timestamp", None)
        payload.pop("repository_state_at_materialization", None)
        payload.pop("environment_fingerprint", None)
        source_locks = payload.get("source_locks")
        if isinstance(source_locks, dict):
            source_locks["sha256"] = "<source-lock-provenance>"
    if "study1_shared_execution_dedupe" in old:
        raise MaterializationAuditError(
            f"{bundle_id} unexpectedly had the Study-1 dedupe contract in v5."
        )
    _assert_equal(
        new.get("study1_shared_execution_dedupe"),
        study1_shared_execution_dedupe_contract(),
        label=f"{bundle_id}.v6 dedupe contract",
    )
    new.pop("study1_shared_execution_dedupe")
    old_cells = old.get("cells")
    new_cells = new.get("cells")
    if not isinstance(old_cells, list) or not isinstance(new_cells, list):
        raise MaterializationAuditError(
            f"{bundle_id} has no ordered manifest cells."
        )
    _assert_equal(len(old_cells), len(new_cells), label="manifest cell count")
    active_gradient_policy = dict(STUDY1_BUNDLE_POLICIES)[bundle_id]
    for old_cell, new_cell in zip(old_cells, new_cells):
        if not isinstance(old_cell, dict) or not isinstance(new_cell, dict):
            raise MaterializationAuditError(
                f"{bundle_id} has a malformed manifest cell."
            )
        _assert_equal(
            old_cell.get("cell_id"),
            new_cell.get("cell_id"),
            label=f"{bundle_id}.ordered manifest cell identity",
        )
        if "preservation_execution_gate" in old_cell:
            raise MaterializationAuditError(
                f"{old_cell.get('cell_id')} unexpectedly had a v5 gate."
            )
        if old_cell.get("preservation_contract_id") is not None:
            _assert_equal(
                new_cell.pop("preservation_execution_gate", None),
                preservation_execution_gate_contract(
                    active_gradient_policy=active_gradient_policy
                ),
                label=(
                    f"{old_cell.get('cell_id')}.manifest preservation gate"
                ),
            )
        elif "preservation_execution_gate" in new_cell:
            raise MaterializationAuditError(
                f"Non-preservation cell {old_cell.get('cell_id')} has a gate."
            )
    old_progression = old.get("execution_progression_contract")
    new_progression = new.get("execution_progression_contract")
    if not isinstance(old_progression, dict) or not isinstance(
        new_progression, dict
    ):
        raise MaterializationAuditError(
            f"{bundle_id} has no progression contract."
        )
    old_gate_ids = list(old_progression.get("required_objective_gate_ids", ()))
    new_gate_ids = list(new_progression.get("required_objective_gate_ids", ()))
    _assert_equal(
        new_gate_ids,
        [*old_gate_ids, PRESERVATION_GATE_ID],
        label=f"{bundle_id}.preservation gate addition",
    )
    new_progression["required_objective_gate_ids"] = old_gate_ids
    return old, new, [
        "$.sha256",
        "$.materialization_timestamp",
        "$.repository_state_at_materialization",
        "$.environment_fingerprint",
        "$.source_locks.sha256",
        "$.study1_shared_execution_dedupe",
        "$.cells.*.preservation_execution_gate[when applicable]",
        "$.execution_progression_contract.required_objective_gate_ids"
        "[+preservation_policy_semantics]",
    ]


def _normalize_source_lock_pair(
    old_payload: Mapping[str, Any],
    new_payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    old = copy.deepcopy(dict(old_payload))
    new = copy.deepcopy(dict(new_payload))
    for payload in (old, new):
        payload.pop("sha256", None)
        payload.pop("implementation_sources", None)
    return old, new, ["$.sha256", "$.implementation_sources"]


def _normalize_expected_pair(
    old_payload: Mapping[str, Any],
    new_payload: Mapping[str, Any],
    *,
    bundle_id: str,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    old = copy.deepcopy(dict(old_payload))
    new = copy.deepcopy(dict(new_payload))
    old.pop("sha256", None)
    new.pop("sha256", None)
    old_cells = old.get("cells")
    new_cells = new.get("cells")
    if not isinstance(old_cells, dict) or not isinstance(new_cells, dict):
        raise MaterializationAuditError(
            f"{bundle_id} has malformed expected-artifact cells."
        )
    _assert_equal(set(old_cells), set(new_cells), label="expected cell ids")
    for cell_id in sorted(old_cells):
        old_cell = old_cells[cell_id]
        new_cell = new_cells[cell_id]
        if not isinstance(old_cell, dict) or not isinstance(new_cell, dict):
            raise MaterializationAuditError(
                f"Malformed expected-artifact row for {cell_id}."
            )
        if "execution_fulfillment" in old_cell:
            raise MaterializationAuditError(
                f"{cell_id} unexpectedly had execution fulfillment in v5."
            )
        fulfillment = new_cell.pop("execution_fulfillment", None)
        if not isinstance(fulfillment, Mapping):
            raise MaterializationAuditError(
                f"{cell_id} has no v6 execution-fulfillment contract."
            )
        fulfillment_kind = str(fulfillment.get("fulfillment_kind", ""))
        reference_fulfilled = (
            fulfillment_kind == "shared_result_reference_v1"
        )
        old_artifacts = old_cell.get("expected_run_artifacts")
        new_artifacts = new_cell.get("expected_run_artifacts")
        if not isinstance(old_artifacts, dict) or not isinstance(
            new_artifacts, dict
        ):
            raise MaterializationAuditError(
                f"{cell_id} has malformed expected run artifacts."
            )
        _assert_equal(
            set(old_artifacts),
            set(new_artifacts),
            label=f"{cell_id}.expected run artifact roles",
        )
        for role in sorted(old_artifacts):
            old_artifact = old_artifacts[role]
            new_artifact = new_artifacts[role]
            if not isinstance(old_artifact, dict) or not isinstance(
                new_artifact, dict
            ):
                raise MaterializationAuditError(
                    f"{cell_id}/{role} has a malformed artifact contract."
                )
            for key in (
                "fulfillment_kind",
                "direct_file_required",
                "reference_receipt_required",
            ):
                if key in old_artifact:
                    raise MaterializationAuditError(
                        f"{cell_id}/{role} unexpectedly had {key} in v5."
                    )
            _assert_equal(
                new_artifact.pop("fulfillment_kind", None),
                fulfillment_kind,
                label=f"{cell_id}/{role}.fulfillment_kind",
            )
            _assert_equal(
                new_artifact.pop("direct_file_required", None),
                not reference_fulfilled,
                label=f"{cell_id}/{role}.direct_file_required",
            )
            _assert_equal(
                new_artifact.pop("reference_receipt_required", None),
                reference_fulfilled,
                label=f"{cell_id}/{role}.reference_receipt_required",
            )
        if "preservation_execution_gate" in old_cell:
            raise MaterializationAuditError(
                f"{cell_id} unexpectedly had a preservation gate in v5."
            )
        if cell_id.endswith("__singleton_plateau"):
            expected_gate = preservation_execution_gate_contract(
                active_gradient_policy=(
                    dict(STUDY1_BUNDLE_POLICIES)[bundle_id]
                )
            )
            _assert_equal(
                new_cell.pop("preservation_execution_gate", None),
                expected_gate,
                label=f"{cell_id}.preservation gate",
            )
        elif "preservation_execution_gate" in new_cell:
            raise MaterializationAuditError(
                f"Non-preservation cell {cell_id} has a preservation gate."
            )
        for cell in (old_cell, new_cell):
            protocol = cell.get("protocol")
            template = cell.get("execution_template")
            if not isinstance(protocol, dict) or not isinstance(template, dict):
                raise MaterializationAuditError(
                    f"{cell_id} has malformed expected bindings."
                )
            protocol["sha256"] = "<protocol-digest>"
            template["sha256"] = "<execution-template-digest>"
    return old, new, [
        "$.sha256",
        "$.cells.*.protocol.sha256",
        "$.cells.*.execution_template.sha256",
        "$.cells.*.execution_fulfillment",
        "$.cells.*.expected_run_artifacts.*."
        "{fulfillment_kind,direct_file_required,reference_receipt_required}",
        "$.cells.*.preservation_execution_gate[when applicable]",
    ]


def _normalize_template_pair(
    old_payload: Mapping[str, Any],
    new_payload: Mapping[str, Any],
    *,
    cell_id: str,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    old = copy.deepcopy(dict(old_payload))
    new = copy.deepcopy(dict(new_payload))
    for payload in (old, new):
        payload.pop("sha256", None)
        protocol = payload.get("protocol")
        locks = payload.get("input_source_lock_hashes")
        if not isinstance(protocol, dict) or not isinstance(locks, dict):
            raise MaterializationAuditError(
                f"{cell_id} has malformed execution-template bindings."
            )
        protocol["sha256"] = "<protocol-digest>"
        locks["implementation_source_inventory_sha256"] = (
            "<implementation-inventory>"
        )
        locks["source_locks_manifest_sha256"] = "<source-lock-manifest>"
    if "execution_fulfillment" in old:
        raise MaterializationAuditError(
            f"{cell_id} unexpectedly had execution fulfillment in v5."
        )
    if not isinstance(new.pop("execution_fulfillment", None), Mapping):
        raise MaterializationAuditError(
            f"{cell_id} has no v6 execution-fulfillment template field."
        )
    return old, new, [
        "$.sha256",
        "$.protocol.sha256",
        "$.input_source_lock_hashes."
        "{implementation_source_inventory_sha256,"
        "source_locks_manifest_sha256}",
        "$.execution_fulfillment",
    ]


def _singleton_child_identity_receipt(
    baselines: Mapping[str, Any],
) -> dict[str, Any]:
    resolver = _problem_resolver_from(baselines)
    rows: list[dict[str, Any]] = []
    adapter = SinglePauliWordCandidateAdapter()
    for regime_id, expected in EXPECTED_SINGLETON_CHILD_IDENTITIES.items():
        problem = resolver(regime_id, 3)
        parent = adapter.parent_inventory(problem).receipt.to_dict()
        child_inventory = adapter.global_executable_pool(problem)
        child = child_inventory.receipt.to_dict()
        shared = child_inventory.metadata.get("shared_pool_manifest")
        if not isinstance(shared, Mapping):
            raise MaterializationAuditError(
                f"{regime_id} global child pool has no shared manifest."
            )
        observed = {
            "parent_count": int(parent["count"]),
            "parent_ordered_labels_sha256": parent[
                "ordered_labels_sha256"
            ],
            "parent_ordered_pool_sha256": parent["ordered_pool_sha256"],
            "child_count": int(child["count"]),
            "child_ordered_labels_sha256": child[
                "ordered_labels_sha256"
            ],
            "child_ordered_pool_sha256": child["ordered_pool_sha256"],
            "child_receipt_sha256": child["sha256"],
            "shared_contract_ordered_pool_sha256": shared[
                "ordered_pool_hash"
            ],
            "historical_source_runtime_ordered_pool_sha256": expected[
                "historical_source_runtime_ordered_pool_sha256"
            ],
        }
        _assert_equal(
            observed, expected, label=f"{regime_id}.singleton child identity"
        )
        historical_trace = _load_mapping(
            V5_SOURCE_ROOT
            / "resolver_traces"
            / "raw"
            / f"{regime_id}__nph3__singleton_plateau.json"
        )
        historical_settings = (
            historical_trace.get("settings_reused", {}).get("settings", {})
        )
        _assert_equal(
            historical_settings.get("shared_pauli_pool_ordered_pool_hash"),
            expected["historical_source_runtime_ordered_pool_sha256"],
            label=f"{regime_id}.historical singleton runtime hash",
        )
        rows.append(
            {
                "regime_id": regime_id,
                "nph": 3,
                **observed,
                "historical_runtime_contract": {
                    "shared_pauli_pool_mode": historical_settings.get(
                        "shared_pauli_pool_mode"
                    ),
                    "shared_pauli_pool_symmetry_policy": (
                        historical_settings.get(
                            "shared_pauli_pool_symmetry_policy"
                        )
                    ),
                },
                "change_classification": (
                    "new_explicit_global_hard_guard_child_identity_receipt_v1"
                ),
                "reason": (
                    "shared_guard_legal_codeword_fixed_sector_and_intrinsic_"
                    "singleton_identity_repair_v1"
                ),
            }
        )
    return _digested(
        {
            "schema": "ra_adapt_v6_singleton_child_pool_identity_v1",
            "status": "passed",
            "rows": rows,
            "pool_identity_hashes_masked_from_normalized_diff": False,
        }
    )


def _build_normalized_diff_receipt(
    staging_root: Path,
    *,
    baselines: Mapping[str, Any],
) -> dict[str, Any]:
    bundle_rows: list[dict[str, Any]] = []
    protocol_rows: list[dict[str, Any]] = []
    total_append = 0
    total_always = 0
    for bundle_id, _policy in STUDY1_BUNDLE_POLICIES:
        old_bundle = V5_ROOT / bundle_id
        new_bundle = staging_root / bundle_id
        old_manifest = _load_canonical_digested(
            old_bundle / "bundle_manifest.json", label="v5 bundle manifest"
        )
        new_manifest = _load_canonical_digested(
            new_bundle / "bundle_manifest.json", label="v6 bundle manifest"
        )
        old_norm, new_norm, manifest_allowed = _normalize_manifest_pair(
            old_manifest, new_manifest, bundle_id=bundle_id
        )
        _require_normalized_equal(
            old_norm, new_norm, label=f"{bundle_id}.bundle_manifest"
        )
        old_locks = _load_canonical_digested(
            old_bundle / "source_locks.json", label="v5 source locks"
        )
        new_locks = _load_canonical_digested(
            new_bundle / "source_locks.json", label="v6 source locks"
        )
        old_norm, new_norm, lock_allowed = _normalize_source_lock_pair(
            old_locks, new_locks
        )
        _require_normalized_equal(
            old_norm, new_norm, label=f"{bundle_id}.source_locks"
        )
        old_expected = _load_canonical_digested(
            old_bundle / "expected_artifacts.json",
            label="v5 expected artifacts",
        )
        new_expected = _load_canonical_digested(
            new_bundle / "expected_artifacts.json",
            label="v6 expected artifacts",
        )
        old_norm, new_norm, expected_allowed = _normalize_expected_pair(
            old_expected, new_expected, bundle_id=bundle_id
        )
        _require_normalized_equal(
            old_norm, new_norm, label=f"{bundle_id}.expected_artifacts"
        )
        bundle_rows.append(
            {
                "bundle_id": bundle_id,
                "cell_count": EXPECTED_BUNDLE_CELL_COUNT,
                "ordered_cell_matrix_unchanged": True,
                "source_scientific_locks_unchanged": True,
                "allowed_manifest_paths": manifest_allowed,
                "allowed_source_lock_paths": lock_allowed,
                "allowed_expected_artifact_paths": expected_allowed,
                "normalized_status": "equal",
            }
        )
        for cell_id in sorted(
            str(row["cell_id"]) for row in old_manifest["cells"]
        ):
            old_protocol = _load_canonical_digested(
                old_bundle / "protocols" / f"{cell_id}.json",
                label="v5 protocol",
            )
            new_protocol = _load_canonical_digested(
                new_bundle / "protocols" / f"{cell_id}.json",
                label="v6 protocol",
            )
            old_norm, new_norm, protocol_allowed = _normalize_protocol_pair(
                old_protocol, new_protocol, cell_id=cell_id
            )
            _require_normalized_equal(
                old_norm, new_norm, label=f"{bundle_id}/{cell_id}"
            )
            old_template = _load_canonical_digested(
                old_bundle / "execution_templates" / f"{cell_id}.json",
                label="v5 execution template",
            )
            new_template = _load_canonical_digested(
                new_bundle / "execution_templates" / f"{cell_id}.json",
                label="v6 execution template",
            )
            old_template_norm, new_template_norm, template_allowed = (
                _normalize_template_pair(
                    old_template, new_template, cell_id=cell_id
                )
            )
            _require_normalized_equal(
                old_template_norm,
                new_template_norm,
                label=f"{bundle_id}/{cell_id}.execution_template",
            )
            change_classes = ["materialization_provenance_rebinding"]
            if cell_id.endswith("__append_macro"):
                total_append += 1
                change_classes.append(
                    "append_native_refit_and_selector_scope_repair"
                )
            if cell_id.endswith("__ra_macro_always"):
                total_always += 1
                change_classes.append("typed_full_insertion_request_fidelity")
            protocol_rows.append(
                {
                    "bundle_id": bundle_id,
                    "cell_id": cell_id,
                    "change_classes": change_classes,
                    "allowed_protocol_paths": protocol_allowed,
                    "allowed_execution_template_paths": template_allowed,
                    "parent_inventory_unchanged": (
                        old_protocol["parent_inventory"]
                        == new_protocol["parent_inventory"]
                    ),
                    "executable_pool_unchanged": (
                        old_protocol["executable_pool"]
                        == new_protocol["executable_pool"]
                    ),
                    "normalized_status": "equal",
                }
            )
            if not protocol_rows[-1]["parent_inventory_unchanged"] or not (
                protocol_rows[-1]["executable_pool_unchanged"]
            ):
                raise MaterializationAuditError(
                    f"Pool identity drifted for {bundle_id}/{cell_id}."
                )
    _assert_equal(total_append, 28, label="Append protocol delta count")
    _assert_equal(total_always, 28, label="always protocol delta count")
    child_identity = _singleton_child_identity_receipt(baselines)
    return _digested(
        {
            "schema": "ra_adapt_v5_v6_normalized_diff_v1",
            "status": "passed",
            "baseline_revision": "v5",
            "target_revision": "v6",
            "bundle_rows": bundle_rows,
            "protocol_rows": protocol_rows,
            "protocol_count": len(protocol_rows),
            "append_native_refit_protocol_count": total_append,
            "typed_full_insertion_protocol_count": total_always,
            "ledger_charge_repair": {
                "serialized_estimator_accounting_convention_changed": False,
                "implementation_inventory_bound": True,
                "semantic_change": (
                    "append_refit_chart_gradient_and_metric_construction_"
                    "charges_removed_v1"
                ),
            },
            "singleton_child_pool_identity": child_identity,
            "pool_identity_hashes_masked": False,
            "physics_fields_masked": False,
            "source_archive_or_member_hashes_masked": False,
            "optimizer_budget_seed_or_horizon_fields_masked": False,
            "unexpected_normalized_difference_count": 0,
        }
    )


def _run_cross_loader_validation(
    root: Path,
    *,
    phase: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    bundle_counts: dict[str, int] = {}
    for bundle_id, _policy in STUDY1_BUNDLE_POLICIES:
        bundle_root = root / bundle_id
        manifest = _load_canonical_digested(
            bundle_root / "bundle_manifest.json",
            label=f"{phase} bundle manifest",
        )
        cells = manifest.get("cells")
        if not isinstance(cells, list):
            raise MaterializationAuditError(
                f"{phase}/{bundle_id} has no ordered cells."
            )
        bundle_counts[bundle_id] = len(cells)
        for cell in cells:
            cell_id = str(cell.get("cell_id", ""))
            path = bundle_root / "protocols" / f"{cell_id}.json"
            protocol_payload = _load_canonical_digested(
                path, label=f"{phase} protocol"
            )
            try:
                loaded = load_validated_bundle_protocol(path)
                loaded_sha256 = str(getattr(loaded, "sha256"))
                if loaded_sha256 != protocol_payload["sha256"]:
                    raise MaterializationAuditError(
                        f"Loaded protocol digest drifted for {cell_id}."
                    )
                status = "passed"
                error = None
            except Exception as exc:  # receipt captures every loader failure
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "bundle_id": bundle_id,
                    "cell_id": cell_id,
                    "protocol_path": path.relative_to(
                        REPO_ROOT
                    ).as_posix(),
                    "protocol_sha256": protocol_payload["sha256"],
                    "status": status,
                    **({"error": error} if error is not None else {}),
                }
            )
    total = sum(bundle_counts.values())
    passed = sum(row["status"] == "passed" for row in rows)
    failed = total - passed
    _assert_equal(total, EXPECTED_CROSS_LOADER_COUNT, label=f"{phase}.total")
    _assert_equal(len(rows), total, label=f"{phase}.row count")
    if failed:
        raise MaterializationAuditError(
            f"{phase} cross-loader validation failed {failed}/{total} rows."
        )
    return _digested(
        {
            "schema": "ra_adapt_cross_file_loader_validation_v1",
            "status": "passed",
            "phase": phase,
            "validated_root_path": root.relative_to(
                REPO_ROOT
            ).as_posix(),
            "loader": (
                "pipelines.static_adapt.ra_adapt.bundles."
                "load_validated_bundle_protocol"
            ),
            "bundle_counts": bundle_counts,
            "total_count": total,
            "passed_count": passed,
            "failed_count": failed,
            "rows": rows,
            "completed_utc": _utc_now(),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        }
    )


def _preservation_comparison(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for revision in HISTORICAL_ROOTS:
        old = before[revision]
        new = after[revision]
        result[revision] = {
            "file_count": int(old["file_count"]),
            "pre_tree_sha256": old["tree_sha256"],
            "post_tree_sha256": new["tree_sha256"],
            "unchanged": old == new,
        }
    return result


def main() -> int:
    if V6_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite immutable v6: {V6_ROOT}")
    if not MATERIALIZATIONS_ROOT.is_dir():
        raise MaterializationAuditError(
            f"Materializations root is missing: {MATERIALIZATIONS_ROOT}"
        )
    _darwin_renameatx_np()

    captured_utc = _utc_now()
    repository_state = _repository_state()
    historical_before = _snapshot_historical_materializations()
    _assert_historical_anchors(historical_before)
    implementation_preflight = _implementation_source_inventory(REPO_ROOT)
    source_locks = _load_mapping(SOURCE_LOCKS_INPUT)
    baselines = _load_mapping(PROBLEM_BASELINES)

    staging_root = Path(
        tempfile.mkdtemp(
            prefix=".ra_adapt_unification_post_refactor_v6.staging.",
            dir=MATERIALIZATIONS_ROOT,
        )
    )
    source_before, source_after = _copy_and_verify_source_materialization(
        staging_root
    )
    inheritance_receipt = _write_receipt(
        staging_root / "source_materialization_inheritance_receipt.json",
        {
            "schema": "ra_adapt_source_materialization_inheritance_v1",
            "status": "passed",
            "source_revision": "v5",
            "target_revision": "v6",
            "source_path": V5_SOURCE_ROOT.relative_to(REPO_ROOT).as_posix(),
            "target_path": (
                V6_ROOT / "source_materialization"
            ).relative_to(REPO_ROOT).as_posix(),
            "copy_policy": "byte_identical_no_path_rebasing_v1",
            "historical_embedded_v5_paths_preserved": True,
            "inherited_source_validation_role": (
                "historical_receipt_not_fresh_v6_implementation_validation"
            ),
            "fresh_v6_validation_authority": (
                "each_bundle_source_locks_json_plus_current_implementation_"
                "inventory_v1"
            ),
            "file_count": source_before["file_count"],
            "total_size_bytes": source_before["total_size_bytes"],
            "source_relative_tree_sha256": source_before["tree_sha256"],
            "copied_relative_tree_sha256": source_after["tree_sha256"],
            "files": source_before["files"],
            "files_equal": True,
        },
    )
    preflight_receipt = _write_receipt(
        staging_root / "preflight_receipt.json",
        {
            "schema": "ra_adapt_v6_preflight_receipt_v1",
            "status": "passed",
            "captured_utc": captured_utc,
            "repository_state": repository_state,
            "implementation_inventory": implementation_preflight,
            "visible_source_locks": {
                "path": SOURCE_LOCKS_INPUT.relative_to(REPO_ROOT).as_posix(),
                "file_sha256": _hash_file(SOURCE_LOCKS_INPUT),
                "schema": source_locks.get("schema"),
                "cell_lock_count": len(source_locks.get("cell_locks", {})),
                "global_source_count": len(
                    source_locks.get("global_sources", {})
                ),
            },
            "problem_baselines": {
                "path": PROBLEM_BASELINES.relative_to(REPO_ROOT).as_posix(),
                "file_sha256": _hash_file(PROBLEM_BASELINES),
                "schema": baselines.get("schema"),
            },
            "older_materialization_preservation": historical_before,
            "source_materialization_inheritance_receipt_sha256": (
                inheritance_receipt["sha256"]
            ),
            "execution_authorized": False,
            "submission_state": "not_submitted",
            "submitted": False,
        },
    )

    timestamp = _utc_now()
    materialized_receipts = materialize_study1_bundles(
        staging_root,
        problem_resolver=_problem_resolver_from(baselines),
        source_locks=source_locks,
        validation_horizon=23,
        repository_state=repository_state,
        repo_root=REPO_ROOT,
        full_horizon=50,
        dependency_lock_paths=(REPO_ROOT / "requirements.txt",),
        materialization_timestamp=timestamp,
        verify_source_files=True,
    )
    bundle_summaries = [
        _validate_bundle_surface(
            staging_root / bundle_id,
            expected_implementation_inventory=implementation_preflight,
        )
        for bundle_id, _policy in STUDY1_BUNDLE_POLICIES
    ]
    receipt_by_bundle = {
        receipt.bundle_id: receipt for receipt in materialized_receipts
    }
    for summary in bundle_summaries:
        receipt = receipt_by_bundle[summary["bundle_id"]]
        _assert_equal(
            receipt.cell_count,
            EXPECTED_BUNDLE_CELL_COUNT,
            label=f"{receipt.bundle_id}.materializer receipt cell count",
        )
        _assert_equal(
            receipt.materialization_status,
            "passed",
            label=f"{receipt.bundle_id}.materializer receipt status",
        )
        _assert_equal(
            receipt.bundle_manifest_sha256,
            summary["bundle_manifest_sha256"],
            label=f"{receipt.bundle_id}.manifest receipt binding",
        )
    staged_source_locks = [
        _load_canonical_digested(
            staging_root / bundle_id / "source_locks.json",
            label=f"{bundle_id} source locks",
        )
        for bundle_id, _policy in STUDY1_BUNDLE_POLICIES
    ]
    _assert_equal(
        staged_source_locks[0],
        staged_source_locks[1],
        label="cross-bundle source-lock equality",
    )

    normalized_diff = _build_normalized_diff_receipt(
        staging_root, baselines=baselines
    )
    _write_receipt(
        staging_root / "v5_v6_normalized_diff_receipt.json",
        {
            key: value
            for key, value in normalized_diff.items()
            if key != "sha256"
        },
    )
    staged_loader = _run_cross_loader_validation(
        staging_root, phase="staged_pre_publish"
    )
    _write_receipt(
        staging_root / "cross_file_loader_validation_staged.json",
        {key: value for key, value in staged_loader.items() if key != "sha256"},
    )

    implementation_post_staged_loader = _implementation_source_inventory(
        REPO_ROOT
    )
    _assert_equal(
        implementation_post_staged_loader,
        implementation_preflight,
        label="preflight-to-staged-loader implementation inventory",
    )
    historical_pre_publish = _snapshot_historical_materializations()
    _assert_snapshots_equal(
        historical_before,
        historical_pre_publish,
        label="Staged materialization",
    )
    source_staged_pre_publish = _snapshot_inherited_source(staging_root)
    _assert_inherited_source_unchanged(
        source_staged_pre_publish,
        source_before,
        label="Staged pre-publication",
    )
    _write_receipt(
        staging_root / "staged_materialization_receipt.json",
        {
            "schema": "ra_adapt_v6_staged_materialization_receipt_v1",
            "status": "passed",
            "materialization_revision": "v6",
            "materialization_timestamp": timestamp,
            "preflight_receipt_sha256": preflight_receipt["sha256"],
            "source_inheritance_receipt_sha256": inheritance_receipt["sha256"],
            "inherited_source_materialization": {
                "pinned_v5_snapshot": _snapshot_summary(source_before),
                "initial_copy_snapshot": _snapshot_summary(source_after),
                "staged_pre_publish_snapshot": _snapshot_summary(
                    source_staged_pre_publish
                ),
                "snapshots_exactly_equal": True,
            },
            "normalized_diff_receipt_sha256": normalized_diff["sha256"],
            "staged_loader_validation_sha256": staged_loader["sha256"],
            "bundles": bundle_summaries,
            "implementation_inventory": {
                "preflight_sha256": implementation_preflight["sha256"],
                "post_staged_loader_sha256": (
                    implementation_post_staged_loader["sha256"]
                ),
                "stable": True,
            },
            "older_materialization_preservation": _preservation_comparison(
                historical_before, historical_pre_publish
            ),
            "atomic_publish_ready": True,
            "atomic_publish_method": (
                "darwin_renameatx_np_RENAME_EXCL_v1"
            ),
            "atomic_publish_no_replace": True,
            "execution_authorized": False,
            "submission_state": "not_submitted",
            "submitted": False,
        },
    )

    _atomic_rename_no_replace(staging_root, V6_ROOT)
    source_immediate_post_publish = _snapshot_inherited_source(V6_ROOT)
    _assert_inherited_source_unchanged(
        source_immediate_post_publish,
        source_before,
        label="Immediate post-publication",
    )

    final_loader = _run_cross_loader_validation(
        V6_ROOT, phase="final_post_publish"
    )
    final_loader_receipt = _write_receipt(
        V6_ROOT / "cross_file_loader_validation.json",
        {key: value for key, value in final_loader.items() if key != "sha256"},
    )
    implementation_post_final_loader = _implementation_source_inventory(
        REPO_ROOT
    )
    _assert_equal(
        implementation_post_final_loader,
        implementation_preflight,
        label="preflight-to-final-loader implementation inventory",
    )
    historical_after = _snapshot_historical_materializations()
    _assert_snapshots_equal(
        historical_before,
        historical_after,
        label="Published v6 materialization",
    )
    final_bundle_summaries = [
        _validate_bundle_surface(
            V6_ROOT / bundle_id,
            expected_implementation_inventory=implementation_preflight,
        )
        for bundle_id, _policy in STUDY1_BUNDLE_POLICIES
    ]
    _assert_equal(
        final_bundle_summaries,
        bundle_summaries,
        label="staged-to-final bundle summaries",
    )
    source_final_pre_receipt = _snapshot_inherited_source(V6_ROOT)
    _assert_inherited_source_unchanged(
        source_final_pre_receipt,
        source_before,
        label="Final pre-receipt",
    )

    v5_final = _load_mapping(V5_FINAL_RECEIPT)
    supersession_chain = list(v5_final.get("supersession_chain", ()))
    if not supersession_chain or supersession_chain[-1].get("revision") != "v5":
        raise MaterializationAuditError(
            "The v5 supersession chain is missing its v5 terminus."
        )
    supersession_chain.append(
        {
            "revision": "v6",
            "path": V6_ROOT.relative_to(REPO_ROOT).as_posix(),
            "status": "passed",
        }
    )
    v6_tree_before_final_receipt = _snapshot_roots((V6_ROOT,))
    final_receipt = _write_receipt(
        V6_ROOT / "final_materialization_receipt.json",
        {
            "schema": "ra_adapt_final_materialization_receipt_v1",
            "status": "passed",
            "campaign_id": (
                "paper_i_ra_adapt_stationarity_comparison_v1"
            ),
            "run_class": "candidate",
            "materialization_revision": "v6",
            "finalized_utc": _utc_now(),
            "atomic_publish": {
                "method": "darwin_renameatx_np_RENAME_EXCL_v1",
                "no_replace": True,
                "unsupported_platform_behavior": "fail_closed",
                "staged_loader_validation_sha256": staged_loader["sha256"],
                "final_loader_validation_sha256": (
                    final_loader_receipt["sha256"]
                ),
                "staged_and_final_loader_rows": EXPECTED_CROSS_LOADER_COUNT,
            },
            "bundles": final_bundle_summaries,
            "source_materialization": {
                "path": (
                    V6_ROOT / "source_materialization"
                ).relative_to(REPO_ROOT).as_posix(),
                "status": "inherited_byte_identical",
                "file_count": source_final_pre_receipt["file_count"],
                "relative_tree_sha256": (
                    source_final_pre_receipt["tree_sha256"]
                ),
                "inheritance_receipt_sha256": inheritance_receipt["sha256"],
                "pinned_v5_snapshot": _snapshot_summary(source_before),
                "initial_copy_snapshot": _snapshot_summary(source_after),
                "staged_pre_publish_snapshot": _snapshot_summary(
                    source_staged_pre_publish
                ),
                "immediate_post_publish_snapshot": _snapshot_summary(
                    source_immediate_post_publish
                ),
                "final_pre_receipt_snapshot": _snapshot_summary(
                    source_final_pre_receipt
                ),
                "snapshots_exactly_equal": True,
                "fresh_validation_authority": (
                    "bundle_source_locks_and_implementation_inventory_v1"
                ),
            },
            "normalized_diff": {
                "path": (
                    V6_ROOT / "v5_v6_normalized_diff_receipt.json"
                ).relative_to(REPO_ROOT).as_posix(),
                "sha256": normalized_diff["sha256"],
                "status": "passed",
            },
            "loader_validation": {
                "path": (
                    V6_ROOT / "cross_file_loader_validation.json"
                ).relative_to(REPO_ROOT).as_posix(),
                "sha256": final_loader_receipt["sha256"],
                "total_count": EXPECTED_CROSS_LOADER_COUNT,
                "passed_count": EXPECTED_CROSS_LOADER_COUNT,
                "failed_count": 0,
            },
            "implementation_inventory": {
                "root_count": implementation_preflight["root_count"],
                "file_count": implementation_preflight["file_count"],
                "preflight_sha256": implementation_preflight["sha256"],
                "post_staged_loader_sha256": (
                    implementation_post_staged_loader["sha256"]
                ),
                "post_final_loader_sha256": (
                    implementation_post_final_loader["sha256"]
                ),
                "stable": True,
            },
            "older_materialization_preservation": _preservation_comparison(
                historical_before, historical_after
            ),
            "v6_tree_before_final_receipt": _snapshot_summary(
                v6_tree_before_final_receipt
            ),
            "supersession_chain": supersession_chain,
            "stationarity_winner_selected": False,
            "user_decision_required_after_study_1": True,
            "execution_authorized": False,
            "submission_state": "not_submitted",
            "submitted": False,
        },
    )
    print(
        json.dumps(
            {
                "destination": V6_ROOT.relative_to(REPO_ROOT).as_posix(),
                "status": "passed",
                "bundle_count": len(final_bundle_summaries),
                "cell_count_per_bundle": EXPECTED_BUNDLE_CELL_COUNT,
                "loader_validation": "116/116",
                "implementation_inventory_sha256": (
                    implementation_preflight["sha256"]
                ),
                "final_receipt_sha256": final_receipt["sha256"],
                "execution_authorized": False,
                "submitted": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
