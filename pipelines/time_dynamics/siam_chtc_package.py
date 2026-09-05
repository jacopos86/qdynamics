"""Source-locked one-action CHTC packages for the SIAM Nb=4 pilot.

This module materializes the control plane around the registered SIAM screen
and continuation commands.  It deliberately does not submit jobs, build exact
references, score trajectories, or compile circuits.  Each package contains
one action and one unique raw-output path; exact evaluation remains a separate
post-run transaction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from numbers import Real
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.paper_ii_runs import (
    ARMS,
    EULER_RIDGE1E6_NUMERICS,
    RUNNER_PATH,
    SIAM_NB4_ARMS,
    SIAM_NB4_CONTINUATION_SEGMENT,
    SIAM_NB4_PILOT_SEARCH,
    SIAM_NB4_SCREEN_SEGMENT,
    STATE_MOTION_CONTROL_CALIBRATED,
    build_siam_nb4_continuation_run,
    build_siam_nb4_screen_run,
)
from pipelines.time_dynamics.ap_mclachlan.exchange_config import (
    EXCHANGE_ROUTE_BUNDLE_HISTORICAL,
    EXCHANGE_ROUTE_BUNDLES,
)
from pipelines.time_dynamics.ap_mclachlan.exchange_integration import (
    build_candidate_pool_contract,
)
from pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory import (
    _controller_resume_snapshot,
    _restore_controller_resume_state,
    _support_identity_hash,
)
from pipelines.time_dynamics.ap_mclachlan.state import (
    AP_PARAMETERIZATION_LOGICAL_SHARED,
    state_from_scaffold_runtime_input,
)
from pipelines.time_dynamics.diagnostics.siam_offline_scoring import (
    SIAM_ROUTE_NOT_APPLICABLE,
)
from pipelines.time_dynamics.resume import (
    CONTROLLER_RESUME_SCHEMA_V1,
    RESUME_SCHEMA_V2,
    runtime_input_from_resume_state,
)
from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import (
    _build_parser,
    _resolve_times,
    _solve_repair_config_from_args,
    _support_patch_config_from_args,
)
from pipelines.time_dynamics.siam_activation_ladder import (
    SIAM_NB4_ACTIVATION_CUT_LADDER,
    SIAM_NB4_CANONICAL_LADDER_SHA256,
    canonical_registered_cut,
    ladder_payload,
    resolve_embedded_ladder,
)
from pipelines.time_dynamics.siam_seed import (
    SIAM_SEED_SCHEMA_V2,
    load_siam_runtime_input,
    load_siam_seed_manifest,
    portable_statevector_sha256,
)
from src.quantum.siam_model import (
    build_initial_state,
    build_siam_hamiltonian,
    sector_indices,
)
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix


SIAM_NB4_CHTC_ACTION_SCHEMA_V3 = "siam_nb4_chtc_action_v3"
SIAM_NB4_CHTC_MANIFEST_SCHEMA_V3 = "siam_nb4_chtc_package_manifest_v3"
SIAM_NB4_CHTC_RECEIPT_SCHEMA_V3 = "siam_nb4_chtc_action_receipt_v3"
SIAM_NB4_CHTC_ACTION_SCHEMA_V4 = "siam_nb4_chtc_action_v4"
SIAM_NB4_CHTC_MANIFEST_SCHEMA_V4 = "siam_nb4_chtc_package_manifest_v4"
SIAM_NB4_CHTC_RECEIPT_SCHEMA_V4 = "siam_nb4_chtc_action_receipt_v4"
# V5 adds exchange-route identity (``route_bundle_id`` /
# ``route_contract_sha256``) to the three digest-bound documents.  It is
# emitted only for a ladder on a registered non-historical exchange route: a
# historical package stays byte-identical at V4 so the r00--r07 lineage
# remains reproducible and every V4 reader keeps working unchanged.
SIAM_NB4_CHTC_ACTION_SCHEMA_V5 = "siam_nb4_chtc_action_v5"
SIAM_NB4_CHTC_MANIFEST_SCHEMA_V5 = "siam_nb4_chtc_package_manifest_v5"
SIAM_NB4_CHTC_RECEIPT_SCHEMA_V5 = "siam_nb4_chtc_action_receipt_v5"
SIAM_SOURCE_TREE_CONTRACT_V1 = "siam_source_tree_contract_v1"
SIAM_NB4_RESUME_TERMINAL_PROOF_V1 = "siam_nb4_resume_terminal_proof_v1"
SIAM_SOURCE_EPOCH_CONTRACT_V1 = "siam_source_epoch_contract_v1"
SIAM_SOURCE_EPOCH_CLOSURE_V1 = "siam_source_epoch_closure_v1"
SIAM_SOURCE_EPOCH_REBASELINE_V1 = "siam_source_epoch_rebaseline_v1"

# The canonical arm id is the *arm* identity, not the runner policy: the
# compiled-Q route shares ``--dynamics-policy exchange`` with the historical
# exchange arm but is a different arm and a different ladder.
_ARM_ALIASES: Mapping[str, str] = {
    "apm": "exchange",
    "avqds": "avqds",
    "exchange": "exchange",
    "exchange_q_compiled": "exchange_q_compiled",
    "full_apm": "exchange",
    "full_apm_q_compiled": "exchange_q_compiled",
}
# Method identity is recorded off the arm id; the runner policy is recorded
# off ``Arm.dynamics_policy``.  The two are deliberately separate axes.
_ARM_METHOD_LABELS: Mapping[str, str] = {
    "avqds": "avqds",
    "exchange": "full_apm",
    "exchange_q_compiled": "full_apm_q_compiled",
}
# The compiled-Q route's batch name carries the arm id verbatim so the route
# is legible in the pool.  This mirrors
# ``siam_submission_intent_bridge._accepted_batch_names`` exactly.
_VERBATIM_BATCH_ARMS = frozenset({"exchange_q_compiled"})
# ``siam_nb4_ladder`` names the two comparison *lanes*, not the arms: the route
# is a separate axis there, carried by ``route_id``.  Every exchange-family arm
# is the ladder's ``full_apm`` lane.
_LADDER_LANES: Mapping[str, str] = {
    "avqds": "avqds",
    "exchange": "full_apm",
    "exchange_q_compiled": "full_apm",
}
ROUTE_HISTORICAL = "historical"
_SEGMENT_ALIASES = {
    "continuation": SIAM_NB4_CONTINUATION_SEGMENT,
    "continue": SIAM_NB4_CONTINUATION_SEGMENT,
    SIAM_NB4_CONTINUATION_SEGMENT.segment_id: SIAM_NB4_CONTINUATION_SEGMENT,
    "screen": SIAM_NB4_SCREEN_SEGMENT,
    SIAM_NB4_SCREEN_SEGMENT.segment_id: SIAM_NB4_SCREEN_SEGMENT,
}
_ACTION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SAFE_TRANSFER_PATH = re.compile(r"^[A-Za-z0-9_./-]+$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _argv_sha256(argv: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(
            list(argv), separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _write_bytes(path: Path, content: bytes, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    if executable:
        path.chmod(0o755)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a mutable execution receipt, never package inputs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_pretty_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _relative_transfer_path(root: Path, value: str | Path, *, label: str) -> str:
    candidate = Path(value)
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (root / candidate).resolve()
    )
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must be inside repo_root: {resolved}") from exc
    text = relative.as_posix()
    if not text or not _SAFE_TRANSFER_PATH.fullmatch(text):
        raise ValueError(f"{label} is not a CHTC-safe relative path: {text!r}")
    return text


def _validated_revision(value: str) -> str:
    revision = str(value).strip()
    if not _REVISION.fullmatch(revision):
        raise ValueError(
            "SIAM CHTC source_revision must be one lowercase 40-character "
            f"commit id, got {revision!r}"
        )
    return revision


def build_siam_source_epoch_contract(
    *, source_revision: str, source_tree_files_sha256: str
) -> dict[str, str]:
    """Return the stable source epoch used in composite action identities."""

    revision = _validated_revision(source_revision)
    tree_sha = str(source_tree_files_sha256).strip()
    if not _SHA256.fullmatch(tree_sha):
        raise ValueError("SIAM source epoch requires a source-tree SHA-256")
    identity = {
        "schema": SIAM_SOURCE_EPOCH_CONTRACT_V1,
        "source_revision": revision,
        "source_tree_files_sha256": tree_sha,
    }
    return {
        **identity,
        "source_epoch_id": "siam-nb4-source-"
        + _sha256_bytes(_canonical_json_bytes(identity))[:20],
    }


def build_siam_source_epoch_closure(
    *,
    source_epoch_id: str,
    superseded_by_source_epoch_id: str,
    package_id: str,
    action_id: str,
    source_revision: str,
    source_tree_files_sha256: str,
    submission_receipt_sha256: str,
    package_manifest_sha256: str,
    closed_at: str,
) -> dict[str, Any]:
    """Build a self-digesting terminal receipt for one superseded epoch."""

    old_epoch = _validated_action_id(source_epoch_id)
    new_epoch = _validated_action_id(superseded_by_source_epoch_id)
    if old_epoch == new_epoch:
        raise ValueError("SIAM epoch closure must name a distinct successor")
    revision = _validated_revision(source_revision)
    digests = {
        "source_tree_files_sha256": source_tree_files_sha256,
        "submission_receipt_sha256": submission_receipt_sha256,
        "package_manifest_sha256": package_manifest_sha256,
    }
    if any(not _SHA256.fullmatch(str(value)) for value in digests.values()):
        raise ValueError("SIAM epoch closure requires valid SHA-256 digests")
    timestamp = str(closed_at).strip()
    if not timestamp:
        raise ValueError("SIAM epoch closure requires a closed_at timestamp")
    core: dict[str, Any] = {
        "schema": SIAM_SOURCE_EPOCH_CLOSURE_V1,
        "status": "CLOSED_SUPERSEDED",
        "source_epoch_id": old_epoch,
        "superseded_by_source_epoch_id": new_epoch,
        "closed_to_new_actions": True,
        "closed_at": timestamp,
        "package_id": _validated_action_id(package_id),
        "action_id": _validated_action_id(action_id),
        "source_revision": revision,
        **digests,
    }
    return {
        **core,
        "closure_receipt_sha256": _sha256_bytes(_canonical_json_bytes(core)),
    }


def write_siam_source_epoch_closure(
    path: str | Path, **closure_fields: Any
) -> Path:
    """Write one immutable, self-digesting source-epoch closure receipt."""

    destination = Path(path)
    payload = build_siam_source_epoch_closure(**closure_fields)
    _write_bytes(destination, _pretty_json_bytes(payload))
    return destination


def _verify_source_revision(repo_root: Path, revision: str) -> None:
    """Require a real commit containing this runtime in Git worktrees."""

    if not (repo_root / ".git").exists():
        raise ValueError(
            "SIAM production package preparation requires a Git worktree; "
            "an unversioned source tree cannot satisfy the source lock"
        )
    commit = subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if commit.returncode != 0:
        raise ValueError(f"SIAM source_revision is not a local commit: {revision}")
    required = (
        "pipelines/time_dynamics/siam_chtc_package.py",
        "pipelines/time_dynamics/paper_ii_runs.py",
        "pipelines/time_dynamics/siam_seed.py",
        RUNNER_PATH,
    )
    missing = []
    for relative in required:
        result = subprocess.run(
            ["git", "cat-file", "-e", f"{revision}:{relative}"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            missing.append(relative)
    if missing:
        raise ValueError(
            "SIAM source_revision predates required package runtime files: "
            + ", ".join(missing)
        )
    drift = subprocess.run(
        ["git", "diff", "--quiet", revision, "--", "pipelines", "src"],
        cwd=repo_root,
        check=False,
    )
    if drift.returncode != 0:
        raise ValueError(
            "SIAM package preparation source differs from source_revision; "
            "prepare from its clean pipelines/src tree or exact worktree"
        )
    untracked = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            "pipelines",
            "src",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if untracked.returncode != 0:
        raise ValueError("Unable to audit untracked SIAM runtime source files")
    if untracked.stdout.strip():
        raise ValueError(
            "SIAM package preparation has untracked runtime source files: "
            + ", ".join(untracked.stdout.splitlines())
        )


def _build_source_tree_contract(
    repo_root: Path, revision: str
) -> dict[str, Any]:
    """Describe every tracked ``pipelines``/``src`` byte at one commit.

    The remote execute directory does not contain ``.git``.  Embedding the
    exact archive members in the action lets the worker prove that the Python
    tree it imports is the committed tree rather than merely trusting a
    revision string.
    """

    archived = subprocess.run(
        ["git", "archive", "--format=tar", revision, "pipelines", "src"],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if archived.returncode != 0:
        raise ValueError("Unable to materialize the committed SIAM source tree")
    rows: list[dict[str, Any]] = []
    with tarfile.open(fileobj=io.BytesIO(archived.stdout), mode="r:") as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError("Committed SIAM source archive has an unsafe path")
            if path.parts[0] not in {"pipelines", "src"}:
                raise ValueError("Committed SIAM source archive escaped its roots")
            if member.isfile():
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError("Committed SIAM source member is unreadable")
                content = handle.read()
                kind = "file"
                link_target = None
            elif member.issym():
                content = member.linkname.encode("utf-8")
                kind = "symlink"
                link_target = member.linkname
            else:
                raise ValueError(
                    "Committed SIAM source tree contains an unsupported member"
                )
            rows.append(
                {
                    "path": path.as_posix(),
                    "kind": kind,
                    "mode": int(member.mode) & 0o777,
                    "bytes": len(content),
                    "sha256": _sha256_bytes(content),
                    "link_target": link_target,
                }
            )
    rows.sort(key=lambda row: str(row["path"]))
    if not rows or not {str(row["path"]).split("/", 1)[0] for row in rows} >= {
        "pipelines",
        "src",
    }:
        raise ValueError("Committed SIAM source tree is incomplete")
    return {
        "schema": SIAM_SOURCE_TREE_CONTRACT_V1,
        "source_revision": revision,
        "file_count": len(rows),
        "files_sha256": _sha256_bytes(_canonical_json_bytes(rows)),
        "files": rows,
    }


def _verify_source_tree_contract(
    contract: Mapping[str, Any], *, repo_root: Path, revision: str
) -> dict[str, Any]:
    """Verify the imported runtime tree against its committed-byte contract."""

    document = dict(contract)
    if document.get("schema") != SIAM_SOURCE_TREE_CONTRACT_V1:
        raise ValueError("SIAM action lacks its source-tree contract")
    if document.get("source_revision") != revision:
        raise ValueError("SIAM source-tree revision mismatch")
    rows = document.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("SIAM source-tree contract has no files")
    if document.get("file_count") != len(rows):
        raise ValueError("SIAM source-tree file count mismatch")
    if document.get("files_sha256") != _sha256_bytes(
        _canonical_json_bytes(rows)
    ):
        raise ValueError("SIAM source-tree contract digest mismatch")
    seen: set[str] = set()
    roots: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("SIAM source-tree file row is malformed")
        text = str(row.get("path") or "")
        path_value = PurePosixPath(text)
        if (
            not text
            or path_value.is_absolute()
            or ".." in path_value.parts
            or path_value.parts[0] not in {"pipelines", "src"}
            or text in seen
        ):
            raise ValueError("SIAM source-tree file path is unsafe or duplicated")
        seen.add(text)
        roots.add(path_value.parts[0])
        path = repo_root / Path(*path_value.parts)
        kind = row.get("kind")
        if kind == "file":
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"SIAM source-tree file is missing: {text}")
            content = path.read_bytes()
        elif kind == "symlink":
            if not path.is_symlink():
                raise ValueError(f"SIAM source-tree symlink is missing: {text}")
            content = os.readlink(path).encode("utf-8")
            if row.get("link_target") != os.readlink(path):
                raise ValueError(f"SIAM source-tree symlink target differs: {text}")
        else:
            raise ValueError("SIAM source-tree member kind is unsupported")
        byte_count = row.get("bytes")
        mode = row.get("mode")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count != len(content)
            or row.get("sha256") != _sha256_bytes(content)
        ):
            raise ValueError(f"SIAM source-tree bytes differ: {text}")
        if (
            isinstance(mode, bool)
            or not isinstance(mode, int)
            or (path.lstat().st_mode & 0o111) != (mode & 0o111)
        ):
            raise ValueError(f"SIAM source-tree executable mode differs: {text}")
    if roots != {"pipelines", "src"}:
        raise ValueError("SIAM source-tree contract does not cover both roots")
    return {
        "source_tree_files_sha256": str(document["files_sha256"]),
        "source_tree_file_count": len(rows),
    }


def _validated_action_id(value: str) -> str:
    action_id = str(value).strip()
    if not _ACTION_ID.fullmatch(action_id):
        raise ValueError(
            "SIAM CHTC action_id must contain only letters, digits, '.', '_', "
            f"and '-', got {action_id!r}"
        )
    return action_id


def _validated_receipt_path(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    candidate = PurePosixPath(text)
    if (
        not text
        or candidate.is_absolute()
        or ".." in candidate.parts
        or not _SAFE_TRANSFER_PATH.fullmatch(text)
    ):
        raise ValueError(f"SIAM completion receipt {label} is unsafe: {text!r}")
    return text


def _canonical_arm(value: str) -> str:
    token = str(value).strip().lower().replace("-", "_")
    try:
        arm = _ARM_ALIASES[token]
    except KeyError as exc:
        raise ValueError(
            "SIAM Nb=4 CHTC arm must be AVQDS, full APM (exchange), or "
            "compiled-Q full APM (exchange_q_compiled), got "
            f"{value!r}"
        ) from exc
    if arm not in SIAM_NB4_ARMS:
        raise RuntimeError(f"registered SIAM arm disappeared: {arm}")
    return arm


def _arm_dynamics_policy(canonical_arm: str) -> str:
    """Return the runner ``--dynamics-policy`` value the arm resolves to.

    The run lock records this string as ``policy.structural_policy``, so every
    comparison against a run lock or a runner flag keys off this and never off
    the arm id.  Route bundles share the ``exchange`` policy.
    """

    try:
        return ARMS[str(canonical_arm)].dynamics_policy
    except KeyError as exc:  # pragma: no cover - guarded by _canonical_arm
        raise ValueError(f"unregistered SIAM arm {canonical_arm!r}") from exc


def _arm_method_label(canonical_arm: str) -> str:
    """Return the reported method identity for one arm id."""

    try:
        return _ARM_METHOD_LABELS[str(canonical_arm)]
    except KeyError as exc:  # pragma: no cover - guarded by _canonical_arm
        raise ValueError(
            f"SIAM arm {canonical_arm!r} has no registered method label"
        ) from exc


def _arm_route_bundle_id(canonical_arm: str) -> str:
    """Return the exchange route bundle id this arm actually runs on.

    The comparator arm is on no exchange route at all and reports
    ``not_applicable``, exactly as ``siam_offline_scoring`` does.  An exchange
    arm without an explicit ``--exchange-route-bundle`` flag runs the
    registered historical bundle.
    """

    if _arm_dynamics_policy(canonical_arm) == "avqds":
        return SIAM_ROUTE_NOT_APPLICABLE
    flags = tuple(ARMS[str(canonical_arm)].flags)
    for index, flag in enumerate(flags[:-1]):
        if flag == "--exchange-route-bundle":
            bundle_id = str(flags[index + 1])
            if bundle_id not in EXCHANGE_ROUTE_BUNDLES:
                raise ValueError(
                    f"SIAM arm {canonical_arm!r} names an unregistered "
                    f"exchange route bundle {bundle_id!r}"
                )
            return bundle_id
    return EXCHANGE_ROUTE_BUNDLE_HISTORICAL


def _route_contract_sha256(bundle_id: str) -> str | None:
    """Return the digest the run lock will record for this bundle.

    This reproduces the runner's computation verbatim -- ``asdict`` of the
    registered :class:`ExchangeRouteBundle` hashed as canonical JSON -- so the
    packaged expectation and ``run_lock.policy.guards``
    ``exchange_route_contract_sha256`` are the same string by construction.
    The comparator arm records no route digest.
    """

    if bundle_id == SIAM_ROUTE_NOT_APPLICABLE:
        return None
    bundle = EXCHANGE_ROUTE_BUNDLES.get(str(bundle_id))
    if bundle is None:
        raise ValueError(f"unregistered exchange route bundle {bundle_id!r}")
    return _sha256_bytes(_canonical_json_bytes(asdict(bundle)))


def _canonical_route_id(value: str) -> str:
    """Validate one ladder route id against the ladder's own route table.

    ``siam_nb4_ladder`` imports this module, so the table is resolved lazily.
    ``ROUTE_TOKENS`` is authoritative for which routes exist.
    """

    from pipelines.time_dynamics.siam_nb4_ladder import ROUTE_TOKENS

    route = str(value).strip()
    if route not in ROUTE_TOKENS:
        raise ValueError(
            f"SIAM Nb=4 package route must be one of {sorted(ROUTE_TOKENS)!r}; "
            f"got {value!r}"
        )
    return route


def _assert_arm_on_route(canonical_arm: str, *, route_id: str) -> str:
    """Return the arm's bundle id after binding it to the ladder route.

    The comparator arm belongs to whichever ladder it is packaged for and
    carries no bundle of its own.  An exchange-family arm must run exactly the
    ladder's route.
    """

    bundle_id = _arm_route_bundle_id(canonical_arm)
    if bundle_id == SIAM_ROUTE_NOT_APPLICABLE:
        return bundle_id
    expected = (
        EXCHANGE_ROUTE_BUNDLE_HISTORICAL
        if route_id == ROUTE_HISTORICAL
        else route_id
    )
    if bundle_id != expected:
        raise ValueError(
            f"SIAM arm {canonical_arm!r} runs exchange route {bundle_id!r}, "
            f"not the packaged ladder route {route_id!r}"
        )
    return bundle_id


def _registered_segment(value: str):
    token = str(value).strip().lower()
    try:
        return _SEGMENT_ALIASES[token]
    except KeyError as exc:
        raise ValueError(
            "SIAM Nb=4 segment must be screen_t0_t1_101 or "
            f"continue_t1_t5_401, got {value!r}"
        ) from exc


def _registered_cut(value: float) -> tuple[float, int]:
    return canonical_registered_cut(value)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable UTF-8 JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


@dataclass(frozen=True)
class SIAMSourceEpochRebaseline:
    """Explicit evidence required to reuse one scientific action identifier.

    This is intentionally not a boolean escape hatch.  The named predecessor
    submission, its package manifest, and its terminal epoch-closure receipt
    must all be digest-bound into the new package.
    """

    predecessor_source_epoch_id: str
    predecessor_submission_receipt: str | Path
    predecessor_package_manifest: str | Path
    predecessor_epoch_closure_receipt: str | Path


def _rebaseline_core_contract(
    *,
    predecessor_source_epoch_id: str,
    predecessor_submission: Mapping[str, Any],
    predecessor_manifest: Mapping[str, Any],
    predecessor_closure: Mapping[str, Any],
    predecessor_submission_sha256: str,
    predecessor_manifest_sha256: str,
    predecessor_closure_sha256: str,
    new_source_epoch: Mapping[str, Any],
    new_package_id: str,
    new_action_id: str,
    new_output_path: str,
) -> dict[str, Any]:
    """Validate and summarize one closed-epoch source rebaseline."""

    old_epoch = _validated_action_id(predecessor_source_epoch_id)
    new_epoch = _validated_action_id(
        str(new_source_epoch.get("source_epoch_id") or "")
    )
    if old_epoch == new_epoch:
        raise ValueError("SIAM source rebaseline must create a new source epoch")
    for label, digest in (
        ("predecessor submission", predecessor_submission_sha256),
        ("predecessor manifest", predecessor_manifest_sha256),
        ("predecessor closure", predecessor_closure_sha256),
    ):
        if not _SHA256.fullmatch(str(digest)):
            raise ValueError(f"SIAM {label} SHA-256 is invalid")

    predecessor_action = _validated_action_id(
        str(predecessor_manifest.get("action_id") or "")
    )
    if predecessor_action != new_action_id:
        raise ValueError(
            "SIAM source rebaseline is only for the identical scientific "
            "action ID; use ordinary global-absence packaging otherwise"
        )
    predecessor_package = _validated_action_id(
        str(predecessor_manifest.get("package_id") or "")
    )
    predecessor_revision = _validated_revision(
        str(predecessor_manifest.get("source_revision") or "")
    )
    predecessor_tree_sha = str(
        predecessor_manifest.get("source_tree_files_sha256") or ""
    )
    if not _SHA256.fullmatch(predecessor_tree_sha):
        raise ValueError("SIAM predecessor manifest has no source-tree digest")
    predecessor_output = _validated_receipt_path(
        predecessor_manifest.get("expected_output"),
        label="predecessor output path",
    )
    if predecessor_package == new_package_id:
        raise ValueError("SIAM source rebaseline must use a new package path")
    if predecessor_output == new_output_path:
        raise ValueError("SIAM source rebaseline must use a new output path")
    if predecessor_revision == new_source_epoch.get("source_revision"):
        raise ValueError("SIAM source rebaseline requires a new source revision")
    if predecessor_tree_sha == new_source_epoch.get(
        "source_tree_files_sha256"
    ):
        raise ValueError("SIAM source rebaseline requires a new source tree")

    expected_submission = {
        "package_id": predecessor_package,
        "action_id": predecessor_action,
        "source_revision": predecessor_revision,
        "source_tree_files_sha256": predecessor_tree_sha,
        "package_manifest_sha256": predecessor_manifest_sha256,
    }
    if predecessor_submission.get("status") not in {"SUBMITTED", "COMPLETED"}:
        raise ValueError(
            "SIAM predecessor submission receipt must record SUBMITTED or "
            "COMPLETED before its epoch can be closed"
        )
    if any(
        predecessor_submission.get(field) != value
        for field, value in expected_submission.items()
    ):
        raise ValueError(
            "SIAM predecessor submission does not bind the named package epoch"
        )

    expected_closure = {
        "schema": SIAM_SOURCE_EPOCH_CLOSURE_V1,
        "status": "CLOSED_SUPERSEDED",
        "source_epoch_id": old_epoch,
        "superseded_by_source_epoch_id": new_epoch,
        "package_id": predecessor_package,
        "action_id": predecessor_action,
        "source_revision": predecessor_revision,
        "source_tree_files_sha256": predecessor_tree_sha,
        "submission_receipt_sha256": predecessor_submission_sha256,
        "package_manifest_sha256": predecessor_manifest_sha256,
    }
    if any(
        predecessor_closure.get(field) != value
        for field, value in expected_closure.items()
    ):
        raise ValueError(
            "SIAM predecessor epoch is not closed/superseded by the new epoch"
        )
    if predecessor_closure.get("closed_to_new_actions") is not True:
        raise ValueError(
            "SIAM predecessor epoch closure must forbid new actions"
        )
    closure_core = dict(predecessor_closure)
    closure_self_digest = str(
        closure_core.pop("closure_receipt_sha256", "") or ""
    )
    if (
        not str(closure_core.get("closed_at") or "").strip()
        or not _SHA256.fullmatch(closure_self_digest)
        or closure_self_digest
        != _sha256_bytes(_canonical_json_bytes(closure_core))
    ):
        raise ValueError(
            "SIAM predecessor epoch closure self-digest is invalid"
        )

    return {
        "predecessor_source_epoch_id": old_epoch,
        "new_source_epoch_id": new_epoch,
        "predecessor_package_id": predecessor_package,
        "scientific_action_id": predecessor_action,
        "predecessor_source_revision": predecessor_revision,
        "predecessor_source_tree_files_sha256": predecessor_tree_sha,
        "predecessor_output": predecessor_output,
        "predecessor_submission_receipt_sha256": predecessor_submission_sha256,
        "predecessor_package_manifest_sha256": predecessor_manifest_sha256,
        "predecessor_epoch_closure_receipt_sha256": predecessor_closure_sha256,
        "checks": {
            "identical_scientific_action_id": True,
            "new_package_path": True,
            "new_output_path": True,
            "new_source_revision": True,
            "new_source_tree": True,
            "predecessor_epoch_closed_to_new_actions": True,
            "predecessor_epoch_superseded_by_new_epoch": True,
        },
    }


def _trajectory_points(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    trajectory = payload.get("trajectory")
    if isinstance(trajectory, dict):
        points = trajectory.get("points")
    else:
        points = trajectory
    if not isinstance(points, list) or not all(
        isinstance(row, dict) for row in points
    ):
        raise ValueError("SIAM run carries no valid trajectory point list")
    return points


def _assert_reference_blind_payload(payload: Mapping[str, Any]) -> None:
    run_lock = dict(payload.get("run_lock") or {})
    physics = dict(run_lock.get("physics") or {})
    for field in ("exact_reference_json", "exact_reference_sha256"):
        if field not in physics or physics.get(field) not in {None, ""}:
            raise ValueError("raw SIAM run embeds an exact-reference run lock")

    trajectory = payload.get("trajectory")
    metadata = (
        dict(trajectory.get("metadata") or {})
        if isinstance(trajectory, dict)
        else {}
    )
    for field in ("reference_energy_json", "seed_reference_energy_json"):
        if field not in metadata or metadata.get(field) not in {None, ""}:
            raise ValueError("raw SIAM run embeds reference-energy inputs")
    for field in (
        "uses_exact_reference_for_decision",
        "uses_future_exact_forecast_for_decision",
        "uses_reference_for_decision",
    ):
        if metadata.get(field) is not False:
            raise ValueError(f"raw SIAM run does not certify {field}=false")

    flow = dict(payload.get("decision_data_flow") or {})
    for field in (
        "uses_exact_reference_for_decision",
        "uses_future_exact_forecast_for_decision",
        "uses_reference_for_decision",
    ):
        if flow.get(field) is not False:
            raise ValueError(
                f"raw SIAM decision flow does not certify {field}=false"
            )


def _assert_complete_pool(payload: Mapping[str, Any], *, block_count: int) -> None:
    summary = dict(payload.get("summary") or {})
    initial = dict(payload.get("initial_state") or {})
    final = dict(payload.get("final_state") or {})
    for label, row in (
        ("summary", summary),
        ("initial_state", initial),
        ("final_state", final),
    ):
        if row.get("candidate_pool_complete") is not True:
            raise ValueError(f"SIAM {label} does not certify a complete pool")
        count = row.get("candidate_pool_term_count")
        if isinstance(count, bool) or not isinstance(count, int):
            raise ValueError(f"SIAM {label} carries no integer pool count")
        if count != int(block_count):
            raise ValueError(
                f"SIAM {label} pool count {count!r} does not match seed {block_count}"
            )


def _assert_recorded_statevectors(
    payload: Mapping[str, Any], *, point_count: int, state_dimension: int
) -> None:
    rows = payload.get("plot_rows")
    if not isinstance(rows, list) or len(rows) != point_count:
        raise ValueError("SIAM raw run does not retain one plot row per checkpoint")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"SIAM plot row {index} is not an object")
        real = row.get("psi_real")
        imag = row.get("psi_imag")
        if (
            not isinstance(real, list)
            or not isinstance(imag, list)
            or len(real) != state_dimension
            or len(real) != len(imag)
        ):
            raise ValueError(
                f"SIAM plot row {index} lacks a complete recorded statevector"
            )
        amplitudes = (*real, *imag)
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            for value in amplitudes
        ):
            raise ValueError(
                f"SIAM plot row {index} has nonnumeric/nonfinite amplitudes"
            )


def _validate_resume_payload(
    payload: Mapping[str, Any],
    *,
    arm: str,
    seed_sha256: str,
    seed_block_count: int,
    seed_state_dimension: int,
) -> dict[str, Any]:
    _assert_reference_blind_payload(payload)
    _assert_complete_pool(payload, block_count=seed_block_count)
    points = _trajectory_points(payload)
    if len(points) != len(SIAM_NB4_SCREEN_SEGMENT.times):
        raise ValueError("SIAM continuation resume is not the registered t<=1 screen")
    times = tuple(float(row.get("time")) for row in points)
    if times != SIAM_NB4_SCREEN_SEGMENT.times:
        raise ValueError("SIAM continuation resume uses the wrong ordered time grid")
    _assert_recorded_statevectors(
        payload,
        point_count=len(points),
        state_dimension=seed_state_dimension,
    )

    resume_state = dict(payload.get("resume_state") or {})
    if resume_state.get("schema") != "paper_ii_resume_state_v2":
        raise ValueError("SIAM continuation requires portable resume schema v2")
    if float(resume_state.get("time", math.nan)) != 1.0:
        raise ValueError("SIAM continuation resume does not terminate at t=1")
    if not isinstance(resume_state.get("coordinates"), list) or not isinstance(
        resume_state.get("controller_state"), dict
    ):
        raise ValueError("SIAM continuation resume lacks controller/support state")

    run_lock = dict(payload.get("run_lock") or {})
    physics = dict(run_lock.get("physics") or {})
    policy = dict(run_lock.get("policy") or {})
    if physics.get("seed_sha256") != seed_sha256:
        raise ValueError("SIAM continuation resume does not bind this portable seed")
    if policy.get("structural_policy") != _arm_dynamics_policy(arm):
        raise ValueError("SIAM continuation resume arm differs from requested arm")
    revision = str(run_lock.get("code_revision") or "")
    if not _REVISION.fullmatch(revision):
        raise ValueError("SIAM continuation resume lacks a committed source revision")
    return {
        "run_id": str(payload.get("run_id") or ""),
        "source_revision": revision,
        "time_final": 1.0,
        "point_count": len(points),
    }


def _validate_resume_source(
    path: Path,
    *,
    arm: str,
    seed_sha256: str,
    seed_block_count: int,
    seed_state_dimension: int,
) -> dict[str, Any]:
    payload = _load_json_object(path, label="SIAM continuation resume source")
    return _validate_resume_payload(
        payload,
        arm=arm,
        seed_sha256=seed_sha256,
        seed_block_count=seed_block_count,
        seed_state_dimension=seed_state_dimension,
    )


def _validate_source_completion_receipt(
    path: Path,
    *,
    trajectory_path: Path,
    arm: str,
    activation_cut: float,
    seed_sha256: str,
    resume_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if path.name != "action_receipt.json":
        raise ValueError(
            "SIAM continuation completion input must be named action_receipt.json"
        )
    receipt = _load_json_object(path, label="SIAM screen action receipt")
    trajectory_sha = _sha256(trajectory_path)
    verified_receipt = validate_siam_nb4_completion_receipt(
        receipt,
        expected_source_revision=str(resume_metadata["source_revision"]),
        expected_seed_sha256=seed_sha256,
        expected_trajectory_sha256=trajectory_sha,
        expected_arm=arm,
        expected_segment=SIAM_NB4_SCREEN_SEGMENT.segment_id,
    )
    if verified_receipt["trajectory_bytes"] != trajectory_path.stat().st_size:
        raise ValueError("SIAM screen action receipt trajectory byte-count mismatch")
    if receipt.get("schema") not in {
        SIAM_NB4_CHTC_RECEIPT_SCHEMA_V3,
        SIAM_NB4_CHTC_RECEIPT_SCHEMA_V4,
        SIAM_NB4_CHTC_RECEIPT_SCHEMA_V5,
    }:
        raise ValueError(
            "SIAM continuation requires a v3/v4/v5 screen action receipt with "
            "the complete ordered-grid validation contract"
        )
    if receipt.get("status") != "COMPLETED":
        raise ValueError(
            "SIAM continuation requires source action receipt status COMPLETED; "
            f"got {receipt.get('status')!r}"
        )
    source_action_id = _validated_action_id(str(receipt.get("action_id") or ""))
    if receipt.get("arm") != arm:
        raise ValueError("SIAM screen action receipt arm mismatch")
    if float(receipt.get("activation_cut", math.nan)) != activation_cut:
        raise ValueError("SIAM screen action receipt activation-cut mismatch")
    if receipt.get("seed_sha256") != seed_sha256:
        raise ValueError("SIAM screen action receipt seed mismatch")
    if receipt.get("source_revision") != resume_metadata["source_revision"]:
        raise ValueError("SIAM screen action receipt source revision mismatch")
    if receipt.get("segment") != SIAM_NB4_SCREEN_SEGMENT.segment_id:
        raise ValueError("SIAM screen action receipt segment mismatch")
    receipt_seed_path = receipt.get("seed_path")
    receipt_output_path = receipt.get("trajectory_json")
    if not isinstance(receipt_seed_path, str) or not receipt_seed_path:
        raise ValueError("SIAM screen action receipt lacks its seed path")
    if not isinstance(receipt_output_path, str) or not receipt_output_path:
        raise ValueError("SIAM screen action receipt lacks its trajectory path")
    expected_command = build_siam_nb4_screen_run(
        seed_path=receipt_seed_path,
        arm=arm,
        activation_cut=activation_cut,
        output_json=receipt_output_path,
        code_revision=str(resume_metadata["source_revision"]),
    )
    expected_run_id = expected_command.run_id
    if receipt.get("run_id") != expected_run_id:
        raise ValueError("SIAM screen action receipt run-id mismatch")
    if receipt.get("exact_reference_blind") is not True:
        raise ValueError("SIAM screen action receipt is not exact-reference blind")
    if receipt.get("qiskit_compilation_requested") is not False:
        raise ValueError("SIAM screen action receipt requested Qiskit compilation")

    if receipt.get("trajectory_sha256") != trajectory_sha:
        raise ValueError("SIAM screen action receipt trajectory SHA-256 mismatch")
    trajectory_bytes = receipt.get("trajectory_bytes")
    if (
        isinstance(trajectory_bytes, bool)
        or not isinstance(trajectory_bytes, int)
        or trajectory_bytes != trajectory_path.stat().st_size
    ):
        raise ValueError("SIAM screen action receipt trajectory byte-count mismatch")
    checks = receipt.get("checks")
    required_checks = (
        "source_revision",
        "uncured_local_subdivisions",
        "resume_schema",
        "trajectory_points",
        "ordered_time_grid",
        "initial_time",
        "terminal_time",
    )
    if not isinstance(checks, dict) or any(
        checks.get(field) is not True for field in required_checks
    ):
        raise ValueError(
            "SIAM screen action receipt lacks affirmative source/time-grid checks"
        )
    runner_argv = receipt.get("runner_argv")
    if not isinstance(runner_argv, list) or not runner_argv or not all(
        isinstance(token, str) for token in runner_argv
    ):
        raise ValueError("SIAM screen action receipt lacks its exact runner argv")
    runner_argv_sha = str(receipt.get("runner_argv_sha256") or "")
    if runner_argv_sha != _argv_sha256(runner_argv):
        raise ValueError("SIAM screen action receipt runner argv digest mismatch")
    if tuple(runner_argv) != tuple(expected_command.argv()):
        raise ValueError(
            "SIAM screen action receipt argv differs from the registered command"
        )
    _validate_runner_argv(
        runner_argv,
        seed_path=receipt_seed_path,
        resume_path=None,
        output_path=receipt_output_path,
        source_revision=str(resume_metadata["source_revision"]),
        segment_times=SIAM_NB4_SCREEN_SEGMENT.times,
    )
    return {
        "source_action_id": source_action_id,
        "source_action_receipt_sha256": _sha256(path),
        "source_action_receipt_schema": receipt["schema"],
        "source_run_id": expected_run_id,
        "source_runner_argv_sha256": runner_argv_sha,
        "source_activation_cut": float(receipt["activation_cut"]),
        "source_trajectory_sha256": trajectory_sha,
        "source_trajectory_bytes": trajectory_path.stat().st_size,
        "source_completion_checks": {
            field: True for field in required_checks
        },
    }


def _validate_runner_argv(
    argv: Sequence[str],
    *,
    seed_path: str,
    resume_path: str | None,
    output_path: str,
    source_revision: str,
    segment_times: Sequence[float],
) -> None:
    parser = _build_parser()
    parsed = parser.parse_args(list(argv))
    if parsed.artifact_json not in {None, ""}:
        raise ValueError("SIAM raw run must not consume a scaffold artifact")
    if parsed.siam_seed_json != seed_path:
        raise ValueError("SIAM raw run resolved the wrong portable seed")
    if parsed.resume_from_run_json != resume_path:
        raise ValueError("SIAM raw run resolved the wrong resume input")
    if parsed.output_json != output_path:
        raise ValueError("SIAM raw run resolved the wrong output")
    if parsed.code_revision != source_revision:
        raise ValueError("SIAM raw run resolved the wrong source revision")
    if tuple(_resolve_times(parsed)) != tuple(segment_times):
        raise ValueError("SIAM raw run resolved the wrong ordered time grid")
    if bool(parsed.enable_drive):
        raise ValueError("SIAM hybridization quench must not enable a drive")
    if parsed.reference_energy_json not in {None, ""} or (
        parsed.seed_reference_energy_json not in {None, ""}
    ):
        raise ValueError("SIAM raw run must remain exact-reference blind")
    if not bool(parsed.record_statevector):
        raise ValueError("SIAM raw run must record checkpoint statevectors")
    if not bool(parsed.require_complete_candidate_pool):
        raise ValueError("SIAM raw run must require the complete logical pool")
    if str(parsed.parameterization_mode) != "logical_shared":
        raise ValueError("SIAM raw run must use logical-shared parameters")
    if str(parsed.diagnostic_append_pool_mode) != "none" or str(
        parsed.normalized_candidate_pool_profile
    ) != "none":
        raise ValueError("SIAM raw run must not substitute a diagnostic pool")


def _execution_contract_from_argv(
    argv: Sequence[str],
    *,
    seed_path: Path,
    resume_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Resolve the exact achieved-config contract implied by one command."""

    try:
        parsed = _build_parser().parse_args(list(argv))
    except SystemExit as exc:  # pragma: no cover - guarded by command tests
        raise ValueError("SIAM registered runner argv cannot be parsed") from exc
    support_config = _support_patch_config_from_args(parsed)
    solve_repair_config = _solve_repair_config_from_args(parsed)
    runtime_input = load_siam_runtime_input(seed_path)
    if resume_payload is not None:
        resume_state = dict(resume_payload.get("resume_state") or {})
        runtime_input = runtime_input_from_resume_state(
            resume_state,
            base_runtime_input=runtime_input,
        )
    state = state_from_scaffold_runtime_input(
        runtime_input,
        parameterization_mode=AP_PARAMETERIZATION_LOGICAL_SHARED,
    )
    return {
        "support_patch_config": support_config.to_json_dict(),
        "solve_repair_config": solve_repair_config.to_json_dict(),
        "candidate_pool_contract": build_candidate_pool_contract(
            state,
            support_config=support_config,
        ),
    }


def validate_siam_nb4_completion_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_source_revision: str | None = None,
    expected_seed_sha256: str | None = None,
    expected_trajectory_sha256: str | None = None,
    expected_arm: str | None = None,
    expected_segment: str | None = None,
    expected_action_json_sha256: str | None = None,
    expected_package_manifest_sha256: str | None = None,
    expected_source_tree_files_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a positive v3/v4/v5 worker receipt without reopening it.

    The receipt is self-describing: its exact registered command is rebuilt
    from the recorded seed, resume, arm, cut, segment, output, and source
    revision.  Callers may additionally bind it to an independently verified
    score or stitch record through the ``expected_*`` arguments.
    """

    document = dict(receipt)
    receipt_schema = document.get("schema")
    if receipt_schema not in {
        SIAM_NB4_CHTC_RECEIPT_SCHEMA_V3,
        SIAM_NB4_CHTC_RECEIPT_SCHEMA_V4,
        SIAM_NB4_CHTC_RECEIPT_SCHEMA_V5,
    }:
        raise ValueError(
            "SIAM completion requires a v3, v4, or v5 screen action receipt "
            "(or the corresponding continuation receipt)"
        )
    route_lineage = receipt_schema == SIAM_NB4_CHTC_RECEIPT_SCHEMA_V5
    _SCHEMA_TRIPLE = {
        SIAM_NB4_CHTC_RECEIPT_SCHEMA_V3: (
            SIAM_NB4_CHTC_ACTION_SCHEMA_V3,
            SIAM_NB4_CHTC_MANIFEST_SCHEMA_V3,
        ),
        SIAM_NB4_CHTC_RECEIPT_SCHEMA_V4: (
            SIAM_NB4_CHTC_ACTION_SCHEMA_V4,
            SIAM_NB4_CHTC_MANIFEST_SCHEMA_V4,
        ),
        SIAM_NB4_CHTC_RECEIPT_SCHEMA_V5: (
            SIAM_NB4_CHTC_ACTION_SCHEMA_V5,
            SIAM_NB4_CHTC_MANIFEST_SCHEMA_V5,
        ),
    }
    action_schema, manifest_schema = _SCHEMA_TRIPLE[receipt_schema]
    if document.get("status") != "COMPLETED":
        raise ValueError(
            "SIAM completion receipt requires status COMPLETED; "
            f"got {document.get('status')!r}"
        )

    package_id = _validated_action_id(str(document.get("package_id") or ""))
    action_id = _validated_action_id(str(document.get("action_id") or ""))
    revision = _validated_revision(str(document.get("source_revision") or ""))
    arm = _canonical_arm(str(document.get("arm") or ""))
    if document.get("arm") != arm:
        raise ValueError("SIAM completion receipt arm is not canonical")
    segment = _registered_segment(str(document.get("segment") or ""))
    cut = float(document.get("activation_cut", math.nan))
    if not math.isfinite(cut) or cut <= 0.0:
        raise ValueError("SIAM completion receipt activation cut is invalid")

    seed_sha256 = str(document.get("seed_sha256") or "")
    trajectory_sha256 = str(document.get("trajectory_sha256") or "")
    if not _SHA256.fullmatch(seed_sha256):
        raise ValueError("SIAM completion receipt seed SHA-256 is invalid")
    if not _SHA256.fullmatch(trajectory_sha256):
        raise ValueError("SIAM completion receipt trajectory SHA-256 is invalid")
    action_json_sha256 = str(document.get("action_json_sha256") or "")
    package_manifest_sha256 = str(
        document.get("package_manifest_sha256") or ""
    )
    source_tree_files_sha256 = str(
        document.get("source_tree_files_sha256") or ""
    )
    if any(
        not _SHA256.fullmatch(value)
        for value in (
            action_json_sha256,
            package_manifest_sha256,
            source_tree_files_sha256,
        )
    ):
        raise ValueError(
            "SIAM completion receipt lacks action/package/source-tree digests"
        )
    source_tree_file_count = document.get("source_tree_file_count")
    if (
        isinstance(source_tree_file_count, bool)
        or not isinstance(source_tree_file_count, int)
        or source_tree_file_count <= 0
    ):
        raise ValueError("SIAM completion receipt source-tree count is invalid")

    action_contract = document.get("action_contract")
    manifest_contract = document.get("package_manifest_contract")
    if not isinstance(action_contract, Mapping) or not isinstance(
        manifest_contract, Mapping
    ):
        raise ValueError(
            "SIAM completion receipt lacks embedded action/package contracts"
        )
    action_document = dict(action_contract)
    manifest_document = dict(manifest_contract)
    if _sha256_bytes(_pretty_json_bytes(action_document)) != action_json_sha256:
        raise ValueError("SIAM completion receipt action-contract digest mismatch")
    if (
        _sha256_bytes(_pretty_json_bytes(manifest_document))
        != package_manifest_sha256
    ):
        raise ValueError("SIAM completion receipt package-contract digest mismatch")
    ladder_resolution = resolve_embedded_ladder(
        activation_cut=cut,
        package_index=action_document.get("activation_cut_ladder_index"),
        action_ladder_sha256=action_document.get(
            "activation_cut_ladder_sha256"
        ),
        manifest_ladder_sha256=manifest_document.get(
            "activation_cut_ladder_sha256"
        ),
        manifest_ladder=manifest_document.get("activation_cut_ladder"),
    )
    source_tree = action_document.get("source_tree")
    if not isinstance(source_tree, Mapping):
        raise ValueError("SIAM completion action lacks its source-tree contract")
    source_rows = source_tree.get("files")
    if (
        source_tree.get("schema") != SIAM_SOURCE_TREE_CONTRACT_V1
        or source_tree.get("source_revision") != revision
        or not isinstance(source_rows, list)
        or not source_rows
        or source_tree.get("file_count") != len(source_rows)
        or source_tree.get("files_sha256")
        != _sha256_bytes(_canonical_json_bytes(source_rows))
        or source_tree.get("files_sha256") != source_tree_files_sha256
        or len(source_rows) != source_tree_file_count
    ):
        raise ValueError("SIAM completion source-tree contract is inconsistent")
    seen_source_paths: set[str] = set()
    source_roots: set[str] = set()
    for row in source_rows:
        if not isinstance(row, Mapping):
            raise ValueError("SIAM completion source-tree row is malformed")
        source_path = str(row.get("path") or "")
        source_member = PurePosixPath(source_path)
        source_kind = row.get("kind")
        source_bytes = row.get("bytes")
        source_mode = row.get("mode")
        if (
            not source_path
            or source_member.is_absolute()
            or ".." in source_member.parts
            or source_member.parts[0] not in {"pipelines", "src"}
            or source_path in seen_source_paths
            or source_kind not in {"file", "symlink"}
            or isinstance(source_bytes, bool)
            or not isinstance(source_bytes, int)
            or source_bytes < 0
            or isinstance(source_mode, bool)
            or not isinstance(source_mode, int)
            or not _SHA256.fullmatch(str(row.get("sha256") or ""))
            or (
                source_kind == "file" and row.get("link_target") is not None
            )
            or (
                source_kind == "symlink"
                and not isinstance(row.get("link_target"), str)
            )
        ):
            raise ValueError("SIAM completion source-tree row is unsafe")
        seen_source_paths.add(source_path)
        source_roots.add(source_member.parts[0])
    if source_roots != {"pipelines", "src"}:
        raise ValueError(
            "SIAM completion source-tree contract does not cover both roots"
        )
    if receipt_schema in {
        SIAM_NB4_CHTC_RECEIPT_SCHEMA_V4,
        SIAM_NB4_CHTC_RECEIPT_SCHEMA_V5,
    } and (
        "source_epoch" not in action_document
        or "source_epoch" not in manifest_document
    ):
        raise ValueError("SIAM v4 completion lacks its source-epoch contract")
    if "source_epoch" in action_document:
        expected_source_epoch = build_siam_source_epoch_contract(
            source_revision=revision,
            source_tree_files_sha256=source_tree_files_sha256,
        )
        expected_scientific_identity = {
            "scope": "source_epoch_id/action_id",
            "source_epoch_id": expected_source_epoch["source_epoch_id"],
            "action_id": action_id,
        }
        if (
            action_document.get("source_epoch") != expected_source_epoch
            or manifest_document.get("source_epoch") != expected_source_epoch
            or document.get("source_epoch") != expected_source_epoch
            or action_document.get("scientific_identity")
            != expected_scientific_identity
            or manifest_document.get("scientific_identity")
            != expected_scientific_identity
            or document.get("scientific_identity")
            != expected_scientific_identity
            or document.get("intentional_source_rebaseline")
            != action_document.get("intentional_source_rebaseline")
            or manifest_document.get("intentional_source_rebaseline")
            != action_document.get("intentional_source_rebaseline")
        ):
            raise ValueError(
                "SIAM completion source-epoch identity is inconsistent"
            )
    if route_lineage:
        # The v5 lineage is self-describing: the arm, its declared route and
        # the route contract digest must agree across all three documents and
        # must still be what the registered arm actually runs.
        route_bundle_id = str(document.get("route_bundle_id") or "")
        expected_bundle_id = _arm_route_bundle_id(arm)
        if route_bundle_id != expected_bundle_id:
            raise ValueError(
                "SIAM completion receipt route bundle is not the one its arm "
                "runs"
            )
        expected_route_digest = _route_contract_sha256(expected_bundle_id)
        if document.get("route_contract_sha256") != expected_route_digest:
            raise ValueError(
                "SIAM completion receipt route contract digest is stale"
            )
        for field in ("route_id", "route_bundle_id", "route_contract_sha256"):
            if (
                action_document.get(field) != document.get(field)
                or manifest_document.get(field) != document.get(field)
            ):
                raise ValueError(
                    "SIAM completion route identity is inconsistent across its "
                    "digest-bound documents"
                )
        try:
            _canonical_route_id(str(document.get("route_id") or ""))
        except ValueError as exc:
            raise ValueError(
                "SIAM completion receipt names an unregistered ladder route"
            ) from exc
    elif any(
        field in document
        for field in ("route_id", "route_bundle_id", "route_contract_sha256")
    ):
        raise ValueError(
            "SIAM v3/v4 completion receipt must not declare a ladder route"
        )
    expected_action_fields = {
        "schema": action_schema,
        "package_id": package_id,
        "action_id": action_id,
        "source_revision": revision,
        "arm": arm,
        "segment": segment.segment_id,
        "activation_cut": cut,
        "seed_path": str(document.get("seed_path") or ""),
        "seed_sha256": seed_sha256,
        "trajectory_json": str(document.get("trajectory_json") or ""),
        "expected_run_id": str(document.get("run_id") or ""),
        "expected_runner_argv_sha256": str(
            document.get("runner_argv_sha256") or ""
        ),
        "exact_reference_blind": True,
        "compile_terminal_cost": False,
        "qiskit_compilation_requested": False,
    }
    execution_contract = action_document.get("expected_execution_contract")
    if any(
        action_document.get(field) != value
        for field, value in expected_action_fields.items()
    ) or not isinstance(execution_contract, Mapping) or any(
        not isinstance(execution_contract.get(field), Mapping)
        for field in (
            "support_patch_config",
            "solve_repair_config",
            "candidate_pool_contract",
        )
    ):
        raise ValueError("SIAM completion action contract differs from its receipt")
    expected_manifest_fields = {
        "schema": manifest_schema,
        "package_id": package_id,
        "action_id": action_id,
        "source_revision": revision,
        "action_json_sha256": action_json_sha256,
        "runner_argv_sha256": str(document.get("runner_argv_sha256") or ""),
        "expected_run_id": str(document.get("run_id") or ""),
        "expected_output": str(document.get("trajectory_json") or ""),
        "source_tree_contract": SIAM_SOURCE_TREE_CONTRACT_V1,
        "source_tree_files_sha256": source_tree_files_sha256,
        "source_tree_file_count": source_tree_file_count,
    }
    if any(
        manifest_document.get(field) != value
        for field, value in expected_manifest_fields.items()
    ):
        raise ValueError("SIAM completion package contract differs from its receipt")
    manifest_files = manifest_document.get("files")
    action_file_row = (
        manifest_files.get("action")
        if isinstance(manifest_files, Mapping)
        else None
    )
    if not isinstance(action_file_row, Mapping) or action_file_row.get(
        "sha256"
    ) != action_json_sha256:
        raise ValueError("SIAM completion package does not bind its action file")
    trajectory_bytes = document.get("trajectory_bytes")
    if (
        isinstance(trajectory_bytes, bool)
        or not isinstance(trajectory_bytes, int)
        or trajectory_bytes <= 0
    ):
        raise ValueError("SIAM completion receipt trajectory byte count is invalid")

    seed_path = _validated_receipt_path(document.get("seed_path"), label="seed path")
    output_path = _validated_receipt_path(
        document.get("trajectory_json"), label="trajectory path"
    )
    if document.get("exact_reference_blind") is not True:
        raise ValueError("SIAM completion receipt is not exact-reference blind")
    if document.get("qiskit_compilation_requested") is not False:
        raise ValueError("SIAM completion receipt requested Qiskit compilation")

    checks = document.get("checks")
    required_checks = (
        "source_revision",
        "uncured_local_subdivisions",
        "resume_schema",
        "trajectory_points",
        "ordered_time_grid",
        "initial_time",
        "terminal_time",
        "configuration_lock",
        "segment_initial_state",
        "segment_initial_energy",
        "state_norm_sector",
        "resume_reconstruction",
        "resume_support_parameters",
        "resume_terminal_statevector",
        "controller_resume_reconstruction",
        "controller_resume_terminal_state",
        *(
            ("initial_seed_state", "initial_seed_energy")
            if segment is SIAM_NB4_SCREEN_SEGMENT
            else ("resume_boundary_state", "resume_boundary_energy")
        ),
    )
    if not isinstance(checks, Mapping) or any(
        checks.get(field) is not True for field in required_checks
    ):
        raise ValueError(
            "SIAM completion receipt lacks affirmative source/time-grid checks"
        )

    resume_proof = document.get("resume_terminal_proof")
    if not isinstance(resume_proof, Mapping) or resume_proof.get(
        "schema"
    ) != SIAM_NB4_RESUME_TERMINAL_PROOF_V1:
        raise ValueError(
            "SIAM completion receipt lacks its terminal resume proof"
        )
    if (
        resume_proof.get("resume_schema") != RESUME_SCHEMA_V2
        or resume_proof.get("controller_schema")
        != CONTROLLER_RESUME_SCHEMA_V1
        or float(resume_proof.get("terminal_time", math.nan))
        != float(segment.times[-1])
    ):
        raise ValueError(
            "SIAM completion receipt terminal resume proof has wrong identity"
        )
    logical_count = resume_proof.get("logical_coordinate_count")
    pauli_count = resume_proof.get("runtime_pauli_coordinate_count")
    velocity_count = resume_proof.get("previous_accepted_velocity_count")
    checkpoint_index = resume_proof.get("terminal_checkpoint_index")
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                logical_count,
                pauli_count,
                velocity_count,
                checkpoint_index,
            )
        )
        or int(pauli_count) < int(logical_count)
        or int(velocity_count) != int(logical_count)
    ):
        raise ValueError(
            "SIAM completion receipt terminal resume proof has invalid counts"
        )
    proof_digests = (
        "support_contract_sha256",
        "theta_contract_sha256",
        "controller_state_sha256",
        "reconstructed_state_semantic_sha256",
        "terminal_state_semantic_sha256",
    )
    if any(
        not _SHA256.fullmatch(str(resume_proof.get(field) or ""))
        for field in proof_digests
    ) or resume_proof.get("reconstructed_state_semantic_sha256") != (
        resume_proof.get("terminal_state_semantic_sha256")
    ):
        raise ValueError(
            "SIAM completion receipt terminal resume proof has invalid digests"
        )
    state_infidelity = resume_proof.get("reconstructed_state_infidelity")
    if (
        isinstance(state_infidelity, bool)
        or not isinstance(state_infidelity, Real)
        or not math.isfinite(float(state_infidelity))
        or float(state_infidelity) < 0.0
        or float(state_infidelity) > 1.0e-12
    ):
        raise ValueError(
            "SIAM completion receipt terminal resume proof failed state parity"
        )

    runner_argv = document.get("runner_argv")
    if not isinstance(runner_argv, list) or not runner_argv or not all(
        isinstance(token, str) for token in runner_argv
    ):
        raise ValueError("SIAM completion receipt lacks its exact runner argv")
    runner_argv_sha256 = str(document.get("runner_argv_sha256") or "")
    if runner_argv_sha256 != _argv_sha256(runner_argv):
        raise ValueError("SIAM completion receipt runner argv digest mismatch")
    try:
        parsed = _build_parser().parse_args(list(runner_argv))
    except SystemExit as exc:
        raise ValueError(
            "SIAM completion receipt runner argv cannot be parsed"
        ) from exc
    resume_path = parsed.resume_from_run_json
    if resume_path not in {None, ""}:
        resume_path = _validated_receipt_path(resume_path, label="resume path")
    else:
        resume_path = None
    if segment is SIAM_NB4_SCREEN_SEGMENT:
        if resume_path is not None:
            raise ValueError("SIAM screen completion receipt unexpectedly resumes")
        expected_command = build_siam_nb4_screen_run(
            seed_path=seed_path,
            arm=arm,
            activation_cut=cut,
            output_json=output_path,
            code_revision=revision,
        )
    else:
        if resume_path is None:
            raise ValueError("SIAM continuation completion receipt lacks resume input")
        expected_command = build_siam_nb4_continuation_run(
            seed_path=seed_path,
            arm=arm,
            resume_from_run_json=resume_path,
            activation_cut=cut,
            output_json=output_path,
            code_revision=revision,
        )
    if tuple(runner_argv) != tuple(expected_command.argv()):
        raise ValueError(
            "SIAM completion receipt argv differs from the registered command"
        )
    if document.get("run_id") != expected_command.run_id:
        raise ValueError("SIAM completion receipt run-id mismatch")
    _validate_runner_argv(
        runner_argv,
        seed_path=seed_path,
        resume_path=resume_path,
        output_path=output_path,
        source_revision=revision,
        segment_times=segment.times,
    )

    expected_values = {
        "source revision": (revision, expected_source_revision),
        "seed SHA-256": (seed_sha256, expected_seed_sha256),
        "trajectory SHA-256": (trajectory_sha256, expected_trajectory_sha256),
        "arm": (arm, None if expected_arm is None else _canonical_arm(expected_arm)),
        "segment": (
            segment.segment_id,
            None
            if expected_segment is None
            else _registered_segment(expected_segment).segment_id,
        ),
        "action JSON SHA-256": (
            action_json_sha256,
            expected_action_json_sha256,
        ),
        "package manifest SHA-256": (
            package_manifest_sha256,
            expected_package_manifest_sha256,
        ),
        "source-tree SHA-256": (
            source_tree_files_sha256,
            expected_source_tree_files_sha256,
        ),
    }
    for label, (observed, expected) in expected_values.items():
        if expected is not None and observed != expected:
            raise ValueError(f"SIAM completion receipt {label} mismatch")

    return {
        "schema": receipt_schema,
        "package_id": package_id,
        "action_id": action_id,
        "run_id": expected_command.run_id,
        "source_revision": revision,
        "seed_sha256": seed_sha256,
        "trajectory_sha256": trajectory_sha256,
        "trajectory_bytes": int(trajectory_bytes),
        "arm": arm,
        "activation_cut": cut,
        "activation_cut_canonical_ladder_index": (
            ladder_resolution.canonical_index
        ),
        "activation_cut_package_ladder_index": ladder_resolution.package_index,
        "activation_cut_package_ladder_contract": (
            ladder_resolution.package_contract
        ),
        "activation_cut_identity_migrated": ladder_resolution.migrated,
        "segment": segment.segment_id,
        "runner_argv_sha256": runner_argv_sha256,
        "action_json_sha256": action_json_sha256,
        "package_manifest_sha256": package_manifest_sha256,
        "source_tree_files_sha256": source_tree_files_sha256,
        "source_tree_file_count": int(source_tree_file_count),
        "checks": {field: True for field in required_checks},
        "resume_terminal_proof": dict(resume_proof),
    }


def _file_row(path: Path, *, repo_root: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(repo_root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


@dataclass(frozen=True)
class PreparedSIAMCHTCAction:
    """Paths and outer digests for one inert, not-yet-submitted package."""

    package_dir: Path
    action: Path
    manifest: Path
    worker: Path
    wrapper: Path
    submit: Path
    seed: Path
    resume: Path | None
    resume_action_receipt: Path | None
    action_sha256: str
    manifest_sha256: str


def prepare_siam_nb4_chtc_action(
    *,
    repo_root: str | Path,
    package_dir: str | Path,
    seed_path: str | Path,
    image_path: str | Path,
    source_revision: str,
    action_id: str,
    activation_cut: float,
    arm: str,
    segment: str,
    route_id: str = ROUTE_HISTORICAL,
    resume_from_run_json: str | Path | None = None,
    resume_action_receipt_json: str | Path | None = None,
    request_cpus: int = 2,
    request_memory_gb: int = 10,
    request_disk_gb: int = 40,
    max_runtime_seconds: int = 43200,
    prepared_at: str | None = None,
    intentional_source_rebaseline: SIAMSourceEpochRebaseline | None = None,
) -> PreparedSIAMCHTCAction:
    """Materialize one immutable SIAM screen or continuation CHTC package.

    The package is written through a sibling temporary directory and installed
    only when every digest and command invariant has passed.  Existing package
    or output paths are never overwritten.

    ``route_id`` names the ladder's exchange route.  The historical route emits
    the frozen v4 documents byte for byte; a registered non-historical route
    emits v5 documents that additionally carry the route bundle id and its
    contract digest, and requires the caller's action id to be exactly the one
    ``siam_nb4_ladder.siam_nb4_action_id`` builds for that rung.
    """

    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise ValueError(f"repo_root is not a directory: {root}")
    out = Path(package_dir)
    out = out.resolve() if out.is_absolute() else (root / out).resolve()
    package_rel = _relative_transfer_path(root, out, label="package_dir")
    if out.exists():
        raise FileExistsError(f"refusing to replace SIAM CHTC package: {out}")
    if not _ACTION_ID.fullmatch(out.name):
        raise ValueError(f"package directory name is not CHTC-safe: {out.name!r}")

    revision = _validated_revision(source_revision)
    _verify_source_revision(root, revision)
    source_tree = _build_source_tree_contract(root, revision)
    source_epoch = build_siam_source_epoch_contract(
        source_revision=revision,
        source_tree_files_sha256=str(source_tree["files_sha256"]),
    )
    action_name = _validated_action_id(action_id)
    canonical_arm = _canonical_arm(arm)
    registered_segment = _registered_segment(segment)
    cut, cut_index = _registered_cut(activation_cut)
    canonical_route = _canonical_route_id(route_id)
    route_bundle_id = _assert_arm_on_route(canonical_arm, route_id=canonical_route)
    route_contract_sha = _route_contract_sha256(route_bundle_id)
    historical_route = canonical_route == ROUTE_HISTORICAL
    if not historical_route:
        from pipelines.time_dynamics.siam_nb4_ladder import siam_nb4_action_id

        expected_action_id = siam_nb4_action_id(
            arm=_LADDER_LANES[canonical_arm],
            segment=registered_segment.segment_id,
            activation_cut=cut,
            route_id=canonical_route,
        )
        if action_name != expected_action_id:
            raise ValueError(
                "SIAM Nb=4 route package action id must be the ladder identity "
                f"{expected_action_id!r}; got {action_name!r}"
            )

    source_seed = Path(seed_path).resolve()
    if not source_seed.is_file() or source_seed.stat().st_size <= 0:
        raise ValueError(f"SIAM portable seed is missing or empty: {source_seed}")
    seed_manifest = load_siam_seed_manifest(source_seed)
    if seed_manifest.schema != SIAM_SEED_SCHEMA_V2:
        raise ValueError(
            "SIAM CHTC packages require portable seed schema v2, got "
            f"{seed_manifest.schema!r}"
        )
    seed_content = source_seed.read_bytes()
    seed_sha = _sha256_bytes(seed_content)
    seed_block_count = int(seed_manifest.pool["block_count"])
    seed_state_dimension = 1 << int(seed_manifest.layout["n_qubits"])

    source_image_rel = _relative_transfer_path(root, image_path, label="image_path")
    source_image = root / source_image_rel
    if not source_image.is_file() or source_image.stat().st_size <= 0:
        raise ValueError(f"SIAM runtime image is missing or empty: {source_image}")
    image_sha = _sha256(source_image)
    image_bytes = source_image.stat().st_size

    resume_source: Path | None = None
    resume_content: bytes | None = None
    resume_payload_for_contract: dict[str, Any] | None = None
    resume_metadata: dict[str, Any] | None = None
    completion_receipt_source: Path | None = None
    completion_receipt_content: bytes | None = None
    completion_receipt_metadata: dict[str, Any] | None = None
    if registered_segment.requires_resume:
        if resume_from_run_json in {None, ""}:
            raise ValueError("SIAM continuation package requires a screen run JSON")
        if resume_action_receipt_json in {None, ""}:
            raise ValueError(
                "SIAM continuation package requires an explicit completed "
                "source action_receipt.json"
            )
        resume_source = Path(str(resume_from_run_json)).resolve()
        if not resume_source.is_file() or resume_source.stat().st_size <= 0:
            raise ValueError(
                f"SIAM continuation resume is missing or empty: {resume_source}"
            )
        resume_metadata = _validate_resume_source(
            resume_source,
            arm=canonical_arm,
            seed_sha256=seed_sha,
            seed_block_count=seed_block_count,
            seed_state_dimension=seed_state_dimension,
        )
        resume_content = resume_source.read_bytes()
        parsed_resume = json.loads(resume_content.decode("utf-8"))
        if not isinstance(parsed_resume, dict):
            raise ValueError("SIAM continuation resume source must be an object")
        resume_payload_for_contract = parsed_resume
        completion_receipt_source = Path(
            str(resume_action_receipt_json)
        ).resolve()
        if (
            not completion_receipt_source.is_file()
            or completion_receipt_source.stat().st_size <= 0
        ):
            raise ValueError(
                "SIAM continuation source action receipt is missing or empty: "
                f"{completion_receipt_source}"
            )
        completion_receipt_metadata = _validate_source_completion_receipt(
            completion_receipt_source,
            trajectory_path=resume_source,
            arm=canonical_arm,
            activation_cut=cut,
            seed_sha256=seed_sha,
            resume_metadata=resume_metadata,
        )
        completion_receipt_content = completion_receipt_source.read_bytes()
    elif resume_from_run_json not in {None, ""} or resume_action_receipt_json not in {
        None,
        "",
    }:
        raise ValueError(
            "SIAM screen package must start from the immutable seed without "
            "resume inputs"
        )

    output_rel = (
        Path("raw_outputs") / out.name / action_name / "run.json"
    ).as_posix()
    if (root / output_rel).exists():
        raise FileExistsError(f"refusing to overwrite SIAM raw output: {output_rel}")
    seed_rel = (Path(package_rel) / "input" / "seed.json").as_posix()
    resume_rel = (
        (Path(package_rel) / "input" / "resume" / "screen_run.json").as_posix()
        if registered_segment.requires_resume
        else None
    )
    completion_receipt_rel = (
        (
            Path(package_rel)
            / "input"
            / "resume"
            / "action_receipt.json"
        ).as_posix()
        if registered_segment.requires_resume
        else None
    )
    rebaseline_payload: dict[str, Any] | None = None
    rebaseline_contents: dict[str, bytes] = {}
    if intentional_source_rebaseline is not None:
        if not isinstance(
            intentional_source_rebaseline, SIAMSourceEpochRebaseline
        ):
            raise TypeError(
                "intentional_source_rebaseline must be a "
                "SIAMSourceEpochRebaseline evidence object"
            )
        rebaseline_sources = {
            "predecessor_submission_receipt": Path(
                intentional_source_rebaseline.predecessor_submission_receipt
            ),
            "predecessor_package_manifest": Path(
                intentional_source_rebaseline.predecessor_package_manifest
            ),
            "predecessor_epoch_closure_receipt": Path(
                intentional_source_rebaseline.predecessor_epoch_closure_receipt
            ),
        }
        rebaseline_documents: dict[str, dict[str, Any]] = {}
        rebaseline_artifacts: dict[str, dict[str, Any]] = {}
        for label, source in rebaseline_sources.items():
            source_rel = _relative_transfer_path(
                root, source, label=label
            )
            resolved_source = root / source_rel
            if not resolved_source.is_file() or resolved_source.stat().st_size <= 0:
                raise ValueError(
                    f"SIAM source-rebaseline evidence is missing: {resolved_source}"
                )
            content = resolved_source.read_bytes()
            packaged_rel = (
                Path(package_rel)
                / "input"
                / "source_rebaseline"
                / f"{label}.json"
            ).as_posix()
            rebaseline_contents[label] = content
            rebaseline_documents[label] = _load_json_object(
                resolved_source, label=f"SIAM {label}"
            )
            rebaseline_artifacts[label] = {
                "source_path": source_rel,
                "packaged_path": packaged_rel,
                "bytes": len(content),
                "sha256": _sha256_bytes(content),
            }
        rebaseline_core = _rebaseline_core_contract(
            predecessor_source_epoch_id=(
                intentional_source_rebaseline.predecessor_source_epoch_id
            ),
            predecessor_submission=rebaseline_documents[
                "predecessor_submission_receipt"
            ],
            predecessor_manifest=rebaseline_documents[
                "predecessor_package_manifest"
            ],
            predecessor_closure=rebaseline_documents[
                "predecessor_epoch_closure_receipt"
            ],
            predecessor_submission_sha256=rebaseline_artifacts[
                "predecessor_submission_receipt"
            ]["sha256"],
            predecessor_manifest_sha256=rebaseline_artifacts[
                "predecessor_package_manifest"
            ]["sha256"],
            predecessor_closure_sha256=rebaseline_artifacts[
                "predecessor_epoch_closure_receipt"
            ]["sha256"],
            new_source_epoch=source_epoch,
            new_package_id=out.name,
            new_action_id=action_name,
            new_output_path=output_rel,
        )
        rebaseline_payload = {
            "schema": SIAM_SOURCE_EPOCH_REBASELINE_V1,
            "mode": "intentional_source_rebaseline",
            **rebaseline_core,
            "artifacts": rebaseline_artifacts,
        }

    if registered_segment is SIAM_NB4_SCREEN_SEGMENT:
        command = build_siam_nb4_screen_run(
            seed_path=seed_rel,
            arm=canonical_arm,
            activation_cut=cut,
            output_json=output_rel,
            code_revision=revision,
        )
    else:
        assert resume_rel is not None
        command = build_siam_nb4_continuation_run(
            seed_path=seed_rel,
            arm=canonical_arm,
            resume_from_run_json=resume_rel,
            activation_cut=cut,
            output_json=output_rel,
            code_revision=revision,
        )
    argv = tuple(command.argv())
    _validate_runner_argv(
        argv,
        seed_path=seed_rel,
        resume_path=resume_rel,
        output_path=output_rel,
        source_revision=revision,
        segment_times=registered_segment.times,
    )
    argv_sha = _argv_sha256(argv)
    execution_contract = _execution_contract_from_argv(
        argv,
        seed_path=source_seed,
        resume_payload=resume_payload_for_contract,
    )
    ladder_values = list(ladder_payload(SIAM_NB4_ACTIVATION_CUT_LADDER))
    ladder_sha = SIAM_NB4_CANONICAL_LADDER_SHA256

    action_schema = (
        SIAM_NB4_CHTC_ACTION_SCHEMA_V4
        if historical_route
        else SIAM_NB4_CHTC_ACTION_SCHEMA_V5
    )
    manifest_schema = (
        SIAM_NB4_CHTC_MANIFEST_SCHEMA_V4
        if historical_route
        else SIAM_NB4_CHTC_MANIFEST_SCHEMA_V5
    )
    action_payload: dict[str, Any] = {
        "schema": action_schema,
        "package_id": out.name,
        "action_id": action_name,
        "source_revision": revision,
        "source_tree": source_tree,
        "source_epoch": source_epoch,
        "scientific_identity": {
            "scope": "source_epoch_id/action_id",
            "source_epoch_id": source_epoch["source_epoch_id"],
            "action_id": action_name,
        },
        "intentional_source_rebaseline": rebaseline_payload,
        "arm": canonical_arm,
        "method_label": _arm_method_label(canonical_arm),
        "segment": registered_segment.segment_id,
        "activation_cut": cut,
        "activation_cut_ladder_index": cut_index,
        "activation_cut_ladder_sha256": ladder_sha,
        "seed_path": seed_rel,
        "seed_schema": SIAM_SEED_SCHEMA_V2,
        "seed_sha256": seed_sha,
        "seed_pool_block_count": seed_block_count,
        "seed_state_dimension": seed_state_dimension,
        "resume": None,
        "trajectory_json": output_rel,
        "expected_run_id": command.run_id,
        "expected_runner_argv_sha256": argv_sha,
        "expected_execution_contract": execution_contract,
        "expected_point_count": len(registered_segment.times),
        "expected_time_initial": registered_segment.times[0],
        "expected_time_final": registered_segment.times[-1],
        "exact_reference_blind": True,
        "compile_terminal_cost": False,
        "qiskit_compilation_requested": False,
        "record_statevectors": True,
        "require_complete_candidate_pool": True,
        "output_uniqueness_scope": "package_id/action_id",
    }
    if not historical_route:
        action_payload["route_id"] = canonical_route
        action_payload["route_bundle_id"] = route_bundle_id
        action_payload["route_contract_sha256"] = route_contract_sha
    if (
        resume_content is not None
        and resume_metadata is not None
        and completion_receipt_content is not None
        and completion_receipt_metadata is not None
    ):
        assert resume_rel is not None and resume_source is not None
        assert completion_receipt_rel is not None
        assert completion_receipt_source is not None
        action_payload["resume"] = {
            "source_path": str(resume_source),
            "packaged_path": resume_rel,
            "sha256": _sha256_bytes(resume_content),
            "source_action_receipt_path": str(completion_receipt_source),
            "packaged_action_receipt_path": completion_receipt_rel,
            **completion_receipt_metadata,
            **resume_metadata,
        }

    action_content = _pretty_json_bytes(action_payload)
    action_sha = _sha256_bytes(action_content)
    action_rel = (Path(package_rel) / "input" / "action.json").as_posix()
    manifest_rel = (Path(package_rel) / "input" / "manifest.json").as_posix()
    worker_rel = (Path(package_rel) / "run_action.py").as_posix()
    wrapper_rel = (Path(package_rel) / "run_action_apptainer.sh").as_posix()
    submit_rel = (Path(package_rel) / "submit.sub").as_posix()

    worker_content = (
        "#!/usr/bin/env python3\n"
        '"""Execute one digest-bound exact-blind SIAM CHTC action."""\n'
        "from pathlib import Path\n"
        "import sys\n"
        "from pipelines.time_dynamics.siam_chtc_package import "
        "run_prepared_siam_action\n\n"
        f"PACKAGE_DIR = Path({package_rel!r})\n"
        f"ACTION_ID = {action_name!r}\n"
        f"ACTION_SHA256 = {action_sha!r}\n"
        f"SOURCE_REVISION = {revision!r}\n\n"
        "if __name__ == '__main__':\n"
        "    if len(sys.argv) != 2 or sys.argv[1] != ACTION_ID:\n"
        "        raise SystemExit(f'usage: run_action.py {ACTION_ID}')\n"
        "    raise SystemExit(run_prepared_siam_action(\n"
        "        package_dir=PACKAGE_DIR,\n"
        "        expected_action_id=ACTION_ID,\n"
        "        expected_action_sha256=ACTION_SHA256,\n"
        "        expected_source_revision=SOURCE_REVISION,\n"
        "    ))\n"
    ).encode("utf-8")
    worker_sha = _sha256_bytes(worker_content)

    quoted_image = shlex.quote(source_image_rel)
    quoted_worker = shlex.quote(worker_rel)
    wrapper_content = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        f"ACTION_ID={shlex.quote(action_name)}\n"
        'if [[ "${1:-}" != "$ACTION_ID" || "$#" -ne 1 ]]; then\n'
        '  echo "usage: run_action_apptainer.sh $ACTION_ID" >&2\n'
        "  exit 2\n"
        "fi\n"
        "mkdir -p raw_outputs\n"
        f"WORKER={quoted_worker}\n"
        f'EXPECTED_WORKER_SHA256="{worker_sha}"\n'
        f'IMAGE="${{PROJECT_IMAGE:-{quoted_image}}}"\n'
        f"EXPECTED_IMAGE_BYTES={image_bytes}\n"
        f'EXPECTED_IMAGE_SHA256="{image_sha}"\n\n'
        'if [[ ! -f "$WORKER" ]]; then\n'
        '  echo "Missing digest-bound SIAM worker: $WORKER" >&2\n'
        "  exit 2\n"
        "fi\n"
        'if [[ "$(sha256sum "$WORKER" | awk \'{print $1}\')" '
        '!= "$EXPECTED_WORKER_SHA256" ]]; then\n'
        '  echo "SIAM worker SHA-256 mismatch" >&2\n'
        "  exit 2\n"
        "fi\n"
        'if [[ ! -f "$IMAGE" ]]; then\n'
        '  echo "Missing Apptainer image: $IMAGE" >&2\n'
        "  exit 2\n"
        "fi\n"
        'if [[ "$(wc -c < "$IMAGE" | tr -d " ")" != "$EXPECTED_IMAGE_BYTES" ]]; then\n'
        '  echo "Apptainer image byte count mismatch" >&2\n'
        "  exit 2\n"
        "fi\n"
        'if [[ "$(sha256sum "$IMAGE" | awk \'{print $1}\')" '
        '!= "$EXPECTED_IMAGE_SHA256" ]]; then\n'
        '  echo "Apptainer image SHA-256 mismatch" >&2\n'
        "  exit 2\n"
        "fi\n"
        "if command -v apptainer >/dev/null 2>&1; then\n"
        '  APPTAINER_BIN="$(command -v apptainer)"\n'
        "elif command -v singularity >/dev/null 2>&1; then\n"
        '  APPTAINER_BIN="$(command -v singularity)"\n'
        "else\n"
        '  echo "No apptainer/singularity on this execute node." >&2\n'
        "  exit 127\n"
        "fi\n\n"
        'ROOT="$PWD"\n'
        '"$APPTAINER_BIN" exec --cleanenv --bind "$ROOT:/work" "$IMAGE" \\\n'
        f"  bash -lc 'cd /work && PYTHONPATH=. PYTHONUNBUFFERED=1 python3 "
        f"{quoted_worker} \"$@\"' -- \"$ACTION_ID\"\n"
    ).encode("utf-8")

    # Keep the fully hyphenated pre-route form for every historical batch name,
    # byte for byte; the compiled-Q arm carries its arm id verbatim.  Both forms
    # are exactly what ``siam_submission_intent_bridge`` accepts.
    if canonical_arm in _VERBATIM_BATCH_ARMS:
        batch_name = (
            f"siam-nb4-{registered_segment.segment_id.replace('_', '-')}-"
            f"{canonical_arm}-rung{cut_index}"
        )
    else:
        batch_name = (
            f"siam-nb4-{registered_segment.segment_id}-{canonical_arm}-"
            f"rung{cut_index}"
        ).replace("_", "-")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (
            request_cpus,
            request_memory_gb,
            request_disk_gb,
            max_runtime_seconds,
        )
    ):
        raise ValueError("SIAM CHTC resource requests must be positive integers")
    submit_content = (
        "universe = vanilla\n"
        f"executable = {wrapper_rel}\n"
        f"arguments = {action_name}\n"
        "should_transfer_files = YES\n"
        "when_to_transfer_output = ON_EXIT\n"
        "transfer_executable = True\n"
        "preserve_relative_paths = True\n"
        f"transfer_input_files = pipelines, src, {package_rel}, {source_image_rel}\n"
        "transfer_output_files = raw_outputs\n"
        f"log = {package_rel}/logs/{batch_name}.$(Cluster).$(Process).log\n"
        f"output = {package_rel}/logs/{batch_name}.$(Cluster).$(Process).out\n"
        f"error = {package_rel}/logs/{batch_name}.$(Cluster).$(Process).err\n"
        "stream_output = False\n"
        "stream_error = False\n"
        "requirements = TARGET.HasSIF\n"
        f"request_cpus = {int(request_cpus)}\n"
        f"request_memory = {int(request_memory_gb)}GB\n"
        f"request_disk = {int(request_disk_gb)}GB\n"
        f'+SIAMSourceEpochId = "{source_epoch["source_epoch_id"]}"\n'
        f'+SIAMScientificActionId = "{action_name}"\n'
        "+SIAMIntentionalSourceRebaseline = "
        f"{'True' if rebaseline_payload is not None else 'False'}\n"
        f"+MaxRuntime = {int(max_runtime_seconds)}\n"
        f'+JobBatchName = "{batch_name}"\n'
        "queue 1\n"
    ).encode("utf-8")

    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{out.name}.", dir=str(out.parent))
    )
    try:
        temp_action = temporary / "input" / "action.json"
        temp_seed = temporary / "input" / "seed.json"
        temp_manifest = temporary / "input" / "manifest.json"
        temp_worker = temporary / "run_action.py"
        temp_wrapper = temporary / "run_action_apptainer.sh"
        temp_submit = temporary / "submit.sub"
        temp_resume = temporary / "input" / "resume" / "screen_run.json"
        temp_completion_receipt = (
            temporary / "input" / "resume" / "action_receipt.json"
        )
        _write_bytes(temp_action, action_content)
        _write_bytes(temp_seed, seed_content)
        if resume_content is not None:
            _write_bytes(temp_resume, resume_content)
        if completion_receipt_content is not None:
            _write_bytes(temp_completion_receipt, completion_receipt_content)
        for label, content in rebaseline_contents.items():
            _write_bytes(
                temporary
                / "input"
                / "source_rebaseline"
                / f"{label}.json",
                content,
            )
        _write_bytes(temp_worker, worker_content, executable=True)
        _write_bytes(temp_wrapper, wrapper_content, executable=True)
        _write_bytes(temp_submit, submit_content)
        _write_bytes(temporary / "logs" / ".gitkeep", b"")

        files: dict[str, Any] = {
            "action": _file_row(temp_action, repo_root=temporary.parent),
            "seed": _file_row(temp_seed, repo_root=temporary.parent),
            "worker": _file_row(temp_worker, repo_root=temporary.parent),
            "wrapper": _file_row(temp_wrapper, repo_root=temporary.parent),
            "submit_descriptor": _file_row(
                temp_submit, repo_root=temporary.parent
            ),
        }
        # Replace the random temporary prefix with immutable final paths.
        for row in files.values():
            suffix = Path(row["path"]).relative_to(temporary.name)
            row["path"] = (Path(package_rel) / suffix).as_posix()
        if resume_content is not None:
            files["resume"] = _file_row(temp_resume, repo_root=temporary.parent)
            files["resume"]["path"] = resume_rel
        if completion_receipt_content is not None:
            files["resume_action_receipt"] = _file_row(
                temp_completion_receipt, repo_root=temporary.parent
            )
            files["resume_action_receipt"]["path"] = completion_receipt_rel
        for label in rebaseline_contents:
            temp_evidence = (
                temporary
                / "input"
                / "source_rebaseline"
                / f"{label}.json"
            )
            files[label] = _file_row(
                temp_evidence, repo_root=temporary.parent
            )
            assert rebaseline_payload is not None
            files[label]["path"] = rebaseline_payload["artifacts"][label][
                "packaged_path"
            ]

        if rebaseline_payload is None:
            action_identity_preflight = [
                "verify_action_id_absent_from_prior_packages_and_scheduler_history"
            ]
        else:
            action_identity_preflight = [
                "verify_composite_source_epoch_action_id_absent_from_prior_packages_and_scheduler_history",
                "verify_historical_same_action_ids_use_distinct_source_epochs_and_bind_immediate_predecessor",
                "verify_predecessor_epoch_closed_or_superseded",
            ]

        manifest_payload = {
            "schema": manifest_schema,
            "status": "PREPARED_NOT_SUBMITTED",
            "prepared_at": prepared_at
            or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "package_id": out.name,
            "action_count": 1,
            "action_id": action_name,
            "source_revision": revision,
            "source_tree_contract": SIAM_SOURCE_TREE_CONTRACT_V1,
            "source_tree_files_sha256": source_tree["files_sha256"],
            "source_tree_file_count": source_tree["file_count"],
            "source_epoch": source_epoch,
            "scientific_identity": action_payload["scientific_identity"],
            "intentional_source_rebaseline": rebaseline_payload,
            "action_json_sha256": action_sha,
            "runner_argv_sha256": argv_sha,
            "expected_run_id": command.run_id,
            "expected_output": output_rel,
            "seed_schema": SIAM_SEED_SCHEMA_V2,
            "seed_sha256": seed_sha,
            "activation_cut_ladder": ladder_values,
            "activation_cut_ladder_sha256": ladder_sha,
            "expected_image": {
                "path": source_image_rel,
                "bytes": image_bytes,
                "sha256": image_sha,
            },
            "checks": {
                "complete_logical_pool": True,
                "exact_reference_blind": True,
                "portable_seed_schema": True,
                "qiskit_compilation_requested": False,
                "record_statevectors": True,
                "resume_input_present": resume_content is not None,
                "source_rebaseline_evidence_present": (
                    rebaseline_payload is not None
                ),
                "source_locked": True,
                "unique_output": True,
            },
            "required_remote_preflight": [
                "verify_source_archive_revision_and_digest",
                "verify_package_file_digests",
                "verify_image_size_and_sha256",
                *action_identity_preflight,
                "verify_output_path_absent",
                "verify_quota",
                "condor_submit_dry_run_exact_descriptor",
            ],
            "files": files,
            "worker_sha256": worker_sha,
            "wrapper_sha256": _sha256(temp_wrapper),
            "submit_descriptor_sha256": _sha256(temp_submit),
        }
        if not historical_route:
            manifest_payload["route_id"] = canonical_route
            manifest_payload["route_bundle_id"] = route_bundle_id
            manifest_payload["route_contract_sha256"] = route_contract_sha
        _write_bytes(temp_manifest, _pretty_json_bytes(manifest_payload))
        os.rename(temporary, out)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    action_path = root / action_rel
    manifest_path = root / manifest_rel
    return PreparedSIAMCHTCAction(
        package_dir=out,
        action=action_path,
        manifest=manifest_path,
        worker=root / worker_rel,
        wrapper=root / wrapper_rel,
        submit=root / submit_rel,
        seed=root / seed_rel,
        resume=None if resume_rel is None else root / resume_rel,
        resume_action_receipt=(
            None
            if completion_receipt_rel is None
            else root / completion_receipt_rel
        ),
        action_sha256=action_sha,
        manifest_sha256=_sha256(manifest_path),
    )


def _verify_manifest_files(
    manifest: Mapping[str, Any], *, repo_root: Path
) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("SIAM package manifest has no file digest table")
    checks = manifest.get("checks")
    if not isinstance(checks, dict) or not isinstance(
        checks.get("resume_input_present"), bool
    ):
        raise ValueError("SIAM package manifest lacks its resume-input contract")
    rebaseline_present = checks.get(
        "source_rebaseline_evidence_present", False
    )
    if not isinstance(rebaseline_present, bool):
        raise ValueError("SIAM package manifest has an invalid rebaseline flag")
    expected_labels = {
        "action",
        "seed",
        "worker",
        "wrapper",
        "submit_descriptor",
    }
    if checks["resume_input_present"]:
        expected_labels.update({"resume", "resume_action_receipt"})
    if rebaseline_present:
        expected_labels.update(
            {
                "predecessor_submission_receipt",
                "predecessor_package_manifest",
                "predecessor_epoch_closure_receipt",
            }
        )
    if set(files) != expected_labels:
        raise ValueError(
            "SIAM package file digest table has the wrong exact membership: "
            f"expected={sorted(expected_labels)!r}, got={sorted(files)!r}"
        )
    top_level_digest_fields = {
        "action": "action_json_sha256",
        "seed": "seed_sha256",
        "worker": "worker_sha256",
        "wrapper": "wrapper_sha256",
        "submit_descriptor": "submit_descriptor_sha256",
    }
    for label, raw in files.items():
        if not isinstance(raw, dict):
            raise ValueError(f"SIAM package file row {label!r} is malformed")
        relative = Path(str(raw.get("path") or ""))
        if relative.is_absolute():
            raise ValueError(f"SIAM package file {label!r} has an absolute path")
        path = (repo_root / relative).resolve()
        try:
            path.relative_to(repo_root.resolve())
        except ValueError as exc:
            raise ValueError(
                f"SIAM package file {label!r} escapes the repository root"
            ) from exc
        if not path.is_file():
            raise ValueError(f"SIAM package file {label!r} is missing: {path}")
        byte_count = raw.get("bytes")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or path.stat().st_size != byte_count
        ):
            raise ValueError(f"SIAM package file {label!r} has the wrong size")
        if _sha256(path) != raw.get("sha256"):
            raise ValueError(f"SIAM package file {label!r} fails SHA-256")
        top_level = top_level_digest_fields.get(label)
        if top_level is not None and manifest.get(top_level) != raw.get("sha256"):
            raise ValueError(
                f"SIAM package manifest {top_level} disagrees with its file row"
            )


def _verify_source_epoch_control_plane(
    *, action: Mapping[str, Any], manifest: Mapping[str, Any], repo_root: Path
) -> None:
    """Fail closed on source-epoch identity and any rebaseline exception."""

    source_epoch = action.get("source_epoch")
    if not isinstance(source_epoch, Mapping):
        raise ValueError("SIAM action lacks its source-epoch contract")
    expected_epoch = build_siam_source_epoch_contract(
        source_revision=str(action.get("source_revision") or ""),
        source_tree_files_sha256=str(
            dict(action.get("source_tree") or {}).get("files_sha256") or ""
        ),
    )
    if dict(source_epoch) != expected_epoch or manifest.get(
        "source_epoch"
    ) != expected_epoch:
        raise ValueError("SIAM action/manifest source epoch is inconsistent")
    action_id = _validated_action_id(str(action.get("action_id") or ""))
    scientific_identity = {
        "scope": "source_epoch_id/action_id",
        "source_epoch_id": expected_epoch["source_epoch_id"],
        "action_id": action_id,
    }
    if action.get("scientific_identity") != scientific_identity or manifest.get(
        "scientific_identity"
    ) != scientific_identity:
        raise ValueError("SIAM composite scientific identity is inconsistent")

    checks = manifest.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("SIAM manifest lacks source-epoch checks")
    rebaseline_present = checks.get("source_rebaseline_evidence_present")
    if not isinstance(rebaseline_present, bool):
        raise ValueError("SIAM manifest lacks its source-rebaseline mode")
    action_rebaseline = action.get("intentional_source_rebaseline")
    manifest_rebaseline = manifest.get("intentional_source_rebaseline")
    preflight = manifest.get("required_remote_preflight")
    if not isinstance(preflight, list) or not all(
        isinstance(value, str) for value in preflight
    ):
        raise ValueError("SIAM manifest has invalid remote preflight requirements")
    global_absence = (
        "verify_action_id_absent_from_prior_packages_and_scheduler_history"
    )
    if not rebaseline_present:
        if action_rebaseline is not None or manifest_rebaseline is not None:
            raise ValueError("SIAM ordinary package contains rebaseline evidence")
        if global_absence not in preflight or any(
            "composite_source_epoch_action_id" in value for value in preflight
        ):
            raise ValueError(
                "SIAM ordinary package must retain global action-ID absence"
            )
        return

    if (
        not isinstance(action_rebaseline, Mapping)
        or dict(action_rebaseline) != manifest_rebaseline
    ):
        raise ValueError("SIAM rebaseline action/manifest evidence differs")
    if global_absence in preflight:
        raise ValueError(
            "SIAM source rebaseline must not claim global action-ID absence"
        )
    required_rebaseline_preflight = {
        "verify_composite_source_epoch_action_id_absent_from_prior_packages_and_scheduler_history",
        "verify_historical_same_action_ids_use_distinct_source_epochs_and_bind_immediate_predecessor",
        "verify_predecessor_epoch_closed_or_superseded",
    }
    if not required_rebaseline_preflight.issubset(set(preflight)):
        raise ValueError("SIAM source-rebaseline preflight contract is incomplete")

    rebaseline = dict(action_rebaseline)
    artifacts = rebaseline.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("SIAM source rebaseline lacks packaged evidence")
    labels = (
        "predecessor_submission_receipt",
        "predecessor_package_manifest",
        "predecessor_epoch_closure_receipt",
    )
    documents: dict[str, dict[str, Any]] = {}
    normalized_artifacts: dict[str, dict[str, Any]] = {}
    for label in labels:
        artifact = artifacts.get(label)
        if not isinstance(artifact, Mapping):
            raise ValueError(f"SIAM source rebaseline lacks {label}")
        artifact_row = dict(artifact)
        packaged_path = _validated_receipt_path(
            artifact_row.get("packaged_path"),
            label=f"{label} packaged path",
        )
        evidence_path = repo_root / packaged_path
        byte_count = artifact_row.get("bytes")
        digest = str(artifact_row.get("sha256") or "")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count <= 0
            or not evidence_path.is_file()
            or evidence_path.stat().st_size != byte_count
            or not _SHA256.fullmatch(digest)
            or _sha256(evidence_path) != digest
        ):
            raise ValueError(f"SIAM source-rebaseline {label} fails its lock")
        documents[label] = _load_json_object(
            evidence_path, label=f"SIAM packaged {label}"
        )
        normalized_artifacts[label] = artifact_row

    core = _rebaseline_core_contract(
        predecessor_source_epoch_id=str(
            rebaseline.get("predecessor_source_epoch_id") or ""
        ),
        predecessor_submission=documents["predecessor_submission_receipt"],
        predecessor_manifest=documents["predecessor_package_manifest"],
        predecessor_closure=documents[
            "predecessor_epoch_closure_receipt"
        ],
        predecessor_submission_sha256=normalized_artifacts[
            "predecessor_submission_receipt"
        ]["sha256"],
        predecessor_manifest_sha256=normalized_artifacts[
            "predecessor_package_manifest"
        ]["sha256"],
        predecessor_closure_sha256=normalized_artifacts[
            "predecessor_epoch_closure_receipt"
        ]["sha256"],
        new_source_epoch=expected_epoch,
        new_package_id=str(action.get("package_id") or ""),
        new_action_id=action_id,
        new_output_path=str(action.get("trajectory_json") or ""),
    )
    expected_rebaseline = {
        "schema": SIAM_SOURCE_EPOCH_REBASELINE_V1,
        "mode": "intentional_source_rebaseline",
        **core,
        "artifacts": normalized_artifacts,
    }
    if rebaseline != expected_rebaseline:
        raise ValueError("SIAM source-rebaseline proof is not canonical")


def _assert_completed_configuration(
    payload: Mapping[str, Any], *, action: Mapping[str, Any]
) -> None:
    """Bind the emitted run lock and structural summary to the staged action."""

    expected_execution = dict(action.get("expected_execution_contract") or {})
    expected_support = expected_execution.get("support_patch_config")
    expected_solve_repair = expected_execution.get("solve_repair_config")
    expected_candidate_pool = expected_execution.get("candidate_pool_contract")
    if not all(
        isinstance(value, Mapping)
        for value in (
            expected_support,
            expected_solve_repair,
            expected_candidate_pool,
        )
    ):
        raise ValueError("completed SIAM action lacks its execution contract")

    run_lock = dict(payload.get("run_lock") or {})
    if run_lock.get("schema") != "paper_ii_run_lock_v4":
        raise ValueError("completed SIAM trajectory has the wrong run-lock schema")
    physics = dict(run_lock.get("physics") or {})
    if (
        physics.get("family_key") != "siam"
        or physics.get("n_ph_max") is not None
        or physics.get("drive_profile") is not None
    ):
        raise ValueError("completed SIAM trajectory changed the frozen physics lane")
    registered_times = _registered_segment(str(action["segment"])).times
    expected_grid = {
        "t_initial": registered_times[0],
        "t_final": registered_times[-1],
        "point_count": len(registered_times),
        "ordered_values_sha256": _sha256_bytes(
            json.dumps(
                [float(value) for value in registered_times],
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ),
    }
    if physics.get("time_grid") != expected_grid:
        raise ValueError("completed SIAM trajectory changed the ordered time grid")

    numerics = dict(run_lock.get("numerics") or {})
    inverse = dict(numerics.get("inverse_policy") or {})
    expected_inverse = {
        "pinv_rcond": float(EULER_RIDGE1E6_NUMERICS.pinv_rcond),
        "ridge_lambda": float(EULER_RIDGE1E6_NUMERICS.ridge_lambda),
        "solve_damping": float(EULER_RIDGE1E6_NUMERICS.solve_damping),
    }
    if numerics.get("integrator_method") != EULER_RIDGE1E6_NUMERICS.integrator:
        raise ValueError("completed SIAM trajectory changed the Euler integrator")
    if inverse != expected_inverse:
        raise ValueError("completed SIAM trajectory changed the inverse policy")
    repair = dict(numerics.get("solve_repair") or {})
    if repair != dict(expected_solve_repair):
        raise ValueError("completed SIAM trajectory changed the solve-repair policy")
    required_inverse_ladders = {
        "ridge_ladder": [expected_inverse["ridge_lambda"]],
        "pinv_rcond_ladder": [expected_inverse["pinv_rcond"]],
        "solve_damping_ladder": [expected_inverse["solve_damping"]],
    }
    if any(
        repair.get(key) != value
        for key, value in required_inverse_ladders.items()
    ):
        raise ValueError(
            "completed SIAM trajectory changed the canonical inverse ladder"
        )
    required_repair = {
        "enabled": True,
        "local_subdivision_enabled": True,
        "max_local_subdivisions": int(
            STATE_MOTION_CONTROL_CALIBRATED.max_local_subdivisions or 0
        ),
        "state_motion_l2_step_max": float(
            STATE_MOTION_CONTROL_CALIBRATED.state_motion_l2_step_max or 0.0
        ),
        "profile_id": "ap_mclachlan_state_space_solve_repair_v2",
    }
    if any(repair.get(key) != value for key, value in required_repair.items()):
        raise ValueError("completed SIAM trajectory changed the state-motion repair")

    required_effective_inverse = {
        "effective_ridge_lambdas": [expected_inverse["ridge_lambda"]],
        "effective_pinv_rconds": [expected_inverse["pinv_rcond"]],
        "effective_solve_dampings": [expected_inverse["solve_damping"]],
    }
    for interval_index, point in enumerate(_trajectory_points(payload)[:-1]):
        integration = point.get("integration_to_next")
        if not isinstance(integration, Mapping):
            raise ValueError(
                f"completed SIAM interval {interval_index} lacks integration telemetry"
            )
        repair_summary = integration.get("repair_summary")
        if not isinstance(repair_summary, Mapping) or any(
            repair_summary.get(key) != value
            for key, value in required_effective_inverse.items()
        ):
            raise ValueError(
                "completed SIAM interval "
                f"{interval_index} changed the effective inverse policy"
            )

    shared = dict(run_lock.get("shared_protocol") or {})
    if shared.get("numerics") != numerics:
        raise ValueError("completed SIAM shared numerics differ from its run lock")
    expected_controller = {
        "composed_parameter_step_max": None,
        "local_subdivision_enabled": True,
        "max_local_subdivisions": int(
            STATE_MOTION_CONTROL_CALIBRATED.max_local_subdivisions or 0
        ),
        "parameter_displacement_max": None,
        "state_motion_l2_step_max": float(
            STATE_MOTION_CONTROL_CALIBRATED.state_motion_l2_step_max or 0.0
        ),
    }
    if shared.get("time_step_controller") != expected_controller:
        raise ValueError("completed SIAM trajectory changed the time-step controller")
    candidate_pool = dict(shared.get("candidate_pool") or {})
    if candidate_pool != dict(expected_candidate_pool):
        raise ValueError("completed SIAM trajectory changed the ordered logical pool")
    expected_pool_count = int(action["seed_pool_block_count"])
    if (
        candidate_pool.get("schema") != "paper_ii_effective_candidate_pool_v1"
        or candidate_pool.get("append_occurrence_policy") != "layer_reuse"
        or candidate_pool.get("candidate_pool_order") != "frozen_pool_order"
        or candidate_pool.get("candidate_pool_raw") != expected_pool_count
        or candidate_pool.get("candidate_pool_deduplicated") != expected_pool_count
        or not _SHA256.fullmatch(
            str(candidate_pool.get("ordered_atom_contract_sha256") or "")
        )
    ):
        raise ValueError("completed SIAM trajectory changed the logical pool contract")

    policy = dict(run_lock.get("policy") or {})
    dynamics_policy = _arm_dynamics_policy(str(action["arm"]))
    if policy.get("structural_policy") != dynamics_policy:
        raise ValueError("completed SIAM trajectory resolved the wrong arm")
    guards = dict(policy.get("guards") or {})
    # A V5 action declares which exchange route it was packaged for.  Bind it
    # to the route the run actually resolved.  A V4 action declares nothing,
    # so the historical lineage keeps its exact prior acceptance set.
    declared_bundle_id = action.get("route_bundle_id")
    if declared_bundle_id is not None and dynamics_policy == "exchange":
        route_contract = guards.get("exchange_route_contract")
        if not isinstance(route_contract, Mapping):
            raise ValueError(
                "completed SIAM exchange trajectory records no exchange route "
                "contract"
            )
        if str(route_contract.get("bundle_id") or "") != str(declared_bundle_id):
            raise ValueError(
                "completed SIAM trajectory ran a different exchange route "
                "than the packaged action"
            )
        declared_digest = action.get("route_contract_sha256")
        if declared_digest is not None and guards.get(
            "exchange_route_contract_sha256"
        ) != str(declared_digest):
            raise ValueError(
                "completed SIAM trajectory changed the exchange route contract"
            )
    expected_guards = {
        "max_structural_pool_size": SIAM_NB4_PILOT_SEARCH.pool_size,
        "max_joint_patch_evaluations": (
            SIAM_NB4_PILOT_SEARCH.joint_patch_evaluations
        ),
        "max_certification_attempts_per_level": (
            SIAM_NB4_PILOT_SEARCH.certification_attempts_per_level
        ),
        "max_certification_attempts_per_deletion_branch": (
            SIAM_NB4_PILOT_SEARCH.certification_attempts_per_deletion_branch
        ),
        "max_insertion_batch_size": 1,
        "residual_ratio_threshold": 0.02,
    }
    if any(guards.get(key) != value for key, value in expected_guards.items()):
        raise ValueError("completed SIAM trajectory changed the search guards")

    summary = dict(payload.get("summary") or {})
    if summary.get("parameterization_mode") != "logical_shared":
        raise ValueError("completed SIAM trajectory changed parameterization mode")
    support = dict(summary.get("support_patch_config") or {})
    if support != dict(expected_support):
        raise ValueError("completed SIAM trajectory changed the frozen support policy")
    cut = float(action["activation_cut"])
    expected_gate = (
        "mclachlan_l2" if dynamics_policy == "exchange" else "residual_ratio"
    )
    expected_support = {
        "dynamics_policy": dynamics_policy,
        "insertion_l2_cut": cut,
        "avqds_l2_cut": cut,
        "insertion_gate_mode": expected_gate,
        "append_occurrence_policy": "layer_reuse",
        "allow_incomplete_candidate_pool": False,
        "max_structural_pool_size": SIAM_NB4_PILOT_SEARCH.pool_size,
        "max_joint_patch_evaluations": (
            SIAM_NB4_PILOT_SEARCH.joint_patch_evaluations
        ),
        "max_certification_attempts_per_level": (
            SIAM_NB4_PILOT_SEARCH.certification_attempts_per_level
        ),
        "max_certification_attempts_per_deletion_branch": (
            SIAM_NB4_PILOT_SEARCH.certification_attempts_per_deletion_branch
        ),
        "max_insertion_rounds_per_checkpoint": (
            SIAM_NB4_PILOT_SEARCH.rounds_per_checkpoint
        ),
        "max_insertion_batch_size": 1,
        "append_schur_max_condition_number": None,
        "uses_future_exact_forecast_for_decision": False,
        "uses_reference_for_decision": False,
    }
    if any(support.get(key) != value for key, value in expected_support.items()):
        raise ValueError(
            "completed SIAM trajectory changed its activation cut or locked policy"
        )


def _assert_completed_start_boundary(
    payload: Mapping[str, Any],
    *,
    action: Mapping[str, Any],
    seed_manifest: Any,
    resume_payload: Mapping[str, Any] | None,
) -> dict[str, bool]:
    """Verify a screen seed or continuation-resume boundary and all states."""

    rows = list(payload.get("plot_rows") or ())
    if not rows:
        raise ValueError("completed SIAM trajectory has no checkpoint states")
    cfg = seed_manifest.physical_config()
    n_bath = int(cfg.n_bath)
    prepared_seed = np.asarray(
        build_initial_state(n_bath, hopping=float(cfg.hopping)), dtype=complex
    ).reshape(-1)
    sector = np.asarray(sector_indices(n_bath), dtype=int)
    sector_mask = np.zeros(prepared_seed.size, dtype=bool)
    sector_mask[sector] = True
    states: list[np.ndarray] = []
    for index, row in enumerate(rows):
        state = np.asarray(row["psi_real"], dtype=float) + 1.0j * np.asarray(
            row["psi_imag"], dtype=float
        )
        norm = float(np.real(np.vdot(state, state)))
        leakage = float(np.real(np.vdot(state[~sector_mask], state[~sector_mask])))
        if abs(norm - 1.0) > 1.0e-8 or leakage > 1.0e-8:
            raise ValueError(
                f"completed SIAM checkpoint {index} violates norm/sector guards"
            )
        states.append(state)
    segment = _registered_segment(str(action["segment"]))
    if segment is SIAM_NB4_SCREEN_SEGMENT:
        if resume_payload is not None:
            raise ValueError("SIAM screen completion unexpectedly has resume data")
        expected_state = prepared_seed
        h_post = np.asarray(
            hamiltonian_matrix(
                build_siam_hamiltonian(
                    n_bath,
                    u=float(cfg.u),
                    v=float(cfg.v_final),
                    hopping=float(cfg.hopping),
                )
            ),
            dtype=complex,
        )
        expected_energy = float(
            np.real(np.vdot(expected_state, h_post @ expected_state))
        )
        boundary_checks = {
            "segment_initial_state": True,
            "segment_initial_energy": True,
            "initial_seed_state": True,
            "initial_seed_energy": True,
            "state_norm_sector": True,
        }
        state_error = "completed SIAM initial state differs from the prepared seed"
        energy_error = "completed SIAM initial energy differs from the seed"
    else:
        if resume_payload is None:
            raise ValueError("SIAM continuation completion lacks its resume boundary")
        resume_rows = list(resume_payload.get("plot_rows") or ())
        if not resume_rows or not isinstance(resume_rows[-1], Mapping):
            raise ValueError("SIAM continuation resume lacks its terminal state")
        resume_row = resume_rows[-1]
        expected_state = np.asarray(
            resume_row.get("psi_real"), dtype=float
        ) + 1.0j * np.asarray(resume_row.get("psi_imag"), dtype=float)
        if expected_state.size != prepared_seed.size:
            raise ValueError("SIAM continuation resume boundary dimension differs")
        expected_energy_value = resume_row.get("energy_expectation")
        if isinstance(expected_energy_value, bool) or not isinstance(
            expected_energy_value, Real
        ):
            raise ValueError("SIAM continuation resume boundary energy is missing")
        expected_energy = float(expected_energy_value)
        if not math.isfinite(expected_energy):
            raise ValueError("SIAM continuation resume boundary energy is nonfinite")
        boundary_checks = {
            "segment_initial_state": True,
            "segment_initial_energy": True,
            "resume_boundary_state": True,
            "resume_boundary_energy": True,
            "state_norm_sector": True,
        }
        state_error = "completed SIAM continuation start differs from its resume boundary"
        energy_error = "completed SIAM continuation energy differs from its resume boundary"

    initial = states[0]
    fidelity = float(abs(np.vdot(expected_state, initial)) ** 2)
    if 1.0 - min(fidelity, 1.0) > 1.0e-12:
        raise ValueError(state_error)
    observed_energy = rows[0].get("energy_expectation")
    if isinstance(observed_energy, bool) or not isinstance(observed_energy, Real):
        raise ValueError("completed SIAM initial energy is missing")
    if not math.isfinite(float(observed_energy)) or abs(
        float(observed_energy) - expected_energy
    ) > 1.0e-10:
        raise ValueError(energy_error)
    return boundary_checks


def _assert_completed_terminal_resume(
    payload: Mapping[str, Any],
    *,
    seed_runtime_input: Any,
) -> tuple[dict[str, bool], dict[str, Any]]:
    """Reconstruct and bind the terminal physical and controller resume.

    A schema label is not evidence that a trajectory can actually continue.
    Reuse the two AP-McLachlan resume loaders here, then compare their rebuilt
    support, angles, state, and controller snapshot with the independently
    serialized terminal trajectory records.
    """

    resume = payload.get("resume_state")
    if not isinstance(resume, Mapping) or resume.get("schema") != RESUME_SCHEMA_V2:
        raise ValueError("completed SIAM trajectory lacks resume schema v2")
    if str(resume.get("problem_family") or "").strip().lower() != "siam":
        raise ValueError("completed SIAM terminal resume has the wrong family")
    if resume.get("parameterization_mode") != AP_PARAMETERIZATION_LOGICAL_SHARED:
        raise ValueError(
            "completed SIAM terminal resume changed parameterization mode"
        )

    try:
        reconstructed_input = runtime_input_from_resume_state(
            resume,
            base_runtime_input=seed_runtime_input,
        )
        reconstructed_state = state_from_scaffold_runtime_input(
            reconstructed_input,
            parameterization_mode=AP_PARAMETERIZATION_LOGICAL_SHARED,
        )
    except Exception as exc:
        raise ValueError(
            "completed SIAM terminal resume cannot be reconstructed"
        ) from exc

    theta = np.asarray(reconstructed_state.theta_runtime, dtype=float).reshape(-1)
    if not np.all(np.isfinite(theta)):
        raise ValueError("completed SIAM terminal resume angles are nonfinite")
    theta_logical = np.asarray(
        getattr(reconstructed_input, "theta_logical", ()), dtype=float
    ).reshape(-1)
    if not np.array_equal(theta, theta_logical):
        raise ValueError(
            "completed SIAM terminal resume logical/runtime angles differ"
        )

    points = _trajectory_points(payload)
    if not points:
        raise ValueError("completed SIAM trajectory lacks a terminal point")
    terminal_point = points[-1]
    terminal_theta_value = terminal_point.get("theta_runtime")
    if not isinstance(terminal_theta_value, list):
        raise ValueError("completed SIAM terminal point lacks its angles")
    terminal_theta = np.asarray(terminal_theta_value, dtype=float).reshape(-1)
    if not np.all(np.isfinite(terminal_theta)) or not np.array_equal(
        terminal_theta, theta
    ):
        raise ValueError(
            "completed SIAM terminal resume angles differ from the terminal point"
        )

    final_state = payload.get("final_state")
    if not isinstance(final_state, Mapping):
        raise ValueError("completed SIAM trajectory lacks its final support")
    rebuilt_final = reconstructed_state.to_json_dict()
    support_fields = (
        "parameterization_mode",
        "active_parameter_count",
        "logical_parameter_count",
        "runtime_parameter_count",
        "runtime_pauli_parameter_count",
        "runtime_coordinate_labels",
        "runtime_pauli_coordinate_labels",
        "selected_term_labels",
        "candidate_pool_term_count",
        "candidate_pool_complete",
    )
    if any(
        final_state.get(field) != rebuilt_final.get(field)
        for field in support_fields
    ):
        raise ValueError(
            "completed SIAM terminal resume support differs from final_state"
        )

    rows = list(payload.get("plot_rows") or ())
    if not rows or not isinstance(rows[-1], Mapping):
        raise ValueError("completed SIAM trajectory lacks a terminal statevector")
    terminal_row = rows[-1]
    terminal_state = np.asarray(
        terminal_row.get("psi_real"), dtype=float
    ) + 1.0j * np.asarray(terminal_row.get("psi_imag"), dtype=float)
    reconstructed_vector = np.asarray(
        reconstructed_state.prepare_state(theta), dtype=complex
    ).reshape(-1)
    if terminal_state.size != reconstructed_vector.size:
        raise ValueError(
            "completed SIAM terminal resume statevector dimension differs"
        )
    overlap_squared = float(abs(np.vdot(reconstructed_vector, terminal_state)) ** 2)
    state_infidelity = max(0.0, 1.0 - min(overlap_squared, 1.0))
    if not math.isfinite(state_infidelity) or state_infidelity > 1.0e-12:
        raise ValueError(
            "completed SIAM terminal resume does not reproduce the terminal statevector"
        )
    reconstructed_state_sha = portable_statevector_sha256(reconstructed_vector)
    terminal_state_sha = portable_statevector_sha256(terminal_state)
    if reconstructed_state_sha != terminal_state_sha:
        raise ValueError(
            "completed SIAM terminal resume state identity differs from checkpoint"
        )

    controller_payload = resume.get("controller_state")
    if not isinstance(controller_payload, Mapping):
        raise ValueError("completed SIAM terminal resume lacks controller state")
    try:
        (
            checkpoint_index,
            prune_runtime_state,
            repair_hold_state,
            previous_accepted_theta_dot,
        ) = _restore_controller_resume_state(controller_payload)
        canonical_controller = _controller_resume_snapshot(
            checkpoint_index=checkpoint_index,
            prune_runtime_state=prune_runtime_state,
            repair_hold_state=repair_hold_state,
            previous_accepted_theta_dot=previous_accepted_theta_dot,
        )
    except Exception as exc:
        raise ValueError(
            "completed SIAM terminal controller resume cannot be reconstructed"
        ) from exc
    if canonical_controller != dict(controller_payload):
        raise ValueError(
            "completed SIAM terminal controller resume is not canonical"
        )

    trajectory = payload.get("trajectory")
    trajectory_controller = (
        trajectory.get("controller_resume_state")
        if isinstance(trajectory, Mapping)
        else None
    )
    if not isinstance(trajectory_controller, Mapping) or dict(
        trajectory_controller
    ) != canonical_controller:
        raise ValueError(
            "completed SIAM terminal controller resume differs from trajectory"
        )
    terminal_index = terminal_point.get("index")
    terminal_row_index = terminal_row.get("index")
    if (
        isinstance(terminal_index, bool)
        or not isinstance(terminal_index, int)
        or terminal_row_index != terminal_index
        or int(checkpoint_index) != terminal_index
    ):
        raise ValueError(
            "completed SIAM terminal controller checkpoint index differs"
        )

    resume_time = float(resume.get("time", math.nan))
    point_time = float(terminal_point.get("time", math.nan))
    row_time = float(terminal_row.get("time", math.nan))
    if (
        not all(
            math.isfinite(value)
            for value in (resume_time, point_time, row_time)
        )
        or resume_time != point_time
        or resume_time != row_time
    ):
        raise ValueError("completed SIAM terminal resume time differs")

    expected_previous: np.ndarray | None = None
    if len(points) > 1:
        previous_fixed_step = points[-2].get("fixed_step")
        previous_theta_dot = (
            previous_fixed_step.get("theta_dot")
            if isinstance(previous_fixed_step, Mapping)
            else None
        )
        if not isinstance(previous_theta_dot, list):
            raise ValueError(
                "completed SIAM trajectory lacks the prior accepted velocity"
            )
        expected_previous = np.asarray(previous_theta_dot, dtype=float).reshape(-1)
        if not np.all(np.isfinite(expected_previous)):
            raise ValueError(
                "completed SIAM prior accepted velocity is nonfinite"
            )
    observed_previous = (
        None
        if previous_accepted_theta_dot is None
        else np.asarray(previous_accepted_theta_dot, dtype=float).reshape(-1)
    )
    if (expected_previous is None) != (observed_previous is None) or (
        expected_previous is not None
        and observed_previous is not None
        and not np.array_equal(expected_previous, observed_previous)
    ):
        raise ValueError(
            "completed SIAM terminal controller velocity differs from trajectory"
        )
    previous_count = (
        0 if observed_previous is None else int(observed_previous.size)
    )
    if observed_previous is not None and previous_count != int(theta.size):
        raise ValueError(
            "completed SIAM terminal controller velocity dimension differs"
        )
    if prune_runtime_state.support_identity_hash != _support_identity_hash(
        reconstructed_state
    ):
        raise ValueError(
            "completed SIAM terminal controller support identity differs"
        )

    support_contract = {
        "coordinates": list(resume.get("coordinates") or ()),
        "runtime_coordinate_labels": rebuilt_final["runtime_coordinate_labels"],
        "runtime_pauli_coordinate_labels": rebuilt_final[
            "runtime_pauli_coordinate_labels"
        ],
        "selected_term_labels": rebuilt_final["selected_term_labels"],
    }
    theta_contract = {
        "theta_runtime": [float(value) for value in theta],
        "theta_logical": [float(value) for value in theta_logical],
    }
    proof = {
        "schema": SIAM_NB4_RESUME_TERMINAL_PROOF_V1,
        "resume_schema": RESUME_SCHEMA_V2,
        "controller_schema": CONTROLLER_RESUME_SCHEMA_V1,
        "terminal_time": resume_time,
        "terminal_checkpoint_index": int(terminal_index),
        "logical_coordinate_count": int(theta.size),
        "runtime_pauli_coordinate_count": int(
            reconstructed_state.runtime_pauli_parameter_count
        ),
        "previous_accepted_velocity_count": previous_count,
        "support_contract_sha256": _sha256_bytes(
            _canonical_json_bytes(support_contract)
        ),
        "theta_contract_sha256": _sha256_bytes(
            _canonical_json_bytes(theta_contract)
        ),
        "controller_state_sha256": _sha256_bytes(
            _canonical_json_bytes(canonical_controller)
        ),
        "reconstructed_state_semantic_sha256": reconstructed_state_sha,
        "terminal_state_semantic_sha256": terminal_state_sha,
        "reconstructed_state_infidelity": state_infidelity,
    }
    checks = {
        "resume_reconstruction": True,
        "resume_support_parameters": True,
        "resume_terminal_statevector": True,
        "controller_resume_reconstruction": True,
        "controller_resume_terminal_state": True,
    }
    return checks, proof


def _validate_completed_run(
    payload: Mapping[str, Any],
    *,
    action: Mapping[str, Any],
    seed_manifest: Any,
    seed_runtime_input: Any,
    resume_payload: Mapping[str, Any] | None = None,
) -> tuple[dict[str, bool], dict[str, Any]]:
    points = _trajectory_points(payload)
    expected_count = int(action["expected_point_count"])
    expected_initial = float(action["expected_time_initial"])
    expected_final = float(action["expected_time_final"])
    times = tuple(float(row.get("time")) for row in points)
    registered_times = _registered_segment(str(action["segment"])).times
    checks = {
        "source_revision": (
            dict(payload.get("run_lock") or {}).get("code_revision")
            == action["source_revision"]
        ),
        "uncured_local_subdivisions": (
            dict(payload.get("summary") or {}).get(
                "uncured_local_subdivision_count"
            )
            == 0
        ),
        "resume_schema": (
            dict(payload.get("resume_state") or {}).get("schema")
            == "paper_ii_resume_state_v2"
        ),
        "trajectory_points": len(points) == expected_count,
        "ordered_time_grid": times == registered_times,
        "initial_time": bool(times) and times[0] == expected_initial,
        "terminal_time": bool(times) and times[-1] == expected_final,
    }
    if not all(checks.values()):
        raise ValueError(f"completed SIAM trajectory failed boundaries: {checks}")
    _assert_reference_blind_payload(payload)
    _assert_complete_pool(
        payload, block_count=int(action["seed_pool_block_count"])
    )
    _assert_recorded_statevectors(
        payload,
        point_count=expected_count,
        state_dimension=int(action["seed_state_dimension"]),
    )
    _assert_completed_configuration(payload, action=action)
    boundary_checks = _assert_completed_start_boundary(
        payload,
        action=action,
        seed_manifest=seed_manifest,
        resume_payload=resume_payload,
    )
    run_lock = dict(payload.get("run_lock") or {})
    physics = dict(run_lock.get("physics") or {})
    policy = dict(run_lock.get("policy") or {})
    if physics.get("seed_sha256") != action["seed_sha256"]:
        raise ValueError("completed SIAM trajectory resolved the wrong seed")
    if policy.get("structural_policy") != _arm_dynamics_policy(
        str(action["arm"])
    ):
        raise ValueError("completed SIAM trajectory resolved the wrong arm")
    resume_checks, resume_proof = _assert_completed_terminal_resume(
        payload,
        seed_runtime_input=seed_runtime_input,
    )
    return ({
        **checks,
        "configuration_lock": True,
        **boundary_checks,
        **resume_checks,
    }, resume_proof)


def run_prepared_siam_action(
    *,
    package_dir: str | Path,
    expected_action_id: str,
    expected_action_sha256: str,
    expected_source_revision: str,
) -> int:
    """Execute one prepared action after rechecking its complete control plane."""

    root = Path.cwd().resolve()
    package = root / Path(package_dir)
    action_path = package / "input" / "action.json"
    manifest_path = package / "input" / "manifest.json"
    if _sha256(action_path) != expected_action_sha256:
        raise SystemExit("SIAM action JSON fails its worker-embedded SHA-256 lock")
    action = _load_json_object(action_path, label="SIAM action")
    manifest = _load_json_object(manifest_path, label="SIAM package manifest")
    action_schema = action.get("schema")
    if action_schema not in {
        SIAM_NB4_CHTC_ACTION_SCHEMA_V4,
        SIAM_NB4_CHTC_ACTION_SCHEMA_V5,
    }:
        raise SystemExit("SIAM action schema mismatch")
    route_lineage = action_schema == SIAM_NB4_CHTC_ACTION_SCHEMA_V5
    expected_manifest_schema = (
        SIAM_NB4_CHTC_MANIFEST_SCHEMA_V5
        if route_lineage
        else SIAM_NB4_CHTC_MANIFEST_SCHEMA_V4
    )
    if manifest.get("schema") != expected_manifest_schema:
        raise SystemExit("SIAM package manifest schema mismatch")
    if action.get("action_id") != expected_action_id:
        raise SystemExit("SIAM action identity mismatch")
    if action.get("source_revision") != expected_source_revision:
        raise SystemExit("SIAM action source revision mismatch")
    if manifest.get("action_json_sha256") != expected_action_sha256:
        raise SystemExit("SIAM manifest does not bind the worker action")
    if manifest.get("source_revision") != expected_source_revision:
        raise SystemExit("SIAM manifest source revision mismatch")
    if manifest.get("source_tree_contract") != SIAM_SOURCE_TREE_CONTRACT_V1:
        raise SystemExit("SIAM manifest source-tree schema mismatch")
    if manifest.get("source_tree_files_sha256") != dict(
        action.get("source_tree") or {}
    ).get("files_sha256") or manifest.get("source_tree_file_count") != dict(
        action.get("source_tree") or {}
    ).get("file_count"):
        raise SystemExit("SIAM manifest does not bind the source-tree contract")
    if manifest.get("action_id") != expected_action_id or manifest.get(
        "package_id"
    ) != action.get("package_id"):
        raise SystemExit("SIAM manifest action/package identity mismatch")
    if manifest.get("expected_output") != action.get("trajectory_json"):
        raise SystemExit("SIAM manifest output identity mismatch")
    _verify_manifest_files(manifest, repo_root=root)
    try:
        _verify_source_epoch_control_plane(
            action=action, manifest=manifest, repo_root=root
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    source_tree_evidence = _verify_source_tree_contract(
        dict(action.get("source_tree") or {}),
        repo_root=root,
        revision=expected_source_revision,
    )

    if action.get("exact_reference_blind") is not True:
        raise SystemExit("SIAM raw action is not declared exact-reference blind")
    if action.get("compile_terminal_cost") is not False or action.get(
        "qiskit_compilation_requested"
    ) is not False:
        raise SystemExit("SIAM raw action must not request Qiskit compilation")
    if action.get("seed_schema") != SIAM_SEED_SCHEMA_V2:
        raise SystemExit("SIAM action does not require the portable seed schema")

    seed_path = root / str(action["seed_path"])
    if _sha256(seed_path) != action.get("seed_sha256"):
        raise SystemExit("SIAM seed fails its action-bound SHA-256 lock")
    seed_manifest = load_siam_seed_manifest(seed_path)
    if seed_manifest.schema != SIAM_SEED_SCHEMA_V2:
        raise SystemExit("SIAM seed is not portable schema v2")
    if int(seed_manifest.pool["block_count"]) != int(
        action["seed_pool_block_count"]
    ):
        raise SystemExit("SIAM seed pool count differs from the action lock")
    expected_state_dimension = 1 << int(seed_manifest.layout["n_qubits"])
    if expected_state_dimension != int(action["seed_state_dimension"]):
        raise SystemExit("SIAM seed state dimension differs from the action lock")

    segment = _registered_segment(str(action["segment"]))
    arm = _canonical_arm(str(action["arm"]))
    cut = float(action["activation_cut"])
    if not math.isfinite(cut) or cut <= 0.0:
        raise SystemExit("SIAM action activation cut is invalid")
    try:
        ladder_resolution = resolve_embedded_ladder(
            activation_cut=cut,
            package_index=action.get("activation_cut_ladder_index"),
            action_ladder_sha256=action.get("activation_cut_ladder_sha256"),
            manifest_ladder_sha256=manifest.get(
                "activation_cut_ladder_sha256"
            ),
            manifest_ladder=manifest.get("activation_cut_ladder"),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    resume = action.get("resume")
    resume_path: str | None = None
    resume_payload: dict[str, Any] | None = None
    if segment.requires_resume:
        if not isinstance(resume, dict):
            raise SystemExit("SIAM continuation action lacks a resume receipt")
        resume_path = str(resume.get("packaged_path") or "")
        if _sha256(root / resume_path) != resume.get("sha256"):
            raise SystemExit("SIAM continuation resume fails its SHA-256 lock")
        completion_receipt_path = str(
            resume.get("packaged_action_receipt_path") or ""
        )
        if not completion_receipt_path:
            raise SystemExit(
                "SIAM continuation action lacks a completion receipt input"
            )
        if _sha256(root / completion_receipt_path) != resume.get(
            "source_action_receipt_sha256"
        ):
            raise SystemExit(
                "SIAM continuation completion receipt fails its SHA-256 lock"
            )
        resume_payload = _load_json_object(
            root / resume_path,
            label="SIAM continuation resume source",
        )
        verified_resume = _validate_resume_payload(
            resume_payload,
            arm=arm,
            seed_sha256=str(action["seed_sha256"]),
            seed_block_count=int(action["seed_pool_block_count"]),
            seed_state_dimension=int(action["seed_state_dimension"]),
        )
        for field in ("source_revision", "time_final", "point_count"):
            if resume.get(field) != verified_resume[field]:
                raise SystemExit(
                    f"SIAM continuation resume {field} receipt mismatch"
                )
        verified_completion = _validate_source_completion_receipt(
            root / completion_receipt_path,
            trajectory_path=root / resume_path,
            arm=arm,
            activation_cut=cut,
            seed_sha256=str(action["seed_sha256"]),
            resume_metadata=verified_resume,
        )
        for field, value in verified_completion.items():
            if resume.get(field) != value:
                raise SystemExit(
                    f"SIAM continuation completion {field} receipt mismatch"
                )
    elif resume is not None:
        raise SystemExit("SIAM screen action unexpectedly contains a resume input")

    output_path = str(action["trajectory_json"])
    if segment is SIAM_NB4_SCREEN_SEGMENT:
        command = build_siam_nb4_screen_run(
            seed_path=str(action["seed_path"]),
            arm=arm,
            activation_cut=cut,
            output_json=output_path,
            code_revision=expected_source_revision,
        )
    else:
        assert resume_path is not None
        command = build_siam_nb4_continuation_run(
            seed_path=str(action["seed_path"]),
            arm=arm,
            resume_from_run_json=resume_path,
            activation_cut=cut,
            output_json=output_path,
            code_revision=expected_source_revision,
        )
    argv = tuple(command.argv())
    _validate_runner_argv(
        argv,
        seed_path=str(action["seed_path"]),
        resume_path=resume_path,
        output_path=output_path,
        source_revision=expected_source_revision,
        segment_times=segment.times,
    )
    execution_contract = _execution_contract_from_argv(
        argv,
        seed_path=seed_path,
        resume_payload=resume_payload,
    )
    if execution_contract != action.get("expected_execution_contract"):
        raise SystemExit(
            "resolved SIAM execution configuration differs from the action lock"
        )
    argv_sha = _argv_sha256(argv)
    if (
        argv_sha != action.get("expected_runner_argv_sha256")
        or argv_sha != manifest.get("runner_argv_sha256")
    ):
        raise SystemExit("resolved SIAM runner command fails its digest lock")
    if command.run_id != action.get("expected_run_id"):
        raise SystemExit("resolved SIAM runner command fails its run-id lock")

    trajectory = root / output_path
    receipt = trajectory.with_name("action_receipt.json")
    if trajectory.exists() or receipt.exists():
        raise SystemExit("refusing to overwrite a prior SIAM action output")
    trajectory.parent.mkdir(parents=True, exist_ok=True)
    base_receipt: dict[str, Any] = {
        "schema": (
            SIAM_NB4_CHTC_RECEIPT_SCHEMA_V5
            if route_lineage
            else SIAM_NB4_CHTC_RECEIPT_SCHEMA_V4
        ),
        "package_id": action["package_id"],
        "action_id": expected_action_id,
        "run_id": command.run_id,
        "source_revision": expected_source_revision,
        "source_epoch": action["source_epoch"],
        "scientific_identity": action["scientific_identity"],
        "intentional_source_rebaseline": action[
            "intentional_source_rebaseline"
        ],
        "action_json_sha256": expected_action_sha256,
        "package_manifest_sha256": _sha256(manifest_path),
        **source_tree_evidence,
        "seed_path": str(action["seed_path"]),
        "seed_sha256": action["seed_sha256"],
        "arm": arm,
        "activation_cut": cut,
        "segment": segment.segment_id,
        "exact_reference_blind": True,
        "qiskit_compilation_requested": False,
        "runner_argv_sha256": argv_sha,
        "runner_argv": list(argv),
        "trajectory_json": output_path,
        # Embed the exact immutable control-plane documents so downstream
        # validators can recompute their digests instead of trusting
        # self-declared hexadecimal strings in the receipt.
        "action_contract": action,
        "package_manifest_contract": manifest,
    }
    if route_lineage:
        # Route identity travels with the receipt so a downstream reader never
        # has to reopen the package to learn which exchange route ran.
        for field in ("route_id", "route_bundle_id", "route_contract_sha256"):
            if field not in action or action.get(field) != manifest.get(field):
                raise SystemExit(
                    "SIAM route package action and manifest disagree on "
                    f"{field}"
                )
            base_receipt[field] = action[field]
        if _arm_route_bundle_id(arm) != action["route_bundle_id"]:
            raise SystemExit("SIAM action arm is not on its declared route")
        if _route_contract_sha256(
            str(action["route_bundle_id"])
        ) != action["route_contract_sha256"]:
            raise SystemExit("SIAM action route contract digest is stale")
    _write_json(receipt, {**base_receipt, "status": "RUNNING"})
    try:
        subprocess.run([sys.executable, RUNNER_PATH, *argv], check=True)
        if not trajectory.is_file() or trajectory.stat().st_size <= 0:
            raise RuntimeError("SIAM runner returned without a nonempty trajectory")
        payload = _load_json_object(trajectory, label="completed SIAM trajectory")
        checks, resume_terminal_proof = _validate_completed_run(
            payload,
            action=action,
            seed_manifest=seed_manifest,
            seed_runtime_input=load_siam_runtime_input(seed_path),
            resume_payload=resume_payload,
        )
        _write_json(
            receipt,
            {
                **base_receipt,
                "status": "COMPLETED",
                "checks": checks,
                "resume_terminal_proof": resume_terminal_proof,
                "trajectory_bytes": trajectory.stat().st_size,
                "trajectory_sha256": _sha256(trajectory),
            },
        )
    except BaseException as exc:
        _write_json(
            receipt,
            {
                **base_receipt,
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "trajectory_present": trajectory.is_file(),
            },
        )
        raise
    return 0


__all__ = [
    "PreparedSIAMCHTCAction",
    "SIAMSourceEpochRebaseline",
    "SIAM_NB4_ACTIVATION_CUT_LADDER",
    "SIAM_NB4_CHTC_ACTION_SCHEMA_V3",
    "SIAM_NB4_CHTC_ACTION_SCHEMA_V4",
    "SIAM_NB4_CHTC_ACTION_SCHEMA_V5",
    "SIAM_NB4_CHTC_MANIFEST_SCHEMA_V3",
    "SIAM_NB4_CHTC_MANIFEST_SCHEMA_V4",
    "SIAM_NB4_CHTC_MANIFEST_SCHEMA_V5",
    "SIAM_NB4_CHTC_RECEIPT_SCHEMA_V3",
    "SIAM_NB4_CHTC_RECEIPT_SCHEMA_V4",
    "SIAM_NB4_CHTC_RECEIPT_SCHEMA_V5",
    "SIAM_NB4_RESUME_TERMINAL_PROOF_V1",
    "SIAM_SOURCE_EPOCH_CLOSURE_V1",
    "SIAM_SOURCE_EPOCH_CONTRACT_V1",
    "SIAM_SOURCE_EPOCH_REBASELINE_V1",
    "SIAM_SOURCE_TREE_CONTRACT_V1",
    "build_siam_source_epoch_closure",
    "build_siam_source_epoch_contract",
    "prepare_siam_nb4_chtc_action",
    "run_prepared_siam_action",
    "validate_siam_nb4_completion_receipt",
    "write_siam_source_epoch_closure",
]
