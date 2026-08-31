"""Run locks: bind a trajectory to the inputs that define it.

A method comparison is only a comparison if every arm demonstrably shared the
same physics.  Before this module, that was asserted by reading launch scripts,
which is how one arm silently ran another arm's configuration and produced
byte-identical results that nothing flagged.

A run lock records, inside the result artifact itself, the identity of
everything a reader would need to reproduce or to prove comparability:

* **physics** — seed artifact path and content hash, Hamiltonian family, phonon
  cutoff, drive profile, and the reporting time grid.  The path is provenance;
  only the content hash participates in comparison identity;
* **numerics** — integrator, inverse policy, and repair configuration;
* **shared protocol** — the time-step controller and exact ordered effective
  candidate-pool contract, together with the physics and numerics;
* **policy** — the structural decision rule and its guards;
* **code** — the commit the run executed.

The comparison fingerprint deliberately excludes structural policy,
activation cut, and code: two arms *should* differ in policy and may use
different cuts to reach matched accuracy, but must agree on their seed,
physics, candidate pool, numerical solve, and time-step controller.
``assert_comparable`` enforces exactly that separation.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

RUN_LOCK_SCHEMA_V1 = "paper_ii_run_lock_v1"
RUN_LOCK_SCHEMA_V2 = "paper_ii_run_lock_v2"
RUN_LOCK_SCHEMA_V3 = "paper_ii_run_lock_v3"
RUN_LOCK_SCHEMA_V4 = "paper_ii_run_lock_v4"


def file_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return "missing"
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_code_revision(repo_root: str | Path | None = None) -> str:
    """Return the checked-out revision, marking a dirty worktree explicitly."""

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root) if repo_root else None,
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            rev = out.stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(repo_root) if repo_root else None,
                capture_output=True, text=True, timeout=10,
            )
            suffix = "-dirty" if dirty.returncode == 0 and dirty.stdout.strip() else ""
            return f"{rev}{suffix}"
    except Exception:
        pass
    return "unknown"


def build_run_lock(
    *,
    seed_artifact_json: str | None,
    family_key: str | None,
    n_ph_max: int | None,
    times: Sequence[float],
    drive_profile: Mapping[str, Any] | None,
    integrator_method: str,
    inverse_policy: Mapping[str, Any],
    solve_repair: Mapping[str, Any] | None,
    structural_policy: str,
    guards: Mapping[str, Any],
    verified_seed_sha256: str | None = None,
    time_step_controller: Mapping[str, Any] | None = None,
    candidate_pool: Mapping[str, Any] | None = None,
    exact_reference_json: str | None = None,
    repo_root: str | Path | None = None,
    code_revision: str | None = None,
) -> dict[str, Any]:
    """Assemble the lock recorded inside a trajectory artifact."""

    times_list = [float(t) for t in times]
    ordered_time_grid_sha256 = hashlib.sha256(
        json.dumps(times_list, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    observed_seed_sha256 = file_sha256(seed_artifact_json)
    if verified_seed_sha256 is None:
        bound_seed_sha256 = observed_seed_sha256
    else:
        bound_seed_sha256 = str(verified_seed_sha256).strip().lower()
        if len(bound_seed_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in bound_seed_sha256
        ):
            raise ValueError("verified_seed_sha256 must be a SHA-256 digest.")
        if seed_artifact_json is None:
            raise ValueError(
                "verified_seed_sha256 requires a seed_artifact_json path."
            )
        if observed_seed_sha256 != bound_seed_sha256:
            raise ValueError(
                "Seed artifact changed after verified load: "
                f"expected {bound_seed_sha256}, observed {observed_seed_sha256}."
            )
    physics = {
        "seed_artifact_json": seed_artifact_json,
        "seed_sha256": bound_seed_sha256,
        "family_key": None if family_key is None else str(family_key),
        "n_ph_max": None if n_ph_max is None else int(n_ph_max),
        "time_grid": {
            "t_initial": times_list[0] if times_list else None,
            "t_final": times_list[-1] if times_list else None,
            "point_count": len(times_list),
            "ordered_values_sha256": ordered_time_grid_sha256,
        },
        "drive_profile": dict(drive_profile) if drive_profile else None,
        "exact_reference_json": exact_reference_json,
        "exact_reference_sha256": file_sha256(exact_reference_json),
    }
    numerics = {
        "integrator_method": str(integrator_method),
        "inverse_policy": dict(inverse_policy),
        "solve_repair": dict(solve_repair) if solve_repair else None,
    }
    shared_protocol = {
        "physics_fingerprint": physics_fingerprint(physics),
        "numerics": numerics,
        "time_step_controller": (
            dict(time_step_controller) if time_step_controller else None
        ),
        "candidate_pool": dict(candidate_pool) if candidate_pool else None,
    }
    lock = {
        "schema": RUN_LOCK_SCHEMA_V4,
        "physics": physics,
        "physics_fingerprint": physics_fingerprint(physics),
        "numerics": numerics,
        "shared_protocol": shared_protocol,
        "comparison_fingerprint": comparison_fingerprint(shared_protocol),
        "policy": {
            "structural_policy": str(structural_policy),
            "guards": dict(guards),
        },
        "code_revision": (
            str(code_revision).strip()
            if code_revision is not None
            else resolve_code_revision(repo_root)
        ),
    }
    return lock


def physics_fingerprint(physics: Mapping[str, Any]) -> str:
    """Hash of the physics a comparison must hold fixed.

    Excludes location-only provenance.  The seed content hash identifies the
    prepared input, while ``seed_artifact_json`` may legitimately differ after
    the same bytes are copied into separate CHTC packages.  Exact-reference
    fields are reporting artifacts rather than trajectory inputs.
    """

    subset = {
        key: value
        for key, value in physics.items()
        if key
        not in (
            "seed_artifact_json",
            "exact_reference_json",
            "exact_reference_sha256",
        )
    }
    blob = json.dumps(subset, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def comparison_fingerprint(shared_protocol: Mapping[str, Any]) -> str:
    """Hash everything that must remain shared across structural arms."""

    blob = json.dumps(
        dict(shared_protocol), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class IncomparableRunsError(ValueError):
    """Raised when runs do not share the declared comparison protocol."""


def assert_comparable(locks: Iterable[Mapping[str, Any]]) -> str:
    """Require one physics, pool, numerical stack, and step controller.

    Call this before aggregating arms into a table or figure.  It is the check
    that turns "we believe these ran the same physics" into something the
    artifacts prove.
    """

    seen: dict[str, list[str]] = {}
    for lock in locks:
        if not lock.get("comparison_fingerprint"):
            fp = "missing"
        else:
            physics = lock.get("physics")
            shared_protocol = lock.get("shared_protocol")
            if not isinstance(physics, Mapping) or not isinstance(
                shared_protocol, Mapping
            ):
                fp = "missing"
            else:
                canonical_shared = dict(shared_protocol)
                canonical_shared["physics_fingerprint"] = physics_fingerprint(
                    physics
                )
                fp = comparison_fingerprint(canonical_shared)
        label = str(
            (lock.get("policy") or {}).get("structural_policy", "unknown")
        )
        seen.setdefault(fp, []).append(label)
    if not seen:
        raise IncomparableRunsError("No run locks supplied.")
    if "missing" in seen:
        raise IncomparableRunsError(
            "Run lock lacks the shared-protocol comparison fingerprint; "
            "pool, numerics, and time-step parity cannot be established."
        )
    if len(seen) > 1:
        detail = "; ".join(f"{fp[:12]}: {sorted(v)}" for fp, v in sorted(seen.items()))
        raise IncomparableRunsError(
            "Runs do not share one physics/pool/numerics/time-step protocol, "
            f"so they cannot be compared: {detail}"
        )
    return next(iter(seen))


__all__ = [
    "RUN_LOCK_SCHEMA_V1",
    "RUN_LOCK_SCHEMA_V2",
    "RUN_LOCK_SCHEMA_V3",
    "RUN_LOCK_SCHEMA_V4",
    "IncomparableRunsError",
    "assert_comparable",
    "build_run_lock",
    "comparison_fingerprint",
    "file_sha256",
    "physics_fingerprint",
    "resolve_code_revision",
]
