"""Run locks: bind a trajectory to the inputs that define it.

A method comparison is only a comparison if every arm demonstrably shared the
same physics.  Before this module, that was asserted by reading launch scripts,
which is how one arm silently ran another arm's configuration and produced
byte-identical results that nothing flagged.

A run lock records, inside the result artifact itself, the identity of
everything a reader would need to reproduce or to prove comparability:

* **physics** — seed artifact path and content hash, Hamiltonian family, phonon
  cutoff, drive profile, and the reporting time grid;
* **numerics** — integrator, inverse policy, and repair configuration;
* **policy** — the structural decision rule and its guards;
* **code** — the commit the run executed.

The physics fingerprint deliberately excludes policy and code: two arms of a
comparison *should* differ in policy, and may differ in code revision, but must
agree on physics or the comparison is meaningless.  ``assert_comparable``
enforces exactly that separation.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

RUN_LOCK_SCHEMA_V1 = "paper_ii_run_lock_v1"


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


def _code_revision(repo_root: str | Path | None = None) -> str:
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
    exact_reference_json: str | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Assemble the lock recorded inside a trajectory artifact."""

    times_list = [float(t) for t in times]
    physics = {
        "seed_artifact_json": seed_artifact_json,
        "seed_sha256": file_sha256(seed_artifact_json),
        "family_key": None if family_key is None else str(family_key),
        "n_ph_max": None if n_ph_max is None else int(n_ph_max),
        "time_grid": {
            "t_initial": times_list[0] if times_list else None,
            "t_final": times_list[-1] if times_list else None,
            "point_count": len(times_list),
        },
        "drive_profile": dict(drive_profile) if drive_profile else None,
        "exact_reference_json": exact_reference_json,
        "exact_reference_sha256": file_sha256(exact_reference_json),
    }
    lock = {
        "schema": RUN_LOCK_SCHEMA_V1,
        "physics": physics,
        "physics_fingerprint": physics_fingerprint(physics),
        "numerics": {
            "integrator_method": str(integrator_method),
            "inverse_policy": dict(inverse_policy),
            "solve_repair": dict(solve_repair) if solve_repair else None,
        },
        "policy": {
            "structural_policy": str(structural_policy),
            "guards": dict(guards),
        },
        "code_revision": _code_revision(repo_root),
    }
    return lock


def physics_fingerprint(physics: Mapping[str, Any]) -> str:
    """Hash of the physics a comparison must hold fixed.

    Excludes the exact-reference pointer: a reference is a reporting artifact,
    not an input to the trajectory, and one arm may carry it while another does
    not without the two becoming incomparable.
    """

    subset = {k: v for k, v in physics.items()
              if k not in ("exact_reference_json", "exact_reference_sha256")}
    blob = json.dumps(subset, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class IncomparableRunsError(ValueError):
    """Raised when runs that are being compared do not share their physics."""


def assert_comparable(locks: Iterable[Mapping[str, Any]]) -> str:
    """Require every run to share one physics fingerprint; return it.

    Call this before aggregating arms into a table or figure.  It is the check
    that turns "we believe these ran the same physics" into something the
    artifacts prove.
    """

    seen: dict[str, list[str]] = {}
    for lock in locks:
        fp = str(lock.get("physics_fingerprint") or "missing")
        label = str(
            (lock.get("policy") or {}).get("structural_policy", "unknown")
        )
        seen.setdefault(fp, []).append(label)
    if not seen:
        raise IncomparableRunsError("No run locks supplied.")
    if len(seen) > 1:
        detail = "; ".join(f"{fp[:12]}: {sorted(v)}" for fp, v in sorted(seen.items()))
        raise IncomparableRunsError(
            "Runs do not share a physics fingerprint, so they cannot be "
            f"compared: {detail}"
        )
    return next(iter(seen))


__all__ = [
    "RUN_LOCK_SCHEMA_V1",
    "IncomparableRunsError",
    "assert_comparable",
    "build_run_lock",
    "file_sha256",
    "physics_fingerprint",
]
