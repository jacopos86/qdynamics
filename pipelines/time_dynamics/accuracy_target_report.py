"""Compare methods at matched accuracy, not at a matched threshold.

Each adaptive method carries its own activation threshold, and on this problem
that threshold is not a usable common comparison axis.  This report treats the
threshold as a per-cell search variable and compares resources at a declared
accuracy target.

This report instead fixes an accuracy target and asks, per cell, what each
method costs to reach it:

* the minimum-final-support threshold whose mean |dE| meets the target, as a
  quick diagnostic before Qiskit compilation;
* cells that do not yet meet the target are marked EXTEND LADDER, and the next
  tighter rung must be run.

Usage:
    PYTHONPATH=. python3 -m pipelines.time_dynamics.accuracy_target_report \
        --root output/frontier --target 1e-4
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _seed_path(run: dict[str, Any]) -> Path:
    raw = run.get("source_artifact_json")
    if raw is None:
        raw = ((run.get("run_lock") or {}).get("physics") or {}).get(
            "seed_artifact_json"
        )
    if raw is None:
        raise ValueError("run has no recorded seed artifact")
    path = Path(str(raw))
    return path if path.is_absolute() else REPO / path


def _effective_threshold(run: dict[str, Any], arm: str) -> float:
    config = (run.get("summary") or {}).get("support_patch_config") or {}
    key = "avqds_l2_cut" if arm == "avqds" else "insertion_l2_cut"
    if config.get(key) is None:
        raise ValueError(f"run does not record {key}")
    return float(config[key])


def _mean_energy_error(
    run_path: Path,
    *,
    arm: str,
    declared_threshold: float,
    reference_rows: list[dict[str, Any]] | None = None,
) -> tuple[float, int, dict[str, Any]]:
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "output"))
    from exact_driven_reference import exact_rows  # noqa: E402

    run = json.loads(run_path.read_text())
    actual_threshold = _effective_threshold(run, arm)
    if not np.isclose(actual_threshold, declared_threshold, rtol=1.0e-12, atol=0.0):
        raise ValueError(
            f"{run_path}: directory declares cut {declared_threshold:g}, "
            f"artifact records {actual_threshold:g}"
        )
    ref = (
        exact_rows(str(_seed_path(run)), str(run_path))
        if reference_rows is None
        else reference_rows
    )
    rows = run["plot_rows"]
    if len(rows) != len(ref):
        raise ValueError(
            f"{run_path}: trajectory/reference length mismatch "
            f"({len(rows)} != {len(ref)})"
        )
    err = [abs(r["energy_expectation"] - x["energy"]) for r, x in zip(rows, ref)]
    return (
        float(np.mean(err)),
        int(run["summary"]["runtime_parameter_count_final"]),
        run,
    )


def collect(root: Path) -> dict[tuple[str, str], list[tuple[float, float, int]]]:
    """(drive, arm) -> [(threshold, mean |dE|, support), ...] sorted by threshold."""

    out: dict[tuple[str, str], list[tuple[float, float, int]]] = {}
    locks_by_drive: dict[str, list[dict[str, Any]]] = {}
    reference_cache: dict[str, list[dict[str, Any]]] = {}
    for run_path in sorted(root.glob("*/run.json")):
        m = re.match(r"(?P<drive>[a-z0-9]+)_(?P<arm>avqds|append_only|exchange)_cut(?P<cut>[0-9.e+-]+)$",
                     run_path.parent.name)
        if not m:
            continue
        threshold = float(m["cut"])
        raw_run = json.loads(run_path.read_text())
        lock = raw_run.get("run_lock")
        if not isinstance(lock, dict):
            raise ValueError(f"{run_path}: missing run_lock")
        fingerprint = str(lock.get("physics_fingerprint", "")).strip()
        if not fingerprint:
            raise ValueError(f"{run_path}: missing physics_fingerprint")
        if fingerprint not in reference_cache:
            sys.path.insert(0, str(REPO))
            sys.path.insert(0, str(REPO / "output"))
            from exact_driven_reference import exact_rows

            reference_cache[fingerprint] = exact_rows(
                str(_seed_path(raw_run)), str(run_path)
            )
        err, support, run = _mean_energy_error(
            run_path,
            arm=m["arm"],
            declared_threshold=threshold,
            reference_rows=reference_cache[fingerprint],
        )
        locks_by_drive.setdefault(m["drive"], []).append(lock)
        out.setdefault((m["drive"], m["arm"]), []).append(
            (threshold, err, support)
        )
    from pipelines.time_dynamics.run_lock import assert_comparable

    for drive, locks in locks_by_drive.items():
        try:
            assert_comparable(locks)
        except ValueError as exc:
            raise ValueError(f"drive {drive!r} contains incomparable runs") from exc
    for key in out:
        out[key].sort(key=lambda r: -r[0])
    return out


def _next_tighter_cut(cut: float) -> float:
    """Continue the 1--3 logarithmic ladder below ``cut``."""

    if not np.isfinite(cut) or cut <= 0.0:
        raise ValueError("cut must be finite and positive")
    exponent = int(np.floor(np.log10(cut)))
    mantissa = cut / (10.0 ** exponent)
    if mantissa >= 2.0:
        return 1.0 * (10.0 ** exponent)
    return 3.0 * (10.0 ** (exponent - 1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="output/frontier")
    ap.add_argument("--target", type=float, default=1.0e-4)
    args = ap.parse_args(argv)

    data = collect(Path(args.root))
    drives = sorted({k[0] for k in data})
    arms = sorted({k[1] for k in data})

    print(f"Accuracy target: mean |dE| <= {args.target:g}\n")
    print(f"{'drive':12s} {'arm':12s} {'best |dE|':>11s} {'at cut':>9s} "
          f"{'target rung / next action':>28s} {'support there':>14s}")
    for drive in drives:
        for arm in arms:
            series = data.get((drive, arm))
            if not series:
                continue
            best = min(series, key=lambda r: r[1])
            meeting = [r for r in series if r[1] <= args.target]
            if meeting:
                # Quick diagnostic only.  Final resource reporting recompiles
                # this ansatz in Qiskit and reports N2q, total depth, and D2q.
                pick = min(meeting, key=lambda r: (r[2], r[1]))
                verdict = f"{pick[0]:.1e}"
                support = f"{pick[2]}"
            else:
                next_cut = _next_tighter_cut(min(r[0] for r in series))
                verdict = f"EXTEND TO {next_cut:.1e}"
                support = "--"
            print(f"{drive:12s} {arm:12s} {best[1]:11.2e} {best[0]:9.1e} "
                  f"{verdict:>24s} {support:>14s}")
        print()

    print("Cells reaching the target on the current ladder, by arm:")
    for arm in arms:
        reached = sum(
            1 for drive in drives
            if data.get((drive, arm)) and any(r[1] <= args.target for r in data[(drive, arm)])
        )
        total = sum(1 for drive in drives if data.get((drive, arm)))
        print(f"  {arm:12s} {reached}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
