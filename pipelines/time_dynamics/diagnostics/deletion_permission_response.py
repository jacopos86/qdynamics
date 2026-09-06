"""Pre-permission angle sum versus post-refit certification, from stored runs.

For every deletion-bearing finalist an AP-McLachlan run materialized, pair the
batch's accumulated effective rotation angle ``a_D = sum_{mu in D} |theta_mu c_mu|``
(Paper II Eq. deletion_angle_sum, evaluated on the support the finalist was
nominated from) with the post-refit ray distance and velocity-continuity value
the certification recorded.  Supports are replayed round by round inside each
time point: a coordinate appended in an earlier round of the same time point
enters later rounds at zero angle, so a deletion of it has ``a_D = 0``.

Angles are taken at time-point entry; a certified deletion-bearing commit
refits the retained angles before later rounds, and that adjustment is not
replayed.  Rotation coefficients ``c_mu`` come from every run's final
``resume_state.coordinates``; a label absent from all of them is reported as
unattributed rather than guessed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_PAULI_PREFIX = re.compile(r"^pauli:")
_INSERTION_SUFFIX = re.compile(r"::insr\d+c\d+o\d+::r\d+::[a-z]+$")
_REPLAY_SUFFIX = re.compile(r"::inserted_round\d+$")

BINS: tuple[tuple[float, float, str], ...] = (
    (0.0, 0.0, "a_D = 0 (same-time-point append)"),
    (0.0, 2.0e-3, "0 < a_D < 2e-3"),
    (2.0e-3, 5.0e-3, "2e-3 <= a_D < 5e-3"),
    (5.0e-3, 1.0e-2, "5e-3 <= a_D < 1e-2"),
    (1.0e-2, 2.0e-2, "1e-2 <= a_D < 2e-2"),
    (2.0e-2, 5.0e-2, "2e-2 <= a_D < 5e-2"),
    (5.0e-2, 1.0e-1, "5e-2 <= a_D < 1e-1"),
    (1.0e-1, math.inf, "a_D >= 1e-1"),
)


def _norm(label: str) -> str:
    return _PAULI_PREFIX.sub("", str(label))


def _term(label: str) -> str:
    """Generator-term identity: the coefficient is a property of ``atom::rN::word``."""

    value = _REPLAY_SUFFIX.sub("", _norm(label))
    while True:
        stripped = _INSERTION_SUFFIX.sub("", value)
        if stripped == value:
            return value
        value = stripped


def _coefficient(coef: Mapping[str, float], label: str) -> float | None:
    direct = coef.get(_norm(label))
    if direct is not None:
        return direct
    return coef.get(_term(label))


def coefficient_map(docs: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for doc in docs:
        for entry in (doc.get("resume_state") or {}).get("coordinates") or ():
            value = abs(float(entry["coeff_real"]))
            out[_norm(entry["label"])] = value
            out.setdefault(_term(entry["label"]), value)
    return out


def _entry_theta(prev: Mapping[str, Any], point: Mapping[str, Any]) -> tuple[list[float] | None, str]:
    """Angles entering ``point``: recorded when the integrator stores them.

    The raw-coordinate (delta-theta) controller does not record ``theta_next``;
    there the entry angles are estimated to first order from the previous time
    point's angles and velocity, ``theta + dt * theta_dot``.
    """

    integration = prev.get("integration_to_next") or {}
    if integration.get("theta_next") is not None:
        return [float(v) for v in integration["theta_next"]], "recorded"
    theta = prev.get("theta_runtime")
    theta_dot = (prev.get("fixed_step") or {}).get("theta_dot")
    if theta is None or theta_dot is None or len(theta) != len(theta_dot):
        return None, "unavailable"
    dt = float(point["time"]) - float(prev["time"])
    return [float(a) + dt * float(b) for a, b in zip(theta, theta_dot)], "euler_estimate"


def attempt_rows(
    doc: Mapping[str, Any], *, run: str, coef: Mapping[str, float]
) -> tuple[list[dict[str, Any]], Counter]:
    """Replay supports round by round and attribute ``a_D`` to every attempt."""

    rows: list[dict[str, Any]] = []
    skipped: Counter = Counter()
    points = doc["trajectory"]["points"]
    for k in range(1, len(points)):
        prev, point = points[k - 1], points[k]
        entry_labels = list(((prev.get("fixed_step") or {}).get("support_labels")) or ())
        entry_theta, theta_source = _entry_theta(prev, point)
        if not entry_labels or entry_theta is None or len(entry_theta) != len(entry_labels):
            skipped["no_entry_support"] += 1
            continue
        labels = entry_labels
        theta = list(entry_theta)
        metadata = (point.get("patch_decision") or {}).get("metadata") or {}
        rounds = metadata.get("selection_round_work") or ()
        # Post-refit ray and velocity values are recorded only for the final
        # round's attempts (``metadata.attempts``, same order).
        final_attempts = list(metadata.get("attempts") or ())
        for r, work in enumerate(rounds):
            attempts = list(((work.get("estimator_work") or {}).get("certification_attempts")) or ())
            gated = final_attempts if (r == len(rounds) - 1 and len(final_attempts) == len(attempts)) else [{}] * len(attempts)
            committed = None
            for attempt, gate in zip(attempts, gated):
                removed = [int(i) for i in attempt.get("removed_runtime_indices") or ()]
                if removed:
                    if any(i >= len(labels) for i in removed):
                        skipped["index_out_of_range"] += 1
                        continue
                    coeffs = [None if theta[i] == 0.0 else _coefficient(coef, labels[i]) for i in removed]
                    if any(c is None and theta[i] != 0.0 for i, c in zip(removed, coeffs)):
                        skipped["coefficient_unattributed"] += 1
                        a_d = None
                    else:
                        a_d = float(sum(abs(theta[i]) * (c or 0.0) for i, c in zip(removed, coeffs)))
                    rows.append(
                        {
                            "run": run,
                            "time": float(point["time"]),
                            "round": r,
                            "kind": str(attempt.get("kind")),
                            "family": str(attempt.get("family")),
                            "deleted_count": len(removed),
                            "a_D": a_d,
                            "ray_bound": None if a_d is None else math.sin(min(math.pi / 2.0, a_d)),
                            "same_time_point_append": bool(
                                a_d is not None and all(theta[i] == 0.0 for i in removed)
                            ),
                            "entry_theta_source": theta_source,
                            "post_refit_ray": gate.get("ray_distance"),
                            "post_refit_velocity": gate.get("smoothness_eta"),
                            "reason": str(attempt.get("reason")),
                            "certified": attempt.get("reason") == "certified",
                        }
                    )
                if attempt.get("reason") == "certified":
                    committed = attempt
            if committed is None:
                continue
            # Apply the round's commit to the working support: deletions first
            # (by entry-of-round runtime index), then the insertion at its cut.
            removed = sorted({int(i) for i in committed.get("removed_runtime_indices") or ()}, reverse=True)
            for i in removed:
                del labels[i]
                del theta[i]
            width = int(committed.get("patched_runtime_parameter_count", 0)) - (
                int(committed.get("base_runtime_parameter_count", 0)) - len(removed)
            )
            for atom, cut in committed.get("inserted_selection") or ():
                for j in range(max(width, 0)):
                    labels.insert(int(cut) + j, f"{atom}::inserted_round{r}")
                    theta.insert(int(cut) + j, 0.0)
                width = 0
    return rows, skipped


def summarize(rows: Sequence[Mapping[str, Any]], *, ray_max: float, velocity_max: float) -> list[dict[str, Any]]:
    out = []
    for lo, hi, name in BINS:
        if lo == hi == 0.0:
            sel = [r for r in rows if r["a_D"] is not None and r["a_D"] == 0.0]
        else:
            sel = [r for r in rows if r["a_D"] is not None and lo <= r["a_D"] < hi and r["a_D"] > 0.0]
        rays = [float(r["post_refit_ray"]) for r in sel if r["post_refit_ray"] is not None]
        vels = [float(r["post_refit_velocity"]) for r in sel if r["post_refit_velocity"] is not None]
        out.append(
            {
                "bin": name,
                "tested": len(sel),
                "ray_known": len(rays),
                "certified": sum(bool(r["certified"]) for r in sel),
                "ray_within": sum(v <= ray_max for v in rays),
                "velocity_within": sum(v <= velocity_max for v in vels),
                "median_post_refit_ray": None if not rays else float(np.median(rays)),
                "median_ray_over_a_D": (
                    None
                    if not sel or lo == hi == 0.0
                    else float(np.median([float(r["post_refit_ray"]) / r["a_D"] for r in sel if r["post_refit_ray"] is not None]))
                ),
            }
        )
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="name=path/to/run.json entries")
    parser.add_argument("--ray-max", type=float, default=2.0e-3)
    parser.add_argument("--velocity-max", type=float, default=1.0e-3)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args(argv)
    named = []
    for entry in args.runs:
        name, _, path = entry.partition("=")
        named.append((name or Path(path).parent.name, json.loads(Path(path).read_text(encoding="utf-8"))))
    coef = coefficient_map([doc for _, doc in named])
    rows: list[dict[str, Any]] = []
    skipped: Counter = Counter()
    for name, doc in named:
        r, s = attempt_rows(doc, run=name, coef=coef)
        rows.extend(r)
        skipped.update(s)
    table = summarize(rows, ray_max=args.ray_max, velocity_max=args.velocity_max)
    print(f"runs {len(named)}; deletion-bearing finalists {len(rows)}; skipped {dict(skipped)}")
    print("| a_D bin | tested | certified | with post-refit values | ray within | velocity within | median post-refit ray | median ray / a_D |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for t in table:
        n = t["tested"]; m = t["ray_known"]
        pct = lambda v, d: "" if not d else f"{v} ({v / d:.0%})"  # noqa: E731
        med = "" if t["median_post_refit_ray"] is None else f"{t['median_post_refit_ray']:.2e}"
        ratio = "" if t["median_ray_over_a_D"] is None else f"{t['median_ray_over_a_D']:.2f}"
        print(f"| {t['bin']} | {n} | {pct(t['certified'], n)} | {m} | {pct(t['ray_within'], m)} | {pct(t['velocity_within'], m)} | {med} | {ratio} |")
    sources = Counter(r["entry_theta_source"] for r in rows)
    print("entry-angle source of attributed rows:", dict(sources))
    print("\nper run: accepted exchange commits by what they deleted")
    for name, _ in named:
        ex = [r for r in rows if r["run"] == name and r["kind"] == "exchange" and r["certified"]]
        same = sum(r["same_time_point_append"] for r in ex)
        dormant = sum((not r["same_time_point_append"]) and r["a_D"] is not None and r["a_D"] < 2.0e-3 for r in ex)
        rotated = sum(r["a_D"] is not None and r["a_D"] >= 2.0e-3 for r in ex)
        print(f"  {name}: exchanges {len(ex)} = same-time-point append {same} + dormant entry coordinate (a_D < 2e-3) {dormant} + rotated (a_D >= 2e-3) {rotated}")
    if args.output_json is not None:
        args.output_json.write_text(json.dumps({"rows": rows, "table": table, "skipped": dict(skipped)}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
