"""Compare methods at matched accuracy, not at a matched threshold.

Each adaptive method carries its own accuracy threshold, and on this problem
that threshold is not a usable comparison axis: tightening it improves accuracy
monotonically on 1 of 3 drives for the comparator and 0 of 3 for this route,
while a 100x range moves support by only ~10 coordinates. Reading either
method's quality off one setting of its own dial is therefore meaningless.

This report instead fixes an accuracy target and asks, per cell, what each
method costs to reach it:

* the cheapest threshold whose mean |dE| meets the target, and the support size
  there -- that is the comparable number;
* cells that no threshold in the ladder reaches are reported as NOT CONVERGED,
  which is an outcome rather than a gap to be averaged over.

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

import numpy as np

REPO = "/Users/jakestrobel/local_repos/Holstein_test_fullclone_3"
SEED = "chtc/paper_ii_production_v2/input/seeds/hh_snake_nph1.json"


def _mean_energy_error(run_path: Path) -> tuple[float, int]:
    sys.path.insert(0, REPO)
    sys.path.insert(0, f"{REPO}/output")
    from exact_driven_reference import exact_rows  # noqa: E402

    run = json.loads(run_path.read_text())
    ref = exact_rows(SEED, str(run_path))
    rows = run["plot_rows"]
    err = [abs(r["energy_expectation"] - x["energy"]) for r, x in zip(rows, ref)]
    return float(np.mean(err)), int(run["summary"]["runtime_parameter_count_final"])


def collect(root: Path) -> dict[tuple[str, str], list[tuple[float, float, int]]]:
    """(drive, arm) -> [(threshold, mean |dE|, support), ...] sorted by threshold."""

    out: dict[tuple[str, str], list[tuple[float, float, int]]] = {}
    for run_path in sorted(root.glob("*/run.json")):
        m = re.match(r"(?P<drive>[a-z0-9]+)_(?P<arm>avqds|append_only|exchange)_cut(?P<cut>[0-9.e+-]+)$",
                     run_path.parent.name)
        if not m:
            continue
        err, support = _mean_energy_error(run_path)
        out.setdefault((m["drive"], m["arm"]), []).append(
            (float(m["cut"]), err, support)
        )
    for key in out:
        out[key].sort(key=lambda r: -r[0])
    return out


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
          f"{'cheapest meeting target':>24s} {'support there':>14s}")
    for drive in drives:
        for arm in arms:
            series = data.get((drive, arm))
            if not series:
                continue
            best = min(series, key=lambda r: r[1])
            meeting = [r for r in series if r[1] <= args.target]
            if meeting:
                # cheapest = fewest coordinates among those meeting the target
                pick = min(meeting, key=lambda r: (r[2], r[1]))
                verdict = f"{pick[0]:.1e}"
                support = f"{pick[2]}"
            else:
                verdict = "NOT CONVERGED"
                support = "--"
            print(f"{drive:12s} {arm:12s} {best[1]:11.2e} {best[0]:9.1e} "
                  f"{verdict:>24s} {support:>14s}")
        print()

    print("Cells that reach the target, by arm:")
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
