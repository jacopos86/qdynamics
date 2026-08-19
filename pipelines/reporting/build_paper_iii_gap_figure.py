#!/usr/bin/env python3
"""Emit the Paper III excited-state gap figure.

Single-plot figure (one plot per figure, repo plotting convention): the
lowest six excitation gaps omega_nu = E_nu - E_0 of every benchmark
regime, comparing the exact sector-restricted spectrum against the
complete fixed linear-response class and the geometry-selected support
with certified exchange. Where a method's marker departs from the exact
ladder, that root is unresolved by that method; the max-error tables
report the same information as a single scalar per regime.

Regenerable from the committed multi-root evidence JSON. Reporting only;
never feeds controller decisions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MULTIROOT_JSON = (
    REPO_ROOT
    / "output/diagnostics/paper_iii_multiroot_sweep_20260818_v1/multiroot_sweep_epsstop.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "MATH/paper_details/generated/paper_iii_gap_figure.pdf"

_ARMS = (
    ("fixed_linear_response_complete", "fixed linear-response class", "#c05020", "s"),
    ("exchange_dominance_R6", "selected + certified exchange", "#2673b8", "o"),
)
_REGIME_LABEL = {
    "weak_weak": r"weak--weak",
    "intermediate_weak": r"interm.--weak",
    "strong_weak_u8": r"strong--weak",
    "weak_strong": r"weak--strong",
    "intermediate_strong": r"interm.--strong",
    "strong_strong_u8": r"strong--strong",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multiroot-json", type=Path, default=MULTIROOT_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.multiroot_json).read_text(encoding="utf-8"))
    regimes = [r for r in _REGIME_LABEL if r in payload["regimes"]]
    if not regimes:
        raise SystemExit("no known regimes in the multiroot evidence")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    width = 1.0
    xticks: list[float] = []
    xlabels: list[str] = []

    for slot, regime in enumerate(regimes):
        record = payload["regimes"][regime]
        e0 = record.get("reference_ground_energy")
        refs = record["reference_excitations"]
        if e0 is None:
            raise SystemExit(
                "multiroot evidence lacks reference_ground_energy; rerun "
                "pipelines/exact_bench/paper_iii_qse_multiroot_sweep.py"
            )
        base = slot * (width + 0.45)
        offsets = [base + width * (i / (len(refs) - 1)) for i in range(len(refs))]
        xticks.append(base + width / 2.0)
        xlabels.append(_REGIME_LABEL[regime])

        exact_gaps = [float(r) - float(e0) for r in refs]
        ax.plot(
            offsets,
            exact_gaps,
            "_",
            color="#222222",
            markersize=16,
            markeredgewidth=2.0,
            label="exact sector" if slot == 0 else None,
            zorder=1,
        )
        for arm_key, arm_label, color, marker in _ARMS:
            arm = record["arms"].get(arm_key)
            if arm is None or "root_energies" not in arm:
                continue
            gaps = [
                None if e is None else float(e) - float(e0) for e in arm["root_energies"]
            ]
            xs = [x for x, g in zip(offsets, gaps) if g is not None]
            ys = [g for g in gaps if g is not None]
            ax.plot(
                xs,
                ys,
                marker,
                color=color,
                markersize=5,
                markerfacecolor="none",
                markeredgewidth=1.3,
                label=arm_label if slot == 0 else None,
                zorder=2,
            )

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel(r"excitation gap $\omega_\nu = E_\nu - E_0$")
    ax.set_xlabel("regime (six lowest excitations, left to right within each group)")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=8, loc="upper left")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    png = args.output.with_suffix(".png")
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"wrote {args.output}")
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
