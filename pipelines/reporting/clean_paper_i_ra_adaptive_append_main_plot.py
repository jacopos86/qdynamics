#!/usr/bin/env python3
"""Build the tentative Paper-I main singleton comparison figure.

Published Append-ADAPT tolerance-matched baseline versus the adaptive-campaign
append-RA arm (complete) and commutation-reduced insertion arm (completed and
live partial cells), drawn in the manuscript's clean two-by-three format.
Baseline loading and the k* (baseline-plateau) rule are delegated to or copied
exactly from the campaign comparison builder. Series are distinguished by
linestyle and sparse marker shape as well as color, for color-blind and
black-and-white legibility. Cell trajectories are read from summary.json only;
the multi-gigabyte result.json payloads are never opened.
Evidence class: preliminary (partial campaign), author-authorized 2026-08-18.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
CMP_PATH = ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "build_paper_i_ra_phase123_qiskit_comparison_pdf_20260817.py"
)
CELLS = ROOT / "output/local_runs/paper_i_ra_allphase_adaptive_20260817_comparison_data"
WORKDIR = CELLS / "workdir"
LIVE_TAILS = CELLS / "live_tails"
OUT_DIR = ROOT / "MATH/paper_details/figures/paper_i_ra_adaptive_append_main_20260818"
OUTPUT = OUT_DIR / "paper_i_ra_adaptive_append_main.pdf"
PROVENANCE = OUT_DIR / "paper_i_ra_adaptive_append_main_provenance.json"

REGIMES = [
    ("weak_weak", "Weak--weak", 3),
    ("intermediate_weak", "Intermediate--weak", 3),
    ("strong_weak_u8", "Strong--weak", 3),
    ("weak_strong", "Weak--strong", 7),
    ("intermediate_strong", "Intermediate--strong", 7),
    ("strong_strong_u8", "Strong--strong", 7),
]

# Okabe--Ito colors; series also differ by linestyle and marker shape.
BASE_COLOR = "#0072B2"      # dashed, no markers
RA_APPEND_COLOR = "#009E73"  # solid, sparse solid circles
RA_COLOR = "#8B0000"         # blood red; solid, sparse solid diamonds
ERR_FLOOR = 1e-16
APPEND_ARM = "forced_append_ra"
INSERTION_ARM = "forced_always_open_position_phase0"
_WEAK_ROW_SCALE = ((6e-17, 4e0), [1e0, 1e-4, 1e-8, 1e-12, 1e-16])
PANEL_SCALES = {
    "weak_weak": _WEAK_ROW_SCALE,
    "intermediate_weak": _WEAK_ROW_SCALE,
    "strong_weak_u8": _WEAK_ROW_SCALE,
    "weak_strong": ((3e-5, 4e0), [1e0, 1e-2, 1e-4]),
    "intermediate_strong": ((3e-6, 4e0), [1e0, 1e-2, 1e-4]),
    "strong_strong_u8": ((3e-10, 4e0), [1e0, 1e-3, 1e-6, 1e-9]),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_cell_summary(cell_dir: Path) -> dict | None:
    """Trajectory from summary.json only (never the huge result.json)."""
    path = cell_dir / "run/summary/summary.json"
    if not path.is_file():
        return None
    summary = json.loads(path.read_text())
    trace = summary.get("accepted_error_trace") or []
    if not trace:
        return None
    return {
        "cell_name": cell_dir.name,
        "summary_path": path,
        "points": [
            (int(row["controller_round"]),
             max(float(row["absolute_energy_error"]), ERR_FLOOR))
            for row in trace
        ],
        "live": False,
    }


def _err_at(points: list[tuple[int, float]], k: int) -> float | None:
    return dict(points).get(k)


def main() -> int:
    spec = importlib.util.spec_from_file_location("cmp_builder", CMP_PATH)
    cmp_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmp_mod)

    hist_plateau = cmp_mod.load_hist_plateau()
    exact = {r: v["exact_energy"] for r, v in hist_plateau.items()}
    hist_append = cmp_mod.load_hist_append(exact, WORKDIR)

    cells: dict[tuple[str, str], dict] = {}
    for arm in (APPEND_ARM, INSERTION_ARM):
        for regime, *_ in REGIMES:
            entry = _load_cell_summary(
                CELLS / f"x_allphase_maxk50__{arm}__{regime}"
            )
            if entry is not None:
                cells[(arm, regime)] = entry
    # Live partial trajectories fill insertion holes only; completed wins.
    for key, entry in cmp_mod.load_live_cells(LIVE_TAILS, exact).items():
        if key[0] == INSERTION_ARM and key not in cells:
            entry = dict(entry)
            # The AI_LOG line at depth d records the energy ENTERING round d
            # (verified: tail k=2 equals summary k=1 on a completed cell), so
            # live points are shifted one round left and the pre-round-1
            # entry is dropped.
            entry["points"] = [
                (k - 1, e) for k, e in entry["points"] if k >= 2
            ]
            if not entry["points"]:
                continue
            entry["live"] = True
            cells[key] = entry

    plt.rcParams.update({"font.family": "serif", "font.size": 8.5})
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 5.15), sharex=True)
    stats: dict[str, dict] = {}
    for index, (ax, (regime, title, nph)) in enumerate(zip(axes.flat, REGIMES)):
        base = hist_append[regime]
        cell = cells[(APPEND_ARM, regime)]
        ins = cells.get((INSERTION_ARM, regime))

        # k*: baseline-plateau onset, exactly the comparison builder's rule.
        target = base["err_final"] * (10 ** 0.1)
        kstar = next(
            (kk for kk, err in base["points"] if err <= target),
            base["k_final"],
        )

        bk = [k for k, _ in base["points"]]
        be = [e for _, e in base["points"]]
        rk = [k for k, _ in cell["points"]]
        re_ = [e for _, e in cell["points"]]
        ax.plot(bk, be, color=BASE_COLOR, lw=1.3, ls="--")
        ax.plot(
            rk, re_, color=RA_APPEND_COLOR, lw=1.6,
            marker="o", markersize=4.0, markevery=[0, len(rk) - 1],
            markerfacecolor=RA_APPEND_COLOR, markeredgewidth=0.9,
        )
        ins_stats = None
        if ins is not None:
            ik = [k for k, _ in ins["points"]]
            ie = [e for _, e in ins["points"]]
            ax.plot(
                ik, ie, color=RA_COLOR, lw=1.6,
                marker="D", markersize=4.2, markevery=[0, len(ik) - 1],
                markerfacecolor=RA_COLOR, markeredgewidth=0.9,
            )
            ins_stats = {
                "latest_k": ik[-1],
                "latest_error": ie[-1],
                "error_at_kstar": _err_at(ins["points"], kstar),
                "live_partial": bool(ins.get("live")),
                "cell_name": ins.get("cell_name"),
            }
        ax.set_yscale("log")
        ax.set_xlim(0, 50)
        # Ranges anchored at 10^0; the weak-Holstein row shares its floor,
        # while each strong-Holstein panel floors at its own data scale.
        ylims, yticks = PANEL_SCALES[regime]
        ax.set_ylim(*ylims)
        ax.set_yticks(yticks)
        ax.grid(True, which="major", alpha=0.25, lw=0.55)
        ax.set_title(rf"{title} ($n_{{\rm ph}}={nph}$)", fontsize=9.4)
        ax.set_xlabel("ADAPT iteration")
        stats[regime] = {
            "kstar_baseline_plateau": kstar,
            "baseline_error_at_kstar": _err_at(base["points"], kstar),
            "baseline_terminal_k": bk[-1],
            "baseline_terminal_error": be[-1],
            "ra_append_error_at_kstar": _err_at(cell["points"], kstar),
            "ra_append_terminal_k": rk[-1],
            "ra_append_terminal_error": re_[-1],
            "cell_name": cell["cell_name"],
            "insertion": ins_stats,
        }
    axes[0, 0].set_ylabel(r"same-cutoff $|\Delta E|$")
    axes[1, 0].set_ylabel(r"same-cutoff $|\Delta E|$")
    legend = [
        Line2D(
            [0], [0], color=RA_COLOR, lw=1.6, marker="D", markersize=4.2,
            markerfacecolor=RA_COLOR, label="RA",
        ),
        Line2D(
            [0], [0], color=RA_APPEND_COLOR, lw=1.6, marker="o", markersize=4,
            markerfacecolor=RA_APPEND_COLOR, label="RA, append only",
        ),
        Line2D([0], [0], color=BASE_COLOR, lw=1.3, ls="--",
               label="Append-ADAPT VQE"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=3, frameon=False,
               fontsize=7.2)
    fig.subplots_adjust(
        left=0.065, right=0.995, bottom=0.085, top=0.855, hspace=0.39,
        wspace=0.24,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    plt.close(fig)

    summary_bindings = {}
    for regime, row in stats.items():
        path = CELLS / row["cell_name"] / "run/summary/summary.json"
        summary_bindings[f"{APPEND_ARM}__{regime}"] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256(path),
        }
        ins = row["insertion"]
        if ins and not ins["live_partial"] and ins.get("cell_name"):
            ipath = CELLS / ins["cell_name"] / "run/summary/summary.json"
            summary_bindings[f"{INSERTION_ARM}__{regime}"] = {
                "path": str(ipath.relative_to(ROOT)),
                "sha256": _sha256(ipath),
            }
    record = {
        "schema": "paper_i_ra_adaptive_append_main_v7",
        "evidence_class": "preliminary_partial_campaign_author_authorized_20260818",
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": _sha256(OUTPUT),
        "display": {
            "series": [
                "published_append_tolmatch_r50 (blue dashed); series per comparison page 1 (forced-k50 arms, exact horizon 50)",
                "forced_append_ra (green solid, solid circles)",
                f"{INSERTION_ARM} (blood-red solid, solid diamonds; "
                "completed and live partial cells)",
            ],
            "excluded_arms": [
                "plateau/forced/floors arms (unreported diagnostics)",
            ],
            "kstar_rule": (
                "baseline plateau onset: first baseline round with error at or "
                "below err_final * 10^0.1 (comparison-builder rule); reported "
                "in prose, not marked on panels"
            ),
            "markers": "series-distinguishing; drawn at first and final iteration only", "y_axis": "anchored at 1e0; weak row shared to 1e-16, strong panels floored at own data scale (ws 3e-5, is 3e-6, ss 3e-10)",
        },
        "per_regime": stats,
        "sources": {
            "cells_dir": str(CELLS.relative_to(ROOT)),
            "cell_summaries": summary_bindings,
            "live_tails_dir": str(LIVE_TAILS.relative_to(ROOT)),
            "published_baseline_provenance": {
                "path": str(cmp_mod.PAPER_PACKAGE_PROVENANCE),
                "sha256": _sha256(cmp_mod.PAPER_PACKAGE_PROVENANCE),
            },
            "loader_module": {
                "path": str(CMP_PATH.relative_to(ROOT)),
                "sha256": _sha256(CMP_PATH),
            },
        },
    }
    PROVENANCE.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    def _fmt(value):
        return f"{value:.2e}" if isinstance(value, float) else str(value)

    for regime, row in stats.items():
        ins = row["insertion"]
        print(
            f"{regime:20s} k*={row['kstar_baseline_plateau']:>2d} "
            f"base@k*={_fmt(row['baseline_error_at_kstar'])} "
            f"ra@k*={_fmt(row['ra_append_error_at_kstar'])} "
            f"| ra term k={row['ra_append_terminal_k']} "
            f"{_fmt(row['ra_append_terminal_error'])} "
            + (
                f"| ins@k*={_fmt(ins['error_at_kstar'])} "
                f"latest k={ins['latest_k']} {_fmt(ins['latest_error'])} "
                f"live={ins['live_partial']}"
                if ins else "| ins absent"
            )
        )
    print("wrote", OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
