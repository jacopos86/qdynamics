"""Corrected exchange-face campaign against the current manuscript values.

One standalone PDF: the Table I columns of the manuscript for each cell as
three rows (APM as reported, APM corrected, AVQDS comparator), the
generator-level structural-action table for both APM trajectories with the
count of committed exchange patches, and per-cell observable panels (total
energy, doublon occupation, charge imbalance) with their absolute errors
against the exact reference.

"Reported" values come from the locked selector comparison the manuscript
tables were generated from; "corrected" values come from a comparison built
by the same tool over the corrected campaign's archives.  Both are read, not
recomputed here.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.diagnostics.q_compiled_manuscript_assets import (
    CONTROLLER_ORDER,
    CONTROLLER_TEX,
    DRIVE_ORDER,
    DRIVE_TEX,
    ERROR_UNIT,
    STEP_UNIT,
    WORK_UNIT,
    _cost_record,
    _generator_motion,
    _load_json,
    _recovered_estimator_total,
    _scaled,
    _trajectory_rows,
)

DRIVE_NAME = {
    "weakslow": "weak--slow",
    "fastweak": "weak--fast",
    "fastfast": "weak--fastest",
    "slowstrong": "strong--slow",
    "midres": "strong--mid",
    "strongfast": "strong--fast",
}


def _work_total(route: Mapping[str, Any]) -> tuple[int | None, bool]:
    work = route["estimator_work"]
    if work.get("logical_total_estimator_work_status") == "exact":
        return int(work["logical_total_estimator_work"]), True
    recovered = _recovered_estimator_total(route)
    return recovered, False


def _manuscript_row(table_tex: Path, drive: str, controller: str, method: str) -> list[str] | None:
    """The APM or AVQDS row printed in the manuscript's Table I for a cell (already scaled)."""

    lines = table_tex.read_text(encoding="utf-8").splitlines()
    head = f"{DRIVE_TEX[drive]} & {CONTROLLER_TEX[controller]} & APM"
    for index, line in enumerate(lines):
        if line.startswith(head):
            if method == "APM":
                cells = [c.strip() for c in line.rstrip("\\ ").split("&")]
                return cells[3:12] if len(cells) >= 12 else None
            if index + 1 < len(lines):
                cells = [c.strip() for c in lines[index + 1].rstrip("\\ ").split("&")]
                if len(cells) >= 12 and cells[2] == "AVQDS":
                    return cells[3:12]
    return None


def _fallback_avqds_row(table_tex: Path, drive: str, controller: str) -> list[str] | None:
    return _manuscript_row(table_tex, drive, controller, "AVQDS")


def _manuscript_motion_row(table_tex: Path, drive: str, controller: str) -> list[str] | None:
    """The row of the manuscript's Table II for a cell: E, ins., del., exch., del-then-re-ins., ins-then-del., tested/failed."""

    head = f"{DRIVE_TEX[drive]} & {CONTROLLER_TEX[controller]} & "
    for line in table_tex.read_text(encoding="utf-8").splitlines():
        if line.startswith(head) and "APM" not in line and "pending" not in line:
            cells = [c.strip() for c in line.rstrip("\\ ").split("&")]
            if len(cells) == 9:
                return cells[2:9]
    return None


def _accuracy_cells(route: Mapping[str, Any]) -> list[str]:
    acc = route["accuracy"]
    return [
        _scaled(acc["mean_abs_energy_error"], ERROR_UNIT),
        _scaled(acc["mean_abs_doublon_error"], ERROR_UNIT),
        _scaled(acc["mean_abs_charge_imbalance_error"], ERROR_UNIT),
    ]


def _exchange_commits(route: Mapping[str, Any]) -> int:
    counts = (route.get("support_motion") or {}).get("decision_reason_counts") or {}
    return int(counts.get("accepted_generalized_exchange_exchange", 0))


def _match_text(pair: Mapping[str, Any] | None) -> str:
    if not pair or pair.get("worst_component_ratio") is None:
        return "--"
    verdict = "yes" if pair.get("passes") else "no"
    return f"{verdict} ({float(pair['worst_component_ratio']):.2f})"


def build(
    *,
    locked: Mapping[str, Any],
    corrected: Mapping[str, Any],
    locked_cost_root: Path,
    corrected_cost_root: Path,
    manuscript_table: Path,
    output_dir: Path,
    extra_sections: Sequence[str] = (),
    matched_column: bool = True,
    intro_extra: str = "",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    locked_cells = {c["cell_id"]: c for c in locked["cells"]}
    rows: list[dict[str, Any]] = []
    for raw in corrected["cells"]:
        q_new = raw["routes"].get("q_compiled")
        if q_new is None:
            continue
        cell_id = str(raw["cell_id"])
        old = locked_cells.get(cell_id, {})
        q_old = old.get("routes", {}).get("q_compiled")
        av = old.get("routes", {}).get("avqds") or raw["routes"].get("avqds")
        rows.append(
            {
                "cell_id": cell_id,
                "drive": str(raw["drive"]),
                "controller": str(raw["controller"]),
                "q_new": q_new,
                "q_old": q_old,
                "avqds": av,
                "pair_new": raw["pairwise_accuracy"].get("q_compiled__avqds"),
                "pair_old": old.get("pairwise_accuracy", {}).get("q_compiled__avqds"),
                "reference": raw["reference"],
            }
        )
    rows.sort(key=lambda r: (DRIVE_ORDER[r["drive"]], CONTROLLER_ORDER[r["controller"]]))

    # ---- Table I -------------------------------------------------------
    t1 = [
        r"\begin{table}[H]\centering\scriptsize\setlength{\tabcolsep}{3pt}",
        r"\caption{Manuscript Table~I columns for every corrected cell.  \emph{reported}: the "
        r"AP-McLachlan trajectory in the current manuscript; \emph{corrected}: the same cell "
        r"re-run with the declared exchange insertion frontier ($W=8$), which lets the search "
        r"enumerate true-exchange patches; AVQDS: the locked comparator.  $\overline{|\Delta E|}$, "
        r"$\overline{|\Delta D|}$, $\overline{|\Delta n|}$ in $10^{-4}$ over all 501 reporting "
        r"times; $\mathcal E$ in $10^{8}$; $N_{\rm step}$ in $10^{3}$; \emph{exch.} counts "
        r"committed patches that both delete and insert"
        + (r"; \emph{matched} is the declared matched-accuracy rule against AVQDS with the worst "
           r"componentwise factor in parentheses.}" if matched_column else ".}"),
        r"\begin{tabular}{lllrrrrrrrrrr" + ("l" if matched_column else "") + r"}\toprule",
        r"$(A,\omega)$ & controller & trajectory & $\overline{|\Delta E|}$ & $\overline{|\Delta D|}$ & "
        r"$\overline{|\Delta n|}$ & $N_\theta$ & $N_{2q}$ & $D_{2q}$ & $D_c$ & $\mathcal E$ & "
        r"$N_{\rm step}$ & exch." + (r" & matched" if matched_column else "") + r" \\ \midrule",
    ]
    t2 = [
        r"\begin{table}[H]\centering\scriptsize\setlength{\tabcolsep}{3.5pt}",
        r"\caption{Structural actions at generator granularity for the reported and corrected "
        r"AP-McLachlan trajectories (manuscript Table~II).  \emph{Inserted} and \emph{deleted} "
        r"count structural events; \emph{exch.} counts time points whose committed patch both "
        r"deletes and inserts; \emph{del.\ then re-ins.} a generator deleted and later "
        r"re-inserted, \emph{ins.\ then del.} a generator inserted and later deleted; "
        r"\emph{finalists tested/failed} are deletion-bearing finalists materialized and sent to "
        r"the hard gates, over those failing a gate after materialization; \emph{exchange commits} "
        r"counts accepted exchange patches over all selector rounds (several may commit inside one "
        r"time point).}",
        r"\begin{tabular}{lllrrrrrrcr}\toprule",
        r"$(A,\omega)$ & controller & trajectory & $\mathcal E$ & ins. & del. & exch. & "
        r"\shortstack{del.\ then\\re-ins.} & \shortstack{ins.\ then\\del.} & "
        r"\shortstack{finalists\\tested/failed} & \shortstack{exchange\\commits} \\ \midrule",
    ]
    manifest_rows = []
    for r in rows:
        drive, controller = r["drive"], r["controller"]
        head = [DRIVE_TEX[drive], CONTROLLER_TEX[controller]]
        for label, route, root, pair in (
            ("APM reported", r["q_old"], locked_cost_root, r["pair_old"]),
            ("APM corrected", r["q_new"], corrected_cost_root, r["pair_new"]),
            ("AVQDS", r["avqds"], locked_cost_root, None),
        ):
            first = label == "APM reported"
            lead = head if first else ["", ""]
            if route is None:
                fallback = _manuscript_row(manuscript_table, drive, controller, "AVQDS" if label == "AVQDS" else "APM")
                if fallback is None:
                    t1.append(" & ".join(lead + [label] + ["--"] * (11 if matched_column else 10)) + r" \\")
                    continue
                tail = (["0"] if label != "AVQDS" else ["--"]) + (["--"] if matched_column else [])
                t1.append(" & ".join(lead + [label] + fallback + tail) + r" \\")
                if label == "APM reported":
                    motion = _manuscript_motion_row(manuscript_table, drive, controller)
                    if motion is not None:
                        t2.append(" & ".join(head + ["reported"] + motion + ["--"]) + r" \\")
                continue
            cost, _ = _cost_record(root, r["cell_id"], "avqds" if label == "AVQDS" else "q_compiled")
            total, exact = _work_total(route)
            work_text = "--" if total is None else (("" if exact else r"$\geq$") + _scaled(total, WORK_UNIT))
            exch = "--" if label == "AVQDS" else str(_generator_motion(route)["exchanges"])
            t1.append(
                " & ".join(
                    lead
                    + [label]
                    + _accuracy_cells(route)
                    + [str(int(route["n_theta"])), str(cost["N2q"]), str(cost["D2q"]), str(cost["Dc"]),
                       work_text, _scaled(int(route["time_step_count"]), STEP_UNIT), exch]
                    + ([("--" if label == "AVQDS" else _match_text(pair))] if matched_column else [])
                )
                + r" \\"
            )
            if label != "AVQDS":
                motion = _generator_motion(route)
                gates = route["deletion_gate_outcomes"]
                t2.append(
                    " & ".join(
                        (head if label == "APM reported" else ["", ""])
                        + [label.split()[1], work_text, str(motion["inserted"]), str(motion["deleted"]),
                           str(motion["exchanges"]), str(motion["reinserted"]), str(motion["retired"]),
                           f"{int(gates['deletion_bearing_finalist_attempt_count'])}/{int(gates['post_delete_gate_failure_count'])}",
                           str(_exchange_commits(route))]
                    )
                    + r" \\"
                )
        t1.append(r"\addlinespace[2pt]")
        t2.append(r"\addlinespace[2pt]")
        manifest_rows.append({"cell_id": r["cell_id"], "corrected_sources": r["q_new"]["run_sources"],
                              "reported_sources": None if r["q_old"] is None else r["q_old"]["run_sources"]})
    t1 += [r"\bottomrule\end{tabular}\end{table}"]
    t2 += [r"\bottomrule\end{tabular}\end{table}"]

    # ---- figures ---------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_tex = []
    for r in rows:
        manifest_path = Path(r["reference"]["manifest_path"])
        manifest = _load_json(manifest_path)
        with np.load(manifest_path.parent / str(manifest["data_file"])) as ref:
            t_ref = np.asarray(ref["time"], dtype=float)
            exact = (
                np.asarray(ref["energy"], dtype=float),
                np.asarray(ref["doublon"], dtype=float),
                np.asarray(ref["site_occupations"], dtype=float)[:, 0]
                - np.asarray(ref["site_occupations"], dtype=float)[:, 1],
            )

        def series(route):
            rows_ = _trajectory_rows([Path(s["path"]) for s in route["run_sources"]])
            t = np.asarray([x["time"] for x in rows_], dtype=float)
            return t, (
                np.asarray([x["energy_expectation"] for x in rows_], dtype=float),
                np.asarray([x["doublon"] for x in rows_], dtype=float),
                np.asarray([x["site_occupations"][0] - x["site_occupations"][1] for x in rows_], dtype=float),
            )

        curves = [("APM corrected", r["q_new"], "#8b1a1a", "-", 1.3)]
        if r["q_old"] is not None:
            curves.append(("APM reported", r["q_old"], "#d08a00", "-.", 1.0))
        if r["avqds"] is not None:
            curves.append(("AVQDS", r["avqds"], "#1f5aa6", "--", 1.0))
        plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.7})
        fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.0), sharex=True)
        ylabels = (r"$E(t)$", r"$D(t)$", r"$\Delta n(t)$")
        for j in range(3):
            axes[0, j].plot(t_ref, exact[j], color="black", linestyle=":", linewidth=1.0, label="exact")
        for label, route, color, style, width in curves:
            t, vals = series(route)
            if not np.allclose(t, t_ref, rtol=0.0, atol=1e-12):
                raise ValueError(f"{r['cell_id']}: {label} is not on the reporting grid")
            for j in range(3):
                axes[0, j].plot(t, vals[j], color=color, linestyle=style, linewidth=width, label=label)
                axes[1, j].semilogy(t, np.abs(vals[j] - exact[j]) + 1e-16, color=color, linestyle=style, linewidth=width)
        for j in range(3):
            axes[0, j].set_ylabel(ylabels[j]); axes[1, j].set_ylabel("|error|"); axes[1, j].set_xlabel(r"$t$")
            for a in axes[:, j]:
                a.grid(alpha=0.18, linewidth=0.5)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=False, ncol=4)
        fig.tight_layout(rect=(0, 0, 1, 0.93), pad=0.5, w_pad=0.9)
        fname = f"{r['cell_id']}.pdf"
        fig.savefig(output_dir / "figures" / fname, bbox_inches="tight")
        plt.close(fig)
        figure_tex.append(
            r"\begin{figure}[H]\centering\includegraphics[width=\linewidth]{figures/%s}" % fname
            + r"\caption{%s $%s$, %s: total energy, doublon occupation and charge imbalance "
            r"(top) and absolute errors against the exact reference (bottom).}\end{figure}"
            % (DRIVE_NAME[r["drive"]], DRIVE_TEX[r["drive"]].strip("$"), CONTROLLER_TEX[r["controller"]])
        )

    generated = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d %H:%MZ")
    n_cells = len(rows)
    tex = "\n".join(
        [
            r"\documentclass[10pt]{article}",
            r"\usepackage[margin=1.4cm,landscape]{geometry}\usepackage{booktabs,graphicx,float,amsmath,amssymb}",
            r"\usepackage[T1]{fontenc}\setlength{\parskip}{4pt}",
            r"\begin{document}",
            r"\section*{Corrected exchange-face campaign against the current manuscript values}",
            f"Generated {generated}.  {n_cells} of 11 corrected cells returned so far; the remaining cells "
            r"(weak--slow $\delta\theta$, weak--slow state motion, strong--slow state motion) are appended "
            r"when they land.  \emph{Corrected} trajectories run the exchange-face search with the declared "
            r"insertion frontier ($W=8$, profile \texttt{paper\_declared\_exchange\_v1}), which admits every "
            r"deletion-cardinality rung the joint-work cap had refused whole; every other setting is the "
            r"reported cell's.  Under the reported permission bound, a deletion batch is admitted only when "
            r"$\sin[\min(\pi/2,\sum_{\mu\in D}|\theta_\mu c_\mu|)]\le2\times10^{-3}$, so every committed "
            r"exchange below removes a coordinate whose accumulated effective angle was at most "
            r"$2\times10^{-3}$ at the time point.",
            intro_extra,
            *t1,
            *t2,
            *extra_sections,
            r"\clearpage",
            r"\section*{Observables of the corrected cells}",
            *figure_tex,
            r"\end{document}",
            "",
        ]
    )
    (output_dir / "paper_ii_corrected_vs_reported.tex").write_text(tex, encoding="utf-8")
    manifest = {
        "generated_utc": generated,
        "cells": manifest_rows,
        "locked_comparison": str(locked.get("generated_utc")),
        "corrected_comparison": str(corrected.get("generated_utc")),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--locked-comparison", type=Path, required=True)
    parser.add_argument("--corrected-comparison", type=Path, required=True)
    parser.add_argument("--locked-cost-root", type=Path, required=True)
    parser.add_argument("--corrected-cost-root", type=Path, required=True)
    parser.add_argument("--manuscript-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args(argv)
    build(
        locked=_load_json(args.locked_comparison),
        corrected=_load_json(args.corrected_comparison),
        locked_cost_root=args.locked_cost_root,
        corrected_cost_root=args.corrected_cost_root,
        manuscript_table=args.manuscript_table,
        output_dir=args.output_dir,
    )
    if args.compile:
        subprocess.run(["tectonic", "paper_ii_corrected_vs_reported.tex"], cwd=args.output_dir, check=True)
        print(args.output_dir / "paper_ii_corrected_vs_reported.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
