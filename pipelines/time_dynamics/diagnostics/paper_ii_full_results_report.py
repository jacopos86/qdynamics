"""All Paper II campaign results in one PDF, with observables and no pass/fail gate.

Sections: corrected campaign against the reported values (Table I and Table II
columns, no matched-accuracy verdict), the pre-permission threshold pilot, the
activation-threshold rungs for the cells that left the factor-two rule, the
SIAM Nb=4 ladder (both arms), and the per-cell observables.  Every number is a
measured value or a declared parameter; the manuscript's matched-accuracy rule
is not applied.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.diagnostics.exact_reference_store import ExactReferenceStore
from pipelines.time_dynamics.diagnostics.paper_ii_corrected_campaign_report import (
    DRIVE_NAME,
    build,
    hh_hamiltonian_text,
)
from pipelines.time_dynamics.diagnostics.physical_errors import physical_errors
from pipelines.time_dynamics.diagnostics.q_compiled_manuscript_assets import (
    CONTROLLER_TEX,
    DRIVE_TEX,
    ERROR_UNIT,
    STEP_UNIT,
    WORK_UNIT,
    _load_json,
    _scaled,
)

SEED = "chtc/paper_ii_production_v2/input/seeds/hh_snake_nph1.json"


def _run_record(run_path: Path, store: ExactReferenceStore) -> dict[str, Any]:
    """Errors, support, terminal cost, estimator count, steps and series of one HH run."""

    doc = _load_json(run_path)
    rows = doc["plot_rows"]
    summary = doc["summary"]
    ref = store.get_or_build(seed_path=SEED, run_path=run_path, times=[r["time"] for r in rows])
    metrics = physical_errors(rows, ref.rows)
    cost = _load_json(run_path.with_name("terminal_qiskit_cost.json"))["rows"][0]
    work = _load_json(run_path.with_name("exact_estimator_work.json"))["estimator_work"]
    t = np.asarray([r["time"] for r in rows], dtype=float)
    series = (
        np.asarray([r["energy_expectation"] for r in rows], dtype=float),
        np.asarray([r["doublon"] for r in rows], dtype=float),
        np.asarray([r["site_occupations"][0] - r["site_occupations"][1] for r in rows], dtype=float),
    )
    exact = (
        np.asarray([r["energy"] for r in ref.rows], dtype=float),
        np.asarray([r["doublon"] for r in ref.rows], dtype=float),
        np.asarray([r["site_occupations"][0] - r["site_occupations"][1] for r in ref.rows], dtype=float),
    )
    config = summary["support_patch_config"]
    return {
        "dE": metrics.mean_abs_energy_error,
        "dD": metrics.mean_abs_doublon_error,
        "dn": 2.0 * metrics.mean_max_abs_site_occupation_error,
        "n_theta": int(summary["active_parameter_count_final"]),
        "N2q": int(cost["N2q"]), "D2q": int(cost["D2q"]), "Dc": int(cost["Dc"]),
        "work": int(work["logical_total_estimator_work"]),
        "work_exact": work.get("logical_total_estimator_work_status") == "exact",
        "steps": int(summary["attempted_internal_step_count"]),
        "exchanges": int(summary.get("accepted_exchange_count") or 0),
        "tau_pre": config.get("exchange_permission_ray_distance_max"),
        "tau_m": float(config["insertion_l2_cut"]),
        "permitted": summary.get("deletion_permission_permitted_set_count"),
        "evaluated": summary.get("deletion_permission_evaluated_set_count"),
        "t": t, "series": series, "exact": exact,
        "drive": doc["hamiltonian"].get("drive_profile") or {},
    }


def _row(label: str, r: Mapping[str, Any], lead: Sequence[str] = ()) -> str:
    work = ("" if r["work_exact"] else r"$\geq$") + _scaled(r["work"], WORK_UNIT)
    return " & ".join(
        list(lead) + [label, _scaled(r["dE"], ERROR_UNIT), _scaled(r["dD"], ERROR_UNIT), _scaled(r["dn"], ERROR_UNIT),
                      str(r["n_theta"]), str(r["N2q"]), str(r["D2q"]), str(r["Dc"]), work, _scaled(r["steps"], STEP_UNIT), str(r["exchanges"])]
    ) + r" \\"


HEAD = (r"$\overline{|\Delta E|}$ & $\overline{|\Delta D|}$ & $\overline{|\Delta n|}$ & $N_\theta$ & $N_{2q}$ & "
        r"$D_{2q}$ & $D_c$ & $\mathcal E$ & $N_{\rm step}$ & exch.")


def _multi_figure(runs: Sequence[tuple[str, Mapping[str, Any]]], output: Path, title: str,
                  avqds: Mapping[str, Any] | None = None) -> None:
    """Three observables: exact reference, the corrected AP-McLachlan variants, the AVQDS comparator."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.7})
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.4), sharex=True)
    t_ref = runs[0][1]["t"]; exact = runs[0][1]["exact"]
    for j in range(3):
        axes[j].plot(t_ref, exact[j], color="black", linestyle=":", linewidth=1.1, label="exact")
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(runs)))
    for (label, r), color in zip(runs, colors):
        for j in range(3):
            axes[j].plot(r["t"], r["series"][j], color=color, linewidth=0.9, label=label)
    if avqds is not None:
        for j in range(3):
            axes[j].plot(avqds["t"], avqds["series"][j], color="#1f5aa6", linestyle="--", linewidth=1.0, label="AVQDS")
    for j, yl in enumerate((r"$E(t)$", r"$D(t)$", r"$\Delta n(t)$")):
        axes[j].set_ylabel(yl); axes[j].set_xlabel(r"$t$"); axes[j].grid(alpha=0.18, linewidth=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False, ncol=min(4, len(labels)), fontsize=7)
    fig.suptitle(title, y=0.93, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.86), pad=0.5, w_pad=0.9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def pilot_section(control: Path, pilots: Sequence[Path], store: ExactReferenceStore, figdir: Path) -> str:
    runs = [("$2\\times10^{-3}$ (manuscript bound)", _run_record(control, store))]
    for p in pilots:
        r = _run_record(p, store)
        runs.append((f"${r['tau_pre']:g}$".replace("e-0", r"\times10^{-").replace("e-", r"\times10^{-") + ("}$" if "10^{" in f"{r['tau_pre']:g}".replace("e-0", r"\times10^{-") else ""), r))
    # clean labels: plain floats -> 5\times10^{-3}
    labels = []
    for label, r in runs:
        if r["tau_pre"] is None:
            labels.append(label)
        else:
            m, e = f"{r['tau_pre']:.0e}".split("e")
            labels.append(rf"${int(m)}\times10^{{{int(e)}}}$")
    runs = list(zip(labels, [r for _, r in runs]))
    runs.sort(key=lambda x: (x[1]["tau_pre"] is not None, x[1]["tau_pre"] or 0.0))
    lines = [
        r"\section*{Pre-permission threshold pilot}",
        r"\textbf{Hamiltonian.} " + hh_hamiltonian_text(drive=runs[0][1]["drive"]) + ".",
        r"Strong--fast $(2.4,3)$ $\delta\theta$, $t=0\to20$.  The manuscript admits a deletion batch $D$ only when its ray bound "
        r"$\sin[\min(\pi/2,\sum_{\mu\in D}|\theta_\mu c_\mu|)]$ lies below the post-refit tolerance $2\times10^{-3}$; each pilot "
        r"run tests that bound at $\tau_{\rm pre}$ instead, leaving the post-refit ray and velocity certificates unchanged.  "
        r"Columns as in the corrected-campaign table; \emph{permitted} is the number of deletion sets the permission test admitted "
        r"over the number it evaluated.",
        r"\begin{table}[H]\centering\scriptsize\setlength{\tabcolsep}{3.5pt}\begin{tabular}{lrrrrrrrrrrl}\toprule",
        r"$\tau_{\rm pre}$ & " + HEAD + r" & permitted / evaluated \\ \midrule",
    ]
    for label, r in runs:
        lines.append(_row(label, r)[:-3] + f" & {r['permitted']:,} / {r['evaluated']:,}" + r" \\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    _multi_figure(runs, figdir / "pilot_strongfast_dtheta.pdf", "pre-permission threshold pilot, strong--fast delta-theta")
    lines += [r"\begin{figure}[H]\centering\includegraphics[width=\linewidth]{figures/pilot_strongfast_dtheta.pdf}"
              r"\caption{Total energy, doublon occupation and charge imbalance of the pilot trajectories against the exact reference "
              r"(the strong--fast AVQDS comparator has no fresh trajectory yet).  " + hh_hamiltonian_text(drive=runs[0][1]["drive"]) + r".}\end{figure}"]
    return "\n".join(lines)


def rung_section(cells: Mapping[str, Sequence[tuple[str, Path]]], store: ExactReferenceStore, figdir: Path,
                 avqds: Mapping[str, Path]) -> str:
    lines = [
        r"\section*{Activation-threshold rungs for the two corrected cells outside the factor-two rule}",
        r"\textbf{Hamiltonian.} " + hh_hamiltonian_text() + r"; the drive of each cell is stated in its caption.",
        r"The activation threshold $\tau_m$ of Eq.~(realized\_mclachlan\_distance) is stepped on the factor-three ladder the "
        r"reported cells were chosen on; every run keeps the declared insertion frontier and the manuscript's permission bound.  "
        r"The AVQDS comparator row is the locked one.",
    ]
    for cell_id, entries in cells.items():
        drive, controller = cell_id.split("__", 1)
        lines += [
            rf"\subsection*{{{DRIVE_NAME[drive]} {DRIVE_TEX[drive]}, {CONTROLLER_TEX[controller]}}}",
            r"\begin{table}[H]\centering\scriptsize\setlength{\tabcolsep}{3.5pt}\begin{tabular}{lrrrrrrrrrr}\toprule",
            r"$\tau_m$ & " + HEAD + r" \\ \midrule",
        ]
        runs = []
        for label, path in entries:
            r = _run_record(path, store)
            m, e = f"{r['tau_m']:.2e}".split("e")
            tex = rf"${float(m):g}\times10^{{{int(e)}}}$" + (f" ({label})" if label else "")
            lines.append(_row(tex, r))
            runs.append((tex, r))
        a = None
        if cell_id in avqds:
            a = _run_record(avqds[cell_id], store)
            lines.append(_row("AVQDS (locked)", a))
        lines += [r"\bottomrule\end{tabular}\end{table}"]
        fname = f"rungs_{cell_id}.pdf"
        _multi_figure(runs, figdir / fname, f"{DRIVE_NAME[drive]} {controller.replace('_', ' ')}: activation-threshold rungs", avqds=a)
        lines += [rf"\begin{{figure}}[H]\centering\includegraphics[width=\linewidth]{{figures/{fname}}}"
                  r"\caption{Total energy, doublon occupation and charge imbalance of the rung trajectories, the exact reference and the "
                  r"AVQDS comparator.  " + hh_hamiltonian_text(drive=runs[0][1]["drive"]) + r".}\end{figure}"]
    return "\n".join(lines)


def siam_section(score_files: Sequence[Path], figdir: Path) -> str:
    rows = []
    for f in score_files:
        s = _load_json(f)
        m = re.search(r"siam_nb4_qc_(avqds|full_apm)_(screen|continue)_r(\d+)_cut([0-9a-z]+)_source_([0-9a-f]{8})", str(f))
        if not m:
            continue
        arm, seg, rung, cut, rev = m.groups()
        run = _load_json(f.with_name("run.json"))["summary"]
        rows.append({
            "arm": "AVQDS" if arm == "avqds" else "APM (exchange face)",
            "segment": "$t\\in[0,1]$" if seg == "screen" else "$t\\in[1,5]$",
            "rung": int(rung), "cut": cut.replace("em", r"\times10^{-").replace("1em", "") ,
            "burden": s["summary"]["aggregate_accuracy_burden"],
            "components": s["summary"].get("normalized_accuracy_components", {}),
            "infidelity": s["summary"].get("mean_infidelity"),
            "n_theta": run.get("active_parameter_count_final"),
            "exchanges": run.get("accepted_exchange_count"),
            "classification": s["summary"]["classification"],
            "series": s.get("series") or s.get("trajectory") or {},
            "path": f,
        })
    rows.sort(key=lambda r: (r["arm"], r["rung"]))
    comp_names = sorted({k for r in rows for k in r["components"]})
    phys = _load_json(Path("chtc/siam_nb4_qc_full_apm_screen_r00_cut1em02_source_812d20c3_20260905/input/seed.json"))["physical"]
    siam_text = (
        rf"single-impurity Anderson model in a finite box, $N_b={int(phys['n_bath'])}$ bath sites ({int(phys['n_qubits'])} qubits), "
        rf"hopping $t={float(phys['hopping']):g}$, $U={float(phys['u']):g}$, hybridization quench $V: {float(phys['v_pre_quench']):g}\to{float(phys['v_post_quench']):.4g}$ at $t=0$, "
        rf"$N_\uparrow={int(phys['electron_counts']['up'])}$, $N_\downarrow={int(phys['electron_counts']['down'])}$"
    )
    lines = [
        r"\section*{SIAM $N_b=4$ (10 qubits) activation-threshold ladder}",
        r"\textbf{Hamiltonian.} " + siam_text + ".",
        r"Screens over $t\in[0,1]$ (101 points) at the activation cut of each rung, both arms on the compiled-$Q$ route; the AP-McLachlan "
        r"arm uses the declared insertion frontier and the manuscript's permission bound.  Errors are normalized by the RMS excursion of the "
        r"exact reference (the scorer's response components); \emph{burden} is their aggregate.  No target or guard is applied here.",
        r"\begin{table}[H]\centering\scriptsize\setlength{\tabcolsep}{3.5pt}\begin{tabular}{llrl" + "r" * len(comp_names) + r"rrr}\toprule",
        r"arm & segment & rung & cut & " + " & ".join(rf"${n.replace('_', '_{')}}}$" if "_" in n else n for n in comp_names)
        + r" & burden & $N_\theta$ & exch. \\ \midrule",
    ]
    for r in rows:
        cut_tex = {"1em02": "$10^{-2}$", "3em03": r"$3.33\times10^{-3}$", "1em03": "$10^{-3}$", "3em04": r"$3.33\times10^{-4}$",
                   "1em04": "$10^{-4}$", "3em05": r"$3.33\times10^{-5}$", "1em05": "$10^{-5}$", "3em06": r"$3.33\times10^{-6}$",
                   "1em06": "$10^{-6}$", "3em07": r"$3.33\times10^{-7}$", "1em07": "$10^{-7}$"}
        m = re.search(r"cut([0-9a-z]+)_source", str(r["path"])); ct = cut_tex.get(m.group(1), m.group(1)) if m else ""
        comps = " & ".join(f"{r['components'].get(n, float('nan')):.3g}" for n in comp_names)
        lines.append(f"{r['arm']} & {r['segment']} & {r['rung']} & {ct} & {comps} & {r['burden']:.3g} & {r['n_theta']} & "
                     f"{'--' if r['exchanges'] is None else r['exchanges']}" + r" \\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    # Observables of the latest screen of each arm against the exact reference.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    contract = _load_json(rows[0]["path"]).get("observable_contract", {}) if rows else {}
    lines += [
        r"Observables (offline, from the checkpoint statevectors): impurity number $N_d=n_{d\uparrow}+n_{d\downarrow}$, impurity "
        r"double occupancy $D_d$, impurity magnetization $m_d$, impurity spin square $S_d^2=\tfrac34(N_d-2D_d)$, impurity--bath spin "
        r"correlators $C_1,\dots,C_4$ by bath site, and the integrated screening $\Sigma_r=S_d^2+\sum_{j\le r}C_j$.  Each response "
        r"component above is the mean absolute error of that observable divided by its RMS excursion along the exact reference; the "
        r"table below gives the mean absolute errors and the excursions themselves for the latest screen of each arm.",
    ]
    names = ("N_d", "D_d", "m_d", "S_d2", "C_1", "C_2", "C_3", "C_4", "Sigma_4")
    latest = {}
    for r in rows:
        if r["segment"].startswith("$t\\in[0,1]") or "0,1" in r["segment"]:
            if r["arm"] not in latest or r["rung"] > latest[r["arm"]]["rung"]:
                latest[r["arm"]] = r
    if latest:
        lines += [r"\begin{table}[H]\centering\scriptsize\setlength{\tabcolsep}{4pt}\begin{tabular}{ll" + "r" * len(names) + r"}\toprule",
                  r"arm & quantity & " + " & ".join(f"${n.replace('S_d2', 'S_d^2').replace('Sigma_4', chr(92) + 'Sigma_4')}$" for n in names) + r" \\ \midrule"]
        for arm, r in latest.items():
            prm = _load_json(r["path"]).get("primitive_response_metrics", {})
            def val(n, key):
                e = prm.get(n) or prm.get(n.replace("S_d2", "S_d2")) or {}
                v = e.get(key); return "--" if v is None else f"{v:.2e}"
            lines.append(f"{arm} (rung {r['rung']}) & mean abs.\ error & " + " & ".join(val(n, "mean_absolute_error") for n in names) + r" \\")
            lines.append(f" & RMS excursion of the exact reference & " + " & ".join(val(n, "response_scale") for n in names) + r" \\")
        lines += [r"\bottomrule\end{tabular}\end{table}"]
        for arm, r in latest.items():
            s = _load_json(r["path"]); ser = s["series"]; t = np.asarray(ser["time"], dtype=float)
            plt.rcParams.update({"font.size": 7.5, "axes.linewidth": 0.7})
            fig, axes = plt.subplots(3, 3, figsize=(7.4, 5.2), sharex=True)
            for ax, n in zip(axes.flat, names):
                ax.plot(t, np.asarray(ser["exact"][n], dtype=float), color="black", linestyle=":", linewidth=1.1, label="exact")
                ax.plot(t, np.asarray(ser["observed"][n], dtype=float), color="#8b1a1a" if "APM" in arm else "#1f5aa6", linewidth=1.0, label=arm)
                ax.set_ylabel("$" + n.replace("S_d2", "S_d^2").replace("Sigma_4", "\\Sigma_4") + "$"); ax.grid(alpha=0.18, linewidth=0.5)
            for ax in axes[-1]:
                ax.set_xlabel(r"$t$")
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=False, ncol=2)
            fig.tight_layout(rect=(0, 0, 1, 0.97), pad=0.5, w_pad=0.9)
            fname = f"siam_observables_{'avqds' if 'AVQDS' in arm else 'apm'}_r{r['rung']:02d}.pdf"
            fig.savefig(figdir / fname, bbox_inches="tight"); plt.close(fig)
            lines += [rf"\begin{{figure}}[H]\centering\includegraphics[width=0.92\linewidth]{{figures/{fname}}}"
                      rf"\caption{{{arm}, rung {r['rung']} screen ($t\in[0,1]$): observables against the exact reference.  " + siam_text + r".}\end{figure}"]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--locked-comparison", type=Path, required=True)
    parser.add_argument("--corrected-comparison", type=Path, required=True)
    parser.add_argument("--locked-cost-root", type=Path, required=True)
    parser.add_argument("--corrected-cost-root", type=Path, required=True)
    parser.add_argument("--manuscript-table", type=Path, required=True)
    parser.add_argument("--reference-store", type=Path, required=True)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--pilot-run", type=Path, action="append", default=[])
    parser.add_argument("--rung-run", action="append", default=[], help="cell_id=path/to/run.json (rung label read from the run)")
    parser.add_argument("--avqds-run", action="append", default=[], help="cell_id=path/to/run.json (locked AVQDS comparator)")
    parser.add_argument("--siam-score", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args(argv)
    store = ExactReferenceStore(args.reference_store)
    figdir = args.output_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    sections = [pilot_section(args.control_run, args.pilot_run, store, figdir)] if args.pilot_run else []
    rungs: dict[str, list[tuple[str, Path]]] = {}
    for item in args.rung_run:
        cell, _, path = item.partition("=")
        rungs.setdefault(cell, []).append(("", Path(path)))
    for cell in rungs:
        rungs[cell].sort(key=lambda e: -_load_json(e[1])["summary"]["support_patch_config"]["insertion_l2_cut"])
    avqds = {k: Path(v) for k, _, v in (item.partition("=") for item in args.avqds_run)}
    if rungs:
        sections.append(rung_section(rungs, store, figdir, avqds))
    if args.siam_score:
        sections.append(siam_section(args.siam_score, figdir))
    build(
        locked=_load_json(args.locked_comparison), corrected=_load_json(args.corrected_comparison),
        locked_cost_root=args.locked_cost_root, corrected_cost_root=args.corrected_cost_root,
        manuscript_table=args.manuscript_table, output_dir=args.output_dir,
        extra_sections=[r"\clearpage"] + sections, matched_column=False,
        intro_extra=r"This report applies no accuracy guard or pass/fail rule: every arm's errors are printed and left to the reader.",
    )
    if args.compile:
        subprocess.run(["tectonic", "paper_ii_corrected_vs_reported.tex"], cwd=args.output_dir, check=True)
        print(args.output_dir / "paper_ii_corrected_vs_reported.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
