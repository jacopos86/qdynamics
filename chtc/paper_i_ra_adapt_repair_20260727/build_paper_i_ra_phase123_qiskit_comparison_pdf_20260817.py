#!/usr/bin/env python3
"""Paper-I-method comparison PDF: historical vs all-phase-adaptive RA routes.

Six regime convergence panels plus the cost-tuple table
(N2q, D2q, Dc, W1q, S_alg), comparing:

- plateau insertion (historical cluster 9605157, k50) vs plateau-RA
  subpolicies (plateau_append_phase0, plateau_position_phase0);
- append insertion (historical cluster 9398375, r70 clipped to k<=50) vs
  append-RA subpolicies (append_ra, append_position_phase0);
- always_open_position_phase0 overlays once cells complete.

New-route cells are read from worker archives extracted under --ra-cells-dir
(subdirs x_<execution_id_prefix>/run/summary/summary.json). Regenerate with
fresh extractions as campaign cells land; panels without RA data yet are
annotated as pending. This script lives outside the pipelines/src source
inventory on purpose: rebuilding it never perturbs sealed campaign packages.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
REGIMES = (
    "weak_weak",
    "intermediate_weak",
    "strong_weak_u8",
    "weak_strong",
    "intermediate_strong",
    "strong_strong_u8",
)
HIST_PLATEAU_GLOB = str(HERE / "retrieved_phase0_completed_20260809/9605157.*_completed_report_adapter.json")
# The published paper's append baseline: the six POWELL tolerance-matched
# r50 append trajectories adopted by author direction on 2026-08-16 —
# resolved through the published package's figure provenance so the series
# is exactly what Paper_I.pdf plots.
PAPER_PACKAGE_PROVENANCE = Path(
    "/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/"
    "output/paper_packages/Paper_I_full_20260816/figures/"
    "paper_i_ra_vs_append_matched_singleton_plateau_20260812/"
    "paper_i_ra_vs_append_matched_singleton_plateau_provenance.json"
)
ERR_FLOOR = 1e-16
K_MAX = 50

FAMILIES = {
    # series key: (panel family, label, color, linestyle)
    "hist_plateau": ("plateau", "plateau historical 9605157 [P0: endpoint-gen, fixed24]", "#8b1a1a", "--"),
    "hist_append": ("append", "append published tolmatch r50 [P0: endpoint-gen]", "#0b3d91", "--"),
    "plateau_append_phase0": ("plateau", "plateau-RA [P0: endpoint-generators]", "#e05c4b", "-"),
    "plateau_position_phase0": ("plateau", "plateau-RA [P0: position-records]", "#f2a03d", "-"),
    "append_ra": ("append", "append-RA [P0: endpoint-generators]", "#2a7fd4", "-"),
    "append_position_phase0": ("append", "append-RA [P0: position-records] (retired cross-check)", "#39b3a6", "-"),
    "always_open_position_phase0": ("open", "always-open-RA [P0: position-records]", "#7a4fbf", "-"),
    "forced_append_ra": ("append", "forced-k50 append-RA [P0: endpoint-gen]", "#0b6b2e", "-."),
    "forced_plateau_append_phase0": ("plateau", "forced-k50 plateau-RA [P0: endpoint-gen]", "#0b6b2e", "-."),
    "forced_plateau_position_phase0": ("plateau", "forced-k50 plateau-RA [P0: position-rec]", "#4a7d0b", "-."),
    "forced_always_open_position_phase0": ("open", "forced-k50 always-open-RA [P0: position-rec]", "#2e6b5e", "-."),
}
TERSE_STATUS = {
    "append_ra": "running 9662333",
    "plateau_append_phase0": "running 9662333",
    "plateau_position_phase0": "running 9662333",
    "always_open_position_phase0": "running 9662334",
    "forced_append_ra": "planned (forced)",
    "forced_plateau_append_phase0": "planned (forced)",
    "forced_plateau_position_phase0": "planned (forced)",
    "forced_always_open_position_phase0": "planned (forced)",
}


def load_hist_plateau() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in sorted(glob.glob(HIST_PLATEAU_GLOB)):
        adapter = json.loads(Path(path).read_text())
        costs = adapter["terminal"]["costs"]
        rows[str(adapter["regime_id"])] = {
            "points": [(int(p["k"]), float(p["error"])) for p in adapter["points"]],
            "exact_energy": float(adapter["exact_same_cutoff_energy"]),
            "k_final": int(adapter["terminal"]["k"]),
            "err_final": float(adapter["terminal"]["error"]),
            "costs": {k: int(costs[k]) for k in ("N2q", "D2q", "Dc", "W1q", "S_alg")},
        }
    return rows


def _append_salg_prefix(result_path: Path, cache_path: Path) -> dict[int, int]:
    """Cumulative charged S_alg per append round from the run ledger."""

    if cache_path.is_file():
        return {int(k): int(v) for k, v in json.loads(cache_path.read_text()).items()}
    payload = json.loads(result_path.read_text())
    occurrences = payload["result_payload"]["estimator_call_ledger"]["occurrences"]
    per_round: dict[int, int] = {}
    for row in occurrences:
        if not row.get("charged", False):
            continue
        scope = str(row.get("consumer_scope", ""))
        if not scope.startswith("append_round_"):
            continue
        round_index = int(scope.split("append_round_")[1].split(":")[0])
        per_round[round_index] = per_round.get(round_index, 0) + 1
    prefix: dict[int, int] = {}
    total = 0
    for round_index in sorted(per_round):
        total += per_round[round_index]
        prefix[round_index] = total
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(prefix))
    return prefix


def load_hist_append(exact_by_regime: dict[str, float], workdir: Path) -> dict[str, dict]:
    provenance = json.loads(PAPER_PACKAGE_PROVENANCE.read_text())
    rows: dict[str, dict] = {}
    for replacement in provenance["adopted_replacements"]:
        regime = str(replacement["regime"])
        summary_path = Path(replacement["source"]["summary"]["path"])
        summary = json.loads(summary_path.read_text())
        exact = exact_by_regime[regime]
        points = [
            (
                int(row["controller_round"]),
                max(abs(float(row["energy_after"]) - exact), ERR_FLOOR),
            )
            for row in summary["accepted_history"]
            if 1 <= int(row["controller_round"]) <= K_MAX
        ]
        terminal = summary["resources"]["terminal_compiled_resources"]
        accounting = summary["estimator_accounting"]
        salg_prefix: dict[int, int] = {}
        result_ref = replacement["source"].get("result")
        if isinstance(result_ref, dict) and workdir is not None:
            try:
                salg_prefix = _append_salg_prefix(
                    Path(result_ref["path"]),
                    Path(workdir) / f"append_salg_prefix_{regime}.json",
                )
            except (OSError, KeyError, ValueError, TypeError):
                salg_prefix = {}
        rows[regime] = {
            "points": points,
            "k_final": int(summary["controller_rounds_completed"]),
            "err_final": points[-1][1] if points else float("nan"),
            "clipped": False,
            "salg_by_round": salg_prefix,
            "costs": {
                "N2q": int(terminal["compiled_count_2q_total"]),
                "D2q": int(terminal["compiled_depth_2q_total"]),
                "Dc": int(terminal["compiled_depth_total"]),
                "W1q": int(terminal["qiskit_pretranspile_pauli_1q_work_total"]),
                "S_alg": int(accounting["S_alg"]),
            },
        }
    return rows


def load_ra_cells(cells_dir: Path) -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    for path in sorted(cells_dir.glob("x_*/run/summary/summary.json")):
        summary = json.loads(path.read_text())
        cell_name = path.parents[2].name  # x_allphase_maxk50__<arm>__<regime>
        arm = next((a for a in FAMILIES if f"__{a}__" in cell_name), None)
        regime = next((r for r in REGIMES if cell_name.endswith(r) or f"__{r}" in cell_name), None)
        trace = summary.get("accepted_error_trace") or []
        if arm is None or regime is None or not trace:
            continue
        work = summary.get("canonical_all_work") or {}
        salg_by_round: dict[int, int] = {}
        result_path = path.parents[1] / "result/result.json"
        if result_path.is_file():
            try:
                payload = json.loads(result_path.read_text())
                prefix_rows = payload["run"]["canonical_reporting"][
                    "accepted_prefix_work"
                ]
                salg_by_round = {
                    index + 1: int(row["s_alg"])
                    for index, row in enumerate(prefix_rows)
                }
            except (OSError, KeyError, ValueError, TypeError):
                salg_by_round = {}
        rows[(arm, regime)] = {
            "salg_by_round": salg_by_round,
            "points": [
                (int(row["controller_round"]), max(float(row["absolute_energy_error"]), ERR_FLOOR))
                for row in trace
            ],
            "k_final": int(trace[-1]["controller_round"]),
            "err_final": max(float(trace[-1]["absolute_energy_error"]), ERR_FLOOR),
            "costs": {"S_alg": int(work.get("s_alg", 0))},
            "horizon_scope": str(summary.get("horizon_scope", "")),
        }
    return rows


SHORT = {
    "hist_plateau": "plateau hist (9605157, k50)",
    "hist_append": "append hist (published tolmatch r50)",
    "plateau_append_phase0": "plateau-RA (append-P0)",
    "plateau_position_phase0": "plateau-RA (position-P0)",
    "append_ra": "append-RA",
    "append_position_phase0": "append-RA (position-P0, cross-check)",
    "always_open_position_phase0": "always-open-RA",
}
RUNNING_STATUS = {
    "append_ra": "running (CHTC 9662333)",
    "plateau_append_phase0": "running (CHTC 9662333)",
    "plateau_position_phase0": "running (CHTC 9662333)",
    "always_open_position_phase0": "running (CHTC 9662334)",
    "forced_append_ra": "forced-k50: awaiting submission",
    "forced_plateau_append_phase0": "forced-k50: awaiting submission",
    "forced_plateau_position_phase0": "forced-k50: awaiting submission",
    "forced_always_open_position_phase0": "forced-k50: awaiting submission",
}
SHORT.update({
    "forced_append_ra": "forced-k50 append-RA",
    "forced_plateau_append_phase0": "forced-k50 plateau-RA (append-P0)",
    "forced_plateau_position_phase0": "forced-k50 plateau-RA (position-P0)",
    "forced_always_open_position_phase0": "forced-k50 always-open-RA",
})
ARM_PAGES = (
    (
        "Arm 1 - Append: published baseline vs natural-RA vs forced-k50",
        "Append-only insertion. Baseline = published Paper-I POWELL "
        "tolerance-matched append (r50). Natural-RA stops at the "
        "authenticated Phase-III no-positive terminal (max k=50); "
        "forced-k50 force-admits through no-positive rounds to exactly "
        "k=50.",
        ("hist_append", "append_ra", "forced_append_ra"),
        (),
    ),
    (
        "Arm 2 - Plateau (append-endpoint Phase 0): historical vs natural vs forced",
        "Plateau-commutation insertion with endpoint-generator Phase 0. "
        "Baseline = historical plateau campaign 9605157 (k50). Forced-k50 "
        "tests whether the historical deep staircase returns under the "
        "new signed Qiskit scoring.",
        ("hist_plateau", "plateau_append_phase0", "forced_plateau_append_phase0"),
        (),
    ),
    (
        "Arm 3 - Plateau (position-record Phase 0): historical vs natural vs forced",
        "Plateau-commutation insertion with position-record Phase 0 "
        "(representation ablation of Arm 2). Baseline = historical "
        "plateau campaign 9605157 (k50).",
        ("hist_plateau", "plateau_position_phase0", "forced_plateau_position_phase0"),
        (),
    ),
    (
        "Arm 4 - Always-open: natural vs forced (historical plateau context)",
        "Always-commutation-reduced insertion: every reduced position "
        "available each round. New arm with no historical counterpart; "
        "historical plateau shown for context.",
        ("hist_plateau", "always_open_position_phase0",
         "forced_always_open_position_phase0"),
        (),
    ),
)
FAMILY_PAGES = (
    (
        "Append family: published Paper-I append vs append-RA",
        "Append-only insertion: each round may only add one operator at the "
        "circuit end. Baseline = the published Paper-I POWELL tolerance-"
        "matched append trajectories (r50). RA = all-phase-adaptive "
        "shortlists, Qiskit-costed Phases I-III, authenticated natural-"
        "terminal stopping under max k=50.",
        ("hist_append", "append_ra", "forced_append_ra"),
        ("forced-k50 append-RA (planned): Phase-III no-positive rounds "
         "force-admit the argmax signed-score candidate; every cell runs "
         "to exactly k=50.",),
    ),
    (
        "Plateau family: historical plateau vs plateau-RA subpolicies",
        "Plateau-commutation insertion: interior positions open per the "
        "commutation-reduction plateau law. Baseline = historical Paper-I "
        "plateau campaign 9605157 (k50, fixed shortlists, Phase II-III "
        "Qiskit; P0 = endpoint generators). TWO distinct RA subpolicies "
        "run here: [P0: endpoint-generators] ranks whole generators at "
        "the append endpoint; [P0: position-records] ranks "
        "(generator, insertion-position) pairs. Same insertion law, "
        "different Phase-0 population identity.",
        ("hist_plateau", "plateau_append_phase0", "plateau_position_phase0",
         "forced_plateau_append_phase0", "forced_plateau_position_phase0"),
        ("forced-k50 plateau arms (planned): forced admission through "
         "no-positive plateaus, exact k=50 - directly tests whether the "
         "historical deep staircase returns under the new scoring.",),
    ),
    (
        "Always-open insertion (new arm; historical plateau for context)",
        "Always-commutation-reduced insertion: every reduced position is "
        "available every round (maximal insertion freedom). Phase 0 is "
        "position-records by construction (the open insertion domain IS a "
        "position-record concept; no endpoint-generator variant exists). "
        "No historical counterpart; the historical plateau baseline is "
        "context only. One natural cluster (9662334) runs now; the "
        "forced-k50 twin is planned.",
        ("hist_plateau", "always_open_position_phase0",
         "forced_always_open_position_phase0"),
        ("forced-k50 always-open (planned): same freedom, forced to "
         "exact k=50.",),
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ra-cells-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--status-note", default="")
    args = parser.parse_args()

    hist_plateau = load_hist_plateau()
    exact = {r: v["exact_energy"] for r, v in hist_plateau.items()}
    hist_append = load_hist_append(exact, args.workdir)
    ra = load_ra_cells(args.ra_cells_dir)

    def entry_for(key: str, regime: str):
        if key == "hist_plateau":
            return hist_plateau.get(regime)
        if key == "hist_append":
            return hist_append.get(regime)
        return ra.get((key, regime))

    forced_present = any(arm.startswith("forced_") for arm, _ in ra)
    pages = ARM_PAGES if forced_present else FAMILY_PAGES

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        # Cover page: full campaign map.
        fig, ax = plt.subplots(figsize=(15, 11.5))
        ax.axis("off")
        done = {}
        for (arm, regime) in ra:
            done.setdefault(arm, []).append(regime)
        cover_rows = [
            ["historical plateau", "cluster 9605157 (fixed shortlists, phase II-III Qiskit)", "complete (6/6)", "plateau pages"],
            ["published append", "POWELL tolmatch r50 (adopted in Paper_I.pdf 2026-08-16)", "complete (6/6)", "append page"],
            ["append-RA (natural)", "CHTC 9662333 procs 0-5", f"{len(done.get('append_ra', []))}/6 complete", "append page"],
            ["plateau-RA append-P0 (natural)", "CHTC 9662333 procs 6-11", f"{len(done.get('plateau_append_phase0', []))}/6 complete", "plateau page(s)"],
            ["plateau-RA position-P0 (natural)", "CHTC 9662333 procs 12-17", f"{len(done.get('plateau_position_phase0', []))}/6 complete", "plateau page(s)"],
            ["always-open-RA (natural)", "CHTC 9662334 procs 0-5", f"{len(done.get('always_open_position_phase0', []))}/6 complete", "always-open page"],
            ["forced-k50 (4 arms x 6 regimes)", "24 cells; forced admission at no-positive rounds, exact k=50", ("data present" if forced_present else "awaiting submission"), "per-arm pages once data lands"],
            ["append-position-RA", "withdrawn by user (single append-RA arm policy); one banked strong_weak cross-check", "retired", "-"],
        ]
        cover = ax.table(
            cellText=cover_rows,
            colLabels=["route / campaign", "provenance", "status", "PDF location"],
            loc="center", cellLoc="left",
            colWidths=[0.22, 0.40, 0.16, 0.18],
        )
        cover.auto_set_font_size(False)
        cover.set_fontsize(9)
        cover.scale(1.0, 1.8)
        ax.set_title(
            "Paper-I RA campaign map - all routes, one page each family/arm\n"
            "All RA science: gradient-only Phase 0, adaptive shortlists 0-III, "
            "Qiskit-costed Phases I-III, POWELL(200), seeds 7/7, singleton "
            "admission, no pruning/beam.\n" + (args.status_note or ""),
            fontsize=12, pad=28,
        )
        pdf.savefig(fig)
        plt.close(fig)

        for page_title, description, keys, planned_notes in pages:
            fig = plt.figure(figsize=(15, 11.5))
            grid = fig.add_gridspec(
                3, 3, height_ratios=(1.0, 1.0, 1.45),
                hspace=0.42, top=0.865, bottom=0.015,
            )
            axes = [
                [fig.add_subplot(grid[r, c]) for c in range(3)]
                for r in range(2)
            ]
            for index, regime in enumerate(REGIMES):
                ax = axes[index // 3][index % 3]
                pending_here = []
                for key in keys:
                    entry = entry_for(key, regime)
                    if entry is None:
                        if not key.startswith("hist_"):
                            pending_here.append(
                                SHORT.get(key, key)
                                + " ["
                                + TERSE_STATUS.get(key, "pending")
                                + "]"
                            )
                        continue
                    _, label, color, style = FAMILIES[key]
                    points = entry["points"]
                    ax.plot([q[0] for q in points], [q[1] for q in points],
                            style, color=color, linewidth=1.4,
                            marker="o" if style == "-" else None,
                            markersize=2.5, label=label)
                ax.set_yscale("log")
                ax.set_title(regime.replace("_", " "), fontsize=11)
                ax.grid(True, which="both", alpha=0.25)
                ax.set_xlim(0, K_MAX + 2)
                if pending_here:
                    ax.text(0.97, 0.95,
                            "pending:\n" + "\n".join(pending_here),
                            transform=ax.transAxes, ha="right", va="top",
                            fontsize=7, color="#666666",
                            bbox=dict(boxstyle="round", fc="#f5f5f5",
                                      ec="#cccccc"))
                if index // 3 == 1:
                    ax.set_xlabel("accepted controller round $k$")
                if index % 3 == 0:
                    ax.set_ylabel(r"$|E_k - E_{\mathrm{exact}}|$")
            handles, labels = [], []
            for ax_row in axes:
                for ax in ax_row:
                    for handle, label in zip(*ax.get_legend_handles_labels()):
                        if label not in labels:
                            handles.append(handle)
                            labels.append(label)
            fig.legend(handles, labels, loc="upper right", ncol=1,
                       fontsize=8, frameon=False,
                       bbox_to_anchor=(0.99, 0.985))

            table_ax = fig.add_subplot(grid[2, :])
            table_ax.axis("off")
            dual_point = "hist_append" in keys
            columns = ["regime", "series", "point", "err",
                       "N2q", "D2q", "Dc", "W1q", "S_alg"]
            rows = []

            def err_at(entry, k):
                for kk, err in entry["points"]:
                    if kk == k:
                        return err
                return None

            for regime in REGIMES:
                hist = entry_for(keys[0], regime)
                ra_entries = [
                    (key, entry_for(key, regime))
                    for key in keys
                    if not key.startswith("hist_")
                ]
                ra_present = [(k, e) for k, e in ra_entries if e is not None]
                if dual_point and hist is not None and ra_present:
                    # Plateau onset of the append comparator: first round
                    # within 0.1 decades of its r50 terminal error.
                    target = hist["err_final"] * (10 ** 0.1)
                    kstar = next(
                        (kk for kk, err in hist["points"] if err <= target),
                        hist["k_final"],
                    )
                    costs = hist.get("costs", {})
                    salg_hist = hist.get("salg_by_round", {})
                    rows.append([
                        regime, SHORT.get(keys[0], keys[0]), "k=50",
                        f"{hist['err_final']:.2e}",
                        str(costs.get("N2q", "pend.")),
                        str(costs.get("D2q", "pend.")),
                        str(costs.get("Dc", "pend.")),
                        str(costs.get("W1q", "pend.")),
                        str(costs.get("S_alg", "pend.")),
                    ])
                    e_kstar = err_at(hist, kstar)
                    rows.append([
                        regime, SHORT.get(keys[0], keys[0]),
                        f"k*={kstar} (plateau)",
                        f"{e_kstar:.2e}" if e_kstar is not None else "-",
                        "pend.", "pend.", "pend.", "pend.",
                        str(salg_hist.get(kstar, "n/a")),
                    ])
                    for key, entry in ra_present:
                        salg_ra = entry.get("salg_by_round", {})
                        rows.append([
                            regime, SHORT.get(key, key),
                            f"k={entry['k_final']} (terminal)",
                            f"{entry['err_final']:.2e}",
                            "pend.", "pend.", "pend.", "pend.",
                            str(entry["costs"].get("S_alg", "pend.")),
                        ])
                        e_ra = err_at(entry, kstar)
                        rows.append([
                            regime, SHORT.get(key, key),
                            f"k*={kstar}",
                            f"{e_ra:.2e}" if e_ra is not None
                            else f"ended k={entry['k_final']}",
                            "pend.", "pend.", "pend.", "pend.",
                            str(salg_ra.get(kstar, "pend.")),
                        ])
                    continue
                for key in keys:
                    entry = entry_for(key, regime)
                    if entry is None:
                        continue
                    costs = entry.get("costs", {})
                    rows.append([
                        regime, SHORT.get(key, key),
                        f"k={entry['k_final']}",
                        f"{entry['err_final']:.2e}",
                        str(costs.get("N2q", "pend.")),
                        str(costs.get("D2q", "pend.")),
                        str(costs.get("Dc", "pend.")),
                        str(costs.get("W1q", "pend.")),
                        str(costs.get("S_alg", "pend.")),
                    ])
            for key in keys:
                if key.startswith("hist_"):
                    continue
                missing = [r for r in REGIMES if (key, r) not in ra]
                if missing:
                    rows.append([
                        ", ".join(m.replace("_", " ") for m in missing),
                        SHORT.get(key, key),
                        "-", RUNNING_STATUS.get(key, "pending"),
                        "-", "-", "-", "-", "-",
                    ])
            widths = [0.17, 0.21, 0.06, 0.13, 0.08, 0.08, 0.08, 0.08, 0.11]
            table = table_ax.table(cellText=rows, colLabels=columns,
                                   loc="upper center", cellLoc="center",
                                   colWidths=widths)
            table.auto_set_font_size(False)
            table.set_fontsize(6.5)
            table.scale(1.0, 1.02)

            planned_text = "  |  ".join(planned_notes)
            fig.text(0.06, 0.004, "PLANNED: " + planned_text,
                     fontsize=8, color="#555555", ha="left")
            fig.suptitle(page_title, fontsize=13, y=0.985)
            note = description
            if args.status_note:
                note += "   [" + args.status_note + "]"
            fig.text(0.06, 0.878, note, fontsize=8.5, color="#333333",
                     ha="left", wrap=True, va="bottom")
            pdf.savefig(fig)
            plt.close(fig)
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
