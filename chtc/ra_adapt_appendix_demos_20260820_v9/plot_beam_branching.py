"""Beam branching figure: every executed child against the winning lineage."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXACT = -2.1414935642714075
FLOOR = 1.0e-16
TARGET = 1.0e-6

run = Path(sys.argv[1]); out = Path(sys.argv[2])
# Accept either a published campaign cell or a probe run directory.
_published = run / "accepted_run_evidence.json"
ev = (
    json.loads(_published.read_text())
    if _published.exists()
    else json.loads((run / "result.json").read_text())
    ["scientific_receipts"]["accepted_run_evidence"]
)
diag = ev["beam_search_diagnostics"]
rounds_rows = diag["rounds"]
accepted = ev["accepted_rounds"]

initial_err = abs(1.25 - EXACT)
err_by_branch, segments = {}, []
winner_ids = {str(b) for b in ev["beam_accounting"]["winning_branch_ids"]}
for index, row in enumerate(rounds_rows, start=1):
    survivors = {str(x) for x in (row.get("survivor_branch_ids") or ())}
    for child in row["children"]:
        cid, pid = str(child["branch_id"]), child.get("parent_branch_id")
        perr = initial_err if pid is None else err_by_branch.get(str(pid), initial_err)
        cerr = max(abs(float(child["accepted_energy"]) - EXACT), FLOOR)
        err_by_branch[cid] = cerr
        segments.append({
            "round": index, "p": perr, "c": cerr,
            "winner": cid in winner_ids,
            "survived": cid in survivors,
            "dominated": bool(child.get("pareto_dominated", False)),
        })

main_r = [0] + [int(r["controller_round"]) for r in accepted]
main_e = [initial_err] + [max(abs(float(r["accepted_energy"]) - EXACT), FLOOR) for r in accepted]
marker = next((r for r, e in zip(main_r[1:], main_e[1:]) if e <= TARGET), None)

fig, ax = plt.subplots(figsize=(7.2, 4.6))
WIN, ALT = "#3B6EA5", "#B0555C"
labelled = False
for s in segments:
    if s["winner"]:
        continue
    label = None
    if not labelled:
        labelled = True
        label = "branch refit and not retained"
    ax.plot([s["round"] - 1, s["round"]], [s["p"], s["c"]], color=ALT,
            alpha=0.55, linewidth=0.85, linestyle=(0, (2.5, 1.6)),
            zorder=1, label=label)
    ax.plot([s["round"]], [s["c"]], marker="o", markersize=2.6, color=ALT,
            alpha=0.55, zorder=2)

ax.plot(main_r, main_e, color=WIN, marker="o", markersize=4.4, linewidth=2.0,
        label="winning lineage (reported trajectory)", zorder=4)
ax.axhline(TARGET, color="0.45", linewidth=0.9, linestyle=":", zorder=0)
ax.text(0.25, TARGET * 1.7, r"$10^{-6}t$", fontsize=8, color="0.35")
ax.set_yscale("log")
ax.set_xlabel("accepted iteration $k$")
ax.set_ylabel(r"$|E_k - E_{\rm ED}|\;/\;t$")
ax.set_title("Fork-local beam search, cost-weighted with Pareto filtering\n"
             "L=3 Hubbard, $U/t=1.25$ — every executed branch shown", fontsize=10)
ax.grid(alpha=0.25, linewidth=0.6)
ax.legend(fontsize=8, loc="lower left", framealpha=0.95)
total = len(segments)
ax.text(0.985, 0.965,
        f"branches executed: {total}\n"
        f"all-branch $S_{{\\rm alg}}$: {ev['beam_accounting']['all_executed_s_alg']:,}\n"
        f"winning lineage $S_{{\\rm alg}}$: {ev['beam_accounting']['winning_lineage_s_alg']:,}",
        transform=ax.transAxes, fontsize=7.4, va="top", ha="right", family="monospace",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white",
              "edgecolor": "0.75", "linewidth": 0.6, "alpha": 0.95})
out.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(out, dpi=220)
fig.savefig(out.with_suffix(".pdf"))
print(f"wrote {out}  ({total} branches)")
