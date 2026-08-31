"""L=3 noise panel: RA-append and AVQE at three noise levels."""
import hashlib
import json
import statistics
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3.worktrees/appendix-demos-20260820")
CELL_ROOT = (
    REPO
    / "output/local_runs/ra_adapt_appendix_noise_l3_20260821_v1/cells"
)
CORRECTION_SOURCE = Path(
    "/Users/jakestrobel/local_repos/ADAPT---Paper-I/"
    "Paper_I_overleaf_20260818/figures/ra_optional_controls_main/"
    "estimator_reuse_corrections_20260821.json"
)
CORRECTIONS = json.loads(CORRECTION_SOURCE.read_text())

EXACT, FLOOR = -2.1414935642714075, 1.0e-16
COMPARATOR = sys.argv[1] if len(sys.argv) > 1 else "high"
OUT = (
    Path(sys.argv[2])
    if len(sys.argv) > 2
    else Path(__file__).resolve().parent / "figures/ra_appendix_l3_noise_v9.png"
)

AVQE_TERMINAL_COMPILE = {
    "low": {"N2q": 374, "D2q": 344, "Dc": 1413},
    "high": {"N2q": 464, "D2q": 430, "Dc": 1736},
    "extreme": {"N2q": 244, "D2q": 208, "Dc": 829},
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def ra_curve(level):
    summary_path = CELL_ROOT / f"l3_noise_{level}_h20/cell_summary.json"
    rows = json.loads(summary_path.read_text())["accepted_rounds"]
    r, e = [0], [abs(1.25 - EXACT)]
    for row in rows:
        r.append(int(row["controller_round"]))
        e.append(max(float(row["same_cutoff_absolute_error"]), FLOOR))
    terminal = {
        **rows[-1]["compile"],
        "S_alg": CORRECTIONS["noise"]["k20"][level]["corrected"],
    }
    return r, e, terminal, summary_path


def avqe_curve(level):
    summary_path = CELL_ROOT / f"l3_aavqe_noise_{level}_h20/cell_summary.json"
    rows = json.loads(summary_path.read_text())["accepted_rounds"]
    r, e = [0], [abs(1.25 - EXACT)]
    for row in rows:
        r.append(int(row["controller_round"]))
        e.append(max(float(row["same_cutoff_absolute_error"]), FLOOR))
    terminal = {
        **AVQE_TERMINAL_COMPILE[level],
        "S_alg": int(rows[-1]["cumulative_s_alg"]),
    }
    return r, e, terminal, summary_path

LEVELS = (("low", "#4C78A8"), ("high", "#E45756"), ("extreme", "#8172B2"))
PLATEAU_FROM = 10
fig, ax = plt.subplots(figsize=(7.4, 4.5))
lines = []
sources = {}
for level, color in LEVELS:
    r, e, terminal, source = ra_curve(level)
    sources[f"ra_{level}"] = source
    # Measured plateau, not an analytic floor: the plotted error is the exact
    # energy at the parameters the noisy optimizer reached, so no closed-form
    # measurement floor applies.  The line is the median over k >= PLATEAU_FROM.
    plateau = statistics.median(e[PLATEAU_FROM:])
    ax.axhline(plateau, color=color, linewidth=0.9, linestyle=(0, (1, 2.5)),
               alpha=0.8, zorder=0)
    ax.text(20.4, plateau, f"{plateau:.1e}", fontsize=6.2, color=color,
            va="center", ha="left")
    ax.plot(r, e, color=color, marker="o", markersize=3.6, linewidth=1.8,
            label=f"RA-append, {level} noise", zorder=3)
    t = terminal
    lines.append(f"RA {level:<8} floor {e[-1]:.2e}  "
                 f"({t['N2q']}, {t['D2q']}, {t['Dc']}, {t['S_alg']})")

for level, color in LEVELS:
    rounds, errors, t, source = avqe_curve(level)
    sources[f"avqe_{level}"] = source
    ax.plot(rounds, errors, color=color, marker="x", markersize=4.2,
            linewidth=1.4, linestyle="--", alpha=0.85,
            label=f"AVQE, {level} noise", zorder=2)
    lines.append(f"AVQE {level:<7} floor {errors[-1]:.2e}  "
                 f"({t['N2q']}, {t['D2q']}, {t['Dc']}, {t['S_alg']})")

ax.set_yscale("log")
ax.set_xlabel("accepted iteration $k$")
ax.set_ylabel(r"$|E_k - E_{\rm ED}|\;/\;t$")
ax.set_title(r"$L=3$ Hubbard, $U/t=1.25$ — RA-append vs AVQE at three noise levels", fontsize=10)
ax.grid(alpha=0.25, linewidth=0.6)
ax.legend(fontsize=7.0, loc="upper right", framealpha=0.95, ncol=2)
ax.text(0.015, 0.03, "\n".join(lines), transform=ax.transAxes, fontsize=6.6,
        va="bottom", ha="left", family="monospace",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
              "edgecolor": "0.75", "linewidth": 0.6, "alpha": 0.95})
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout(); fig.savefig(OUT, dpi=220); fig.savefig(OUT.with_suffix(".pdf"))
provenance = {
    "schema": "paper_i_l3_noise_figure_v10_reuse_corrected",
    "generator": {
        "path": str(Path(__file__).resolve()),
        "sha256": sha256(Path(__file__).resolve()),
        "command": f"python3 {Path(__file__).resolve()} {COMPARATOR} {OUT}",
    },
    "estimator_corrections": {
        "path": str(CORRECTION_SOURCE),
        "sha256": sha256(CORRECTION_SOURCE),
    },
    "sources": {
        key: {"path": str(path), "sha256": sha256(path)}
        for key, path in sorted(sources.items())
    },
    "outputs": {
        "png": str(OUT),
        "png_sha256": sha256(OUT),
        "pdf": str(OUT.with_suffix(".pdf")),
        "pdf_sha256": sha256(OUT.with_suffix(".pdf")),
    },
}
OUT.with_name(OUT.stem + "_provenance.json").write_text(
    json.dumps(provenance, indent=2, sort_keys=True) + "\n"
)
print(f"wrote {OUT}")
