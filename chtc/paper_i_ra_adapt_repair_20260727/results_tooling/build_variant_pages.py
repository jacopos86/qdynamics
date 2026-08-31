import importlib.util, json, sys, re, datetime
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

spec = importlib.util.spec_from_file_location(
    "p1builder",
    "/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/chtc/"
    "paper_i_ra_adapt_repair_20260727/"
    "build_paper_i_ra_phase123_qiskit_comparison_pdf_20260817.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["p1builder"] = mod
spec.loader.exec_module(mod)
exact = {r: v["exact_energy"] for r, v in mod.load_hist_plateau().items()}
REGIMES = list(mod.REGIMES)

# r51-75 of the first continuation batch ran without the published
# tolmatch Powell overlay (invalid; quarantined); plot the valid r<=50
# prefix only until the tolmatch rerun (cluster 9665004) completes.
aavqe = {r: [(k, max(abs(e - exact[r]), 1e-16)) for k, e in rows if k <= 50]
         for r, rows in json.load(open("aavqe_full_hist.json")).items()}

# live samples -> (eid_arm, regime) -> [(round, |err|)]
WT = "/Users/jakestrobel/local_repos/Holstein_test_fullclone_3.worktrees/ra-refactor-stage2/chtc/paper_i_ra_adapt_repair_20260727"
eid_map = {}
for cluster, qfile in (
    (9664259, f"{WT}/paper_i_ra_allphase_adaptive_maximum_k75_extension_20260819_v4_chtc/queue_resume.tsv"),
    (9664325, f"{WT}/ra_adapt_costexp_depthfix_sixregime_20260820_v2_chtc/queue_fresh.tsv"),
):
    for proc, line in enumerate(open(qfile)):
        eid_map[(cluster, proc)] = line.split("\t")[0].strip()

live = defaultdict(dict)
for line in open("live_samples.jsonl"):
    try:
        d = json.loads(line)
    except ValueError:
        continue
    row = d["row"]
    eid = d.get("eid") or eid_map.get((d["cluster"], d["proc"]), "")
    m = re.match(r"(?:sixregime|allphase_k75)__(.+?)__((?:weak|intermediate|strong)_\w+?)__nph\d", eid)
    if not m:
        continue
    arm, regime = m.group(1), m.group(2)
    k = int(row["depth"]) - (1 if d["cluster"] == 9664259 else 0)
    if k < 1:
        continue
    live[(arm, regime)][k] = max(abs(float(row["energy"]) - exact[regime]), 1e-16)
live = {key: sorted(v.items()) for key, v in live.items()}

# completed B summaries
completed = {}
for d, regime in (("bsum12", "weak_weak"), ("bsum13", "intermediate_weak"), ("bsum14", "strong_weak_u8")):
    tr = json.load(open(f"{d}/run/summary/summary.json"))["accepted_error_trace"]
    completed[("b_depth_append_ra", regime)] = [
        (int(r["controller_round"]), max(float(r["absolute_energy_error"]), 1e-16)) for r in tr]

PAGES = [
    ("PAGE 2 - RA continuation arms (k50->75, natural-terminal lineage, live)", 9664259, [
        ("append_ra", "append-RA (continuation)", "#d55e00", "^"),
        ("plateau_position_phase0", "plateau-insertion RA [position-P0]", "#009e73", "s"),
        ("always_open_position_phase0", "always-open-RA", "#cc79a7", "o"),
    ]),
    ("PAGE 3 - Matrix A: costexp min-cost position realization (forced, live)", 9664325, [
        ("a_costexp_always_open", "always-open-RA + pos-fix", "#cc79a7", "o"),
        ("a_costexp_plateau_position", "plateau-RA + pos-fix", "#009e73", "s"),
    ]),
    ("PAGE 4 - Matrix B: three-term Qiskit cost + pos-fix (forced, live+complete)", 9664325, [
        ("b_depth_append_ra", "append-RA + 3-term cost", "#d55e00", "^"),
        ("b_depth_costexp_always_open", "always-open-RA + 3-term + pos-fix", "#cc79a7", "o"),
        ("b_depth_costexp_plateau_position", "plateau-RA + 3-term + pos-fix", "#009e73", "s"),
    ]),
]

stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M CDT")
out = ("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/pdf/"
       "paper_i_ra_allphase_adaptive_20260817/ra_adapt_variant_pages_20260820.pdf")
with PdfPages(out) as pdf:
    for title, cluster, series in PAGES:
        fig = plt.figure(figsize=(15, 9))
        grid = fig.add_gridspec(2, 3, hspace=0.42, top=0.86, bottom=0.08)
        for index, regime in enumerate(REGIMES):
            ax = fig.add_subplot(grid[index // 3, index % 3])
            ks, errs = zip(*aavqe[regime])
            ax.semilogy(ks, errs, "--", color="#0072b2", lw=1.6,
                        label="AAVQE (r<=50; tolmatch r75 rerun in flight)" if index == 0 else None)
            for arm, label, color, marker in series:
                pts = completed.get((arm, regime)) or live.get((arm, regime))
                is_live = (arm, regime) not in completed and (arm, regime) in live
                if pts:
                    k2, e2 = zip(*pts)
                    ax.semilogy(k2, e2, "-", color=color, marker=marker, markersize=2.5, lw=1.2,
                                label=(label + (" [live]" if is_live else "")) if index == 0 else None)
            ax.axvline(50, color="0.6", ls=":", lw=0.8)
            ax.set_xlim(0, 78)
            ax.set_title(regime.replace("_", " "), fontsize=11)
            ax.grid(True, which="both", alpha=0.25)
            if index >= 3:
                ax.set_xlabel("accepted controller round $k$")
            if index % 3 == 0:
                ax.set_ylabel(r"$|E_k - E_{\mathrm{exact}}|$")
        fig.legend(loc="upper left", bbox_to_anchor=(0.055, 0.965), fontsize=8, ncol=2)
        fig.suptitle(title + f"  -  refreshed {stamp}", y=0.97, fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)
    # Matrix C placeholder page
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.axis("off")
    ax.text(0.5, 0.5, "PAGE 5 - Matrix C: three-term cost + alpha-star Schur candidate warm start\n"
            "18 forced cells sealed (identity commit cc86d9eb). Live-lane engagement wiring in final\n"
            "integration; panels populate from first live samples after submission.",
            ha="center", va="center", fontsize=13)
    pdf.savefig(fig); plt.close(fig)
print("written:", out)
