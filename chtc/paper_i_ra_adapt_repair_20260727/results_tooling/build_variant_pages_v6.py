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

LR = "/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/local_runs"
SCRATCH = ("/private/tmp/claude-501/-Users-jakestrobel-Documents-Holstein-"
           "implementation-Holstein-test-fullclone-3/"
           "91434114-4541-47ef-95e4-2b8bd90bd929/scratchpad")
OTHER = ("/private/tmp/claude-501/-Users-jakestrobel-Documents-Holstein-"
         "implementation-Holstein-test-fullclone-3/"
         "4759e8af-c607-49b1-b24e-7240b4844aff/scratchpad")
WT = ("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3.worktrees/"
      "ra-refactor-stage2/chtc/paper_i_ra_adapt_repair_20260727")

# AAVQE reference: authenticated tolmatch r1-75 (cluster 9665004)
aavqe = {r: [(k, max(abs(e - exact[r]), 1e-16)) for k, e in rows]
         for r, rows in json.load(open(
             f"{LR}/paper_i_aavqe_r50_to_r75_tolmatch_20260820_v2/"
             "aavqe_full_hist_tolmatch_9665004.json")).items()}
aavqe_cost = json.load(open(
    f"{LR}/paper_i_aavqe_r50_to_r75_tolmatch_20260820_v2/"
    "aavqe_r75_cost_tuples_9665004.json"))
bcost = json.load(open(
    f"{LR}/ra_adapt_costexp_depthfix_sixregime_20260820_v2/"
    "completed_cost_tuples.json"))

eid_map = {}
for cluster, qfile in (
    (9664259, f"{WT}/paper_i_ra_allphase_adaptive_maximum_k75_extension_20260819_v4_chtc/queue_resume.tsv"),
    (9664325, f"{WT}/ra_adapt_costexp_depthfix_sixregime_20260820_v2_chtc/queue_fresh.tsv"),
    (9665210, f"{WT}/ra_adapt_warmstart_sixregime_20260820_v1_chtc/queue_fresh.tsv"),
):
    for proc, line in enumerate(open(qfile)):
        eid_map[(cluster, proc)] = line.split("\t")[0].strip()

TAILS = ("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/"
         "local_runs/paper_i_ra_allphase_adaptive_20260817_comparison_data/live_tails")
live = defaultdict(dict)
import glob as _glob
for tf in _glob.glob(TAILS + "/allphase_k75__*.tail"):
    m = re.match(r"allphase_k75__(.+?)__((?:weak|intermediate|strong)_\w+?)__nph\d",
                 tf.split("/")[-1])
    if not m:
        continue
    arm, regime = m.group(1), m.group(2)
    for line in open(tf):
        if '"hardcoded_adapt_iter"' not in line:
            continue
        i = line.find("{")
        try:
            row = json.loads(line[i:])
        except ValueError:
            continue
        k = int(row["depth"]) - 1
        if k >= 1:
            live[(arm, regime)][k] = max(abs(float(row["energy"]) - exact[regime]), 1e-16)
for line in open(f"{SCRATCH}/live_samples_fresh.jsonl"):
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
    if d["cluster"] == 9664259:
        continue  # continuation arms come from the merged grounded tails
    k = int(row["depth"])
    if k < 1:
        continue
    live[(arm, regime)][k] = max(abs(float(row["energy"]) - exact[regime]), 1e-16)
live = {key: sorted(v.items()) for key, v in live.items()}

completed = {}
CDD = ("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/"
       "local_runs/paper_i_ra_allphase_adaptive_20260817_comparison_data")
import glob as _g2, os as _os
for prefix in ("x_allphase_maxk50__", "x_allphase_k75__"):  # k75 second: overrides
    for cd in sorted(_g2.glob(f"{CDD}/{prefix}*")):
        name = _os.path.basename(cd)[len(prefix):]
        for reg in REGIMES:
            if name.endswith("__" + reg):
                arm, regime = name[: -(len(reg) + 2)], reg
                break
        else:
            continue
        if arm not in ("append_ra", "plateau_position_phase0", "always_open_position_phase0"):
            continue
        sp = f"{cd}/run/summary/summary.json"
        if not _os.path.exists(sp):
            continue
        tr = json.load(open(sp)).get("accepted_error_trace") or []
        if tr:
            completed[(arm, regime)] = [
                (int(r["controller_round"]), max(float(r["absolute_energy_error"]), 1e-16))
                for r in tr]
for d, regime in (("bsum12", "weak_weak"), ("bsum13", "intermediate_weak"), ("bsum14", "strong_weak_u8")):
    tr = json.load(open(f"{OTHER}/{d}/run/summary/summary.json"))["accepted_error_trace"]
    completed[("b_depth_append_ra", regime)] = [
        (int(r["controller_round"]), max(float(r["absolute_energy_error"]), 1e-16)) for r in tr]

# completed Bundle-4 cells: traces from the AP .out write-back streams
import glob as _g4, os as _os4
_B4 = json.load(open("/private/tmp/claude-501/-Users-jakestrobel-Documents-"
                     "Holstein-implementation-Holstein-test-fullclone-3/"
                     "91434114-4541-47ef-95e4-2b8bd90bd929/scratchpad/kstar_tables.json")).get("b4", {})
for _f in _g4.glob(f"{SCRATCH}/matrix_outs/9665210.*-sixregime__c_ws_depth_append_ra__*.out"):
    _m = re.search(r"sixregime__(.+?)__((?:weak|intermediate|strong)_\w+?)__nph\d", _os4.path.basename(_f))
    if not _m or f"{_m.group(1)}|{_m.group(2)}" not in _B4:
        continue
    _arm, _reg = _m.group(1), _m.group(2)
    _pts = {}
    for _line in open(_f):
        if '"hardcoded_adapt_iter"' not in _line:
            continue
        _i = _line.find("{")
        try:
            _row = json.loads(_line[_i:])
        except ValueError:
            continue
        _pts[int(_row["depth"])] = max(abs(float(_row["energy"]) - exact[_reg]), 1e-16)
    if _pts:
        completed[(_arm, _reg)] = sorted(_pts.items())

PAGES = [
    ("Page 1 - Bundle 1: Old-safety/back-up paper I jobs - old route, old cost term (strong sector continuing to 75; cluster 9664259)", "1", [
        ("append_ra", "append-RA", "#d55e00", "^"),
        ("plateau_position_phase0", "plateau-insertion RA", "#009e73", "s"),
        ("always_open_position_phase0", "always-insertion RA", "#cc79a7", "o"),
    ], None),
    ("Page 2 - Bundle 2: Old-safety #2 - Insert & Plateau with new positions in equivalence classes (cluster 9664325)", "2", [
        ("a_costexp_always_open", "always-insertion RA", "#cc79a7", "o"),
        ("a_costexp_plateau_position", "plateau-insertion RA", "#009e73", "s"),
    ], None),
    ("Page 3 - Bundle 3: Final RA paper I facing, no-warm-start, yes-position-cost - 2q depth + 2q count + circuit depth (cluster 9664325)", "3", [
        ("b_depth_append_ra", "append-RA", "#d55e00", "^"),
        ("b_depth_costexp_always_open", "always-insertion RA", "#cc79a7", "o"),
        ("b_depth_costexp_plateau_position", "plateau-insertion RA", "#009e73", "s"),
    ], None),
    ("Page 4 - Bundle 4: Final Paper I candidate warm-start - alpha* from the Phase-III scored optimum (cluster 9665210)", "4", [
        ("c_ws_depth_append_ra", "append-RA (warm start)", "#d55e00", "^"),
        ("c_ws_depth_costexp_always_open", "always-insertion RA (warm start)", "#cc79a7", "o"),
        ("c_ws_depth_costexp_plateau_position", "plateau-insertion RA (warm start)", "#009e73", "s"),
    ], "Submitted 2026-08-20 ~10:05 CT after kernel-lane engagement proof (commits 24e23efa, 274be6dd); "
       "18 forced cells running, no cell complete yet -> no terminal results to formalize; "
       "promotion into the deliverable tables follows completion (campaign directive)."),
]

KT = json.load(open("/private/tmp/claude-501/-Users-jakestrobel-Documents-"
                     "Holstein-implementation-Holstein-test-fullclone-3/"
                     "91434114-4541-47ef-95e4-2b8bd90bd929/scratchpad/kstar_tables.json"))
KSTAR = KT["kstar"]

def _pair(a, b, fmt="{:.1e}"):
    fa = "-" if a is None else (fmt.format(a) if isinstance(a, float) else str(a))
    fb = "-" if b is None else (fmt.format(b) if isinstance(b, float) else str(b))
    return f"{fa} / {fb}"

def _tup_pair(ts, tt, key):
    a = (ts or {}).get(key)
    b = (tt or {}).get(key)
    return _pair(a, b, "{}")

REG_SHORT = {"weak_weak": "ww", "intermediate_weak": "iw", "strong_weak_u8": "swu8",
             "weak_strong": "ws", "intermediate_strong": "is", "strong_strong_u8": "ssu8"}

def fmt(v):
    return "-" if v is None else (f"{v:,}".replace(",", " ") if isinstance(v, int) else v)

stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M CDT")
out = ("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/pdf/"
       "paper_i_ra_allphase_adaptive_20260817/ra_adapt_variant_pages_20260820.pdf")
npages = len(PAGES)
with PdfPages(out) as pdf:
    for pageno, (title, deliv, series, note) in enumerate(PAGES, start=1):
        fig = plt.figure(figsize=(15, 10.5))
        grid = fig.add_gridspec(2, 3, hspace=0.40, top=0.855, bottom=0.42)
        for index, regime in enumerate(REGIMES):
            ax = fig.add_subplot(grid[index // 3, index % 3])
            ks, errs = zip(*aavqe[regime])
            ax.semilogy(ks, errs, "--", color="#0072b2", lw=1.6,
                        label="AAVQE baseline (r75, 9665004)" if index == 0 else None)
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

        # ---- k*/k_term cost table: AAVQE + every method on this page ----
        b1t, b3t = KT["b1"], KT["b3"]
        rows, colors = [], []
        for regime in REGIMES:
            av = KT["aavqe"][regime]
            rows.append(["AAVQE", regime, _pair(av["k_star"], av["k_term"], "{}"),
                         _pair(av["dE_star"], av["dE_term"]),
                         _tup_pair(av["tuple_star"], av["tuple_term"], "N2q"),
                         _tup_pair(av["tuple_star"], av["tuple_term"], "D2q"),
                         _tup_pair(av["tuple_star"], av["tuple_term"], "Dc"),
                         _pair(av["salg_star"], av["salg_term"], "{}")])
            colors.append("#dce6f1")
            for arm, label, _, _ in series:
                key = f"{arm}|{regime}"
                comp = None
                if key in KT.get("b4", {}) and pageno == 4:
                    comp = KT["b4"][key]
                elif key in b3t and pageno == 3:
                    comp = b3t[key]
                elif pageno == 1:
                    comp = b1t.get(key)
                if comp is not None:
                    rows.append([label, regime,
                                 _pair(comp["k_star"], comp["k_term"], "{}"),
                                 _pair(comp["dE_star"], comp["dE_term"]),
                                 _tup_pair(comp["tuple_star"], comp["tuple_term"], "N2q"),
                                 _tup_pair(comp["tuple_star"], comp["tuple_term"], "D2q"),
                                 _tup_pair(comp["tuple_star"], comp["tuple_term"], "Dc"),
                                 _pair(comp["salg_star"], comp["salg_term"], "{}")])
                    colors.append("white")
                elif (arm, regime) in live:
                    pts = dict(live[(arm, regime)])
                    kt_live = max(pts)
                    ks = KSTAR[regime]
                    ls = KT.get("live_star", {}).get(key) or {}
                    ts = ls.get("tuple_star")
                    rows.append([label + " [live]", regime,
                                 _pair(ks if (ks in pts or ls) else None, kt_live, "{}"),
                                 _pair(ls.get("dE_star", pts.get(ks)), pts[kt_live]),
                                 _tup_pair(ts, None, "N2q"),
                                 _tup_pair(ts, None, "D2q"),
                                 _tup_pair(ts, None, "Dc"),
                                 _pair(ls.get("salg_star"), None, "{}")])
                    colors.append("#fdf3e3")
        tab_ax = fig.add_axes([0.05, 0.045, 0.90, 0.30])
        tab_ax.axis("off")
        table = tab_ax.table(
            cellText=rows,
            colLabels=["series", "regime", "k* / k_term", "dE k*/term", "N2q k*/term",
                       "D2q k*/term", "Dc k*/term", "S_alg k*/term"],
            cellLoc="center", loc="upper center")
        table.auto_set_font_size(False)
        table.set_fontsize(5.8)
        table.scale(1, 0.85)
        for (r, c0), cell in table.get_celld().items():
            if r > 0:
                cell.set_facecolor(colors[r - 1])
        fig.text(0.05, 0.006,
                 "k* = first AAVQE round within 0.1 decade of its r75 terminal error (plateau point). "
                 "ALL cost tuples: routed FakeMarrakesh ol=1 seed_transpiler=7 (incl. routing/SWAP), compiled from "
                 "accepted-word prefixes (matrix-lane scored marginals retained in archives). "
                 "Live rows: k_term costs and S_alg land at completion.",
                 fontsize=6.5, color="0.35")
        if note:
            fig.text(0.05, 0.358, note, fontsize=7.5, color="#8b0000")
        fig.legend(loc="upper left", bbox_to_anchor=(0.055, 0.930), fontsize=8, ncol=2)
        fig.suptitle(f"{title}  -  refreshed {stamp}", y=0.99, fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)
print("written:", out)
