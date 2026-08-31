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
for _of in _glob.glob(f"{SCRATCH}/matrix_outs/*-sixregime__*.out"):
    _m = re.search(r"sixregime__(.+?)__((?:weak|intermediate|strong)_\w+?)__nph\d", _of.split("/")[-1])
    if not _m:
        continue
    for _line in open(_of):
        if '"hardcoded_adapt_iter"' not in _line:
            continue
        _i = _line.find("{")
        try:
            _row = json.loads(_line[_i:])
        except ValueError:
            continue
        _k = int(_row["depth"])
        if _k >= 1:
            live[(_m.group(1), _m.group(2))][_k] = max(
                abs(float(_row["energy"]) - exact[_m.group(2)]), 1e-16)
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
_KTE = json.load(open("/private/tmp/claude-501/-Users-jakestrobel-Documents-"
                     "Holstein-implementation-Holstein-test-fullclone-3/"
                     "91434114-4541-47ef-95e4-2b8bd90bd929/scratchpad/kstar_tables.json"))
_B4 = _KTE.get("b4", {})
for _sec in ("b2", "b3", "b4", "b5"):
    for _k, _e in _KTE.get(_sec, {}).items():
        if _e.get("trace"):
            _a, _r = _k.split("|")
            completed[(_a, _r)] = [(int(kk), max(float(er), 1e-16)) for kk, er in _e["trace"]]
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

# Bundle 5 (B3-MR) live tails: strong cells still running on 9674089
import collections as _c5
_b5live = _c5.defaultdict(dict)
for _bf in _glob.glob("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/local_runs/paper_i_hh_b3_measured_residual_20260821/live_tails/b3mr__*.tail"):
    _m = re.search(r"b3mr__(mr_\w+?)__((?:weak|intermediate|strong)_\w+?)__nph\d", _bf.split("/")[-1])
    if not _m:
        continue
    for _line in open(_bf):
        _i = _line.find("{")
        try:
            _row = json.loads(_line[_i:])
        except ValueError:
            continue
        _k = int(_row["depth"])
        if _k >= 1:
            _b5live[(_m.group(1), _m.group(2))][_k] = max(
                abs(float(_row["energy"]) - exact[_m.group(2)]), 1e-16)
for _key, _pts in _b5live.items():
    live[_key] = sorted(_pts.items())

PAGES = [
    ("Page 1 - Bundle 1: Old-safety/back-up paper I jobs - old route, old cost term (strong sector continuing to 75; cluster 9664259)", "1", [
        ("append_ra", "append-RA", "#d55e00", "^"),
        ("plateau_position_phase0", "plateau-insertion RA", "#009e73", "s"),
        ("always_open_position_phase0", "always-insertion RA", "#cc79a7", "o"),
    ], None),
    ("Page 2 - Bundle 2: Old-safety #2 - Insert & Plateau with new positions in equivalence classes (cluster 9664325)", "2", [
        ("append_ra", "append-RA (Bundle 1 reference)", "#d55e00", "^"),
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
    ], "Bundle 3 + alpha* warm start (candidate amplitude seeded from its Phase-III scored optimum). Warm-start gate: "
       "50 seeded admissions / 0 cold fallbacks in every completed cell. AUDIT 2026-08-20: on every completed matched "
       "comparison (append arm, weak trio) Bundle 4 equals Bundle 3 accuracy at k* to 7+ digits at 2.8-5.8% lower S_alg. "
       "Remaining cells running; rows fill as they complete."),
    ("Page 5 - Bundle 5: Bundle 3 without the stationarity assumption - measured residual + marginal candidate gain (cluster 9674089)", "5", [
        ("mr_append_ra", "append-RA (measured residual)", "#d55e00", "^"),
        ("mr_always_insertion", "always-insertion RA (measured residual)", "#cc79a7", "o"),
        ("mr_plateau_insertion", "plateau-insertion RA (measured residual)", "#009e73", "s"),
    ], "Selector flip vs Bundle 3: active gradient MEASURED every round (measured_residual_response_v1, charged to S_alg) and "
       "candidates credited marginal gain only (joint_minus_active_only_supported_trust_v1). NOT one-axis vs Bundle 3: this lane "
       "cannot carry the three-term Qiskit cost or min-cost position realization (both abort at runtime), so Bundle 5 selects with "
       "no compile-cost term and plain commutation insertion. Weak rows final (k*/r45); strong rows [live], running to 75."),
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
import os as _osb
BUNDLE3_ONLY = bool(_osb.environ.get("BUNDLE3_ONLY"))
FINAL_B3_B4 = bool(_osb.environ.get("FINAL_B3_B4"))
if FINAL_B3_B4:
    # THE deliverable: page 1 = Bundle 3, page 2 = Bundle 4.
    PAGES = [p for p in PAGES if p[1] in ("3", "4", "5")]
    out = ("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/pdf/"
           "paper_i_ra_allphase_adaptive_20260817/paper_i_final_results.pdf")
elif BUNDLE3_ONLY:
    PAGES = [p for p in PAGES if p[1] == "3"]
    out = ("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/pdf/"
           "paper_i_ra_allphase_adaptive_20260817/bundle3_final_results.pdf")
else:
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
                comp_pts = completed.get((arm, regime))
                live_pts = live.get((arm, regime))
                if comp_pts and live_pts:
                    last = comp_pts[-1][0]
                    pts = comp_pts + [p for p in live_pts if p[0] > last]
                elif comp_pts:
                    pts = comp_pts
                else:
                    pts = live_pts
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
        WR = KT.get("weak_report_45_b5" if deliv == "5" else "weak_report_45", {}).get("regimes", {})
        WEAK_REGS = set(WR)
        def _rep45(regime, arm_key):
            """(k_label, dE, tuple, salg) for the r45 report side, or None."""
            if deliv not in ("3", "4", "5"):
                return None
            e = WR.get(regime, {}).get(arm_key)
            if not e and deliv == "5" and arm_key == "AAVQE":
                e = KT.get("weak_report_45", {}).get("regimes", {}).get(regime, {}).get(arm_key)
            if not e:
                return None
            return 45, e.get("dE"), e.get("tuple"), e.get("salg")
        rows, colors = [], []
        for regime in REGIMES:
            av = KT["aavqe"][regime]
            rep = _rep45(regime, "AAVQE") if deliv in ("3", "4", "5") and regime in WEAK_REGS else None
            if rep:
                k2, de2, tu2, sa2 = rep
                rows.append(["AAVQE", regime, _pair(av["k_star"], k2, "{}"),
                             _pair(av["dE_star"], de2),
                             _tup_pair(av["tuple_star"], tu2, "N2q"),
                             _tup_pair(av["tuple_star"], tu2, "D2q"),
                             _tup_pair(av["tuple_star"], tu2, "Dc"),
                             _pair(av["salg_star"], sa2, "{}")])
            else:
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
                if deliv == "5":
                    comp = KT.get("b5", {}).get(key)
                elif deliv == "4":
                    comp = KT.get("b4", {}).get(key)
                elif deliv == "3":
                    comp = b3t.get(key)
                elif deliv == "2":
                    comp = KT.get("b2", {}).get(key)
                if comp is None and (deliv == "1" or arm == "append_ra"):
                    comp = b1t.get(key)
                if comp is not None:
                    rep = _rep45(regime, arm) if deliv in ("3", "4", "5") and regime in WEAK_REGS else None
                    if rep:
                        k2, de2, tu2, sa2 = rep
                        salg_cell = _pair(comp["salg_star"], sa2, "{}")
                        if comp["salg_star"] is None and sa2 is None:
                            salg_cell = "in transit"
                        rows.append([label, regime,
                                     _pair(comp["k_star"], k2, "{}"),
                                     _pair(comp["dE_star"], de2),
                                     _tup_pair(comp["tuple_star"], tu2, "N2q"),
                                     _tup_pair(comp["tuple_star"], tu2, "D2q"),
                                     _tup_pair(comp["tuple_star"], tu2, "Dc"),
                                     salg_cell])
                    else:
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
        tab_ax = fig.add_axes([0.05, 0.048, 0.90, 0.295])
        tab_ax.axis("off")
        table = tab_ax.table(
            cellText=rows,
            colLabels=["series", "regime", "k* / k_term", "dE k*/term", "N2q k*/term",
                       "D2q k*/term", "Dc k*/term", "S_alg k*/term"],
            cellLoc="center", loc="upper center", bbox=[0.0, 0.0, 1.0, 1.0])
        table.auto_set_font_size(False)
        table.set_fontsize(5.6)
        for (r, c0), cell in table.get_celld().items():
            if r > 0:
                cell.set_facecolor(colors[r - 1])
        fig.text(0.05, 0.026,
                 "WEAK-sector rows report k*/r45 (author convention; AAVQE ran to 75, RA to 50, plateaus unaffected). Strong rows: k*/k_term. "
                 "k* = AAVQE plateau round: first round within 0.1 decade of its r75 terminal error, when entered by "
                 "round 60; weak_strong and intermediate_strong do not plateau, so k* = 75 (terminal) there. ALL cost "
                 "tuples: routed FakeMarrakesh ol=1 seed_transpiler=7 (incl. routing/SWAP).",
                 fontsize=6.5, color="0.35")
        fig.text(0.05, 0.012,
                 "Append arms: acceptance-order prefixes (their true ordering). Insertion arms: round-k* circuits "
                 "reconstructed by selected_position replay (exact). Live rows: k_term costs and S_alg land at "
                 "completion; matrix-lane scored marginals retained in archives.",
                 fontsize=6.5, color="0.35")
        if deliv == "3" and KT.get("weak_report_45"):
            wr = KT["weak_report_45"]
            fe = lambda v: ("-" if v is None else f"{v:.1e}")
            regs = wr["regimes"]
            l1 = (f"WEAK-SECTOR REPORT HOOK (manuscript): report at iteration {wr['k_rep']}. "
                  "AAVQE ran to 75 without breaking its plateau; RA ran to 50 with plateaus unaffected. dE@45 (ww/iw/swu8): "
                  f"AAVQE {fe(regs['weak_weak']['AAVQE']['dE'])}/{fe(regs['intermediate_weak']['AAVQE']['dE'])}/{fe(regs['strong_weak_u8']['AAVQE']['dE'])}, "
                  f"append {fe(regs['weak_weak'].get('b_depth_append_ra',{}).get('dE'))}/{fe(regs['intermediate_weak'].get('b_depth_append_ra',{}).get('dE'))}/{fe(regs['strong_weak_u8'].get('b_depth_append_ra',{}).get('dE'))}, "
                  f"always {fe(regs['weak_weak'].get('b_depth_costexp_always_open',{}).get('dE'))}/{fe(regs['intermediate_weak'].get('b_depth_costexp_always_open',{}).get('dE'))}/{fe(regs['strong_weak_u8'].get('b_depth_costexp_always_open',{}).get('dE'))}, "
                  f"plateau {fe(regs['weak_weak'].get('b_depth_costexp_plateau_position',{}).get('dE'))}/{fe(regs['intermediate_weak'].get('b_depth_costexp_plateau_position',{}).get('dE'))}/{fe(regs['strong_weak_u8'].get('b_depth_costexp_plateau_position',{}).get('dE'))}.")
            exc = wr["exception"]["dE_by_round_43_50"]
            l2 = ("EXCEPTION: always-insertion strong_weak_u8 continues past the report point: "
                  f"r43 {float(exc['43']):.1e} -> r45 {float(exc['45']):.1e} -> r46 {float(exc['46']):.1e} -> r47 {float(exc['47']):.1e}, terminal r50 {float(exc['50']):.1e}. "
                  "Plateau cells ran all 50 rounds; exit-holds = completion-kind guard only (accepted 2026-08-20). r45 tuples: weak_report_45.")
            fig.text(0.05, 0.368, l1, fontsize=6.6, color="#8b0000")
            fig.text(0.05, 0.356, l2, fontsize=6.6, color="#8b0000")
        elif note:
            import textwrap as _tw
            _y = 0.368
            for _ln in _tw.wrap(note, 205)[:3]:
                fig.text(0.05, _y, _ln, fontsize=6.6, color="#8b0000")
                _y -= 0.012
        fig.legend(loc="upper left", bbox_to_anchor=(0.055, 0.930), fontsize=8, ncol=2)
        shown = title
        if FINAL_B3_B4:
            shown = re.sub(r"^Page \d+ - ", f"Page {pageno} - ", title)
        fig.suptitle(f"{shown}  -  refreshed {stamp}", y=0.99, fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)
print("written:", out)
if BUNDLE3_ONLY or FINAL_B3_B4:
    import hashlib, subprocess
    def _sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    inputs = {
        "kstar_tables.json": f"{SCRATCH}/kstar_tables.json",
        "aavqe_full_hist_tolmatch_9665004.json": f"{LR}/paper_i_aavqe_r50_to_r75_tolmatch_20260820_v2/aavqe_full_hist_tolmatch_9665004.json",
        "aavqe_r75_cost_tuples_9665004.json": f"{LR}/paper_i_aavqe_r50_to_r75_tolmatch_20260820_v2/aavqe_r75_cost_tuples_9665004.json",
        "aavqe_salg_prefix_r75_9665004.json": f"{LR}/paper_i_aavqe_r50_to_r75_tolmatch_20260820_v2/aavqe_salg_prefix_r75_9665004.json",
        "live_samples_fresh.jsonl": f"{SCRATCH}/live_samples_fresh.jsonl",
        "queue_fresh.tsv (9664325)": f"{WT}/ra_adapt_costexp_depthfix_sixregime_20260820_v2_chtc/queue_fresh.tsv",
    }
    import glob as _gm
    _pats = [f"{SCRATCH}/matrix_outs/9664325.*-*.out"]
    if FINAL_B3_B4:
        _pats.append(f"{SCRATCH}/matrix_outs/9665210.*-*.out")
        _pats.append("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/local_runs/paper_i_hh_b3_measured_residual_20260821/logs/*.out")
        _pats.append("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/local_runs/paper_i_hh_b3_measured_residual_20260821/live_tails/*.tail")
    for _pat in _pats:
        for p in sorted(_gm.glob(_pat)):
            inputs[f"stream:{_osb.path.basename(p)}"] = p
    try:
        commit = subprocess.check_output(
            ["git", "-C", "/Users/jakestrobel/local_repos/Holstein_test_fullclone_3.worktrees/ra-refactor-stage2",
             "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    manifest = {
        "artifact": _osb.path.basename(out),
        "artifact_sha256": _sha(out),
        "bundle": ("page 1 = Bundle 3 (no-warm-start, yes-position-cost); page 2 = Bundle 4 (+ alpha* warm start); page 3 = Bundle 5 (measured residual, no cost term)"
                   if FINAL_B3_B4 else "Bundle 3 - Final RA paper I facing, no-warm-start, yes-position-cost"),
        "cluster": ([9664325, 9665210, 9674089] if FINAL_B3_B4 else 9664325),
        "generated": stamp,
        "renderer": "results_tooling/build_variant_pages_v7.py (" + ("FINAL_B3_B4=1" if FINAL_B3_B4 else "BUNDLE3_ONLY=1") + ")",
        "tooling_commit": commit,
        "inputs": {k: {"path": v, "sha256": _sha(v)} for k, v in inputs.items() if _osb.path.exists(v)},
        "conventions": {
            "cost_tuples": "routed FakeMarrakesh ol=1 seed_transpiler=7, incl. routing/SWAP",
            "insertion_kstar_ordering": "insertion_position_replay_v1 (exact round-k* circuits)",
            "salg": "accepted_prefix_work cumulative records (authenticated per-round ledger)",
        },
    }
    mpath = out.replace(".pdf", "_manifest.json")
    json.dump(manifest, open(mpath, "w"), indent=1)
    print("manifest:", mpath)
