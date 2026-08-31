#!/usr/bin/env python3
"""One page per bundle: six regime convergence panels + per-series results table.

Self-contained: reads kstar_tables.json (worktree copy) for ingested bundles and
the local .out streams for bundles still in flight. No dependency on deleted
session scratchpads (the reason build_variant_pages_v7.py can no longer run).

Usage: python3 build_bundle_pages_v8.py [--only 3,5]
"""
import argparse, datetime, glob, importlib.util, json, os, re, sys
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt

LR = "/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/local_runs"
KTP = ("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3.worktrees/ra-refactor-stage2/"
       "chtc/paper_i_ra_adapt_repair_20260727/results_tooling/kstar_tables.json")
OUT = ("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/pdf/"
       "paper_i_ra_allphase_adaptive_20260817/paper_i_bundle_pages_20260823.pdf")
HERE = "/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/chtc/paper_i_ra_adapt_repair_20260727"

spec = importlib.util.spec_from_file_location(
    "p1b", f"{HERE}/build_paper_i_ra_phase123_qiskit_comparison_pdf_20260817.py")
mod = importlib.util.module_from_spec(spec); sys.modules["p1b"] = mod; spec.loader.exec_module(mod)
exact = {r: v["exact_energy"] for r, v in mod.load_hist_plateau().items()}
REGIMES = list(mod.REGIMES)

KT = json.load(open(KTP))
KSTAR = KT["kstar"]

_LS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_status.json")
LIVE = json.load(open(_LS)) if os.path.exists(_LS) else {}
STATE_NAMES = {1: "idle - not started", 2: "RUNNING", 4: "completed", 5: "held"}

def cell_state(cluster, arm, regime):
    """Live queue state for one cell, resolved through the cluster's execution ids.

    cluster may be a list when a bundle spans an original cluster plus a
    resubmission (Bundle 3's strong insertion cells live in 9675078).
    """
    for c in (cluster if isinstance(cluster, (list, tuple)) else [cluster]):
        r = _cell_state_one(str(c), arm, regime)
        if r:
            return r
    return None


def _cell_state_one(cluster, arm, regime):
    d = LIVE.get(str(cluster))
    if not d:
        return None
    for proc, eid in d["eids"].items():
        if f"__{arm}__{regime}__" in eid:
            st = d["status"].get(str(proc)) or d["status"].get(proc)
            if not st:
                return "left queue (completed)"
            code = st["status"]
            if code == 5:
                mem, req = st.get("mem"), st.get("req")
                try:
                    if mem not in (None, "undefined") and int(mem) > int(req) * 0.95:
                        return f"held - OOM ({mem}/{req} MB)"
                except (TypeError, ValueError):
                    pass
                try:
                    return "held - finished (exit)" if int(mem) > 512 else "held - crashed early"
                except (TypeError, ValueError):
                    return "held"
            if code == 2:
                return f"RUNNING (peak {st.get('mem')} MB)"
            return STATE_NAMES.get(code, str(code))
    return None

aavqe = {r: [(k, max(abs(e - exact[r]), 1e-16)) for k, e in rows]
         for r, rows in json.load(open(
             f"{LR}/paper_i_aavqe_r50_to_r75_tolmatch_20260820_v2/"
             "aavqe_full_hist_tolmatch_9665004.json")).items()}

ORANGE, GREEN, PINK = "#d55e00", "#009e73", "#cc79a7"

LIVETAILS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "livetails")


def live_trace(cluster, arm, regime):
    """Partial trace for a cell that is still RUNNING.

    Running jobs never have a .out on the access point (stream_output=False),
    so their progress is only visible through condor_tail. fetch_live_tails.sh
    captures it; this reads what it captured, so an in-flight cell plots its
    rounds so far instead of showing nothing.
    """
    for c in (cluster if isinstance(cluster, (list, tuple)) else [cluster]):
        d = LIVE.get(str(c))
        if not d:
            continue
        for proc, eid in d["eids"].items():
            if f"__{arm}__{regime}__" not in eid:
                continue
            st = d["status"].get(str(proc)) or d["status"].get(proc)
            if not st or st.get("status") != 2:
                continue
            path = os.path.join(LIVETAILS, f"{c}.{proc}.out")
            if not os.path.exists(path):
                continue
            rows = {}
            for line in open(path, errors="ignore"):
                if '"hardcoded_adapt_iter"' not in line:
                    continue
                i = line.find("{")
                try:
                    rec = json.loads(line[i:])
                except ValueError:
                    continue
                rows[int(rec["depth"])] = max(
                    abs(float(rec["energy"]) - exact[regime]), 1e-16
                )
            if rows:
                return sorted(rows.items())
    return None


def stream_trace(pattern, regime):
    """Per-round dE from a live .out stream (bundles not yet ingested)."""
    hits = glob.glob(pattern)
    if not hits:
        return None
    rows = {}
    for line in open(hits[0], errors="ignore"):
        if '"hardcoded_adapt_iter"' not in line:
            continue
        i = line.find("{")
        try:
            d = json.loads(line[i:])
        except ValueError:
            continue
        rows[int(d["depth"])] = max(abs(float(d["energy"]) - exact[regime]), 1e-16)
    return sorted(rows.items()) or None

# (page title, KT section or None, cluster id, [(arm,label,colour,marker)], stream glob, note)
PAGES = [
    ("Bundle 3 [SINGLETON pool]: Paper-I facing - stationary, flat 948/6508 singleton pool, "
     "sealed Qiskit cost in selection (2q count + 2q depth + circuit depth) (cluster 9664325)", "b3", ["9664325", "9675078"], [
        ("b_depth_append_ra", "append-RA", ORANGE, "^"),
        ("b_depth_costexp_always_open", "always-insertion RA", PINK, "o"),
        ("b_depth_costexp_plateau_position", "plateau-insertion RA", GREEN, "s")], None,
     "Executes the pool traversal Paper I describes: the deduplicated flat Pauli-word pool screened "
     "through Phases 0-III, identity exposure, no in-pipeline splitting. Strong-sector insertion cells "
     "never completed. NOT resumable - condor_rm destroyed the sandboxes before any checkpoint "
     "transferred, so they were RESUBMITTED FRESH 2026-08-23 as cluster 9675078 using B3's own frozen "
     "protocols (identical config, horizon 75). Their state is shown in the table below."),

    ("Bundle 4 [SINGLETON pool]: Bundle 3 + alpha* candidate warm start from the Phase-III "
     "scored optimum (cluster 9665210)", "b4", "9665210", [
        ("c_ws_depth_append_ra", "append-RA (warm start)", ORANGE, "^"),
        ("c_ws_depth_costexp_always_open", "always-insertion RA (warm start)", PINK, "o"),
        ("c_ws_depth_costexp_plateau_position", "plateau-insertion RA (warm start)", GREEN, "s")], None,
     "Warm-start gate: 50 seeded admissions / 0 cold fallbacks in every completed cell. On matched "
     "completed comparisons (append arm, weak trio) Bundle 4 equals Bundle 3 accuracy at k* to 7+ digits "
     "at 2.8-5.8% lower S_alg. Cluster cancelled 2026-08-22; rows are those that completed."),

    ("Bundle 5 (= 'b3mr') [MACRO-to-SINGLETON pool]: stationarity dropped - measured residual; "
     "~123 macro parents split to singleton children after Phase I (cluster 9674089)", "b5", "9674089", [
        ("mr_append_ra", "append-RA (measured residual)", ORANGE, "^"),
        ("mr_always_insertion", "always-insertion RA (measured residual)", PINK, "o"),
        ("mr_plateau_insertion", "plateau-insertion RA (measured residual)", GREEN, "s")], None,
     "COMPLETE 18/18, all cells self-certified (ledger closure passed, paper_i_table_work_status ok). "
     "ROUTE CAVEAT: not one-axis vs Bundle 3. Phases 0-I shortlist ~123 MACRO parents which are split into "
     "Pauli children between Phase I and Phase II (split_parent_count 6-8, split_child_count 24-69), where "
     "Paper I specifies the flat 948/6508 pool screened throughout with no in-pipeline construction. It also "
     "carries no compile-cost term and plain commutation insertion. S_alg values are duplication-corrected."),

    ("Bundle 6 [MACRO-to-SINGLETON pool]: Bundle 5 + alpha* Schur candidate warm start (cluster 9674592)", "b6", "9674592", [
        ("mr_append_ra", "append-RA (+alpha*)", ORANGE, "^"),
        ("mr_always_insertion", "always-insertion RA (+alpha*)", PINK, "o"),
        ("mr_plateau_insertion", "plateau-insertion RA (+alpha*)", GREEN, "s")],
     None,
     "One axis vs Bundle 5: candidate coordinate seeded at +/-alpha* (Schur trust step) instead of zero. "
     "Warm start seeded in every round of every lane, no guard rejections. COMPLETE 18/18, all cells self-certified. "
     "MEASURED RESULT over the full matrix: alpha* costs modestly MORE estimator work on median - S_alg at k* worse in "
     "12/18 cells (median 1.049) and at terminal worse in 14/18 (median 1.068) - so it does not pay for itself. Accuracy is "
     "genuinely mixed rather than uniformly worse: dE at k* is unchanged in the weak sector but swings both ways in the "
     "strong sector (always W-S 0.23x better, append I-S 2.67x worse). Inherits Bundle 5's macro-parent route, so it is "
     "not Paper-I faithful regardless."),

    ("Bundle 7 [SINGLETON pool]: measured residual on the flat Paper-I singleton pool "
     "(the pool AVQE scans) - GRAPH-PROXY cost (cluster 9675059)", None, "9675059", [
        ("mr_append_ra", "append-RA (flat pool)", ORANGE, "^"),
        ("mr_always_insertion", "always-insertion RA (flat pool)", PINK, "o"),
        ("mr_plateau_insertion", "plateau-insertion RA (flat pool)", GREEN, "s")],
     LR + "/paper_i_hh_b7_mr_flat_singleton_20260822/logs/*__{arm}__{regime}__nph*.out",
     "REMOVED 2026-08-23 at the user's instruction; its two running cells were destroyed by condor_rm rather than vacated, so ~16h was lost with no archive. Not resubmitted. Rows below are the state at removal. Fixed Bundle 5's route defect: explicit GlobalSinglePauliWordCandidateAdapter puts the "
     "flat 948/6508 pool into Phase 0 with identity exposure (verified: split_parent_count 0). COST MODEL: graph proxy "
     "(marrakesh_graph_span_v1) - an analytic Steiner-span estimate of c_hat_2q/c_hat_d/c_hat_1q, NO transpilation, weights "
     "1.0 dN2q + 0.1 dD + 0.01 size from the oracle config. This is NOT cost-free: AVQE carries no cost policy, backend cost "
     "mode or weights at all, so B7 vs AVQE differs on cost awareness as well as on the selector. In flight; live traces."),

    ("Bundle 8 [SINGLETON pool]: Bundle 7 with TRANSPILED-QISKIT-MARGINAL cost + min-cost positions (cluster 9675127)", None, "9675127", [
        ("mr_append_ra", "append-RA (flat pool + cost)", ORANGE, "^"),
        ("mr_always_insertion", "always-insertion RA (flat pool + cost)", PINK, "o"),
        ("mr_plateau_insertion", "plateau-insertion RA (flat pool + cost)", GREEN, "s")],
     LR + "/paper_i_hh_b8_mr_flat_qiskit_cost_20260822/logs/*__{arm}__{regime}__nph*.out",
     "SUPERSEDED BY BUNDLE 9 - DO NOT USE AS EXACT-COST EVIDENCE. B8 was intended as 'Bundle 3 with stationarity off', but its completed cells scored on the graph-span PROXY while recording an exact-transpile cost policy: read back from cluster 9675127's append checkpoint, compile_cost_source is marrakesh_graph_span_v1 in all 100 occurrences and backend_transpile_v1 appears zero times. The label said exact marginals; the numbers were the analytic Steiner span. Cause: whether the Qiskit cost is routed was decided by identity-set membership rather than by the declared cost axis, and B8's identity (paper_i_ra_adapt_nonstationary_incremental_full_response_v2) is in none of those sets, so the entire Qiskit block was skipped. Its twelve min-cost insertion cells additionally died in Phase I with 'Cost-expanded Qiskit member lost a routed penalty input', because cost-expanded realization needs the all-phase transpile route that this lane could not carry. Cluster held 2026-08-24; three append cells completed but carry proxy numbers. Bundle 9 is the corrected campaign."),

    ("Bundle 9 [SINGLETON pool]: measured residual + ALL-PHASE exact Qiskit transpile cost "
     "+ min-cost positions - the campaign Bundle 8 was meant to be (continuation cluster 9679133)",
     "b9", ["9679133", "9675590", "9675584"], [
        ("b_depth_append_ra", "append-RA (measured, exact cost)", ORANGE, "^"),
        ("b_depth_costexp_always_open", "always-insertion RA (min-cost)", PINK, "o"),
        ("b_depth_costexp_plateau_position", "plateau-insertion RA (min-cost)", GREEN, "s")],
     LR + "/paper_i_hh_b9_mr_measured_costexp_20260824/logs/*__{arm}__{regime}__nph*.out",
     "Submitted 2026-08-24 after removing an artificial restriction that confined semantic-closure routes "
     "to a stationary gradient. That restriction had no recorded rationale and confined the ONLY all-phase "
     "Qiskit implementation - and min-cost realization with it - to stationary lanes, which is why Bundle 8 "
     "declared an exact-cost policy and silently scored on the graph-span proxy. B9 runs B3's route with "
     "stationarity off: verified backend_transpile_v1 in phases I/II/III (NOT the proxy), "
     "explicit_depth_qiskit_cost_v1 at the sealed 0.30 dN2q + 0.30 dD2q + 0.25 dD weights, and "
     "termwise_cross_component_commutation_min_cost_realization_v1 on both insertion arms. "
     "EARLY-TERMINATION CAVEAT: the first wave stopped at depth 34-40 with 'no positive feasible candidate' "
     "- with exact compiled cost subtracted, no candidate paid for itself once the measured residual "
     "converged (grad 1e-5..1e-7). All such cells still self-certified (ledger closure passed, work_status ok) "
     "and their archives are valid evidence. The measured route variants were missing from "
     "PAPER_I_RA_PHASE3_FORCED_ADMISSION_ROUTE_VARIANTS, so they fell through to the RAISE policy instead of "
     "forced admission; that is fixed and the affected cells are being continued to the full horizon."),

]

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="", help="comma list of bundle numbers, e.g. 3,5")
args = ap.parse_args()
want = {s.strip() for s in args.only.split(",") if s.strip()}
pages = [p for p in PAGES if not want or re.search(r"Bundle (\d)", p[0]).group(1) in want]

stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M CDT")
def fmt_pair(a, b, f="{:.1e}"):
    g = lambda v: "-" if v is None else (f.format(v) if isinstance(v, float) else str(v))
    return f"{g(a)} / {g(b)}"


def fmt_k45(ent, regime):
    """Weak-sector reporting point: dE and template 2q count at iteration 45.

    AAVQE runs to 75 and RA to 50 in the weak sector, so 45 is the deepest
    round both reach with plateaus unaffected. Blank in the strong sector.
    """
    if regime not in ("weak_weak", "intermediate_weak", "strong_weak_u8"):
        return ""
    if not ent:
        return "-"
    dE = ent.get("dE_45")
    c2q = (ent.get("tuple_45_proxy") or {}).get("c2q")
    if dE is None and c2q is None:
        return "-"
    return f"{'-' if dE is None else f'{dE:.1e}'} | ({'-' if c2q is None else c2q})"


def fmt_dual(ent, field, proxy_field):
    """compiled star/term | (template star/term).

    The bracketed pair is the deterministic template surrogate, not a
    transpiled measurement: it carries no routing/seed variance.
    """
    if not ent:
        return "- / -"
    q = fmt_pair((ent.get("tuple_star") or {}).get(field),
                 (ent.get("tuple_term") or {}).get(field), "{}")
    p_ = fmt_pair((ent.get("tuple_star_proxy") or {}).get(proxy_field),
                  (ent.get("tuple_term_proxy") or {}).get(proxy_field), "{}")
    return f"{q} | ({p_})"

