"""Bundle 6 ingest: all available cells, weak (nph3/50) and strong (nph7/75).

Same recovery route as B5: trace from the .out stream, per-round S_alg from the
archive checkpoint, cost tuples from exact insertion replay, plus the run's own
closure receipts. Writes KT["b6"].
"""
import importlib.util, json, sys, re, glob, os, tarfile
from pathlib import Path
import ijson

HERE = "/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/chtc/paper_i_ra_adapt_repair_20260727"
spec = importlib.util.spec_from_file_location("p1b", f"{HERE}/build_paper_i_ra_phase123_qiskit_comparison_pdf_20260817.py")
mod = importlib.util.module_from_spec(spec); sys.modules["p1b"] = mod; spec.loader.exec_module(mod)
exact = {r: v["exact_energy"] for r, v in mod.load_hist_plateau().items()}

CD = Path("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/local_runs/paper_i_ra_allphase_adaptive_20260817_comparison_data")
WORKDIR = CD / "workdir"
B6 = Path("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/local_runs/paper_i_hh_b6_mr_alpha_star_warmstart_20260822")
KTP = Path("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3.worktrees/ra-refactor-stage2/chtc/paper_i_ra_adapt_repair_20260727/results_tooling/kstar_tables.json")
RT = Path("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3.worktrees/ra-refactor-stage2/chtc/paper_i_ra_adapt_repair_20260727/results_tooling")

KT = json.load(open(KTP)); KSTAR = KT["kstar"]; KT.setdefault("b6", {})
src = (RT / "fill_completed_generalized.py").read_text()
ns = {}
for fn in ("checkpoint_salg_stream", "stream_rows", "replay_order"):
    m = re.search(rf"(def {fn}\(.*?)(?=\ndef |\npatterns)", src, re.S)
    exec(compile(m.group(1), fn, "exec"), {"re": re, "json": json, "tarfile": tarfile}, ns)
checkpoint_salg_stream, stream_rows, replay_order = ns["checkpoint_salg_stream"], ns["stream_rows"], ns["replay_order"]

CLOSURE = {
    "adapt_vqe.continuation.active_prefix_estimator_ledger_closure.passed": "ledger_closure_passed",
    "adapt_vqe.continuation.active_prefix_estimator_ledger_closure.status": "ledger_closure_status",
    "adapt_vqe.estimator_call_accounting.complete": "estimator_accounting_complete",
    "adapt_vqe.controller_measurement_work_summary.controller_numeric_validation.paper_i_table_work_status": "paper_i_table_work_status",
    "adapt_vqe.checkpoint_reason": "checkpoint_reason",
    "adapt_vqe.ansatz_depth": "checkpoint_ansatz_depth",
    "adapt_vqe.S_alg": "checkpoint_terminal_salg",
}
def receipts(tp):
    out = {}
    try:
        with tarfile.open(tp, "r:gz") as t:
            m = next(x for x in t if x.name.endswith("run/checkpoints/current.json"))
            for pre, ev, val in ijson.parse(t.extractfile(m)):
                if pre in CLOSURE and ev in ("string","number","boolean","integer","double"):
                    out.setdefault(CLOSURE[pre], val)
                if len(out) == len(CLOSURE): break
    except Exception as e:
        out["_error"] = f"{type(e).__name__}: {e}"
    return out

def ckpt_energy(tp):
    with tarfile.open(tp, "r:gz") as t:
        m = next(x for x in t if x.name.endswith("run/checkpoints/current.json"))
        for pre, ev, val in ijson.parse(t.extractfile(m)):
            if pre == "adapt_vqe.energy" and ev in ("number","double","integer"):
                return float(val)

done = 0
for arc in sorted(glob.glob(str(B6 / "transfer/*.tar.gz"))):
    m = re.search(r"b6mr__(mr_\w+?)__((?:weak|intermediate|strong)_\w+?)__nph(\d)", os.path.basename(arc))
    if not m: continue
    arm, regime, nph = m.group(1), m.group(2), int(m.group(3))
    key = f"{arm}|{regime}"
    if KT["b6"].get(key, {}).get("salg_term") is not None:
        print("skip (done):", key, flush=True); continue
    outs = glob.glob(str(B6 / f"logs/*__{arm}__{regime}__nph{nph}.out"))
    if not outs:
        print("no stream:", key, flush=True); continue
    # Pick by CONTENT, not filename order: a failed earlier cluster can leave a
    # zero-byte .out that sorts first and would silently look like no data.
    best, rows = None, {}
    for cand in outs:
        r = stream_rows(cand)
        if len(r) > len(rows):
            best, rows = cand, r
    kt_ = 50 if nph == 3 else 75
    if any(r not in rows for r in range(1, kt_ + 1)):
        print(f"incomplete {key}: {len(rows)}/{kt_}", flush=True); continue
    per = checkpoint_salg_stream(arc)
    errs = {k: max(abs(rows[k][0] - exact[regime]), 1e-16) for k in rows}
    ks = KSTAR[regime]; is_app = arm == "mr_append_ra"
    tup = lambda k: mod._routed_prefix_tuple(f"replay_b6_{arm}_{regime}", replay_order(rows, k, is_app), regime, k, WORKDIR)
    dE_star = errs[ks]; dE_term = errs[kt_]
    if ks == kt_:
        ce = ckpt_energy(arc)
        if ce is not None:
            dE_star = dE_term = max(abs(ce - exact[regime]), 1e-16)
    print(f"building tuples {key} (k*={ks}, term={kt_}) ...", flush=True)
    KT["b6"][key] = {
        "k_star": ks, "k_term": kt_, "dE_star": dE_star, "dE_term": dE_term,
        "tuple_star": tup(ks), "tuple_term": tup(ks) if ks == kt_ else tup(kt_),
        "salg_star": per.get(ks), "salg_term": per.get(kt_),
        "trace": sorted(errs.items()),
        "ordering_method": "acceptance_order_exact_for_append" if is_app else "insertion_position_replay_v1",
        "dE_energy_source": ("terminal accepted energy from checkpoint adapt_vqe.energy (k*==k_term)"
                             if ks == kt_ else "stream hardcoded_adapt_iter energy at k* (k*<k_term)"),
        "source": "B6 alpha-star warm start: stream trace + checkpoint S_alg receipts",
        "run_self_certification": receipts(arc),
    }
    json.dump(KT, open(KTP, "w"), indent=1)
    done += 1
    print(f"FILLED {key}: dE* {dE_star:.3e} salg {per.get(ks)}/{per.get(kt_)} "
          f"closure={KT['b6'][key]['run_self_certification'].get('ledger_closure_passed')}", flush=True)
print(f"FILL_B6_DONE ({done} newly filled, {len(KT['b6'])}/18 total)", flush=True)
