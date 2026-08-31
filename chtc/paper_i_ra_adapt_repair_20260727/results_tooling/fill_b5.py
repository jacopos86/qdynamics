"""Bundle 5 (B3-MR, measured residual, cluster 9674089) ingest.

Weak cells completed all 50 rounds; their result.json was lost to a
worker receipt-path crash, so: trace from the AP .out streams, per-round
S_alg from the archive checkpoints, cost tuples from exact insertion
replay. Writes KT["b5"] and KT["weak_report_45_b5"]."""
import importlib.util, json, sys, re, glob, os
from pathlib import Path

HERE = "/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/chtc/paper_i_ra_adapt_repair_20260727"
spec = importlib.util.spec_from_file_location("p1b", f"{HERE}/build_paper_i_ra_phase123_qiskit_comparison_pdf_20260817.py")
mod = importlib.util.module_from_spec(spec); sys.modules["p1b"] = mod; spec.loader.exec_module(mod)
exact = {r: v["exact_energy"] for r, v in mod.load_hist_plateau().items()}

S = Path("/private/tmp/claude-501/-Users-jakestrobel-Documents-Holstein-implementation-Holstein-test-fullclone-3/91434114-4541-47ef-95e4-2b8bd90bd929/scratchpad")
CD = Path("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/local_runs/paper_i_ra_allphase_adaptive_20260817_comparison_data")
WORKDIR = CD / "workdir"
B5 = Path("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/local_runs/paper_i_hh_b3_measured_residual_20260821")
KT = json.load(open(S / "kstar_tables.json"))
KSTAR = KT["kstar"]

sys.path.insert(0, str(S))
fg_spec = importlib.util.spec_from_file_location("fillgen", str(S / "fill_completed_generalized.py"))
# fill_completed_generalized runs at import; reuse its helpers via source exec of just the functions instead
src = open(S / "fill_completed_generalized.py").read()
ns = {}
for fn in ("checkpoint_salg_stream", "stream_rows", "replay_order"):
    m = re.search(rf"(def {fn}\(.*?)(?=\ndef |\npatterns)", src, re.S)
    exec(compile(m.group(1), fn, "exec"), {"re": re, "json": json, "tarfile": __import__("tarfile")}, ns)
checkpoint_salg_stream, stream_rows, replay_order = ns["checkpoint_salg_stream"], ns["stream_rows"], ns["replay_order"]

KT.setdefault("b5", {})
KT.setdefault("weak_report_45_b5", {"regimes": {}})
for out in sorted(glob.glob(str(B5 / "logs/*nph3.out"))):
    m = re.search(r"b3mr__(mr_\w+?)__((?:weak|intermediate|strong)_\w+?)__nph3", os.path.basename(out))
    arm, regime = m.group(1), m.group(2)
    key = f"{arm}|{regime}"
    rows = stream_rows(out)
    if not rows or any(k not in rows for k in range(1, 51)):
        print("incomplete stream:", key); continue
    arch = glob.glob(str(B5 / f"transfer/b3mr__{arm}__{regime}__nph3__*.tar.gz"))
    per = checkpoint_salg_stream(arch[0]) if arch else {}
    is_append = arm == "mr_append_ra"
    errs = {k: max(abs(rows[k][0] - exact[regime]), 1e-16) for k in rows}
    ks, kt_, kr = KSTAR[regime], 50, 45
    def tup(k):
        return mod._routed_prefix_tuple(f"replay_b5_{arm}_{regime}", replay_order(rows, k, is_append), regime, k, WORKDIR)
    KT["b5"][key] = {
        "k_star": ks, "k_term": kt_,
        "dE_star": errs[ks], "dE_term": errs[kt_],
        "tuple_star": tup(ks), "tuple_term": tup(kt_),
        "salg_star": per.get(ks), "salg_term": per.get(kt_),
        "trace": sorted(errs.items()),
        "ordering_method": "acceptance_order_exact_for_append" if is_append else "insertion_position_replay_v1",
        "source": "B3-MR partial archive: stream trace + checkpoint S_alg receipts (worker receipt crash, science complete)",
    }
    KT["weak_report_45_b5"]["regimes"].setdefault(regime, {})[arm] = {
        "dE": errs[kr], "tuple": tup(kr), "salg": per.get(kr),
    }
    print(f"b5 filled {key}: dE* {errs[ks]:.1e} dE45 {errs[kr]:.1e} dE50 {errs[kt_]:.1e} salg {per.get(ks)}/{per.get(kr)}/{per.get(kt_)}", flush=True)

json.dump(KT, open(S / "kstar_tables.json", "w"), indent=1)
print("FILL_B5_DONE")
