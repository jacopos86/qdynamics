"""Complete b1 rows for cells whose only local record is the archive:
trace + accepted_history (labels + insertion positions) + prefix-work
S_alg, with replayed routed tuples at k* and k_term."""
import importlib.util, json, sys, re, glob, os, tarfile, gc
from pathlib import Path

HERE = "/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/chtc/paper_i_ra_adapt_repair_20260727"
spec = importlib.util.spec_from_file_location("p1b", f"{HERE}/build_paper_i_ra_phase123_qiskit_comparison_pdf_20260817.py")
mod = importlib.util.module_from_spec(spec); sys.modules["p1b"] = mod; spec.loader.exec_module(mod)
exact = {r: v["exact_energy"] for r, v in mod.load_hist_plateau().items()}

S = Path("/private/tmp/claude-501/-Users-jakestrobel-Documents-Holstein-implementation-Holstein-test-fullclone-3/91434114-4541-47ef-95e4-2b8bd90bd929/scratchpad")
CD = Path("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/local_runs/paper_i_ra_allphase_adaptive_20260817_comparison_data")
WORKDIR = CD / "workdir"
KT = json.load(open(S / "kstar_tables.json"))
KSTAR = KT["kstar"]
PLAIN = ("append_ra", "plateau_position_phase0", "always_open_position_phase0")

def balanced_slice(raw, anchor, open_b=b"[", close_b=b"]"):
    m = re.search(anchor, raw)
    if not m:
        return None
    i = raw.index(open_b, m.end() - 1)
    depth = 0
    for mm in re.finditer(b"[" + re.escape(open_b) + re.escape(close_b) + b"]", raw):
        if mm.start() < i:
            continue
        if mm.group() == open_b:
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return raw[i:mm.end()]
    return None

for f in sorted(glob.glob(str(CD / "archives/allphase_maxk50__*.tar.gz"))):
    m = re.match(r"allphase_maxk50__(.+?)__((?:weak|intermediate|strong)_\w+?)__nph\d", os.path.basename(f))
    if not m or m.group(1) not in PLAIN:
        continue
    arm, regime = m.group(1), m.group(2)
    key = f"{arm}|{regime}"
    if key in KT["b1"] and KT["b1"][key].get("salg_star") is not None:
        continue
    with tarfile.open(f, "r:gz") as tar:
        members = tar.getmembers()
        rj = [mm for mm in members if mm.name.endswith("result.json")
              and "source/" not in mm.name]
        sm = [mm for mm in members if mm.name.endswith("summary.json")
              and "source/" not in mm.name]
        if not rj:
            print("no result:", key, flush=True)
            continue
        raw = tar.extractfile(rj[0]).read()
        summary = json.loads(tar.extractfile(sm[0]).read()) if sm else None
    seg = balanced_slice(raw, rb'"accepted_history"\s*:\s*\[')
    hist = json.loads(seg) if seg else []
    seg2 = balanced_slice(raw, rb'"accepted_prefix_work"\s*:\s*\[')
    work = json.loads(seg2) if seg2 else []
    del raw; gc.collect()
    if not hist:
        print("no accepted_history:", key, flush=True)
        continue
    rows = {}
    for e in hist:
        rnd = int(e["controller_round"])
        rows[rnd] = (float(e["energy_after"]), e["selected_label"].split("::")[-1],
                     int(e.get("insertion_position", rnd - 1)))
    per_salg = {}
    for idx, e in enumerate(work):
        if isinstance(e, dict):
            sa = e.get("s_alg", e.get("S_alg"))
            if isinstance(sa, int):
                per_salg[int(e.get("controller_round", idx + 1))] = sa
    kt_ = max(rows)
    ks = KSTAR[regime]
    errs = {r: max(abs(rows[r][0] - exact[regime]), 1e-16) for r in rows}
    def order(k):
        seq = []
        for r in range(1, k + 1):
            _, w, p = rows[r]
            if p < 0 or p > len(seq):
                p = len(seq)
            seq.insert(p, w)
        return seq
    is_append = arm == "append_ra"
    def tup(k):
        if k > kt_ or any(r not in rows for r in range(1, k + 1)):
            return None
        words = [rows[r][1] for r in range(1, k + 1)] if is_append else order(k)
        return mod._routed_prefix_tuple(f"arch_{arm}_{regime}", words, regime, k, WORKDIR)
    KT["b1"][key] = {
        "k_star": ks if ks in rows else None, "k_term": kt_,
        "dE_star": errs.get(ks), "dE_term": errs[kt_],
        "tuple_star": tup(ks), "tuple_term": tup(kt_),
        "salg_star": per_salg.get(ks), "salg_term": per_salg.get(kt_),
        "trace": sorted(errs.items()),
        "ordering_method": "acceptance_order_exact_for_append" if is_append else "insertion_position_replay_v1",
    }
    print("b1 archive-filled:", key, "k*", ks, "kt", kt_,
          "salg", per_salg.get(ks), "/", per_salg.get(kt_), flush=True)

json.dump(KT, open(S / "kstar_tables.json", "w"), indent=1)
print("B1_ARCHIVE_DONE", flush=True)
