"""Finish Bundle-1 append k75 recovery: shift check, table rows with
routed tuples at k*/k_term, terminal S_alg, and per-round S_alg from the
crash checkpoints (fast bracket walk). Tails were already replaced."""
import importlib.util, json, sys, re, glob, tarfile
from pathlib import Path

HERE = "/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/chtc/paper_i_ra_adapt_repair_20260727"
spec = importlib.util.spec_from_file_location("p1b", f"{HERE}/build_paper_i_ra_phase123_qiskit_comparison_pdf_20260817.py")
mod = importlib.util.module_from_spec(spec); sys.modules["p1b"] = mod; spec.loader.exec_module(mod)
exact = {r: v["exact_energy"] for r, v in mod.load_hist_plateau().items()}

S = Path("/private/tmp/claude-501/-Users-jakestrobel-Documents-Holstein-implementation-Holstein-test-fullclone-3/91434114-4541-47ef-95e4-2b8bd90bd929/scratchpad")
CD = Path("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/local_runs/paper_i_ra_allphase_adaptive_20260817_comparison_data")
V1 = Path("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/local_runs/paper_i_ra_allphase_adaptive_maximum_k75_extension_20260819/v1_crash_tarballs")
KT = json.load(open(S / "kstar_tables.json"))
KSTAR = KT["kstar"]
SALG_TERM = {"weak_strong": 850241, "intermediate_strong": 861886, "strong_strong_u8": 760909}

def fast_balanced_end(buf, i):
    depth = 0
    for m in re.finditer(rb"[\[\]]", buf, ):
        if m.start() < i:
            continue
        if m.group() == b"[":
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return m.end()
    return None

# streams from the replaced tails (raw AI_LOG lines with prefix)
streams = {}
for tail in sorted(CD.glob("live_tails/allphase_k75__append_ra__*.tail")):
    regime = re.search(r"append_ra__(.+?)__nph", tail.name).group(1)
    rows = {}
    for line in open(tail):
        if '"hardcoded_adapt_iter"' not in line:
            continue
        i = line.find("{")
        try:
            d = json.loads(line[i:])
        except ValueError:
            continue
        rows[int(d["depth"])] = (float(d["energy"]), d["best_op"].split("::")[-1])
    streams[regime] = rows
    print(regime, "tail depths", min(rows), "-", max(rows), flush=True)

# shift check vs bit-identical maxk50 prefix
shifts = {}
for regime, rows in streams.items():
    sp = CD / f"x_allphase_maxk50__append_ra__{regime}/run/summary/summary.json"
    tr = json.load(open(sp))["accepted_error_trace"]
    acc = {int(x["controller_round"]): float(x["accepted_energy"]) for x in tr}
    dm1 = abs(rows[51][0] - acc[50]) if 51 in rows and 50 in acc else float("inf")
    d0 = abs(rows[50][0] - acc[50]) if 50 in rows and 50 in acc else float("inf")
    shifts[regime] = 1 if dm1 <= d0 else 0
    print(f"{regime}: |E(d51)-E(r50)|={dm1:.2e} |E(d50)-E(r50)|={d0:.2e} -> shift={shifts[regime]}", flush=True)

# per-round S_alg probe (fast walk, bounded head)
salg_rounds = {}
for f in sorted(V1.glob("*.tar.gz")):
    regime = re.search(r"append_ra__(.+?)__nph", f.name).group(1)
    with tarfile.open(f, "r:gz") as tar:
        member = next(m for m in tar if m.name.endswith("run/checkpoints/current.json"))
        buf = tar.extractfile(member).read(192 << 20)
    m = re.search(rb'"active_prefix_checkpoints"\s*:\s*\[', buf)
    per = {}
    if m:
        end = fast_balanced_end(buf, m.end() - 1)
        if end:
            try:
                entries = json.loads(buf[m.end() - 1:end])
            except ValueError:
                entries = []
            for e in entries:
                rnd = e.get("controller_round") or e.get("active_ansatz_depth")
                sa = None
                o = e.get("estimator_prefix")
                if isinstance(o, dict):
                    sa = (o.get("cumulative_executed_queries") or {}).get("S_alg")
                if sa is None and isinstance(e.get("S_alg"), int):
                    sa = e["S_alg"]
                if rnd is not None and isinstance(sa, int):
                    per[int(rnd)] = sa
            if entries:
                print(regime, "prefix entries", len(entries), "keys", sorted(entries[0])[:8],
                      "salg points", len(per), flush=True)
    salg_rounds[regime] = per
    del buf

for regime, rows in streams.items():
    shift = shifts[regime]
    ks = KSTAR[regime]
    words = [rows[d][1] for d in sorted(rows)]
    if shift:
        words = words[1:]
    k_term = max(rows) - shift
    errs = {d - shift: max(abs(rows[d][0] - exact[regime]), 1e-16) for d in rows if d - shift >= 1}
    print(f"compiling {regime} k*={ks} k_term={k_term}", flush=True)
    entry = {
        "k_star": ks if ks in errs else None, "k_term": k_term,
        "dE_star": errs.get(ks), "dE_term": errs[k_term],
        "tuple_star": mod._routed_prefix_tuple(f"v1crash_append_{regime}", words, regime, ks, CD / "workdir") if ks <= len(words) else None,
        "tuple_term": mod._routed_prefix_tuple(f"v1crash_append_{regime}", words, regime, k_term, CD / "workdir"),
        "salg_star": salg_rounds.get(regime, {}).get(ks),
        "salg_term": SALG_TERM[regime],
    }
    KT["b1"][f"append_ra|{regime}"] = entry
    print("b1 updated:", regime, entry["tuple_star"], entry["tuple_term"],
          "salg*", entry["salg_star"], flush=True)

json.dump(KT, open(S / "kstar_tables.json", "w"), indent=1)
print("RECOVERY_V2_DONE", flush=True)
