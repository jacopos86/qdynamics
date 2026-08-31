"""Extract/compile everything needed for the k*/k_term cost tables.

Writes kstar_tables.json:
  kstar[regime]
  aavqe[regime]  = {dE_star, dE_term, tuple_star, tuple_term, salg_star, salg_term}
  b1[(arm|regime)] = same fields (completed Bundle-1 cells, routed convention)
  b3[(arm|regime)] = same fields (completed Bundle-3 trio, matrix-lane convention)
"""
import importlib.util, json, sys, re, glob, os, tarfile, gc
from pathlib import Path

HERE = "/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/chtc/paper_i_ra_adapt_repair_20260727"
spec = importlib.util.spec_from_file_location("p1b", f"{HERE}/build_paper_i_ra_phase123_qiskit_comparison_pdf_20260817.py")
mod = importlib.util.module_from_spec(spec); sys.modules["p1b"] = mod; spec.loader.exec_module(mod)
exact = {r: v["exact_energy"] for r, v in mod.load_hist_plateau().items()}
REGIMES = list(mod.REGIMES)

CD = Path("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/local_runs/paper_i_ra_allphase_adaptive_20260817_comparison_data")
WORKDIR = CD / "workdir"
TOLM = Path("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/local_runs/paper_i_aavqe_r50_to_r75_tolmatch_20260820_v2")

hist = {r: [(int(k), max(abs(e - exact[r]), 1e-16), e) for k, e in rows]
        for r, rows in json.load(open(TOLM / "aavqe_full_hist_tolmatch_9665004.json")).items()}
kstar = {}
for r, rows in hist.items():
    term = rows[-1][1]
    entry = next((k for k, e, _ in rows if e <= term * 10 ** 0.1), 75)
    # a sector only "demonstrates a plateau" if the band is entered well
    # before the horizon; otherwise the matched point is the terminal.
    kstar[r] = entry if entry <= 60 else 75

# ---- AAVQE words r1-75 (accepted_operator_labels) from the archives ----
def balanced_list(raw, start):
    i = raw.index(b"[", start); depth = 0
    for k in range(i, len(raw)):
        c = raw[k:k+1]
        if c == b"[": depth += 1
        elif c == b"]":
            depth -= 1
            if depth == 0: return raw[i:k+1]
    raise ValueError

words_file = TOLM / "aavqe_words_r75_9665004.json"
if words_file.is_file():
    aavqe_words = json.load(open(words_file))
else:
    aavqe_words = {}
    for f in sorted(glob.glob(str(TOLM / "transfer/*.tar.gz"))):
        regime = re.search(r"aavqe_r75cont__(.+?)__nph", f).group(1)
        with tarfile.open(f, "r:gz") as tar:
            member = next(m for m in tar if m.name.endswith("run/result.json"))
            raw = tar.extractfile(member).read()
        m = re.search(rb'"accepted_operator_labels"\s*:', raw)
        labels = json.loads(balanced_list(raw, m.end()))
        del raw; gc.collect()
        assert len(labels) == 75, (regime, len(labels))
        aavqe_words[regime] = [lb.split("::")[-1] for lb in labels]
        print("words", regime, len(aavqe_words[regime]), flush=True)
    json.dump(aavqe_words, open(words_file, "w"))

# per-round S_alg r1-50 caches (anchor lane)
salg_prefix = {}
for r in REGIMES:
    p = WORKDIR / f"append_salg_prefix_{r}.json"
    salg_prefix[r] = {int(k): v for k, v in json.load(open(p)).items()} if p.is_file() else {}
aavqe_cost_term = json.load(open(TOLM / "aavqe_r75_cost_tuples_9665004.json"))

out = {"kstar": kstar, "aavqe": {}, "b1": {}, "b3": {}}
for r in REGIMES:
    ks = kstar[r]
    errs = {k: e for k, e, _ in hist[r]}
    tup_star = mod._routed_prefix_tuple(f"aavqe_r75_{r}", aavqe_words[r], r, ks, WORKDIR)
    tup_term = mod._routed_prefix_tuple(f"aavqe_r75_{r}", aavqe_words[r], r, 75, WORKDIR)
    out["aavqe"][r] = {
        "k_star": ks, "k_term": 75,
        "dE_star": errs[ks], "dE_term": errs[75],
        "tuple_star": tup_star, "tuple_term": tup_term,
        "salg_star": salg_prefix[r].get(ks),
        "salg_term": aavqe_cost_term[r]["S_alg"],
    }
    print("aavqe", r, "k*", ks, tup_star, tup_term, flush=True)

# ---- Bundle-1 completed cells: words_cache + routed compiles ----
for prefix in ("x_allphase_maxk50__", "x_allphase_k75__"):
    for cd in sorted(glob.glob(str(CD / (prefix + "*")))):
        name = os.path.basename(cd)[len(prefix):]
        for reg in REGIMES:
            if name.endswith("__" + reg):
                arm, regime = name[: -(len(reg) + 2)], reg
                break
        else:
            continue
        if arm not in ("append_ra", "plateau_position_phase0", "always_open_position_phase0"):
            continue
        sp = os.path.join(cd, "run/summary/summary.json")
        wc = os.path.join(cd, "words_cache.json")
        if not (os.path.exists(sp) and os.path.exists(wc)):
            continue
        tr = json.load(open(sp)).get("accepted_error_trace") or []
        if not tr:
            continue
        words = json.load(open(wc))
        if isinstance(words, dict):
            words = words.get("words") or []
        errs = {int(x["controller_round"]): max(float(x["absolute_energy_error"]), 1e-16) for x in tr}
        kt = max(errs)
        ks = kstar[regime]
        cell = os.path.basename(cd)
        entry = {"k_star": ks if ks in errs else None, "k_term": kt,
                 "dE_star": errs.get(ks), "dE_term": errs[kt],
                 "tuple_star": None, "tuple_term": None,
                 "salg_star": None, "salg_term": None}
        if words:
            if ks <= len(words) and ks in errs:
                entry["tuple_star"] = mod._routed_prefix_tuple(cell, words, regime, ks, WORKDIR)
            if kt <= len(words):
                entry["tuple_term"] = mod._routed_prefix_tuple(cell, words, regime, kt, WORKDIR)
        out["b1"][f"{arm}|{regime}"] = entry
        print("b1", arm, regime, "k*", entry["k_star"], "kt", kt,
              entry["tuple_star"], entry["tuple_term"], flush=True)

# ---- Bundle-3 completed trio: per-round marginals from their own results ----
def last_series(raw, key):
    return [int(x) for x in re.findall(rb'"' + key.encode() + rb'"\s*:\s*(\d+)', raw)]

for f in sorted(glob.glob("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/local_runs/ra_adapt_costexp_depthfix_sixregime_20260820_v2/transfer/sixregime__b_depth_append_ra__*.tar.gz")):
    m = re.search(r"sixregime__(.+?)__((?:weak|intermediate|strong)_\w+?)__nph\d", f)
    arm, regime = m.group(1), m.group(2)
    with tarfile.open(f, "r:gz") as tar:
        members = tar.getmembers()
        rj = [mm for mm in members if mm.name.endswith("result.json")
              and "/source/" not in mm.name and not mm.name.startswith("./source")]
        sm = [mm for mm in members if mm.name.endswith("run/summary/summary.json")]
        if not sm:
            print("SKIP (no summary)", f, flush=True)
            continue
        summary = json.loads(tar.extractfile(sm[0]).read())
        if rj:
            raw = tar.extractfile(rj[0]).read()
            n2q = last_series(raw, "compiled_two_qubit_count")
            d2q = last_series(raw, "compiled_two_qubit_depth")
            dc = last_series(raw, "compiled_total_depth")
            del raw; gc.collect()
        else:
            print("NO result.json in", f, "- summary-only row", flush=True)
            n2q = d2q = dc = []
    tr = summary["accepted_error_trace"]
    errs = {int(x["controller_round"]): max(float(x["absolute_energy_error"]), 1e-16) for x in tr}
    kt = max(errs)
    ks = kstar[regime]
    # align: last entry of each series = round kt
    def at(series, k):
        idx = len(series) - 1 - (kt - k)
        return series[idx] if 0 <= idx < len(series) else None
    caw = summary.get("canonical_all_work") or {}
    out["b3"][f"{arm}|{regime}"] = {
        "k_star": ks if ks in errs else None, "k_term": kt,
        "dE_star": errs.get(ks), "dE_term": errs[kt],
        "tuple_star": ({"N2q": at(n2q, ks), "D2q": at(d2q, ks), "Dc": at(dc, ks)}
                       if ks in errs else None),
        "tuple_term": {"N2q": at(n2q, kt), "D2q": at(d2q, kt), "Dc": at(dc, kt)},
        "salg_star": None, "salg_term": caw.get("s_alg"),
    }
    print("b3", arm, regime, out["b3"][f"{arm}|{regime}"], flush=True)

json.dump(out, open("/private/tmp/claude-501/-Users-jakestrobel-Documents-Holstein-implementation-Holstein-test-fullclone-3/91434114-4541-47ef-95e4-2b8bd90bd929/scratchpad/kstar_tables.json", "w"), indent=1)
print("DONE kstar_tables.json", flush=True)
