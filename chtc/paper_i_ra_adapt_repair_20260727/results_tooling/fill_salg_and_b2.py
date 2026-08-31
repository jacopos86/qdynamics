"""S_alg at k*/k_term for all completed cells (accepted_prefix_work), plus
full Bundle-2 rows (six completed weak-trio insertion cells: traces,
replayed tuples, S_alg)."""
import importlib.util, json, sys, re, glob, os, tarfile, gc
from pathlib import Path

HERE = "/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/chtc/paper_i_ra_adapt_repair_20260727"
spec = importlib.util.spec_from_file_location("p1b", f"{HERE}/build_paper_i_ra_phase123_qiskit_comparison_pdf_20260817.py")
mod = importlib.util.module_from_spec(spec); sys.modules["p1b"] = mod; spec.loader.exec_module(mod)
exact = {r: v["exact_energy"] for r, v in mod.load_hist_plateau().items()}

S = Path("/private/tmp/claude-501/-Users-jakestrobel-Documents-Holstein-implementation-Holstein-test-fullclone-3/91434114-4541-47ef-95e4-2b8bd90bd929/scratchpad")
CD = Path("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/local_runs/paper_i_ra_allphase_adaptive_20260817_comparison_data")
WORKDIR = CD / "workdir"
LR = Path("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/local_runs")
KT = json.load(open(S / "kstar_tables.json"))
KSTAR = KT["kstar"]

def salg_prefix_from_raw(raw):
    """{round: cumulative s_alg} from the accepted_prefix_work array."""
    m = re.search(rb'"accepted_prefix_work"\s*:\s*\[', raw)
    if not m:
        return {}
    i = m.end() - 1
    depth = 0
    end = None
    for mm in re.finditer(rb"[\[\]]", raw):
        if mm.start() < i:
            continue
        if mm.group() == b"[":
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                end = mm.end()
                break
    if end is None:
        return {}
    try:
        entries = json.loads(raw[i:end])
    except ValueError:
        return {}
    per = {}
    for idx, e in enumerate(entries):
        if not isinstance(e, dict):
            continue
        rnd = int(e.get("controller_round", idx + 1))
        sa = e.get("s_alg", e.get("S_alg"))
        if isinstance(sa, int):
            per[rnd] = sa
    return per

def read_archive(path):
    with tarfile.open(path, "r:gz") as tar:
        members = tar.getmembers()
        rj = [mm for mm in members if mm.name.endswith("result.json")
              and "/source/" not in mm.name and not mm.name.startswith("./source")
              and not mm.name.startswith("source/")]
        sm = [mm for mm in members if mm.name.endswith("summary.json")
              and "/source/" not in mm.name and not mm.name.startswith("./source")
              and not mm.name.startswith("source/")]
        summary = json.loads(tar.extractfile(sm[0]).read()) if sm else None
        raw = tar.extractfile(rj[0]).read() if rj else None
    return summary, raw

def stream_rows(out_path):
    rows = {}
    for line in open(out_path):
        if '"hardcoded_adapt_iter"' not in line:
            continue
        i = line.find("{")
        try:
            d = json.loads(line[i:])
        except ValueError:
            continue
        rows[int(d["depth"])] = (float(d["energy"]), d["best_op"].split("::")[-1],
                                 int(d.get("selected_position", -1)))
    return rows

def replay_order(rows, k):
    seq = []
    for r in range(1, k + 1):
        _, w, p = rows[r]
        if p < 0 or p > len(seq):
            p = len(seq)
        seq.insert(p, w)
    return seq

# ---- Bundle 2: six completed weak-trio insertion cells ----
KT.setdefault("b2", {})
for f in sorted(glob.glob(str(LR / "ra_adapt_costexp_depthfix_sixregime_20260820_v2/transfer/sixregime__a_costexp_*.tar.gz"))):
    m = re.search(r"sixregime__(.+?)__((?:weak|intermediate|strong)_\w+?)__nph\d", f)
    arm, regime = m.group(1), m.group(2)
    summary, raw = read_archive(f)
    if summary is None:
        print("SKIP no summary:", os.path.basename(f), flush=True)
        continue
    per_salg = salg_prefix_from_raw(raw) if raw else {}
    del raw; gc.collect()
    tr = summary["accepted_error_trace"]
    errs = {int(x["controller_round"]): max(float(x["absolute_energy_error"]), 1e-16) for x in tr}
    kt_ = max(errs)
    ks = KSTAR[regime]
    outs = glob.glob(str(S / f"matrix_outs/*sixregime__{arm}__{regime}__*.out"))
    rows = stream_rows(outs[0]) if outs else {}
    ok = all(k in rows for k in range(1, kt_ + 1))
    caw = summary.get("canonical_all_work") or {}
    entry = {
        "k_star": ks if ks in errs else None, "k_term": kt_,
        "dE_star": errs.get(ks), "dE_term": errs[kt_],
        "tuple_star": mod._routed_prefix_tuple(f"replay_{arm}_{regime}", replay_order(rows, ks), regime, ks, WORKDIR) if ok and ks <= kt_ else None,
        "tuple_term": mod._routed_prefix_tuple(f"replay_{arm}_{regime}", replay_order(rows, kt_), regime, kt_, WORKDIR) if ok else None,
        "salg_star": per_salg.get(ks),
        "salg_term": caw.get("s_alg", per_salg.get(kt_)),
        "trace": sorted(errs.items()),
        "ordering_method": "insertion_position_replay_v1",
    }
    KT["b2"][f"{arm}|{regime}"] = entry
    print("b2", arm, regime, "k*", ks, "salg", entry["salg_star"], "/", entry["salg_term"],
          entry["tuple_star"], entry["tuple_term"], flush=True)

# ---- fill salg_star (and missing salg_term) for b3/b4 trios ----
for section, pattern in (
    ("b3", str(LR / "ra_adapt_costexp_depthfix_sixregime_20260820_v2/transfer/sixregime__b_depth_append_ra__*.tar.gz")),
    ("b4", str(LR / "ra_adapt_warmstart_sixregime_20260820_v1/transfer/sixregime__c_ws_depth_append_ra__*.tar.gz")),
):
    for f in sorted(glob.glob(pattern)):
        m = re.search(r"sixregime__(.+?)__((?:weak|intermediate|strong)_\w+?)__nph\d", f)
        key = f"{m.group(1)}|{m.group(2)}"
        if key not in KT[section]:
            continue
        entry = KT[section][key]
        if entry.get("salg_star") is not None:
            continue
        summary, raw = read_archive(f)
        if raw is None:
            continue
        per_salg = salg_prefix_from_raw(raw)
        del raw; gc.collect()
        ks = KSTAR[m.group(2)]
        entry["salg_star"] = per_salg.get(ks)
        if entry.get("salg_term") is None:
            entry["salg_term"] = per_salg.get(entry["k_term"])
        print(section, key, "salg", entry["salg_star"], "/", entry["salg_term"], flush=True)

# ---- b1 cells with local archives: salg pairs ----
b1_archives = {}
for f in glob.glob(str(CD / "archives/allphase_maxk50__*.tar.gz")):
    m = re.match(r"allphase_maxk50__(.+?)__((?:weak|intermediate|strong)_\w+?)__nph\d", os.path.basename(f))
    if m and m.group(1) in ("append_ra", "plateau_position_phase0", "always_open_position_phase0"):
        b1_archives[f"{m.group(1)}|{m.group(2)}"] = f
for key, f in sorted(b1_archives.items()):
    if key not in KT["b1"]:
        continue
    entry = KT["b1"][key]
    if entry.get("salg_star") is not None and entry.get("salg_term") is not None:
        continue
    summary, raw = read_archive(f)
    if raw is None:
        print("b1 no result:", key, flush=True)
        continue
    per_salg = salg_prefix_from_raw(raw)
    del raw; gc.collect()
    regime = key.split("|")[1]
    ks = KSTAR[regime]
    if entry.get("salg_star") is None:
        entry["salg_star"] = per_salg.get(ks)
    if entry.get("salg_term") is None:
        entry["salg_term"] = per_salg.get(entry["k_term"])
    print("b1", key, "salg", entry["salg_star"], "/", entry["salg_term"], flush=True)

json.dump(KT, open(S / "kstar_tables.json", "w"), indent=1)
print("SALG_B2_DONE", flush=True)
