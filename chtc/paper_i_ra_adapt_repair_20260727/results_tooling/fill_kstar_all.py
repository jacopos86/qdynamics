"""Compile k* Qiskit tuples (+ available S_alg) for every RA cell whose
trajectory has passed its regime's k*, and full rows for newly-completed
cells (Bundle-4 append trio). Updates kstar_tables.json in place."""
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

def parse_line(line):
    if '"hardcoded_adapt_iter"' not in line:
        return None
    i = line.find("{")
    if i < 0:
        return None
    try:
        d = json.loads(line[i:])
    except ValueError:
        return None
    return int(d["depth"]), float(d["energy"]), d["best_op"].split("::")[-1]

# ---- matrix streams: .out write-backs, fresh harvest wins on overlap ----
streams = {}
for f in sorted(glob.glob(str(S / "matrix_outs/*.out"))):
    m = re.search(r"sixregime__(.+?)__((?:weak|intermediate|strong)_\w+?)__nph\d", os.path.basename(f))
    if not m:
        continue
    key = (m.group(1), m.group(2))
    rows = streams.setdefault(key, {})
    for line in open(f):
        p = parse_line(line)
        if p:
            rows[p[0]] = (p[1], p[2])
eid_map = {}
WT = "/Users/jakestrobel/local_repos/Holstein_test_fullclone_3.worktrees/ra-refactor-stage2/chtc/paper_i_ra_adapt_repair_20260727"
for cluster, qf in ((9664325, f"{WT}/ra_adapt_costexp_depthfix_sixregime_20260820_v2_chtc/queue_fresh.tsv"),
                    (9665210, f"{WT}/ra_adapt_warmstart_sixregime_20260820_v1_chtc/queue_fresh.tsv")):
    for proc, line in enumerate(open(qf)):
        eid_map[(cluster, proc)] = line.split("\t")[0].strip()
for line in open(S / "live_samples_fresh.jsonl"):
    try:
        d = json.loads(line)
    except ValueError:
        continue
    if d["cluster"] == 9664259:
        continue
    eid = d.get("eid") or eid_map.get((d["cluster"], d["proc"]), "")
    m = re.match(r"sixregime__(.+?)__((?:weak|intermediate|strong)_\w+?)__nph\d", eid)
    if not m:
        continue
    row = d["row"]
    streams.setdefault((m.group(1), m.group(2)), {})[int(row["depth"])] = (
        float(row["energy"]), row["best_op"].split("::")[-1])

# ---- Bundle-1 insertion strong streams from merged tails (k = depth-1) ----
for f in glob.glob(str(CD / "live_tails/allphase_k75__*_position_phase0__*.tail")):
    m = re.match(r"allphase_k75__(.+?)__((?:weak|intermediate|strong)_\w+?)__nph\d", os.path.basename(f))
    if not m:
        continue
    rows = {}
    for line in open(f):
        p = parse_line(line)
        if p and p[0] >= 2:
            rows[p[0] - 1] = (p[1], p[2])
    if rows:
        streams[(m.group(1), m.group(2))] = rows

# ---- compile k* tuples for every stream past k* with contiguous words ----
live_star = {}
for (arm, regime), rows in sorted(streams.items()):
    ks = KSTAR[regime]
    if ks not in rows:
        continue
    if any(k not in rows for k in range(1, ks + 1)):
        print(f"SKIP {arm}|{regime}: word gap below k*={ks}", flush=True)
        continue
    words = [rows[k][1] for k in sorted(rows) if k <= max(rows)]
    words = [rows[k][1] for k in range(1, max(rows) + 1) if k in rows]
    # contiguous prefix only (gaps above k* are fine for the k* compile)
    prefix = []
    for k in range(1, max(rows) + 1):
        if k not in rows:
            break
        prefix.append(rows[k][1])
    tup = mod._routed_prefix_tuple(f"kstarfill_{arm}_{regime}", prefix, regime, ks, WORKDIR)
    live_star[f"{arm}|{regime}"] = {
        "k_star": ks,
        "dE_star": max(abs(rows[ks][0] - exact[regime]), 1e-16),
        "tuple_star": tup,
    }
    print(f"live_star {arm}|{regime} k*={ks} {tup}", flush=True)
KT["live_star"] = live_star

# ---- newly completed cells: Bundle-4 append trio (+ routed unification) ----
def last_series(raw, key):
    return [int(x) for x in re.findall(rb'"' + key.encode() + rb'"\s*:\s*(\d+)', raw)]

def checkpoint_prefix_salg(tar, want_rounds):
    """Per-round cumulative S_alg from the archived checkpoint, if exposed."""
    try:
        member = next(m for m in tar if m.name.endswith("run/checkpoints/current.json"))
    except StopIteration:
        return {}
    fh = tar.extractfile(member)
    buf = fh.read(64 << 20)  # prefix checkpoints sit at the head
    m = re.search(rb'"active_prefix_checkpoints"\s*:\s*\[', buf)
    if not m:
        return {}
    i = m.end() - 1; depth = 0; end = None
    for k in range(i, len(buf)):
        c = buf[k:k+1]
        if c == b"[": depth += 1
        elif c == b"]":
            depth -= 1
            if depth == 0:
                end = k + 1; break
    if end is None:
        return {}
    try:
        entries = json.loads(buf[i:end])
    except ValueError:
        return {}
    per = {}
    for e in entries:
        rnd = e.get("controller_round") or e.get("active_ansatz_depth")
        sa = None
        o = e.get("estimator_prefix")
        if isinstance(o, dict):
            sa = (o.get("cumulative_executed_queries") or {}).get("S_alg")
        if sa is None:
            sa = e.get("S_alg")
        if rnd is not None and isinstance(sa, int):
            per[int(rnd)] = sa
    return per

KT.setdefault("b4", {})
for f in sorted(glob.glob(str(LR / "ra_adapt_warmstart_sixregime_20260820_v1/transfer/sixregime__c_ws_depth_append_ra__*.tar.gz"))):
    m = re.search(r"sixregime__(.+?)__((?:weak|intermediate|strong)_\w+?)__nph\d", f)
    arm, regime = m.group(1), m.group(2)
    with tarfile.open(f, "r:gz") as tar:
        members = tar.getmembers()
        sm = [mm for mm in members if mm.name.endswith("run/summary/summary.json")]
        if not sm:
            print("no summary:", f, flush=True)
            continue
        summary = json.loads(tar.extractfile(sm[0]).read())
        salg_rounds = checkpoint_prefix_salg(iter(members) and tar, None)
    tr = summary["accepted_error_trace"]
    errs = {int(x["controller_round"]): max(float(x["absolute_energy_error"]), 1e-16) for x in tr}
    kt_ = max(errs)
    ks = KSTAR[regime]
    rows = streams.get((arm, regime), {})
    prefix = []
    for k in range(1, kt_ + 1):
        if k not in rows:
            break
        prefix.append(rows[k][1])
    caw = summary.get("canonical_all_work") or {}
    KT["b4"][f"{arm}|{regime}"] = {
        "k_star": ks if ks in errs else None, "k_term": kt_,
        "dE_star": errs.get(ks), "dE_term": errs[kt_],
        "tuple_star": mod._routed_prefix_tuple(f"kstarfill_{arm}_{regime}", prefix, regime, ks, WORKDIR) if len(prefix) >= ks else None,
        "tuple_term": mod._routed_prefix_tuple(f"kstarfill_{arm}_{regime}", prefix, regime, kt_, WORKDIR) if len(prefix) >= kt_ else None,
        "salg_star": salg_rounds.get(ks),
        "salg_term": caw.get("s_alg"),
    }
    print("b4", arm, regime, KT["b4"][f"{arm}|{regime}"], flush=True)

# ---- Bundle-3 trio: routed unification + checkpoint S_alg@k* ----
for f in sorted(glob.glob(str(LR / "ra_adapt_costexp_depthfix_sixregime_20260820_v2/transfer/sixregime__b_depth_append_ra__*.tar.gz"))):
    m = re.search(r"sixregime__(.+?)__((?:weak|intermediate|strong)_\w+?)__nph\d", f)
    arm, regime = m.group(1), m.group(2)
    key = f"{arm}|{regime}"
    if key not in KT["b3"]:
        continue
    rows = streams.get((arm, regime), {})
    entry = KT["b3"][key]
    kt_ = entry["k_term"]
    ks = KSTAR[regime]
    prefix = []
    for k in range(1, kt_ + 1):
        if k not in rows:
            break
        prefix.append(rows[k][1])
    with tarfile.open(f, "r:gz") as tar:
        salg_rounds = checkpoint_prefix_salg(tar, None)
    entry["tuple_star_matrix"] = entry.get("tuple_star")
    entry["tuple_term_matrix"] = entry.get("tuple_term")
    if len(prefix) >= ks:
        entry["tuple_star"] = mod._routed_prefix_tuple(f"kstarfill_{arm}_{regime}", prefix, regime, ks, WORKDIR)
    if len(prefix) >= kt_:
        entry["tuple_term"] = mod._routed_prefix_tuple(f"kstarfill_{arm}_{regime}", prefix, regime, kt_, WORKDIR)
    if entry.get("salg_star") is None:
        entry["salg_star"] = salg_rounds.get(ks)
    print("b3 unified", key, entry["tuple_star"], entry["tuple_term"], "salg*", entry["salg_star"], flush=True)

json.dump(KT, open(S / "kstar_tables.json", "w"), indent=1)
print("DONE fill_kstar_all", flush=True)
