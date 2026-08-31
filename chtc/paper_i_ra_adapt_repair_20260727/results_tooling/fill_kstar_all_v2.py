"""k* tuples for every RA cell past its plateau point.

Insertion arms are compiled from the REPLAYED insertion ordering
(selected_position replay r1..k*) — exact round-k* circuits. Append arms
use the acceptance-order prefix (identical to their true ordering).
Spot-check cells also compile the truncated (acceptance-order)
approximation and report the delta. Completed cells (Bundle-3/4 trios)
get unified routed tuples + checkpoint per-round S_alg where exposed."""
import importlib.util, json, sys, re, glob, os, tarfile
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

APPEND_ARMS = {"append_ra", "b_depth_append_ra", "c_ws_depth_append_ra"}
SPOT = {"a_costexp_plateau_position|weak_weak", "always_open_position_phase0|strong_strong_u8"}

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
    return (int(d["depth"]), float(d["energy"]),
            d["best_op"].split("::")[-1], int(d.get("selected_position", -1)))

def fast_balanced_end(buf, i):
    depth = 0
    for m in re.finditer(rb"[\[\]]", buf):
        if m.start() < i:
            continue
        if m.group() == b"[":
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return m.end()
    return None

def checkpoint_prefix_salg(tar):
    try:
        member = next(m for m in tar if m.name.endswith("run/checkpoints/current.json"))
    except StopIteration:
        return {}
    buf = tar.extractfile(member).read(96 << 20)
    m = re.search(rb'"active_prefix_checkpoints"\s*:\s*\[', buf)
    if not m:
        return {}
    end = fast_balanced_end(buf, m.end() - 1)
    if not end:
        return {}
    try:
        entries = json.loads(buf[m.end() - 1:end])
    except ValueError:
        return {}
    per = {}
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
    return per

def replay_order(rows, k):
    """Insertion-ordered word list at round k via selected_position replay."""
    seq = []
    for r in range(1, k + 1):
        _, w, p = rows[r]
        if p < 0 or p > len(seq):
            p = len(seq)
        seq.insert(p, w)
    return seq

# ---- streams ----
streams = {}
for f in sorted(glob.glob(str(S / "matrix_outs/*.out"))):
    m = re.search(r"sixregime__(.+?)__((?:weak|intermediate|strong)_\w+?)__nph\d", os.path.basename(f))
    if not m:
        continue
    rows = streams.setdefault((m.group(1), m.group(2)), {})
    for line in open(f):
        p = parse_line(line)
        if p:
            rows[p[0]] = (p[1], p[2], p[3])
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
        float(row["energy"]), row["best_op"].split("::")[-1],
        int(row.get("selected_position", -1)))
# Bundle-1 insertion strong tails (k = depth-1)
for f in glob.glob(str(CD / "live_tails/allphase_k75__*_position_phase0__*.tail")):
    m = re.match(r"allphase_k75__(.+?)__((?:weak|intermediate|strong)_\w+?)__nph\d", os.path.basename(f))
    if not m:
        continue
    rows = {}
    for line in open(f):
        p = parse_line(line)
        if p and p[0] >= 2:
            rows[p[0] - 1] = (p[1], p[2], p[3])
    if rows:
        streams[(m.group(1), m.group(2))] = rows

# append-position sanity: every append row should insert at the end
for (arm, regime), rows in streams.items():
    if arm in APPEND_ARMS:
        bad = [k for k in rows if 0 <= rows[k][2] != k - 1]
        if bad:
            print(f"WARN append position anomaly {arm}|{regime}: rounds {bad[:5]}", flush=True)

# ---- k* compiles for everything past k* ----
live_star = {}
spot_report = {}
for (arm, regime), rows in sorted(streams.items()):
    ks = KSTAR[regime]
    if ks not in rows or any(k not in rows for k in range(1, ks + 1)):
        if ks in rows or max(rows) >= ks:
            print(f"SKIP {arm}|{regime}: gap below k*={ks}", flush=True)
        continue
    key = f"{arm}|{regime}"
    truncated = [rows[k][1] for k in range(1, ks + 1)]
    if arm in APPEND_ARMS:
        ordering, method = truncated, "acceptance_order_exact_for_append"
    else:
        ordering, method = replay_order(rows, ks), "insertion_position_replay_v1"
    tup = mod._routed_prefix_tuple(f"replay_{arm}_{regime}", ordering, regime, ks, WORKDIR)
    live_star[key] = {
        "k_star": ks,
        "dE_star": max(abs(rows[ks][0] - exact[regime]), 1e-16),
        "tuple_star": tup,
        "ordering_method": method,
    }
    print(f"star {key} k*={ks} [{method}] {tup}", flush=True)
    if key in SPOT and arm not in APPEND_ARMS:
        if ordering == truncated:
            spot_report[key] = {"identical_ordering": True, "replayed": tup, "truncated": tup}
        else:
            tup_trunc = mod._routed_prefix_tuple(f"trunc_{arm}_{regime}", truncated, regime, ks, WORKDIR)
            spot_report[key] = {"identical_ordering": False, "replayed": tup, "truncated": tup_trunc}
        print("SPOT", key, spot_report[key], flush=True)

# extra spot coverage: report ordering-identity below k* for every insertion cell
for (arm, regime), rows in sorted(streams.items()):
    ks = KSTAR[regime]
    if arm in APPEND_ARMS or ks not in rows or any(k not in rows for k in range(1, ks + 1)):
        continue
    ident = replay_order(rows, ks) == [rows[k][1] for k in range(1, ks + 1)]
    print(f"ordering@k* {arm}|{regime}: {'identical' if ident else 'DIFFERS'}", flush=True)

KT["live_star"] = live_star
KT["spot_check"] = spot_report

# ---- Bundle-4 completed trio ----
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
        salg_rounds = checkpoint_prefix_salg(tar)
    tr = summary["accepted_error_trace"]
    errs = {int(x["controller_round"]): max(float(x["absolute_energy_error"]), 1e-16) for x in tr}
    kt_ = max(errs)
    ks = KSTAR[regime]
    rows = streams.get((arm, regime), {})
    prefix = [rows[k][1] for k in range(1, kt_ + 1) if k in rows]
    ok = len(prefix) == kt_
    caw = summary.get("canonical_all_work") or {}
    KT["b4"][f"{arm}|{regime}"] = {
        "k_star": ks if ks in errs else None, "k_term": kt_,
        "dE_star": errs.get(ks), "dE_term": errs[kt_],
        "tuple_star": mod._routed_prefix_tuple(f"replay_{arm}_{regime}", prefix, regime, ks, WORKDIR) if ok and ks <= kt_ else None,
        "tuple_term": mod._routed_prefix_tuple(f"replay_{arm}_{regime}", prefix, regime, kt_, WORKDIR) if ok else None,
        "salg_star": salg_rounds.get(ks) or (KT.get("b4", {}).get(f"{arm}|{regime}") or {}).get("salg_star"),
        "salg_term": caw.get("s_alg") or (KT.get("b4", {}).get(f"{arm}|{regime}") or {}).get("salg_term"),
    }
    print("b4", arm, regime, KT["b4"][f"{arm}|{regime}"], flush=True)

# ---- Bundle-3 trio: routed unification + checkpoint S_alg ----
for f in sorted(glob.glob(str(LR / "ra_adapt_costexp_depthfix_sixregime_20260820_v2/transfer/sixregime__b_depth_append_ra__*.tar.gz"))):
    m = re.search(r"sixregime__(.+?)__((?:weak|intermediate|strong)_\w+?)__nph\d", f)
    arm, regime = m.group(1), m.group(2)
    key = f"{arm}|{regime}"
    if key not in KT["b3"]:
        continue
    entry = KT["b3"][key]
    kt_ = entry["k_term"]
    ks = KSTAR[regime]
    rows = streams.get((arm, regime), {})
    prefix = [rows[k][1] for k in range(1, kt_ + 1) if k in rows]
    ok = len(prefix) == kt_
    with tarfile.open(f, "r:gz") as tar:
        salg_rounds = checkpoint_prefix_salg(tar)
    entry.setdefault("tuple_star_matrix", entry.get("tuple_star"))
    entry.setdefault("tuple_term_matrix", entry.get("tuple_term"))
    if ok and ks <= kt_:
        entry["tuple_star"] = mod._routed_prefix_tuple(f"replay_{arm}_{regime}", prefix, regime, ks, WORKDIR)
    if ok:
        entry["tuple_term"] = mod._routed_prefix_tuple(f"replay_{arm}_{regime}", prefix, regime, kt_, WORKDIR)
    if entry.get("salg_star") is None:
        entry["salg_star"] = salg_rounds.get(ks)
    print("b3 unified", key, entry["tuple_star"], entry["tuple_term"], "salg*", entry["salg_star"], flush=True)

json.dump(KT, open(S / "kstar_tables.json", "w"), indent=1)
print("FILL_V2_DONE", flush=True)
