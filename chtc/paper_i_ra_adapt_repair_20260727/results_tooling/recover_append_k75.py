"""Recover Bundle-1 append strong k75 cells from the v1 crash write-backs.

1. Parse the three 9663770 .out streams (full r1..k_term trajectories+words).
2. Verify the depth->round shift against the bit-identical maxk50 prefix.
3. Replace the truncated live tails with the full streams.
4. Probe crash checkpoints for per-round cumulative S_alg (k* values).
5. Compile routed tuples at k* and k_term; update kstar_tables.json b1 rows.
"""
import importlib.util, json, sys, re, glob, os, tarfile, io, gzip
from pathlib import Path

HERE = "/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/chtc/paper_i_ra_adapt_repair_20260727"
spec = importlib.util.spec_from_file_location("p1b", f"{HERE}/build_paper_i_ra_phase123_qiskit_comparison_pdf_20260817.py")
mod = importlib.util.module_from_spec(spec); sys.modules["p1b"] = mod; spec.loader.exec_module(mod)
exact = {r: v["exact_energy"] for r, v in mod.load_hist_plateau().items()}

S = Path("/private/tmp/claude-501/-Users-jakestrobel-Documents-Holstein-implementation-Holstein-test-fullclone-3/91434114-4541-47ef-95e4-2b8bd90bd929/scratchpad")
CD = Path("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/local_runs/paper_i_ra_allphase_adaptive_20260817_comparison_data")
TAILS = CD / "live_tails"
V1 = Path("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/local_runs/paper_i_ra_allphase_adaptive_maximum_k75_extension_20260819/v1_crash_tarballs")
KSTAR = json.load(open(S / "kstar_tables.json"))["kstar"]

streams = {}
for out in sorted(S.glob("9663770.*-allphase_k75__append_ra__*.out")):
    regime = re.search(r"append_ra__(.+?)__nph", out.name).group(1)
    rows = {}
    for line in open(out):
        if '"hardcoded_adapt_iter"' not in line:
            continue
        i = line.find("{")
        try:
            d = json.loads(line[i:])
        except ValueError:
            continue
        rows[int(d["depth"])] = (line[i:].strip(), float(d["energy"]),
                                 d["best_op"].split("::")[-1])
    streams[regime] = rows
    print(regime, "stream depths", min(rows), "-", max(rows))

# ---- shift verification: stream depth d vs maxk50 trace round d-1 ----
for regime, rows in streams.items():
    sp = CD / f"x_allphase_maxk50__append_ra__{regime}/run/summary/summary.json"
    tr = json.load(open(sp))["accepted_error_trace"]
    acc = {int(x["controller_round"]): float(x["accepted_energy"]) for x in tr}
    d51 = rows.get(51)
    if d51 is None or 50 not in acc:
        print(regime, "shift check: prefix data missing"); continue
    dm1 = abs(d51[1] - acc[50])
    d0 = abs(rows[50][1] - acc[50]) if 50 in rows else float("inf")
    shift = 1 if dm1 <= d0 else 0
    print(f"{regime}: |E(d51)-E(r50)|={dm1:.2e}  |E(d50)-E(r50)|={d0:.2e}  -> shift={shift}")
    streams[regime] = (rows, shift)

# ---- replace tails with full streams ----
for regime, (rows, shift) in streams.items():
    tail = TAILS / f"allphase_k75__append_ra__{regime}__nph7.tail"
    with open(tail, "w") as f:
        for d in sorted(rows):
            f.write(rows[d][0] + "\n")
    print("tail replaced:", tail.name, len(rows), "rows")

# ---- per-round S_alg probe from crash checkpoints (bounded stream read) ----
def prefix_salg(tar_path):
    """Read current.json from the tarball only up to the end of
    active_prefix_checkpoints; return {round: cumulative S_alg} if present."""
    with tarfile.open(tar_path, "r:gz") as tar:
        member = next(m for m in tar if m.name.endswith("run/checkpoints/current.json"))
        fh = tar.extractfile(member)
        buf = b""
        start = None
        while True:
            chunk = fh.read(1 << 22)
            if not chunk:
                break
            buf += chunk
            if start is None:
                m = re.search(rb'"active_prefix_checkpoints"\s*:\s*\[', buf)
                if m:
                    start = m.end() - 1
            if start is not None:
                depth = 0; end = None
                for k in range(start, len(buf)):
                    c = buf[k:k+1]
                    if c == b"[": depth += 1
                    elif c == b"]":
                        depth -= 1
                        if depth == 0:
                            end = k + 1; break
                if end is not None:
                    entries = json.loads(buf[start:end])
                    return entries
            if len(buf) > (1 << 31):
                return None
    return None

salg_by_round = {}
for f in sorted(V1.glob("*.tar.gz")):
    regime = re.search(r"append_ra__(.+?)__nph", f.name).group(1)
    entries = prefix_salg(f)
    if not entries:
        print(regime, "prefix checkpoints: none/too deep"); continue
    e0 = entries[0]
    keys = sorted(e0)
    per = {}
    for e in entries:
        rnd = e.get("controller_round") or e.get("active_ansatz_depth")
        sa = None
        for path in (("estimator_prefix", "cumulative_executed_queries", "S_alg"),
                     ("S_alg",), ("cumulative_executed_queries", "S_alg")):
            o = e
            for p in path:
                o = o.get(p) if isinstance(o, dict) else None
                if o is None:
                    break
            if isinstance(o, int):
                sa = o; break
        if rnd is not None and sa is not None:
            per[int(rnd)] = sa
    print(regime, "prefix entries", len(entries), "sample keys", keys[:8],
          "per-round S_alg points:", len(per))
    if per:
        salg_by_round[regime] = per

# terminal S_alg totals (from checkpoint heads, verified above)
SALG_TERM = {"weak_strong": 850241, "intermediate_strong": 861886,
             "strong_strong_u8": 760909}

# ---- routed tuples + kstar table update ----
kt_path = S / "kstar_tables.json"
KT = json.load(open(kt_path))
for regime, (rows, shift) in streams.items():
    ks = KSTAR[regime]
    words = [rows[d][2] for d in sorted(rows)]
    if shift:
        words = words[1:]  # drop the reference row; word i -> round i+1
    k_term = max(rows) - shift
    errs = {d - shift: max(abs(rows[d][1] - exact[regime]), 1e-16) for d in rows if d - shift >= 1}
    entry = {
        "k_star": ks if ks in errs else None, "k_term": k_term,
        "dE_star": errs.get(ks), "dE_term": errs[k_term],
        "tuple_star": mod._routed_prefix_tuple(f"v1crash_append_{regime}", words, regime, ks, CD / "workdir") if ks <= len(words) else None,
        "tuple_term": mod._routed_prefix_tuple(f"v1crash_append_{regime}", words, regime, k_term, CD / "workdir"),
        "salg_star": (salg_by_round.get(regime) or {}).get(ks),
        "salg_term": SALG_TERM[regime],
    }
    KT["b1"][f"append_ra|{regime}"] = entry
    print("b1 updated:", regime, "k*", entry["k_star"], "k_term", k_term,
          entry["tuple_star"], entry["tuple_term"], "salg", entry["salg_star"], entry["salg_term"])
json.dump(KT, open(kt_path, "w"), indent=1)
print("DONE")
