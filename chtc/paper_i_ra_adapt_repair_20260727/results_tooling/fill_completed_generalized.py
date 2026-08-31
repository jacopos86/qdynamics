"""Ingest EVERY completed matrix archive (Bundles 2/3/4, all arms) into
kstar_tables.json: trace, dE, replayed/append routed tuples at k* and
k_term, and S_alg pairs from accepted_prefix_work. Idempotent: cells
already filled are skipped unless the archive is newer than the entry."""
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

SECTION = [("a_costexp_", "b2"), ("b_depth_", "b3"), ("c_ws_", "b4")]
APPEND_ARMS = {"append_ra", "b_depth_append_ra", "c_ws_depth_append_ra"}

def salg_prefix_from_raw(raw):
    m = re.search(rb'"accepted_prefix_work"\s*:\s*\[', raw)
    if not m:
        return {}
    i = m.end() - 1
    depth = 0
    for mm in re.finditer(rb"[\[\]]", raw):
        if mm.start() < i:
            continue
        if mm.group() == b"[":
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                try:
                    entries = json.loads(raw[i:mm.end()])
                except ValueError:
                    return {}
                per = {}
                for idx, e in enumerate(entries):
                    if isinstance(e, dict):
                        sa = e.get("s_alg", e.get("S_alg"))
                        if isinstance(sa, int):
                            per[int(e.get("controller_round", idx + 1))] = sa
                return per
    return {}

def checkpoint_salg_stream(tar_path):
    """Per-round cumulative S_alg from a PARTIAL archive's checkpoint
    (exit-held cells ship checkpoints but no result.json)."""
    per = {}
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            member = next(m for m in tar if m.name.endswith("run/checkpoints/current.json"))
            fh = tar.extractfile(member)
            overlap = b""; cur = None
            while True:
                chunk = fh.read(32 << 20)
                if not chunk:
                    break
                buf = overlap + chunk; ev = []
                for m in re.finditer(rb'"active_ansatz_depth"\s*:\s*(\d+)', buf):
                    ev.append((m.start(), "d", int(m.group(1))))
                for m in re.finditer(rb'"cumulative_executed_queries"\s*:\s*\{\s*"S_alg"\s*:\s*(\d+)', buf):
                    ev.append((m.start(), "s", int(m.group(1))))
                for pos, kind, val in sorted(ev):
                    if pos < len(overlap) - (1 << 20):
                        continue
                    if kind == "d":
                        cur = val
                    elif cur is not None:
                        per.setdefault(cur, val)
                overlap = buf[-(1 << 20):]
    except (StopIteration, OSError, tarfile.TarError):
        return {}
    return per


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

def replay_order(rows, k, is_append):
    if is_append:
        return [rows[r][1] for r in range(1, k + 1)]
    seq = []
    for r in range(1, k + 1):
        _, w, p = rows[r]
        if p < 0 or p > len(seq):
            p = len(seq)
        seq.insert(p, w)
    return seq

patterns = [
    str(LR / "ra_adapt_costexp_depthfix_sixregime_20260820_v2/transfer/sixregime__*.tar.gz"),
    str(LR / "ra_adapt_warmstart_sixregime_20260820_v1/transfer/sixregime__*.tar.gz"),
]
for pattern in patterns:
    for f in sorted(glob.glob(pattern)):
        m = re.search(r"sixregime__(.+?)__((?:weak|intermediate|strong)_\w+?)__nph\d", f)
        if not m:
            continue
        arm, regime = m.group(1), m.group(2)
        section = next((sec for pre, sec in SECTION if arm.startswith(pre)), None)
        if section is None:
            continue
        key = f"{arm}|{regime}"
        KT.setdefault(section, {})
        prior = KT[section].get(key)
        mtime = os.path.getmtime(f)
        if prior and prior.get("salg_term") is not None and prior.get("_archive_mtime") == mtime:
            continue
        with tarfile.open(f, "r:gz") as tar:
            members = tar.getmembers()
            sm = [mm for mm in members if mm.name.endswith("run/summary/summary.json")]
            if not sm:
                partial = True
            else:
                partial = False
                summary = json.loads(tar.extractfile(sm[0]).read())
            rj = [mm for mm in members if mm.name.endswith("result.json")
                  and "source/" not in mm.name]
            raw = tar.extractfile(rj[0]).read() if rj else None
        per_salg = salg_prefix_from_raw(raw) if raw else {}
        del raw; gc.collect()
        if partial:
            # exit-held: trace from stream, per-round S_alg from checkpoint
            outs_p = sorted(glob.glob(str(S / f"matrix_outs/*sixregime__{arm}__{regime}__*.out")))
            rows_p = stream_rows(outs_p[-1]) if outs_p else {}
            if not rows_p or any(k not in rows_p for k in range(1, max(rows_p) + 1)):
                print("pending (partial, no usable stream):", os.path.basename(f), flush=True)
                continue
            per_p = checkpoint_salg_stream(f)
            kt_p = max(rows_p); ks_p = KSTAR[regime]
            is_app_p = arm in APPEND_ARMS
            errs_p = {k: max(abs(rows_p[k][0] - exact[regime]), 1e-16) for k in rows_p}
            KT[section][key] = {
                "k_star": ks_p if ks_p in errs_p else None, "k_term": kt_p,
                "dE_star": errs_p.get(ks_p), "dE_term": errs_p[kt_p],
                "tuple_star": mod._routed_prefix_tuple(f"replay_{arm}_{regime}", replay_order(rows_p, ks_p, is_app_p), regime, ks_p, WORKDIR) if ks_p <= kt_p else None,
                "tuple_term": mod._routed_prefix_tuple(f"replay_{arm}_{regime}", replay_order(rows_p, kt_p, is_app_p), regime, kt_p, WORKDIR),
                "salg_star": per_p.get(ks_p), "salg_term": per_p.get(kt_p),
                "trace": sorted(errs_p.items()),
                "ordering_method": "acceptance_order_exact_for_append" if is_app_p else "insertion_position_replay_v1",
                "source": "partial archive (exit-held): stream trace + checkpoint S_alg receipts",
                "_archive_mtime": mtime,
            }
            print(f"{section} PARTIAL-filled {key}: k*{ks_p} kt{kt_p} salg {per_p.get(ks_p)}/{per_p.get(kt_p)}", flush=True)
            continue
        tr = summary["accepted_error_trace"]
        errs = {int(x["controller_round"]): max(float(x["absolute_energy_error"]), 1e-16) for x in tr}
        kt_ = max(errs)
        ks = KSTAR[regime]
        outs = sorted(glob.glob(str(S / f"matrix_outs/*sixregime__{arm}__{regime}__*.out")))
        rows = stream_rows(outs[-1]) if outs else {}
        ok = all(k in rows for k in range(1, kt_ + 1))
        is_append = arm in APPEND_ARMS
        caw = summary.get("canonical_all_work") or {}
        entry = {
            "k_star": ks if ks in errs else None, "k_term": kt_,
            "dE_star": errs.get(ks), "dE_term": errs[kt_],
            "tuple_star": mod._routed_prefix_tuple(f"replay_{arm}_{regime}", replay_order(rows, ks, is_append), regime, ks, WORKDIR) if ok and ks <= kt_ else None,
            "tuple_term": mod._routed_prefix_tuple(f"replay_{arm}_{regime}", replay_order(rows, kt_, is_append), regime, kt_, WORKDIR) if ok else None,
            "salg_star": per_salg.get(ks),
            "salg_term": caw.get("s_alg", per_salg.get(kt_)),
            "trace": sorted(errs.items()),
            "ordering_method": "acceptance_order_exact_for_append" if is_append else "insertion_position_replay_v1",
            "_archive_mtime": mtime,
        }
        KT[section][key] = entry
        print(f"{section} filled {key}: k* {entry['k_star']} kt {kt_} "
              f"salg {entry['salg_star']}/{entry['salg_term']} "
              f"tuples {'ok' if entry['tuple_term'] else 'missing-words'}", flush=True)

json.dump(KT, open(S / "kstar_tables.json", "w"), indent=1)
print("FILL_COMPLETED_DONE", flush=True)
