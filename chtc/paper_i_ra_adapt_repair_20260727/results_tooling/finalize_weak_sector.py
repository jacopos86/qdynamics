"""Weak-sector finalization per user ruling 2026-08-20:
- all six b_depth costexp weak cells completed their 50 rounds (streams);
  build their b3 rows from streams (replayed tuples at k* and 50);
- plateau trio terminal S_alg from partial-tarball checkpoint headers;
- weak_report_45: dE and replayed tuples at the reporting point k=45
  for AAVQE + all nine Bundle-3 weak cells, plus the measured
  always/strong_weak_u8 descent exception."""
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
TOLM = LR / "paper_i_aavqe_r50_to_r75_tolmatch_20260820_v2"
KT = json.load(open(S / "kstar_tables.json"))
KSTAR = KT["kstar"]
WEAK = ("weak_weak", "intermediate_weak", "strong_weak_u8")
K_REP = 45

def stream_rows(path):
    rows = {}
    for line in open(path):
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

def checkpoint_salg(tar_path):
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            member = next(m for m in tar if m.name.endswith("run/checkpoints/current.json"))
            head = tar.extractfile(member).read(4096)
    except (StopIteration, OSError):
        return None, None
    ms = re.search(rb'"S_alg"\s*:\s*(\d+)', head)
    mo = re.search(rb'"N_H_outer"\s*:\s*(\d+)', head)
    return (int(ms.group(1)) if ms else None, int(mo.group(1)) if mo else None)

# ---- six costexp weak cells: completed rows from streams ----
for arm in ("b_depth_costexp_always_open", "b_depth_costexp_plateau_position"):
    for regime in WEAK:
        outs = sorted(glob.glob(str(S / f"matrix_outs/*sixregime__{arm}__{regime}__*.out")))
        if not outs:
            continue
        rows = stream_rows(outs[-1])
        kt_ = max(rows)
        if kt_ < 50 or any(k not in rows for k in range(1, kt_ + 1)):
            print("incomplete stream:", arm, regime, kt_, flush=True)
            continue
        ks = KSTAR[regime]
        errs = {k: max(abs(rows[k][0] - exact[regime]), 1e-16) for k in rows}
        salg_term = None
        if "plateau" in arm:
            proc = {"weak_weak": 24, "intermediate_weak": 25, "strong_weak_u8": 26}[regime]
            t = LR / f"ra_adapt_costexp_depthfix_sixregime_20260820_v2/transfer/sixregime__{arm}__{regime}__nph3__9664325__{proc}.tar.gz"
            if t.exists():
                sa, nho = checkpoint_salg(t)
                if sa and nho == kt_:
                    salg_term = sa
                print(f"checkpoint salg {arm}|{regime}: {sa} at N_H_outer={nho}", flush=True)
        prior = KT["b3"].get(f"{arm}|{regime}", {})
        KT["b3"][f"{arm}|{regime}"] = {
            "k_star": ks, "k_term": kt_,
            "dE_star": errs[ks], "dE_term": errs[kt_],
            "tuple_star": mod._routed_prefix_tuple(f"replay_{arm}_{regime}", replay_order(rows, ks, False), regime, ks, WORKDIR),
            "tuple_term": mod._routed_prefix_tuple(f"replay_{arm}_{regime}", replay_order(rows, kt_, False), regime, kt_, WORKDIR),
            "salg_star": prior.get("salg_star"),
            "salg_term": salg_term or prior.get("salg_term"),
            "trace": sorted(errs.items()),
            "ordering_method": "insertion_position_replay_v1",
            "completed_from": "streams+checkpoint (archives pending)" if salg_term is None else "streams+checkpoint",
        }
        print("b3 finalized", arm, regime, "k*", ks, "term", kt_, "salg_term", salg_term, flush=True)

# ---- weak_report_45: dE + replayed tuples at the reporting point ----
aavqe_hist = {r: dict((int(k), max(abs(e - exact[r]), 1e-16), ) if False else (int(k), max(abs(e - exact[r]), 1e-16)) for k, e in rows)
              for r, rows in json.load(open(TOLM / "aavqe_full_hist_tolmatch_9665004.json")).items()}
aavqe_words = json.load(open(TOLM / "aavqe_words_r75_9665004.json"))
aavqe_salg = json.load(open(TOLM / "aavqe_salg_prefix_r75_9665004.json"))
report = {"k_rep": K_REP, "regimes": {}}
for regime in WEAK:
    entry = {"AAVQE": {
        "dE": aavqe_hist[regime][K_REP],
        "tuple": mod._routed_prefix_tuple(f"aavqe_r75_{regime}", aavqe_words[regime], regime, K_REP, WORKDIR),
        "salg": aavqe_salg[regime][str(K_REP)],
    }}
    for arm in ("b_depth_append_ra", "b_depth_costexp_always_open", "b_depth_costexp_plateau_position"):
        outs = sorted(glob.glob(str(S / f"matrix_outs/*sixregime__{arm}__{regime}__*.out")))
        if not outs:
            continue
        rows = stream_rows(outs[-1])
        if any(k not in rows for k in range(1, K_REP + 1)):
            continue
        is_append = arm == "b_depth_append_ra"
        entry[arm] = {
            "dE": max(abs(rows[K_REP][0] - exact[regime]), 1e-16),
            "tuple": mod._routed_prefix_tuple(f"replay_{arm}_{regime}", replay_order(rows, K_REP, is_append), regime, K_REP, WORKDIR),
            "salg": None,  # per-round ledger arrives with archives
        }
        print("report45", arm, regime, entry[arm]["dE"], flush=True)
    report["regimes"][regime] = entry
# the measured descent exception (always-insertion strong_weak_u8)
outs = sorted(glob.glob(str(S / "matrix_outs/*sixregime__b_depth_costexp_always_open__strong_weak_u8__*.out")))
rows = stream_rows(outs[-1])
exc = {str(k): max(abs(rows[k][0] - exact["strong_weak_u8"]), 1e-16) for k in range(43, 51)}
report["exception"] = {
    "cell": "b_depth_costexp_always_open|strong_weak_u8",
    "statement": "always-insertion strong_weak_u8 descends past the report point",
    "dE_by_round_43_50": exc,
}
KT["weak_report_45"] = report
json.dump(KT, open(S / "kstar_tables.json", "w"), indent=1)
print("WEAK_FINALIZE_DONE", flush=True)
