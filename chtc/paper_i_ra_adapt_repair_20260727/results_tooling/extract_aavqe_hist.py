import json, tarfile, glob, re, gc
out = {}
for f in sorted(glob.glob("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/local_runs/paper_i_aavqe_r50_to_r75_continuation_20260819/transfer/*.tar.gz")):
    regime = re.search(r"aavqe_r75cont__(.+?)__nph", f).group(1)
    with tarfile.open(f, "r:gz") as tar:
        member = next(m for m in tar if m.name.endswith("run/result.json"))
        raw = tar.extractfile(member).read()
    # slice just the top-level "history" array to keep memory bounded
    i = raw.find(b'"history":')
    assert i != -1, regime
    j = raw.index(b"[", i)
    depth = 0
    for k in range(j, len(raw)):
        c = raw[k:k+1]
        if c == b"[": depth += 1
        elif c == b"]":
            depth -= 1
            if depth == 0: break
    rows = json.loads(raw[j:k+1])
    del raw; gc.collect()
    out[regime] = [
        (int(r["controller_round"]), float(r["energy_after"]))
        for r in rows if isinstance(r, dict) and "controller_round" in r
    ]
    print(regime, "rounds:", len(out[regime]), "last:", out[regime][-1])
json.dump(out, open("aavqe_full_hist.json", "w"))
print("saved aavqe_full_hist.json")
