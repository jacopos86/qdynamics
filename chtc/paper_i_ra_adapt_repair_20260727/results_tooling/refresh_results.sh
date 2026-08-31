#!/bin/bash
# One-command results refresh: harvest -> ingest completions -> compile
# new k* entries (cached) -> rebuild both PDFs + manifest. Idempotent;
# one execute-node contact per running cell, one scp per package.
set -u
cd "$(dirname "$0")"
SSH="ssh -n -S $HOME/.ssh/cm-chtc-agent.sock -o BatchMode=yes jsstrobel@ap2001.chtc.wisc.edu"
echo "== refresh $(date '+%F %T')"
scp -q -o ControlPath=$HOME/.ssh/cm-chtc-agent.sock "jsstrobel@ap2001.chtc.wisc.edu:~/ra_adapt_costexp_depthfix_sixregime_20260820_v2/logs/*.out" matrix_outs/ 2>/dev/null
scp -q -o ControlPath=$HOME/.ssh/cm-chtc-agent.sock "jsstrobel@ap2001.chtc.wisc.edu:~/ra_adapt_warmstart_sixregime_20260820_v1/logs/*.out" matrix_outs/ 2>/dev/null
./refresh_tails.sh 2>/dev/null | tail -2
python3 - <<'PYEOF'
# merge freshly captured raw tails into grounded tails (fresh wins per depth)
import json, glob, os
TAILS = "/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/local_runs/paper_i_ra_allphase_adaptive_20260817_comparison_data/live_tails"
def parse(line):
    if '"hardcoded_adapt_iter"' not in line: return None, None
    i = line.find("{")
    if i < 0: return None, None
    try: return json.loads(line[i:])["depth"], line[i:].strip()
    except (ValueError, KeyError): return None, None
for rawf in glob.glob(TAILS + "/*.tar.gz.tail"):
    tailf = rawf[:-len(".tar.gz.tail")] + ".tail"
    merged = {}
    if os.path.exists(tailf):
        for line in open(tailf):
            d, p = parse(line)
            if d is not None: merged[d] = p
    for line in open(rawf):
        d, p = parse(line)
        if d is not None: merged[d] = p
    with open(tailf, "w") as f:
        for d in sorted(merged): f.write(merged[d] + "\n")
    os.remove(rawf)
PYEOF
./sample_live_all.sh 2>&1 | tail -1
python3 -u fill_completed_generalized.py | tail -5
python3 -u fill_kstar_all_v2.py 2>&1 | tail -3
FINAL_B3_B4=1 python3 build_variant_pages_v7.py   # THE deliverable: 2 pages (B3, B4)
BUNDLE3_ONLY=1 python3 build_variant_pages_v7.py  # legacy 1-pager (frozen handoff references it)
python3 build_variant_pages_v7.py                 # 4-page archive view
echo "== refresh done $(date '+%F %T')"
