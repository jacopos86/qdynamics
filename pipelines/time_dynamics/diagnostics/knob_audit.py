"""Which knobs actually fired across completed runs? Evidence for simplification."""
import json, glob
from collections import Counter

runs = sorted(glob.glob("output/*/*/run.json")) + sorted(glob.glob("output/*/run.json"))
gate_reasons = Counter(); stop_reasons = Counter()
repair = Counter(); subdiv = Counter(); policy_fields = Counter()
score_terms = Counter(); n = 0
for path in runs:
    try:
        run = json.load(open(path))
    except Exception:
        continue
    n += 1
    for p in run.get("trajectory", {}).get("points", []):
        m = (p.get("patch_decision") or {}).get("metadata") or {}
        if m.get("stop_reason"): stop_reasons[m["stop_reason"]] += 1
        for a in (m.get("attempts") or []):
            gate_reasons[str(a.get("reason", ""))[:34]] += 1
        itg = p.get("integration_to_next") or {}
        rs = itg.get("repair_summary") or {}
        if itg.get("local_subdivision_applied"): subdiv["subdivision_applied"] += 1
        for k in ("kink_triggered", "condition_triggered", "rho_num_triggered"):
            if rs.get(k): repair[k] += 1
        if itg.get("repair_applied"): repair["repair_applied"] += 1
        cfg_used = rs.get("selected_candidate_id") or rs.get("selected_policy")
        if cfg_used: policy_fields[str(cfg_used)] += 1
print(f"runs scanned: {n}\n")
print("== certification gate outcomes ==")
for k, v in gate_reasons.most_common(10): print(f"  {v:6d}  {k}")
print("\n== selector stop reasons ==")
for k, v in stop_reasons.most_common(8): print(f"  {v:6d}  {k}")
print("\n== numerical repair triggers ==")
for k, v in (repair + subdiv).most_common(12): print(f"  {v:6d}  {k}")
print("\n== repair candidate/policy actually selected ==")
for k, v in policy_fields.most_common(8): print(f"  {v:6d}  {k}")
