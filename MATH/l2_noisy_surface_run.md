# L=2 HH Noisy Surface Run Note

This note is the terminal-agent handoff for the **local/offline oracle-scoring + final-audit noisy surface** used for the direct L=2 Hubbard-Holstein (`hh`) ADAPT run.

It is **not** the real-runtime / paid-hardware path. It is the local fake-backend surface using:

- entrypoint: `python -m pipelines.hardcoded.adapt_pipeline`
- continuation mode: `phase3_v1`
- pool: `pareto_lean_l2`
- noisy oracle / audit backend: `FakeNighthawk`
- mitigation lanes: `none`, `readout`, `full`

## Objective

Map the L=2 HH noisy surface across three mitigation lanes while keeping the physics point and ADAPT surface fixed.

In the current `phase3_v1` route, phase-3 scouting uses the noisy oracle on this surface, but inner re-optimization remains exact unless `--phase3-oracle-inner-objective-mode noisy_v1` is explicitly enabled.

## Fixed physics point

- problem: `hh`
- `L=2`
- `t=1.0`
- `U=4.0`
- `dv=0.0`
- `omega0=1.0`
- `g_ep=0.5`
- `n_ph_max=1`
- boundary: `open`
- ordering: `blocked`
- boson encoding: `binary`

## Fixed algorithm point

- `--adapt-continuation-mode phase3_v1`
- `--adapt-pool pareto_lean_l2`
- `--adapt-max-depth 30`
- `--adapt-eps-grad 1e-5`
- `--adapt-maxiter 800`
- `--adapt-seed 7`
- `--phase1-score-z-alpha 1.0`
- `--adapt-no-finite-angle-fallback`
- `--phase1-no-prune`
- `--initial-state-source adapt_vqe`
- `--skip-pdf`

Important constraint:

- `pareto_lean_l2` is intentionally narrow and is only valid for `L=2` and `n_ph_max=1`.

## Fixed noisy surface

Use the same local noisy surface for both phase-3 oracle evaluation and final audit:

- `--phase3-oracle-gradient-mode backend_scheduled`
- `--phase3-oracle-use-fake-backend`
- `--phase3-oracle-backend-name FakeNighthawk`
- `--phase3-oracle-shots 2048`
- `--phase3-oracle-repeats 8`
- `--phase3-oracle-aggregate mean`
- `--phase3-oracle-seed 7`
- `--phase3-oracle-execution-surface expectation_v1`
- `--phase3-oracle-seed-transpiler 7`
- `--phase3-oracle-transpile-optimization-level 1`
- `--final-noise-audit-mode backend_scheduled`
- `--final-noise-audit-backend-name FakeNighthawk`
- `--final-noise-audit-use-fake-backend`
- `--final-noise-audit-shots 2048`
- `--final-noise-audit-repeats 8`
- `--final-noise-audit-aggregate mean`
- `--final-noise-audit-seed 7`
- `--final-noise-audit-seed-transpiler 7`
- `--final-noise-audit-transpile-optimization-level 1`
- `--final-noise-audit-strict`

## Mitigation lanes

Run these three lanes:

1. `none`
   - `--phase3-oracle-mitigation none`
   - `--final-noise-audit-mitigation none`
2. `readout`
   - `--phase3-oracle-mitigation readout`
   - `--phase3-oracle-local-readout-strategy mthree`
   - `--final-noise-audit-mitigation readout`
   - `--final-noise-audit-local-readout-strategy mthree`
3. `full`
   - `--phase3-oracle-mitigation readout`
   - `--phase3-oracle-local-readout-strategy mthree`
   - `--phase3-oracle-zne-scales 1,3,5`
   - `--phase3-oracle-local-gate-twirling`
   - `--phase3-oracle-dd-sequence XpXm`
   - `--final-noise-audit-mitigation readout`
   - `--final-noise-audit-local-readout-strategy mthree`
   - `--final-noise-audit-zne-scales 1,3,5`
   - `--final-noise-audit-local-gate-twirling`
   - `--final-noise-audit-dd-sequence XpXm`

## Recommended run wrapper

Save the following as, for example, `run_l2_noisy_surface.sh`, execute it from the repo root, and keep the generated run directory under `artifacts/agent_runs/`.

```bash
#!/usr/bin/env bash
set -euo pipefail

if [ ! -f AGENTS.md ]; then
  echo "Run this script from the repo root (the directory that contains AGENTS.md)." >&2
  exit 1
fi

TAG="${1:-$(date -u +%Y%m%dT%H%M%SZ)_direct_hh_L2_phase3_v1_fake_matrix_pareto_lean_l2}"
RUN_DIR="artifacts/agent_runs/$TAG"
RESULTS_DIR="$RUN_DIR/results"
CASES_DIR="$RUN_DIR/cases"
PROGRESS_JSON="$RUN_DIR/progress.json"
SUMMARY_JSON="$RESULTS_DIR/summary.json"
SCRIPT_NAME="${BASH_SOURCE[0]:-run_l2_noisy_surface.sh}"

mkdir -p "$RUN_DIR/logs" "$RESULTS_DIR" "$CASES_DIR"
exec > >(tee -a "$RUN_DIR/logs/stdout.log") 2> >(tee -a "$RUN_DIR/logs/stderr.log" >&2)
printf '%q ' "$SCRIPT_NAME" "$@" > "$RUN_DIR/logs/command.sh"
printf '\n' >> "$RUN_DIR/logs/command.sh"

python - "$PROGRESS_JSON" <<'PY'
from pathlib import Path
import json, sys, time
path = Path(sys.argv[1])
path.write_text(json.dumps({
    "status": "running",
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "pool": "pareto_lean_l2",
    "completed": []
}, indent=2) + "\n")
PY

CURRENT_LABEL=""
trap 'status=$?; if [ $status -ne 0 ] && [ -f "$PROGRESS_JSON" ]; then python - "$PROGRESS_JSON" "$CURRENT_LABEL" "$status" <<'"'"'PY'"'"'
from pathlib import Path
import json, sys, time
path = Path(sys.argv[1])
label = sys.argv[2]
status = int(sys.argv[3])
data = json.loads(path.read_text()) if path.exists() else {}
data["status"] = "failed"
data["failed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
data["failed_case"] = label or None
data["exit_code"] = status
path.write_text(json.dumps(data, indent=2) + "\n")
PY
fi' EXIT

base=(
  python -m pipelines.hardcoded.adapt_pipeline
  --L 2 --problem hh --t 1.0 --u 4.0 --boundary open --ordering blocked --boson-encoding binary
  --omega0 1.0 --g-ep 0.5 --n-ph-max 1
  --adapt-continuation-mode phase3_v1
  --adapt-pool pareto_lean_l2
  --adapt-max-depth 30 --adapt-eps-grad 1e-5 --adapt-maxiter 800 --adapt-seed 7
  --phase1-score-z-alpha 1.0
  --adapt-no-finite-angle-fallback
  --phase1-no-prune
  --phase3-oracle-gradient-mode backend_scheduled
  --phase3-oracle-use-fake-backend
  --phase3-oracle-backend-name FakeNighthawk
  --phase3-oracle-shots 2048
  --phase3-oracle-repeats 8
  --phase3-oracle-aggregate mean
  --phase3-oracle-seed 7
  --phase3-oracle-execution-surface expectation_v1
  --phase3-oracle-seed-transpiler 7
  --phase3-oracle-transpile-optimization-level 1
  --final-noise-audit-mode backend_scheduled
  --final-noise-audit-backend-name FakeNighthawk
  --final-noise-audit-use-fake-backend
  --final-noise-audit-shots 2048
  --final-noise-audit-repeats 8
  --final-noise-audit-aggregate mean
  --final-noise-audit-seed 7
  --final-noise-audit-seed-transpiler 7
  --final-noise-audit-transpile-optimization-level 1
  --final-noise-audit-strict
  --initial-state-source adapt_vqe
  --skip-pdf
)

run_case() {
  local label="$1"
  shift
  local case_dir="$CASES_DIR/$label"
  local case_logs="$case_dir/logs"
  local out_json="$RESULTS_DIR/${label}.json"
  CURRENT_LABEL="$label"
  mkdir -p "$case_logs"
  local cmd=("${base[@]}" "$@" --output-json "$out_json")
  printf '%q ' "${cmd[@]}" > "$case_logs/command.sh"
  printf '\n' >> "$case_logs/command.sh"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] START $label" | tee -a "$RUN_DIR/logs/stdout.log"
  "${cmd[@]}" > "$case_logs/stdout.log" 2> "$case_logs/stderr.log"
  python - "$PROGRESS_JSON" "$label" "$out_json" <<'PY'
import json, sys, time
progress_path, label, out_json = sys.argv[1:4]
with open(progress_path) as f:
    data = json.load(f)
completed = [x for x in data.get("completed", []) if x.get("label") != label]
completed.append({
    "label": label,
    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "output_json": out_json,
})
data["completed"] = completed
with open(progress_path, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
PY
  CURRENT_LABEL=""
}

run_case none \
  --phase3-oracle-mitigation none \
  --final-noise-audit-mitigation none

run_case readout \
  --phase3-oracle-mitigation readout \
  --phase3-oracle-local-readout-strategy mthree \
  --final-noise-audit-mitigation readout \
  --final-noise-audit-local-readout-strategy mthree

run_case full \
  --phase3-oracle-mitigation readout \
  --phase3-oracle-local-readout-strategy mthree \
  --phase3-oracle-zne-scales 1,3,5 \
  --phase3-oracle-local-gate-twirling \
  --phase3-oracle-dd-sequence XpXm \
  --final-noise-audit-mitigation readout \
  --final-noise-audit-local-readout-strategy mthree \
  --final-noise-audit-zne-scales 1,3,5 \
  --final-noise-audit-local-gate-twirling \
  --final-noise-audit-dd-sequence XpXm

python - "$RESULTS_DIR" "$SUMMARY_JSON" "$PROGRESS_JSON" <<'PY'
from pathlib import Path
import json, sys, time
results_dir = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
progress_path = Path(sys.argv[3])
rows = {}
for label in ("none", "readout", "full"):
    path = results_dir / f"{label}.json"
    with path.open() as f:
        data = json.load(f)
    audit = data.get("final_noise_audit_v1") or {}
    deltas = audit.get("deltas") or {}
    av = data.get("adapt_vqe", {}) if isinstance(data.get("adapt_vqe"), dict) else {}
    rows[label] = {
        "output_json": str(path),
        "pool_type": av.get("pool_type"),
        "pool_size": av.get("pool_size"),
        "ansatz_depth": av.get("ansatz_depth", data.get("ansatz_depth")),
        "stop_reason": av.get("stop_reason", data.get("stop_reason")),
        "exact_abs_delta_e": av.get("abs_delta_e", data.get("abs_delta_e")),
        "audit_exact_target_abs_error": deltas.get("exact_target_abs_error"),
        "audit_exact_target_delta_e": deltas.get("exact_target_delta_e"),
        "audit_energy": (audit.get("result") or {}).get("requested_estimate_energy"),
    }
none_val = rows["none"].get("audit_exact_target_abs_error")
if none_val is not None:
    for label in ("readout", "full"):
        val = rows[label].get("audit_exact_target_abs_error")
        rows[label]["audit_abs_error_improvement_vs_none"] = (None if val is None else float(none_val) - float(val))
summary = {
    "status": "completed",
    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "primary_metric": "final_noise_audit_v1.deltas.exact_target_abs_error",
    "cases": rows,
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
with progress_path.open() as f:
    progress = json.load(f)
progress["status"] = "completed"
progress["completed_at"] = summary["completed_at"]
with progress_path.open("w") as f:
    json.dump(progress, f, indent=2)
    f.write("\n")
print("SUMMARY_WRITTEN", summary_path)
PY
```

## Expected artifact layout

The wrapper above writes:

- `artifacts/agent_runs/<tag>/logs/command.sh`
- `artifacts/agent_runs/<tag>/logs/stdout.log`
- `artifacts/agent_runs/<tag>/logs/stderr.log`
- `artifacts/agent_runs/<tag>/progress.json`
- `artifacts/agent_runs/<tag>/cases/none/logs/{command.sh,stdout.log,stderr.log}`
- `artifacts/agent_runs/<tag>/cases/readout/logs/{command.sh,stdout.log,stderr.log}`
- `artifacts/agent_runs/<tag>/cases/full/logs/{command.sh,stdout.log,stderr.log}`
- `artifacts/agent_runs/<tag>/results/none.json`
- `artifacts/agent_runs/<tag>/results/readout.json`
- `artifacts/agent_runs/<tag>/results/full.json`
- `artifacts/agent_runs/<tag>/results/summary.json`

## Notes for the terminal agent

- Run this from the repo root.
- Keep this route **local** and **fake-backend**; do not silently switch to Runtime / paid hardware.
- This surface is intentionally L=2-only because it depends on `pareto_lean_l2`.
- The first depth is slow because each candidate probe uses repeated noisy oracle evaluations.
- The primary scalar to compare across the three lanes is:
  - `final_noise_audit_v1.deltas.exact_target_abs_error`

## Provenance

This note is anchored to the validated lean-pool evidence and the restarted local noisy run setup from 2026-04-05.
