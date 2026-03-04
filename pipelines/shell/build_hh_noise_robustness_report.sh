#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUNNER="${REPO_ROOT}/pipelines/exact_bench/hh_noise_robustness_seq_report.py"
JSON_OUT="${REPO_ROOT}/artifacts/json/hh_noise_robustness_L2_report.json"
PDF_OUT="${REPO_ROOT}/docs/HH noise robustness report.PDF"

if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python is required." >&2
  exit 2
fi

CORE_FILES=(
  "src/quantum/pauli_letters_module.py"
  "src/quantum/pauli_words.py"
  "src/quantum/qubitization_module.py"
)

for rel in "${CORE_FILES[@]}"; do
  if ! git -C "${REPO_ROOT}" diff --quiet -- "${rel}"; then
    echo "ERROR: operator-core file has unstaged modifications: ${rel}" >&2
    exit 1
  fi
  if ! git -C "${REPO_ROOT}" diff --cached --quiet -- "${rel}"; then
    echo "ERROR: operator-core file has staged modifications: ${rel}" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "${JSON_OUT}")"
mkdir -p "$(dirname "${PDF_OUT}")"

CMD=(
  python "${RUNNER}"
  --output-json "${JSON_OUT}"
  --output-pdf "${PDF_OUT}"
)

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

printf 'Running: '
printf '%q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

if [[ ! -f "${JSON_OUT}" ]]; then
  echo "ERROR: expected JSON output missing: ${JSON_OUT}" >&2
  exit 1
fi

if [[ ! -f "${PDF_OUT}" ]]; then
  echo "ERROR: expected PDF output missing: ${PDF_OUT}" >&2
  exit 1
fi

JSON_OUT="${JSON_OUT}" python - <<'PY'
from pathlib import Path
import json
import math
import os

path = Path(os.environ["JSON_OUT"])
payload = json.loads(path.read_text(encoding="utf-8"))
required = [
    "settings",
    "stage_pipeline",
    "transitions",
    "pool_B",
    "hardcoded_superset",
    "dynamics_noiseless",
    "dynamics_noisy",
    "comparisons",
    "summary",
    "equation_registry",
    "plot_contracts",
    "diagnostics",
]
missing = [k for k in required if k not in payload]
if missing:
    raise SystemExit("ERROR: missing required top-level JSON keys: " + ", ".join(missing))

hardcoded = payload.get("hardcoded_superset", {}).get("profiles", {})
if "static" not in hardcoded:
    raise SystemExit("ERROR: hardcoded superset missing static profile.")
if "drive" not in hardcoded:
    raise SystemExit("ERROR: hardcoded superset missing drive profile.")
for profile in ("static", "drive"):
    traj = hardcoded.get(profile, {}).get("trajectory", [])
    if not traj:
        raise SystemExit(f"ERROR: hardcoded superset profile '{profile}' has empty trajectory.")

eq_registry = payload.get("equation_registry", {})
plot_contracts = payload.get("plot_contracts", {})
if len(eq_registry) < 40:
    raise SystemExit(f"ERROR: equation registry too small ({len(eq_registry)} < 40).")
if len(plot_contracts) < 18:
    raise SystemExit(f"ERROR: plot contracts too small ({len(plot_contracts)} < 18).")

summary = payload.get("summary", {})
completed = int(summary.get("noisy_modes_completed", 0))
if completed < 1:
    raise SystemExit(
        "ERROR: no noisy mode completed. "
        "Use --require-at-least-one-noisy false if you want non-fatal mode availability."
    )

# Numeric finiteness spot checks
stage = payload.get("stage_pipeline", {})
for key in ("warm_start", "adapt_pool_b", "conventional_vqe"):
    rec = stage.get(key, {})
    if "energy" in rec and not math.isfinite(float(rec["energy"])):
        raise SystemExit(f"ERROR: non-finite stage energy for {key}")
    if "delta_abs" in rec and not math.isfinite(float(rec["delta_abs"])):
        raise SystemExit(f"ERROR: non-finite stage delta_abs for {key}")

print("JSON schema gate: PASS")
PY

PDF_OUT="${PDF_OUT}" JSON_OUT="${JSON_OUT}" python - <<'PY'
from pathlib import Path
import json
import os

try:
    from PyPDF2 import PdfReader
except Exception as exc:
    raise SystemExit(f"ERROR: PyPDF2 required for PDF manifest gate: {exc}")

pdf = Path(os.environ["PDF_OUT"])
reader = PdfReader(str(pdf))
if len(reader.pages) < 1:
    raise SystemExit("ERROR: PDF has no pages")

txt = (reader.pages[0].extract_text() or "")
required_snippets = [
    "PARAMETER MANIFEST",
    "Model",
    "Ansatz",
    "Drive enabled",
    "t (hopping)",
    "U (interaction)",
    "dv (disorder)",
]
missing = [s for s in required_snippets if s not in txt]
if missing:
    raise SystemExit("ERROR: parameter manifest gate failed; missing: " + ", ".join(missing))

full_text = "\n".join((page.extract_text() or "") for page in reader.pages)
required_sections = [
    "SECTION: RESULTS SUMMARY",
    "SECTION: RESULTS STAGE TRANSITIONS",
    "SECTION: RESULTS STATIC OVERLAYS",
    "SECTION: RESULTS DRIVE OVERLAYS",
    "SECTION: RESULTS METHOD COMPARISON",
    "SECTION: RESULTS NOISE DETAILS",
    "SECTION: APPENDIX DEFINITIONS USED",
    "SECTION: APPENDIX FORMULA ATLAS",
    "SECTION: APPENDIX PLOT CONTRACTS",
]
missing_sections = [s for s in required_sections if s not in full_text]
if missing_sections:
    raise SystemExit("ERROR: section gate failed; missing: " + ", ".join(missing_sections))

if "[PLOT_CAPTION]" not in full_text:
    raise SystemExit("ERROR: caption gate failed ([PLOT_CAPTION] not found).")

caption_count = full_text.count("[PLOT_CAPTION]")
json_payload = json.loads(Path(os.environ["JSON_OUT"]).read_text(encoding="utf-8"))
is_smoke = bool(json_payload.get("settings", {}).get("smoke_test_intentionally_weak", False))
caption_floor = 12 if is_smoke else 20
if caption_count < caption_floor:
    raise SystemExit(
        f"ERROR: insufficient caption-strip density ({caption_count} < {caption_floor})."
    )

print("PDF manifest gate: PASS")
PY

# Deterministic static-content gate (optional skip via env flag)
if [[ "${HH_NOISE_SKIP_DETERMINISM:-0}" != "1" ]]; then
  TMP1="/tmp/hh_noise_robustness_det_1.json"
  TMP2="/tmp/hh_noise_robustness_det_2.json"
  python "${RUNNER}" \
    --output-json "${TMP1}" \
    --skip-pdf \
    --require-at-least-one-noisy \
    "$@"
  python "${RUNNER}" \
    --output-json "${TMP2}" \
    --skip-pdf \
    --require-at-least-one-noisy \
    "$@"

  TMP1="${TMP1}" TMP2="${TMP2}" python - <<'PY'
from pathlib import Path
import json
import hashlib
import os

def _stable_subset(payload: dict) -> dict:
    return {
        "settings": payload.get("settings", {}),
        "pool_B": payload.get("pool_B", {}),
        "transitions": payload.get("transitions", {}),
        "hardcoded_superset": {
            "profiles": {
                "static": {
                    "final": payload.get("hardcoded_superset", {}).get("profiles", {}).get("static", {}).get("final", {}),
                },
                "drive": {
                    "final": payload.get("hardcoded_superset", {}).get("profiles", {}).get("drive", {}).get("final", {}),
                },
            }
        },
        "equation_registry_count": len(payload.get("equation_registry", {})),
        "plot_contract_count": len(payload.get("plot_contracts", {})),
        "stage_pipeline": {
            "warm_start": {
                "energy": payload.get("stage_pipeline", {}).get("warm_start", {}).get("energy"),
                "delta_abs": payload.get("stage_pipeline", {}).get("warm_start", {}).get("delta_abs"),
                "stop_reason": payload.get("stage_pipeline", {}).get("warm_start", {}).get("stop_reason"),
            },
            "adapt_pool_b": {
                "energy": payload.get("stage_pipeline", {}).get("adapt_pool_b", {}).get("energy"),
                "delta_abs": payload.get("stage_pipeline", {}).get("adapt_pool_b", {}).get("delta_abs"),
                "stop_reason": payload.get("stage_pipeline", {}).get("adapt_pool_b", {}).get("stop_reason"),
            },
            "conventional_vqe": {
                "energy": payload.get("stage_pipeline", {}).get("conventional_vqe", {}).get("energy"),
                "delta_abs": payload.get("stage_pipeline", {}).get("conventional_vqe", {}).get("delta_abs"),
                "stop_reason": payload.get("stage_pipeline", {}).get("conventional_vqe", {}).get("stop_reason"),
            },
        },
        "dynamics_noiseless": payload.get("dynamics_noiseless", {}),
    }

p1 = json.loads(Path(os.environ["TMP1"]).read_text(encoding="utf-8"))
p2 = json.loads(Path(os.environ["TMP2"]).read_text(encoding="utf-8"))

s1 = json.dumps(_stable_subset(p1), sort_keys=True).encode("utf-8")
s2 = json.dumps(_stable_subset(p2), sort_keys=True).encode("utf-8")

h1 = hashlib.sha256(s1).hexdigest()
h2 = hashlib.sha256(s2).hexdigest()
if h1 != h2:
    raise SystemExit("ERROR: deterministic static-content gate failed (hash mismatch).")

print("Determinism gate: PASS")
PY
fi

echo "OK: HH noise robustness report generated"
echo "  JSON: ${JSON_OUT}"
echo "  PDF : ${PDF_OUT}"
