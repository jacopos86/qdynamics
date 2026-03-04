#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUNNER="${REPO_ROOT}/pipelines/exact_bench/hh_noise_model_repo_guide.py"
JSON_OUT="${REPO_ROOT}/artifacts/json/hh_noise_model_repo_guide_index.json"
PDF_OUT="${REPO_ROOT}/docs/HH_noise_model_repo_guide.pdf"

if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python is required." >&2
  exit 2
fi

if [[ ! -f "${RUNNER}" ]]; then
  echo "ERROR: runner missing: ${RUNNER}" >&2
  exit 1
fi

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
import os

path = Path(os.environ["JSON_OUT"])
payload = json.loads(path.read_text(encoding="utf-8"))
required = [
    "scope",
    "sources",
    "modules",
    "symbols",
    "cli_flags",
    "import_edges",
    "docs",
    "tests",
    "generated_at_utc",
    "run_command",
]
missing = [k for k in required if k not in payload]
if missing:
    raise SystemExit("ERROR: missing required top-level JSON keys: " + ", ".join(missing))

roots = payload.get("scope", {}).get("root_modules", [])
required_roots = {
    "pipelines.exact_bench.noise_oracle_runtime",
    "pipelines.exact_bench.hh_noise_hardware_validation",
    "pipelines.exact_bench.hh_noise_robustness_seq_report",
    "pipelines.exact_bench.hh_seq_transition_utils",
}
if not required_roots.issubset(set(roots)):
    raise SystemExit("ERROR: JSON scope missing required root modules.")

if len(payload.get("symbols", [])) < 30:
    raise SystemExit("ERROR: symbol index unexpectedly small (<30).")

print("JSON schema gate: PASS")
PY

PDF_OUT="${PDF_OUT}" python - <<'PY'
from pathlib import Path
import os

try:
    from PyPDF2 import PdfReader
except Exception as exc:
    raise SystemExit(f"ERROR: PyPDF2 required for PDF gate: {exc}")

pdf = Path(os.environ["PDF_OUT"])
reader = PdfReader(str(pdf))
if len(reader.pages) < 1:
    raise SystemExit("ERROR: PDF has no pages")

p1 = (reader.pages[0].extract_text() or "")
required_manifest = [
    "PARAMETER MANIFEST",
    "Model",
    "Ansatz",
    "Drive enabled",
    "t (hopping)",
    "U (interaction)",
    "dv (disorder)",
]
missing_manifest = [x for x in required_manifest if x not in p1]
if missing_manifest:
    raise SystemExit("ERROR: parameter manifest gate failed; missing: " + ", ".join(missing_manifest))

full_text = "\n".join((page.extract_text() or "") for page in reader.pages)
required_sections = [
    "SECTION: SCOPE + SOURCES",
    "SECTION: ARCHITECTURE MAP",
    "SECTION: ORACLE CONTRACT",
    "SECTION: VALIDATION PIPELINE CONTRACT",
    "SECTION: ROBUSTNESS REPORT CONTRACT",
    "SECTION: TEST + DOC COVERAGE",
    "SECTION: REFACTOR PLAYBOOK",
    "SECTION: APPENDIX SYMBOL INDEX",
    "SECTION: APPENDIX CLI FLAG INDEX",
    "SECTION: APPENDIX IMPORT EDGES",
    "SECTION: COMMAND + PROVENANCE",
]
missing = [s for s in required_sections if s not in full_text]
if missing:
    raise SystemExit("ERROR: required section markers missing: " + ", ".join(missing))

if "SECTION: APPENDIX SYMBOL INDEX" not in full_text:
    raise SystemExit("ERROR: symbol appendix marker missing")

print("PDF gate: PASS")
PY

echo "OK: HH noise model repo guide generated"
echo "  JSON: ${JSON_OUT}"
echo "  PDF : ${PDF_OUT}"
