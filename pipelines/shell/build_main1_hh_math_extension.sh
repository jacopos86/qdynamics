#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SOURCE_MD="${REPO_ROOT}/docs/main1_hh_math_extension.md"
OUTPUT_PDF="${REPO_ROOT}/docs/Main1_HH_Math_Extension.pdf"
MIN_PAGES=55
MAX_PAGES=60
STRICT_MANIFEST=1

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --source-md <path>        Markdown source (default: docs/main1_hh_math_extension.md)
  --output-pdf <path>       Output PDF (default: docs/Main1_HH_Math_Extension.pdf)
  --min-pages <int>         Minimum allowed page count (default: 55)
  --max-pages <int>         Maximum allowed page count (default: 60)
  --strict-manifest         Enforce first-page manifest key checks (default: on)
  --no-strict-manifest      Disable first-page manifest key checks
  -h, --help                Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-md)
      SOURCE_MD="$2"
      shift 2
      ;;
    --output-pdf)
      OUTPUT_PDF="$2"
      shift 2
      ;;
    --min-pages)
      MIN_PAGES="$2"
      shift 2
      ;;
    --max-pages)
      MAX_PAGES="$2"
      shift 2
      ;;
    --strict-manifest)
      STRICT_MANIFEST=1
      shift
      ;;
    --no-strict-manifest)
      STRICT_MANIFEST=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if ! command -v pandoc >/dev/null 2>&1; then
  echo "ERROR: pandoc is required" >&2
  exit 2
fi
if ! command -v pdflatex >/dev/null 2>&1; then
  echo "ERROR: pdflatex is required" >&2
  exit 2
fi
if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python is required" >&2
  exit 2
fi

if [[ ! -f "${SOURCE_MD}" ]]; then
  echo "ERROR: source markdown not found: ${SOURCE_MD}" >&2
  exit 1
fi

REQUIRED_HEADINGS=(
  "# 1. Parameter Manifest and Reader Contract"
  "# 8. Inner Optimizer Deep Dive"
  "# 9. ADAPT Math and Selection Mechanics"
  "# 10. HH-VA (Termwise and Layerwise)"
  "# 11. PAOP/PI Operator Pool Deep Dive"
)

for h in "${REQUIRED_HEADINGS[@]}"; do
  if ! grep -Fq "$h" "${SOURCE_MD}"; then
    echo "ERROR: missing required heading: $h" >&2
    exit 1
  fi
done

if ! grep -Fq "e/x/y/z" "${SOURCE_MD}"; then
  echo "ERROR: notation check failed: missing internal Pauli notation e/x/y/z" >&2
  exit 1
fi
if ! grep -Fq "q_{n-1}" "${SOURCE_MD}"; then
  echo "ERROR: notation check failed: missing q_(n-1)...q_0 statement" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_PDF}")"

pushd "${REPO_ROOT}" >/dev/null
pandoc "${SOURCE_MD}" \
  --from markdown+tex_math_dollars \
  --pdf-engine=pdflatex \
  --toc \
  --number-sections \
  -V geometry:margin=0.8in \
  -V fontsize=10pt \
  -o "${OUTPUT_PDF}"
popd >/dev/null

if [[ ! -f "${OUTPUT_PDF}" ]]; then
  echo "ERROR: expected PDF not generated: ${OUTPUT_PDF}" >&2
  exit 1
fi

OUTPUT_PDF="${OUTPUT_PDF}" MIN_PAGES="${MIN_PAGES}" MAX_PAGES="${MAX_PAGES}" STRICT_MANIFEST="${STRICT_MANIFEST}" python - <<'PY'
from pathlib import Path
import os
import re
from PyPDF2 import PdfReader

pdf_path = Path(os.environ["OUTPUT_PDF"])
min_pages = int(os.environ["MIN_PAGES"])
max_pages = int(os.environ["MAX_PAGES"])
strict_manifest = int(os.environ["STRICT_MANIFEST"])

reader = PdfReader(str(pdf_path))
pages = len(reader.pages)
if pages < min_pages:
    raise SystemExit(f"ERROR: PDF page count below threshold: {pages} < {min_pages}")
if pages > max_pages:
    raise SystemExit(f"ERROR: PDF page count above threshold: {pages} > {max_pages}")

scan_pages = min(5, len(reader.pages))
scan_text = []
for idx in range(scan_pages):
    scan_text.append(reader.pages[idx].extract_text() or "")
early_text = "\n".join(scan_text)
if strict_manifest:
    required_patterns = [
        r"Model family/name",
        r"Ansatz types covered",
        r"Drive enabled",
        r"Core physical parameters",
        r"HH-defining parameters",
    ]
    missing = [pat for pat in required_patterns if re.search(pat, early_text, re.IGNORECASE) is None]
    if missing:
        raise SystemExit(
            "ERROR: strict-manifest failed in early-summary pages (1-5); missing keys: "
            + ", ".join(missing)
        )

print(f"PDF page count OK: {pages}")
print("First-page manifest check: " + ("strict-pass" if strict_manifest else "skipped"))
PY

echo "Build complete: ${OUTPUT_PDF}"
echo "Source: ${SOURCE_MD}"
echo "Page window: [${MIN_PAGES}, ${MAX_PAGES}]"
