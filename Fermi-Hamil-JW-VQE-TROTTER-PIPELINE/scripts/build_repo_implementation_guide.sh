#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBREPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${SUBREPO_ROOT}/.." && pwd)"

DOC_MD="${SUBREPO_ROOT}/docs/repo_implementation_guide.md"
PDF_OUT="${SUBREPO_ROOT}/docs/Repo implementation guide.PDF"
ASSETS_DIR="${SUBREPO_ROOT}/docs/repo_guide_assets"
SUMMARY_JSON="${ASSETS_DIR}/repo_guide_summary.json"
ARTIFACT_METRICS_JSON="${ASSETS_DIR}/repo_guide_artifact_metrics.json"
PYTEST_SNAPSHOT="${ASSETS_DIR}/pytest_snapshot.txt"

REQUIRED_SECTIONS=(
  "## 1. Reader Contract And Review Goal"
  "## 2. Physics-To-Code Invariants"
  "## 3. Hamiltonian Assembly And JW Contracts"
  "## 4. Hartree-Fock Construction And Half-Filling Placement"
  "## 5. VQE Internals: Ansatz, Theta, And Inner Optimization"
  "## 6. Trotter Dynamics Internals And Ordering Semantics"
  "## 7. Time-Dependent Drive Implementation"
  "## 8. Exact Reference Semantics And exact_steps_multiplier"
  "## 9. Trajectory Observables: Formulas To Code"
  "## 10. Plot Generation Internals And Meaning"
  "## 11. Canonical Artifact Audits (L=2,3,4)"
  "## 12. Minimal Compare/Qiskit Validation Role"
  "## 13. Extension Safety: Safe And High-Risk Edits"
  "## 14. Ultra-Brief Run Appendix"
)

CANONICAL_CASES=(
  "${SUBREPO_ROOT}/artifacts/json/H_L2_static_t1.0_U4.0_S64_heavy.json"
  "${SUBREPO_ROOT}/artifacts/json/H_L3_static_t1.0_U4.0_S128_heavy.json"
  "${SUBREPO_ROOT}/artifacts/json/H_L4_vt_t1.0_U4.0_S256_dyn.json"
)

if ! command -v pandoc >/dev/null 2>&1; then
  echo "ERROR: pandoc is required." >&2
  exit 2
fi
if ! command -v pdflatex >/dev/null 2>&1; then
  echo "ERROR: pdflatex is required." >&2
  exit 2
fi
if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python is required." >&2
  exit 2
fi

mkdir -p "${ASSETS_DIR}"

# ---------------------------------------------------------------------------
# Operator-core immutability gate
# ---------------------------------------------------------------------------

CORE_FILES=(
  "src/quantum/pauli_letters_module.py"
  "src/quantum/pauli_words.py"
  "src/quantum/qubitization_module.py"
)

for rel in "${CORE_FILES[@]}"; do
  if ! git -C "${WORKSPACE_ROOT}" diff --quiet -- "${rel}"; then
    echo "ERROR: operator-core file has unstaged modifications: ${rel}" >&2
    exit 1
  fi
  if ! git -C "${WORKSPACE_ROOT}" diff --cached --quiet -- "${rel}"; then
    echo "ERROR: operator-core file has staged modifications: ${rel}" >&2
    exit 1
  fi
done

# ---------------------------------------------------------------------------
# Canonical artifact presence gate
# ---------------------------------------------------------------------------

for case_path in "${CANONICAL_CASES[@]}"; do
  if [[ ! -f "${case_path}" ]]; then
    echo "ERROR: required canonical artifact missing: ${case_path}" >&2
    exit 1
  fi
done

# ---------------------------------------------------------------------------
# Regenerate assets and check deterministic re-run
# ---------------------------------------------------------------------------

"${SUBREPO_ROOT}/scripts/generate_repo_guide_assets.py" >/tmp/repo_guide_gen_1.json

SUBREPO_ROOT="${SUBREPO_ROOT}" python - <<'PY'
from pathlib import Path
import hashlib
import json
import os

subrepo = Path(os.environ["SUBREPO_ROOT"])
assets = subrepo / "docs" / "repo_guide_assets"
manifest = sorted(p for p in assets.glob("*.png"))
if len(manifest) < 30:
    raise SystemExit(f"ERROR: expected at least 30 generated diagrams, found {len(manifest)}")

sha = {}
for p in manifest:
    sha[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()

(assets / ".determinism_before.json").write_text(json.dumps(sha, indent=2), encoding="utf-8")
PY

"${SUBREPO_ROOT}/scripts/generate_repo_guide_assets.py" >/tmp/repo_guide_gen_2.json

SUBREPO_ROOT="${SUBREPO_ROOT}" python - <<'PY'
from pathlib import Path
import hashlib
import json
import os

subrepo = Path(os.environ["SUBREPO_ROOT"])
assets = subrepo / "docs" / "repo_guide_assets"
before_path = assets / ".determinism_before.json"
before = json.loads(before_path.read_text(encoding="utf-8"))
after = {}
for p in sorted(assets.glob("*.png")):
    after[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()

if before != after:
    raise SystemExit("ERROR: diagram generation is not deterministic across consecutive runs")

before_path.unlink(missing_ok=True)
PY

if [[ ! -f "${SUMMARY_JSON}" ]]; then
  echo "ERROR: expected summary JSON not found: ${SUMMARY_JSON}" >&2
  exit 1
fi
if [[ ! -f "${ARTIFACT_METRICS_JSON}" ]]; then
  echo "ERROR: expected artifact metrics JSON not found: ${ARTIFACT_METRICS_JSON}" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Summary/evidence/contrast gates
# ---------------------------------------------------------------------------

SUMMARY_JSON="${SUMMARY_JSON}" ARTIFACT_METRICS_JSON="${ARTIFACT_METRICS_JSON}" python - <<'PY'
from pathlib import Path
import json
import os
import re

summary = json.loads(Path(os.environ["SUMMARY_JSON"]).read_text(encoding="utf-8"))
metrics = json.loads(Path(os.environ["ARTIFACT_METRICS_JSON"]).read_text(encoding="utf-8"))

# Evidence gate
required_functions = {
    "_simulate_trajectory",
    "_evolve_trotter_suzuki2_absolute",
    "_evolve_piecewise_exact",
    "_site_resolved_number_observables",
    "_spin_orbital_bit_index",
    "_run_hardcoded_vqe",
    "vqe_minimize",
    "hartree_fock_bitstring",
    "hartree_fock_statevector",
}
ev = summary.get("implementation_evidence", {})
found = set(ev.get("found_functions", []))
missing = sorted(required_functions - found)
if missing:
    raise SystemExit("ERROR: implementation evidence missing required function anchors: " + ", ".join(missing))

# Artifact metrics gate
cases = metrics.get("cases", [])
if len(cases) < 3:
    raise SystemExit(f"ERROR: expected 3 canonical cases in artifact metrics, found {len(cases)}")
invalid = [c.get("case", "?") for c in cases if not bool(c.get("valid", False))]
if invalid:
    raise SystemExit("ERROR: invalid canonical artifact metrics for case(s): " + ", ".join(invalid))

# Contrast gate
style = summary.get("style_palette", {})
for key in ["text_primary", "text_secondary", "edge_dark", "edge_mid"]:
    if key not in style:
        raise SystemExit(f"ERROR: missing style palette key in summary: {key}")

hex_re = re.compile(r"^#[0-9A-Fa-f]{6}$")
for key in ["text_primary", "text_secondary"]:
    val = str(style.get(key, ""))
    if not hex_re.match(val):
        raise SystemExit(f"ERROR: invalid hex color for {key}: {val}")
    # Ensure text is dark enough (simple luminance heuristic)
    r = int(val[1:3], 16)
    g = int(val[3:5], 16)
    b = int(val[5:7], 16)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if lum > 120:
        raise SystemExit(f"ERROR: {key} is too light for white background: {val}")
PY

# ---------------------------------------------------------------------------
# Markdown content gates
# ---------------------------------------------------------------------------

if [[ ! -f "${DOC_MD}" ]]; then
  echo "ERROR: guide markdown not found: ${DOC_MD}" >&2
  exit 1
fi

for section in "${REQUIRED_SECTIONS[@]}"; do
  if ! grep -Fq "${section}" "${DOC_MD}"; then
    echo "ERROR: missing required section heading: ${section}" >&2
    exit 1
  fi
done

DIAGRAM_REF_COUNT="$(grep -Eo 'repo_guide_assets/[0-9]{2}_[^)]+\.png' "${DOC_MD}" | sort -u | wc -l | tr -d ' ')"
if [[ "${DIAGRAM_REF_COUNT}" -lt 30 ]]; then
  echo "ERROR: expected at least 30 embedded diagram references, found ${DIAGRAM_REF_COUNT}" >&2
  exit 1
fi

DOC_MD="${DOC_MD}" python - <<'PY'
from pathlib import Path
import re
import os

doc = Path(os.environ["DOC_MD"])
docs_dir = doc.parent
text = doc.read_text(encoding="utf-8")

# Local link/path gate (ignore remote URLs and anchors)
link_re = re.compile(r'\[[^\]]+\]\(([^)]+)\)')
missing = []
for m in link_re.finditer(text):
    raw = m.group(1).strip()
    if raw.startswith(("http://", "https://", "mailto:", "#")):
        continue
    target = (docs_dir / raw).resolve()
    if not target.exists():
        missing.append(raw)
if missing:
    raise SystemExit("ERROR: missing local markdown link targets: " + ", ".join(sorted(set(missing))))

# Ultra-brief run appendix cap: <= 220 words, <= 40 non-blank lines
lines = text.splitlines()
start = None
end = len(lines)
for i, line in enumerate(lines):
    if line.strip() == "## 14. Ultra-Brief Run Appendix":
        start = i + 1
        continue
    if start is not None and line.startswith("## ") and i > start:
        end = i
        break
if start is None:
    raise SystemExit("ERROR: missing ultra-brief run appendix heading")

section_lines = lines[start:end]
non_blank = [ln for ln in section_lines if ln.strip()]
words = sum(len(re.findall(r"\b\w+\b", ln)) for ln in non_blank)
if len(non_blank) > 40:
    raise SystemExit(f"ERROR: run appendix too long by lines: {len(non_blank)} > 40")
if words > 220:
    raise SystemExit(f"ERROR: run appendix too long by words: {words} > 220")
PY

# ---------------------------------------------------------------------------
# Test snapshot gate
# ---------------------------------------------------------------------------

pushd "${WORKSPACE_ROOT}" >/dev/null
pytest -q | tee "${PYTEST_SNAPSHOT}"
popd >/dev/null

if ! grep -Eq '[0-9]+ passed' "${PYTEST_SNAPSHOT}"; then
  echo "ERROR: pytest snapshot did not report passing tests." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Build PDF
# ---------------------------------------------------------------------------

pushd "${SUBREPO_ROOT}" >/dev/null
pandoc docs/repo_implementation_guide.md --resource-path=docs --from gfm --pdf-engine=pdflatex --toc --number-sections -V geometry:margin=0.8in -V fontsize=10pt -o "docs/Repo implementation guide.PDF"
popd >/dev/null

if [[ ! -f "${PDF_OUT}" ]]; then
  echo "ERROR: expected PDF not generated: ${PDF_OUT}" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# PDF quality gate
# ---------------------------------------------------------------------------

PDF_OUT="${PDF_OUT}" python - <<'PY'
from pathlib import Path
from PyPDF2 import PdfReader
import os

pdf = Path(os.environ["PDF_OUT"])
reader = PdfReader(str(pdf))
pages = len(reader.pages)
if pages < 60:
    raise SystemExit(f"ERROR: PDF page count below threshold: {pages} < 60")
if pages > 75:
    raise SystemExit(f"ERROR: PDF page count above threshold: {pages} > 75")
print(f"PDF page count OK: {pages}")
PY

echo "Build complete: ${PDF_OUT}"
echo "Summary JSON:  ${SUMMARY_JSON}"
echo "Metrics JSON:  ${ARTIFACT_METRICS_JSON}"
echo "Test snapshot: ${PYTEST_SNAPSHOT}"
