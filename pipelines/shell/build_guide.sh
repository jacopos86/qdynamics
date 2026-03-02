#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DOC_MD="${REPO_ROOT}/docs/repo_implementation_guide.md"
PDF_OUT="${REPO_ROOT}/docs/Repo implementation guide.PDF"
ASSETS_DIR="${REPO_ROOT}/docs/repo_guide_assets"
SUMMARY_JSON="${ASSETS_DIR}/repo_guide_summary.json"
ARTIFACT_METRICS_JSON="${ASSETS_DIR}/repo_guide_artifact_metrics.json"
PYTEST_SNAPSHOT="${ASSETS_DIR}/pytest_snapshot.txt"

REQUIRED_SECTIONS=(
  "## 1. Reader Contract And Review Goal (HH-First)"
  "## 2. Non-Negotiable Invariants"
  "## 3. Current Repo Topology And Active Codepaths"
  "## 4. HH Hamiltonian Assembly And JW Contracts"
  "## 5. HF And Reference-State Construction (Hubbard + HH)"
  "## 6. VQE Internals: Ansatz Families, Theta, Restarts"
  "## 7. ADAPT Internals And Pool Semantics (PAOP/LF)"
  "## 8. Trotter Ordering Semantics And Non-Commutation"
  "## 9. Drive Implementation, Reference Split, Safe-Test"
  "## 10. Observable Computations And Trajectory Contracts"
  "## 11. Plot Generation Internals And Interpretation"
  "## 12. Artifact-Grounded Audits (HH Primary, Hubbard Context)"
  "## 13. Extension Safety: Safe And High-Risk Edits"
  "## 14. Ultra-Brief Run Appendix"
)

CANONICAL_HH_CASES=(
  "${REPO_ROOT}/artifacts/json/hc_hh_L2_static_t1.0_U2.0_g1.0_nph1_strong.json"
  "${REPO_ROOT}/artifacts/json/hc_hh_L2_drive_J1_U2_g1_nph1_from_adapt.json"
  "${REPO_ROOT}/artifacts/json/adapt_hh_L2_static_t1.0_U2.0_g1.0_nph1_paop_deep.json"
)

CANONICAL_HUBBARD_CONTEXT_CASES=(
  "${REPO_ROOT}/artifacts/json/hc_hubbard_L3_static_t1.0_U4.0_S128_heavy.json"
  "${REPO_ROOT}/artifacts/json/hc_hubbard_L4_drive_t1.0_U4.0_S256.json"
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
  if ! git -C "${REPO_ROOT}" diff --quiet -- "${rel}"; then
    echo "ERROR: operator-core file has unstaged modifications: ${rel}" >&2
    exit 1
  fi
  if ! git -C "${REPO_ROOT}" diff --cached --quiet -- "${rel}"; then
    echo "ERROR: operator-core file has staged modifications: ${rel}" >&2
    exit 1
  fi
done

# ---------------------------------------------------------------------------
# Canonical artifact presence gate (HH primary required; Hubbard context optional)
# ---------------------------------------------------------------------------

for case_path in "${CANONICAL_HH_CASES[@]}"; do
  if [[ ! -f "${case_path}" ]]; then
    echo "ERROR: required HH canonical artifact missing: ${case_path}" >&2
    exit 1
  fi
done

for case_path in "${CANONICAL_HUBBARD_CONTEXT_CASES[@]}"; do
  if [[ ! -f "${case_path}" ]]; then
    echo "WARN: optional Hubbard context artifact missing: ${case_path}" >&2
  fi
done

# ---------------------------------------------------------------------------
# Regenerate assets and check deterministic re-run
# ---------------------------------------------------------------------------

"${REPO_ROOT}/reports/guide_assets.py" >/tmp/repo_guide_gen_1.json

REPO_ROOT="${REPO_ROOT}" python - <<'PY'
from pathlib import Path
import hashlib
import json
import os

repo_root = Path(os.environ["REPO_ROOT"])
assets = repo_root / "docs" / "repo_guide_assets"
manifest = sorted(p for p in assets.glob("*.png"))
if len(manifest) < 35:
    raise SystemExit(f"ERROR: expected at least 35 generated diagrams, found {len(manifest)}")

sha = {}
for p in manifest:
    sha[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()

(assets / ".determinism_before.json").write_text(json.dumps(sha, indent=2), encoding="utf-8")
PY

"${REPO_ROOT}/reports/guide_assets.py" >/tmp/repo_guide_gen_2.json

REPO_ROOT="${REPO_ROOT}" python - <<'PY'
from pathlib import Path
import hashlib
import json
import os

repo_root = Path(os.environ["REPO_ROOT"])
assets = repo_root / "docs" / "repo_guide_assets"
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
    "_run_hardcoded_adapt_vqe",
    "vqe_minimize",
    "hartree_fock_bitstring",
    "hartree_fock_statevector",
    "hubbard_holstein_reference_state",
    "evaluate_drive_waveform",
    "build_gaussian_sinusoid_density_drive",
    "make_pool",
    "_make_paop_core",
    "jw_current_hop",
}
ev = summary.get("implementation_evidence", {})
found = set(ev.get("found_functions", []))
missing = sorted(required_functions - found)
if missing:
    raise SystemExit("ERROR: implementation evidence missing required function anchors: " + ", ".join(missing))

# Artifact metrics gate
cases = metrics.get("cases", [])
if len(cases) < 3:
    raise SystemExit(f"ERROR: expected at least 3 canonical cases in artifact metrics, found {len(cases)}")
hh_primary = [c for c in cases if c.get("group") == "hh_primary"]
if len(hh_primary) < 3:
    raise SystemExit(f"ERROR: expected 3 HH primary cases in artifact metrics, found {len(hh_primary)}")
invalid_hh = [c.get("case", "?") for c in hh_primary if not bool(c.get("valid", False))]
if invalid_hh:
    raise SystemExit("ERROR: invalid HH primary artifact metrics for case(s): " + ", ".join(invalid_hh))

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
if [[ "${DIAGRAM_REF_COUNT}" -lt 35 ]]; then
  echo "ERROR: expected at least 35 embedded diagram references, found ${DIAGRAM_REF_COUNT}" >&2
  exit 1
fi

DOC_MD="${DOC_MD}" SUMMARY_JSON="${SUMMARY_JSON}" python - <<'PY'
from pathlib import Path
import re
import os

doc = Path(os.environ["DOC_MD"])
docs_dir = doc.parent
text = doc.read_text(encoding="utf-8")
summary = Path(os.environ["SUMMARY_JSON"]).read_text(encoding="utf-8")

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

# Stale-path gate
deprecated = "Fermi-Hamil-JW-VQE-TROTTER-PIPELINE"
if deprecated in text:
    raise SystemExit(f"ERROR: deprecated path prefix found in markdown: {deprecated}")
if deprecated in summary:
    raise SystemExit(f"ERROR: deprecated path prefix found in summary json: {deprecated}")
PY

# ---------------------------------------------------------------------------
# Test snapshot gate
# ---------------------------------------------------------------------------

pushd "${REPO_ROOT}" >/dev/null
pytest -q | tee "${PYTEST_SNAPSHOT}"
popd >/dev/null

if ! grep -Eq '[0-9]+ passed' "${PYTEST_SNAPSHOT}"; then
  echo "ERROR: pytest snapshot did not report passing tests." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Build PDF
# ---------------------------------------------------------------------------

pushd "${REPO_ROOT}" >/dev/null
pandoc docs/repo_implementation_guide.md --resource-path=docs --from gfm --pdf-engine=pdflatex --toc --number-sections -V geometry:margin=0.8in -V fontsize=10pt -o "docs/Repo implementation guide.PDF"
popd >/dev/null

if [[ ! -f "${PDF_OUT}" ]]; then
  echo "ERROR: expected PDF not generated: ${PDF_OUT}" >&2
  exit 1
fi

echo "Build complete: ${PDF_OUT}"
echo "Summary JSON:  ${SUMMARY_JSON}"
echo "Metrics JSON:  ${ARTIFACT_METRICS_JSON}"
echo "Test snapshot: ${PYTEST_SNAPSHOT}"
