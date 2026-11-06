#!/usr/bin/env bash
# Pre-upload gate for this lane: rehearse the inner runner from the bundle's
# own contents and nothing else.
#
# The bundle is extracted into a scratch root that lies inside no checkout,
# with no symlink to the worktree, and the runner is exercised there at
# h = 1e-8: the import gate, a first segment, the continuation the lane's
# generator writes from that segment's receipt, the negative case of a
# driver that dies before its output directory exists, and the lifecycle
# validator on every submit file in the extracted lane.  What the runner sees
# is what the execute node sees: paper_5/src, this lane, the validator.
#
# Why the scratch root must lie outside every checkout: paper5.repo locates
# the parent repository by climbing the *resolved* ancestors of
# paper_5/src/paper5/repo.py for src/quantum/pauli_polynomial_class.py.  A
# copy under a checkout's output/, or a symlink to the worktree's paper_5,
# finds the parent and passes a rehearsal the execute node fails; cluster
# 10554184 died at import that way after two such rehearsals had passed.
# This script refuses such a root.
#
# Usage, from the worktree root, with a python3 that carries numpy, scipy,
# cvxpy and clarabel first on PATH (the runner's pip bootstrap must never
# trigger in a rehearsal, and this script refuses to start if it would):
#
#   PATH=/Users/lordskyler/Holstein_test/.venv/bin:$PATH \
#     bash chtc/paper_v_cwrmf_event_guard_accel_20260915/rehearse_bundle.sh \
#       [bundle.tgz] [scratch-root]
#
# Defaults: the bundle build_bundle.py wrote; a fresh mktemp -d under
# ${TMPDIR:-/tmp}.  On success the last line is
#   BUNDLE_REHEARSAL_OK <sha256>
# and that digest is appended to bundle.rehearsed beside the bundle; the
# digest must equal bundle.sha256 for the bundle to be uploaded.  Any other
# outcome is a failed gate: the last line is BUNDLE_REHEARSAL_FAILED <step>
# and the scratch root is left in place to read.
set -uo pipefail

LANE="chtc/paper_v_cwrmf_event_guard_accel_20260915"
WORKTREE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
BUNDLE="${1:-${WORKTREE}/output/local_runs/paper_v_cwrmf_event_guard_accel_20260915/paper_v_cwrmf_event_guard_accel_20260915.tgz}"
SCRATCH="${2:-}"
PARENT_MARKER="src/quantum/pauli_polynomial_class.py"
MISSING_PARENT_MESSAGE="Could not locate parent repo root with src/quantum."

step() { printf '\n=== %s ===\n' "$*"; }
fail() { printf '\nBUNDLE_REHEARSAL_FAILED %s\n' "$1" >&2; exit "${2:-1}"; }

step "0 the bundle"
[[ -f "${BUNDLE}" ]] || fail "bundle-missing:${BUNDLE}" 2
if command -v sha256sum >/dev/null 2>&1; then
  BUNDLE_SHA="$(sha256sum "${BUNDLE}" | awk '{print $1}')"
else
  BUNDLE_SHA="$(shasum -a 256 "${BUNDLE}" | awk '{print $1}')"
fi
BUNDLE_BYTES="$(wc -c < "${BUNDLE}" | tr -d ' ')"
echo "bundle ${BUNDLE}"
echo "sha256 ${BUNDLE_SHA} bytes ${BUNDLE_BYTES}"

step "1 a scratch root inside no checkout"
if [[ -z "${SCRATCH}" ]]; then
  SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/paper_v_cwrmf_bundle_rehearsal.XXXXXX")" || fail "mktemp" 2
else
  mkdir -p "${SCRATCH}" || fail "scratch-root-mkdir:${SCRATCH}" 2
  [[ -z "$(ls -A "${SCRATCH}")" ]] || fail "scratch-root-not-empty:${SCRATCH}" 2
fi
SCRATCH="$(cd "${SCRATCH}" && pwd -P)"
probe="${SCRATCH}"
while :; do
  [[ -e "${probe}/${PARENT_MARKER}" ]] && fail "scratch-root-under-a-checkout:${probe}" 2
  [[ "${probe}" == "/" ]] && break
  probe="$(dirname "${probe}")"
done
case "${SCRATCH}/" in
  "${WORKTREE}/"*) fail "scratch-root-inside-the-worktree:${SCRATCH}" 2 ;;
esac
echo "scratch root ${SCRATCH}"

step "2 extract; nothing may link outside"
tar -xzf "${BUNDLE}" -C "${SCRATCH}" || fail "extract"
links="$(find "${SCRATCH}" -type l | head -5)"
[[ -z "${links}" ]] || { echo "${links}" >&2; fail "symlinks-in-bundle" 2; }
[[ -d "${SCRATCH}/paper_5/src/paper5" ]] || fail "bundle-lacks-paper_5/src" 2
[[ -f "${SCRATCH}/${LANE}/run_cwrmf_event_guard_accel.sh" ]] || fail "bundle-lacks-the-runner" 2
[[ -f "${SCRATCH}/${LANE}/make_resume_submit.py" ]] || fail "bundle-lacks-the-generator" 2
[[ -f "${SCRATCH}/chtc/validate_condor_submit_lifecycle.py" ]] || fail "bundle-lacks-the-validator" 2
chmod +x "${SCRATCH}/${LANE}"/*.sh
( cd "${SCRATCH}" && ls -1 && ls -1 "${LANE}" )

step "3 the interpreter the runner will call"
command -v python3 || fail "no-python3" 2
python3 --version
python3 -c "import numpy, scipy, cvxpy, clarabel" 2>/dev/null \
  || fail "conic-stack-missing-on-this-python3 (put a venv with numpy scipy cvxpy clarabel first on PATH; the runner's pip bootstrap must not run in a rehearsal)" 2

cd "${SCRATCH}" || fail "cd-scratch"
unset PYTHONPATH PYTHONSAFEPATH

step "4 import gate: paper5 and the driver from paper_5/src alone"
if ! where="$(PYTHONPATH=paper_5/src python3 -c 'import pathlib, paper5, paper5.stability.apcm_carried_witness_analysis; print(pathlib.Path(paper5.__file__).resolve())' 2>import_gate.err)"; then
  cat import_gate.err >&2
  if grep -q "${MISSING_PARENT_MESSAGE}" import_gate.err; then
    echo "the bundle's paper_5/src needs the parent repository at import: the failure class of cluster 10554184" >&2
  fi
  fail "import-gate"
fi
echo "paper5 imported from ${where}"
case "${where}/" in
  "${SCRATCH}/"*) ;;
  *) fail "import-gate-resolved-outside-the-scratch-root:${where}" ;;
esac
PYTHONPATH=paper_5/src python3 -m paper5.stability.apcm_carried_witness_analysis --help > /dev/null \
  || fail "driver-help"

receipt_of() {  # receipt_of <run_id>
  python3 - "$1" <<'PY'
import json, sys
from pathlib import Path
for path in sorted(Path("archives_by_receipt").glob("*/receipt.json")):
    if json.loads(path.read_text())["run_id"] == sys.argv[1]:
        print(path)
        break
PY
}
receipt_field() {  # receipt_field <receipt> <field>
  python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"
}
show_receipt() {
  python3 -c 'import json, sys; r = json.load(open(sys.argv[1])); print({k: r[k] for k in ("run_id", "final_time", "initial_state_file", "expected_initial_time", "stop_reason", "initial_time", "last_time", "driver_exit_status")})' "$1"
}
run_submit_arguments() {  # run the runner with the arguments/environment line of a .sub
  python3 - "$1" <<'PY' > rehearsal_invocation.sh
import shlex, sys
from pathlib import Path
arguments, environment = None, []
for line in Path(sys.argv[1]).read_text().splitlines():
    stripped = line.strip()
    if stripped.startswith("arguments = "):
        arguments = stripped.split("=", 1)[1].split()
    elif stripped.startswith("environment = "):
        environment = shlex.split(stripped.split("=", 1)[1])[0].split()
for entry in environment:
    print(f"export {entry}")
print("bash chtc/paper_v_cwrmf_event_guard_accel_20260915/run_cwrmf_event_guard_accel.sh "
      + " ".join(shlex.quote(a) for a in arguments))
PY
  cat rehearsal_invocation.sh
  bash rehearsal_invocation.sh
}

step "5 segment 01 from the prepared ground state"
bash "${LANE}/run_cwrmf_event_guard_accel.sh" rbundle_s01_t0000_t2000 2e-8 1e-8 2e-4 none none
status=$?; echo "runner exit: ${status}"
[[ "${status}" -eq 0 ]] || fail "segment-01-runner-exit-${status}"
S01="$(receipt_of rbundle_s01_t0000_t2000)"
[[ -n "${S01}" ]] || fail "segment-01-no-receipt"
show_receipt "${S01}"
[[ "$(receipt_field "${S01}" stop_reason)" == "final_time" ]] || fail "segment-01-stop-reason"
[[ "$(receipt_field "${S01}" driver_exit_status)" == "0" ]] || fail "segment-01-driver-exit"

step "6 the continuation the generator writes from that receipt, run"
python3 "${LANE}/make_resume_submit.py" "${S01}" \
  --segment-length 2e-8 --campaign-final-time 6e-8 --time-label-scale 1e11 || fail "generator"
S02_SUB="$(ls -1 "${LANE}"/submit_cwrmf_accel_rbundle_s02_t2000_t4000_*.sub 2>/dev/null | head -1)"
[[ -n "${S02_SUB}" ]] || fail "generator-wrote-nothing"
run_submit_arguments "${S02_SUB}"
status=$?; echo "runner exit: ${status}"
[[ "${status}" -eq 0 ]] || fail "segment-02-runner-exit-${status}"
S02="$(receipt_of rbundle_s02_t2000_t4000)"
[[ -n "${S02}" ]] || fail "segment-02-no-receipt"
show_receipt "${S02}"
[[ "$(receipt_field "${S02}" stop_reason)" == "final_time" ]] || fail "segment-02-stop-reason"
[[ "$(receipt_field "${S02}" initial_time)" == "2e-08" ]] || fail "segment-02-initial-time"

step "7 negative: a driver that dies before its output directory exists still leaves a receipt"
bash "${LANE}/run_cwrmf_event_guard_accel.sh" rbundle_s99_badfloor 2e-8 1e-8 not-a-number none none
status=$?; echo "runner exit: ${status}"
[[ "${status}" -ne 0 ]] || fail "bad-floor-was-accepted"
S99="$(receipt_of rbundle_s99_badfloor)"
[[ -n "${S99}" ]] || fail "dead-driver-left-no-receipt (the runner died at tar: the failure mode of cluster 10554184)"
show_receipt "${S99}"
[[ "$(receipt_field "${S99}" stop_reason)" == "unrecorded" ]] || fail "dead-driver-stop-reason"
[[ "$(receipt_field "${S99}" driver_exit_status)" != "0" ]] || fail "dead-driver-exit-status"

step "8 lifecycle validator on every submit file in the extracted lane"
python3 chtc/validate_condor_submit_lifecycle.py "${LANE}"/*.sub || fail "lifecycle"
echo LIFECYCLE_OK

step "9 the two trajectories tile the interval"
python3 - <<'PY' || fail "tiling"
import numpy as np
from pathlib import Path
times = {p.parent.name: np.load(p)["times"]
         for p in sorted(Path("raw_outputs").glob("*/*/autonomous_trajectory.npz"))}
for name, t in times.items():
    print(name, np.array2string(t, precision=12))
a = times["rbundle_s01_t0000_t2000"]
b = times["rbundle_s02_t2000_t4000"]
assert np.isclose(a[0], 0.0) and np.isclose(a[-1], 2e-8), a
assert np.isclose(b[0], 2e-8) and np.isclose(b[-1], 4e-8), b
PY

RECORD="$(dirname "${BUNDLE}")/bundle.rehearsed"
printf '%s  %s  %s  %s\n' "${BUNDLE_SHA}" "${BUNDLE_BYTES}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${SCRATCH}" >> "${RECORD}"
echo "recorded in ${RECORD}"
echo "scratch root kept for reading: ${SCRATCH}"
printf '\nBUNDLE_REHEARSAL_OK %s\n' "${BUNDLE_SHA}"
