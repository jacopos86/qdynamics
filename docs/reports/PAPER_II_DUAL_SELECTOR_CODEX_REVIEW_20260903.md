# Paper II dual-selector Codex review — 2026-09-03

## Scope

This report records the independent Codex review and hardening of the Fable
implementation ending at commit `895212b6a58dc302636a871bf0fe5a4146242aff`.
It covers the three immutable Paper II selector routes:

1. `paper_ii_exchange_legacy_proxy_fixed_v1` — realized `Q`, with the
   historical singleton-cost regression repaired over the active population;
2. `paper_ii_exchange_q_compiled_v1` — realized stabilized `Q` with the
   complete-patch compiled-resource factor;
3. `paper_ii_exchange_gamma_compiled_v1` — exact projection ordinate
   `Gamma` with the same complete-patch compiled-resource factor.

The final comparison campaign is exactly fifteen already-qualified matched
cells times these three APM routes: 45 fresh `t=0..20`, 501-point, full-exchange
APM trajectories and no new AVQDS or precursor/screen trajectories. Historical
AVQDS rows remain locked comparators.

## Review findings and corrections

The Fable implementation established the main route architecture, but the
review found four scientific-control defects and two execution-control defects
that had to be corrected before production use.

### Scientific controls

- **Regime-local candidate closure.** Large-miss ranking now contains only
  insertion-bearing patches (pure append plus true exchange). Maintenance
  ranking contains only pure deletions plus stay. Inactive faces no longer
  influence the population median/MAD normalization or the winner.
- **Realized gate authority.** Large-miss materialization requires a strict
  realized `L^2` improvement after the complete patch is solved. Maintenance
  uses the realized-`Q` loss budget. Proxy or exact-geometry ranking cannot
  bypass either authority.
- **Shared Q/Gamma frontier.** The compiled Q and Gamma routes seal the same
  candidate population and share one compiled-cost vector. A candidate must
  have finite values for both ordinates to enter that shared scientific
  frontier; this prevents route-dependent missing-data populations.
- **Materialization identity.** The accepted support word is reconstructed
  from the cached compiled placement cuts, then checked against the sealed word
  hash. The committed run payload binds the route contract, population hash,
  cost marginal, selected cuts, decision hash, and gate outcome.

### Execution controls

- **Killable compilation.** Every Qiskit compile runs in a spawned process,
  including the one-worker case. A 300-second hard timeout terminates the
  affected pool; a compiler failure rejects the affected candidate rather than
  silently substituting `K=1`.
- **Honest compile-budget accounting.** The compile-work guard counts only
  candidates in the active regime population, excludes stay and inactive
  faces, and is admitted transactionally with the joint search-work guard.
  The campaign explicitly pins four scoring/compile workers, while preserving
  bitwise scientific parity across 1, 2, and 4 workers.

The review also restored the calibration adapter's historical telemetry alias
for `exact_reference_controller_inputs`. Without this narrow compatibility
fix, verified old trajectories were incorrectly reported as calibration
errors solely because the field lived under the older
`uses_exact_reference_for_decision` spelling.

## Evidence-lock recovery

The final matched-cell lock was originally a deliberately blocked skeleton.
The review added a strict one-way hydrator and recovered its authoritative
post-coverage sources:

- promotion preview: 15 selected pairs;
- ledger: 371 records, 34 qualified `t20` method arms, 2 pending;
- 30 screen plus 30 continuation run payloads, all SHA-verified;
- 30 stitched trajectories, each exactly 501 points on `t=0..20`;
- 15 exact-reference manifests, all SHA-verified;
- final coverage-ingestion receipt and 29-page comprehensive results PDF.

Hydration does not choose cuts or cells. It requires the existing fifteen-cell
lock and the verified promotion preview to agree exactly, verifies every
referenced digest and run lock, and leaves submission blocked until a tracked
refresh-resolution receipt is supplied.

## Verification completed before calibration

- Focused selector/compile/campaign tests: 193 passed, 32 third-party Qiskit
  deprecation warnings, peak process-tree RSS 478 MB under an 8 GB RAM guard.
- Calibration adapter self-test: passed.
- Real-seed smoke trajectories completed for all three routes.
- Q compiled worker-count parity: workers 1, 2, and 4 produced the identical
  scientific decision digest
  `6c48b062d6ada5cd39574162b14949066344c03e22acf1b37adb37643f8c97a8`.
- Q and Gamma routes produced identical cost-population hashes at every smoke
  checkpoint; only the physical ranking ordinate differs.
- Maximum observed smoke process-tree RSS was 1,209 MB.

## Production boundary

No result from the new selectors is adopted into a manuscript table, figure,
source map, or scientific claim by this work. The campaign produces comparison
evidence for later author review. The old visible-results refresh block may be
retired only after the recovered evidence, existing visual-QA receipt, and a
fresh calibration receipt are bound by a tracked resolution receipt.
