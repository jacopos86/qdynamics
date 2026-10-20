# Independent review: divergence parameter map, and the stopped weak-cutoff increment 024 (Claude, 2026-10-18)

*Revised 2026-10-18, text only, under `classical-review-scope-20261018-003`: cutoff statements qualified as observed finite-cutoff event agreement and sensitivity (see the disposition block at the end).*

**Verdict.**
1. **The original map is accurate and complete within its scope.** Its
   headline counts recompute exactly from the committed grid files:
   - exact-correlated grid: 41 of 54 cells fire by `t = 2000`, 34 of 54
     by `500`;
   - Hartree–Fock grid: 44 of 54 by `2000`, 42 by `500`;
   - low coupling: 8 of 18 exact-correlated cells fire at
     `lambda_ep <= 0.75`, against 10 of 18 Hartree–Fock.
2. **The handoff's reading rule was applied as written.** The result is
   "mixed".
3. **The cutoff finding is material:**
   - at `lambda_ep = 1.5`, the cutoff-16 exact correlated preparation
     fires at `141.41`;
   - the observed finite-cutoff event sequence (cutoffs 20 to 32)
     settles descriptively near `127.868`, but no adjacent pair passed the
     declared `1e-12` coordinate criterion, so no convergence pair is
     accepted;
   - this bears on every record number that used that preparation at
     cutoff 16.
4. **Increment 024 has two parts, judged separately:**
   - its one cutoff-24 cell completed successfully;
   - the increment as a whole failed its `1200 s` gate, because closure
     work took it to `1636.84 s`.
5. **Holds.** Root hold 025 stands. No evidence is adopted (section 9).

**Scope.** Read-only. No propagation, grid cell or comparator was rerun,
no Codex worktree or shared code was edited; the only computations were
SHA-256 recomputation and counting rows of committed JSON. The reviewer
wrote the map's handoff; section 4 holds the execution against it. This
lane runs no chained repaired segments, so the three-lane parity failure
and its holds do not touch the map's results (root's source review says
the same).

---

## 1. Hash verifications performed

| artifact | expected | actual | match |
|---|---|---|---|
| message `codex-to-claude-20261010-013.md` | `b078b9a4…` | `b078b9a4df55ceb73e2c5ed3e8f7b695b89fd774ba7b2856cda045bedbcaea73` | yes |
| map report (disk; committed at `d9a8ad29` and unchanged at `ae98da08`; 358 lines) | `3bdd6453…4d` | `3bdd64532d78a3356e4979fa958da5e1f2d745ecdce0a41ce38509a8fdc0314d` (all three) | yes |
| weak-cutoff report (disk, and at `ae98da08`; 74 lines) | `689854a5…4f` | `689854a55f886861c4cdd0a03c856712ff1fee0e41c8f7bef5b2aa12950e0b4f` | yes |
| dispatch `divergence-weak-cutoff-diagnostic-20261010-024.md` | `0cdbd3bc…d7` | `0cdbd3bcc7e954c21e4261a21e5433329949e8d3b3998ce22e326feccffe07d5` | yes |
| root hold `divergence-budget-hold-20261011-025.md` | none supplied | `0d89e54bf8c6248a1854ed33916ae7b073e847e4a782cdc99565fc1474cd988c` (recorded here) | — |
| handoff snapshot (report line 18) | `4532c31b…7c` | blob at `6b27596e`, tracked file and campaign copy: equal | yes |
| primary lock (report line 20) | `f198c7e0…12` | campaign copy equal | yes |
| frozen runner `divergence_parameter_map.py` (024 dispatch line 19) | `5fd49ac6…a1` | `5fd49ac6e2b89cd4334b2e0093bc33c3b6feba1b46c6b0233eb7f6615770fea1` | yes |
| `grid_exact_correlated.json` (the 024 comparator lock) | `1bd6117a…bb` | equal | yes |
| map `input_manifest.json`: 35 inputs and 5 scientific-source entries, all recomputed | `expected = actual` | 35 of 35 and 5 of 5 | yes |
| map `artifact_manifest.json`, all entries recomputed | as listed | 209 of 209 | yes |
| 024 `input_manifest.json` and `artifact_manifest.json`, all entries | as listed | 12 of 12 and 18 of 18 | yes |
| 024 `closure_receipt.json` (gitignored output, not committed) | — | `1bda7e1f0b6d873bb8d11f3c497cee039b1126b9f8342cbdafcb86c32246d990`; it records `whole_increment_seconds_through_commit_and_checks` `1636.837143` | — |

## 2. The map's claims, against its receipts and code

| claim (report line) | receipt checked | finding |
|---|---|---|
| reproductions (44, 52–58): exact-correlated `141.4057435874271`, crossings `1.0362131873909188` / `3.6820770750527934` / `0.16070471250561877`, sampled maximum through 140 `24.242142679210755`; HF `54.46129581272155` / `27.963394366414704`; locked `t = 60` start `433.8964202711016` (historical tolerances) and `434.18784227515005` (refined); Eq. (112) none through 400, maximum `2.261301078269013` | `reproduction_gates.json`, `reproduce_*.json` | supported. Each meets the handoff's increment-1 acceptance (section 4) |
| resolution (63, 69–76): 62 of 62 complete; `1e-10` spreads `0.00137793582` (exact) and `6.2129542e-07` (HF); tolerance shifts below spreads | `resolution_summary.json` | supported. The decision rule gives one start per cell, and the report applies it |
| cutoff (82, 88–97): events `141.41 → 128.73 → 127.87 → 127.87 → 127.87` at cutoffs 16/20/24/28/32 (`lambda_ep = 1.5`), and `26.05 → 20.95 → 20.84` at 16/24/32 (`lambda_ep = 3`); no adjacent coordinate change below `1e-12` | `cutoff_summary.json` | supported. Raw values: `141.4057435874271`, `128.72767713494915`, `127.86911133032973`, `127.86796450662091`, `127.86796193643005` |
| grid counts (103, 110, 118–142) | both grid files, recounted by this review | 34/41/8 and 42/44/10, exactly as reported |
| drive-0 events from a preparation the exact dynamics leave stationary (105, 125–128) | grid rows | supported. A magnitude event at zero drive belongs to the closure alone |
| restarts (148–233): 13 locked canonical inputs; 4 of 5 fire within 400 elapsed; the `lambda_ep = 1.25` start at `t = 20` has a four-draw spread of `9.64` | `restart_summary.json` (SHA-256 recorded at line 194), stage files | supported |
| Eq. (112) (238, 244–254): 5 of 9 fire by 2000; all 9 cross below `-1e-8` in the boson-moment and retained-Gram eigenvalues; 6 of 9 in the electronic one | `control_summary.json` | supported. Message 013's "all nine lose positivity" should be read as boson-moment and retained-Gram positivity: the electronic count is 6 of 9 |
| suite 528 = 508 + 20 (283) | `final_test_receipt.json` | supported |

**One pattern the receipts show and the report does not state.** This is
a reviewer reading, not adopted. All eight low-coupling events in the
exact-correlated grid occur at drive amplitude `1` or more:
- `lambda_ep = 0.25`: drives 1, 1.5 and 2;
- `lambda_ep = 0.5`: drives 1.5 and 2;
- `lambda_ep = 0.75`: drives 1, 1.5 and 2.

All eight fire before `t = 436.11`, far from the `2000` censoring horizon.
The ten low-coupling cells that do not fire show no event through `2000`.
Seven of those ten nonetheless cross below `-1e-8` in the electronic
eigenvalue.

So the "mixed" count is not on a numerical knife edge in time. It is one
cell short of the "generic" threshold by count. At weak coupling it
sorts by drive amplitude rather than scattering.

## 3. The cutoff finding and what it touches

At `lambda_ep = 1.5`, `gamma = 0.5`, the exact correlated preparation at
cutoff 16 has tail probability `7.64e-6`. It differs from cutoff 32 by
`8.864e-4` in the largest coordinate (report line 274). Its unrepaired
event, `141.41`, sits `13.54` units above the observed cutoff-32 event,
`127.86796`.

The raw events are `141.4057`, `128.7277`, `127.86911`, `127.867965` and
`127.867962` at cutoffs 16, 20, 24, 28 and 32. Adjacent event changes
fall from `1.15e-3` (24 to 28) to `2.6e-6` (28 to 32). This settling is a
descriptive observation of finite-cutoff event agreement. It is not an
accepted convergence pair: no adjacent pair passed the handoff's declared
`1e-12` coordinate criterion (the change from 28 to 32 is `1.08e-10`).
It is not an infinite-cutoff event accuracy either.

**What this touches, as a consequence to be dispositioned rather than
adopted.** Every record quantity that used the exact correlated
preparation or the exact driven reference at cutoff 16 inherits that
preparation's cutoff error:
- the `141.41` event, which the closure-instability report also quotes
  and converges in integration tolerance (not in cutoff);
- the cheap repair's task-11 accuracy rows, scored against a cutoff-16
  exact reference;
- the boundary sweep's exact states.

Within the record, preparation and reference share the cutoff, so
internal comparisons are like for like. Statements about the physical
model at `lambda_ep >= 1.5` are cutoff-sensitive at 16: the observed
event shifts by `13.54` units between cutoffs 16 and 32.

At weak coupling the observed initial-coordinate changes are small:
- `lambda_ep <= 0.75`: coordinate changes from 16 to 24 are at most
  `7.4e-9` (line 270);
- `lambda_ep = 0.25`: increment 024 measured `1.39e-14`.

These are coordinate proximities, not propagated error bounds on any
event or count, and the review does not conclude that the low-coupling
count is free of cutoff artifacts. The only propagated weak-coupling
check is increment 024's single cutoff-16/24 pair for one cell
(section 7). That establishes neither infinite-cutoff event accuracy nor
stability of the weak-coupling grid count.

## 4. Were the handoff's first-increment gates executed as written

The contract is my snapshot `4532c31b…`
(`agent_guidance/paper-v/handoffs/codex-paper-v-divergence-parameter-map-20261010.md`
at `6b27596e`).

| handoff gate | executed | note |
|---|---|---|
| increment 0 (lines 177–210): lock, byte-identical copies, readable inputs | yes, with the provisioned worktree | 29 primary + 2 supplemental locks; copies identical |
| acceptance (lines 211–217): residual norms; seven runs, with `closed_archive_exact_correlated` at final time `140.0`, maximum `24.24214267917956`, electronic crossing `1.0362131873909188`; fourteen exact-state events, with one `None` at `24.44` | yes, all met (report line 13) | — |
| stop (lines 218–220) | not met | — |
| increment 1 acceptance (lines 294–305): within `1e-6` of `1.0362132`, `3.6820771` and `0.1607047`; within `1e-4` of `24.2421`; within `1e-12` of the residual; event within `1e-4` of `141.4057436`; start-60 event within `1e-3` of `433.8964`; Eq. (112) maximum within `1e-3` of `2.2613` | yes, all met | the event agrees to `1.3e-8` |
| increment 2 decision rule (lines 340–349), increment 3 rules (lines 375–383), increment 4 reading rule (lines 418–431) | yes, applied literally | "mixed"; no cutoff stated as converged |
| C10 20-minute bound (line 747) | yes, per process and per declared sub-increment | — |
| C20 parity gate | not applicable: this lane runs no chained repaired segments (C20 scope; handoff section 0) | the lane says so (report line 14) |

**Handoff defects the review owns.**
1. **Sampling step.** My increment 1 used sample step `0.0025` for the
   reproductions. The record's `unrepaired_t140.json` was sampled at
   `0.005`, so the record's sampled maxima reproduce only at `0.005`. The
   lane found this, kept both protocols distinct and recorded it as a
   deviation (report line 38). Event times are unaffected; the defect was
   mine.
2. **Cutoff criterion.** My increment-3 convergence criterion (adjacent
   change below `1e-12`) is stricter than any adjacent change the cutoff
   ladder produces. The rule correctly reported "none". A useful
   criterion would also look at the event sequence, as section 3 does.
3. **Reading rule.** My increment-4 rule's bands are count thresholds
   with no allowance for cells one short. That is acceptable as declared,
   but the "mixed" label carries less structure than the receipts do
   (section 2).

## 5. Qualifications against the handoff's evidence standards

The standards are handoff lines 512–524. The report meets them
throughout and exceeds them in places.

Where it meets them:
- every no-event statement carries its window;
- events are given to two decimals with full digits kept as
  reproducibility metadata (112);
- resolution is measured at the baseline only and not extended to every
  cell (112);
- the restart starts are described as "descriptive, not a matched-state
  comparison" against the historical HF starts (157, 216);
- perturbations carry no physical-admissibility certificate (64).

Where it exceeds them:
- the restart controls add four-draw spreads and a tolerance pair per
  start;
- they expose the `9.64`-unit spread at `lambda_ep = 1.25`, a cell the
  baseline resolution would have misjudged.

No Jacobian, exponent or convergence study duplicates the
closure-instability work (46, 65).

## 6. Defects and unsupported statements in the map report

None of substance.

- **Line 3, "including 8/18 at lambda_ep<=0.75".** Exact as stated. It
  would be more informative with the drive structure of section 2.
- **Line 82, "a cutoff effect, not cutoff independence".** Supported.
  The report's own table (90–94) also shows the descriptive
  finite-cutoff settling of section 3, without an accepted convergence
  pair. The report does not overstate here.

## 7. Increment 024: the successful cell, kept separate from the failed whole-increment gate

### The cell observation

Accepted as an observation, not as an increment.
- **The cell itself:** one new exact-correlated cell at
  `lambda_ep = 0.25`, drive 1, `gamma = 0.5`, cutoff 24, under the
  frozen runner and the map protocol (DOP853 `1e-10` / `1e-12`, sampling
  `0.0025`, threshold `1e4`).
  - it completed in `13.566 s`, return code 0, with no deadline signal;
  - it reached the threshold at `t = 436.10264591071893`;
  - the stored cutoff-16 comparator, configured through `500`, terminated
    at `436.1026459107307`;
  - signed difference `-1.18e-11` (weak-cutoff report, lines 3 and 17).
- **Agreement with the comparator.** The initial coordinates differ by
  at most `1.393e-14`. All nine positivity crossing times agree to the
  printed digits (lines 41–51).
- **Its receipts verify:** input manifest 12 of 12, artifact manifest 18
  of 18, comparator lock and runner hash.

The observation is one finite cutoff pair (16 and 24) for one cell,
with observed event agreement of `1.18e-11`. It establishes neither that
cell's infinite-cutoff event accuracy nor the stability of the
weak-coupling grid count, and it excludes no cutoff artifact. It changes
none of the map's counts.

### The whole-increment gate

Failed. The 024 dispatch (line 31) bounds the whole selected increment,
including reads, scoring, report, graph and scope checks and commit,
below `1200 s`:
- the first explicit over-bound check measured `1529.108808 s` (report
  line 5; `validation.json`);
- the closing receipt measured `1586.161324 s`;
- the closure through commit and checks measured `1636.837143 s`
  (`closure_receipt.json`).

The overrun came from closure work, not from the scientific process.
Root hold 025 (lines 7–11) records the failure, preserves the cell
observation separately and stops further execution. The report's
disposition is "Status: blocked" with an explicit operational stop
(lines 5 and 10). That is the correction 025 required.

### What the failed gate establishes

The increment's operational discipline failed. It does not establish
that the cell's measurement is wrong: the process receipts and output
hashes stand. It does not reflect on the original map, which stays
accepted within its exploratory scope.

### One provenance gap

The final `1636.84 s` lives only in `closure_receipt.json` under the
gitignored output directory and in root's campaign README. The
committed `validation.json` carries the earlier `1586.16 s`. Since a
commit cannot contain its own timing, the gap is unavoidable, but the
durable record understates the final figure. A one-line note in a later
coordinator record would close it. This review asks nothing of the lane,
whose 024 scope is closed.

## 8. What the map does and does not establish

**It establishes**, at cutoff 16, `gamma = 0.5`, the two `t = 0`
preparations, and windows of 500 and 2000:
- the unrepaired archive closure reaches the `1e4` magnitude event in
  most cells;
- it does so at zero drive from a stationary exact state for
  `lambda_ep >= 1.5`;
- it does so from four of five exact driven states of the correlated
  preparation within 400 elapsed;
- under Eq. (112) subtraction it does so in five of nine cells, and in
  all nine it crosses below `-1e-8` in the boson-moment and
  retained-Gram eigenvalues;
- the initial-contraction cutoff moves the strong-coupling events by
  several to thirteen time units.

**It does not establish:**
- divergence as a finite-time singularity;
- boundedness for any cell without an event, beyond its window;
- a cutoff-converged event or count at any coupling, including the
  low-coupling count;
- physical admissibility of the perturbations;
- any statement about repaired dynamics.

## 9. Holds and evidence status

Every hold remains binding:
- root hold 025, which stops all further execution in 024; no
  comparator rerun or new increment is selected;
- the holds that do not touch this lane's results, as root's source
  review notes: messages 014, 015, 016, 017 and 019, cone increment 3,
  and the three failed strict parity gates of the repaired lanes
  (regularized, repair-early, cone), each failed by
  `1.9395497882168655e-4`.

The map at `d9a8ad29` remains exploratory. Its counts, its cutoff
sensitivity and its censoring are not evidence. The 024 cell is an
observation preserved separately from its failed increment. This review
adopts no evidence and releases nothing. All outputs remain
`exploratory_local_not_promoted`.

## Disposition of classical-review-scope-20261018-003

Message `classical-review-scope-20261018-003`
(`/Users/lordskyler/Documents/ChatGPT/Holstein/campaign-20261010/classical-review-scope-followup-20261018-003.md`,
SHA-256 `2ebb3296cc0fcb2f632eee07a02724a6fb6f4e9bafabfb685543e671ff9291b4`,
11 lines, verified before reading). Input version: `be7b1b2b…`, 278 lines.
Text-only. No source or test edit, rerun, new gate or tolerance change.
The map, the 024 audits and the 024 closure stay closed.

| correction | disposition | where |
|---|---|---|
| verdict item 3: "cutoff-converged sequence" | accepted: now the observed finite-cutoff event sequence, settling descriptively, with no adjacent pair passing the declared `1e-12` criterion and no accepted convergence pair | verdict, item 3 |
| section 3: convergence reading of the event sequence | accepted: raw sequence and adjacent changes kept as a descriptive observation; no accepted convergence pair; no infinite-cutoff accuracy | section 3, second paragraph |
| section 3: "the map's low-coupling count is therefore not a cutoff artifact" from small coordinate changes | accepted: replaced. Coordinate proximity is not a propagated event or count error bound; no artifact exclusion is claimed | section 3, last paragraphs |
| section 7: the cutoff-16/24 agreement "confirms … not a cutoff artifact" | accepted: one finite pair for one cell establishes neither infinite-cutoff event accuracy nor grid-count stability; the successful cell observation is preserved | section 7, "The cell observation", closing paragraph |
| consistency: section 3 bullet on the closure-instability event, section 6 note on line 82, section 8 non-establishment list | accepted, to remove remaining convergence wording | sections 3, 6 and 8 |
| preserved unchanged | already present | raw sequence; successful cell observation; failed whole-increment gate; root 025; holds 014–017 and 019; conditional cone increment 3; 1200-second whole-increment bounds |
