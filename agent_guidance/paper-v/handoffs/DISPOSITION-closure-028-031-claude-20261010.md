# Disposition of root reviews 028, 029, 030 (classical part) and 031: closure instability and the classical consolidation

**Revision line (2026-10-18, closure-disposition-scope-20261018-005).**
Text-only corrections: rows 028-1 and section 2's capped-map line, and the
amendment-table rows C02, C05, C11 and C32. See the closing block.

**Status.** These are author dispositions, written 2026-10-18 by the
existing closure-instability author, with every accepted amendment applied
in place.

- No run, test, new agent, source edit, default, tolerance or execution.
- No evidence adoption and no manuscript edit.
- Nothing here releases or selects anything.
- The three current strict resume-parity failures, root 025, the cone
  increment-3 hold and root messages 014 to 017 and 019 stay binding.
- None of the guard, hybrid or chain studies proposed in report 13.12 and
  14.7 or in the consolidation's section 8 is selected.

## Inputs, each SHA-256 verified before reading

| input | SHA-256 |
|---|---|
| `codex-to-claude-20261010-013.md` (paragraphs 35 to 53) | `b078b9a4df55ceb73e2c5ed3e8f7b695b89fd774ba7b2856cda045bedbcaea73` |
| `closure-instability-corrections-review-20261011-028.md` | `d511e046432c9c475593a90496a4a59696943e2be047e09ec2254b0564040e35` |
| `closure-hybrid-review-20261011-029.md` | `daf4f68336b067580311c13bc4edd29a16330d2085a8b37e9629657da6e1104f` |
| `consolidated-results-review-20261011-030.md` (classical part only) | `14c14fd4da0355840c944263d15cac0a6fb90fa6f683dad9fc533338f9442f3b` |
| `classical-replacement-review-20261011-031.md` | `f0e4f659478afbb60bd8baafa586e8b50de393d11a7be5bbcd0a5ed41bfb6878` |

All four reviews are under
`/Users/lordskyler/Documents/ChatGPT/Holstein/campaign-20261010/`. The
reviewed versions were confirmed before any edit:

- the report at `36b686b3` hashes to `a4eb9088...` (review 028);
- the report before this edit hashes to `13e9f8ff...` (review 029,
  section 14 at `8390a83b`);
- the consolidation at `2c096bad` hashes to `332dbbc3...` (review 031).

Review 030's classical findings were checked against `2c096bad` before being
accepted, as message 013 asks. The root receipt indexes under
`claude-receipts/` were not edited.

The code claims in the reviews were checked against the receipts, without
running anything:

- `minors_family.py` lines 219 to 224 increment `nonconv` once per
  whole-batch failure, and line 236 records every 400 steps;
- `maps2.run` records every `record_every_time` (one unit) and keeps no
  bound residual;
- `run_spec.py` lines 62 to 68 restore without validation, and line 112
  hashes after execution;
- all 234 hybrid re-selections are infeasible;
- every post-event Gram minimum is negative (largest `-7.75e-6`, smallest
  `-0.196`), with 99 of 234 above the pre-event value;
- every extension minimum is negative (best `-1.85e-4`);
- the exact-correlated `Delta = 20` arm reached the 200-iteration cap;
- all 13 arms have `resumes = 0`;
- the 117 escape statuses are 76 success and 41 terminal event.

## Outputs

| file | before | after |
|---|---|---|
| `agent_guidance/paper-v/overnight/closure-instability-20261010.md` (sections 13 and 14 amended, revision note at the top) | `13e9f8ffe87cd64b12c363491dc720b4d0a6e700af817ea57bda7dc68726b05a` (2152 lines) | `30187293aeb89b3e355599560930ba26c5a79cd4bd263413d61cdbcb0fe0c710` (2416 lines) |
| `MATH/paper_facing/paper_V_high_u_gkba/classical_repair_and_guard_results_20261011.md` (revision 2, note at the top) | `332dbbc32c634fabda8bf791ca7f61b0785b386421aa0dd5fda3783256a632a2` (1080 lines) | `bc8d37530c98856dd035bfd14dbe8cad2b5797cd9b2b89ede7548601c750882f` (1344 lines) |
| `paper_5/notes/electron_phonon_closure_worklog_20260803.md` | --- | a dated 2026-10-18 block appended (W01), earlier blocks untouched |

The committed versions stay in history: `36b686b3`, `8390a83b`, `3a9981f7`
and `2c096bad`.

**Counts.** 28 findings: 25 accepted, 3 accepted with amendment, 0
contested. They are applied as 70 report amendments (R01 to R70), 52
consolidation amendments (C01 to C52) and one worklog block (W01).

## 1. Findings and decisions

"Accept with amendment" means the finding is accepted, and the replacement
adds a measured qualification that the finding did not state. The
amendment IDs point to the table in section 3, which gives the line and the
replacement text.

| finding (review and line, quoted short) | decision | changes (file: amendment IDs) | replacement, in substance |
|---|---|---|---|
| **028-1** (028 line 13): "Separate the conditional compact-set argument from failed map enforcement"; batch-step not member-step counts; per-row acceptance; drop-trace batch lacks residuals; exact `x17 = -omega*n/(2g)`; trace control not uniquely necessary | accept with amendment | report: R01, R07, R08, R09, R21, R22, R23, R24; consolidation: C13, C15, C16, C19, C21, C22; worklog W01 | The compact set is a conditional statement about the set, true for a trajectory only while every condition holds. Counts are batch steps, with the member count not retained. Per-row status is given for every drop-one and keep-only cell. The exact witness is used, and trace control is needed but not shown to be the only control. **Amendment:** the receipts show that in the fired keep-only cells the unconverged residual is in the energy (at most `2.2e-11`), while the kept bounds hold to `1e-12`. Two fired cells (Hartree--Fock `trace`+`n_diag`; exact correlated `trace`) converged to the recorded tolerance (`1e-12`) at every step, and for their 8 members and 600-unit window the event still occurred under the capped algorithm. Finite-tolerance enforcement is not exact set membership, so no row establishes insufficiency of the exactly enforced set, and the conditional compactness algebra stands. (Wording corrected under closure-disposition-scope-20261018-005.) |
| **028-2** (028 line 15): "Qualify finite-grid necessity, confinement and heating claims"; extrema recorded every 400 steps are sampled; worklog wording | accept | report: R02, R10, R20, R23, R25, R26, R29, R34, R36, R37, R38, R70; consolidation: C17, C18, C19, C20; worklog W01 | Finite ablations with their 8 members, 600-unit window and `1e4` threshold. "Supremum" becomes the largest sampled value (records every 400 steps). A sampled energy maximum cannot bound every point. The heating has not levelled off by 3000, which is not heating without bound. The worklog's "heats without bound", "any five ... confine" and "necessary" are corrected in W01. |
| **028-3** (028 line 17): "Name the tested cones and limit the predictor/starting-state conclusions" | accept | report: R04, R11, R12, R13, R14, R15, R39, R40, R41; consolidation: C08, C09, C10, C11, C12, C29, C30, C40, C41 | The selected minors and the retained 8-by-8 Gram are necessary conditions, not complete physical representability. 13.10 becomes "starts satisfying the tested necessary constraints". The 16-against-64-member and provenance mismatch is stated, with the direct 128-event result. No universal predictor or independence claim is made; the weak correlations, censoring, depth threshold and in-sample tuning are retained. |
| **028-4** (028 line 19): "Bind accuracy and order to a metric, preparation, horizon and tested refinement" | accept | report: R05, R06, R17, R18, R33, R35; consolidation: C23, C24, C25, C27 | "Does not improve accuracy" is attributed to the critic's comparison with the cheap repair. Against the bare closure the gains and losses are given metric by metric. The step-halving ratios are an observation of the tested always-on maps, not an order theorem. The triggered guard's agreement with the bare closure before its first trigger is not a benefit, and its Hartree--Fock trigger at `t = 37.26` hits the cap unconverged. |
| **028-5** (028 line 21): "Keep computed-energy preservation and representability repair distinct"; energy-neutral projection can change later pulse work; a one-pass clip need not stay in the cone | accept | report: R27, R28, R30, R31, R33; consolidation: C14, C26, C28 | The target is the computed post-RK4, pre-map energy, not the exact reference and not an accumulated drive-work reference. "Does not touch that difference" is withdrawn: rounded agreement at `t = 1` does not establish equal work histories, and the attribution is not proved. The one-pass Gram and boson maps are named as such, with their negative eigenvalues; no name implies an enforced intersection. |
| **028-6** (028 line 23): "Preserve the repaired-chain prerequisite hold and qualify timing"; 8 of 10 failed re-selections | accept | report: R03, R42, R43, R44; consolidation: C34, C51 | The cheap-repair timing includes 8 of 10 failed re-selections. Minimum-of-three, historical and cross-solver walls are descriptive, not matched-accuracy guarantees. Nothing here repairs the three current strict resume-parity failures. The repair seconds are corrected to the receipt's `19.0` s. |
| **028-7** (028 line 25): "Do not substitute process watchdogs for whole-increment acceptance" | accept | report: R44, R45; consolidation: C48 | The per-process 19.5-minute cap and the batching do not establish the 1200-second whole-increment gate, which is recorded as unestablished. |
| **028-8** (028 line 27): "Separate final hash integrity from immutable execution provenance" | accept | report: R44, R46; consolidation: C51 | `run_spec.py` restores checkpoints without validating source, input or settings, and hashes the state artifact after execution. The manifest holds no `paper_5/src` file. Hashes prove artifact integrity, not fixed provenance across resumes. |
| **029-1** (029 line 7): "Correct the claim that the Gram is held at repair times" | accept | report: R53, R63; consolidation: C35 | Every post-repair, post-guard Gram minimum is negative (`-7.75e-6` to `-0.196`). Only 99 of 234 events leave it above the pre-event value: some repairs reduce the violation. |
| **029-2** (029 line 9): "Use metric-dependent, finite comparisons"; exact-correlated `Delta = 10` improves some 60-unit metrics; one `h/2` arm | accept | report: R48, R49, R50, R64; consolidation: C31, C36 | Exact correlated is metric-dependent and nonmonotone (`Delta = 5`, 10, 20 better on site-0 and all 31 at `T = 60`; `Delta = 1`, 2 worse). The `R §14.0` item 2 contradiction is resolved from its own table and recorded. One refinement shows its own difference only. |
| **029-3** (029 line 11): "Label the executed fallback and its failed gates"; 222 + 12 infeasible; negative extension minima; exact-correlated `Delta = 20` cap without a residual receipt; zero-cumulant assumption | accept | report: R51, R53, R54, R58, R62; consolidation: C32, C35, C36 | Labelled a capped fallback hybrid, not an accepted repair: 234 infeasible re-selections, extension minima negative (best `-1.85e-4`), and the `Delta = 20` cap without a bound-residual receipt. Agreement of the zero-cumulant dictionary at the initial product state does not establish later correctness. |
| **029-4** (029 line 13): "Qualify the reference and skipped-selection statements" | accept | report: R51, R55, R62, R66; consolidation: C33 | Computed energy is supplied directly, with no separate finite-step drive-work accumulator. The refused branch returns the carried lift, which is source-level support for equal inputs. Solver and RNG effects were not audited and bitwise equality was not run, so the skipped cost stays derived. |
| **029-5** (029 line 15): "Keep censoring, cost and timing scope explicit" | accept | report: R52, R57, R60, R61, R67, R68, R69; consolidation: C34, C36 | 117 nonfailure solver statuses (76 complete, 41 terminal events). Zero of 8 is censored, with an upper bound of `0.31`. Accuracy is one trajectory. Arm walls exclude preparation, checkpoint, escape, scoring and report, so whole-increment compliance is unestablished. The repair timing includes 8 of 10 failures. |
| **030-1** (030 line 23): capped maps and sampled observations; exact `x17` | accept | consolidation: C13, C15, C16, C17, C21, C22 | Checked against `2c096bad`: the sampling labels were already partly added (unit records, twelve sampled times), but the capped-map, supremum and `-0.408 n` points persisted (031-1 to 031-3). Now amended. |
| **030-2** (030 line 25): necessary constraints versus physical representability; in-sample predictor scope | accept | consolidation: C08, C09, C10, C11, C12, C29, C30, C40, C41 | This persisted in `2c096bad` (031-4). Now amended. |
| **030-3** (030 line 27): finite necessity, order and flow conclusions | accept | consolidation: C03 to C07, C19, C20, C24, C37 to C39, C42 | This persisted in part (031-3, 031-5). Now amended. |
| **030-4** (030 line 29): retain metric-dependent improvements; energy-neutral projection may alter drive work | accept | consolidation: C25, C26, C27, C31 | The exact-correlated `Delta = 10` gains were already restored in `2c096bad` (031 records that as resolved, and it is preserved). The remaining `T = 4` contradiction, the critic attribution and the pulse-work qualification are amended now. |
| **030-5** (030 line 31): hybrid failure disposition and cost scope; segmented work against the 1200-second gate | accept | consolidation: C32 to C36, C48 to C50 | Partly present in `2c096bad` (222 infeasible, negative extension minima). The 12 refinement selections, negative post-event Gram, the `Delta = 20` cap, the skipped-selection limits and the whole-increment gate are added now. |
| **030-close** (030 line 33): historical rescoring does not repair the parity failures; 014 to 017, 019, cone increment 3 and 025 binding | accept | report: R44, R67; consolidation: C51 | Stated as binding holds. |
| **031-1** (031 line 26): conditional compactness against the executed capped map; "tight" requires optimality | accept | consolidation: C13, C15, C19, C43, C44, C46, C47 | The exact constraint set and the capped algorithm are labelled separately, with per-row residual scope. Failed enforcement cannot establish insufficiency. "Tight" becomes "sharper", with global optimality not shown. |
| **031-2** (031 line 28): batch-step counts; largest recorded sampled value with its cadence | accept | consolidation: C17, C21, C22 | Restored as batch-step counts; "largest recorded sampled value (records every 400 steps)". |
| **031-3** (031 line 30): exact trace witness; finite necessity | accept | consolidation: C16, C19, C20 | `x17 = -omega n / (2 g)`; trace control needed but not unique; finite 8-member outcomes of the tested algorithms. |
| **031-4** (031 line 32): retained-Gram positivity is not physical representability; 16 against 64; equal medians | accept | consolidation: C08 to C12, C29, C30, C40, C41 | Named "starts satisfying the tested necessary constraints". The unmatched ensembles are kept. Equal medians are reported as a finite, censored, in-sample association, not "discriminates nothing". |
| **031-5** (031 line 34): scope mechanism and convergence statements; cite an exact divergence-free identity separately | accept with amendment | consolidation: C03 to C07, C37, C38, C39, C42, C45 | Crossings are robust to the tested refinements, not certified. One backward-Euler matrix is not a nonstiffness certificate. Finite-time stretching and finite growth are not a global attractor or unboundedness. The singularity disclaimer is kept. **Amendment:** the exact identity exists and is cited (`QF`, `475c47a7`): `tr J` vanishes identically from the polynomial coefficients. So by Liouville's theorem the smooth flow preserves volume and cannot contract to a lower-dimensional attractor, and that statement is kept, separate from the floating-point traces and the Lyapunov sum. |
| **031-6** (031 line 36): order and accuracy metric-specific; the `T = 4` contradiction; the critic quotation | accept | consolidation: C24, C25, C27, C31 | `T = 4` is metric-dependent: `Delta = 1` improves occupation `.0318 < .0356` and energy `.0277 < .0437` and worsens all-31 `.1364 > .1313`; `Delta = 2` slightly improves energy. No tested arm improves all three. The quotation is attributed, with its limited comparison. |
| **031-7** (031 line 38): pulse-work and skipped-reselection claims | accept | consolidation: C14, C26, C32, C33 | Computed pre-map energy is not an accumulated drive-work reference. Rounded `t = 1` energies do not establish equal work. The count is 222 main-arm plus 12 refinement selections. Skipped equality is source-level only, and its cost is derived. |
| **031-8** (031 line 40): retain the hybrid failure and provenance limits; pin the numbers-check audit artifact and `DI` | accept with amendment | consolidation: C01, C02, C35, C51 | The negative post-event Gram, the `Delta = 20` cap and the provenance limits are added. `DI` is pinned (`37e4067d`, SHA-256 `87508b2c...`). **Amendment:** the numbers-check artifact (`classical_summary_verify.json` in the coordinating session's scratchpad) was not preserved and cannot be pinned. The first revision note now says its counts are author-reported. |
| **031-9** (031 line 42): process segmentation is not campaign acceptance; the exact-correlated estimates extrapolate Hartree--Fock profiles | accept | consolidation: C48, C49, C50 | Any selected study must fit the 1200-second complete-increment bound, with bounded resumable subincrements and the fail-on-error and checkpoint gates. The exact-correlated estimates are labelled as extrapolations. Nothing is selected. |

## 2. The points message 013 names, answered

| point (013) | answer | where |
|---|---|---|
| Non-convergence counts are batch-step counts, not failing-member counts | Accepted. Every count is now "batch steps", with a note that the member count was not retained. | R21, R22, C21, C22 |
| The driver records extrema every 400 steps, so "sampled maxima" wherever the text says supremum | Accepted. Every supremum in sections 13 and 14 and in the consolidation is now a sampled maximum with its cadence. Full-step quantities (bound violation, energy residual) are distinguished. | R10, R20, R21, R29, R34, R36, R38, C17 |
| A capped map failing kept constraints cannot establish insufficiency of the exactly enforced intersection | Accepted, with the receipt nuance that most fired keep-only rows fail only on the energy residual, and two converged to the recorded `1e-12` tolerance. Finite-tolerance enforcement is not exact set membership. | R01, R22, R24, C15, C19 |
| The selected minors and the retained 8-by-8 Gram are necessary conditions, not complete physical representability | Accepted. | R04, R11, R15, R39 to R41, C08 to C12, C29, C30, C40 |
| Finite and protocol qualifications: ablations, in-sample thresholds, step-halving ratios, 16 against 64, 3000-unit heating | Accepted. | R02, R14, R17, R33, R37, R41, C18 to C20, C24, C30 |
| Keep the accuracy metrics apart; the cheap-repair comparison and the pulse-work discrepancy are separate | Accepted. | R05, R18, R28, C25 to C27 |
| Energy-neutral projection can affect later pulse work; computed-energy targeting has no separate drive-work accumulator | Accepted. | R27, R28, R55, C14, C26 |
| Use the exact trace counterexample `x17 = -omega n / (2 g)` | Accepted. | R08, C16, C47 (item 35) |
| The cheap-repair timing includes 8 of 10 re-selection failures | Accepted. | R03, R43, R51, R61, C34 |
| `run_spec.py` does not validate at resume and hashes after execution | Accepted, stated as a provenance limit. | R44, R46, C51 |
| Process watchdogs do not establish the whole-increment 1200-second gate | Accepted. | R44, R45, R60, R67, R69, C36, C48 |
| The retained Gram is not held even at repair times, and repairs sometimes reduce its violation | Accepted (99 of 234). | R53, R63, C35 |
| Exact-correlated `Delta = 10` improves some 60-unit metrics; the effect is metric-dependent and nonmonotone | Accepted. The `R §14.0` item 2 contradiction is resolved from its table. | R48, C31 |
| 222 + 12 infeasible re-selections; negative extension minima; capped fallback; `Delta = 20` cap without a bound-residual receipt | Accepted. | R51, R53, R58, R62, C32, C35 |
| Skipped selection: source-level support; solver and RNG effects and bitwise equality not verified; cost derived | Accepted. | R51, R62, R66, C33 |
| All 13 arms reached 60 with no resume; all 117 escape integrations have nonfailure statuses | Accepted, as what the receipts prove. | R52, R68, C36 |
| Censoring, one-trajectory accuracy, one refinement and the zero-cumulant assumption remain limits | Accepted. | R52, R54, R64, C36 |
| 031: the `T = 4` contradiction; improvement is metric-dependent, and no tested arm improves all three | Accepted. | R50, C31 |
| 031: "sharper", not "tight" | Accepted. | R01, R07, C15, C46, C47 (item 38) |
| 031: preserve the U-free scope, the sampling and timing qualifications, the task-11 exception and the metric-dependent exact-correlated gains | Preserved unchanged. | consolidation lines 17 to 21, sections 1, 2.3 and 5 |
| The three failed parity gates and root 025 stay binding; no section-8 study is selected | Stated in both files. | R44, R67, C48, C51 |

## 3. Every amendment, with its line in the revised file

The line is that of the first line of the replacement in the revised file
(the report after 2026-10-18, 2416 lines; the consolidation revision 2,
1344 lines).

- **The report.** The revision line (005) is at lines 3 to 11, and the revision note at lines 13 to 48.
- **The consolidation.**
  - The revision line (005) is at lines 22 to 32, and revision note 2 at lines 34 to 98.
  - The first revision note follows it.
  - C52 is the in-place rewording of the keep-`trace`, `n_diag`, `a`
    bullet.
- **W01** is the worklog block dated 2026-10-18. It corrects, without
  editing them:
  - the earlier blocks' "representability loss predicts nothing", "no
    single family of bounds is necessary", "necessary", "heats without
    bound", "confines", "any five of the six families confine over 600
    units", "the sparse repair does not help" and "the trajectories are
    identical";
  - the 222-event count.
| amendment | file | line | replacement text (opening; the full text is at that line) |
|---|---|---:|---|
| R01 | report | 1099 | 4. **The compactness statement is conditional: it holds for a set, with a sharper constant than section 11's, and only with trace control.** It bounds every coordinate while a bounded-above Eq. (22) energy, the trace pin and ... |
| R02 | report | 1113 | 5. **A finite ablation of the capped map (13.5).** With the exact-energy hold, dropping any single family of bounds gives no `1e4` event in 600 units from either `t = 0` preparation (8 members each). In this finite test, no ... |
| R03 | report | 1132 | the recorded `t = 0` to 60 wall. That is `2.1e3` or `1.3e3` times the minors map. The cheap repair's timing includes 8 of 10 failed lift re-selections, so the ratios compare the measured protocols, not equally accurate methods. |
| R04 | report | 1136 | 7. **Section 10 (R2, 13.3).** In these 19 ensembles, loss of the retained-Gram condition does not discriminate escape. Under the bare closure the retained Gram is negative at the start, or within 12 units of it, in all 19 ... |
| R05 | report | 1150 | 8. **The maps (critic, 13.4).** The critic's assessment, carried verbatim, is: *the minors clip is a candidate cheap divergence guard, not a representability repair, and it does not improve accuracy.* Its "does not improve ... |
| R06 | report | 1170 | `9.7e-3` of member-steps. Its thresholds were tuned on the same ensembles. Before its first trigger it is the bare closure, so its agreement with bare dynamics is not an accuracy benefit. From Hartree--Fock its single trigger, at ... |
| R07 | report | 1326 | minors and `Tr C[q] = 0` gives a compact set. This is a conditional statement about the set: it applies to a trajectory only while every one of those conditions holds. The lower bound, sharper than section 11's (its global ... |
| R08 | report | 1343 | `Tr C[q]`. Take `N_00 = N_11 = n`, `x17 = -omega n / (2 g)` (about `-0.408 n` at `omega = 0.5`, `g = 0.6124`) and every other coordinate zero. The energy `2 omega n + 4 g x17` is then exactly 0, the Gram's minimum eigenvalue is ... |
| R09 | report | 1356 | 13.5 tests that in a finite ablation of the capped map, with each row's convergence status. |
| R10 | report | 1365 | - Given the largest recorded energy (sampled every 400 steps, so not a bound at every point of the interval), the inequality bounds `max\|x\|` by `14` to `29` at the sampled states of every minors run. |
| R11 | report | 1403 | It is not a 16-feature matrix and not the 36-row extension. Positivity of this Gram and of the selected minors are necessary conditions for a physical moment state. They are not a complete test of physical representability, and ... |
| R12 | report | 1428 | events: rho `-0.04` to `0.25`, `p` `0.047` to `0.95`. The section-10 table and the receipts give `-0.0443` for the repaired `t = 40` ensemble; this line printed `-0.01` before the 2026-10-18 revision. |
| R13 | report | 1449 | about its median) the Gram is `1.2` to `6.2` units earlier (receipt `d_threshold_scan.json`, low end `1.228`; this line printed `1.3` before the 2026-10-18 revision). |
| R14 | report | 1453 | to 577 units to the event. These leads are for the five ensembles of section 10's table (of 19), on which the thresholds were tuned, so they are in-sample. Section 10's fixed-threshold table covers 9 of the 23 stored thresholds. |
| R15 | report | 1468 | `543`). In these ensembles, loss of the retained-Gram condition is a property of the bare flow from every tested state and does not discriminate escape. That is a finite, in-sample observation with censoring (23 of 64 fired by ... |
| R16 | report | 1487 | - **What the minors map targets after every step.** These are held to the recorded residual: at most `3.6e-12` without the energy hold, and with it `1e-12` except at the non-converged batch steps listed in 13.5. The `maps2` and ... |
| R17 | report | 1514 | - **The tested always-on maps behave as first order in `h`.** This is an observation on one unperturbed trajectory per preparation at three step sizes. It is not an order theorem, a continuum limit, or a statement about the ... |
| R18 | report | 1544 | - **The critic's assessment, carried verbatim:** *the minors clip is a candidate cheap divergence guard, not a representability repair, and it does not improve accuracy.* The critic compared the clip with the cheap repair. ... |
| R19 | report | 1550 | ## 13.5 A finite ablation of the bound families (the necessity ablation and run 2) |
| R20 | report | 1568 | Each cell gives the members fired by 600 and the maximum absolute coordinate at 600 over survivors. In brackets is its largest value at the records taken every 400 steps, one per unit: a sampled maximum, not a bound over the ... |
| R21 | report | 1586 | `t = 10` to 600 in every drop-one cell. Three Hartree--Fock cells did not converge at every step: drop `a` at 120 batch steps (largest remaining bound violation `6.1e-3`), drop `n_diag` at 10 (`5.8e-4`) and drop `trace` at 7 ... |
| R22 | report | 1613 | Each keep-only cell's convergence, as non-converged batch steps (the number of failing members was not retained) / largest recorded bound violation / largest energy residual after the map (`family_*.json`): \| families kept \| ... |
| R23 | report | 1638 | - With the exact-energy hold, in this finite test (8 members per cell, 600 units, `1e4` threshold), dropping any single family gives 0 events and a sampled maximum of at most `3.80`. Three Hartree--Fock cells and the exact `t = ... |
| R24 | report | 1643 | - With the capped alternation, keeping only the trace pin, only `N_qq >= 0`, both, or `N_qq >= 0` with the anomalous bounds lets every member cross the threshold. The three together give no event from Hartree--Fock and 8 events ... |
| R25 | report | 1654 | - The correlation bounds alone, with the exact-energy hold, give no event in 600. They do not keep the coordinates small from Hartree--Fock (`169` at 600). |
| R26 | report | 1657 | - With the energy not held (one application of the remaining families per step, residual at most `3.6e-12`), dropping the correlation bounds lets 7 of 8 exact-correlated members cross the threshold and drives the Hartree--Fock ... |
| R27 | report | 1673 | both at most `1e-12`, for at most 200 alternations. `E_target` is the computed energy after the RK4 step, before the map; it is not taken from the exact reference. The map itself does no work. By changing the state, though, it ... |
| R28 | report | 1749 | - The Eq. (22) RMS that remains under the exact hold (`0.035` from exact correlated, `0.044` to `0.050` from Hartree--Fock) is consistent with a difference between the work the pulse does on the closure and on the exact state. At ... |
| R29 | report | 1779 | \| map \| preparation \| fired \| maximum absolute coordinate at 600 (in brackets, the largest sampled value; records every 400 steps) \| Eq. (22) change, `t = 10` to 600 \| smallest sampled retained Gram eigenvalue \| smallest ... |
| R30 | report | 1790 | unit records. It heats (`+0.77`, `+1.81`) and has no event in 600 units (8 members per preparation). It is one clip-and-map-back pass per step, which need not land in the cone, and the retained Gram stays negative. |
| R31 | report | 1793 | - `gram_exact` holds the energy and has no event in 600 units. The energy root after the one-pass clip destroys positivity again: the Gram falls to `-1.76` |
| R32 | report | 1803 | and `-1.4e-3` (exact correlated, sampled minima) and has no event in 600 units from either preparation. The retained |
| R33 | report | 1807 | - Order in `h` (`order_check.json`): every tested map, `gram` and `boson_minors_exact` included, behaves as first order on the tested trajectories (ratios `1.91` to `2.39`). No order theorem is shown. - None of these names means ... |
| R34 | report | 1822 | \| start \| members \| fired by 600 \| fraction of member-steps mapped \| maximum absolute coordinate at 600 (in brackets, the largest sampled value; records every 400 steps) \| Eq. (22) change over 600 \| |
| R35 | report | 1840 | minors maps. Its scores move by `4.6e-11` between `h` and `h/2`. Before its first trigger the triggered guard is the bare closure, so its agreement with the bare scores to `T = 20` is not an independent accuracy benefit. Its zero ... |
| R36 | report | 1854 | \| start \| 600 \| 1200 \| 1800 \| 2400 \| 3000 \| largest sampled maximum absolute coordinate (records every 400 steps) \| |
| R37 | report | 1859 | No member fires in 3000 units (8 members per start, `1e4` threshold). The recorded heating has not levelled off by 3000. That does not show heating without bound. In both |
| R38 | report | 1863 | change `+339` exact correlated, `+115` exact `t = 60`, at the records) and reach sampled maximum absolute coordinates of `285` and `124`. By the bound of 13.2 (c), the `1e4` event needs `E >= 4.87e3`. The energy is sampled, so ... |
| R39 | report | 1872 | ## 13.10 Escape from starts satisfying the tested necessary constraints (run 7) |
| R40 | report | 1875 | `x0 + eps u` (seeds 1 to 16, `eps = 1e-4` and `1e-3`), carried into a set satisfying a tested necessary condition (not a certificate of physical representability), then the bare closure |
| R41 | report | 1901 | `588.7` (64 members, unprojected). The comparison does not isolate a projection effect: - the projected ensembles have 16 members, against section 9's 64; - the exact stored states and the perturbed moment vectors have different ... |
| R42 | report | 1943 | - `19.0` s for the 480 Dykstra repairs (one per SSPRK stage; receipt `cheap_repair_timing_run1/summary.json`; this line printed the second run's `18.8` before the 2026-10-18 revision); |
| R43 | report | 1948 | The recorded chain's wall over `t = 0` to 60 is `12,990.087` s, or `216.5` s per unit. That run was in another session, over a different horizon. These timings measure the protocol as run, including 8 of 10 failed lift ... |
| R44 | report | 1998 | (13.5) only the energy residual (at most `1.0e-12`) is recorded, not the remaining bound violation. - **Execution provenance.** `run_spec.py` restores a checkpoint without validating saved source, input or settings hashes (its ... |
| R45 | report | 2040 | waiting loop, and rerun through `maps2` in 100-unit batches. No process of this round ran 20 minutes. That is a per-process statement; it does not establish the 1200-second whole-increment gate (13.12). |
| R46 | report | 2051 | manifest, so no earlier receipt changed. The manifest file itself has SHA-256 `777ae638beb3707da344c47cd2b9ca7936e2b8ee580aa98e41940f8ac0615da0`. These hashes bind the current bytes; they do not prove fixed source, input or ... |
| R47 | report | 2079 | Over `[0, 4]` the same arm is `4.6` (receipt values; this line printed `4.5` before the 2026-10-18 revision) and `26` times worse on the site-0 |
| R48 | report | 2082 | 2. **From the exact correlated preparation the effect of the sparse repair is metric-dependent and not monotone in `Delta`.** Before the 2026-10-18 revision this item said the sparse repair does not improve on the guard, which ... |
| R49 | report | 2091 | - Eq. (22) RMS `0.0610` at `Delta = 1` and `0.0412` at `Delta = 2`, against `0.0354`. |
| R50 | report | 2098 | 3. **Over the guard alone, from Hartree--Fock, the `T = 20` and `T = 60` scores improve steadily as `Delta` shrinks.** `Delta = 1` lowers the `T = 60` scores by 13 percent (site-0), 42 percent (Eq. (22)) and 20 percent (all 31). ... |
| R51 | report | 2114 | - Averaged per arm, the lift re-selection takes `2.4` to `3.0` s of a mean event wall of `2.5` to `3.1` s. It returned "infeasible" in all 222 main-arm events and in the 12 events of the `h/2` arm (234 in all), so the carried ... |
| R52 | report | 2132 | - From exact correlated, 0 of 8 fire at `Delta = 5`, 10 and 20; 3 of 8 at `Delta = 1`; 4 of 8 at `Delta = 2`; 6 of 8 for the guard alone. - These are finite, censored observations (8 members, 600 units). With 0 of 8 fired, the ... |
| R53 | report | 2139 | 6. **The retained Gram is not held, even at the repair times.** - Every recorded post-repair, post-guard Gram minimum in the 234 events is negative (largest `-7.75e-6`, smallest `-0.196`). - Only 99 of the 234 events leave it ... |
| R54 | report | 2171 | - This is an assumption of this lane. It equals the exact-moment dictionary of the Hartree--Fock product state to `4.4e-16`. It differs from the exact correlated ground state's dictionary by up to `0.378` (`probe_hybrid_*.json`). ... |
| R55 | report | 2183 | The energy reference is the computed Eq. (22) energy carried into the repair (`hybrid.repair_event`). The closure step changes the energy only by the drive's work plus integration error, and the guard holds it. So this ... |
| R56 | report | 2199 | - across all 234 events (222 main-arm, 12 in the `h/2` arm), the energy after the repair and the guard differs from the reference by at most `3.6e-15`. |
| R57 | report | 2207 | wall in the serial session, divided by 60. It counts guard segments and repair events only, not the route build, checkpoints, the escape test, scoring or the report. |
| R58 | report | 2273 | alternation reached the 200-iteration cap (`0.59` s per unit of guard alone in that arm). `maps2.run` records no bound residual, so there is no receipt of the bounds at those steps. |
| R59 | report | 2277 | - **The pulse work.** Repairs during the pulse (`Delta = 1`, 2) change the state, and so can change the work the drive does on the closure. The Eq. (22) RMS rises above the guard's (`0.0610` and `0.0412` against `0.0354`). The ... |
| R60 | report | 2285 | 13:58 to 14:16; one-minute load average `2.2` to `6.0`, mostly `3.0` to `4.3`, on 14 cores). It does not establish the 1200-second whole-increment gate: the escape test, scoring, the report and its review ran afterwards, and no ... |
| R61 | report | 2296 | `lift_reselection_failures`), `17.9` s of Dykstra repairs and `2.4` s of velocity evaluations. That is the protocol's cost including those failures, not the cost of an accepted repaired trajectory. |
| R62 | report | 2315 | Every re-selection returned "infeasible": 222 of 222 main-arm events across both preparations, and 12 of 12 in the `h/2` arm. The solved lift failed the measured-candidate check, and the refused branch returns a copy of the ... |
| R63 | report | 2346 | Dykstra at 120 iterations does not reach the extension cone: every recorded extension minimum after a repair is negative (best `-1.85e-4`). The retained Gram is not held at the repair times either. Every post-repair, post-guard ... |
| R64 | report | 2364 | This is the observed difference for this one arm. It does not show that the whole table is converged, or certify a continuum limit. The other arms were run at `h` only. |
| R65 | report | 2374 | - **`Delta` below 1.** From Hartree--Fock the `T = 20` and `T = 60` scores improve monotonically down to `Delta = 1`; the `T = 4` scores are metric-dependent. |
| R66 | report | 2379 | - **The hybrid with the re-selection skipped, run directly.** Its cost is derived above. Equal repair inputs have source-level support (the refused branch returns the carried lift). Solver and RNG side effects were not audited, ... |
| R67 | report | 2394 | - **`h/2` for the other arms**, and for the exact correlated preparation. - **The whole-increment gate.** Per-process timers and the serial session do not establish the 1200-second whole-increment bound, which is unestablished. - ... |
| R68 | report | 2413 | cleanly at a segment boundary after 18 minutes. All 13 arms (12 main, one at `h/2`) reached `t = 60` with no resume (`resumes = 0` in every `arm_*.json`). Had a resume occurred, `run_arm.py` would have restored the checkpoint ... |
| R69 | report | 2422 | - `watchdog.sh`: stops any process at 19.5 minutes. It stopped none. A per-process limit does not establish the whole-increment gate (14.7). |
| R70 | report | 1973 | (13.9). Whether the capped hold keeps the sampled coordinates bounded beyond 600 is not measured. |
| C01 | consolidation | 105 | value is used and why. I disagree with none of the fixes. The line numbers in the list below refer to that revision (commit `2c096bad`), not to revision 2. The numbers-check artifact itself (the coordinating session's ... |
| C02 | consolidation | 213 | \| `R §13`, `§14` (amended) \| the same report after the 2026-10-18 amendments of sections 13 and 14 (dispositions of reviews 028 to 031), SHA-256 `30187293aeb89b3e355599560930ba26c5a79cd4bd263413d61cdbcb0fe0c710` \| uncommitted ... (the binding hash on consolidation line 213, formerly line 202; this row first quoted the intermediate `cd5c6b5b...`) |
| C03 | consolidation | 303 | 2. **The threshold crossings are robust to the tested refinements.** Every event time converges under halving and quartering the step, under tightening the tolerance, and across one explicit and one implicit method, at all three ... |
| C04 | consolidation | 311 | 3. **Volume and energy.** - **The exact identity.** The archive field is exactly divergence-free. The trace of its Jacobian vanishes identically, from its polynomial coefficients (`tr A_1 = tr D = 0`, `sum_i A_2[i, i, :] = 0`; ... |
| C05 | consolidation | 335 | - By the exact identity the flow map preserves volume wherever it exists; no compact set can uniformly attract a bounded open neighbourhood of strictly larger volume; noncompact invariant sets are not excluded (`x' = x`, `y' = -y`). (Corrected under closure-disposition-scope-20261018-005.) |
| C06 | consolidation | 350 | 4. **Positive finite-time stretching from the exact `t = 60` state (transient chaos on these diagnostics).** Positive finite-time exponents and finite coordinate growth are observations. They do not prove a global chaotic ... |
| C07 | consolidation | 377 | - So the energy level set is not compact. The energy, the only conserved quantity used here, does not confine those entries (`R §6`). |
| C08 | consolidation | 479 | ## 3. The retained-Gram necessary condition as a predictor |
| C09 | consolidation | 481 | The tested condition is that the retained 8-by-8 joint Gram of `(1, b0, b1, b0^dag, b1^dag, sigma_x, sigma_y, sigma_z)`, with the shared trace removed, is positive semidefinite (`run_unrepaired.py`). It is measured by its minimum ... |
| C10 | consolidation | 493 | 1. **The tested condition fails within units, from every tested state.** |
| C11 | consolidation | 513 | 3. **In this ensemble the fired and censored groups cross at the same median time.** ... It does not establish an absence of predictive information. (Corrected under closure-disposition-scope-20261018-005.) |
| C12 | consolidation | 519 | 4. **Within one ensemble it orders the events weakly or not at all** (finite, in-sample). |
| C13 | consolidation | 557 | The exact constraint set and the capped algorithm that targets it are separate things, and are kept separate below. **What the clip targets** after every RK4 step (`R §13.4`). These are held only to the recorded residual: - ... |
| C14 | consolidation | 598 | root of `E(x + s grad E) = E_target`. `E_target` is the computed energy after the RK4 step, before the map. It is not taken from the exact reference, and it is not an independently accumulated drive-work reference. The map itself ... |
| C15 | consolidation | 609 | `\|r\| <= 1`, the twelve 2-by-2 Gram minors and `Tr C[q] = 0`. That set is then compact. A trajectory stays in it only while every condition actually holds, which the capped map does not guarantee (item 2), and sampled extrema ... |
| C16 | consolidation | 626 | - **This certificate needs trace control.** The retained Gram subtracts `Tr C[q]`. Take `N_00 = N_11 = n`, `x17 = -omega n / (2 g)` (about `-0.408 n` here) and all else zero. The energy `2 omega n + 4 g x17` is then exactly 0, ... |
| C17 | consolidation | 637 | \| start \| median energy change at 3000 \| largest energy change at 3000 \| largest recorded sampled value of the largest coordinate over 3000 units (records every 400 steps) \| median largest coordinate at 3000 \| |
| C18 | consolidation | 642 | Some members run away between 1200 and 1800 units (`R §13.9`). The recorded heating has not levelled off by 3000, but that does not show heating without bound. |
| C19 | consolidation | 648 | 2. **A finite ablation of the families, with the capped map.** 8 members per cell, perturbation size `1e-4`, 600 units, `1e4` threshold, from both `t = 0` preparations (`R §13.5`). These outcomes are those of the tested capped ... |
| C20 | consolidation | 673 | - **With the energy not held** (one application per step, residual at most `3.6e-12`), dropping the correlation bounds lets 7 of 8 exact-correlated members cross the threshold (`502` to `576` units). Hartree--Fock members reach a ... |
| C21 | consolidation | 683 | - **Non-converged steps.** Three Hartree--Fock drop-one cells with exact energy did not converge at every step. The counts are batch steps: `minors_family.py` counts one non-converged step whenever the whole batch fails to ... |
| C22 | consolidation | 696 | Keep-only cells, as non-converged batch steps / largest energy residual after the map (`family_*.json`; the largest recorded bound violation is at most `1e-12`, except `2.1e-10` in the two Hartree--Fock `corr` cells): \| families ... |
| C23 | consolidation | 720 | - **Its thresholds were tuned on the same ensembles** (`R §13.3`). No held-out test exists. - Before its first trigger it is the bare closure, so its agreement with bare dynamics is not an independent accuracy benefit. - Its zero ... |
| C24 | consolidation | 747 | 5. **Observed first-order behaviour in `h`** (`R §13.4`). With a map after every RK4 step, the ratio of successive step-halving differences is `1.91` to `2.39` at `t = 20`, against `16.0` for the bare closure. This holds for the ... |
| C25 | consolidation | 755 | observables** (`R §13.6`; one unperturbed trajectory per row, scores over `[0, T]`). The last column is the largest change of any of the twelve scores between `h` and `h/2`. The guard can improve a selected metric over the bare ... |
| C26 | consolidation | 806 | - **The remaining energy error.** What is left of the Eq. (22) RMS under the hold (`0.035` exact correlated, `0.044` to `0.050` Hartree--Fock) is consistent with a difference between the work the drive does on the closure and on ... |
| C27 | consolidation | 817 | - **The critic's assessment, verbatim** (`R §13.4`, from the completeness critic of 2026-10-11): *the minors clip is a candidate cheap divergence guard, not a representability repair, and it does not improve accuracy.* Its ... |
| C28 | consolidation | 824 | units, from both `t = 0` preparations; minima over unit records). None has an event in 600 units (0 of 8 per preparation). Each is a one-pass clip and map-back, or such a pass followed by an energy root, so none is a projection ... |
| C29 | consolidation | 836 | 8. **Starts satisfying the tested necessary constraints escape as before** (`R §13.10`). The repaired |
| C30 | consolidation | 847 | The projected ensembles have 16 members against section 9's 64, and the exact stored states and the perturbed moment vectors have different provenance. So the means do not isolate a projection effect. In these tests, satisfying ... |
| C31 | consolidation | 940 | 3. **Over the guard, from Hartree--Fock.** The `T = 20` and `T = 60` scores improve steadily as `Delta` shrinks. `Delta = 1` lowers the `T = 60` scores by 13, 42 and 20 percent, at `14.5` times the guard's wall. - At `T = 4` the ... |
| C32 | consolidation | 950 | 4. **The lift re-selection costs most of the wall and always returned the carried lift.** It returned "infeasible" in all 222 main-arm events and in the 12 events of the `h/2` arm, 234 in all. (Heading corrected under closure-disposition-scope-20261018-005.) |
| C33 | consolidation | 958 | - The refused branch returns a copy of the carried lift. That is source-level support for equal repair inputs if the re-selection were skipped. Solver and RNG side effects were not audited, and no skipped-selection run checked ... |
| C34 | consolidation | 967 | - With the re-selection, the hybrid is 100 to 930 times cheaper than the cheap repair's `332.1` s per unit in the same session. That wall includes 8 of 10 failed lift re-selections. The ratio compares the measured protocols, not ... |
| C35 | consolidation | 976 | 6. **The repair does not reach its cone, and the Gram is not held.** - The Dykstra repair at 120 iterations never reaches the extension cone: the minimum after every event is `-1.85e-4` to `-4.1e-3`. - Every post-repair, ... |
| C36 | consolidation | 987 | 7. **Step dependence.** The `h/2` check on Hartree--Fock `Delta = 5` moves the scores by at most `1.8e-4`. That is one tested refinement of one arm; it does not show that the table is converged. The other arms were not checked. ... |
| C37 | consolidation | 1022 | of three, `t = 0` to 20, one trajectory). On these diagnostics its failure shows no energy drift and no stiffness at the tested step: |
| C38 | consolidation | 1029 | number `1.008` to `1.038` wherever the largest coordinate is below 5 (`R §0` item 6). That is the conditioning of one matrix at one step on sampled bounded states, not a general nonstiffness certificate. |
| C39 | consolidation | 1032 | - **What breaks: correlation blow-up.** The failure is growth of the off-diagonal entries of `delta rho_12^q`, which the energy does not contain. Past the `1e4` threshold they reach `1e10` within `0.25` to `0.33` time units on ... |
| C40 | consolidation | 1056 | - This is loss of a necessary condition for joint representability, not of the electronic density matrix alone. |
| C41 | consolidation | 1061 | - In the tested, partly censored ensembles the joint loss does not discriminate which states blow up. |
| C42 | consolidation | 1067 | - Energy drift is not observed at the sampled times, and stiffness is not indicated at the tested step on sampled bounded states. Neither is excluded in general. |
| C43 | consolidation | 1076 | - **What the clip targets cheaply.** The capped clip targets: |
| C44 | consolidation | 1082 | With the energy bounded above and every bound held, these bounds define a compact set. That is an analytical, conditional bound. The capped map holds the bounds only to its recorded residuals, not at every batch step. Its ... |
| C45 | consolidation | 1122 | moments) under a nonlinear flow, volume-preserving by the exact divergence-free identity (`QF`); |
| C46 | consolidation | 1181 | 14. **The compactness bound with `4 sqrt 2 g`** → valid but loose; the sharper bound of section 4 supersedes it (its global optimality is not shown). |
| C47 | consolidation | 1252 | (`R §14.3` table; section 5, item 2 here). Withdrawn in revision 2 (2026-10-18), after reviews 028 to 031: 35. **`x17 = -0.408 n` gives exactly zero energy** → the exact witness is `x17 = -omega n / (2 g)` (section 4, item 1). ... |
| C48 | consolidation | 1281 | Every compute figure below is an **estimate**, named with the measured wall it comes from. None of these studies is selected. - **The campaign bound.** Any selected study would have to fit the campaign's 1200-second bound for a ... |
| C49 | consolidation | 1308 | - The same measured walls, which are Hartree--Fock profiles. - Estimate, extrapolated from those profiles: about `3.6` to `5.8` hours, as many separately bounded subincrements. |
| C50 | consolidation | 1304 | - Estimate: about `1.2` to `1.9` hours per chain, as many separately bounded subincrements. |
| C51 | consolidation | 1361 | - Estimate: about 1 minute per arm at `Delta >= 5` and about 3.5 minutes at `Delta = 1`. 15. **Immutable execution provenance.** `run_spec.py` restores checkpoints without validating saved source, input or settings hashes, and ... |
| C52 | consolidation | 668 | gives no event from Hartree--Fock (0 fired) but 8 from exact correlated in the 600-unit window. |

## 4. What follows, and what does not

- **This round.** Nothing was run. The amended files keep every number of
  their committed versions except the four receipt corrections, each noted
  in place:

  | quantity | was | now |
  |---|---|---|
  | within-ensemble rank correlation | `-0.01` | `-0.04` |
  | equal-skill lead | `1.3` | `1.2` |
  | Dykstra seconds | `18.8` | `19.0` |
  | the `[0, 4]` occupation ratio | `4.5` | `4.6` |

- **What the amendments add.** They add receipt facts that were already on
  disk: per-row residuals, 234 re-selections, 99 of 234 events, 117 solver
  statuses and the `x17` witness. They also pin `DI` and the exact
  divergence-free identity.
- **What stays binding.** The three current strict resume-parity failures
  (first divergence at `t = 0.1225`, about `1.94e-4` against `1e-12`, as
  root records them), root 025, the cone increment-3 hold and root messages
  014 to 017 and 019.
- **Nothing is selected.** No guard, hybrid, chain, dictionary,
  skipped-selection or sparse-repair study or policy follows. Nor does any
  implementation, rerun, default, tolerance, output, route, reference,
  physical benchmark or execution.
- **What stays with the user.** Evidence adoption and manuscript insertion.
- **Ownership.** Report, consolidation, worklog and module ownership are
  unchanged.

## Disposition of closure-disposition-scope-20261018-005

**Input.** The message is
`closure-disposition-scope-followup-20261018-005.md`, SHA-256
`4ac9c98f8c9245e30a2cd6d62870ce492bc6208dd5a6f2c30aec2b143649f36e` (14
lines). It was verified before reading. Its three inputs matched their
stated hashes before the edit: `e08bc128...`, `30187293...` and
`bc8d3753...`.

**What changed.** Text only. No run, measurement, source or test edit,
selection or policy. Every correction is accepted; none is contested.

| correction | decision | lines touched (in the files as revised) |
|---|---|---|
| 1. The "no attracting set of lower dimension" claim (C05) is too broad for noncompact sets. | accepted | consolidation 335 to 350 (section 1, item 3). The flow map preserves volume wherever it exists, under stated existence assumptions. No compact set can uniformly attract a bounded open neighbourhood of strictly larger volume. Noncompact invariant sets are not excluded (`x' = x`, `y' = -y`). Also report revision line 3 to 11 (sections 0 and 6 read with that qualification), and disposition row C05 (line 234). |
| 2. "In this ensemble it does not discriminate escape" (C11) overstates equal medians. | accepted | consolidation 513 to 518 (heading and body) and 1061 to 1065 (C41, the same wording in section 6). Report, for consistency: 1136 to 1144 (13.0 item 7) and 1468 to 1473 (13.3). Disposition row C11 (line 240). Each now reports the finite observation, equal median crossing times of the fired and censored groups, without concluding an absence of predictive information. |
| 3. "Changes nothing" (C32) contradicts the body. | accepted | consolidation 950 to 951 (heading: "always returned the carried lift"). Disposition row C32 (line 261). |
| 4. The C02 summary quoted the intermediate report hash `cd5c6b5b...`. | accepted | disposition row C02 (line 231) now quotes the binding `30187293...` on consolidation line 213 (formerly 202). No manifest or source re-audit. |
| 5. "Exactly held subset" in two converged fired cells (row 028-1). | accepted | disposition row 028-1 (line 83) and section 2's capped-map row (line 118). Report 1647 to 1653 (13.5). Consolidation 660 to 666 (section 4, item 2). Each now says the capped algorithm held the kept bounds and the energy to the recorded tolerance (`1e-12`) at every step, for 8 members and 600 units. Finite-tolerance enforcement is not exact set membership, so no row establishes insufficiency of the exactly enforced set. The conditional compactness algebra is unchanged. |

**Follow-on updates.** All line references were recomputed after the
insertions:

- the consolidation's revision note 2 (lines 34 to 98);
- every row of section 3's amendment table;
- the revision-note spans in section 3.

**What stays binding.** The parity holds, cone increment 3, root 025 and
the 1200-second whole-increment gate.

**Final hashes after 005.**

| file | SHA-256 | lines |
|---|---|---:|
| report | `12a0ffb2b1d29ad79c3087b64d457a673ed98158ed8654ce10dbeefb5f03d0f3` | 2433 |
| consolidation | `bb024fa81afcd12e89f7a12c8ee869a8a2142cc5454f9843bcf4af2dea081640` | 1373 |

The consolidation's line 213 now binds the report both at `286fb368`
(`30187293...`) and after these corrections (`12a0ffb2...`).
