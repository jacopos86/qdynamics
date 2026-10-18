# Independent review: electronic cone lane — prerequisite report, diagnostic 022, and release 012 against the receipt at `74466441` (Claude, 2026-10-18)

*Revised 2026-10-18, text only, under `classical-review-scope-20261018-003`: Candidate B restated as a descriptive observed-margin diagnostic (see the disposition block at the end).*

**Verdict.**
1. **The report is accurate.** Its increment 0 is blocked: the strict
   resume-parity gate fails by `1.9395497882168655e-4`, bitwise the same
   failure as the other two repaired lanes.
2. **Its independently authorized audits are sound and correctly scoped.**
   - The principal-Gram identity holds on 16,034 stored samples (largest
     error `3.11e-15`).
   - The electronic one-body density matrix stays positive at all 7,200
     reconstructed task-11 stage repairs.
   - Diagnostic 014 establishes, by intervention, the cause of the first
     `t = 0.09` roundoff difference.
3. **Diagnostic 022 is a sound bounded experiment.** It shows that the
   jointly differing pre-selection inputs suffice to reproduce the
   `t = 0.12` decision split. It does not identify which component
   matters or where the difference grew.
4. **All seven corrections in release 012 are valid.** Three correct real
   defects in my receipt at `74466441`.
5. **Increment 3 stays held.** No evidence is adopted (section 9).

**Scope.** This is a read-only review. I ran no propagation, re-selection
or audit and repeated no Codex measurement. I edited no Codex worktree and
no shared code. The only computations were SHA-256 recomputation and
reading saved JSON. The cone chat owns the shared causal diagnosis; this
review checks its scope and does not extend it. The reviewer wrote both
the handoff and the architecture receipt that this lane and release 012
build on.

---

## 1. Hash verifications performed

| artifact | expected | actual | match |
|---|---|---|---|
| message `codex-to-claude-20261010-013.md` | `b078b9a4…` | `b078b9a4df55ceb73e2c5ed3e8f7b695b89fd774ba7b2856cda045bedbcaea73` | yes |
| report (disk; committed at `f2879811` and unchanged at `c1375aa2`) | `145773a7…7a` | `145773a7fb69fac9dec445fd2d23a40300aa504f22ae40212a54fc2abebcc97a` (all three) | yes |
| diagnostic `cone-reselection-diagnostic-20261010.md` (disk, and at `c1375aa2`; 115 lines) | `fd4b869a…73` | `fd4b869a073e7074cdaae07f3b1bcbe4dd8a8a199a891ff0d1c4c3d27098c673` | yes |
| release `cone-architecture-release-20261010-012.md`, and its frozen copy in the lane's data | `65407575…25` | `654075758bf4cbcecab11ed8a0e068483bac73370bf38b892e04d6766e35ac25` (both) | yes |
| my receipt: tracked file, the blob at `74466441`, and the lane's frozen copy `architecture_receipt.md` | `7932fc02…ab` | `7932fc02392838e4529b5e945e2c1fa8949074133154edc8b9c02ad5a59661ab` (all three) | yes |
| handoff snapshot | `73128739…b5` | blob at `6b27596e`, tracked file and campaign copy: `73128739ec2101d684d6d6c8741b037fd1ba1fdbf3ad090ccb4c9cc2084a8fb5` | yes |
| diagnostic 022 dispatch `cone-reselection-diagnostic-20261010-022.md` | `47e87342…2d` (diagnostic line 9) | `47e87342026b306fa8d0098250c47eb5a44333913d6ca6951e4e49fd32fdcc2d` | yes |
| root's parity source review | `0082d8b9…5a` | equal | yes |
| root-verified parity receipts: acceptance, execution, segment-B resume receipt | `70181404…20`, `35472f80…e8`, `36495918…5d` (source review) | equal | yes |
| `cheap_projection_route.py` in the lane worktree (report line 59) | `25c4b0e3…4d` | equal, so the shared module is unchanged | yes |
| `input_manifest.json` (report line 12) | `ffb27a4a…2f` | equal; all 107 entries recomputed against their files: 107 OK | yes |
| `driver_revision_manifest.json` | as listed | all 26 entries OK | yes |
| `result_inventory.json` | as listed | all 134 unique entries OK | yes |
| `trajectory_inventory.json` | as listed | 40 seeded-random entries of 267, all OK | yes |
| diagnostic 022 `input_manifest.json` and `artifact_inventory.json` | as listed | 24 of 24, and 40 of 51 sampled, all OK | yes |
| parity trajectories | unbroken / A / B | `48fe68ef…` / `99e3edaf…` / `f6224b86…`, byte-identical to the regularized and repair lanes' files, which come from three distinct fixed drivers (`6d5eb134…`, `4225bfab…`, `8320827e…`) | yes |

## 2. The report's claims, against its receipts and code

| claim (report line) | receipt or code checked | finding |
|---|---|---|
| HF parity: seam bitwise in 56/70 and 31, reference delta 0, maximum `1.94e-4`, first failure at sample 49, `t = 0.1225` (26, 104–117) | `parity_hf_default_acceptance.json`; checker `drivers/parity_check.py` exits nonzero on any failed criterion (`:52-122`) | supported |
| task-11 parity not attempted after the HF failure (26, 113) | — | consistent with the handoff's stop clause (section 4) |
| identity on 16,034 stored samples: HF 8,013 plus task-11 8,021; largest full-spectrum error `3.1086244689504383e-15` (3, 142–145) | `identity_hf_t20.json`, `identity_task11_t20.json` | supported. The test compares the ordered principal spectrum to `(2λ−, 2λ−, 2λ+, 2λ+)`, which is stronger than my receipt's minimum-only identity |
| task-11 electronic minimum `0.0620598870597599` at `t ≈ 7.21` (143) | identity receipt | supported; a number the record lacked and the handoff asked for (handoff increment 1) |
| 10 task-11 checks show the scaled retained matrix is `D G D`, `D = diag(1, 1/sqrt 17 ×4, 1, 1, 1)`, error `5.55e-17` (145) | same | supported; agrees with `cheap_projection_route.py:773-786` (`phonon_envelope = 16`) |
| scan: 69 completed supported cheap-repair trajectories, none crosses `-1e-8`; 0 supported unrepaired-closure trajectories (34, 147–157) | `scan_input_manifest.json`, `electronic_scan_classified.json` | supported, and correctly bounded to snapshot-inventoried inputs as the addendum required |
| 7,200 reconstructed stage repairs electronically positive; retained joint Gram violated at most of them (4, 159–169) | `stage_inputs_task11_w{02,24,68}.json`, `stage_inputs_acceptance.json`, `share_reference_complete_acceptance.json` (192 aggregates, difference 0) | supported. The replay steps by adjacent stored-time differences, not by the integrator's fixed `h`, and the report says so (161) |
| passive observer: cone-memory input, cone output, affine-memory input and affine output; electronic principal minima at least `0.1232` in every category; only the affine output has zero affine-structure residual (171–188) | merged summary | supported; correctly qualified as default-path observations that do not test `w = a + e` (188) |
| diagnostic 014: changing only the initial native clock (`0x1.6666666666667p-4` against `…666p-4`) reproduces each saved `t = 0.09` output bitwise; state difference `2.89e-15` (50, 119–134) | `clock_counterfactual_one_step_014.json` and its manifests | supported. This is a controlled intervention, so "causal" is justified **for the first roundoff difference only**, and the report confines it there (50, 132) |
| no demonstrated missing persisted state, reference, configuration, solver object or repair history (136) | source review; diagnostic 022's fresh-cache reproduction | supported as "not demonstrated", which is how it is phrased |
| 521 passed, 9 warnings; 11 driver/gate tests (98, 218–221) | `final_suite_execution.json`, `gate_tests_final_execution.json` | supported |

## 3. Diagnostic 022 (`c1375aa2`)

**What it did, and how it reads.**
- **Recovery and reproduction pass.** It recovered the two pre-selection
  bundles at `t = 0.1175` by one inherited SSPRK step each, and
  reproduced both recorded decisions bitwise from fresh routes:
  - unbroken declines (95 iterations);
  - resumed accepts (90 iterations, displacement `1.007414834495438`).
- **The 2×2 crossing is clean.** It crossed each bundle with each
  predecessor clock and found the decision follows the bundle. Within a
  bundle, the effective inputs are bitwise unchanged by the clock
  exchange (lines 3, 49–58).

**What it supports.** The jointly differing bundle is sufficient for the
split:
- state `1.4432899320127035e-14` apart;
- standardized warm lift `1.0571648808091472e-15` apart;
- reference bitwise equal (lines 61–73).

All four completions are uncertified, with zero face dimensions and
infinite KKT diagnostic (line 58). Fresh-cache reproduction shows prior
solver history is not needed for these two decisions (line 43).

**What it explicitly does not establish** (lines 3, 115), and the review
agrees:
- whether the lower moments or the warm lift matter;
- where between `t = 0.0875` and `0.1175` the bundle difference grew;
- a source defect;
- general solver instability;
- a safe fix;
- passage of the original parity gate.

**Its one disclosed limitation does not affect the result.** An unused
failure path would stop generically on a missing candidate (line 99).
All four candidates exist with shape `(70,)`.

**Bounds held.** The numerical batch took `106.86 s`; the whole selection
stayed within its declared `1200 s` (lines 81–85).

## 4. Were the handoff's first-increment gates executed as written

The contract is my snapshot `73128739…`
(`agent_guidance/paper-v/handoffs/codex-paper-v-electronic-cone-projection-20261010.md`
at `6b27596e`).

| handoff gate | executed | note |
|---|---|---|
| increment 0 (lines 198–281): lock, byte-identical copies of both driver sets, readable inputs, per-segment references of both stored chains, fixes to both segment drivers, both parity gates | yes, except the second gate | the HF gate failed first; the stop clause (lines 290–293) then applies, so not attempting the task-11 gate is correct |
| acceptance (lines 282–289) | met except "both parity gates pass" | stored continuity at most `8.88e-16` (report table, lines 260–276) |
| C20 (lines 1043–1085) and gate (lines 1086–1098) | yes, with the addendum's executable checker | — |
| increment 1 stop clause (lines 356–360: "an identity failure larger than `1e-10` is not a stop … strengthens the case") | moot: the identity holds to `3.11e-15` | **withdrawn** by my receipt (section 1, `ARCHITECTURE…:116-121`) and by release 012 item 1; under both, an identity failure is a stop for diagnosis |
| increment 2 reading rule (lines 401–404: no stage input below `-1e-8` means the option comparison is expected to be bitwise) | the lane did **not** apply it, and correctly so (report 44, 188) | withdrawn by my receipt section 2 and by release item 2 |
| increments 3–7 | held | correct |

**Handoff defects the review owns.**
- My increment 0 sample checker printed booleans and never failed; the
  addendum corrected this.
- My C20 gate predicted strict parity from a correct seam, a prediction
  the record had already contradicted
  (`cheap_repair_route_results_20260918.md:645-650`).
- Two of my handoff rules were mathematically wrong: the increment-1
  identity stop clause and the increment-2 inertness rule. Both are now
  withdrawn.

## 5. Release 012's executable corrections, against my receipt at `74466441`

Each correction is checked against `cheap_projection_route.py` at
`25c4b0e3…` and my receipt's text. Lines are of the receipt.

| release item | receipt text it amends | review |
|---|---|---|
| 1. identity only on raw-coordinate, affine or reconstructed states; not on cone or correction-memory blocks | §1 (`:104-121`) states the identity "for every coordinate vector the code can produce" but never warns against feature-space iterates | **valid precision.** `y_k`, `c_k`, `z_k` and `w_k` are feature vectors outside the affine image. Their `S`-block need not have the `[[1, rᵀ], [r, I + iεr]]` form, so no ρ exists for them. The lane's observer confirms only affine outputs have zero structure residual (report 173). The receipt's claim that every cone output lies in `E` still holds, because a principal submatrix of a PSD block is PSD |
| 2. activation evaluated on `w_k = a_k + e_k`; a default-path affine audit is a zero-memory counterfactual; no electronic step at the final cap; test activation on a violating non-final iterate; withdraw "only an affine point below the floor can activate" | §2 (`:143-145`): "The only iterates at which `E` can act are the affine iterates `a_k`"; placement at `:155` already has `if k < K - 1` | **valid correction of a real over-statement.** The claim is true only before the first activation, while `e_k = 0`; afterwards `w_k = a_k + e_k` can leave `E` with `a_k` inside it. The induction for the bitwise prediction (`:166-172`) survives, because it is stated under all `a_k` in `E`, which keeps `e_k = 0` throughout |
| 3. preserve the `None`-path record schema: capture pre-edit stage key sets with `record_stages=True`; turn the printed Boolean comparison into assertions with nonzero exit; exclude timing fields; compare NaN masks | step 3a (`:346-366`) records repairs and a rollout without stage records; test 1 (`:415-417`) then asks for "exactly the pre-edit keys"; the step-3a comparison (`:362`) only prints | **valid; corrects two real defects.** Test 1 could not be executed from the receipt's own recording, and the double-recording check could pass by being read rather than asserted |
| 4. reject non-finite `epsilon` as well as values outside `[0, 0.5]`; no default-path eigenanalysis or record keys; inactive projection bitwise; 16-feature cylinder | §6 step 3c (`:395`) refuses values outside `[0, 1/2]` without saying non-finite; §2 and §3 already require the rest | **valid hardening.** A NaN passes `epsilon < 0 or epsilon > 0.5` |
| 5. validate reconstruction, residual, Weyl bound and mapped identity; separate the three outcomes; a cap with unmet floors proves neither infeasibility nor convergence; handle skipped or already-feasible repairs with iterations 0 and no invented early exit | §3 item 4 and §4 (`:247-293`) define the checks and outcomes but leave the skipped-repair case (`cheap_projection_route.py:1233-1242` returns `unchanged()`) without a classification | **valid completion** of my section 4 |
| 6. a positive-floor test must exhibit an explicitly feasible successful repair and assert success before the floor; the one-iteration cap test requires the last-affine return without treating residual floors as exceptions | test 5 (`:429-433`) is conditional ("when `floors_met_at_return` is true …") | **valid; corrects a real defect.** My test would pass vacuously if no repair ever succeeded |
| 7. convergence claims need a feasible intersection and compatible orthogonal projections; the declared-metric energy direction is oblique; electronic positivity is weaker than joint-Gram positivity | §3 (`:200-204`) and §5 (`:319`) say the option at `epsilon = 0` "cannot change the intersection the alternation converges toward" | **valid correction of an over-statement.** The intersection is unchanged, but convergence to it is not established: `K` is finite, the intersection may be empty with the hyperplane on (no constrained chain holds a cone past `t = 8`), and under `raw_coordinate_motion_scales` the hyperplane is met along an oblique direction (`cheap_projection_route.py` docstring, settings block, and `energy_correction_direction`). The sentence should read "cannot change the intersection of the sets; whether the alternation reaches a point of it is not established" |

**Summary of section 5.**
- Items 1, 2 and 7 correct over-statements in my receipt.
- Items 3 and 6 correct defects that would have made two of my receipt's
  tests vacuous or unexecutable.
- Items 4 and 5 add needed cases.

The release keeps the receipt's architecture: the cone/affine/electronic
order, the last-affine cap return, the 2ε floor, the 16-feature cylinder
and the bitwise `None` path. I accept every correction, and the receipt
should be read with them. The release is conditional: it takes effect
only after increments 0–2 pass and are reported. Increment 0 has not
passed.

## 6. Defects and unsupported statements

1. **"Bounded diagnostic 014 … This establishes the first roundoff
   difference's cause"** (`REPORT-…:50`, heading `:119`). This is
   supported by a controlled intervention and correctly bounded. The
   coordinator has warned against inferring causation from time-grid
   differences; that warning does not apply to this intervention. It
   would apply to any extension of 014 to the `t = 0.12` split, which the
   report does not make (`:132`, `:134`). No defect.
2. **The increment-2 replay uses adjacent stored-time differences**
   (`:161`), not the integrator's `h`. Its 7,200 counts are therefore
   observations of a reconstruction. The next-stored-state check, at most
   `1.93e-14`, supports the reconstruction but does not certify bitwise
   original stage inputs. Disclosed; it bears on how far section 7 can
   use these counts.
3. **An initial share command requested only `twin_both`** (`:44`). It
   was superseded by the full three-twin reproduction. Disclosed.
   Provenance-only.
4. **The broad scan's exclusions remain unresolved** (`:36`, `:151-155`):
   192 snapshots lack archive coordinates and 87 report no success.
   Disclosed. The "no crossing" statement is confined to the 69
   (`:157`).

I found no unsupported scientific statement.

## 7. Architectural next-step candidates (recorded as candidates only; none is proposed for execution, none is released)

- **Candidate A — a carried global clock.** An opt-in option in
  `integrate_cheap_projection_ssprk3` that builds sample and stage times
  from a carried global step index, `t_n = t_0 + n h` with `n` global,
  instead of `initial_time + linspace(0, span, intervals + 1)`. The
  current form is at `cheap_projection_route.py:1928` per root's source
  review. Under it, a resumed segment's clocks would equal the unbroken
  run's bit for bit.
  - **Basis:** diagnostic 014 showed the first roundoff difference
    (`2.89e-15` at `t = 0.09`) is caused by the native clock. Diagnostic
    022 did not test whether the whole bundle difference at `t = 0.1175`
    descends from that clock.
  - **Test:** the candidate would succeed only by meeting the corrected
    prerequisite of section 8.
  - **Owner:** the cone lane, as sole editor of `cheap_projection_route.py`,
    with impact analysis, a bitwise default-path test and root release.
- **Candidate B — a descriptive observed-margin diagnostic at every
  re-selection.** Record, read-only, each decision's observed distance
  from its acceptance boundary: the difference of solved and carried
  objectives, against the margin, beside certification and face
  dimensions. It changes no policy.
  - It is descriptive only. An objective distance to the boundary does
    not by itself establish how sensitive the optimizer and acceptance
    map are to `1e-14` input perturbations.
  - So it would not flag which decisions such changes could flip, and it
    would not decide whether a passing gate was fortunate.
  - Any perturbation guarantee would need a separately established
    sensitivity or robustness bound. No such bound or measurement is
    requested or selected here.
- **Candidate C — an increment-3 audit that reuses the passive observer
  on exact stage inputs.** The lane's default-path affine outputs had
  electronic principal minima at least `0.124` in every replayed window
  (report table, lines 175–184). As a zero-memory counterfactual (release
  item 2), this would support a bitwise inertness prediction at
  `epsilon = 0` for those windows, but it does not certify one: the inputs
  were reconstructed. If increment 3 is ever released, step 3b should
  capture stage inputs during an actual rollout instead of reconstructing
  them, and should add the HF rollout of receipt step 3b(ii).
- **Not proposed.** Any change to the acceptance policy, such as a
  certified-only rule: the record refuses it for accuracy (CR section 4,
  item 3). Also not proposed: any tolerance relaxation, or a parity gate
  restated as a band. Each is a scientific decision for root and the
  user, outside this review.

## 8. What the failed gate does and does not establish

**It establishes.** At the default HF task-11 repair, the fixed driver
resumes bitwise in 56/70/31 with an exact reference. A two-segment run
spliced at `t = 0.08` then departs from the unbroken run by up to `1.94e-4`
over `t <= 0.2`. The departure follows a different decision at the
uncertified re-selection near `t = 0.12`.

Diagnostics 014 and 022 add, by intervention, two facts:
- the native clock causes the first roundoff difference;
- the differing `1e-14`-level bundle suffices for the split.

This holds identically in the three repaired lanes. It blocks increment
3, every option-on chain, and the task-11 comparison.

**It does not establish:**
- that the clock causes the split;
- which bundle component matters;
- a source defect;
- that any recorded chain is wrong;
- anything about the electronic set's effect, since no option code
  exists;
- any physical statement.

This review waives no tolerance and infers no causation from the
time-grid or optimizer chronology.

**What a corrected prerequisite acceptance would have to show.**
1. **The already selected gate passes, unchanged.** The executable gate
   passes at the HF setting, and then at the task-11 setting:
   - seam bitwise;
   - grid within `1e-12`;
   - all samples within `1e-12` through `t = 0.2`, across the
     re-selections at `0.12` and `0.16`.
2. **The carried-reference receipt passes.** The reference is within
   `1e-14`. This is a distinct obligation: re-selection itself takes
   neither the clock nor the carried energy reference as an input
   (diagnostic 022, line 28).
3. **After any change, the gate passes again with the option on,**
   before increments 5 and 6.
4. **Root decides.** Root reviews the actual corrected acceptance and
   selects any future contract or code change. This review selects none.

**What is and is not known about passing that gate.**
- **What 022 shows.** The observed, jointly differing non-clock bundle
  suffices for the one controlled decision split.
- **What 022 does not show:**
  - that every nonzero pre-selection difference must cause a later
    failure;
  - that bitwise equality of every pre-selection input is necessary to
    pass the existing coordinate-tolerance gate;
  - that inability to achieve such equality would leave chaining unable
    to meet that gate.
- **Limits in both directions.** A finite passing window obtained with
  nonidentical inputs would give no general guarantee for longer
  horizons. Equally, the present evidence gives no universal
  impossibility result for passing with such inputs.
- **A possible strategy, not a gate.** Bitwise identity of the selector's
  inputs between unbroken and segmented runs, with software, solver and
  RNG settings bound, is one possible sufficient reproducibility
  strategy. The selector's inputs are the state, the warm lift and the
  lower moments. Adopting that strategy as a contract would be a future
  candidate requiring root selection; it is not a necessary condition
  and not a gate. Any code change toward it would be a named, default-off
  option with a bitwise default-path test, made in this lane as the single owner of `cheap_projection_route.py`,
  and, released by
  root.

## 9. Holds and evidence status

Every hold remains binding:
- root messages 014 (`cone-parity-prerequisite-20261010-014`), 015
  (`cone-parity-review-20261010-015`), 016, 017 and 019;
- cone increment 3, which release 012 does not release while increment 0
  has failed;
- root hold 025;
- the three strict parity gates (regularized, repair-early, cone), each
  failed by `1.9395497882168655e-4`.

Increments 3–7 remain held and unmeasured: activation counts, option
RMS, wall ratios and task-11 comparisons. No shared-source change, clock
fix or solver-policy change is released. This review adopts no evidence.
All outputs remain `exploratory_local_not_promoted`.

**Consistency note (2026-10-18).** Root's focused wording correction
`review-regularized-scope-20261018-001` (SHA-256 `fe521731…c9`) was
addressed to the regularized review. Its section-7 concern applies
equally here, so the same text-only correction was applied to section 8
and to the wording of candidate B in section 7. Nothing else changed.

## Disposition of classical-review-scope-20261018-003

Message `classical-review-scope-20261018-003` (SHA-256 `2ebb3296…b4`,
11 lines, verified before reading). Input version: `77ff56bd…`, 314
lines. Text-only. No source or test edit, rerun, new gate or tolerance
change. The 001 corrections are not reapplied.

| correction | disposition | where |
|---|---|---|
| Candidate B: remove the claim that the margin measurement would flag `1e-14` flips or decide whether a gate pass was fortunate; restate it as a descriptive observed-margin diagnostic, conditional on a separately established sensitivity or robustness bound for any perturbation guarantee; request or select no such bound | accepted | section 7, Candidate B |
| preserved unchanged | already present | holds 014–017 and 019; conditional cone increment 3; root 025; the 1200-second whole-increment bounds; every other section |
