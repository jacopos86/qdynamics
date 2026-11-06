# Independent review: regularized product-state lane, stopped at its increment-0b gate (Claude, 2026-10-18)

**Verdict.** The report is accurate, and it stopped where the handoff
requires. The fixed driver carries the seam and the Eq. (22) reference
exactly; the strict resume-parity gate fails by `1.9395497882168655e-4`
against `1e-12`. That failure is the same, bit for bit, as the one in the
two other repaired lanes. It blocks every dependent increment. Nothing in
this lane is adopted, and every hold stays binding (section 8).

**Scope.** This is a read-only review. I ran no propagation and repeated no
Codex measurement. I edited no Codex worktree and no shared code. The only
computations were SHA-256 recomputation of the receipts and reading of
saved JSON. The reviewer is the author of the handoff this lane executed,
so section 4 holds that handoff's text against its execution, including
the handoff's own defects.

---

## 1. Hash verifications performed

Every expected hash below was supplied by root message
`codex-to-claude-20261010-013`. I verified that message itself first:
`/Users/lordskyler/Documents/ChatGPT/Holstein/campaign-20261010/codex-to-claude-20261010-013.md`,
actual `b078b9a4df55ceb73e2c5ed3e8f7b695b89fd774ba7b2856cda045bedbcaea73`.

| artifact | expected | actual | match |
|---|---|---|---|
| report `REPORT-regularized-product-state-codex-20261010.md` (on disk) | `037874da…f2` | `037874da4695467ae7b919a7d1a45704ba8752f3017060ff70c2421ac18483f2` | yes |
| the same file as committed at `72d6d67a` (`git show`) | `037874da…f2` | `037874da…f2` | yes |
| edited driver `…-data/drivers/run_divergence_segment.py` (disk, and at `72d6d67a`) | `4225bfab…bc` | `4225bfabe5549ff2ca44bb520c5b68f8ebc1920ebd16490147bc6c87914325bc` | yes |
| the driver's working copy under `output/local_runs/paper_v_regularized_product_state_codex_20261010/` | `4225bfab…bc` | `4225bfab…bc` | yes |
| checker `…-data/drivers/parity_check.py` (disk, at `72d6d67a`, and its output copy) | `cbb6479f…8d` | `cbb6479ff8116f28b0da73546079d33313a56c7ee48dfeb6f33bf0a54b770e8d` | yes |
| handoff snapshot the lane cites (report line 181) | `e2fe9267…10` | the blob at `6b27596e`, the tracked file at this branch's HEAD and the campaign `classical-handoffs/` copy all hash to `e2fe92679028b312682fcf6c1d5c502c4c162bd2ef5a4f9f8468a4c544510710` | yes |
| input lock the lane cites (report line 183) | `e2e2ba4a…88` | the blob at `6b27596e` and the campaign copy: `e2e2ba4a071f5a6b6bb156aaa2d0164f3f4de16302ff9e8b9eb3fab815dbf888` | yes |
| root parity source review `parity-source-review-20261010.md` | `0082d8b9…5a` | `0082d8b9bed428e83b8f0c30593c01427c84365fce20b4ec1bfe60fd1e5f635a` | yes |
| all 75 entries of `result_manifest.json` (recomputed from the files, not sampled) | as listed | 75 of 75 match in hash and byte size; none missing | yes |
| `input_manifest.json` digest recorded in `result_manifest.json` | `2272651d…b5` | `2272651db6e1b32bae9d368b7edcbe47bbcb788eae97ad79561e6ada3150f5b5` | yes |
| 30 seeded-random entries of `input_manifest.json` | as listed | 30 of 30 | yes |
| parity trajectories (report lines 192–194) | `48fe68ef…51`, `99e3edaf…6f`, `f6224b86…6c` | equal | yes |

**Cross-lane check.** The repair-early lane wrote its parity trajectories
with a different fixed driver (SHA-256 `8320827e…`), and so did the
electronic-cone lane (`6d5eb134…`). Both lanes' unbroken and segment-B `trajectory.npz`
files hash to the same `48fe68ef…` and `f6224b86…` as this lane's. The
cone lane's segment-A file is also `99e3edaf…`. So the three recorded
lane runs reproduce bitwise across their three fixed drivers, and the
failure is not an artifact of one lane's driver. This does not establish
unrestricted run-to-run determinism across all processes, machines,
library versions or settings.

## 2. What the report claims, against its receipts and code

| claim (report line) | receipt or code checked | finding |
|---|---|---|
| 49/49 locked inputs and 21/21 byte-identical driver copies (16, 22–23) | `input_manifest.json`; 30 entries recomputed | supported |
| 508 passed, 9 warnings, `830.90 s` end to end (17) | `baseline.json` (`passed 508`, `end_to_end_seconds 830.9`, log SHA recorded) | supported; under the 20-minute bound |
| HF Eq. (22) energy `-3.499999999989493`, truncation weight `6.741274205523951e-13` (16) | `initial_audit.json` | supported; these meet handoff increment 0's acceptance (energy `-3.5` to `1e-9`, truncation weight to two digits) |
| retained joint Gram spectrum (171) and "rank 5" (16, 27) | `initial_audit.json` | the numbers are supported; the reading needs qualifying (section 6, item 1) |
| driver: complete 56/70 checkpoint load, no rebuilt `t = 0` completion, carried reference from the matched sample, fresh output directory (40) | diff of the edited driver against the record's `run_divergence_segment.py`: new `_load_resume` (validates configuration, rule, dictionary, preparation and lift; matches exactly one sample at `1e-9`; requires a re-selection boundary; compares the stored state/lift histories bitwise); output `mkdir` moved after preparation/load and refused if it exists; `energy_reference_value=` passed to the integrator; `resume_input_receipt` and three per-segment reference fields written | supported; implements the handoff's C20 items 1–4 and the addendum's receipt rules |
| checker exits 1 on a genuine hard stop after every structural check passes (40) | `parity_check.py:95-205`: configuration and dimension equality, checkpoint shapes and finiteness, single matched resume sample at a re-selection boundary (`:122-127`), bitwise 56/70 seam on both sides (`:130-135`), bitwise closed seam (`:136`), receipt hashes (`:139-151`), reference within `1e-14` with a hard-stop tier above `1e-12` (`:157-162`), grid within `1e-12` (`:176-180`), coordinates within `1e-12` with a hard-stop tier above `1e-10` (`:185-197`); non-zero exit on any failure (`:228`) | supported; this is the executable gate the addendum required, and it encodes the acceptance/hard-stop tiers explicitly |
| first failure: sample 49, `t = 0.1225`, coordinate 0, `1.57266010158974e-7`; maximum `1.9395497882168655e-4` at coordinate 13, `t = 0.17` (3, 130–132) | `parity_task11_gate.json`, `independent_parity_review.json` | supported, every digit |
| complete seam exact; reference delta 0; grid discrepancy `2.7755575615628914e-17` (127–129) | same receipts | supported |
| decision split at `t ≈ 0.12`: unbroken declines (carried `7.744300602781718`, solved `7.937620280484198`, 95 iterations); resumed accepts (carried `7.744300603050708`, solved `7.274900400659208`, 90 iterations, displacement `1.007414834495438`); both uncertified (140–145) | re-selection records inside `parity_task11_gate.json` | supported; the receipt also shows the next re-selection, at `0.16`, accepted in both runs (resumed certified, unbroken uncertified) |
| reference table: segment A resume `-3.499974910696376` is not its terminal `-3.4999389875638798`; segment B starts at the resume value (149–157) | gate receipt, summaries | supported; this demonstrates why the handoff required the matched sample rather than the terminal one |
| no tolerance, algorithm or setting changed (10, 123); no source or test diff (64, 112) | protected-source diff recorded in `final_validation.json`; `git diff --stat b9f99271 72d6d67a` touches only the report and its data namespace | supported |

## 3. The failure, read against the record

This failure mode is already in the record. The consolidated cheap-repair
summary records it in
`MATH/paper_facing/paper_V_high_u_gkba/cheap_repair_route_results_20260918.md:645-650`:
"a chained run agrees with a single rollout to `2.5e-14` through three
re-selections and can flip a fourth at the feasibility test's edge". The
example there was a `0.89` move followed by `d31` of `3.3e-4` two samples
later. Here the flip comes at the first re-selection after the resume. The
returned candidate changes (solved objective `7.94` against `7.27`), so the
decision changes too. The resulting coordinate difference, `1.94e-4`, is
of the same order as the record's `3.3e-4`.

The cone lane's controlled diagnostic 022 (`c1375aa2`) shows that the
jointly differing pre-selection inputs suffice to reproduce this split:
state `1.44e-14` apart and warm lift `1.06e-15` apart, at an uncertified,
facially degenerate completion. This review does not repeat or extend
that diagnosis; the cone chat owns it.

## 4. Were the handoff's first-increment gates executed as written

The contract is my handoff snapshot `e2fe9267…`
(`agent_guidance/paper-v/handoffs/codex-paper-v-regularized-product-state-20261010.md`
at `6b27596e`, unchanged at this branch's HEAD).

| handoff gate | executed | note |
|---|---|---|
| increment 0 commands (lines 261–343): worktree at `b9f99271`, `shasum -a 256 -c "$LOCK"` (line 279) | yes, with the coordinator-provisioned worktree, as addendum `classical-dispatch-20261010-007` directs | 49/49 OK, rechecked at the end |
| copies byte-identical, readable inputs, full spectrum (lines 280–309) | yes | spectrum recorded in full |
| acceptance (lines 347–356): energy, truncation weight, two smallest eigenvalues within `1e-15`, largest within `5e-4`, rank recorded, five segments `exact_moments`, walls `4,296.9 s`, stored reference continuity `<= 1e-12`, parity gate | all met except the parity gate | stored continuity `8.881784197001252e-16`; walls `4296.889898292022 s` |
| stop (lines 357–362): parity failure, with the two decision records reported if a decision flips | met and honoured: the lane stopped and reported both records | correct |
| C20 driver fix (lines 1042–1084) and parity gate (lines 1085–1098) | yes; exact three commands (report lines 45–48) plus `-B` and `/usr/bin/time` | the gate's tolerances were not touched |
| C16 baseline before any source change (line 956) | yes, in sub-increment 0a | equivalent to the handoff's "increment 1"; deviation disclosed |
| C10 20-minute bound (line 901) | yes: baseline `830.90 s`, parity batch `283.92 s` | — |

**Handoff defects the review owns.** The sample `parity_check.py` in my
increment 0 (lines 315–330) printed diagnostics but never failed. The
addendum replaced it with an executable gate, and the lane built that
gate.

My C20 coordinate gate (`1e-12` over `t <= 0.2`, spanning post-resume
re-selections) could be failed by the record's own documented
non-bitwise mode. I anticipated this in the stop clause but did not
change the gate. The gate behaved as designed. The prediction it encoded,
that a fixed driver plus a correct seam would give strict parity, was
wrong.

## 5. Qualifications against the handoff's evidence standards

The report meets the standards at handoff lines 652–667 and in places
exceeds them.

- The checker compares the full 56-coordinate state and 70-coordinate lift
  at the seam. The handoff asked only for the 31-coordinate state.
- The report keeps the declared carried reference distinct from the
  terminal energy.
- Every receipt hash is recorded.
- It states that the 508-test suite does not certify the strict gate
  (line 145).
- It states that the chronology does not establish timestamp arithmetic
  as the cause (145, 207).
- It separates propagation success from parity (41).
- The 50/50 HF/GS mixing witness (`4.185822272628759e-6` at index 21, line
  171) is the nonlinearity check the handoff's increment-1 test required.
  The report does not claim it as certification.

## 6. Defects and unsupported statements

1. **Rank 5 is a tolerance placement, not a refutation of "4"**
   (`REPORT-…:16` and `:27`).

   **The spectrum** (`:171`) is `-4.44e-16`, `-4.21e-16`, `4.20e-12`,
   `1.05e-11`, then `0.595`, `1.0`, `2.0`, `8.405`. The handoff's tolerance,
   `1e-12 × 8.405 = 8.4e-12`, falls between the third and fourth near-zero
   eigenvalues, so the count at that tolerance is 5. Ten orders of
   magnitude separate `1.05e-11` from `0.595`. The gap structure therefore
   shows four near-null directions: two at machine zero and two at about
   `1e-11`.

   **The handoff's derivation predicted four:** two exact, from the
   idempotent electronic block, and two from the coherent-state null
   vectors, which are exact only at infinite cutoff (truncation weight
   `6.74e-13`). Assigning the pairs that way is my reading; it would need
   the eigenvectors to establish.

   **Correction:** "rank 5 at `1e-12 × largest`; four eigenvalues at or
   below `1.1e-11`". Calling it a "surprise" overstates it. The defect is
   in my handoff's rank rule, which the lane applied literally.
2. **The original three-process supervisor heredoc was not saved**
   (`:42`). The exact argv, deadlines and resource receipts were saved and
   suffice for reproduction. This is disclosed, minor, and not a blocker.
3. **The checker does not enforce two checks the report verified
   independently**: the receipt's `trajectory_sha256` and
   `source_directory` (`:145`; `parity_check.py:139-151`). This is
   disclosed as future hardening and does not change the verdict.

I found no unsupported scientific statement.

## 7. What the failed gate does and does not establish

**It establishes.** At the default HF task-11 setting, the fixed driver's
seam is bitwise exact in the 56 state coordinates, the 70 lift
coordinates and the 31 closed coordinates. The carried Eq. (22) reference
is exact (delta 0).

A two-segment run spliced at `t = 0.08` then departs from the unbroken run
by up to `1.9395497882168655e-4` in the 31 closed scalar coordinates over
`t <= 0.2`. The first excursion above `1e-12` is at `t = 0.1225`, after
the re-selection at `t = 0.12` returns a different candidate and makes a
different decision. Both runs are uncertified at that re-selection.

The result reproduces bitwise in two other lanes (section 1). The strict
prerequisite for every chained setting of this lane is therefore not met.

**It does not establish:**
- the cause of the `1e-14` pre-selection differences;
- that the integrator's time grid or the optimizer is the cause;
- that a source defect exists;
- that any recorded chain is wrong;
- anything about the admixtures, whose preparation module, probes and
  chains were never started.

This review waives no tolerance and infers no causation from the grid or
optimizer chronology.

**What a corrected prerequisite acceptance would have to show.**
1. **The already selected gate passes, unchanged.** The same executable
   gate passes at this lane's own settings, with tolerances unchanged:
   - seam bitwise in 56/70/31;
   - grid within `1e-12`;
   - all 81 samples within `1e-12` through `t = 0.2`, across both
     post-resume re-selections at `0.12` and `0.16`;
   - first at the default HF setting, then at every admixed setting
     before its chain (handoff lines 606–623).
2. **The carried-reference receipt passes.** The reference is within
   `1e-14` at the seam, in checkpoint, receipt and trajectory. This is a
   distinct obligation: re-selection itself takes neither the clock nor
   the carried energy reference as an input.
3. **Root decides.** Root reviews the actual corrected acceptance and
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
  option with a bitwise default-path test, made in the module's single
  owner (the cone lane, for `cheap_projection_route.py`) and released by
  root. This lane adopts no change of its own.

## 8. Holds and evidence status

Every hold remains binding:
- root messages 014, 015, 016, 017 and 019, including this lane's own
  stop steering `product-state-parity-stop-20261010-019`;
- cone increment 3, held by 014/015 notwithstanding the conditional
  release 012;
- root hold 025 on the divergence lane's weak-cutoff increment;
- the three strict parity gates: regularized, repair-early and cone, each
  failed by `1.9395497882168655e-4`.

This review adopts no evidence and releases nothing. The lane's
preparation module, increments 1 to 4, probes, comparisons and chains
remain not started and held. All of the lane's outputs remain
`exploratory_local_not_promoted`.

## Disposition of review-regularized-scope-20261018-001

Message `review-regularized-scope-20261018-001`
(`/Users/lordskyler/Documents/ChatGPT/Holstein/campaign-20261010/regularized-review-scope-followup-20261018-001.md`,
SHA-256 `fe521731b69dacab6083896992af924b1efc38a79a5513617992b9ed197f2ec9`,
15 lines, verified before reading). Text-only changes; nothing rerun, no
code, tolerance, fix or release touched.

| required change | disposition | where |
|---|---|---|
| remove or qualify "A pass with nonzero pre-selection differences would not carry to longer chains" | accepted: removed | section 7, former item 2 |
| remove the implication in item 4 (no bitwise-equal inputs means chaining cannot meet the gate) | accepted: removed | section 7, former item 4 |
| bitwise input identity, with software, solver and RNG settings bound, stated as a possible sufficient strategy or future contract candidate requiring root selection, not as a necessary condition or a gate | accepted | section 7, "What is and is not known", last bullet |
| a finite passing window with nonidentical inputs gives no general longer-horizon guarantee, and the present evidence gives no universal impossibility result either | accepted | section 7, "What is and is not known", third bullet |
| re-selection takes no clock or carried-reference input; the carried-reference receipt is a distinct obligation | accepted | section 7, acceptance item 2; the reference is removed from the selector-input list |
| section 1: the three recorded lane runs reproduce bitwise across their fixed drivers, which does not establish unrestricted run-to-run determinism | accepted | section 1, cross-lane paragraph |
| keep the selected strict gate, the failed parity result, every dependent hold, root 025, the 1200-second complete-increment gates, the stopped-scope verdict and the rank qualification | already present, unchanged | verdict; sections 6 (item 1), 7 and 8 |

The same wording correction was applied, for consistency, to the
parallel passages of `REVIEW-repair-early-then-bare-claude-20261010.md`
and `REVIEW-electronic-cone-projection-claude-20261010.md`.
