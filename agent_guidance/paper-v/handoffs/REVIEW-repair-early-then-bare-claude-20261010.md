# Independent review: repair early, then bare — stopped at strict resume parity (Claude, 2026-10-18)

**Verdict.** The report is accurate. The lane stopped where the handoff
requires. It built a sound fixed driver and an executable gate, both with
tests. The seam and the carried reference are exact. The strict
coordinate gate fails by `1.9395497882168655e-4`. That failure is
bitwise the same as in the regularized and cone lanes.

None of the handover question was measured. Unmeasured are the smallest
`T_r`, its repair wall, and whether a cheaper repair buys the same
handover. Nothing is adopted; every hold stays binding (section 8).

**Scope.** This is a read-only review. I ran no propagation and repeated
no Codex measurement. I edited no Codex worktree and no shared code. The
only computations were SHA-256 recomputation and reading saved JSON. The
reviewer wrote the handoff this lane executed; section 4 holds the
execution against that text.

---

## 1. Hash verifications performed

| artifact | expected (message 013 or the report) | actual | match |
|---|---|---|---|
| message `codex-to-claude-20261010-013.md` | `b078b9a4…` | `b078b9a4df55ceb73e2c5ed3e8f7b695b89fd774ba7b2856cda045bedbcaea73` | yes |
| report `REPORT-repair-early-then-bare-codex-20261010.md` (disk, and committed at `95d9f013`) | `47663d9a…2a` | `47663d9ab124ed8ac5fbec1526853c4a398a8299983792bfe1e7ca18221f432a` | yes |
| handoff snapshot (report line 14) | `22b86506…f9` | blob at `6b27596e`, the tracked file at HEAD and the campaign copy: `22b865060cb9f8b478aa9397686398114657b330b4e3ef923a0d12a4b46e91f9` | yes |
| fixed driver, durable and working copies, and the blob at `95d9f013` (report line 201) | `8320827e…bc` | `8320827eb126d2517cee9aec27320a03070d086d809fdbf9303a9994e5a15fbc` (all three) | yes |
| final checker, durable and working copies, and blob at `95d9f013` | `f28e213f…f5` (input manifest `edited_drivers`) | `f28e213fafaf481ff2beb32504784b411733477985552938e81855deadc668f5` | yes |
| HF and exact-lift helpers (report line 201) | `83f345ac…ac`, `d18bcf14…5f` | equal | yes |
| all 25 entries of `result_manifest.json`, recomputed in full | as listed | 25 of 25 | yes |
| all 85 `inputs` entries of `input_manifest.json`, recomputed in full against the source worktree | `expected = actual` as recorded | 85 of 85 | yes |
| `original_copies` entries of `input_manifest.json` | as recorded | 41 match; 1 differs: the `run_divergence_segment.py` working copy, which records the pre-edit copy hash and was then edited as declared. The edited state is recorded separately under `edited_drivers` and matches | expected, not a defect |
| parity trajectories | unbroken / segment B | `48fe68ef…51` / `f6224b86…6c`, byte-identical to the regularized lane's (driver `4225bfab…`) and the cone lane's (driver `6d5eb134…`) files, so the three recorded lane runs reproduce bitwise across their fixed drivers (this does not establish unrestricted run-to-run determinism across all processes and settings) | yes |

## 2. What the report claims, against its receipts and code

| claim (report line) | receipt or code checked | finding |
|---|---|---|
| 85 locks, 21 copies byte-identical, every NPZ/JSON opened (23) | `input_manifest.json`, fully recomputed (section 1) | supported |
| baseline 508 passed, 9 warnings, `832.13 s` (32) | `baseline_receipt.json` | supported; under the 20-minute bound |
| driver repair: geometry-only rebuild on resume, complete 56/70 load, matched-sample reference, full histories, no reused directory, end-to-end deadline (49) | driver at `8320827e`; tests `paper_5/tests/test_repair_segment_resume.py` (26 tests, incl. `test_resume_does_not_prepare_initial_completion`, `test_existing_output_refused_before_preparation`, `test_new_checkpoint_reference_uses_resume_index`, `test_checkpoint_is_accepted_reselection_boundary`, `test_deadline_covers_preparation`) | supported; implements C20 items 1–4 and the addendum, and goes further by testing them |
| checker exits nonzero on any unmet criterion (64, 154) | `drivers/parity_check.py:24-131`: grids, shapes and finiteness (`:30-48`), configuration and dimension provenance (`:49-59`), single matched sample (`:62-65`), bitwise state/lift on both sides of the seam (`:68-72`), bitwise closed seam (`:73`), receipt hashes (`:75-85`), reference within `1e-14` including the `none` case (`:86-107`), grid within `1e-12` (`:108-112`), coordinates within `1e-12` (`:113-124`); exit code (`:148`); 18 gate tests in `test_repair_parity_gate.py` | supported; this is the executable gate the addendum required |
| first failure sample 49, `t = .1225`, coordinate 0, `1.57266010158974e-7`; maximum `0.00019395497882168655` (62) | `parity_task11_receipt_final.json` | supported, every digit |
| time bits first differ at sample 35 (`0.08750000000000001` against `0.0875`, gap `1.39e-17`); state/lift bits first differ at sample 36; lift differs by `0.6036` at `t = .12` (185) | `parity_chronology.json`, which hashes 12 inputs before and after inspection | supported; root's source review and the cone lane's 014/022 record the same chronology |
| decision split at `.12`: decline (`7.744300602781718` / `7.937620280484198`) against accept (`7.744300603050708` / `7.274900400659208`); both uncertified, facially degenerate, infinite KKT diagnostic (187) | receipt's nearby re-selection records | supported |
| stored chain: 24,059 samples with overlaps, 24,001 after the trim; largest overlap difference `1.687538997430238e-14`; four tails beyond the checkpoint (183) | `overlap_audit.json`, `stored_chain_audit.json` | supported. This answers the handoff's section 3 fact 2 (line 196), which predicted chaining agreement near `2.5e-14` from the record |
| 552 passed = 508 + 44 new; first full-suite attempt's 1 failure and 9 errors were a `--basetemp` invocation error (144–153) | `final_suite_retry2_receipt.json`; the interrupted and failed attempts are preserved | supported, and candidly disclosed |
| "the historical `12,990.087148168066 s` is cumulative through `t = 60`, not additional cost over `(20, 60]`" (120) | stored summaries | supported. It also corrects any reading of the handoff's anchor table that treats `217.3 s` per unit as additive |
| increments 1–6 skipped; `repair_handover_sweep.py` not created (68–138) | `git diff --stat b9f99271 95d9f013`: no `paper_5/src` change; only the two test files, report and data | supported |

## 3. The failure, read against the record

The failure is the record's documented non-bitwise chaining mode, for
which the stop clause of my increment 0 named the response.
`cheap_repair_route_results_20260918.md:645-650` describes it: chained
runs agree to `2.5e-14` through three re-selections and can flip a fourth
at the feasibility test's edge, giving `3.3e-4` in `d31` two samples
later. Here the flip is at the first post-resume re-selection, because
the returned candidate itself differs. The cone lane's diagnostic 022
(`c1375aa2`) reproduces the split from the jointly differing
pre-selection inputs. Those inputs differ by about `1e-14` in state and
`1e-15` in warm lift. This review does not extend that diagnosis.

## 4. Were the handoff's first-increment gates executed as written

The contract is my snapshot `22b86506…`
(`agent_guidance/paper-v/handoffs/codex-paper-v-repair-early-then-bare-20261010.md`
at `6b27596e`).

| handoff gate | executed | note |
|---|---|---|
| increment 0 commands (lines 267–349): lock check, byte-identical copies, readable inputs, stored-chain counts and walls, stored reference per segment, driver fix, parity gate | yes, split into bounded sub-increments; the provisioned worktree was reused per the addendum | the three parity commands (report lines 54–56) are the handoff's, with `--max-wall-seconds 1150` added for the driver's end-to-end deadline |
| acceptance (lines 351–363): 85 OK; copies identical; counts `(n, 31)` summing to 24,059; last times equal the 14 boundaries; walls `4,296.9 s` and `12,990.1 s` within `0.1 s`; stored continuity `<= 1e-12`; parity gate | all met except the parity gate | walls `4,296.889898` and `12,990.087148`; stored continuity at most `1.7763568394002505e-15` (report table, lines 165–178) |
| stop (lines 364–370): parity failure; report the two decision records if a decision flips | met and honoured | the full records are preserved |
| C20 (lines 1156–1198), gate (lines 1199–1211) | yes | tolerances untouched |
| C16 baseline before source change (line 1070), C10 bound (line 1015) | yes | — |
| increment 6's per-variant parity gate (line 651 onward) | not reached; correctly so | — |

**Handoff defects the review owns.** My sample `parity_check.py` (lines
316–331) printed booleans but never failed; the addendum corrected this.
My C20 gate encoded a prediction the record had already contradicted:
that a correct seam implies `1e-12` parity across post-resume
re-selections (section 3). The gate correctly blocked; the prediction was
wrong. The lane's clarification that `12,990.1 s` is cumulative (report
line 120) agrees with the handoff's increment 5, which compares
cumulative walls (line 628).

## 5. Qualifications against the handoff's evidence standards

The standards are handoff lines 771–783. The report meets them, and
exceeds them in three places:
- it writes tests for the driver and the gate, which the handoff did not
  require for artifact drivers;
- its chronology hashes its inputs;
- it keeps reproduction separate from causation (187, 212).

The report says the 552-test suite does not certify the numerical gate
(64, 210). It says the cost and `T_r` are undetermined rather than
estimating them (1–3, 96, 126).

## 6. Defects and unsupported statements

1. **The checker does not encode the addendum's two tolerance tiers**
   (`drivers/parity_check.py:113-124`). Every coordinate difference above
   `1e-12` is a failure. There is no "unresolved" label for differences
   between `1e-12` and `1e-10`, and no "hard stop" label above `1e-10`.
   This is conservative: no false acceptance is possible. The report
   states the hard-stop classification in prose (line 62). Minor.
2. **The checker does not itself check that the resume index is a
   re-selection boundary** (`parity_check.py:62-65`). The driver enforces
   it, and `test_checkpoint_is_accepted_reselection_boundary` tests it.
   Minor.
3. **A merge consideration, not a defect of the lane.** The new suite
   tests `paper_5/tests/test_repair_parity_gate.py:14` (and the resume
   tests) load a driver from `agent_guidance/paper-v/handoffs/…-data/drivers/`.
   If these tests are merged into a shared branch, the package suite would
   depend on a handoff data path. Disposition belongs to whoever merges.
4. **Report line 47** records the latest `gitnexus` CLI failing on a
   missing `libssl.3.dylib` before the installed CLI succeeded. This is
   environment state outside the lane and is correctly disclosed. No
   action.

I found no unsupported scientific statement.

## 7. What the failed gate does and does not establish

**It establishes.** Under the default HF task-11 repair (27 keys, cutoff
16, `lambda_ep = 1.5`, `gamma = 0.5`, drive 1, SSPRK(3,3) at `h = 0.0025`,
Dykstra 120, re-selection every 16 steps at margin `1e-3`, `drive_power`
reference), the fixed driver resumes bitwise. A two-segment run spliced
at `t = 0.08` then differs from the unbroken run by up to `1.94e-4` over
`t <= 0.2`. The departure follows a different decision at the
uncertified re-selection near `t = 0.12`.

Every handover map the lane would have computed depends on chained
repaired segments: the stored chain itself and the increment-6 variants.
None of them can be validated against an unbroken rollout at this
setting under the strict gate.

**It does not establish:**
- anything about the bare continuation from repaired states, since no
  handover was run;
- the cause of the pre-selection differences;
- that the clock or the optimizer is the cause;
- that the stored `t = 0` to `60` chain is wrong. That chain is one
  realization whose segment boundaries are perturbations of exactly this
  kind, as CR section 7 already says.

No tolerance is waived here, and no causation is inferred.

**One thing the failure does bear on: the data source the handover
design assumed.** The design took stored repaired states as free
handover inputs (handoff line 108). Those states remain readable and
locked. A handover result from them would still be a statement about one
chained realization. It could not be validated against an unbroken
repaired rollout until a corrected prerequisite passes.

**What a corrected prerequisite acceptance would have to show.**
1. **The already selected gate passes, unchanged.** The executable gate
   passes at the task-11 setting:
   - seam bitwise in 56/70/31;
   - grid within `1e-12`;
   - all 81 samples within `1e-12` through `t = 0.2`, across the
     re-selections at `0.12` and `0.16`;
   - then the gate passes again at each increment-6 variant's own
     setting before its chain.
2. **The carried-reference receipt passes.** The reference is within
   `1e-14` at the seam. This is a distinct obligation: re-selection
   itself takes neither the clock nor the carried energy reference as an
   input.
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
  option with a bitwise default-path test, made in the shared module's single owner (the cone lane, for
  `cheap_projection_route.py`) and, released by
  root.

## 8. Holds and evidence status

Every hold remains binding:
- root messages 014, 015, 016 (`repair-parity-crosslane-20261010-016`),
  017 and 019;
- cone increment 3, held notwithstanding the conditional release 012;
- root hold 025;
- the three strict parity gates (regularized, repair-early, cone), each
  failed by `1.9395497882168655e-4`.

The lane's increments 1 to 6 remain not run and held: handover sweeps,
resolution, dense maps, long windows, the cost reading and the cheaper
variants. This review adopts no evidence and releases nothing. All
outputs remain `exploratory_local_not_promoted`.

**Consistency note (2026-10-18).** Root's focused wording correction
`review-regularized-scope-20261018-001` (SHA-256 `fe521731…c9`) was
addressed to the regularized review. Its section-7 concern applies
equally here, so the same text-only correction was applied to section 7
and to the cross-lane row of section 1. Nothing else changed.
