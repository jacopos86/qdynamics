# Paper V: route state and plan (2026-09-14)

Status: plan of record for Paper V. Supersedes nothing; the worklog
`paper_5/notes/electron_phonon_closure_worklog_20260803.md` and the working
notes `MATH/paper_facing/paper_V_high_u_gkba/stability_quantum_workflow_working_notes.md`
remain the evidence record. Line references below point into those two files
(`worklog:` and `notes:`) and into `paper_5/src/paper5/stability/`.

## 0. Overnight outcome (2026-09-14 22:30 to 2026-09-15 09:00, branch `claude/paper-v-routes-20260914`)

Three rounds of named experiments, every result in the worklog blocks headed
"2026-09-14: overnight loop" onward and in `agent_guidance/paper-v/overnight/`.
Defaults unchanged; `paper_5` suite green at every commit (301 to 379 tests).
Costs are on the replacement M4 Pro; the exact cutoff-16 reference propagates
to `t=4` in `0.7 s`, which is the scale every moment route is measured against.
Conventions: carried-witness accuracy figures are the offset-subtracted
("dynamic") all-coordinate scalar RMS (readout offset `3.24e-4` at `t=0`);
projective figures have offset `8e-16`; see
`agent_guidance/paper-v/overnight/conventions-audit-20260915.md`.  Over `t=0`
to `0.05` the Eq. (112) archive EOM alone scores `5.46e-7`, better than every
moment route, at no cost; the three carried-witness profiles agree to seven
figures there, so that horizon does not discriminate the guard.

| route | state after the night | cost | verdict |
|---|---|---|---|
| projective fixed-dictionary APCM with the event-triggered outer guard (section 3.4, plan steps 10 and 13) | reaches `t=4` at 31-coordinate RMS `0.026049` against exact (raw archive EOM `0.11090`), retained joint Gram positive, no trigger ever fires; the completion retry ladder alone gives `1.2277e-5` over `t=0` to `0.05` (N block `3.1e-8`, `112 s`), and the carried lift with the face carried across stages gives `1.2875e-5` (N block `1.8e-5`, `14 s`); the two are not combined in any run; every variant fails between `t=5.085` and `t=5.235` on a degenerate moment extension after the drive ends | `3,230 s` to `t=4` (stage retraction is the cost); `14 s` to `t=0.05` with the carried lift; internal-energy RMS over `t=0` to `4` `3.62e-2`, worse than the plain archive EOM (`2.76e-2`) and the unguarded carried witness (`1.39e-2`) | the cheap route; the post-pulse degeneracy is the open problem |
| one-shell outer selector (section 3.4) | `NumericalError` at the preparation and at a forced-trigger stage is removed by solver settings (refinement passes at `1e-14`, equilibration off): accepted at gap `7.46e-7` and `7.55e-7` against the unchanged `1e-6` acceptance | about `75 s` per solve, `3.9 GB` | the event guard can now fire when it needs to; untested inside a rollout |
| carried-witness flow, bounded-depth event-triggered guard, balanced, hard floor `2e-4` (section 3.5, plan steps 4 to 6) | completes `t=0.5` where the record stopped at `t=0.2`; RMS `2.45e-5` over `t=0.1175` to `0.5`; carried Gram never below the floor | `14,722 s` for `0.38` time units, 99.4 percent Clarabel at bundle rank 48 to 61, `6.05 GB` peak | correct but not fast; CHTC-sized, not laptop-sized |
| conic solve itself (plan step 7 and the round-2 replay bench) | 22 solver candidates on 36 recorded instances: none faster under the strict acceptance; `max_threads_8` `1.51x` on the strict `t=0` to `0.01` rollout, qualified against the balanced acceptance only; seeding the bundle by the predictor spectrum does not accept; carrying the accepted working set is `1.46x` faster in strict `t=0` to `0.01` (`2.04x` with threads) and `3.42x` slower in balanced `t=0` to `0.05` | --- | no lever left inside one solve |
| 31-coordinate structured cone correction (section 3.2, plan steps 8 and 9) | ten-coordinate energy-neutral barrier to `t=79.95`: containment only (all-31 scaled RMS `0.7008` against `0.7137` uncorrected; packet route `0.0636`); the record's electronic-band suppression is the joint-Gram controller's, not the closure's; the joint-Gram barrier reaches `t=0.30` in `150 min` even after the 24 and 35 percent speedups | `4.6 ms` per right-hand side (ten-coordinate); joint-Gram out of reach | diagnostic, not a route |
| cheap projection route with periodic lift re-selection (new on 2026-09-15; `cheap_projection_route.py`) | fixed-dictionary state, the declared proximal log-det completion re-selected every N steps and a 30-iteration Dykstra projection onto the affine positive-extension and retained-Gram sets between: N=16 scores `0.0268` over `t=0` to `4` and `0.0577` over `t=0` to `8` with no post-pulse wall; Dykstra alone (frozen lift) `0.131` at `t=4` and `1.42` at `t=20`; the Dykstra-only repair on the carried-witness flow reaches `t=4` at `0.184` but worsens the internal-energy RMS 51-fold | `90 s` to `t=4`, `196 s` to `t=8`, `486 s` to `t=20` (N=16) | the working route as of 2026-09-15 afternoon; at `t=20` the plain hybrid is not a closure (N=4 `0.2596`, N=16 `0.5793` against the both-cones `0.128607`; the error is the phonon-occupation block) while the energy-neutral Dykstra repair (Eq. (22) energy held to its drive-end value, or advanced by the exact drive power) brings N=16 to `0.239` with internal-energy RMS `0.069` and the post-pulse phonon-number excursion from `7.0` to `0.81` against exact `0.72`; N=4 energy-neutral to `t=20` is worse than unconstrained (`0.284` to `0.318`) because 58 to 64 percent of re-selections refuse; the 27-key dictionary with N=16 and the energy-neutral repair reaches `0.0114` at `t=4` and `0.0852` (`drive_power`) or `0.0699` (`post_pulse`) at `t=20` against the both-cones `0.1286`, but with internal-energy RMS `0.0063` / `0.165` against `0.00094`, the retained cone violated by `2.5e-3` / `3.1e-3`, and the site-0 occupation (`2 rho[0,0]` convention) `0.225` against `0.205`; sensitivity measured 2026-09-16: one declared solver setting moves the 15-key chain's scores by 13 to 26 percent and the 27-key chain's `t=4` score by 13 percent (the lift re-selection amplifies round-off through refusal flips; the exact flow amplifies the same perturbation by at most 1.16), so the 27-key chain ties cell A at `t=4` and leads by factors 1.57 at `t=8` and 1.40 at `t=12` within the measured spread, while every ordering among rows closer than 1.3x is unresolved; the tightened member carried to `t=20` (2026-09-16, task 6) puts the 27-key `drive_power` row at `0.064` to `0.067` over `t=0` to `16` (band 5.4 percent, containing the `post_pulse` chain's `0.0639`: a tie) and `0.085` to `0.091` over `t=0` to `20` (band 6.7 percent), so the `post_pulse` chain's `0.0699` at `t=20` is ahead by 1.22 to 1.30, outside the measured band; the `post_pulse` chain's own band (2026-09-16, task 7; its driven interval rerun under the lever, which enters its 100 re-selections) is `0.0699` to `0.0756` at `t=20` (8.2 percent), disjoint from the `drive_power` band by 12.6 percent, so at `t=20` the `post_pulse` chain is ahead by 1.13 to 1.30 under every pairing of the four members and is the best moment-closure row on record there on the all-coordinate score; its band is `0.0163` to `0.0197` at `t=4` (also task 3's 27-key unconstrained number's sensitivity, 20.9 percent), `0.0416` to `0.0454` at `t=8` (containing the 15-key `post_pulse` `0.0425`: a tie), `0.0461` to `0.0608` at `t=12` (24.1 percent, the tightened member better; the `drive_power` lead there is 1.06 on the closest members) and `0.0539` to `0.0639` at `t=16` (overlapping the `drive_power` band: no ordering); the lever's effect on this chain has no sign (`+20.9`, `-8.3`, `-24.1`, `-15.6`, `+8.2` percent at `t=4, 8, 12, 16, 20`); on internal energy the `drive_power` chain stays 26 to 36 times better under any pairing and neither chain holds a cone under either setting; the continuous re-selection rule (measured-candidate acceptance, margin `1e-3`, no anchor) removes the amplifier through `t=12` at 15 keys: one-setting twins spread `0.1`, `0.2`, `3.2` percent at `t=4`, `8`, `12` with scores `0.0272`, `0.0385`, `0.0982` (recorded rule `0.0237` to `0.0280`, `0.0404` to `0.0456`, `0.108` to `0.131`), and with a declared trust radius `0.1` on the frontier move (ball form) the base scores `0.0989` at `t=12`, `0.132` at `t=16` and `0.133` at `t=20` against the both-cones `0.1286` and the recorded rule's `0.26` to `0.33`, with bands 5.5, 19.3 and 56.2 percent (three of four twins within 4.4, 14.3 and 23.0): the second digit is resolved through `t=12`, the first digit only beyond; the anchor-free rule on the 27-key `N=16` `drive_power` chain (2026-09-17, projective task 8; base and three one-setting twins to `t=8`, no trust radius) gives the first sensitivity-resolved 27-key numbers: `0.0100` over `t=0` to `4` (four members `9.96e-3` to `1.001e-2`, spread 0.5 percent), below cell A's `0.01258` by 1.26 and below the recorded chain's whole band `0.0114` to `0.0129`, and `0.0318` to `0.0322` over `t=0` to `8` (spread 1.4 percent), behind the recorded chain's band `0.0259` to `0.0271` by 1.17 to 1.24 (the loss is the phonon-occupation and A blocks), so at `t=8` the recorded chain under the earlier form stays the row; internal energy `0.0039` / `0.0048`, retained joint Gram `-1.3e-3`, extension `-1.4e-4` (tighter than the recorded `-2.2e-3` / `-1.9e-4`), walls `1,883` to `1,963 s` against `2,058 s`; two twins are one trajectory with the base to `3e-8` through `t=6.6`, where the base accepts a `0.96` facial-reduction iterate (the task-9 event at a smaller perturbation) at a cost of `1.1e-3` in the 31 coordinates; the chaining gate under the rule is exact to `2.5e-14` until a re-selection sits at the feasibility test's edge (`t=0.16`: `infeasible` against `accepted`, a `0.89` move, `d31` `3.3e-4` at `t=0.2`), so the bands include the boundary as a perturbation of the twins' kind; the ball trust radius `0.1` on the 27-key four (2026-09-17, projective task 9; base and three one-setting twins to `t=8`) binds only in `(0, 0.64]` (3 to 5 caps of 200, the first the `t=0.04` whole-face answer scaled from `0.345` to `0.1`), does not meet task 8's `t=6.6` move (which does not occur on the capped trajectories), and leaves `t=8` unchanged: four members `0.0314` to `0.0322` (spread 2.3 percent; anchor-free `0.0318` to `0.0322`), behind the recorded band by 1.16 to 1.24, while at `t=4` it costs 4 percent (`0.0100` to `0.0104`, spread 3.9 percent, disjoint from and behind the anchor-free `9.96e-3` to `1.001e-2`, still ahead of the recorded band and below cell A) and puts each twin on its own trajectory from `t<=0.36`; the ball base chained to `t=20` (one run) scores `0.0380` / `0.0482` / `0.0635` at `t=12` / `16` / `20` against the recorded `drive_power` bands `0.0425` to `0.0433` / `0.0638` to `0.0672` / `0.0852` to `0.0909` (ahead by 1.12 to 1.14, 1.32 to 1.39, 1.34 to 1.43) and the `post_pulse` bands `0.0461` to `0.0608` / `0.0539` to `0.0639` / `0.0699` to `0.0756` (ahead by 1.21 to 1.60, 1.12 to 1.32, 1.10 to 1.19), `0.494` of the both-cones `0.1286`, behind the recorded chain at `t=10` by 1.07 to 1.08 (the crossing is in `(10, 12]`), with internal-energy RMS `0.0056` / `0.0058` / `0.0059` (inside the `drive_power` band; both cones `0.00094`), retained joint Gram `-2.4e-3` and extension `-2.2e-4` (the recorded chain's `-2.5e-3` / `-2.0e-4`), the cap binding 11 times in `(8, 14.4]` on answers `0.13` to `0.79` and the last quarter accepting every completion; it is behind the recorded chain in the phonon-occupation block at every horizon past `t=4` (`0.047` against `0.032` at `t=20`) and ahead in rho and B; the three twins chained to `t=20` (2026-09-18) give the ball rule a band at every horizon: `0.0380` to `0.0391` at `t=12` (2.9 percent), `0.0459` to `0.0486` at `t=16` (5.9), `0.0579` to `0.0635` at `t=20` (9.7; the 15-key rule's 56), every member ahead of the recorded `drive_power` bands by 1.09 to 1.14, 1.31 to 1.46, 1.34 to 1.57 and of the `post_pulse` bands by 1.18 to 1.60, 1.11 to 1.39, 1.10 to 1.31 (the narrowest lead, the worst member over the `post_pulse` band at `t=20`, is 1.10), `0.45` to `0.49` of the both-cones `0.1286`, internal-energy RMS `0.0053` to `0.0059` at `t=20`, joint Gram `-1.1e-3` to `-2.4e-3`, extension `-9.8e-5` to `-2.2e-4`; the `dykstra60` twin is the best member past `t=12` (`0.0459` / `0.0579`) and the only one on which the cap acts past `t=8` (seven uncertified answers of `0.72` to `1.14` applied at `0.1`), the other two twins within 0.9 percent of the base; so the 27-key row is task 8's anchor-free `0.0100` at `t=4`, the recorded chain's `0.0259` at `t=8`, and the ball rule's `0.0380` / `0.0482` / `0.0635` at `t=12` / `16` / `20` (adoption pending), each with a four-member band; task 10 (2026-09-18) attributes the gain past `t=8`: the anchor-free rule without the radius, chained from its own `t=8` state to `t=20`, scores `0.0385` / `0.0485` / `0.0629` at `t=12` / `16` / `20` (ratios to the ball base 1.014 / 1.006 / 0.990; per window `(8,12]`, `(12,16]`, `(16,20]` 1.027 / 0.998 / 0.977), inside the ball rule's band at every horizon from `t=8`, so the gain over the recorded chain is the measured-candidate acceptance's and the radius moves six applied displacements of `0.105` to `0.228` and no number (it costs 4 percent at `t=4`; the 15-key 12 percent does not carry to 27 keys); its internal-energy RMS `0.0052` at `t=20` is the ball's `0.0059` scaled by the two declared references (`5.5e-3` against `6.2e-3` above the exact post-pulse energy), cones `-2.3e-3` / `-2.1e-4`; the ball base with Dykstra 120 from `t=8` scores `0.0383` / `0.0459` / `0.0578` (ratios 1.010 / 0.953 / 0.910; the `t=20` number the best on record, 0.14 percent below the `dykstra60` twin's `0.0579`), with the cones over `(8,20]` at `-4.4e-4` / `-4.2e-5` against the ball's `-2.4e-3` / `-2.2e-4` (5.4 and 5.3 times shallower, the retained cone still violated at every sample), the internal energy identical (pinned to the same reference), the phonon-occupation block worse (`0.054` against `0.047` at `t=20`), and the wall 11 percent longer (`5,394 s`; repair time 3.9 times); so the row's numbers and standing do not change, its rule may be stated as the anchor-free measured rule without the radius, and the Dykstra count is the setting that moves the score past `t=12` (30 / 60 / 120: `0.0635` / `0.0579` / `0.0578` at `t=20`) and the cones (`-2.4e-3` / `-1.1e-3` / `-4.4e-4`), the Dykstra-120 member being the `t=20` low edge if adopted; task 11 (2026-09-18) runs the two ingredients together from `t=0` --- the anchor-free measured rule without the radius with Dykstra 120, base to `t=20` and three one-setting twins (`energy_tolerance 1e-3`, Dykstra 240, re-selection `conic_tolerance 1e-10`) to `t=8` --- and changes the row's standing at `t=4` and `t=20` and nowhere else: the base scores `0.00929` / `0.0330` / `0.0406` / `0.0461` / `0.0560` at `t=4` / `8` / `12` / `16` / `20`, the best 27-key numbers on record at `t=4` (four-member band `0.00929` to `0.00995`, 7.1 percent, every member ahead of task 8's whole band by 1.001 to 1.078 --- the worst member tying task 8's best by 0.06 percent --- and `0.74` of cell A) and at `t=20` (one run, ahead of the whole ball band `0.0579` to `0.0635` by 1.03 to 1.13, of the Dykstra-120-from-`t=8` member's `0.0578` by 1.03, of the recorded `drive_power` band by 1.52 to 1.62 and the `post_pulse` band by 1.25 to 1.35, `0.436` of the both-cones `0.1286`), inside the ball band at `t=16` (a tie with its low edge `0.0459` at 1.005) but behind every 27-key chain at `t=8` (the recorded band by 1.22 to 1.27, task 8's and the ball's by 1.03 to 1.05) and behind the ball band at `t=10` and `12` (by 1.03 to 1.08); the deeper repair from `t=0` re-opens the sensitivity the rule had closed (the four spread 7.1 percent at `t=4` and 9.8 at `t=8`, `0.0301` to `0.0330`, against task 8's 0.5 and 1.4 at Dykstra 30: every member branches at the `t=0.08` re-selection, whose answer is `0.86` to `1.42` and whose decision is the member's), gives the shallowest cones of any chain measured from `t=0` (retained joint Gram `-4.5e-4`, extension `-3.9e-5` over `t<=20`, 5.1 and 5.4 times shallower than Dykstra 30, the retained cone still violated at all 4,800 samples past `t=8`; the Dykstra-240 twin `-5.5e-5` / `-7.6e-6` over `t<=8`), internal-energy RMS `0.0053` at `t=20` (its declared reference `5.6e-3` above the exact post-pulse energy, held to `1.8e-15`), site-0 occupation `0.136` at `t=20` in the record's `2 rho[0,0]` convention (both cones `0.205`; the `post_pulse` chain `0.125`), 484 accepted of 500 with 117 solver refusals and no cap, wall `5,184 s` (+8 percent over Dykstra 30; repairs 0.17 of the wall, 3.7 times); the gain is outside the phonon-occupation block (the 27 coordinates outside it `0.0427` and `0.0556` at `t=16` and `20` against the ball's `0.0477` and `0.0656`) and it does not touch the `(4, 8]` phonon-occupation loss: the four phonon-occupation coordinates score `0.0678` over `(4, 8]` (twins `0.0644` to `0.0662`; the same rule at Dykstra 30 `0.0669`; the ball `0.0681`) against the recorded chain's `0.0454`, all of it in `(6, 8]` (`0.0828` against `0.0336`; over `(4, 6]` the rule is ahead, `0.0484` against `0.0547`) and a bias (signed mean errors `-0.073`, `-0.074`, `+0.073` on three coordinates against RMS `0.078`, `0.079`, `0.078`), with every completion of the window accepted at applied moves below `0.015`; so the row is now `0.00929` (this configuration, band `0.00929` to `0.00995`) at `t=4`, `0.0259` (the recorded `drive_power` chain, band `0.0259` to `0.0271`) at `t=8`, `0.0380` and `0.0482` (the ball rule at Dykstra 30, bands `0.0380` to `0.0391` and `0.0459` to `0.0486`) at `t=12` and `16`, and `0.0560` (this configuration, one run, no band of its own) at `t=20`; next the two `t=4` to `8` reruns that locate the loss (task 8's rule from the recorded chain's `t=3.96` state; this base with `certified` candidates from its `t=3.92` state), then the three twins of this configuration from `t=8` to `t=20` for the `t=20` band |
| bitwise-neutral speedups | unguarded CWRMF atom `2x`; `cone_correction.py` `24 percent`; `matrix_reference.py` `35 percent`; all default-path, byte-identical | --- | kept |

Approved by the user on 2026-09-15 morning ("sure do that, i told you multiple routes should be attempted"), with the instruction that several routes stay in play at once and that work proceeds through persistent agents with short tasks (no run over 20 minutes) rather than multi-hour one-shot batches. Recommendation as put: make the projective
route with carried lift and carried face the working route and push it past
the post-pulse degeneracy using the repaired selector; treat the guarded
carried-witness flow as a CHTC job, not a local one; retire the joint-Gram
barrier as a long-horizon route on this machine.

## 1. Verdict

No plan has been executed since 2026-08-18, and no single document ordered
the open work across the routes until this file. What exists:

- the working notes' two ordered workstreams (stability first, quantum
  algorithm second, notes:19-31) and their forward protocols (reproduction
  protocol notes:1748-1793, required causal ablation notes:1393-1435,
  strange-attractor test notes:1928-1965, workflow boundary for the quantum
  algorithm notes:1976-1993, open questions notes:1995-2026);
- the manuscript's McLachlan pathway section (`paper_V_high_u_gkba.tex:172-326`),
  which is the projective route's theory: the projected-velocity argmin, the
  damped normal equations, the sector-gauge McLachlan equations, and a
  four-route split for evaluating the force projection;
- executable, schema-versioned plan documents emitted by the sensitivity and
  Lyapunov drivers (`plan.json`, schemas `paper_v_*_plan_v1`), each carrying
  its scientific question;
- the worklog's terminal next actions: the bounded-depth event-guard pilot at
  `event_gram_hard_floor = 2e-4` (worklog:3846-3854), stopped by the user for a
  thermal concern before its first trigger, and the CHTC lane
  `chtc/paper_v_cwrmf_event_guard_20260818`, never uploaded or submitted
  (worklog:3867-3871);
- the projective route's condition for proceeding (worklog:3321-3324):
  redesign the selector representation, or weaken the every-stage outer-guard
  requirement. Neither was done.

The manuscript `MATH/paper_details/paper_V_high_u_gkba.tex` calls itself a
planning template and still carries placeholder slots for the high-U failure
mode and the quantum encoding; its McLachlan pathway section is written. The last substantive Paper V commit on `main` is `ba7f2ac9`
(2026-08-18).

## 2. State on this machine (2026-09-14)

| item | state |
|---|---|
| `paper_5` test suite | 301 passed with `PYTHONPATH=paper_5/src` (recorded in `NEW-MACHINE-SETUP.md`); 300 passed, 1 failed without it (a subprocess could not import `paper5`) |
| conic stack | numpy 2.3.5, scipy 1.16.3, cvxpy 1.9.2, clarabel 0.11.1; SCS, OSQP present |
| `output/local_runs/paper_v_*` | absent; every trajectory, score, and plot artifact cited by the worklog was lost with the previous machine |
| `output/pdf/paper_v_results_progression_20260804.pdf` and its figures | absent; source `paper_5/notes/paper_v_results_progression_20260804.tex` present |
| CWRMF driver | `paper5.stability.apcm_carried_witness_analysis` prepares the canonical cutoff-16 correlated preparation in-process and regenerates the exact reference itself, so reruns do not depend on the lost artifacts |
| branch `docs/paper-v-state` | created 2026-09-01 with no commits; this file is its first content |

## 3. Route ledger

Every number is over the stated horizon at the stated configuration.
"Strong coupling" means `lambda = 1.5`, `gamma = 0.5`, drive amplitude 1,
phonon cutoff 16, exact-correlated preparation, unless stated otherwise.

### 3.1 Archive moment EOM repair (31 real coordinates for rho, B, N, A, C)

Retired as the primary propagator by the 2026-08-04 abandonment decision
(worklog:892-936) on the strength of the finite-cutoff non-identifiability
witness: at phonon cutoff 3, at the single time point t = 0, two constructed
spin-exchange-symmetric density operators agree in the retained
31-coordinate tuple to a maximum difference of 3.25e-19 and in the combined
exact (rho, B, N, A) velocity to 1.75e-18 in Euclidean norm, while their exact
C velocities differ by 2.90343e-2 in Euclidean norm (worklog:356-373). The
record's own scope statement is that this rules out "a universal exact
deterministic velocity depending only on (rho, B, N, A, C) on any state class
containing the witness pair" (worklog:897-899); it is not a proof that no
archive-compatible closure exists. Best repaired
result before closure: the same-spin Pauli-repaired archive EOM gave
31-coordinate RMS error 0.057742 against the exact cutoff-16 trajectory over
t = 0 to 4, single pulse, dt = 0.01, versus 0.11066 raw (worklog:1835-1842).
Every later autonomous finite-memory variant was gated and rejected
(worklog:1835-2044). Declared endpoint: a further attempt "would have to
change the model class to nonlinear, gauge-invariant compression of the
adaptive packet state" (worklog:2471-2475). Not reopened in this plan.

### 3.2 Structured cone correction on the 31-coordinate archive EOM (`cone_correction.py`)

Works as containment; not a closure repair.

| quantity | value | horizon and configuration | source |
|---|---:|---|---|
| complete 31-coordinate l2 time-RMS distance from the exact cutoff-16 reference, raw then corrected | 0.585740 then 0.263960 | t = 0 to 100, weak point `lambda_ep = 0.5`, `gamma = 0.5`, double pulse, RK4 h = 0.01 | worklog:1430-1442 |
| minimum bosonic eigenvalue, raw then energy-neutral matrix barrier | -3.05033e-1 then 2.89314e-5 | t = 0 to 20, strong coupling, DOP853 rtol 1e-9 atol 1e-11 | notes:1850-1859 |
| maximum post-pulse drift from E(4): raw, full barrier, energy-neutral barrier | 2.41557e-6, 6.54778e-2, 2.53904e-6 | same run | notes:1850-1859 |
| fraction of sampled times with the barrier active, unconstrained then energy-neutral | 36.16 percent then 55.11 percent | same run | notes:1861-1866 |
| Hellinger distance of the polarization spectrum from exact, raw then corrected | 0.6381 then 0.1782 | 10 <= t <= 100, `g/t_hop = 0.6124` | worklog:2934-2941 |
| electronic-band weight of that spectrum: exact, raw, corrected | 0.0569, 0.5225, 0.0026 | same window | worklog:2934-2941 |

Declared next (notes:1868-1872), not executed: extend the energy-neutral
barrier through the late amplitude-validation horizon and compare selected
observables with exact driven-dimer propagation.

Solver facts (code): custom constraint generation plus scipy SLSQP at four
call sites (`cone_correction.py:465, 637, 1369, 1470`); no cvxpy; the
correction is solved once per right-hand-side evaluation inside the ODE
closures (`cone_correction.py:1911, 1968, 2022`), so DOP853 pays it several
times per step and again on rejected steps; the 31-fold joint-Gram basis is
rebuilt per call (`1279-1288`); no caching, warm start, or timing anywhere.

### 3.3 Fixed-dictionary APCM with the positive degree-four completion (60 coordinates, both cones)

| quantity | value | horizon and configuration | source |
|---|---:|---|---|
| all-coordinate scalar RMS and autonomous wall time: A full cone + controller; B zero-cumulant prior + controller; D positive M4, controller off | 0.012580 / 3399.6 s; 0.046475 / 52.9 s; 0.0125814 / 3268.6 s | t = 0 to 4, SSPRK(3,3) h = 0.0025, identical stored initial 60-vector | worklog:2608-2664 |
| 31-coordinate scalar RMS against exact, both cones then uncorrected archive EOM | 0.128607 then 0.193205 | t = 0 to 20, h = 0.0025, 5832.34 s, hidden retraction on 5998 of 8000 steps | worklog:2781-2792, 2898-2903 |
| internal-energy RMS, both cones then uncorrected | 0.000939628 then 0.0182361 | same run | worklog:2781-2792 |

The accuracy comes from the positive degree-four completion, not from the
retained joint-Gram controller (cell D versus A), at a roughly 64-fold
runtime cost over cell B. The spin-exchange block-diagonal accelerator ran
4.925x faster on t = 0 to 0.05 but failed solver-level parity (maximum
60-coordinate state difference 0.245845 over t = 0 to 4) and was not promoted
(worklog:2723-2762). The t = 240 continuation died at the t = 31 checkpoint
as a process-lifecycle failure (worklog:2831-2837).

### 3.4 Projective guard (URPG) selector, pre-rollout form (`apcm_projective_guard.py`)

The memoryless selector stage of the "projective method directly on the
moments". The selector minimizes the invariant target image on the one-shell
outer cone and then restricts one common feasible witness to the current
variables; a separate minimum-norm lift under the equality `R_D y = q*` is the
declared canonical tie-break. This selector has never run per stage; the
amendment's dynamically executable form is the carried-witness flow of
section 3.5, which has propagated (worklog:3331-3334).

| quantity | value | horizon and configuration | source |
|---|---:|---|---|
| one outer-aware target solve at the cutoff-16 correlated preparation | about 500 s, then 171 s after the gate reorder; objective 0.01126712315 unchanged | t = 0, one evaluation | worklog:3267-3269, 3282-3288 |
| four-way relative outer audit at the same preparation | 820 s, about 2.7 GiB peak resident | t = 0, one evaluation | worklog:3229-3233 |
| minimum-norm lift at the same preparation | `NumericalError`; witness recorded as noncanonical `outer_target_stage` | t = 0 | worklog:3253-3255; `apcm_projective_guard.py:2478, 2489-2491` |
| frozen-face selector | 117 s, `PrimalInfeasible`, minimum scaled eigenvalue -0.04746; face-map singular values from order 1 to 1e-20 | t = 0 | worklog:3290-3299 |
| PSD cutting-plane QP | still indefinite after 512 cuts on the correlated preparation | t = 0 | worklog:3301-3309 |
| warm-started CVXPY/SCS at eps = 1e-5 | 80.1 s, 16,550 iterations, reconstructed Gram minimum scaled eigenvalue -3.15e-5 while reporting optimal | t = 0 | worklog:3311-3316 |

No per-stage rollout of this selector has run. The fixed-dictionary
integrator exists (`projected_apcm.py:745`, containment before every RHS
call) and one SSPRK step at h = 0.0025 with the earlier nested selector and no
retraction cost about 7.8 s and 331 MB (worklog:3076-3078). At 171 s per
outer solve and three stages per step, an every-stage outer guard is about
513 s per step, which the record calls "not an executable controller"
(worklog:3267-3269). The audit recorded four attempts, not three: the gate
reorder was kept, but it is the keyword `accept_first_provisional=False` on
`_solve_explicit_target_qp` (`apcm_projective_guard.py:1152`) and on
`select_outer_feasible_projective_guard` (`2422`), and no caller sets it
True; the frozen-face selector is the only one the record marks "rejected";
the cutting plane "remains a guarded diagnostic and is not connected to
propagation"; SCS "fails both the runtime and physicality requirements".

### 3.5 Carried-witness radial moment flow, CWRMF (`apcm_carried_witness.py`)

The executable form of the 2026-08-06 projective-guard amendment
(worklog:3331-3334): 44 retained archive-plus-entrance coordinates plus 340
completion coordinates carrying one affine 62-row unified Gram; SSPRK2; the
guard is a conic repair on that Gram at each radial atom. This is where the
projective route and the physicality-cone correction meet, and it is the
route whose cone solve must be made faster.

| quantity | value | horizon and configuration | source |
|---|---:|---|---|
| autonomous solver time, strict always-on guard then no guard (the no-guard mode removes the 62-row PSD optimization and the endpoint rejection together; in the strict profile only the adaptive bundle tier runs, `enable_full_cone_fallback=False`) | 43,692 s (51 scored chunks, peak child RSS 3.27 GB) then 18.93 s | t = 0 to 0.5, h = 0.0025, strong coupling | worklog:3495-3498, 3597; `apcm_carried_witness.py:94` |
| coordinate RMS difference between the no-guard and strict retained trajectories | 6.57e-7 | same interval | worklog:3599-3600 |
| dynamic all-coordinate scalar RMS against the exact cutoff-16 reference (fixed canonical-embedding offset subtracted), strict CWRMF then matched raw 31-coordinate archive EOM | 3.19e-5 then 2.13e-2 | t = 0 to 0.5 | worklog:3484-3493 |
| balanced mixed-step profile (h = 0.01 to t = 0.4, then 0.005): time and all-coordinate scalar RMS | 17,561 s; 3.56e-5 | t = 0 to 0.5 | worklog:3523-3536 |
| minimum extended 62-row Gram eigenvalue without the guard | -1.27089e-2 | t = 0 to 0.5 | worklog:3590-3598 |
| no guard, full driven interval: time; 31-coordinate RMS error, CWRMF then raw archive EOM; interaction-energy RMS, CWRMF then raw | 153.53 s; 0.48049 then 0.11090; 0.01388 then 0.02759 | t = 0 to 4, h = 0.0025 | worklog:3620-3633 |
| minimum extended 62-row Gram and retained joint-Gram eigenvalues without the guard | -26.4455 and -17.6151 | t = 0 to 4 | worklog:3632-3633 |
| first time the retained joint Gram falls below -1e-8 without the guard | t = 1.355 | t = 0 to 4 | worklog:3645-3651 |
| active spectral bundle rank | reached 57 of the 61 nontrivial directions, mean 50.15, read by the record as evidence that the earlier 8- and 16-mode caps were artificial; the balanced coarse phase eventually required all 61 critical modes, "so the remaining cost is structural rather than solely a consequence of excessive solver precision" | t = 0.1 to 0.5; balanced phase h = 0.01 to t = 0.4 | worklog:3475-3477, 3537-3539 |
| event-triggered guard, strict numerics | t = 0.05 in 6.43 s at all-coordinate scalar RMS 3.236e-6 (always-on 3.24e-6); stopped at t = 0.08 after 933.6 s when a triggered repair returned independent residual 2.622e-9 against acceptance 1e-9 | t = 0 to 0.08 | worklog:3796-3816 |
| event-triggered guard, balanced numerics | passed the first triggered conic repair near t = 0.0825 (maximum readable-rate residual 6.34e-2), then stopped at t = 0.2 after 2,184 s with `NumericalError` in both the bundle and full-cone solves; the record's reading: "The conic repair is therefore depth-limited: it succeeded from -2.3e-4 and failed from -5.9e-4" (unshifted endpoint 62-row Gram minimum). Both triggers fired through the negative-mode coupling measure exceeding 1e-2, since neither depth crossed the default hard floor 1e-3 (`apcm_carried_witness.py:82-83, 2621, 2643`); the 2e-4 override makes depth the trigger | t = 0 to 0.2 | worklog:3821-3830 |
| minimum-norm completion retraction | -3.3e-6 restored to -8.2e-9 in 30 iterations, 0.5 s, no conic backend; stalls from -4e-5 and deeper | one triggered atom | worklog:3841-3844 |

Code facts: one `perf_counter` per atom is the only instrumentation
(`apcm_carried_witness.py:2729, 2961, 3264-3266`); `_clarabel_svec_upper` is a
pure-Python double loop called 340 times per bundle outer iteration at 7,503
packed floats each at rank 61 (`apcm_positive_extension.py:177-189`); the
step-scaled completion coefficient tensor, a (340, 62, 62) complex array of
about 20.9 MB, is rebuilt on every solver-tier entry
(`apcm_carried_witness.py:1090, 1933, 2160, 2352`); the retraction and the
coupling measure stream that tensor once per working mode (`2559-2568`,
`3040-3052`); every Clarabel call is pinned to `max_threads = 1` (`958, 1269,
1826, 2021, 2416`); `integrate_cwrmf_ssprk2` terminates the trajectory on a
failed atom instead of halving the step (`3180-3196`), although the record's
operating rule is local step halving (worklog:3539-3541).

State change, 2026-09-15 (worklog entry "CWRMF checkpoint resume completed",
report `agent_guidance/paper-v/overnight/cwrmf-chtc-20260915b.md`): a
checkpoint resume is now exact under the carried seed policies.
`checkpoint.npz` carries `bundle_accepted_rank` always and the accepted 62-row
working set whenever the active seed policy reads it, and a resumed rollout
under `carried_accepted_basis` with rank margin 2 reproduces the unbroken
strict `t = 0` to `0.005` rollout at `h = 0.0025` exactly, where the previous
rank-only fallback left `2.026859e-08` on the 384 carried coordinates.  The
bounded-segment CHTC lane for step 6 is built and packaged as
`chtc/paper_v_cwrmf_event_guard_accel_20260915/` (eight segments of `0.5` time
units to `t = 4`, balanced, hard floor `2e-4`, subdivision depth 2, 8 CPUs,
12 GB); it is not uploaded and not submitted.  The round-3 pair
`carried_accepted_basis` with rank margin 2 is **not** in that lane's submit
files: it is a strict-profile acceleration and a measured `3.42x` regression
on the balanced profile the lane runs.

### 3.6 Context: multi-coherent adaptive packet route

The only representation that propagates stably to long horizon: all-31
scaled RMS against exact cutoff-16 over t = 0 to 98 of 0.063649 (ordinary
K = 6 packet 0.140791), minimum exact-state fidelity 0.923677, joint-Gram
minimum -1.72e-15 (worklog:2503-2523). It is not a moment closure and carries
no online cone correction; it is the comparison line every moment route must
be reported against.

## 4. Plan

Order is by dependency. Each step names its gate. All heavy runs go through
`pipelines/shell/ram_guard.py` under the 18 GB aggregate contract; recorded
peaks are 3.27 GB (CWRMF strict) and about 2.7 GiB (URPG audit).

### 4.1 Re-anchor (both routes depend on this)

1. Rerun the no-guard CWRMF over t = 0 to 0.5 at h = 0.0025 (recorded
   18.93 s) and a strict always-on prefix over t = 0 to 0.05 (about 20 steps;
   the recorded strict cost is 43,692 s over 200 steps). These are the parity
   anchors every acceleration is checked against; both regenerate the exact
   reference in-process.

   ```bash
   cd <checkout> && PYTHONPATH=paper_5/src /Users/lordskyler/Holstein_test/.venv/bin/python pipelines/shell/ram_guard.py --limit-mb 6000 -- /Users/lordskyler/Holstein_test/.venv/bin/python -m paper5.stability.apcm_carried_witness_analysis --output-directory output/local_runs/paper_v_cwrmf_no_gram_guard_t05_h0025_cutoff16_20260914_v1 --final-time 0.5 --time-step 0.0025 --no-gram-guard
   ```

   ```bash
   cd <checkout> && PYTHONPATH=paper_5/src /Users/lordskyler/Holstein_test/.venv/bin/python pipelines/shell/ram_guard.py --limit-mb 6000 -- /Users/lordskyler/Holstein_test/.venv/bin/python -m paper5.stability.apcm_carried_witness_analysis --output-directory output/local_runs/paper_v_cwrmf_strict_t005_h0025_cutoff16_20260914_v1 --final-time 0.05 --time-step 0.0025 --numerical-profile strict
   ```

   Gate: no-guard all-coordinate scalar RMS reproduces 3.19169e-5 over t = 0
   to 0.5 and the strict prefix reproduces 3.24e-6 over t = 0 to 0.05.

### 4.2 Cone correction: make the CWRMF guard fast

2. Instrument below the radial atom: per-tier `perf_counter` accumulators
   for the 62-row eigendecomposition, coefficient assembly and svec packing,
   the Clarabel solve, and the retraction, reported next to
   `maximum_atom_seconds`. Gate: the strict prefix rerun attributes its wall
   time by tier. Without this the ranking below is by code reading.
3. Land the bitwise-neutral accelerations, each gated on reproducing the
   strict t = 0.05 prefix to roundoff and on the 20 carried-witness tests:
   vectorize `_clarabel_svec_upper`; cache the step-scaled completion
   coefficient tensor once per rollout; batch the per-mode contractions in
   the retraction and the coupling measure; precompile the time-independent
   commutator structure in `desired_completion_velocity` and
   `_moment_derivative` (`861-922`, `308-327`, note at `604-605`). The last
   item also speeds the unguarded atom, which is the event guard's steady
   state (about 0.047 s per atom from 18.93 s over 400 atoms).
4. Add local step halving to `integrate_cwrmf_ssprk2` on a failed atom or
   unresolved convex node, bounded by a declared subdivision depth
   (`projected_apcm.py` already carries `maximum_subdivisions`). This removes
   the human restart that ended three recorded runs.
5. Run the bounded-depth event-triggered guard the record prescribed and
   never completed: balanced profile, `--event-gram-hard-floor 2e-4`, t = 0 to
   0.5, h = 0.0025, chunked through `checkpoint.npz`.

   ```bash
   cd <checkout> && PYTHONPATH=paper_5/src /Users/lordskyler/Holstein_test/.venv/bin/python pipelines/shell/ram_guard.py --limit-mb 6000 -- /Users/lordskyler/Holstein_test/.venv/bin/python -m paper5.stability.apcm_carried_witness_analysis --output-directory output/local_runs/paper_v_cwrmf_event_guard_balanced_hardfloor2em4_t05_h0025_cutoff16_20260914_v1 --final-time 0.5 --time-step 0.0025 --numerical-profile balanced --event-triggered-guard --event-gram-hard-floor 2e-4
   ```

   Gate: reaches t = 0.5 with minimum shifted 62-row Gram at or above the
   declared floor and all-coordinate scalar RMS within a stated factor of the
   strict 3.19e-5; report wall time against 18.93 s and 43,692 s.
6. Decisive cone result: the same guard over the full driven interval t = 0
   to 4, scored against the regenerated exact cutoff-16 reference and placed
   next to the two existing t = 4 lines (raw archive EOM 0.11090; unguarded
   CWRMF 0.48049 on the 31-coordinate RMS). This is the number Paper V does
   not have: a guarded moment closure through the pulse that beats the raw
   archive EOM at tolerable cost. Run on CHTC through the existing lane once
   step 5 has validated the depth bound locally; the lane's argument contract
   is `run_id final_time time_step event_gram_hard_floor`, and its README
   gives the t = 4 form. Submission needs the user's Duo login.
7. Only if steps 5 and 6 show the repair count, not the repair cost,
   dominating: raise Clarabel `max_threads` for local runs, gated on
   reproducing an accepted atom's candidate velocity, because multithreaded
   factorization is not bitwise-neutral and the 2026-08-05 four-thread
   accelerator failed parity.

### 4.3 Cone correction: the 31-coordinate structured barrier

8. Memoize the structured correction on the exact (state, derivative) pair
   inside the RHS closures and hoist the per-call invariants (31-fold basis
   rebuild, identity-metric Cholesky, equality SVD). Replace the SLSQP
   projection in `_project_origin_onto_halfspaces` with a warm-started
   active-set minimum-norm-point solve; keep the analytic single-cut path.
   Gate: the cross-backend norm-agreement test at rtol 1e-8 and bitwise
   reproduction of one pinned t = 20 energy-neutral run.
9. Execute the declared next calculation (notes:1868-1872): energy-neutral
   barrier through t = 100 at strong coupling with observables scored against
   exact driven-dimer propagation, reported next to the packet route's t = 98
   line. This settles whether the barrier's electronic-band suppression
   (0.0026 against exact 0.0569 over 10 <= t <= 100) is intrinsic to the
   31-coordinate closure or a controller artifact.

### 4.4 Projective route: make the one-shell selector executable

10. Make the outer guard event-triggered, reusing the CWRMF machinery
    verbatim: unconstrained fixed-dictionary predictor, endpoint Gram
    eigendecomposition, the shared `negative_mode_coupling` measure, and
    `select_outer_feasible_projective_guard` only on triggered stages. This
    is one of the record's two named ways forward. Needs decision D1.
11. Replace the two-solve selector (target image, then unique minimum-norm
    lift under `R_D y = q*`) by one strictly convex solve: the image quadratic
    in the whitened metric plus a fixed positive-definite frontier term,
    reported as a declared model parameter. The lift equality is exactly
    what returned `NumericalError`; a strictly convex objective needs no
    tie-break and removes the null lift space that defeated the cutting
    plane. Acceptance: the four-way relative outer audit at the cutoff-16
    preparation, compared against the recorded witness (objective
    0.01126712315, target-image displacement 0.5193246). Needs decision D2.
12. Repair the two conditioning entries: Tikhonov-damped or truncated face
    solve in place of the componentwise division by face singular values
    (`apcm_projective_guard.py:1762-1763`, rank floor 1e-12 at
    `1674-1684`), and carry the active face across stages by continuation
    with the existing consistency gate and fallback, instead of freezing it.
13. First projective rollout: fixed-dictionary APCM, SSPRK(3,3), h = 0.0025,
    t = 0 to 0.05, containment after every stage, event-triggered outer
    guard, scored offline against the regenerated exact reference and placed
    next to the CWRMF strict t = 0.05 figure (3.24e-6). Then t = 0 to 4 next
    to the step 6 line.

### 4.5 Reporting

14. Rebuild `output/pdf/paper_v_results_progression_20260804.pdf` from the
    regenerated artifacts through the existing plotting workflow; do not
    hand-edit numbers.
15. Fill the manuscript's two placeholder slots only from steps 6, 9, and
    13; adoption is the user's decision.

## 5. Decisions for the user

D1. Is the amendment's requirement that the invariant target image
`q_D = R_D y` be minimized on the one-shell outer cone at every SSPRK stage
still binding, or may the outer cone be enforced as an event-triggered guard
keyed on the endpoint 62-row Gram minimum and the negative-mode coupling
measure, as the carried-witness flow already does? Recommendation: weaken it
to event-triggered. The record names this as one of its two ways forward, and
its mode audit supports an event-triggered guard over always-on optimization
(worklog:3655-3663).

D2. Does the lexicographic selector (minimum invariant target image, then the
unique minimum-norm lift under `R_D y = q*`) remain the declared selector, or
may it be replaced by one strictly convex objective with a declared frontier
term? Recommendation: replace it; the lift equality is the part that failed.

D3. Is the strict profile (acceptance `10 x solver_tolerance = 1e-9`,
`psd_inflation 1e-11`) the reporting contract for the 62-row Gram cone, or is
the balanced profile (declared unshifted floor -1e-8, acceptance 1e-7)
acceptable as the reported cone with its floor stated in the caption?
Recommendation: balanced as the reported cone, strict kept only as the
parity anchor; the strict event-guard run stopped at t = 0.08 on a residual
of 2.622e-9 that balanced admits.

## 6. Verification ledger

Ten load-bearing claims from the synthesis were each checked by three
independent refuters (one against the worklog and notes, one against the
source, one against definitions and horizons). Every correction below is
already folded into sections 1 and 3.

| claim | verdict | correction applied |
|---|---|---|
| no written Paper V plan exists | refuted 3/3 | the notes' workstreams and protocols, the manuscript's McLachlan pathway section, and the drivers' `plan.json` schemas are plans; what was missing is an executed, cross-route order (section 1) |
| the projective route never propagated a trajectory | refuted 3/3 | the carried-witness flow is the amendment's executable form and propagated to t = 0.5; only the one-shell selector stage never ran per stage (sections 3.4, 3.5) |
| the cone solve is the entire CWRMF runtime | refuted 2/3 | the no-guard mode removes the 62-row PSD optimization and endpoint rejection together; strict runs only the bundle tier; the numbers stand (section 3.5) |
| the unguarded flow fails over the full pulse | stands 3/3 | precision added: the RMS is on the 31 retained coordinates; interaction-energy RMS is better without the guard (0.01388 versus 0.02759); the record calls it "not an immediate observable failure" |
| the conic repair is depth-limited at -2.3e-4 versus -5.9e-4 | refuted 2/3 | the sentence is the record's; both triggers fired through the coupling measure, not depth; the 2e-4 override is what makes depth the trigger (section 3.5) |
| the retraction cannot replace the conic repair | stands 3/3 | vocabulary: "minimum-norm completion retraction", a fast path for hysteresis-scale retriggers; single-atom measurement, not a trajectory |
| bundle compression is nearly exhausted | refuted 3/3 | the record says "structural rather than solely a consequence of excessive solver precision"; 57 of 61 was read as evidence against the earlier mode caps (section 3.5) |
| one outer solve costs 171 s and is unusable per stage | stands 3/3 | the gate reorder is `accept_first_provisional=False` by default with no caller setting it (section 3.4) |
| all three per-stage accelerations were rejected | refuted 2/3 | four attempts; the gate reorder was kept; only the frozen face is marked rejected; the others are "guarded diagnostic" and "fails both requirements" (section 3.4) |
| the archive-EOM repair route is closed by the witness | refuted 2/3 | the witness is at cutoff 3, t = 0; the record's scope is "any state class containing the witness pair"; the route was retired by decision, not closed by proof (section 3.1) |

## 7. Overnight autonomous protocol (2026-09-14, user-directed)

The user asked for unattended work on the routes above, "throw things at the
wall and see what sticks", constrained to Paper V. Rules for the loop:

- Checkout: `/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914`,
  branch `claude/paper-v-routes-20260914`. Every edit, run, and commit happens
  there. The main checkout is read-only for this work; other agents use it.
- Writable paths: `paper_5/**`, `agent_guidance/paper-v/**`,
  `MATH/paper_facing/paper_V_high_u_gkba/**`, `MATH/paper_details/paper_V_high_u_gkba.tex`,
  `chtc/paper_v_*/**`, `pipelines/open_dynamics/**` (Paper V pipelines only),
  and `output/local_runs/paper_v_*` (ignored by git). Nothing under Paper I--IV
  lanes, `src/`, `test/`, `pipelines/static_adapt`, `pipelines/time_dynamics`,
  or the shared contracts. A change outside the list is reverted, not kept.
- Python: `PYTHONPATH=paper_5/src` with the main checkout's `.venv`; every
  heavy process runs under `pipelines/shell/ram_guard.py --limit-mb 5000`; at
  most three heavy processes at once; aggregate under 18 GB.
- Defaults do not change. Every new behaviour is a named experiment behind a
  flag or profile with strict and balanced defaults untouched; the decisions
  D1--D3 are assumed in their recommended form for exploration only, and the
  record says so wherever a number depends on them.
- Gates before a change is kept: the 20 carried-witness tests, the full
  `paper_5` suite, and bitwise (or stated-tolerance) reproduction of the
  section 4.1 anchors for any change on the guarded path.
- Recording: one dated entry per completed experiment appended to
  `paper_5/notes/electron_phonon_closure_worklog_20260803.md` in the record's
  vocabulary, numbers with definition and horizon, artifact paths, and a
  kept/dropped verdict; section 3 of this file is updated in place when a
  route's state changes. Commits stage explicit paths only.
- Subagents and workflow agents run on Opus with `model: "opus"`.
- Stop conditions: a gate cannot be met after two independent attempts, the
  memory contract would be exceeded, or a step needs a user decision beyond
  D1--D3; then record the blocker and wait for the next wakeup.
- Never: submit to CHTC (needs the user's Duo login), promote evidence,
  rewrite the manuscript's claims, delete artifacts, or touch another
  agent's jobs.
