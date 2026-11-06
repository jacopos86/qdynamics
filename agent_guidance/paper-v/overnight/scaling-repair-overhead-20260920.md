# scaling, repair-overhead: what the corrected closure costs outside the
# conic solve, and how each piece grows

Round of 2026-09-20, worktree
`/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914`,
branch `claude/paper-v-routes-20260914`, HEAD `63852f96`.  Nothing here
propagates a closure, prepares a state, changes a default or edits a
module.  Every measurement is either a read of a recorded artifact or a
timing of the module's own operation at a synthetic operand of the same
shape.  Scripts and records are under
`output/local_runs/paper_v_scaling_20260920/`.

**The negative result first.**  The premise of the key is wrong: the cheap
route's repair is not the corrected closure's cost.  The Dykstra repair,
the unguarded velocity and everything else together carry `0.197` of the
task-11 chain's `5,183.6 s` wall to `t = 20`, and `0.077` of the wall of
the same rule at the Dykstra count the rest of the record uses.  `0.803`
of it is the declared proximal log-det completion the route re-selects
every 16 steps, and that re-selection *is* a conic solve --- 500 of them
over `t = 0` to `20`, each a sequence of Clarabel interior-point programs
over a positive semidefinite cone of side `2 n` with `n` the extension row
count.  Its measured cost is `2.554e-9 * n^5.133` seconds per Clarabel
interior-point iteration and its measured memory is `74.8 * (n (2 n + 1))^2`
bytes, so the piece that would dominate first as `L` grows is the piece
that already dominates at `L = 2`, and it dominates harder.

---

## 1. The measured tier split, from the record's own chains

Every rollout's `summary.json` records `lift_reselection_seconds`,
`repair_seconds` and `velocity_seconds` beside `autonomous_wall_seconds`.
Summed over the segments of each chain (`aggregate_tiers.py`,
`tier_split.json`):

| chain, at 27 entrance keys unless stated | horizon | wall | re-selection | Dykstra repair | unguarded velocity | everything else |
|---|---|---:|---:|---:|---:|---:|
| anchor-free measured rule, Dykstra 120 (task 11 base) | `t = 20` | `5,183.6 s` | `0.803` | `0.173` | `0.024` | `0.000` |
| the same rule at Dykstra 30 (task 8 base) | `t = 20` | `4,788.1 s` | `0.923` | `0.051` | `0.026` | `0.000` |
| ball rule, Dykstra 30 (task 9 base) | `t = 20` | `4,844.7 s` | `0.924` | `0.050` | `0.026` | `0.000` |
| recorded `drive_power` chain, Dykstra 30 | `t = 20` | `4,794.4 s` | `0.921` | `0.052` | `0.026` | `0.000` |
| recorded `post_pulse` chain, Dykstra 30 | `t = 20` | `3,137.8 s` | `0.921` | `0.062` | `0.016` | `0.001` |
| 15-key ball base, Dykstra 30 | `t = 20` | `817.8 s` | `0.715` | `0.153` | `0.129` | `0.002` |

The three tiers account for the whole rollout wall to three decimal
places; the `build_seconds` the summaries report separately (`19.4 s` at
27 keys, `1.3 s` at 15) is outside it.  Artifacts read:
`output/local_runs/paper_v_dictionary_energy_power_27key_n16_measured_dyk120_base_20260918m_seg{1..6}`,
`..._measured_base_20260916i_seg*`, `..._measured_dykstra60_20260916i_seg*`,
`..._measured_dyk120_dykstra240_20260918m_seg*`,
`..._ball_base_20260917k_seg*`, `..._ball_dykstra60_20260917k_seg*`,
`..._27key_n16_20260915e_seg*`, `..._postpulse_27key_n16_20260915e_seg*`,
`output/local_runs/paper_v_dictionary_energy_none_27key_n16_measured_dyk120_base_20260918o_seg*`,
and `output/local_runs/paper_v_cheap_projection_task10_{ball,carried}_{base,dykstra60}_t20_20260916_v1`.

Per opportunity and per solve, on the same reads:

| quantity, with its definition | 15 entrance keys (24 extension rows, 44 state, 62 frontier) | 27 entrance keys (36 extension rows, 56 state, 70 frontier) |
|---|---:|---:|
| seconds per repair opportunity, Dykstra 30, all opportunities applied and skipped | `5.222 ms` | `10.098 ms` |
| the same at Dykstra 60 | `9.641 ms` | `19.347 ms` |
| seconds per stage velocity evaluation, `drive_power` mode (two `unguarded_velocity` calls each) | `4.412 ms` | `5.13 ms` |
| seconds per lift re-selection, retry ladder included | `1.170 s` | `8.325 s` |

## 2. The repair's per-opportunity wall is linear in the declared Dykstra
## count, and the two parts scale differently

Fitting `a + b k` on the one-setting twins that differ only in
`dykstra_iterations` (`dykstra_count_fit.py`, `dykstra_fit.json`):

| entrance keys, extension rows | fixed part `a` | per alternation pass `b` | check |
|---|---:|---:|---|
| 15 keys, 24 rows (ball) | `0.802 ms` | `0.1473 ms` | --- |
| 15 keys, 24 rows (carried) | `0.534 ms` | `0.1587 ms` | --- |
| 27 keys, 36 rows (measured rule) | `0.850 ms` | `0.3083 ms` | predicts `37.84 ms` at Dykstra 120 against the measured `37.24 ms` (`0.984`), and `74.84` against `67.98` at Dykstra 240 (`0.908`, where the alternation's early exit begins to bind) |
| 27 keys, 36 rows (ball rule) | `1.176 ms` | `0.2983 ms` | --- |

`a` is what `AffineGramCoordinates.repair`
(`paper_5/src/paper5/stability/cheap_projection_route.py:1166`) does once:
two builds of the realified target features, four eigendecompositions of
the two blocks before and after, and the least-squares map back
`scales * (pseudo_inverse @ residual)`.  `b` is one pass of
`AffineGramCoordinates._dykstra` (`:1315`): the cone clip of both blocks
through `_cone_projection`, the two range-basis products of the affine
Gram projection, and the block minima test.

## 3. Each piece's cost law in its own dimension

Dimensions, as the dimension-growth counting names them: `n` the extension
row count, `r = 8` the retained joint Gram row count, `m = n^2 + r^2` the
realified feature dimension, `s` the propagated state coordinate count,
`f` the frontier key count, `p = s + f` the movable coordinate count.
Timings are the module's own functions on synthetic operands of the same
shape, median of repeated calls on the replacement M4 Pro, one process at
a time under `pipelines/shell/ram_guard.py --limit-mb 3000`.

### (a) The eigendecomposition of the Gram and the cone clip

`time_primitives.py` / `repair_primitives.json`, `n` from `24` to `256` at
`r = 8`; `time_two_axis.py` / `two_axis.json` separates `n` from `p`.

| piece | fitted law | range of validity | worst relative residual |
|---|---|---|---|
| one Dykstra alternation pass | `3.199e-7 * n^1.844 * p^0.072` s | `24 <= n <= 160`, `64 <= p <= 1024` | `0.19` |
| the cone clip alone (`_cone_projection`, both blocks) | `2.876e-7 * n^1.870 * p^-0.017` s | same | `0.23` |
| the range-basis products alone | `2.168e-11 * n^2.238 * p^0.963` s | same | `0.53` |

Read once: the alternation pass is **linear in the realified feature
dimension `m`** and is not the eigendecomposition's `n^3` anywhere in the
measured range.  At `n = 36` the `eigh` of the `36`-row block costs
`0.139 ms` against the pass's `0.334 ms`, and the pass's exponent stays at
`1.84` to `256` rows.  What carries it is the `O(m)` packing and unpacking
`_hermitian_features` / `_hermitian_from_features` perform twice per block
per pass, each rebuilding `np.triu_indices(rows, 1)` on every call.  The
synthetic pass reproduces the in-rollout `b` of section 2 to `1.27` at
24 rows and `1.08` at 36.

The `n^3` eigendecomposition overtakes the `O(m)` packing only where
`n^3 / n^1.84` closes the measured factor, which at the fitted
coefficients is beyond `n = 256`; the separate `eigvalsh` timing in
`completion_primitives.json` gives `5.53e-8 * n^2.18` s over
`24 <= n <= 160`, still short of its own cubic.

### (b) The least-squares map back through the affine map

`m` by `p` pseudo-inverse applied to one residual:
`2.904e-11 * n^3.292` s over `24 <= n <= 256` at `p` growing with `n`
(`repair_primitives.json`), which is the `O(m p)` product with `p` growing
at the grid's own rate; the separated fit of the same product shape is the
`range_basis_products` row above, `p^0.963`.  So the map back is
`O(m p) = O(n^2 p)`.

Its factorization --- `np.linalg.pinv` plus `np.linalg.svd` of the `m` by
`p` design, built once per rollout by `AffineGramCoordinates.factorization`
(`:965`) --- is `1.356e-6 * n^2.561` s over `24 <= n <= 160`, that is
`O(m p^2)`: `0.0130 s` at `n = 36` and `0.720 s` at `n = 160`.  It is paid
once and is negligible against the rollout.

### (c) The completion's interior Newton iterations

`SymmetryReducedPositiveExtension._complete_clarabel_newton`
(`paper_5/src/paper5/stability/apcm_positive_extension.py:886`) never hands
Clarabel the log-det problem.  It evaluates a `derivatives` closure in
numpy, and where the interior step is refused it hands Clarabel a
frontier-sized quadratic program over the cone
`_completion_cone_data` builds (`:1285`): one `PSDTriangleConeT(2 n)` on
the realified scaled extension matrix, triangular vector length
`s_c = n (2 n + 1)`, and one `NonnegativeConeT(2 f)` for the frontier box,
with `f` free variables.  The `iterations` a re-selection records is the
**sum of Clarabel's own interior-point iterations** over those programs ---
the facial reference's, each conic Newton step's, and the certificate's ---
not the outer Newton count.

The two costs, measured separately:

| piece | fitted law | range of validity |
|---|---|---|
| one numpy `derivatives` evaluation: `einsum("ai,kab,bj->kij", ...)` | `2.334e-9 * n^2.165 * f^0.976` s | `24 <= n <= 128`, `48 <= f <= 768` |
| `einsum("ab,kbc->kac", ...)` | `2.789e-10 * n^2.411 * f^1.004` s | same |
| `einsum("kab,lba->kl", ...)` | `1.616e-12 * n^2.743 * f^2.013` s (worst relative residual `1.71`: the `f n^2` complex `transformed` array leaves cache at `f n^2 > 4e6` and the measured cost steps up) | same |
| the whole numpy Newton evaluation | `0.934 ms` at `n = 36, f = 70`; `165 ms` at `n = 160, f = 320` | `24 <= n <= 160` |
| **one Clarabel interior-point iteration of the conic program** | `2.554e-9 * n^5.133` s along the record's own two frontier keys per extension row, worst relative residual `0.037` over 8 instances | `12 <= n <= 52`, `24 <= f <= 104` |
| the same with the frontier axis separated | `2.013e-8 * n^4.204 * f^0.286` s, worst relative residual `1.03` over 24 instances --- the frontier axis is not a power law here: at `n = 36` a sixteenfold `f` (`35` to `560`) moves the iteration only `1.58` times | `12 <= n <= 52`, `30 <= f <= 640` |
| the conic program's peak resident bytes above a bare interpreter | `74.8 * s_c^2` bytes, that is `299 n^4` bytes; the coefficient falls from `105` at `n = 16` to `73.7` at `n = 52` and is taken from the last three points | `16 <= n <= 52` |

Scripts: `time_completion.py`, `time_two_axis.py`, `time_conic_qp.py`,
`time_conic_qp_rows.py`, `time_conic_qp_frontier.py`,
`measure_conic_memory.py`, `fit_conic_law.py`; records
`completion_primitives.json`, `two_axis.json`, `conic_qp_rows.json`,
`conic_qp_frontier.json`, `conic_memory.jsonl`, `conic_law.json`.

**The numpy `derivatives` evaluation is not the completion's cost.**  At
the committed dimensions it is `0.934 ms` against `0.2487 s` for one
Clarabel interior-point iteration, a factor `266`.

**The conic law is anchored on the record.**  Fitting `seconds` on the
recorded `iterations` over the 500 re-selections each chain's
`reselections.json` carries (`reselection_iterations.py`,
`reselection_iterations.json`):

| chain | solves | Clarabel iterations per solve, mean | fit | correlation |
|---|---:|---:|---|---:|
| task 11 base, Dykstra 120 | 500 | `35.3` | `1.2201 + 0.20148 * iterations` s | `0.9924` |
| task 9 ball base, Dykstra 30 | 500 | `37.1` | `1.4455 + 0.20211 * iterations` s | `0.9924` |
| task 13, hyperplane off | 500 | `25.9` | `-0.0829 + 0.24229 * iterations` s | `0.9865` |

The synthetic law reads `0.2487 s` per Clarabel iteration at `n = 36`,
`f = 70` against the in-rollout `0.2015 s` on the task-11 chain, a ratio
of `1.23`; and the whole re-selection tier it predicts for a
500-re-selection chain to `t = 20`, `4,396 s`, against the recorded
`4,162 s`, a ratio of `1.06`.

On the task-11 chain the 64 solves that end in
`moment_extension_degeneracy` take `13.089 s` against the certified path's
`7.626 s` over `58.1` against `31.9` Clarabel iterations, and carry
`0.201` of the re-selection tier.

### (d) The velocity evaluation itself

`CheapProjectionRoute.unguarded_velocity` (`:1824`) is
`carried_completion` --- one `extension.matrix` build,
`O((lower + f) n^2)`, plus two `eigvalsh` of the `n`-row matrix, `O(n^3)`
--- and then `FixedDictionaryProjectedAPCM.targets`, the archive right-hand
side with the exact entrance and active commutators.  In `drive_power`
mode the stage evaluation calls it twice.  The two parts the repair shares
account for about `0.25 ms` of the measured `5.13 ms` per stage at 27
keys; the rest is `targets`, whose only declared dimension is the
enumerated word-pair count, `1,296` at the `L = 2` 27-key dictionary
against `78,961` at the `L = 3` dictionary of the same rule
(`count_l_site_construction.py`, run at `--sites 2 3`;
`repair_overhead_counts_L2.json`).  **No `L`-site velocity exists, so this
is a counted structural factor and not a timing.**

## 4. Which piece dominates first as `L` grows

The counts at the same dictionary rule `entrance_plus_two_spin_one_boson`
(`count_l_site_construction.py`; the `L = 2` row reproduces the committed
dictionary exactly --- 27 active keys, 36 extension rows, 70 frontier
keys, 56 state coordinates, 41 lower keys):

| sites | active keys | extension rows `n` | frontier keys `f` | state coordinates `s` | enumerated word pairs |
|---:|---:|---:|---:|---:|---:|
| 2 | 27 | 36 | 70 | 56 | 1,296 |
| 3 | 260 | 281 | 1,555 | 343 | 78,961 |

Reading every fitted law at those two rows, with the task-11 chain's own
schedule (8,000 declared steps to `t = 20`, three stage repairs per step,
`dykstra_iterations = 120`, one re-selection every 16 steps at `35.3`
Clarabel iterations each) --- `read_repair_overhead.py`,
`repair_overhead_reading.json`:

| quantity | `L = 2` | `L = 3` | ratio |
|---|---:|---:|---:|
| seconds per Clarabel interior-point iteration | `0.249` | `9,472` | `3.81e4` |
| seconds per completion re-selection | `10.0` | `3.34e5` | `3.34e4` |
| seconds per Dykstra alternation pass | `0.336 ms` | `18.05 ms` | `53.8` |
| seconds per repair opportunity at Dykstra 120 | `0.0411` | `2.167` | `52.7` |
| seconds per stage velocity, by the word-pair count | `5.13 ms` | `0.313` | `60.9` |
| seconds per declared step | `0.764` | `2.09e4` | `2.74e4` |
| wall to `t = 20` at `h = 0.0025` | `6,111 s` (recorded `5,183.6 s`, ratio `1.18`) | `1.67e8 s` = `4.65e4` hours | `2.74e4` |
| peak resident bytes of the conic program | `0.481 GiB` | `1,744 GiB` | `3,624` |
| tier shares, re-selection / repair / velocity | `0.818` / `0.162` / `0.020` | `1.000` / `0.000` / `0.000` | --- |

**The completion re-selection dominates first and never stops dominating.**
It already carries `0.803` of the measured `L = 2` wall, and it is the one
piece whose exponent in the extension row count is above `2`: `n^5.133`
against the alternation pass's `n^1.844` and the map back's `O(n^2 p)`.
Between two sites and three its share goes from `0.818` to `1.000` on the
fitted laws.  No ordering among the other three pieces can change the
answer, because their sum at `L = 3` is `0.0004` of the step.

**The binding constraint arrives before the wall does.**  The conic
program's dense block is `74.8 (n (2 n + 1))^2` bytes, so the memory
budget caps the extension row count:

| budget | largest extension row count | triangular vector length | seconds per Clarabel iteration | seconds per re-selection | re-selection wall to `t = 20`, 500 solves |
|---|---:|---:|---:|---:|---:|
| `3 GB`, the guard this round used | `n <= 56` | `6,328` | `2.40` | `86.0` | `11.9` hours |
| `18 GB`, the all-agent budget of `agent_guidance/shared/memory-budget.md` | `n <= 87` | `15,225` | `23.1` | `815` | `113` hours |
| `24 GB`, the whole machine | `n <= 94` | `17,766` | `34.3` | `1,212` | `168` hours |

`L = 3` needs `n = 281` at this dictionary rule, three times past the
machine's cap, and `1,744 GiB`.  The repair side has no such wall: at
`n = 281` its range basis is `79,025` by `1,898`, `1.1 GiB`, and its
once-per-rollout `pinv` is `2.9e11` flops.

## 5. What this does and does not settle

- It does not measure an `L`-site closure.  Every Paper V module is
  dimer-specific and nothing here changed that; the `L = 3` column is the
  fitted laws read at counted dimensions, with the velocity scaled by a
  counted word-pair ratio rather than timed.
- The conic law's range of validity is `12 <= n <= 52` in the extension
  row count, which is the largest program that fits the `3 GB` guard.
  Reading it at `n = 281` is an extrapolation across a factor `5.4` in the
  dimension and `3.8e4` in the answer, and the frontier axis is outside
  any power law (`f^0.286` carries a worst relative residual of `1.03`).
  The memory law, `74.8 s_c^2` bytes, is the firmer of the two: its
  coefficient is already flat to `2` percent over the last three measured
  rows and it is a storage count, not a throughput.
- Two cheap structural facts sit inside the completion and were not
  pursued: `reduced_coefficients` is rebuilt at every Newton iteration
  though it depends only on the face's support, and when the warm start is
  strictly interior that support is the identity, so the
  `2.334e-9 n^2.165 f^0.976` seconds it costs buys a copy; and
  `_hermitian_features` / `_hermitian_from_features` rebuild
  `np.triu_indices` on every call, which is what makes the alternation
  pass `O(m)` rather than `O(n^3)` bound.  Neither is the route's cost at
  any dimension measured here.

## 6. Files

Scripts and records, all under
`output/local_runs/paper_v_scaling_20260920/`: `aggregate_tiers.py`,
`tier_split.json`; `dykstra_count_fit.py`, `dykstra_fit.json`;
`time_primitives.py`, `repair_primitives.json`; `time_two_axis.py`,
`two_axis.json`; `time_completion.py`, `completion_primitives.json`;
`time_conic_qp.py`, `time_conic_qp_rows.py`, `conic_qp_rows.json`;
`time_conic_qp_frontier.py`, `conic_qp_frontier.json`;
`measure_conic_memory.py`, `run_conic_memory.sh`, `conic_memory.jsonl`;
`reselection_iterations.py`, `reselection_iterations.json`;
`fit_laws.py`, `fit_conic_law.py`, `conic_law.json`;
`read_repair_overhead.py`, `repair_overhead_reading.json`;
`repair_overhead_counts_L2.json` (the `L = 2` counts of this round's own
`count_l_site_construction.py`).
