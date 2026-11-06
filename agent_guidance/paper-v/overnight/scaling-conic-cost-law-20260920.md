# The conic repair's cost as a function of the moment-matrix dimension

Key `conic-cost-law`, round of 2026-09-20, branch
`claude/paper-v-routes-20260914`. Exploratory local measurement; no default
changed, no module edited, nothing promoted. Scripts and raw rows under
`output/local_runs/paper_v_scaling_20260920/`.

**Negative result first.** There is no single power. Fitted over the whole
recorded range of the positive-semidefinite cone side dimension `n = 12` to
`80`, Clarabel's solve time gives an exponent of `3.24 +/- 0.11`; fitted over
`n = 40` to `80` the same data give `4.17 +/- 0.04`; the synthetic extension
over `n = 84` to `104` gives `4.87 +/- 0.12`. The exponent is not a constant
of the problem -- it rises with `n` -- so any one number quoted for it is a
statement about the window it was fitted in, and extrapolating the small-`n`
exponent understates the cost of a larger cone by orders of magnitude.

---

## 1. What the recorded instances are

`output/local_runs/paper_v_cwrmf_record_strict_t001_h0025_20260915_v1/conic_instances/`
holds 36 Clarabel problems from the strict always-on carried-witness guard,
`t = 0` to `0.01` at `h = 0.0025`, read here through the existing loader
`cwrmf_solver_instances.load_instances`
(`output/local_runs/paper_v_scaling_20260920/conic_instance_shapes.py`).

| tier | count | cone specification | variables | constraint rows | recorded Clarabel solve time |
|---|---:|---|---:|---:|---:|
| `critical_schur` | 3 | `PSDTriangleConeT(12)`, `SecondOrderConeT(110)` | 109 | 188 | `0.0185 s` in total |
| `bundle_cone` | 33 | `PSDTriangleConeT(n)`, `n = 12` to `80`, `SecondOrderConeT(341)` | 340 | `n(n+1)/2 + 341` | `145.80 s` in total |

The `bundle_cone` tier carries `0.9999` of the recorded solve time. Only its
positive-semidefinite block changes size: the variable count stays at 340 and
the second-order block stays at 341 rows at every side dimension, so the cone
row count `m = n(n+1)/2` is the only growing quantity. At `n = 80`,
`m = 3240` and the block is `0.9877` dense (1,088,000 nonzeros of
1,101,600).

Operand structure at `n = 12`, `52` and `80`
(`output/local_runs/paper_v_scaling_20260920/bundle_instance_anatomy.py`):
the objective matrix is diagonal on 340 variables with two distinct entries
(`1.000001` and `1e-6`); the objective vector has 231 nonzeros and norm
`1.7e-9` to `7.0e-9`; the positive-semidefinite right-hand side is the
`svec` of a base matrix with exactly six eigenvalues below `1e-9` at every
recorded side and a negative minimum (`-7.48e-4` at side 12, `-5.41e-3` at
side 80); the second-order block is the velocity ball of radius `10.11` to
`10.44` on the same 340 variables, and the recorded solution's norm is
`9.6e-5` to `1.2e-3`, so that ball is never active.

## 2. Replay on this machine

`output/local_runs/paper_v_scaling_20260920/conic_cost_fit.py` re-solves all
36 recorded operands through `cwrmf_solver_replay.ClarabelCandidate` with no
setting overrides, so the recorded Clarabel settings are the ones used. The
interior-point iteration count is reproduced exactly on all 36; the wall time
is `0.87` to `1.00` of the recorded `solve_time`, the ratio rising with the
side dimension (`14.18 s` replay against `14.46 s` recorded at `n = 80`).
Replay is therefore a valid clock for this machine, and all fits below use
it.

## 3. What the exponent is, and over what window

Ordinary least squares in logarithms. The iteration count is separated out
because it varies between 14 and 28 across the 33 bundle instances and by
itself scales only as `n^{0.148 +/- 0.057}` (`R^2 = 0.18`): the algebraic
cost per interior-point iteration is the quantity that scales.

| quantity | window | exponent in `n` | `R^2` |
|---|---|---:|---:|
| recorded solve time | `n = 12` to `80` | `3.24 +/- 0.11` | 0.968 |
| recorded solve time | `n = 40` to `80` | `4.17 +/- 0.04` | 0.998 |
| replay time per iteration | `n = 12` to `80` | `3.12 +/- 0.14` | 0.945 |
| recorded time per iteration | `n = 40` to `80` | `4.17 +/- 0.04` | 0.998 |
| synthetic time per iteration | `n = 12` to `104` | `3.15 +/- 0.15` | 0.970 |
| synthetic time per iteration | `n = 40` to `104` | `4.22 +/- 0.05` | 0.998 |
| synthetic time per iteration | `n = 84` to `104` | `4.87 +/- 0.12` | 0.998 |
| recorded iteration count | `n = 12` to `80` | `0.148 +/- 0.057` | 0.177 |

Extending the range changes the answer: the exponent measured beyond the
recorded range is `4.87 +/- 0.12`, against `4.17 +/- 0.04` inside it.

## 4. The synthetic problems, and how they differ

`output/local_runs/paper_v_scaling_20260920/synthetic_conic_sweep.py` takes
the recorded side-80 instance as a template and keeps verbatim everything
that does not depend on the side dimension: the diagonal objective matrix on
340 variables, the objective vector, the second-order block of the cone
matrix and of the right-hand side, and the recorded Clarabel settings
dictionary through `build_solver_settings`. Three declared differences:

- the coefficient columns -- one `svec` per variable -- are drawn i.i.d.
  normal and scaled to the recorded side-80 column entry root mean square
  `1.6047`, where the recorded columns come from the step-scaled completion
  coefficient tensor contracted into the active critical bundle basis, so
  their spectra and mutual correlations are not reproduced;
- the base is built feasible by construction: a target matrix carrying the
  recorded side-80 base's eigenvalue spectrum resampled to the requested side
  by linear interpolation in sorted order with negatives clipped to zero (so
  the six eigenvalues below `1e-9` and the positive range to `8.33` are
  carried over and the target meets the cone with a six-dimensional null
  face), minus the coefficient combination of a pseudorandom displacement of
  norm `1.1814e-3`, the recorded side-80 solution's norm. A first attempt
  without this construction returned `PrimalInfeasible` at side 80 and is
  not used;
- the synthetic positive-semidefinite block is fully dense against the
  recorded `0.9877`.

At the eight side dimensions both sources cover, the synthetic per-iteration
time is `1.065` to `1.298` times the replay per-iteration time, geometric
mean `1.208`; the ratio stays near `1.25` up to side 56 and falls to `1.203`
at side 64, `1.105` at side 72 and `1.065` at side 80. The synthetic problems return
`AlmostSolved` from side 40 upward; six of the 33 recorded instances return
it, all at side 68 or above. The fitted exponent over the common window `n = 40` to `80` is
`4.17 +/- 0.04` recorded against `4.22 +/- 0.05` synthetic: within one
standard error, the synthetic problems scale as the real ones.

The sweep stops at side 104. The wall is memory, not time: peak resident
memory over the synthetic solves scales as `n^{2.14 +/- 0.22}` over side 40
to 104, reaching `2,896 MB` at side 104, and the side-112 solve was killed by
`ram_guard` at the declared `3,000 MB` limit. At side 104 one solve costs
`31.2 s`, well inside a twelve-minute process, so time would have allowed
several more side dimensions.

## 5. The cost law

Fitted by nonnegative least squares on the relative scale
(`output/local_runs/paper_v_scaling_20260920/final_conic_cost_law.py`) over
43 solves at side 32 to 104, recorded and synthetic pooled:

```
seconds per interior-point iteration = 9.96e-7 m^1.5 + 1.05e-11 m^3,
    m = n (n + 1) / 2
```

Relative residual root mean square `0.077`, maximum absolute `0.166`. A
2,000-draw resampling of the fitted solves gives `9.33e-7` to `1.07e-6` for
the `m^1.5` coefficient and `9.91e-12` to `1.13e-11` for the `m^3`
coefficient (95 percent). The two terms are equal at `m = 2082`, side 64:
below that the cheaper term dominates, above it the cubic one does, which is
why the measured exponent rises through the range.

**Validity.** Side 32 to 104, that is `m = 528` to `5,460`, on this machine
under the recorded strict settings (`tol_gap_abs = tol_gap_rel = tol_feas =
1e-10`, `max_iter = 600`, `max_threads = 1`, iterative refinement on at
`1e-12`). Below side 32 the law under-predicts by `0.60` to `0.70` of the
measurement on the six solves at side 12 to 26 -- the side-independent parts
(340 variables, the 341-row second-order block) set a floor the law does not
carry -- so it is a lower bound there and must not be used for small cones.
Above side 104 it is extrapolation: the `m^3` term is already `0.75` of the
law's value at side 104, so the extrapolation is essentially the cubic term,
fitted over a range where it is the majority contribution only in the top
quarter.

Total solve time is that rate times the iteration count, which is the
weakly-scaling factor measured in section 3; the recorded bundle instances
have median 22, mean 21.7, range 14 to 28.

| side `n` | `m` | seconds per iteration | seconds per solve at 22 iterations (95 percent) |
|---:|---:|---:|---:|
| 80 | 3,240 | `0.540` | `11.9` (`11.6` to `12.3`) |
| 104 | 5,460 | `2.109` | `46.4` (`44.7` to `48.8`) |
| 128 | 8,256 | `6.649` | `146` (`140` to `155`) |
| 160 | 12,880 | `23.87` | `525` (`500` to `561`) |
| 192 | 18,528 | `69.22` | `1,523` (`1,445` to `1,632`) |
| 256 | 32,896 | `379.3` | `8,345` (`7,900` to `8,961`) |
| 384 | 73,920 | `4,256` | `93,640` (`88,545` to `100,668`) |
| 512 | 131,328 | `23,804` | `523,679` (`495,013` to `563,142`) |

Rows at side 128 and beyond are extrapolation, and their intervals are the
fit's resampling spread only; they do not carry the risk that the local
exponent keeps rising, which the measured rise from `4.17` to `4.87` says it
does.

## 6. Turning the law into a wall-time prediction

Over the 34 guarded carried-witness rollouts on record that make at least one
conic solve -- 1,530 solves, `20,088 s` of rollout wall in total -- the
`clarabel_solve` entry of `tier_seconds` carries `0.9939` of the wall pooled,
with a per-rollout range of `0.9641` (a 16-solve smoke rollout) to `0.9974`
and a median of `0.9929`. The strict recording rollout that produced the 36
instances is `0.9927` (`146.81 s` of `147.89 s`); the longest rollout on
record, the balanced bounded-depth event-triggered guard from `t = 0.1175` to
`0.5`, is `0.9942` (`14,636 s` of `14,722 s`, 381 solves). The wrapper around
Clarabel costs `0.027 s` per call on the recording rollout (`146.81 s` of
tier time against `145.82 s` of Clarabel's own `solve_time` over 36 calls).

So:

```
guarded rollout wall seconds
  = (solves per accepted step) x (steps) x (iterations) x (per-iteration law)
    / 0.9939
```

Solves per accepted step is a property of the guard policy, not of the cone:
`9.00` for the strict always-on guard (36 solves, 4 accepted steps),
`2.50` and `2.49` for the two balanced event-triggered rollouts (50 solves
over 20 steps; 381 over 153). Steps is the horizon divided by `h = 0.0025`,
that is 400 per unit of time.

At the strict guard's 9 solves per step, 400 steps per unit time and 22
iterations, one unit of propagation time at a fixed side dimension costs
`9 x 400 x 22 x rate / 0.9939`: `43,100 s` at side 80, `168,100 s` at side
104, `1.90e6 s` at side 160. At the event-triggered guard's 2.49 solves per
step the same three are `11,900 s`, `46,500 s` and `526,300 s`.

## 7. What this does and does not say about the number of sites

This round measured the cost as a function of the cone side dimension only.
The side dimension of the recorded rollout is set by the adaptive spectral
bundle, which grows two rows per cutting-plane iteration within an atom
(`12, 16, 26, 34, 36, ..., 80` across the recorded sequence) and resumes at
the previous size for the next atom; `maximum_active_critical_modes` is 40
and `mean_active_critical_modes` is 27.8 on the recording rollout, against a
`literal_gram_dimension` of 62. How the side dimension itself depends on the
site count `L` is not measured here and is not derivable from these data:
every Paper V module is dimer-specific. What the law supplies is the second
factor of that product -- given a predicted side dimension at `L` sites, the
conic repair's cost follows from section 5, and its wall from section 6.
