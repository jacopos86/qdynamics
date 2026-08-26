# Paper II — comparison protocol

How every reported comparison in this lane is constructed. The short version:
**methods are compared at fixed accuracy, never at a fixed threshold**. The
primary experiment is factorial: two algorithmic methods are crossed with
three time-step controllers while the inner numerics are locked.

Codex and any other executor should treat this as the contract a reported number
has to satisfy. If a number in the manuscript cannot be traced to a cell built
this way, it is not evidence.

---

## 1. Why not compare at a fixed threshold

Each adaptive method carries a McLachlan-distance cut `L²_cut`. For AVQDS, the
cut closes greedy appends. For generalized exchange, it closes the insertion
faces of the patch family; the pure-deletion face remains eligible. The obvious
comparison is to set both cuts to the same value. That is not sound here, and
the reason is measured rather than assumed.

Sweeping the threshold over a 100× range (1e-2 → 1e-4) on three drives:

- tightening it improves accuracy **monotonically on 1 of 3 drives** for the
  comparator and **0 of 3** for this route;
- a 100× threshold change moves support by only **~10 coordinates**;
- on the weak slow drive at cuts 1e-2 and 3e-3 the compared trajectory arrays
  and support histories are identical (run metadata excluded), because neither
  insertion rule triggers.

So the knob has a dead zone, then a live zone, and is non-monotone inside it.
Extending the ladder to 3e-6 does not fix this: AVQDS on the fast weak drive
runs 5.58e-4 → 3.78e-4 → 8.94e-4 → 4.28e-3 → 1.25e-3 → 2.01e-4 as the cut
tightens, and this route is similarly non-monotone.

Reading a method's quality off one setting of its own dial is therefore
meaningless. Any single-threshold comparison in this lane is superseded.

## 2. Factorial comparison and locked axes

The primary experiment contains two algorithmic methods—APM generalized
exchange and AVQDS greedy growth—and three time-step controllers—tangent-state,
parameter, and composed. Their Cartesian product gives six run configurations.
Calling these six methods is incorrect: the factorial design separately
estimates the method effect, controller effect, and their interaction.

The registry in `pipelines/time_dynamics/paper_ii_runs.py` resolves every
configuration. The campaign layer supplies only registered factor names and
the activation cut; it does not assemble runner flags.

| axis | what is locked | current value |
|---|---|---|
| Physics | seed, regime, drive, reporting grid | `hh_snake_nph1`, six drives, t=10 / 251 checkpoints |
| Shared inner numerics | integrator, Tikhonov ridge, damping, pinv rcond | Euler, 1e-6, 0.0, 1e-10 |
| Step control | which quantity bounds the step | see §3 |
| Candidate pool | membership and order | 125 deduplicated words, cap 128 (non-binding) |
| Guards | enumeration and certification budgets | 50000 / 12 / 2, insertion batch 1 |
| Activation | `L²_cut` | swept per cell, see §4 |

`test/test_time_dynamics_campaign.py` asserts the full factor product, the
Euler/1e-6 lock, method-specific threshold flags, FakeMarrakesh compile profile,
and fail-closed package preparation. `run_lock.py::assert_comparable` verifies
shared physics after execution.

## 3. Step control is an axis, not a property of a method

The controllers bound different finite-step quantities. All are initially
tested under the shared Euler/1e-6 inner numerical lock.

| control | bounds | value |
|---|---|---|
| `state_motion_1e-2` | tangent-state step, then subdivides | 1e-2, subdivision budget 10 |
| `delta_theta_5e-3` | `max_μ |θ̇_μ| dt` | 5e-3 |
| `state_motion_1e-2_plus_parameter_5e-3` | both finite-step bounds | 1e-2 and 5e-3 |

Each controller is paired with both structural methods. The tangent-state
quantity comes from the Gram geometry already acquired for the McLachlan solve;
no trial-state overlap is prepared or measured.
Measured relevance: under state-motion control alone the three largest
step-to-step jumps carry **72–81%** of total error growth; under δθ control,
**7%**. The complete six-configuration matrix will later be repeated under RK4
as a consistency check. That repeat is deferred and does not enlarge the
initial experiment into an Euler-versus-RK4 tuning matrix.

## 4. The fixed-accuracy protocol

For each cell (drive × algorithmic method × time-step controller):

1. Use past frontier runs only to plan the initial worklist. A target-reaching
   prior contributes its resource-cheapest target-reaching cut as an anchor;
   a prior that has not yet reached the target contributes its next tighter
   cut. Include one `1, 3` logarithmic neighbor on each side. A drive without
   prior data receives the full ladder `1e-2, 3e-3, ..., 3e-6`.
2. Record mean |ΔE| against the exact time-ordered trajectory and the final
   support size `N_θ` at each rung.
3. If no rung reaches mean |ΔE| ≤ ε, extend the same `1, 3` logarithmic ladder
   to tighter cuts for that cell. Report `TARGET NOT YET REACHED` while the
   extension is pending. If a declared
   compute cap ends the search, report `TARGET NOT REACHED WITHIN BUDGET` and
   the explored cut range.
4. As a quick diagnostic, report the **minimum-final-support target-reaching
   rung**. This is not the final circuit-cost comparison.
5. For the final result, compile every target-reaching terminal ansatz with
   one locked convention, then select the minimum-`N2q` rung (tie-break by `D2q`, then
   mean |ΔE|) using the same rule in every configuration. Report all compiled
   target-reaching rungs, not only the selected one. Use the locked
   Qiskit transpilation convention and report the compiled two-qubit gate count
   `N2q`, total circuit depth `Dc`, and two-qubit depth `D2q`.

The default target is ε = 1e-4. The prior-informed campaign is prepared by:

```bash
PYTHONPATH=. python3 -m \
  pipelines.time_dynamics.campaigns.paper_ii_factorial_euler_v1 \
  --mode production --prior-root output/frontier
```

### Selection bias — state it, do not hide it

Taking the best rung per arm is a minimum over an irregular ladder. Quantify
that irregularity as the **adjacent-rung ratio**: mean |ΔE| at a cut divided by mean |ΔE| at
the next-looser cut, so a value below 1 means tightening helped and above 1
means it hurt. Over 42 adjacent pairs (3 drives × 2 arms × 7 rungs) the largest
is **19.1×**, the smallest **0.033×**, and **11 of 42 (26%) exceed 1** — one rung
of the cut changes the error by up to a factor of nineteen, and a quarter of the
time in the wrong direction.

The resulting point is an **exact-reference-tuned, per-cell resource
diagnostic**: it answers how much final ansatz was required to reach ε on that
known drive. It is not a QPU-operating rule, because a real run does not know
the exact error with which to choose its threshold, and it does not establish
that the selected threshold transfers to an unseen drive. Equal ladders and an
equal selection rule make the descriptive comparison symmetric, but do not
turn it into held-out validation. Therefore:

- report every evaluated rung, not only the selected one;
- label the selected point as per-cell exact-reference tuning;
- do not quote a best-rung ratio as a headline figure.

## 5. Reporting contract

Per cell: drive, algorithmic method, time-step controller, inner numerics,
ladder rung chosen, mean and
max |ΔE|, mean |Δn_d|, final and peak `N_θ`, insertion/deletion/exchange counts,
certification attempts and budget exhaustion, accepted internal substeps or
right-hand-side evaluations, and wall time. For the selected final ansatz,
report Qiskit-compiled `N2q`, `Dc`, and `D2q`. The compile profile is the Paper-I
lock: `FakeMarrakesh`, native basis `{sx,rz,x,cz,id}`, optimization level 1,
and transpiler seed 7. Record the Qiskit version and resolved backend metadata.

Observable panels follow the form used in this literature: observables against
the exact trajectory plus the resource counts above. The primary matching
metric in this protocol is mean |ΔE|. State infidelity, available in simulator
diagnostics via `--record-statevector`, is a secondary state-level error metric,
not an online controller input. Per-observable deviations may be plotted when
the observable set is declared before inspecting which policy looks better.

## 6. Status

| result | evidence | confidence |
|---|---|---|
| This route's growth arm = AVQDS at cut 1e-3 | 0.96× and 0.98× median, six drives, two integrators | solid |
| rk4 beats Euler ~3.5× for both methods | six drives | solid |
| δθ control removes discrete jumps (72% → 7%) | all arms | solid |
| Threshold is not a usable comparison axis | three drives, eight rungs | solid |
| At fixed accuracy ε=1e-4, the current ladder reaches the target in 3/3 vs 2/3 growth-arm cells; minimum-support target hits are 47 vs 50 and 44 vs 47 | three drives | preliminary, one seed |
| **Two-method × three-controller factorial result** | **not yet measured** | the decisive open run |

The old frontier is a threshold-planning prior only. Until the six new
configurations are measured under the common numerical lock, no factorial
claim about exchange or controller choice is supported.
