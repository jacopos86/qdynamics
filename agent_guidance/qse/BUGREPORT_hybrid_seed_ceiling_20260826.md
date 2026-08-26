# Bug report — adaptive QSE: seeded growth pins to a fixed error ceiling

For Codex. Contract: `agent_guidance/shared/agent-handoff-contract.md`.
Related: `agent_guidance/qse/BUGREPORT_adaptive_qse_root_indexing_20260826.md`
(items 1--3 there were caller-side and are resolved).

## Anchors

| field | value |
|---|---|
| Checkout | `/Users/jakestrobel/local_repos/Holstein_test_fullclone_3` (NOT the Documents mirror) |
| Branch | **Work directly on `paper-ii-exchange-selector`; do not create a branch or worktree.** `git pull --rebase` before each commit. |
| Module | `pipelines/qse_spectra/adaptive_qse_benchmark.py` (yours) |
| Caller | `pipelines/exact_bench/paper_iii_growth_trace_campaign.py::trace_hybrid` (author's; inspect, do not edit -- report needed changes) |
| Ownership | You own `adaptive_qse_benchmark.py` and its tests. `MATH/paper_details/**`, `pipelines/reporting/**`, `pipelines/exact_bench/**` are the author's. |
| Tests | `python3 -m pytest test/test_adaptive_qse_benchmark.py test/test_paper_iii_problem_provider.py -q` currently green. |

## Symptom

Seeding the benchmark with our selected records instead of the fixed
linear-response class makes it converge to a **fixed, seed-independent error
ceiling** and stop, declaring `RESIDUAL_CONVERGED`.

Regime `intermediate_strong` at `n_ph_max=3` (u=1.25, g_ep=0.790569415042),
`target_roots=7`, `eps_residual=1e-14`, `max_dimension=80`, error measured as
`max_{nu<=6} |root_energies[1:][nu] - exact_excitation[nu]|`:

```
seed                       size   best error   at dim   stop
fixed linear-response        18     4.77e-15       63    RESIDUAL_CONVERGED
our selected support, k=8     8     3.71e-01       51    RESIDUAL_CONVERGED
our selected support, k=12   12     3.71e-01       52    RESIDUAL_CONVERGED
our selected support, k=16   16     3.71e-01       50    RESIDUAL_CONVERGED
our selected support, k=24   24     3.71e-01       51    RESIDUAL_CONVERGED
our selected support, k=32   32     3.71e-01       56    RESIDUAL_CONVERGED
```

The ceiling is `3.71e-01` to three digits for every seed size from 8 to 32 --
it does not improve as the seed grows, and growth to dimension ~50 does not
move it. Meanwhile the same code from an 18-record fixed-class seed reaches
machine precision at comparable dimension.

Reproduce with `pipelines/qse_spectra/paper_iii_problem.py::load_problem` for
that regime, `select_static_qse_records(mode="geometry_selected",
geometry_target_roots=6, geometry_cost_discount_alpha=1.0)` for the support,
and `run_adaptive_qse_benchmark` as above.

## Why this is likely a defect rather than physics

A larger, richer seed cannot make a Davidson-type method *worse* if growth is
sound: the seed span is a subspace of what the method could otherwise build,
and additional directions cannot remove reachable states. Two further points:

- the error is **identical across seed sizes**, which points at a systematic
  effect (a missed root, a fixed shift, or a stalled admission path) rather
  than at conditioning, which would vary with seed;
- convergence is declared at residual `<= 1e-14` while the window error is
  `3.7e-01`, so the residual being monitored is not the residual of the roots
  being reported.

That last point is the same class of issue as item 2 in the earlier report,
which we agreed to keep open: a residual-only stop cannot see a root it never
represented.

## What to determine

1. **Is the ceiling a missed root?** Print the reported roots against the exact
   excitations for the k=16 seed. If a target state is absent and a spurious
   one is present, this is Ritz-blindness of the residual stop and the finding
   belongs in the paper -- **say so and do not patch it**.
2. **Is admission stalling?** Log admitted/rejected direction counts per
   iteration for the fixed-class seed versus our k=16 seed. If our seed drives
   candidates below the independence floor so that no useful direction is ever
   admitted, report the floor's role.
3. **Is the residual computed over the reported roots?** Confirm
   `max_root_residual` is taken over the same slots reported in
   `root_energies` after the `R+1` convention. A mismatch here would explain
   convergence-at-wrong-answer directly.
4. **Does the preconditioner assume an orthonormal or well-conditioned seed?**
   Our support is chosen for cost-weighted novelty, not conditioning, and its
   Gram matrix is far from identity. If the preconditioner or the
   independence test assumes otherwise, that is the defect.

## What NOT to do

- Do not tune `eps_residual`, the independence floor, or the denominator floor
  to make the number look better. If a guard is genuinely wrong, fix the guard
  and say why.
- Do not edit the caller or the manuscript. Report required caller changes.
- Do not claim the method is fine because the fixed-class seed works: the
  question is why a superset-quality seed performs worse.

## Report back

Contract section 6, plus: the roots-vs-exact table for the k=16 seed, the
admission log comparison, and a verdict naming which of the four items above
is the cause. If the answer is genuine Ritz-blindness, state it as a finding
with evidence rather than as a bug to be fixed.
