# Bug report — adaptive QSE benchmark: root indexing and terminal state

> **STATUS 2026-08-26: items 1-3 RESOLVED by the author; both were caller-side
> defects in the growth-trace port, not in your module.** Fixed at
> `paper_iii_growth_trace_campaign.py::trace_adaptive_qse`:
> (a) `root_energies` is the lowest N Ritz values INCLUDING the ground root —
> the caller now requests `R+1` and compares `root_energies[1:]`;
> (b) `direction_resources` expects keys `n2q/d2q/dc`, the caller was passing
> `c_hat_*`, which validated as absent and charged every direction zero.
> After the fix the benchmark reaches 2.2e-15 (weak_weak) and 7.8e-15
> (strong_weak_u8) at dimension 16, N2q=300/D2q=540/Dc=1260, 2448 estimator
> queries. **Only the "second issue" below is still open.**

For Codex, follow-up to `HANDOFF_standard_adaptive_qse_benchmark_20260826.md`.
Contract: `agent_guidance/shared/agent-handoff-contract.md`.

## Anchors

| field | value |
|---|---|
| Checkout | `/Users/jakestrobel/local_repos/Holstein_test_fullclone_3` (NOT the Documents mirror) |
| Branch / SHA | `paper-ii-exchange-selector` @ `95551965`. Your work was ported from `codex/adaptive-qse-20260826` into this branch; `pipelines/qse_spectra/adaptive_qse_benchmark.py` and `test/test_adaptive_qse_benchmark.py` are here verbatim. |
| Comparison method | Growth-trace, now LOCKED: `agent_guidance/qse/paper-iii-comparison-protocol.md` section 1. The matched-accuracy campaign you integrated against is superseded; the live consumer is `pipelines/exact_bench/paper_iii_growth_trace_campaign.py::trace_adaptive_qse`. |
| Test baseline | `python3 -m pytest test/test_adaptive_qse_benchmark.py test/test_paper_iii_problem_provider.py test/test_paper_iii_matched_accuracy_campaign.py -q` → currently green. |

## Symptom

In the growth-trace campaign the adaptive-QSE benchmark never reaches any
accuracy target. It halts almost immediately with `RESIDUAL_CONVERGED` at a
near-zero residual while its reported window error stays at 1e0.

Reproduce:

```
python3 pipelines/exact_bench/paper_iii_growth_trace_campaign.py \
  --regime-set hubbard_l2 --k-max 24 --stride 3 --exchange-policy both
```

## Evidence — this is an indexing defect, not physics

`run_adaptive_qse_benchmark(..., target_roots=6)` returns `root_energies`
whose **first entry is the ground state**, so the caller comparing
`root_energies[0:6]` against the six exact excitations is off by one.

Measured (`target_roots=6`, `eps_residual=1e-14`, `max_dimension=40`, seed =
the 10 fixed linear-response records):

```
hubbard_u8  (u=8, g=0, nph=1)   E0 = 0.5279
  ritz roots : 0.5279  1.0000  1.5279  1.5279  9.0000  9.4721
  exact refs :         1.0000  1.5279  1.5279  2.0000  2.0000  2.5279
                ^ ritz[0] == E0 exactly

weak_weak   (u=0.25, g=0.3536, nph=1)   E0 = -0.9173
  ritz roots : -0.9173 -0.0124  0.1211  1.0000  1.0229  1.3010
  exact refs :         -0.0124  0.1211  1.0000  1.0229  1.3010  2.0000
                ^ ritz[1:6] matches refs[0:5] to all printed digits
```

Your increment-2 report said "the free ground-reference root is retained
internally, so Davidson solves R+1 roots and reports the following six
excitation roots." The returned array does not match that description: it
carries the ground root in slot 0 and therefore only R-1 excitations.

## What to fix

1. **Decide and document the contract of `root_energies`.** Either it excludes
   the ground root (then the internal solve must request `R+1` and slice), or
   it includes it (then say so in the docstring and return the ground energy
   separately). Do not leave it implicit — the caller cannot tell.
2. **Make the residual consistent with the reported roots.** `max_root_residual`
   currently converges to ~1e-15 while the reported window is wrong, which is
   what makes the benchmark self-certify a miss. Whatever slot convention you
   choose, residuals must be computed over the same roots that are reported.
3. **Add a regression test that would have caught this.** On a Hamiltonian with
   known spectrum, assert `root_energies` against the exact **excitations**
   (not the full spectrum), and assert `root_energies[0] != ground_energy`
   under the documented contract. The existing 16x16 test passes because it
   compares against the lowest eigenvalues including the ground state.

## Second issue to check while there — separate from the above

At `hubbard_u8`, even after the offset, roots 4-6 read 9.0000/9.4721 against
exact 2.0000/2.5279. Determine whether that is:

- a genuine property (the subspace is exactly H-invariant at dimension 6, so
  the residual is legitimately ~1e-15 while the window is incomplete — the
  Ritz-blindness a residual-only stop cannot detect), or
- a second defect (e.g. seeds silently dropped: 10 seed elements were supplied
  and the reported dimension was 6, so 4 were rejected — at `g_ep=0` the
  phonon records decouple and their q0-projected images may be numerically
  zero).

Report which, with the admission log. **If it is genuine Ritz-blindness, say
so and do not patch it** — that is a real property of a residual-only stopping
rule and it belongs in the paper as a finding, not as a bug.

## Terminology (author's convention)

Call these things **methods** or **benchmarks**, not "arms", in code comments,
docstrings, and report-backs.

## Report back

Contract section 6 format, plus: the corrected `root_energies` against exact
excitations for `hubbard_u8` and `weak_weak` at `nph=1`, and your verdict on
the second issue with evidence.
