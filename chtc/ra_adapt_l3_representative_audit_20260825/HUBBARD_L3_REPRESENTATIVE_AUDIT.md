# Pure-Hubbard (L=3) representative audit

Date: 2026-08-25

## Paper-facing stationary-source promotion

By author decision on 2026-08-25, the Paper-I optional-control panels use the
B9-matched stationary-source, full-joint route recorded under
`paper_i_l3_b9_stationary_joint_controls_20260825`.  This is a declared route
choice, not a post-hoc selection of the globally smallest historical circuit.
It keeps the same physics, pool, adaptive shortlists, exact Qiskit marginal
selection costs, metric-informed Powell directions, and matched tight Powell
tolerances as the measured-residual full-joint controls.  Setting the
source-point active gradient to zero leaves the coupled active-coordinate
response free.

The stationary-source and measured-residual runs have identical accepted
operators, energies, graph-proxy circuit coordinates, beam ancestry, and
pruning events.  Only active-gradient estimator acquisitions change:

| route | first \(\le 10^{-6}\) | error at \(k=12\) | \(\widehat N_{2q}/\widehat D_{2q}/\widehat D_c\) at \(k=12\) | corrected \(\mathcal E\) at \(k=12\) |
|---|---:|---:|---:|---:|
| RA-Append | 8 | \(4.44\times10^{-16}\) | 252/252/463 | 7972 |
| batch | 6 | \(2.22\times10^{-15}\) | 308/308/579 | 8601 |
| beam, selected lineage | 8 | \(1.33\times10^{-15}\) | 236/236/447 | 8020 |
| pruning, all delete--refit work | 8 | \(4.18\times10^{-12}\) | 224/224/405 | 11954 |

The corresponding run IDs in `l3_prefix_metrics.csv` are
`b9_stationary_joint_append`, `b9_stationary_joint_batch`,
`b9_stationary_joint_beam`, and `b9_stationary_joint_prune`.

## Earlier cross-route diagnostic

The best matched pure-Hubbard comparison presently in the local history is
the repaired, native-coordinate, adaptive-shortlist RA route—not the later
whitened manuscript-cost route and not the unmatched historical v10 route.

At each method's first prefix with absolute error at most (10^{-6}), the
repaired adaptive RA-Append route reaches the target at (k=8) with


| route | (k) | (|E-E_{\rm ED}|) | \(\widehat N_{2q}\) | \(\widehat D_{2q}\) | \(\widehat D_c\) | \(\widehat K\) | corrected \(\mathcal E\) |
|---|---:|---:|---:|---:|---:|---:|---:|
| AVQE, tight native Powell | 8 | (1.30\times10^{-12}) | 220 | 220 | 391 | 229.75 | 6510 |
| repaired RA-Append, tight native Powell | 8 | (5.86\times10^{-14}) | 92 | 92 | 203 | 105.95 | 5188 |

Thus the repaired RA prefix reduces the two graph two-qubit coordinates by
(58.2\%\), the new total-template-depth proxy by (48.1\%\), the declared
three-coordinate scalar by (53.9\%\), and corrected estimator work by
(20.3\%\).  It also has the lower energy error at that prefix.

This is the strongest clean algorithmic comparison because both methods use
native Powell coordinates and matched tight tolerances.  It isolates the
selector more cleanly than the currently promoted whitened route, although RA
still supplies metric-informed initial Powell directions while AVQE uses its
default native directions; it is therefore a complete-pipeline comparison,
not a selector-only control.  It is not yet publication-ready:
`RESULTS_SOURCES.md` for the repaired bundle states
that three source fixes underlying the runs were uncommitted.  The exact
protocol must be frozen in a commit and reproduced once before promotion.

## Evaluation-only graph proxy

Every accepted prefix was recounted from the same ordered 24-generator intact
pool and the fixed six-qubit FakeMarrakesh embedding.  The existing graph
coordinates are

\[
\widehat N_{2q}(P)=2\max(|\operatorname{supp}P|-1,0),\qquad
\widehat D_{2q}(P)=2\operatorname{span}_{G}(\operatorname{supp}P).
\]

The audit adds a deterministic total-template-depth coordinate

\[
\widehat D_c(P)=2\operatorname{span}_{G}(\operatorname{supp}P)
                 +2b(P)+1,
\]

where (b(P)=2) when (P) contains a (Y), (b(P)=1) when it contains an
(X) but no (Y), and (b(P)=0) for a (Z)-only word.  Macro-generator
costs are sums over their nonidentity Pauli components, and the ansatz total
adds one layer for parallel Hartree--Fock reference preparation.  The scalar
reported for comparison is

\[
\widehat K=0.30\widehat N_{2q}
          +0.30\widehat D_{2q}
          +0.25\widehat D_c.
\]

This is a routing-deterministic termwise-template proxy.  It does not model
compiler cancellations, interword scheduling, basis-gate translation, or
cross-generator fusion, and therefore must not be labeled compiled depth.
It is currently an evaluation coordinate only.  Making it part of selection
would change the algorithm and require new runs.

## Matched route comparison

The following rows use matched tight Powell tolerances.  Estimator accounting
retains every Hamiltonian/Powell occurrence and removes only repeated staged
gradient or metric primitives already available at the same mathematical
state.  Beam values are selected-lineage totals reconstructed from the
explicit branch ancestry in the estimator ledger.

| evaluated route | comparison prefix | \(\widehat N_{2q}/\widehat D_{2q}/\widehat D_c\) | \(\widehat K\) | corrected \(\mathcal E\) | status |
|---|---:|---:|---:|---:|---|
| AVQE | (k=8), first (\le10^{-6}) | 220/220/391 | 229.75 | 6510 | claim-ready comparator |
| repaired native adaptive RA-Append | (k=8), first (\le10^{-6}) | 92/92/203 | 105.95 | 5188 | best matched candidate; reproduce from committed source |
| repaired native adaptive beam | (k=8), first (\le10^{-6}) | 92/92/203 | 105.95 | 5198 | indistinguishable circuit proxy at target; no advantage over append |
| repaired native adaptive batch | (k=4), first (\le10^{-6}) | 204/204/375 | 216.15 | 3378 | fastest round convergence and lowest estimator count, but larger than RA-Append |
| current whitened RA-Append | (k=9), first (\le10^{-6}) | 284/284/495 | 294.15 | 6474 | current claim-facing bundle |
| current whitened beam | (k=9), first (\le10^{-6}) | 228/228/409 | 239.05 | 6476 | best current-whitened route |
| matched graph-selector whitened RA-Append | (k=9), first (\le10^{-6}) | 284/284/495 | 294.15 | 6378 | diagnostic; needs locked promotion |

At the user's intended display markers—batch at (k=7) and AVQE, RA-Append,
and beam at (k=9)—the later whitened routes behave differently:

| route | (k) | \(\widehat N_{2q}/\widehat D_{2q}/\widehat D_c\) | corrected \(\mathcal E\) |
|---|---:|---:|---:|
| AVQE | 9 | 228/228/409 | 6700 |
| whitened RA-Append | 9 | 284/284/495 | 6474 |
| whitened batch | 7 | 288/288/509 | 5712 |
| whitened beam | 9 | 228/228/409 | 6476 |
| whitened prune, first deletion | 10 | 284/284/495 | 7221 |

The whitened append route therefore saves (3.4\%\) estimator work but uses
(24.6/24.6/21.0\%\) more graph-proxy circuit burden than AVQE at (k=9).
Whitened beam exactly matches AVQE on all three graph coordinates, uses
(3.3\%\) fewer estimator calls, and reaches lower error.  If the manuscript
insists on the whitened refit chart, beam is the only present (k=9) route
that avoids a proxy-circuit regression.

The matched graph-span selector ablation does not recover the old v10 circuit
advantage: at (k=9) it has the same (284/284/495) graph coordinates as the
exact-Qiskit-selector whitened append route.  The graph proxy is therefore not
the cause of the historical v10 advantage by itself.

## Historical v10

Historical v10 remains useful as a mechanism diagnostic.  Its whitened
RA-Append prefix at (k=8) has (104/104/215) and corrected
(\mathcal E=3616), and its beam prefix has (92/92/203).  Those trajectories
were generated with fixed (24/12/1) shortlists, Phase 0 off, the older
five-coordinate graph-span penalty, and loose RA Powell tolerances against a
tight AVQE comparator.  They must not be selected as matched headline
evidence.  The repaired native route recovers and slightly improves their
compact trajectory under matched tight tolerances, so v10 no longer needs to
carry that evidentiary burden.

## Representative-selection rule

Use the following rule before updating Paper I:

1. Fix the physics, ordered 24-generator intact pool, exact-state backend,
   Powell tolerances, estimator convention, and accuracy target.
2. Keep native-coordinate and whitened-coordinate RA as different evaluated
   configurations.
3. Compare all target-satisfying prefixes on the Pareto axes
   \((\widehat N_{2q},\widehat D_{2q},\widehat D_c,\mathcal E)\); do not pick a
   route solely because one weighted scalar is favorable.
4. Prefer the earliest target-satisfying prefix unless a later prefix reduces
   a circuit coordinate through pruning.
5. Require a locked confirmation run for a diagnostic or historical winner.

Under that rule, the provisional main representative is repaired native
adaptive RA-Append at (k=8).  Repaired batch at (k=4) is the batching
representative.  Beam does not improve the native RA proxy at the first target
prefix, although it becomes slightly smaller by (k=9).  No pruning row
improves the first-target native append prefix; pruning should be shown only at
the first accepted-deletion marker and described as a size-control diagnostic.

## Files and reproduction

- Audit script: `audit_l3_representatives.py`
- Source and protocol inventory: `l3_run_inventory.json`
- Every recounted prefix: `l3_prefix_metrics.csv`
- Pareto views: `l3_pareto_frontiers.json`

Run from the `appendix-demos-20260820` worktree root:

```bash
python3 chtc/ra_adapt_l3_representative_audit_20260825/audit_l3_representatives.py
```

The script validates the recount against two locked checks:

- historical v10 RA-Append at (k=12): (114/118/247);
- AVQE at (k=12): (344/344/615).

No Paper-I source, figure, or result table was changed by this audit.
