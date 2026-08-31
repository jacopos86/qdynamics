# Paper II t=20 results replacement contract

Status: active manuscript-promotion contract  
Target source: `MATH/paper_details/time_dynamics_paper_II.tex`  
Target PDF: `MATH/paper_details/time_dynamics_paper_II.pdf`

## Purpose

The active manuscript still contains several provisional result families from
older horizons, numerical settings, structural policies, and incomplete
comparisons. They are superseded as scientific evidence by the staged
matched-accuracy campaign. The four-page campaign report is an author-facing
frontier and evidence report; it is not the manuscript PDF.

The manuscript Results section must be regenerated from accuracy-qualified
full-horizon campaign records rather than updated by accumulating another table
beside the old ones.

## Evidence admission

A run enters the manuscript replacement only when all of the following hold:

1. the `t=5` screen and decision-identical `t=5`-to-`t=20` continuation have
   passed artifact, support-boundary, run-lock, and terminal-cost parity checks;
2. its achieved seed, drive, grid, candidate pool, controller, integrator,
   inverse convention, and compile target match the campaign declaration;
3. its four componentwise errors were computed from the stable cached exact
   trajectory for that physical seed and drive;
4. every error is at most `1e-4`;
5. any compared AVQDS, full-APM, and APM insertion-face arms pass the gates
   independently; and
6. the source ledger, exact-reference artifact, report manifest, and generated
   table/figure inputs carry matching digests.

`gate_pending`, `requires_threshold_extension`, safeguard, failed, diagnostic,
or invalidated trajectories remain visible in the campaign report but never
enter manuscript values or claims.

### Explicit author override

The author may explicitly direct a manuscript refresh before the
exact-reference gate is evaluated.  That instruction admits only trajectories
that have already passed artifact, run-lock, resume-support, and terminal-cost
parity checks.  Such a refresh must label every admitted row and dependent
claim as `author_override_gate_pending`, must omit exact-reference and
matched-accuracy claims, and must carry a digest-bound receipt naming the
override.  It does not change the default admission rule for later results and
is superseded when the stable exact-reference gate is evaluated.

For the August 26 author-directed replacement, the exact-reference errors are
now computed over the full (0\le t\le20) grid, but categorical admission is
not allowed to hide those verified rows.  The manuscript reports the achieved
errors without relabeling a failing component as passing.  This instance is
recorded by the digest-bound promotion receipt.

## Existing blocks to supersede

When the corresponding new evidence is admitted, replace rather than retain:

- the `t=5` three-seed production matrix and its supporting interpretation;
- the fixed-threshold `t=10` frozen-comparator and matched-comparison tables;
- the six preliminary per-drive doublon-error figures;
- the provisional `t=10` integrator table in the main Results section;
- the one-drive insertion-only-debt comparison, figures, and stabilization
  narrative;
- the speculative factorial-interaction paragraph;
- the four placeholder Qiskit-community regime tables;
- the empty append-only-versus-SNAKE seed-track table; and
- the evidence-scope paragraph that says trajectories terminate at `t=1`.

Historical source artifacts remain in Git history and the evidence folders.
They are not deleted merely because the visible manuscript no longer cites
them.

## Replacement result spine

The new Results section contains only:

1. the fixed-accuracy protocol and locked shared numerical configuration;
2. one readable `t=20` row-per-trajectory table for every currently evaluated
   AVQDS, full-APM, and insertion-McLachlan (IM) run;
3. one representative three-panel physics figure showing total energy
   `E(t)`, mean doublon occupation `D(t)`, and charge imbalance `Delta n(t)` for
   the exact reference and every displayed compared arm; full APM is solid
   dark red, AVQDS is dashed blue, and the exact reference is dotted black;
4. in each row, the full-horizon mean infidelity, accepted integration-step
   count, terminal active-coordinate count, and routing-independent terminal
   graph tuple only;
5. a compact scope statement naming the drives/controllers actually admitted.

Componentwise energy and observable errors, log-error curves, and matched-error
curves are qualification diagnostics rather than principal physics panels.
Cumulative estimator-work accounting is postponed and does not block this
replacement.

The graph tuple, mean infidelity, and step count must cover the same identified
run.  The
activation cut is recorded in provenance but omitted from the reader-facing
table. Compiled
cost, continuation-only right-hand-side counts, and continuation-only
structural events do not appear in this main table.

## Atomic promotion

Every ledger mutation or gate decision first refreshes the canonical campaign
report. When a new `t=20` row or comparison becomes manuscript-promotable, the
same workflow must then:

1. generate digest-bound manuscript table and figure inputs;
2. update the source map and claim ledger;
3. replace the superseded manuscript blocks and dependent prose;
4. rebuild `time_dynamics_paper_II.pdf` from the active source;
5. render and inspect every changed page; and
6. record the source, PDF, ledger, exact-reference, table-input, and figure-input
   digests in one promotion receipt.

If manuscript rebuilding or visual verification fails, the promotion is
incomplete and the previous manuscript PDF must not be described as current.
The campaign may continue gathering independent cells, but no failed promotion
may be silently skipped.
