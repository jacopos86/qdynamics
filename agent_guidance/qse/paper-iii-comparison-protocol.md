# Paper III — comparison protocol

How every reported comparison in this lane is constructed. The short version:
**methods are compared at fixed spectral accuracy on identical physics, while
record-based phases share one record alphabet and synthesized-direction methods
declare their different construction family; resources are reported on four
separate axes**. If a number in the manuscript cannot be traced to a cell built
this way, it is not evidence.

This is the Paper III analog of
`agent_guidance/time-dynamics/paper-ii-comparison-protocol.md` and inherits
its stance: fixed-accuracy comparison, locked axes, every rung reported,
failure states named rather than hidden.

---

## 1. Growth traces, not stopping rules (LOCKED, user 2026-08-26)

Every method grows the subspace one direction at a time to a fixed cap and
records, at each step k, the tuple

    (k, N2q, D2q, Dc, max_{nu in mathcal R} |Delta E_nu|).

The comparison is then read off the trace:

    K*(eps_E) = min{ cost over the trace : max_{nu in mathcal R} |Delta E_nu| <= eps_E }.

**No method's own stopping rule chooses the fixed-accuracy crossing.** The
record-selector receipt still determines the canonical method's phase
transition: `residual_converged` terminates record-first cost-weighted QSE,
whereas `pool_exhausted` preserves the exchange-maintained support and starts
Davidson residual synthesis. The reported K*(eps_E) is then read from the
resulting trace rather than from that internal stopping event.

The residual stop, pool-exhaustion receipt, handoff support, and adaptive stop
are all recorded. Exact-reference error is diagnostic and never controls a
stop or handoff.

## 1b. Where exchange enters (LOCKED)

Certified exchange is applied at every recorded support in the canonical
growth trace. The crossing-only variant remains a diagnostic policy. If record
growth exhausts the alphabet, residual synthesis is seeded from the actual
exchange-maintained terminal support, not from the unmaintained acquisition
order.

## 1c. Canonical completeness fallback (LOCKED, user 2026-08-27)

The production construction is `record_first_adaptive_qse_v2`:

1. acquire records with the two-term Delta/K score and certified exchange;
2. stop if the target-root record residual converges;
3. on `pool_exhausted`, preserve the maintained support and synthesize
   preconditioned Davidson residual directions;
4. stop residual synthesis at its declared residual target or dimension cap.

The fixed-alphabet selector is retained as `record_only` and is an ablation,
not the complete method. Retuning the record score after pool exhaustion is
forbidden: no score over a fixed exhausted alphabet can create a missing
direction.

## 2. The native protocol: fixed accuracy, compare resources

The production method takes a residual specification and dimension cap as
inputs and returns a support plus any synthesized continuation and their
resource bill as output. Its native
comparison is therefore

    K*(eps_E) = minimum cost at which a method's certified output
                reaches max_{nu in mathcal R} |Delta E_nu| <= eps_E,

evaluated per regime over a declared error-target ladder
eps_E in {1e-2, 1e-4, 1e-6}. Two failure states, always reported, never
blanked:

- `UNATTAINABLE_WITH_MANIFOLD` — the arm's alphabet/class cannot reach eps_E
  at any budget (e.g. the complete fixed class on a window it does not span).
  No resource ratio may be quoted against this cell.
- `NOT_REACHED_WITHIN_POOL` — an extendable construction did not reach eps_E
  within its declared dimension cap. Report the terminal error and cost.

Fixed-cost and fixed-iteration readings may be *derived* from the same traces
for presentation, but the primary claim axis is K*(eps_E). Prefix-path
("anytime") readings of the production method are NOT evidence: the method
emits certified endpoints, not orderings (established 2026-08-19 after a
mis-benchmark; do not repeat it).

## 3. Locked axes (fairness)

| axis | what is locked |
|---|---|
| Physics | Hamiltonian, sector, exact sector-restricted references, `mathcal R = {1, ..., 6}` window |
| Alphabet | **identical record pool for every record-based phase** (user directive 2026-08-19); direct Davidson, Krylov, and the canonical fallback synthesize directions outside it and must declare that asymmetry |
| Pencil | q0 projection, raw_projected normalization, shared overlap cutoff |
| Cost model | two_qubit_only_v1, graph-span oracle (transpile cross-check reported separately) |
| Selection knobs | production two-term score; arms may differ ONLY in their declared acquisition rule |

**MANDATORY REPORTING (user 2026-08-26): every reported cell carries all four
axes. Reporting one of them alone is a protocol violation** -- a compiled-gate
result is not a depth result and neither is a shot result.

1. `n2q` -- compiled two-qubit gate count (deterministic graph-span proxy);
2. `d2q` -- two-qubit depth (routed-chain proxy);
3. `dc`  -- total circuit depth (`d2q` plus the one-qubit layer channel);
4. `estimator` -- matrix-measurement work over the full (S,H) pencil:
   `qwc_groups` (measurement settings under qubit-wise-commuting basis cover),
   `distinct_words`, and `naive_terms`.

Axes 1-3 come from `PaperIIIProblem.resource_triple`; axis 4 from
`estimator_cost`. Tables may abbreviate as `n2q/d2q/dc` plus a separate
estimator column, but a cell that omits an axis is not evidence. Estimator
work is evaluated at crossing points (where K* is read off), not at every
trace step, for cost reasons; the artifact records which.

For adaptive constructions, admitted record seeds retain their actual compiled
record costs. Only residual-synthesized directions are charged as synthesized
Hamiltonian directions. Charging every seed as a Trotterized residual step is
invalid and supersedes the original v1 resource ratios.

## 4. Comparator admissibility

- External comparators must be faithful implementations of a published
  construction, cited to it. Configurations of our own selector are
  ABLATIONS: they appear in `tab:qse_ablation_matrix` only, never as
  comparator columns (no self-comparison; user directive 2026-08-19).
- `record_only` is the explicit fixed-alphabet ablation of canonical
  record-first cost-weighted QSE.
- Ablations whose outcome is already established (e.g. cost term on/off) are
  not re-reported as comparisons.
- Non-record methods (real-time Krylov) do not consume the alphabet; they are
  benchmarked on the same physics and costed by the same oracle, and labeled
  as a different construction family.

## 5. Ingredient necessity (what is necessary vs unnecessary)

Run `paper_iii_qse_score_ablation.py` at the production residual stop, one
ingredient disabled per arm, support size as an output. Verdicts, per
ingredient, over the regime set:

- `NECESSARY` — removal prevents certification (pool_exhausted) OR degrades
  K*(eps_E) by more than the declared margin (2x cost or 10x error) in at
  least one non-pool-limited regime;
- `UNNECESSARY` — outputs bit-identical or within margin in every
  non-pool-limited regime; the ingredient is then REMOVED from the production
  rule (not zero-weighted) and the manuscript states the removal in one
  sentence without a rejected-variant table;
- `REGIME-CONDITIONAL` — helps in some non-pool-limited regimes and hurts in
  others; reported explicitly, defaults chosen by total resource over the
  regime set.

Pool-limited regimes (every arm exhausts the pool within 10% of one error)
carry no ablation signal and are excluded from verdicts, but shown.

Current verdicts (evidence: score_ablation.json, minimality.json):
metric-novelty weight NECESSARY; residual capture NECESSARY; cost discount
NECESSARY; metric-novelty floor NECESSARY (strong_strong_u8); Ritz gain
UNNECESSARY (regime-conditional, net negative; removed); conditioning penalty
UNNECESSARY (bit-identical; removed); transition visibility UNNECESSARY
(bit-identical; removed).

## 5. Selection-bias caveat (inherited from Paper II)

Where a ladder is searched (residual-stop rungs for our arm; prefix extension
for ordering controls), the selected minimum-cost target-reaching rung is an
exact-reference-tuned per-cell diagnostic, not an operating rule. Report every
evaluated rung; label the selected point; never quote a best-rung ratio as a
headline figure.

## 5b. Shared problem provider (architecture, user-directed 2026-08-26)

Every route obtains its Hamiltonian, operator alphabet, compiled costs, and
exact reference from ONE provider:
`pipelines/qse_spectra/paper_iii_problem.py::load_problem`. Building physics
inside a driver is a protocol violation -- it was how the campaign came to
build the same pool twice per regime with nothing proving the two matched.

The provider guarantees, and every artifact records via
`problem.arm_receipt()`:

- **one alphabet per regime**, handed out by reference (in-process cached);
- **`pool_digest`** -- ordered digest of record names. Equal pool SIZES do not
  prove equal pools; a shared-alphabet claim requires this digest present and
  equal in every arm receipt of a comparison;
- **`reference_key`** -- the content-addressed exact-reference identity;
- **uniform granularity** -- `assert_uniform_granularity` fails closed if a
  pool mixes macro records with pre-split Pauli children. The selection route
  never decomposes a record between phases (Phase I/II/III and exchange all
  treat records as opaque), so a mixed pool would make "one record" mean two
  different resource quantities. Macro-to-singleton decomposition anywhere
  inside a route is forbidden.

## 6. Cached exact references (Paper-II lesson, user-directed)

The exact sector-restricted reference (ground vector + lowest R+1 energies)
is computed ONCE per regime and stored content-addressed under
`output/reference_store/paper_iii_exact_sector/`, keyed by the full physical
identity (family, u, g_ep, n_ph_max, sector, count). Every arm, rung, and
rerun reads the same cached entry; the store verifies identity on read and
never silently overwrites a mismatched entry. Recomputing an identical
reference per run is a protocol violation.

## 7. Displays

Principal displays are physics: the excitation-gap ladder per method
(Fig. gap) and the accuracy--cost frontier. Matched-error curves are not
principal panels; matched-accuracy results are TABULAR (K*(eps_E) cells
with failure states).

## 8. Coded entry point

`pipelines/exact_bench/paper_iii_growth_trace_campaign.py` builds the current
fixed-accuracy cells;
`pipelines/reporting/build_paper_iii_growth_fragments.py` renders them. A
manuscript comparison table not generated from that campaign's JSON is not
evidence.

## Methods text must describe the implementation (2026-08-26)

The manuscript carried a three-phase staged-shortlist architecture that the
production selector never ran: `geometry_selected` scores every admissible
candidate every round in one pass. That discrepancy stood because the phase
text was written as a design and the code was built separately. It is now
resolved -- the methods section describes the one-pass rule.

Standing rule: **a methods claim about record acquisition must be checkable
against `pipelines/qse_spectra/record_selection.py`, and a claim about the
completeness fallback against
`pipelines/qse_spectra/adaptive_qse_benchmark.py`.** In particular, do not
claim selection-time measurement savings from staging unless a staged screen
is actually implemented. The reported estimator advantage comes from support
size (fewer directions imply fewer pencil elements), not from shortlisting.
