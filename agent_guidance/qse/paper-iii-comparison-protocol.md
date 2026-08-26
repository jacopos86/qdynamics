# Paper III — comparison protocol

How every reported comparison in this lane is constructed. The short version:
**methods are compared at fixed spectral accuracy on an identical record
alphabet, and resources are reported on three separate axes**. If a number in
the manuscript cannot be traced to a cell built this way, it is not evidence.

This is the Paper III analog of
`agent_guidance/time-dynamics/paper-ii-comparison-protocol.md` and inherits
its stance: fixed-accuracy comparison, locked axes, every rung reported,
failure states named rather than hidden.

---

## 1. The native protocol: fixed accuracy, compare resources

The production method takes an accuracy specification as INPUT (the residual
stop) and returns a support and its resource bill as OUTPUT. Its native
comparison is therefore

    C*(eps_E) = minimum resource at which a method's certified output
                reaches max_{nu<=R} |Delta E_nu| <= eps_E,

evaluated per regime over a declared error-target ladder
eps_E in {1e-2, 1e-4, 1e-6}. Two failure states, always reported, never
blanked:

- `UNATTAINABLE_WITH_MANIFOLD` — the arm's alphabet/class cannot reach eps_E
  at any budget (e.g. the complete fixed class on a window it does not span).
  No resource ratio may be quoted against this cell.
- `NOT_REACHED_WITHIN_POOL` — an adaptive arm exhausted the shared pool
  before reaching eps_E. Report the terminal error and cost.

Fixed-cost and fixed-iteration readings may be *derived* from the same traces
for presentation, but the primary claim axis is C*(eps_E). Prefix-path
("anytime") readings of the production method are NOT evidence: the method
emits certified endpoints, not orderings (established 2026-08-19 after a
mis-benchmark; do not repeat it).

## 2. Locked axes (fairness)

| axis | what is locked |
|---|---|
| Physics | Hamiltonian, sector, exact sector-restricted references, R=6 window |
| Alphabet | **identical record pool for every record-based arm** (user directive 2026-08-19); per-arm private alphabets are invalid |
| Pencil | q0 projection, raw_projected normalization, shared overlap cutoff |
| Cost model | two_qubit_only_v1, graph-span oracle (transpile cross-check reported separately) |
| Selection knobs | production two-term score; arms may differ ONLY in their declared acquisition rule |

Resource axes are reported separately and never conflated (a compiled-gate
result is not a shot result):

1. compiled two-qubit gates of the record measurement circuits (`total_2q`);
2. measurement settings: QWC basis-cover groups over the full (S,H) pencil
   (`qwc_groups_total`), with distinct-word and naive-term counts;
3. support size k and retained rank.

## 3. Comparator admissibility

- External comparators must be faithful implementations of a published
  construction, cited to it. Configurations of our own selector are
  ABLATIONS: they appear in `tab:qse_ablation_matrix` only, never as
  comparator columns (no self-comparison; user directive 2026-08-19).
- Ablations whose outcome is already established (e.g. cost term on/off) are
  not re-reported as comparisons.
- Non-record methods (real-time Krylov) do not consume the alphabet; they are
  benchmarked on the same physics and costed by the same oracle, and labeled
  as a different construction family.

## 4. Ingredient necessity (what is necessary vs unnecessary)

Run `paper_iii_qse_score_ablation.py` at the production residual stop, one
ingredient disabled per arm, support size as an output. Verdicts, per
ingredient, over the regime set:

- `NECESSARY` — removal prevents certification (pool_exhausted) OR degrades
  C*(eps_E) by more than the declared margin (2x cost or 10x error) in at
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
principal panels; matched-accuracy results are TABULAR (C*(eps_E) cells
with failure states).

## 8. Coded entry point

`pipelines/exact_bench/paper_iii_matched_accuracy_campaign.py` builds the
cells; `build_paper_iii_tex_fragments.py` renders them. A manuscript
comparison table not generated from that campaign's JSON is not evidence.
