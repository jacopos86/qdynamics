# Paper III static-QSE results-acquisition contract

Date: 2026-08-27
Branch: `paper-ii-exchange-selector` (scientific anchor `e6b2c2ca`)
Governing handoff: `prompt-exports/HANDOFF_paper_iii_static_qse_results_pipeline_and_siam_framing_20260827.md`
Status: binding operating contract for the typed static-QSE campaign,
ledger, packaging, ingestion, qualification, and results contract.

## Scope

Paper III is **ground-state preparation followed by static and
frequency-domain QSE results**: excitation energies, transition strengths,
Lehmann Green functions, spectral functions, spectral checks, and resource
comparisons. Time propagation is out of scope for this route. The Paper II
adaptive-dynamics method and the frozen-QSE-to-live-dynamics interface exist
and are validated; their exclusion here is sequencing, not maturity.

## Canonical method and ideal objective

The canonical route is **record-first selection with certified atomic
exchange and Davidson residual completion on pool exhaustion**, public id
`record_first_adaptive_qse`. `record_only` is an ablation, not a second
public algorithm; `early_switch` may run only as an internal ablation and
cannot be promoted from post-run exact data.

The ideal measurable combinatorial objective is:

```text
B* in argmin_{B subset P} K(B)
subject to retained_rank(B) >= |mathcal_R|,
           max_{r in mathcal_R} ||H Psi_r(B) - epsilon_r(B) Psi_r(B)|| <= eps_res,
           and the declared overlap/rank guards.
```

Exact energy error is **not** a constraint of the online objective. The
implemented approximation is greedy: each record acquisition maximizes
`S_i = Delta_i / K_i`, where `Delta_i` is the production two-term
combination of retained-frame metric novelty and target-root residual
capture, and `K_i` is the deterministic graph-span compiled cost
(preset `two_qubit_only_v1`, within-pool normalization, `alpha = 1`,
denominator floor `0.05`). Load-bearing invariants preserved verbatim:

- double Gram--Schmidt projection pass;
- novelty measured against the solver-retained overlap frame;
- the candidate Rayleigh value feeding the pool-relative window check;
- true atomic `1<->1` exchange with whole-patch pencil certification
  (never prune-then-append);
- Davidson residual synthesis starts **only** on `pool_exhausted`, seeded
  by the exchange-maintained terminal support.

Exact spectra qualify and report completed runs; they never select a
record, trigger a fallback, or stop an online algorithm.

Notation: root `r`, target set `mathcal_R`, gain `Delta`, cost `K`,
score `S = Delta/K`, selected basis `mathcal_B`.

## Typed identities

Implemented in `pipelines/qse_spectra/static_campaign.py`. Every id is the
SHA-256 of canonical JSON (sorted keys, compact separators) and stable
across replay.

- `StaticQSEPhysicsKey`: model family, case id, Hamiltonian parameters,
  register layout, conserved sector, cutoff (`n_ph_max` for HH, `n_bath`
  for SIAM).
- `GroundStateArtifactRef`: provider id, manifest path + sha256, state key,
  state payload sha256, qubit count, sector, energy, source revision,
  achieved-settings digest. A path alone is not provenance; states load
  through `pipelines.qse_spectra.io.load_state_json`.
- `StaticQSECellKey`: physics + ground-state artifact id + public method id
  + target set + dimension cap + pool-policy digest + spectral-request
  digest + resource-policy id + **method-policy digest** (the online switch
  rule). **No accuracy target in the identity** (one trace is read post-run
  at `eps_E in {1e-2, 1e-4, 1e-6}`); **no internal phase in the identity**
  (record phase and Davidson phase are stages of one construction).
- `MethodPolicy` (`pipelines/qse_spectra/switch_policy.py`): the declared
  online switch rule. Canonical is `pool_exhausted_only` with no
  parameters. `window_residual_stall` (parameters `stall_rounds`,
  `improvement_ratio`) escalates earlier and is **ablation-only**: a public
  method id carrying a non-canonical policy is refused at construction.
  Changing the switch rule therefore declares *new cells* instead of
  reinterpreting admitted evidence under a rule it never ran; promoting a
  trigger means changing `CANONICAL_METHOD_POLICY`, which re-digests the
  public cells by design.

  **Switch-input boundary (load-bearing):** a trigger predicate reads only
  the selector's online round trace (`max_target_residual_norm`,
  `selected_score`, `window_pressure`, novelty/capture). Exact spectra and
  any window error computed against them are post-run qualification data.
  `assert_online_only_receipt` enforces this structurally by refusing any
  receipt key outside the declared online vocabulary. The stall rule
  compares against a *running best*, not the previous round, because the
  online residual is not monotone (observed on SIAM `N_b=2`: round 5
  `0.531` rises to round 6 `0.717`).
- `StaticQSEAction`: cell + stage + dependency action ids.
- `StaticQSEAttempt`: achieved configuration, source/input/output digests,
  payload references, process disposition, optional qualification.

Public method ids: `record_first_adaptive_qse` (canonical),
`direct_davidson_qse`, `real_time_krylov`, `fixed_linear_response_qse`.
Legacy `record_first_adaptive_qse_v2` artifacts are accepted through an
ingestion alias and normalized; old evidence is not rewritten and no public
v2/v3 algorithm exists.

## Stage graph

```text
ground_state_import
        v
static_qse_trace
   |       |        |
   v       v        v
transitions   green_function   resource_evaluation
   |              |                  |
   +--> spectral_postprocessing <----+   (resource axis feeds qualification)
                     v
              qualification
                     v
           ledger admission -> results contract/PDF
```

`ground_state_import` validates an existing prepared-state artifact and
never reruns Paper I. `static_qse_trace` produces roots, retained-frame
data, support/exchange telemetry, and the complete growth trace. Transition,
Green-function, spectral, and resource stages are post-QSE products and are
separate scheduler actions. `qualification` depends on
`spectral_postprocessing` and `resource_evaluation` because the
qualification record must contain all four resource groups.

**Visibility boundary:** one completed action is independently ingestible as
soon as its own dependencies exist. Never wait for a sibling action, a whole
cluster, a complete method tuple, or a passing gate before admitting it.

## Status vocabulary

- `qualified`: completed, achieved-configuration validated, numerically
  valid, within the declared qualification policy;
- `candidate`: completed and numerically valid but outside one or more
  gates -- values stay intact and visible;
- `pending`: no valid completed attempt exists;
- `gate_pending`: a verified numerical artifact exists but classification is
  temporarily blocked (report state, not an attempt type);
- `not_planned`: a declared matrix slot intentionally not scheduled.

A failed gate never removes numerical values or becomes `pending`. Ledger
admission is achieved-configuration-validated and idempotent; the same
action id with different achieved settings or output digests is a hard
conflict; re-qualification over identical stored outputs appends.

## Declared matrices

**Hubbard--Holstein (first complete matrix):** the six established regimes
(`weak_weak`, `intermediate_weak`, `strong_weak_u8`, `weak_strong`,
`intermediate_strong`, `strong_strong_u8`) at `n_ph_max in {1, 3}`, target
roots `mathcal_R = {1..6}`, all four public methods. Canonical cap 64;
Krylov cap 40. `record_only` appears only in the ablation view. Peierls,
L=3, and dynamics artifacts enter later through explicit legacy/source-map
roles.

**SIAM (additional benchmark, not a novelty claim):**

- fixture: `N_b=2`, `U=1.5`, `V=1/sqrt(2)`, `t_b=1`, spin-blocked JW,
  sector `(N_up, N_down) = (2, 1)` (6 qubits, sector dim 9);
- pilot: `N_b=4`, same `U, V, t_b`, sector `(3, 2)` (10 qubits, dim 100).

Method slots: canonical record-first and direct Davidson. Real-time Krylov
joins only if the generic adapter is faithful without SIAM-specific policy.
The fixed linear-response slot is `not_planned` until a scientifically
defined class exists -- never fabricated for rectangularity.

The spectroscopy Hamiltonian is the interacting `V=1/sqrt(2)` Hamiltonian
and the production seed must be its declared ground-state artifact. The
Paper II decoupled bath/quench state is **forbidden** as the spectroscopy
ground state (rejected structurally by
`pipelines/qse_spectra/siam_static_adapter.py`). The bounded smoke may use
the clearly labeled exact-sector diagnostic fixture; it is never
paper-facing evidence.

SIAM scientific layer provenance: `src/quantum/siam_{model, observables,
pool, reference}.py` ported verbatim from
`codex/paper-ii-search-profiles-20260824` commit `da7b26f3`; commit
`157f7176` is a behavior/provenance reference only.

## Online cost K and reported resource axes

Online `K` stays the deterministic Marrakesh graph-span compiled cost,
preset `two_qubit_only_v1`, within-pool normalization, `alpha=1`, floor
`0.05` (supported for the HH pool by identical selected supports under full
transpilation and Spearman `0.9999`; not re-run as a blocker). For SIAM,
`graph_span_compiled_cost_receipt` provides a bounded offline
ranking-stability receipt (full `N_b=2` pool; deterministic
family-stratified `N_b=4` sample) recording Spearman correlation,
top-decile overlap, backend/transpiler seed, and every compared record. Low
agreement leaves SIAM rows `candidate` and requires an explicit policy
decision before production; the receipt never changes the online cost and
never queries exact spectra.

Every completed result reports **separately** (never one unlabeled cost):

1. analytic/operator proxy fields;
2. deterministic `N2q`, `D2q`, total depth `Dc`;
3. Qiskit-compiled two-qubit count and depth when requested;
4. full-pencil estimator work: pair count, naive terms, distinct Pauli
   words, QWC groups/queries.

## Qualification policy

Typed, serialized, replayable over stored numerical attempts without
rerunning the solver. Required content: ground-state source/sector/qubit
validation and achieved norm; selected support, pool digest, retained and
discarded rank, overlap eigenvalue range/condition estimate, exchange-patch
telemetry; QSE root energies and generalized Ritz residuals; post-run
target-root matching, assignment, degeneracy handling, missing-root count,
principal angles where needed, exact energy errors; transition
amplitudes/strengths with observable identities; Green-function
addition/removal poles, residues, zero-source sectors, grid, Lorentzian
`eta`, anticommutator/source-norm sum-rule deficits; broadened spectral
values, non-negativity tolerance, retarded causality (`Im G^R <=
tolerance`), integrated weight, declared moments, window metrics, optional
reference errors; HH cutoff-boundary diagnostics where applicable; and all
four resource groups.

Structural invalidity (wrong sector, digest, achieved method, grid, missing
mandatory payload) **rejects admission**. A valid numerical result failing
an accuracy, residual, sum-rule, positivity, causality, rank, or
conditioning gate is admitted as `candidate` with values intact.

## Comparator and ablation rules

The HH paper-facing comparator matrix contains only the four public
methods. Record-first and fixed-class record phases must carry the same
ordered pool digest; Davidson and Krylov synthesize outside the alphabet
and must declare the construction asymmetry. Cost-off selection,
crossing-only exchange, record-only growth, and early switching are never
presented as competing algorithms. Settled score-term ablations are
ingested as legacy evidence, not rerun.

## Behavior locks

- The `nph1` growth-trace support sequence: all 93 acquisition-prefix
  supports bit-identical. Any support change is a scientific event -- stop
  and report, never retune.
- Exhaustion-only Davidson handoff stays canonical.
- The canonical results artifact consumes the immutable ledger; manuscript
  assets never rediscover raw runs.

## Implementation surfaces

- `pipelines/qse_spectra/switch_policy.py` -- declared online switch
  policy, its triggers, and the online-only input guard.
- `pipelines/qse_spectra/static_campaign.py` -- identities, stage graph,
  statuses, planner, append-only ledger, declared matrices.
- `pipelines/qse_spectra/static_campaign_artifacts.py` -- result envelope,
  achieved-configuration verification, exact-subset ingestion.
- `pipelines/qse_spectra/static_campaign_package.py` -- deterministic local
  action package and command manifest (packaging, not remote submission).
- `pipelines/qse_spectra/siam_static_adapter.py` -- SIAM physics keys,
  interacting Hamiltonian/pool, ground-state semantics, cost receipt.
- `pipelines/exact_bench/paper_iii_static_qse_cell.py` -- one-action
  producer CLI delegating to the existing scientific implementations.
- `pipelines/reporting/paper_iii_static_results_contract.py` -- ledger-to-
  matrix projection, schema validation, sole verified manuscript-row
  accessor (does not edit the manuscript).
