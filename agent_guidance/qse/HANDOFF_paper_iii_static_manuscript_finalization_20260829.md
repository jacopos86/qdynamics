# Handoff — finalize Paper III as a static spectral-QSE paper

## Objective

Finish the Paper III manuscript and compiled PDF as a static and
frequency-domain QSE paper. Remove the time-dynamics paper from this
manuscript rather than leaving it as deferred placeholders or a secondary
claim. Integrate the locked 12-cell selected-QSE results, static transition
strengths, spectra, and resource comparisons from the production evidence.

Work directly on branch `paper-ii-exchange-selector` in
`/Users/jakestrobel/local_repos/Holstein_test_fullclone_3`. The results and
algorithm decisions are settled; begin the manuscript work without another
design questionnaire.

## Decisions already made

- Paper III contains static/frequency-domain QSE only. Time propagation,
  frozen-QSE dynamics, adaptive-QSE dynamics, escape diagnostics, and live
  McLachlan handoff belong to a later paper.
- Canonical selected QSE uses full proposed-pencil Ky Fan gain divided by
  incremental cost, a novelty admissibility floor without an additive novelty
  reward, the encoded-boson completion pool, and certified atomic `1<->1`
  exchange. It has no implicit Davidson fallback.
- Direct Davidson is the independently initialized energy/cost comparator.
  Complete fixed linear-response QSE is the conventional static spectral
  baseline. Do not retain real-time Krylov merely to fill a comparator column.
- Notation is root $r$, target set $\mathcal R$, gain $\Delta$, cost $K$,
  score $\mathcal S=\Delta/K$, and selected basis $\mathcal B$. Replace cost
  symbol $C$ where it denotes this acquisition cost.
- The exchange scope is an atomic singleton delete-add proposal (`1<->1`),
  solved and certified as one transaction. Do not imply multi-record patches.
- The benchmark uses exact sector ground states to isolate QSE. The general
  method may accept an optimized prepared seed, but the 12 result rows must
  not be described as imported Paper-I variational seeds.

Authoritative evidence:

- `agent_guidance/qse/EVIDENCE_paper_iii_pool_profile_ablation_20260829.md`
- `agent_guidance/qse/EVIDENCE_paper_iii_static_production_20260829.md`
- `output/pdf/paper_iii_static_qse_production_20260829_v1/`

## Authority to implement and repair

Edit the Paper III LaTeX, generated Paper III table/figure fragments, captions,
cross-references, title, abstract, introduction, methods, results, conclusion,
and appendices as needed. Repair ordinary build, layout, stale-reference, and
generated-asset plumbing problems and continue. Preserve the locked algorithm
and evidence identities rather than reviving obsolete hybrid or geometry-score
text.

## Constraints

- Delete or relocate all dynamics material from Paper III; do not summarize it
  as an unvalidated contribution.
- Replace the old novelty-plus-residual score and pool-exhaustion Davidson
  story with the locked full-pencil Ky Fan-per-cost selector and completion
  alphabet.
- Do not claim universal cost dominance: selected QSE wins 10/12 cells at
  $10^{-4}$ and 12/12 at $10^{-6}$. Name the two $10^{-4}$ losses.
- Incorporate the static spectral result: selected median normalized spectral
  $L^1=1.73\times10^{-9}$ and maximum $2.00\times10^{-5}$ across 24 spectra.
- Keep the essential QSE generalized-eigenproblem definition in the main
  methods. Move whitening thresholds, conditioning diagnostics, perturbation
  bounds, and other numerical implementation detail from the former Section H
  to a compact appendix; remove ridge language unless an executed result uses
  it.
- Remove stale Peierls, SIAM, L=3, production-dynamics, and placeholder claims
  unless supported by a current admitted artifact actually used in the paper.
- Preserve supported-overlap truncation, retained-rank checks, double
  Gram--Schmidt, retained-frame novelty, and atomic exchange certification.
- Before each commit run `git pull --rebase`; if the shared dirty checkout
  prevents it, fetch the branch and verify divergence instead. Stage explicit
  Paper III paths only. Never use `git add -A`, stash, force-push, or disturb
  other sessions' changes.

## Definition of done

- The title, abstract, introduction, methods, results, conclusion, figures,
  tables, appendices, and bibliography tell one coherent static-QSE story.
- No dynamics section, dynamics claim, dynamics placeholder, or real-time
  comparator remains.
- Every quantitative claim is traceable to the locked production evidence.
- The notation audit passes for $r$, $\mathcal R$, $\Delta$, $K$,
  $\mathcal S$, and $\mathcal B$.
- The final PDF compiles without placeholders, broken references, overflow,
  clipping, or unreadable figures and is visually inspected page by page.
- Commit and push the manuscript increment, reporting the exact branch,
  commit, tests/build command, and final PDF path.

## True stop conditions

Stop only for a newly discovered scientific contradiction in the locked
artifacts, irrecoverably missing manuscript inputs, or a user-authority issue
that changes the paper's scope. Ordinary prose, notation, LaTeX, generated
asset, and build defects are repair work, not blockers.
