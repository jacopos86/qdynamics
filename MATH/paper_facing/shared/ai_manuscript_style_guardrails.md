# AI Manuscript Style Guardrails

Created: 2026-05-09  
Scope: Future AI-assisted Paper I / Paper II / Paper III drafting and review.

## Hard rules

1. Do not write generic background paragraphs. Every paragraph must do a technical job.
2. Do not overuse "novel," "framework," "pipeline," "robust," "seamless," "leverage," or vague "we propose" language.
3. Every paragraph must be one of: problem, prior limitation, method object, decision rule, benchmark evidence, limitation, or conclusion.
4. Distinguish draft-mode from final-polish mode. In draft-mode, keep `\placeholder{...}`, `\tentative{...}`, red result text, and concise structural labels when they serve as insertion slots or editing handles. Structural paragraph labels may render as red drafting placeholders when the author wants them visible; do not blend labels such as "Problem Hamiltonian" into reader-facing prose unless the user explicitly asks to convert them into real headings.
5. Keep Paper I, Paper II, and Paper III asymmetric:
   - Paper I: ADAPT ansatz construction.
   - Paper II: checkpoint-adaptive live dynamics.
   - Paper III: spectra, QSE excitation manifolds, transition observables, and driven excited-state response.
6. Use exact comparator names: ADAPT-VQE, qubit-ADAPT-VQE, QEB-ADAPT-VQE, Overlap-ADAPT-VQE, TETRIS-ADAPT-VQE, CEO-ADAPT-VQE, Geo-ADAPT-VQE, QSE, qEOM, q-sc-EOM, VQD, MC-VQE, SSVQE, pVQD, adaptive pVQD, AVQDS, AVQDS(T), product formulas, qDRIFT.
7. Avoid "better ADAPT" and "adaptive dynamics is new."
8. Use the stronger boundaries:
   - Paper I: candidate-position acquisition under geometry and cost.
   - Paper II: bidirectional checkpoint-local manifold maintenance.
9. In equations and method prose, use `MATH/Math.md` as the authority.
10. In result prose, cite our figures, tables, benchmark protocols, and ablations.
11. In related-work prose, cite external literature and distinguish overlap precisely, but do not force a related-work rewrite during a sentence-level editing pass.

## Author-control rule for framing

Future agents must not tell the author how to "frame" a paper unless the user
explicitly asks for framing, venue positioning, abstract/introduction strategy,
or final-polish review. When the user is constructing a method section,
appendix, settings inventory, or source-of-truth implementation specification,
the agent's job is to check mathematical consistency, notation, active/default
algorithmic choices, and implementation-contract completeness.

Do not inflate a method-spec review into caveats about publishability, current
results, evidence maturity, or journal positioning. Results may be absent,
deleted, provisional, or intentionally out of scope while the method is being
specified. In that mode, focus on whether the declared equations and runtime
contracts are internally coherent and implementable.

Avoid contrastive AI prose in recommendations and manuscript snippets. Do not
default to "not X but Y," "rather than X," "unlike X," or long caveat ladders.
State the active method object directly.

## Terminology discipline: ADAPT language vs. control language

Do not let the manuscript drift from **variational ansatz construction** into unnecessary **closed-loop decision-system jargon**.

Canonical ADAPT vocabulary is:

- operator pool;
- ansatz;
- adaptive sequence;
- operator selection;
- parameter optimization.

Standard ADAPT can be summarized as greedy variational construction:

\[
\mathcal O_{k+1}=\mathcal O_k\cup\{A_{j^\star}\},
\qquad
\theta^\star=\arg\min_\theta E(\theta).
\]

That is not usually called a controller. It is operator selection followed by reoptimization. In Paper I, prefer external manuscript language such as **adaptive ansatz**, **operator sequence**, **candidate-position selection**, **operator support**, **generator sequence**, and **variational manifold**. Do not use **scaffold** in reader-facing Paper I prose.

Control language becomes appropriate only when Paper II explicitly defines sequential structural actions. There the state and action can be described schematically as

\[
x_k=(\mathcal O_k,\theta_k),
\qquad
u_k\in\{\mathrm{stay},\mathrm{append},\mathrm{prune},\mathrm{branch}\},
\qquad
x_{k+1}=F(x_k,u_k),
\]

or as a decision policy

\[
\pi:\mathcal D_k\mapsto u_k,
\]

where \(\mathcal D_k\) are checkpoint diagnostics. After that definition, **controller** is acceptable for the checkpoint decision policy. Before that definition, prefer **adaptive McLachlan evolution**, **dynamic ansatz refinement**, **checkpoint decision policy**, or **append--prune update rule**.

Practical rule for future writing agents:

- Paper I: use canonical ADAPT/ansatz language first; avoid branding the static selector as a generic controller.
- Paper II: use controller language only for the formal checkpoint policy that chooses stay/append/prune/repair/branch actions.
- In Papers I and II: avoid jargon inflation. If "controller" appears, ask whether **operator sequence**, **adaptive ansatz**, **generator support**, or **variational manifold** would sound more canonical. In Paper I, delete or replace "scaffold" rather than defining it.
- In Paper III: prefer **selected excitation manifold**, **nonorthogonal QSE basis**, **spectral manifold**, **transition strength**, **overlap metric**, and **root tracking**. Use "controller" only for an explicitly defined admission or checkpoint policy.

## Paragraph test

Before accepting a paragraph, ask:

- What is the paragraph's technical job?
- Which claim-source bucket is used?
- What is the evidence or definition anchor?
- Does the paragraph contain repo-native language?
- Does it blur Paper I/Paper II/Paper III scope?
- Could a reviewer identify the comparator and metric?

If any answer is unclear, rewrite.

## Preferred sentence shape

Use concrete objects and verbs:

- "The selector ranks candidate-position records by ..."
- "The adaptive ansatz is represented as an ordered generator sequence ..."
- "After defining the checkpoint policy, the controller admits an append only when ..."
- "The prune lane stores an exact rollback point but does not use exact reference trajectories for decisions."
- "The benchmark reports trajectory error against accumulated compiled two-qubit burden."

Avoid empty claims:

- "We propose a novel framework that leverages a robust pipeline."
- "This approach is better than ADAPT."
- "The method is hardware-ready."
- "The paper should be framed as ..."
- "This is not X but Y ..."
- "Rather than X, we ..."

## Related-work density

Do not cite every sentence. Cite method families at first use, then use boundary sentences:

- Literature-backed claim: cite prior work and state distinction.
- Design/formalism claim: cite `MATH/Math.md` or manuscript equation/algorithm.
- Our-data-backed claim: cite our table/figure/protocol, not prior work.

## Paper I style spine

Author preference for the current Paper I working draft: keep **SNAKE** as the method name after first definition, define it as **Selection--Novel ADAPT Kost Evaluator**, preserve red/tentative result placeholders as future fill slots, and do not propose title/abstract/intro restructuring unless requested.

1. Mixed fermion--boson adaptive ansätze expose coupled expressivity/resource choices.
2. Existing ADAPT variants choose generators or batches, but do not jointly score candidate position, state-space geometry, compiled/measurement cost, and rollback-safe deletion.
3. Frame the novelty as **budgeted adaptive-ansatz acquisition by geometry- and cost-aware candidate-position selection**.
4. Use the static-paper novelty sentence, adjusted to local context:

   > SNAKE selects candidate-position records under a joint geometric and hardware-cost objective, reranks them by reduced-window Schur relaxation, and removes stale generators by rollback-safe generator ablation.

5. Define candidate-position records and the Phase I/II/III selector from `MATH/Math.md`.
6. Present Pareto and ablation evidence only where locked.
7. Mention dynamics only as an application of compact operator supports, not as the method.

## Paper II style spine

1. Real-time variational propagation fails when the inherited manifold becomes underexpressive, ill-conditioned, or overgrown.
2. Existing pVQD/AVQDS/adaptive-pVQD methods motivate projection and growth but do not by themselves supply bidirectional checkpoint-local maintenance.
3. Frame the novelty as **bidirectional checkpoint-local manifold maintenance**, not generic adaptive McLachlan dynamics.
4. Use the dynamics-paper novelty sentence, adjusted to local context:

   > AP-McLachlan maintains a checkpoint-local variational manifold by admitting zero-amplitude Schur-confirmed tangent blocks, pruning verified redundant generators, and allowing exchange patches without breaking trajectory continuity.

5. Define stay/append/prune/repair lanes from `MATH/Math.md`.
6. Keep measurement-compatible decision data explicit.
7. Present trajectory/resource Pareto evidence only where locked.

For Paper II method-source or implementation-spec work, repo-visible runtime
terms may appear when they name the active algorithmic object being specified:
`parameterization_mode`, `per_pauli_term`, `logical_shared`, support atom,
support patch, append ladder, prune ladder, exchange, solve repair, and
drive-aligned ansatz augmentation. Define the mathematical object first, then
give the implementation label. Do not flag those terms as manuscript defects
when the section is intentionally serving as the source of truth for
implementation.

## Paper III style spine

1. Mixed fermion--boson spectra require compact excitation manifolds, transition observables, root tracking, and driven-response diagnostics beyond ground-state preparation.
2. Existing QSE/qEOM/q-sc-EOM/VQD/SSVQE methods motivate excited-state subspaces, but a fixed excitation alphabet can overbuy matrix measurements, worsen conditioning, or miss mixed/polaronic directions.
3. Frame the novelty as **geometry-selected spectral-manifold acquisition**, not generic QSE and not generic adaptive dynamics.
4. Use the spectra-paper novelty sentence, adjusted to local context:

   > The method selects excitation records by probe-transition gain, overlap-metric novelty, residual-Schur spectral gain, conditioning, and measurement cost, then tests whether the selected nonorthogonal manifold remains closed under driven propagation.

5. Define excitation records, QSE metric/regularization, spectral residuals, transition observables, and frozen-vs-live propagation from `MATH/Math.md` or the Paper III method equations.
6. Cite QSE/qEOM/VQD/SSVQE literature before making excited-state novelty claims.
7. Present spectral-window, transition-strength, conditioning, matrix-measurement, and driven-response evidence only where locked.
