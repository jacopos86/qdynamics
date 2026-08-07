# Paper I (RA-ADAPT) — audit worklist for codex

Created 2026-07-27. Target manuscript: `MATH/paper_details/Paper_I.tex`; reader-facing
PDF `MATH/paper_details/Paper_I.pdf`. Companion to
`paper_i_ra_adapt_manuscript_tracker_20260727.md` (insertion-policy G-list); this file
records a whole-paper awkward-prose / poor-notation audit **with the author's decisions applied**.

## How codex should use this file

- **DO** items: implement directly. Concrete before → after is given.
- **CONSIDER** items: **supply a candidate in chat and get author approval before editing.**
  Do not apply these silently.
- Follow `agent_guidance/skills/manuscript-editor/SKILL.md` (no `not X but Y` contrastive
  frames; field-native terminology; smallest local change; rebuild the PDF after edits).
- The file is under concurrent editing. **Re-read each span before editing** and locate by
  content, not by the line numbers below (they will drift).
- Line numbers are as of 2026-07-27 and are pointers only.

---

## DO — implement directly (awkward prose + trivial fixes)

### D1. Introduction opening (`~:125`)
Weak generic opener; `quantum processor units` should be `quantum processing units`;
`speedups include physics simulations such as` is a category slip; you prepare the ground
**state**, not the "ground-state ansatz".

- Before: *"Significant interest has been given to quantum computing in recent years. One
  motivation is that quantum processor units (QPUs) can offer computational speedups. Such
  speedups include physics simulations such as finding the ground-state ansatz of a given
  Hamiltonian \cite{Lloyd1996,AspuruGuzik2005}."*
- After: *"Quantum processing units (QPUs) promise computational speedups for classically
  hard problems, including the simulation of quantum many-body physics
  \cite{Lloyd1996,AspuruGuzik2005}. A central task is preparing the ground state of a given
  Hamiltonian."*
- Keep the following sentence (`The scope of available simulation problems is limited by QPU
  resource constraints ...`) unchanged.

### D2. Current-status sentence (`~:149`)
- `recursively and iteratively builds` → `iteratively builds` (drop the redundant pair).
- `Hartree-Fock` → `Hartree--Fock` (en-dash).
- `the ADAPT-VQE algorithm is sufficiently defined` → `the ADAPT-VQE algorithm is fully specified`.

### D3. Hamiltonian operator gloss (`~:143`)
- Before: *"\(c_{i\sigma}\) and \(b_i\) correspond to fermionic and bosonic annihilation
  operators, respectively, acting on site \(i\), with creation given by \(\dagger\)"*
- After: *"\(c_{i\sigma}\) and \(b_i\) are the fermionic and bosonic annihilation operators
  on site \(i\); their Hermitian conjugates (\(\dagger\)) are the corresponding creation
  operators"*

### D4. Circular partition sentence (`~:1253`)
- Before: *"We denote the partition of \(\mathcal R\) by partitioning \(\mathcal G\)."*
- After: *"We induce a partition of \(\mathcal R\) from a partition of the generator pool
  \(\mathcal G\)."*

### D5. `dominant … dominating` repetition (`~:1333`)
- Before: *"To prevent a numerically gradient-dominant cluster of similar
  physical-response-type records from dominating the record set advanced to the next phase"*
- After: *"To prevent a cluster of similar physical-response records with large gradients
  from crowding out other channels in the set advanced to the next phase"*

### D6. Appendix title case (`~:2036`)
- `\section{Simulated Noise Results}` → `\section{Simulated noise results}` (every other
  appendix head is sentence case).

### D7. Score domain typo (`~:279`)  — minor, author-confirmed
`S` maps the whole record set, not a single record.
- Before: `a scalar scoring function \(S:r\to\mathbb R\)`
- After: `a scalar scoring function \(S^{(t)}:\mathcal R^{(t)}\to\mathbb R\)`
- Note: the score symbol **stays `S`** (see C1); this only fixes the domain.

---

## CONSIDER — bring a candidate to chat first

### C1. `S` overloading — keep score, rename estimator-work metric
`S` is currently the phase score `S^{(t)}(r)`, the Pauli-word support `S_\mu` (`~:604`), the
**logical estimator-work column `S`** (`~:1032`, `~:1848`, all figure tuples), and the phonon
squeeze generator `S_i` (`~:1407`).

Author decision:
- **Score stays `S`** (natural; do not rename).
- Break the clash by renaming the **logical estimator-work metric** (preferred) — candidate
  `\mathcal E` (for estimator). The author notes it is effectively "shots"-like but is not a
  physical shot count, so `\mathcal E` is the safer symbol; `S` may stay if renaming ripples
  too far.
- Alternatively the Pauli-support alias `S_\mu` could be the one renamed instead.

Codex: propose in chat **which** symbol to move (estimator-work vs support) and the
replacement, then apply across body, appendices, figure captions/tuples, and
Table config. Renaming estimator-work `S → \mathcal E` touches `~:1032`, `~:1097`, `~:1105`,
`~:1119–1122`, `~:1138`, `~:1177`, `~:1848–1988`, and the `S_{\rm Append}/S_{\rm Geo}/
S_{\rm RA\text{-}ADAPT}` equations.

### C2. `\lambda` overloading — only the lane map
Author: `\lambda` as a scalar parameter is fine and internally consistent (Holstein coupling
`~:1053`, cost weights `\lambda_x` `~:720`, trust multiplier `~:424`, Gram eigenvalues
`\lambda_i` `~:837`). **Only the lane map is bad.**
- Rename lane map `\lambda:\mathcal G\to\mathcal L` (`~:1254`, and `\lambda(r):=\lambda(A)`)
  → candidate `\ell(\cdot)` or `\operatorname{lane}(\cdot)`.
- Codex: propose the symbol, then apply at `~:1254–1256` and any downstream lane-map use.

### C3. Record-set arrow chain is ugly (`~:296–306`)
Problems: `\operatorname{Pauli-Children}` renders the hyphen as a **minus sign**;
`\operatorname{De-duped \& Symm. Guard}` embeds prose in a math operator;
`\operatorname{Top}_N{(r\in\mathcal R\mid S^{(1)})}` uses `\mid` as "under"; and it disagrees
with the clean form already used at `~:1316`, `\operatorname{Top}_{...}(\mathcal R^{(2)};S^{(2)})`.
- Codex: propose a cleaner rendering — standardize on `\operatorname{Top}_N(\mathcal R;S)`,
  move the arrow annotations to words, or fold the chain into Fig. 1. Bring the candidate to chat.

### C4. `\oplus` / `\ominus` for circuit insert/remove (`~:304`, `~:966`, `~:1838`)
Nonstandard (algebraic symbols for a structural circuit edit), though defined locally.
Author is unsure of a replacement.
- Codex: propose an alternative (e.g. an explicit `\operatorname{insert}/\operatorname{delete}`
  or verbal update rule) **or** justify keeping it with the local definition. Bring candidate.

### C5. Minor symbol clashes (consider only)
`\Pi` = Fubini–Study projector (`~:408`) vs measurement-grouping blocks `\Pi_r` (`~:659`);
`\Lambda` = whitening eigenvalue matrix (`~:881`) vs ablation loss `\Lambda(d)` (`~:1756`);
`K_t(r)` cost factor vs `K` prefix length (`~:1866`). All decorated differently.
- Codex: assess; propose a disambiguation only where clean, otherwise leave and note as accepted.

### C6. Baseline / method naming scheme (author-directed change)
Current baseline has four names (`append-only ADAPT-VQE`, `append-only ADAPT`, `Append-ADAPT`,
`conventional Append-ADAPT`).

Author decision:
- Baseline comparator → **`ADAPT`** (single name).
- The method `RA-ADAPT` → **`RA`**, expanded loosely as "Resource ADAPT" / "Resource-Aware
  ADAPT" (the acronym is allowed to be a bit degenerate). Abstract intro becomes
  `Resource-Aware ADAPT (RA)`.
- Author-facing caution for codex: keep the **baseline `ADAPT`** distinct from the **general
  framework `ADAPT-VQE`** — a bare "ADAPT" can read as the whole class. Confirm the
  disambiguation wording in chat.
- Scope: global (~100+ occurrences), including title check, abstract, figure captions, and
  the estimator-accounting equation subscripts `S_{\rm RA\text{-}ADAPT}` (or
  `\mathcal E_{\rm RA}` under C1). Bring the exact scheme + one-line disambiguation to chat
  before the global rename.

### C7. Redo Algorithm 1 pseudocode (`~:1205–1242`)
Author: the pseudocode is bad and should be **redone**. It also still takes `\{\mathcal P_k\}`
as input and sets `\mathcal R\leftarrow\mathcal G\times\mathcal P_k` (`~:1211`, `~:1217`),
which is wrong when positions depend on `A`; the body correctly uses the per-generator union
`\mathcal R_k=\{(A,p):A\in\mathcal G,\,p\in\mathcal P_k(A)\}` (`~:258`). (Tracker gap G2.)
- Codex: rewrite Algorithm 1 to (a) build positions per generator from the branch state,
  (b) match the body's `\mathcal R_k`, (c) use a well-defined `STOP` consistent with C8,
  (d) keep `RESTRICT`/`REPRESENT`/`SCORE`/`FORM_PROPOSALS`/`INSTANTIATE`/`BEAM_RETAIN`.
  Bring the redrafted pseudocode to chat.

### C8. Stopping condition conflates artificial vs realistic (`~:974–982`)
Author: the paragraph confuses **the artificial stopping condition actually used to obtain the
results** (fixed 50 outer-iteration horizon, a reporting/experimental choice) with **what a
realistic algorithmic stopping condition would be** (gradient plateau / compiled-cost budget /
convergence to an available exact reference).
- Codex: rewrite so the reader clearly sees (a) the reported figures were generated under the
  fixed 50-iteration horizon, stated as an experimental cutoff, and (b) the genuine deployed
  RA stopping rule separately. Do not present the artificial cutoff as the algorithm's
  convergence criterion. (Tracker gap G4.) Then align `STOP` in C7 and the config-table
  `Execution horizon` row (`~:2028`). Bring candidate.

### C9. Config-table "Candidate position" contradicts the figures (`~:2018`)
`Candidate position → appended position` says append-only, but the reported figures
(`~:1097`, `~:1105`) show **plateau-insertion** trajectories.
- Codex: correct the cell to name the actual policies shown (no-insertion; plateau-insertion),
  consistent with C8's framing. (Tracker gap G5.) Bring candidate.

### C10. Abstract accuracy claims not mirrored in Results (low priority)
The abstract's `at least one order of magnitude lower … five of six / three of six` (`~:110`)
has no matching per-regime statement in the Results prose (`~:1124–1150`). Author has
deprioritized results-claims, but since this is now in the abstract:
- Codex: add a one-line per-regime backing near `~:1136`, or the author verifies the 5/6 and
  3/6 counts against the figures. Confirm before editing.

---

## Already resolved (no action)
- Literal `(CITE)` placeholder — **removed** (codex).
- Insertion-policy appendix (`\mathcal T_\tau`, `\bowtie`, `\overline{\mathcal P}_k`, the
  three-policy gate) is **live** at `~:1480–1560`; body references it. Only C7 (Alg.1) and C8
  (stopping) still lag it.
- `bibitem` count 55; uncited-entry cleanup remains the separate codex handoff in the tracker.
