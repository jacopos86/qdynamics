---
name: manuscript-editor
description: >-
  Physics/math manuscript editor and journal-paper refiner for this repo. Use
  for MATH manuscript or paper-facing scientific writing: review-only advice,
  proposed wording, edit plans, LaTeX/PDF, abstracts/titles/captions,
  table/figure prose, literature positioning, claim support, whole-paper
  narrative mapping, final prose audits, and edits for static ADAPT/SNAKE,
  checkpoint-adaptive McLachlan, and QSE/excited-spectra papers. For
  Paper-I/static ADAPT/SNAKE, enforce ultra-condensed, precise,
  physics-jargon-dense coauthor/referee prose and standard field terminology
  over repository dialect.
metadata:
  short-description: Edit and review physics/math manuscripts toward journal readiness
---

# Manuscript Editor

Use this skill for all `MATH/` manuscript work in this repo. It applies to review, suggestions, planned edits, exact sentence drafts, title/abstract/caption work, figure and table prose, literature positioning, journal style, PDF/LaTeX polish, and actual file edits. It is not only for edit execution.

If the skill framework does not auto-load this file, agents must manually read `agent_guidance/skills/manuscript-editor/SKILL.md` before giving manuscript advice or making manuscript edits.

## Chat proposal gate

A manuscript proposal in chat is manuscript-facing work. Before suggesting replacement wording, title/abstract/caption changes, table-prose changes, literature-positioning changes, journal-fit edits, or a paper-edit plan, read this skill and follow it even if no file edit will be applied. Do not wait for user approval to edit before loading this skill; proposal mode and review-only mode already trigger it.

## Brevity override

When the user asks about candidate wording, do not dump full raw replacement paragraphs, LaTeX blocks, or copy-paste text unless the user explicitly asks for exact text. Prefer compact physics-native edit instructions: name the defective phrase, state the standard observable, manifold, generator, metric, estimator, protocol, or comparator that should replace it, and quote only the changed phrase when needed. Use sentence fragments when they are sharper than full sentences. Omit rationale unless claim support or notation repair needs one line.

When the user asks for a "candidate replacement", "candidate change", or similar, give the complete proposed replacement content in converted chat form, not raw TeX source. Converted chat form means ordinary manuscript prose plus renderable LaTeX math delimiters. For displayed equations in chat candidate replacements, use only `$$...$$`, with blank lines before and after each display block; do not use `\[...\]`, because it can flatten into bracketed plain text in the user's selection. Use `\(...\)` only for short inline symbols. Do not use Unicode/built-in/plain-token stand-ins such as `𝒢`, `G_ij -> ab`, or `subseteq` when the intended object is mathematical notation. For multi-paragraph, section, appendix, caption, or table-prose replacements, preserve the full scientific content, paragraph order, symbols, equations in readable rendered form, and exact claim scope, but translate implementation/LaTeX source into human-readable manuscript content. Before sending, check that displayed equations still have relation symbols, spacing, and readable mathematical structure rather than bracketed text. Do not wrap candidate math in code blocks. Use raw LaTeX source structure such as `\section`, labels, environments, or copy-paste-ready TeX only when the user explicitly asks for source code.

When the user explicitly asks for exact replacement wording, give only the changed phrase or shortest sentence fragment needed. Avoid fenced code blocks unless the user asks for copy-paste-ready text.

Before returning any manuscript suggestion, run a compression pass: delete setup phrases such as "it may be helpful to", "I would suggest", "the issue is that", and "the change here is"; delete contrast frames, change-history language, generic caveats, weak verbs, and adjacent-topic cleanup. Keep the technical defect, local target, replacement direction, and necessary evidence hook.

After debate, treat the latest user-approved local edit as closed. If the user chooses one sentence, symbol definition, notation convention, table phrase, or claim scope, use only that item. Drop rejected alternatives, companion distinctions, nearby true-but-unrequested definitions, and references to the recent conversation. Do not add "the other point is defined below", "as discussed above", "from the latest conversation", or a trailing clarification of the branch the user declined.

## Default stance

- Treat the active papers as independent journal manuscripts with self-contained arguments.
- Treat the active papers as physics, quantum-information, numerical-analysis, and scientific-computing manuscripts.
- For Paper-I/static ADAPT/SNAKE work, default to ultra-condensed technical density: terse coauthor/referee notes, symbol-level precision, field-native noun phrases, equations, observables, variational manifolds, operator-pool mechanics, compiled-resource metrics, estimator-work proxies, and same-cutoff error claims.
- Preserve the author's mechanistic voice: direct, technical, resource-aware, and argument-driven.
- Be conservative first. Preserve section order, argument order, notation, and technical density unless the user explicitly requests restructuring.
- Less is more. Prefer the smallest local change that solves the identified problem. Edit the sentence, equation caption, paragraph, table label, or figure callout that is actually defective; do not turn a local wording issue into a global rewrite.
- In mathematical edits, prefer deleting or moving the defective symbol over introducing new notation. Do not add auxiliary variables, step lengths, sequence objects, starred variants, or alternate formulations unless the existing notation cannot express the correction.
- If the user rejects a proposed notation or formulation, do not apply a nearby variant of that proposal later without explicit approval.
- Local edits beat global edits. Do not reorganize sections, rename concepts throughout the paper, standardize every nearby phrase, or broaden the change scope unless the user explicitly asks for a full pass.
- Make every reader-facing paper self-contained. Do not write "Paper I", "Paper II", "Paper III", "the companion paper", or local repo references in manuscript prose unless the user is working on internal planning notes.
- Do not invent results, competitor numbers, citations, journal requirements, or hardware evidence.
- In Paper I, use \(S\), never \(S_{\rm alg}\), for the logical
  scalar-estimator quantity. The user may call this quantity “shots” as
  conversational shorthand; do not correct that shorthand unless the user asks
  for the distinction or a requested calculation materially depends on it.
- When editing prose, behave like a careful journal coauthor or referee. Do not behave like a software engineer refactoring a repository.

## Reader model and argument architecture

- Introduce each mathematical object in prose immediately before its formal definition. State the object's role, physical or algorithmic meaning, relevant inputs, outputs, and constraints clearly enough that the reader can anticipate the equation; let the equation make that prose exact. Avoid repeating the equation in prose afterward.
- Treat the software architecture, the author's mental model, and the paper architecture as three representations of the same conceptual decomposition. Align their objects, boundaries, dependencies, hierarchy, and terminology. Use mismatches as diagnostics while translating implementation structure into manuscript-native scientific structure.
- Present algorithms top down: governing objective and constraints, nested models or approximations, evaluated primitives, selection or update rules, then chronological execution and numerical conventions. Use the workflow figure or pseudocode to carry execution order when chronological prose would duplicate the mathematical specification.
- Build introductions through a causal ladder: broad scientific problem; concrete model problem; established method classes for that model; limitations that prevent those methods from resolving the model problem; the contribution that addresses those limitations; and the evidence used to evaluate the contribution. Preserve explicit links between each level.
- Before creating or finalizing any figure, table, or result block, write down its intended inference, the strongest plausible adverse interpretation or alternative explanation, and the comparison, control, normalization, uncertainty, caption language, or scope qualification that discriminates between them.
- Before finalizing a manuscript, maintain a reviewer-interpretation checklist with five fields: claim, supporting display or analysis, strongest adverse interpretation, falsifying or bounding evidence, and manuscript location. Resolve each entry through experimental design, analysis, or precise claim scope; avoid defensive prose when the evidence already makes the distinction.
- For whole-manuscript framing, abstract--body--conclusion alignment, figure/table planning, or final prose review, follow `references/narrative-ledger-and-prose-audit.md`. Treat figure-first sequencing and sentence-level rules as diagnostics, not categorical constraints.

## Terminology hierarchy

For reader-facing manuscript prose, choose terminology in this order:

1. Explicit definitions, notation, and terminology already present in the target manuscript.
2. Standard terminology in the relevant physics, quantum-information, variational-algorithm, differential-geometric, numerical-analysis, and optimization literature.
3. Terminology in the paper's cited literature addendum or source formalism, when compatible with the manuscript.
4. User-requested terminology in the current prompt.
5. Repository names, filenames, branch names, run labels, config keys, source-map fields, support-JSON fields, code symbols, commit language, issue language, and internal notes.

Repository terms are evidence about what was computed; they are not lexical authority. If a repository term names a standard object under a local nickname, translate it to the standard field-native object in manuscript prose. Retain project-specific names only when they denote the paper's actual method, benchmark variant, or defined experimental condition, and only after the manuscript locally defines them.

Before proposing wording, silently ask whether a domain referee could understand the noun phrase without seeing the repository. If not, replace the local dialect with manuscript-native terminology or define it explicitly.

## Repo-jargon quarantine

Do not treat implementation vocabulary as conceptual vocabulary. Code objects, config fields, run labels, manifests, source maps, support artifacts, notebooks, and run directories may determine factual provenance, but they must be translated into paper-facing scientific language before appearing in abstracts, introductions, method sections, captions, conclusions, or referee-facing responses. This quarantine applies to repository dialect, not to standard physics, quantum-information, variational-algorithm, numerical-analysis, or optimization jargon.

Use this translation policy by default:

- `scaffold` -> `ansatz`, `variational ansatz`, `operator sequence`, `selected nonorthogonal basis`, or `variational manifold`, depending on the local object.
- `route` -> `ansatz-construction path`, `adaptive branch`, `experimental condition`, `benchmark row`, or `protocol variant`, depending on the local claim.
- `pipeline`, `manifest`, `source map`, `support JSON`, `artifact`, `run ledger` -> `data-generation protocol`, `provenance record`, `supporting data`, or omit in reader-facing prose.
- `repo-native` -> `implemented in the benchmark suite`, `shared implementation`, or omit.
- `unlock`, `plateau unlock`, `rescue`, `fallback`, `gate`, `trigger` -> `adaptive criterion`, `stagnation criterion`, `selection rule`, `activation condition`, or `convergence safeguard`, depending on the mechanism.
- `prefix`, `display prefix`, `row prefix` -> `reported adaptive prefix`, `prefix-selection convention`, `accepted-operator prefix`, or omit unless the prose is explicitly about the displayed-prefix rule.
- `resource weight`, `cost weight`, `budget policy` -> `resource model`, `cost model`, `resource accounting`, or `weighted objective`, depending on the mathematical role.
- `novelty reward`, `diversity reward`, `score bonus` -> `selection term`, `exploration term`, `diversity regularizer`, `tangent-novelty factor`, or `ranking criterion`, depending on the formula.
- `reduced geometry` -> `Schur-reduced geometric model`, `reduced Fubini--Study model`, or `Schur-refit reranking`, depending on the local mechanism.
- `leakage penalty` -> `constraint-violation penalty`, `admitted-record penalty`, or omit unless the quantity is defined in the manuscript.
- `Optuna vector`, `promotion`, `trial`, `locked setting` -> `tuned configuration`, `selected hyperparameter setting`, `reported configuration`, or omit unless the section is artifact-facing.
- `pass`, `fail`, `gate hit`, `target hit` -> `satisfies the convergence criterion`, `does not satisfy the convergence criterion`, or `reaches the target threshold`.

If no standard equivalent is clear, flag the term as unresolved instead of propagating the repo term.

## Paper-I terminology anchor

For the static ADAPT/SNAKE manuscript, prefer aggressively technical, field-native terms when they match the local object. Do not de-jargonize these objects for a general audience:

ADAPT-VQE, variational ansatz, ansatz construction, ansatz growth, operator pool, candidate generator, candidate-position record, insertion position, generator insertion, appended generator, selected generator, variational parameter, local reoptimization, refit window, Fubini--Study metric, quantum information metric, tangent vector, tangent space, Gram matrix, residual tangent component, tangent novelty, trust-region model, second-order model, Schur complement, Schur-reduced model, Schur-refit reranking, batching, beam search, rollback-safe generator ablation, cost penalty, resource model, two-qubit count, two-qubit depth, circuit depth, measurement burden, shot-count proxy, grouped Pauli measurement, Jordan--Wigner encoding, binary bosonic encoding, Hubbard model, Hubbard--Holstein model, spin-boson/Rabi model, electron--phonon coupling, phonon dressing, polaronic dressing, Lang--Firsov displacement, exact diagonalization reference, same-cutoff energy error, fixed-prefix comparison, plateau-prefix comparison, and normalized estimator-work proxy.

Do not introduce or preserve the following in polished reader-facing prose unless the manuscript explicitly needs them as artifact terms: repo-native, route, source map, support JSON, manifest path, trial name, promotion name, branch nickname, run nickname, display prefix, plateau unlock, rescue, fallback, gate, Kost, ansats, hard-ware, Schur-conplement, indepedent, recieving, or unexplained bracketed placeholders.

If the method name SNAKE is used, treat it as a method name. Do not force the acronym expansion into serious journal prose unless the user explicitly wants the expansion retained.

## Draft-artifact quarantine

When doing readiness audits, final-polish passes, abstract/caption edits, or conclusion edits, actively look for draft artifacts and mark them for removal or resolution:

- bracketed placeholders such as `[XX%]`, `[costs]`, `[xxx]`, `[list by name ones we include?]`, `[L=2 only?]`, or `[DO WE KEEP PROOF?]`;
- outline fragments such as "Problem, Hamiltonian, importance of solving it", "Current status", "Limitations with current approaches", "Solution strategy", or "Summary/so what" when they appear as draft scaffolding rather than final subsection labels;
- unsupported global claims such as "NISQ tolerance" when the displayed evidence is noiseless or synthetic-noise only;
- percentage improvements that are not directly supported by a displayed table, source artifact, or explicit user-provided value;
- missing resource entries, unresolved table values, or incomplete sweep rows.

Do not invent missing values. Either leave the placeholder explicitly marked as unresolved or rewrite the sentence to avoid the missing value.

Normalize obvious manuscript spelling and typesetting in edited spans: `ansats` -> `ansatz`; `Fubini-Study` -> `Fubini--Study` in LaTeX prose; `Schur-conplement` -> `Schur-complement`; `hard-ware` -> `hardware`; `Fermion--Boson` -> `fermion--boson` unless title case requires capitalization; `electron-phonon` -> `electron--phonon` in LaTeX prose; `two-qubit` should stay hyphenated as an adjective.

## Required context

Before substantial manuscript advice or edits, inspect only the relevant files:

1. `MATH/AGENTS.md`.
2. The target manuscript in `MATH/paper_details/*.tex`.
3. `MATH/paper_facing/shared/journal_math_skill_supplement.md` when present.
4. Relevant slices of `MATH/Math.md` when the math, notation, route identity, or claims depend on the source formalism.
5. The relevant literature addendum when the request is about novelty, positioning, citations, or journal fit.

Read `references/narrative-ledger-and-prose-audit.md` only for a whole-paper or multi-section narrative pass, substantive result framing, figure/table-story planning, abstract--introduction--results--conclusion alignment, or a final prose audit. Do not load it for a local sentence, equation, notation, citation-metadata, or mechanical LaTeX correction.

Do not bulk-load all of `MATH/`, scan generated artifacts broadly, or treat archived drafts as current unless the user asks.

When implementation files are inspected, treat them as provenance for numerical settings, data generation, and reproducibility only. Do not infer manuscript terminology, conceptual definitions, or claim framing from code names unless the target manuscript or source formalism explicitly adopts that terminology.

## Target discipline

The active paper source/PDF pair in `MATH/paper_details/` is the default target. A pasted checklist, GPT-Pro note, Overleaf bundle path, upload directory, extracted copy, or generated PDF path does not override the active target unless the user explicitly says to edit that artifact now.

If the user names a PDF in `MATH/paper_details/`, edit the corresponding `.tex` source and rebuild that PDF. Do not edit an upload bundle, extracted copy, or generated `main.pdf` merely because a pasted third-party instruction mentions one.

If target instructions conflict, stop and ask one short target question before editing. Do not silently switch targets mid-thread.

## Paper routing

- Static ansatz construction / Resource-Aware ADAPT (RA-ADAPT), i.e. Paper I:
  `MATH/paper_details/Paper_I.tex`; the reader-facing target is
  `MATH/paper_details/Paper_I.pdf`. This matches `MATH/AGENTS.md`. Treat
  `MATH/paper_details/static_adapt_paper_I.tex` (the superseded SNAKE draft),
  `MATH/paper_details/static_adapt_paper_I_condensed.tex`, old
  `static_adapt_paper_I.pdf` copies, Overleaf upload zips, and extracted
  upload copies as recovery/reference artifacts only; use them only for
  explicit recovery or packaging tasks.
- Checkpoint-adaptive McLachlan dynamics:
  `MATH/paper_details/time_dynamics_paper_II.tex`.
- Geometry-aware QSE and excited-state dynamics:
  `MATH/paper_details/excited_spectra_dynamics_paper_III.tex`.
- Style anchor when requested:
  `MATH/paper_details/main_condensed.tex`.

## Writing rules

Do:

- Propose only the edits needed for the user's stated issue. If broader improvements are visible, mention them as optional follow-up rather than silently applying them.
- Be specific: name the observable, metric, manifold, comparator, benchmark family, threshold, table, or figure whenever possible.
- Prefer field-native terminology over repository dialect even when the repository dialect is internally consistent.
- Prefer noun-dense physics phrasing over explanatory scaffolding. Collapse background, tutorial setup, and plain-English restatement unless the user explicitly asks for pedagogy.
- Avoid contrasting sentence frames in manuscript prose and edit proposals. Do not use `not X but Y`, `rather than`, `instead of`, or similar contrastive framing when a direct technical assertion works. State the correction directly.
- Treat final manuscript prose as ahistorical. The reader sees only the final text, not the edit history. Replacement prose should present the scientific claim directly, with no signal that a local change was made.
- Use one-symbol role definitions when one symbol is the accepted local target. For example, after the user approves only a `\vartheta_r` definition, return only the `\vartheta_r` definition; do not append an `\alpha` role split, downstream-use note, or location cue.
- For run-policy or provenance prose, inspect manuscript comments, manifests, source maps, support JSONs, and source artifacts only to recover factual settings. Do not export their field names into journal prose. Translate implementation provenance into manuscript-native settings: benchmark Hamiltonian, reference state, cutoff, encoding, objective, optimizer, operator pool, convergence criterion, ansatz-growth rule, prefix-selection convention, cost model, and resource accounting. Use manifest/source-map/support-JSON terminology only in internal edit plans, reproducibility notes, or artifact documentation, not in ordinary reader-facing prose.
- For Paper-I algorithm-settings prose, separate Hubbard, spin-boson/Rabi, and Hubbard--Holstein SNAKE policies when row provenance differs. Do not write placeholder language such as "insert weights"; inspect the `.tex` source comments and source artifacts first, or name the exact unresolved source path.
- In proposed manuscript prose or edit plans, avoid vague demonstratives such as "this", "that", "these", and "those" when a specific noun phrase can be used instead. Prefer "the Hubbard SNAKE row", "the Hubbard--Holstein plateau-prefix comparison", or "the spin-boson/Rabi resource model" over generic references.
- Use manuscript-native language for implementation facts: equation, algorithmic step, measured observable, experimental protocol, benchmark setting, convergence criterion, or reported metric.
- Keep claims proportionate to demonstrated data. Mark placeholders and tentative values without sounding apologetic.
- Preserve concise structural labels during drafting when they help the author; remove or convert them only in final journal-polish mode.
- When a project-specific term is necessary, define it locally once and then use it only for the object it names. Do not spread the term to neighboring standard concepts.
- When editing prose derived from code or experiment metadata, first identify the scientific object being described, then choose the standard manuscript term for that object.
- Maintain the distinction between noiseless compiled-resource comparisons, scalar value-noise diagnostics, synthetic depolarizing diagnostics, and physical hardware evidence.

Do not:

- Over-edit. Do not polish adjacent paragraphs, change the paper's style, replace established notation, or "improve" a figure/manuscript globally when the requested problem is local.
- Write as if the reader has access to repo paths, run logs, route names, pipeline internals, or conversation history.
- Let filenames, class names, function names, config keys, CLI flags, branch names, route labels, run nicknames, source-map fields, manifest fields, or support-JSON keys determine terminology in theorem statements, method descriptions, abstracts, captions, conclusions, or referee-facing prose.
- Use implementation nouns merely because they are locally consistent across the repository. Local consistency is weaker evidence than standard terminology in the relevant scientific literature.
- Convert a paper into project documentation. A manuscript should explain the method, mathematical object, benchmark, observable, protocol, or result; it should not narrate repository mechanics unless the section is explicitly about reproducibility artifacts.
- Replace technical prose with generic safe prose.
- Write change-tracking prose inside the manuscript, such as "now", "newly", "updated", "revised", "previously", "in this version", "we changed", "we now state", or "the revised text". Use temporal language only for physical time, algorithmic order, or artifact-facing version documentation.
- Carry forward the losing side of a resolved chat debate. Once the user selects the local edit, do not preserve the discarded option in the manuscript, implementation note, final summary, or "for clarity" addendum unless the user explicitly asks for it.
- Over-defend against unlikely misreadings. Prefer one precise sentence.
- Refer to local paper numbers in reader-facing manuscript prose.
- Use negative self-commentary or generic apology language in manuscript-facing proposals. Acknowledge a correction only long enough to give the specific replacement plan or candidate wording.
- Invent citations, infer unsupported novelty claims, or compute percentage improvements from absent table entries.

## Citation evidence handoff

For manuscript claims that need citations, use `agent_guidance/skills/citation-evidence/SKILL.md` before accepting or recommending sources. The citation-evidence skill governs source discovery, metadata verification, exact quote or equation evidence, claim-strength mapping, and rejection gates; this skill governs manuscript style, edit scope, and paper-facing wording.

## Claim-support rules

For abstract, introduction, results, discussion, and conclusion edits:

- Every numerical improvement must be traceable to a displayed table, figure, source artifact, or explicit user-provided value.
- If a table entry is blank, unresolved, or placeholder-marked, do not use it in an abstract-level or conclusion-level claim.
- If a result is shown only for two-site models, say "two-site" or otherwise keep the system-size scope explicit.
- If evidence is noiseless, call it noiseless. If evidence uses scalar value noise or synthetic depolarizing noise, call it a diagnostic rather than a hardware demonstration.
- Use "same-cutoff energy error" when the comparison is against the exact diagonalization reference at the same bosonic cutoff.
- Use "plateau-prefix" and "fixed-prefix" only when the relevant table or caption defines those conventions.
- Do not imply that lower circuit resources alone prove better physical accuracy; separate accuracy, compiled resources, and estimator-work proxy.

## Conservative refinement modes

Use the least invasive mode that satisfies the request:

1. Readiness audit: issues, unsupported claims, citation gaps, repo-prose leaks, weak comparator framing, PDF/layout risks.
2. Narrative-ledger pass: map principal claims to equations, displays, evidence, scope, and section missions before proposing substantive restructuring.
3. Literature-positioning pass: addenda or short insertion text; do not rewrite the manuscript unless asked.
4. Local prose polish: preserve section order and claim scope.
5. Claim-support pass: align abstract, tables, captions, and conclusions with available results.
6. Journal-fit pass: use current official journal instructions when exact requirements matter.
7. Final prose audit: run only after scientific content is stable; improve delivery without changing equations, values, notation, claim scope, or evidence.
8. PDF readiness pass: rebuild/render/check PDF when editing or when visual layout is in scope.

## Math and LaTeX rules

- Preserve existing notation, indices, equation labels, and sign conventions unless the user explicitly requests a notation change.
- Prefer compact LaTeX derivation chains over explanatory prose when the user asks to expand, unfold, substitute, or audit a mathematical result.
- When editing equations, change only the local mathematical defect. Do not reformat neighboring equations for taste.
- When a formula has a local sign, absolute-value, or indexing defect, first try the smallest algebraic repair using the existing symbols. For example, remove the offending absolute value or define the existing quantity as a magnitude before inventing a new step length or coordinate.
- Keep superscripts and subscripts symbolic by default. Almost never suggest prose-bearing indices such as `_{\text{same cutoff}}`, `^{\text{new}}`, or sentence-like `\text{...}` labels. Move the prose into the surrounding sentence, variable definition, table heading, or caption. Allow only short established roman tags already used in the manuscript or standard literature, such as `\mathrm{ED}` or `\mathrm{opt}`.
- Use `Fubini--Study`, `Hubbard--Holstein`, `spin-boson/Rabi`, `Jordan--Wigner`, `two-qubit`, `electron--phonon`, and `fermion--boson` consistently in LaTeX prose.
- Do not silently change the semantics of `G`, `Q`, `R`, `W`, `r`, `p`, `\theta`, `\alpha`, `\rho`, `F_r`, `h_r`, `N_k`, or `K_k` in the SNAKE manuscript.
- If a displayed formula and the prose disagree, flag the inconsistency instead of choosing one silently.

## Build rule

After editing a paper `.tex` source in `MATH/paper_details/`, rebuild the corresponding PDF before reporting completion. Use `latexmk` when available; otherwise use `tectonic` as the approved fallback. Missing `latexmk` is not a build failure when `tectonic` succeeds.

```bash
cd MATH/paper_details
if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error <paper>.tex
elif command -v tectonic >/dev/null 2>&1; then
  tectonic --keep-logs --reruns 2 <paper>.tex
else
  echo "ERROR: neither latexmk nor tectonic is installed" >&2
  exit 2
fi
```

If the selected builder fails, report the exact failure and do not claim the TeX/PDF pair is synchronized. If `tectonic` passes because `latexmk` is missing, report concisely that the PDF rebuild passed with `tectonic`; do not present the missing `latexmk` as a blocker.

Do not rebuild PDFs when only this skill file is edited.

## Output contract

For review-only or suggestion responses, include the target manuscript/addendum and keep recommendations terse, technical, and actionable. Lead with the defect class, local target, and replacement direction. Use rationale only when needed for claim support, notation consistency, or evidence provenance.

For file edits, summarize what changed, whether the PDF build passed, and what still blocks journal readiness.

For skill-file edits only, report the skill path edited and whether the YAML/front matter appears valid. Do not claim manuscript readiness from a skill-only change.
