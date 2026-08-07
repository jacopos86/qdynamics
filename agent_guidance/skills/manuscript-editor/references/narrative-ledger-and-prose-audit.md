# Narrative Ledger and Final Prose Audit

Use this reference only for whole-paper or multi-section framing, result-story
planning, abstract--body--conclusion alignment, readiness review, or final
prose polish. Keep local edits on the lighter workflow in `SKILL.md`.

## Narrative ledger

Before proposing a substantive reorganization or polishing high-leverage
sections:

1. State the manuscript's central claim in one sentence.
2. Assign one scientific mission to each section.
3. Map each principal claim through the shortest applicable support chain:

   `claim -> equation -> figure/table -> evidence source -> scope qualifier`

4. Record the strongest plausible adverse interpretation and the evidence or
   qualification that bounds it.
5. Mark unsupported, duplicated, overstated, and orphaned objects before
   drafting replacement prose.

Use this ledger:

| Field | Required content |
|---|---|
| ID | Stable local identifier such as `C1` or `M2` |
| Claim | Atomic manuscript-facing assertion |
| Class | Central result, supporting result, method, definition, or literature claim |
| Location | Section and paragraph, equation, figure, or table |
| Equation | Governing equation or `--` when no equation is needed |
| Display | Supporting figure/table or `--` when no display is needed |
| Evidence | Locked result, derivation, cited primary source, or explicit design choice |
| Scope | Model, system size, regime, comparator, cutoff, noise class, and resource definition needed to keep the claim exact |
| Adverse interpretation | Strongest credible alternative reading or explanation |
| Status | Supported, bounded, duplicate, orphaned, provisional, or unsupported |

Do not require every claim to have both an equation and a display. Method and
definition claims may terminate at an equation or explicit design choice;
literature claims may terminate at a verified primary citation. Every central
result claim must reach a displayed or otherwise inspectable evidence anchor.

Use ledger failures diagnostically:

- A central claim without an evidence anchor is unsupported.
- A figure or table without a claim is orphaned or supplemental.
- An abstract or conclusion claim absent from the body ledger is ungrounded.
- One display carrying several materially different conclusions needs sharper
  scope, an additional control, or fewer claims.
- Repeated claims should have one canonical statement and local
  cross-references.
- An unbounded adverse interpretation requires a control, normalization,
  uncertainty statement, comparator qualification, or narrower claim.

For a telegram outline, write one line per section:

`section mission -> claim IDs -> equations -> figures/tables -> transition`

Use figures as narrative anchors when the evidence naturally supports that
order. Do not impose a fixed figure count, reorder a mature manuscript merely
to become figure-first, or manufacture a display for an equation-native
theoretical contribution.

## Final prose audit

Run this audit only after equations, notation, numerical values, citations,
comparators, and claim scope are stable. Treat every rule as a diagnostic
heuristic subordinate to scientific precision and the author's technical
voice.

### Pass 1: Clutter and repetition

- Delete empty setup phrases, generic importance language, change-history
  narration, and duplicated definitions.
- Replace vague demonstratives with the named scientific object when the noun
  improves precision.
- Preserve qualifications that carry evidence scope; do not delete them as
  mere hedging.

### Pass 2: Verbs and agency

- Prefer direct scientific verbs over avoidable nominalizations.
- Use active voice when the actor or algorithmic operation matters.
- Preserve passive voice when the measured object, mathematical construction,
  or reproducible procedure is the intended subject.

### Pass 3: Sentence architecture

- Split sentences that introduce several unfamiliar objects or independent
  claims.
- Repair buried predicates, parenthetical overload, and ambiguous attachment.
- Preserve long sentences whose clauses express one tightly coupled
  mathematical or comparative relation.

### Pass 4: Terminology and notation

- Reuse established field terms exactly; do not vary technical vocabulary for
  stylistic novelty.
- Verify acronym expansion and first-use definitions.
- Preserve manuscript notation and symbol roles.
- Reject edits that replace physics-native precision with generic prose.

### Pass 5: Cross-section consistency

- Compare the abstract, introduction, results, captions, tables, and conclusion
  for the same values, units, resource definitions, regimes, and comparators.
- Confirm that high-level claims retain the scope qualifiers recorded in the
  narrative ledger.
- Flag scientific or evidentiary inconsistencies for a claim-support pass.
  Never silently repair them during prose polishing.

## Guardrails

- Do not enforce a universal sentence-length threshold.
- Do not enforce one concept per sentence when tightly coupled mathematics
  reads more clearly as one sentence.
- Do not require a fixed number of figures.
- Do not ban arbitrary units categorically; require a defensible normalization
  and an explicit axis definition.
- Do not force the literal phrase “main result”; make the result hierarchy
  unmistakable through placement and direct assertions.
- Do not alter equations, values, citations, notation, or claim scope under a
  prose-only authorization.
- Do not perform a global rewrite when a local correction resolves the defect.

Report findings by scientific consequence: misleading or unsupported first,
then argument-structure failures, then prose defects. Apply only the scope the
user authorizes.
