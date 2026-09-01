---
name: citation-evidence
description: Quote-grounded scientific citation finder and claim-support verifier. Use when a manuscript, paper, review, addendum, or chat proposal needs sources for a scientific claim; when checking whether citations support wording; or when finding or verifying exact quotes, DOI/arXiv metadata, source passages, and claim-strength matches.
metadata:
  short-description: Verify citations against exact source evidence
---

# Citation Evidence

Restored verbatim from commit `517eca4c` on 2026-09-01 (body unchanged; this note is the only addition).

Use this skill for citation-backed scientific claims. It is not just a literature-search workflow: candidate papers are leads only, and a citation may enter manuscript prose only after metadata is verified and the source text supports the intended claim strength.

When the task also asks for manuscript wording, captions, abstracts, tables, LaTeX edits, or paper-facing prose in `MATH/`, use this skill together with `agent_guidance/skills/manuscript-editor/SKILL.md`. This skill governs evidence acceptance; the manuscript-editor skill governs paper style and edit scope.

## Workflow

1. Restate the manuscript claim at the intended strength.
2. Decompose the sentence into atomic predicates: model family, study type, physical sector, condition or method, relation, and scope quantifier.
3. Search for primary papers, review papers, or standard references that could support the claim. Include adversarial searches when quantifier words such as "usually", "commonly", "standard", or "typical" are present.
4. Verify metadata from DOI, arXiv, publisher page, journal page, Crossref, ADS/INSPIRE, or the PDF itself.
5. Extract an exact quote, equation context, or clearly identified passage with a stable locator: page, section, equation, figure, table, or paragraph.
6. Map the source passage to the atomic predicates and classify support.
7. Recommend manuscript wording that matches the verified support level. If the evidence supports only a weaker claim, weaken the manuscript wording rather than stretching the citation.

Keep exact quotes short and limited to the words needed to verify the claim. Prefer source locators and paraphrased context over long copied passages.

## Spark worker dispatch

Prefer direct worker dispatch from the main Codex agent over RepoPrompt's `orchestrate` workflow. The skill should guide the main agent to start, poll, steer, and review explicitly pinned Spark workers itself.

When using RepoPrompt or another multi-agent layer, keep the final accepting authority in the main agent. Use 5.3 Codex Spark agents only as read-only evidence workers.

The main agent should:

1. Decompose the manuscript claim into atomic predicates and assign small, independent search tasks.
2. Start each worker directly with an explicit Spark xhigh model target, such as `codexExec:gpt-5.3-codex-spark-xhigh`, rather than relying on role defaults or `orchestrate`.
3. Send Codex Spark agents narrow prompts for discovery, metadata checking, quote retrieval, or claim mapping. Keep each task to one claim, source family, or citation candidate when possible.
4. Require each Spark agent to return the compact evidence record below.
5. Deduplicate and apply this skill's hard gates in the main agent.
6. Accept, reject, or weaken citations only in the main agent.

Spark-agent constraints:

- Spark agents must not edit, create, delete, move, format, stage, commit, or otherwise mutate repository files.
- Spark agents may search online, open primary sources, inspect publisher/arXiv/DOI pages, and read relevant repository files.
- Spark agents may read manuscript snippets, bibliography files, addenda, source comments, and existing citation evidence only when needed to interpret the claim.
- Spark agents must not run builds, tests, formatters, codegen, PDF rebuilds, artifact aggregation, CHTC jobs, or manuscript-edit commands.
- Spark agents must not call `context_builder`, `oracle`, nested `agent_run`, or any other helper that spawns or routes work to a different model. If a tool has uncertain model routing, do not use it inside the Spark worker.
- Spark agents must not decide that a citation is accepted. They may only return candidate, verified_metadata, quote_verified, or rejected evidence records.

Spark failure handling:

- If a Spark run is still running, steer it to stop searching and return the compact evidence already found.
- If a Spark run is waiting for input, answer only if doing so preserves the read-only, Spark-only constraints.
- If a Spark run reaches a terminal failure such as a context-window error, do not try to continue that run as the evidence worker. Treat any partial transcript as a lead only, then start a fresh Spark xhigh session with a narrower prompt and fewer sources.
- Do not use `context_builder` to rescue a failed Spark run. The retry must remain Spark-only unless the user explicitly approves a non-Spark model.

Use this sub-agent prompt template:

```text
You are a read-only citation-evidence worker using 5.3 Codex Spark.

Task:
{TASK}

Manuscript claim:
{CLAIM}

Atomic predicates:
{ATOMIC_PREDICATES}

Allowed actions:
- Search online for sources.
- Open primary source pages/PDFs and DOI/arXiv/publisher metadata.
- Read relevant repository files only when needed to understand the manuscript claim or existing bibliography.

Forbidden actions:
- Do not edit, create, delete, move, format, stage, commit, or mutate repository files.
- Do not run builds, tests, codegen, PDF rebuilds, CHTC jobs, or artifact aggregation.
- Do not call context_builder, oracle, nested agent_run, or any helper that spawns another model.
- Do not accept citations for the manuscript.
- Do not invent citations, metadata, quotes, page numbers, DOIs, or arXiv IDs.

Return only a compact evidence record using the schema from the citation-evidence skill. If no exact quote or stable locator is found, mark the record rejected or candidate-only.
```

## Support Levels

- S0 reject: no usable support.
- S1 direct precedent: the paper uses the condition.
- S2 repeated precedent: several primary papers use the condition.
- S3 common-practice evidence: a review, meta-statement, or systematic source set supports the practice or standardness claim.
- S4 exact support: the source supports the manuscript wording, including quantifier and scope.

Quantifier discipline is mandatory. One or two examples support "used in prior work", not "usually" or "commonly". Use "usually", "commonly", "standard", or "typical" only with S3 or S4 evidence.

## Hard Rejection Rules

Reject a citation for the current claim when any of these apply:

- Metadata cannot be verified, or DOI/arXiv identity is inconsistent.
- There is no exact quote, equation context, or stable source locator.
- Evidence comes only from search snippets, blogs, LLM summaries, ResearchGate previews, Semantic Scholar summaries, or other secondary-only sources.
- The quote supports only an adjacent claim.
- The source concerns a different model, study type, or physical condition than the manuscript claim.
- The quote concerns a basis, truncation, or reference state rather than the physical initialization or condition being claimed.
- A review is being used to prove a specific primary-paper setup without the review clearly stating that setup.
- Citation details, page numbers, quotes, DOIs, arXiv IDs, or bibliographic fields would need to be guessed.

When evidence supports only a weaker sentence, reject the stronger wording and recommend the weaker wording.

## Evidence Record

Use this compact record when handing citation work between agents or preserving evidence beside a manuscript:

```yaml
claim:
  manuscript_sentence:
  atomic_predicates:
  required_support_level:
candidate:
  title:
  authors:
  year:
  venue:
  doi:
  arxiv:
  source_url:
  source_type:
metadata:
  verified: true/false
  checked_against:
  notes:
evidence:
  exact_quote_or_equation:
  locator:
  accessibility:
claim_mapping:
  support_level: S0/S1/S2/S3/S4
  supports_original_wording: true/false
  safe_wording:
  limitations:
decision:
  accepted: true/false
  rejection_reason:
```

## Output Contract

For citation-finding responses, report the safest supported wording first, then accepted citations with locators and limitations. Include rejected or weaker-only candidates only when they prevent overclaiming or guide the next search.

For manuscript-facing responses, do not insert or recommend citations until the evidence record passes this skill's gates and the final wording matches the support level.
