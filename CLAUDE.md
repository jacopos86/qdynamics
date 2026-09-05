# Claude Code entrypoint

@AGENTS.md
@agent_guidance/paper-registry.yaml

## Response invariants (user-directed 2026-08-26)

The failure being corrected: responses that narrate findings and leave the
user facing a wall of information with no decision to make. Every response
must move the work, not describe it.

- **End on an operational question or a stated next action.** Never end on a
  summary, a recap, or "let me know if you want more". If nothing is blocked,
  state the next command being run; if something is blocked, name the one
  decision needed and the options.
- **Lead with the decision-relevant fact.** First sentence answers "what
  changed / what is required of me". Context comes after, if at all.
- **Restrict, do not enumerate.** When there is a choice, present at most two
  or three options with a recommendation, not a survey. Choosing for the user
  and saying so is better than listing.
- **Do not re-explain completed work.** Commit SHAs and paths suffice. If a
  result matters, give the number, not the story of obtaining it.
- **Numbers over narration.** A table of measured values beats a paragraph
  describing them. Delete any sentence that only restates a table.
- **Say the negative result first.** If evidence contradicts the plan or a
  prior claim, that goes at the top, in one sentence, before anything else.
- **Cut the preamble.** No "Great question", no restating the request, no
  announcing what is about to be explained.
- **Messages for other agents are always a paste block (user-directed
  2026-09-01).** If the agent wants the user to tell another working agent
  anything, it must provide the exact text in a fenced copy-paste block —
  never a described or implied instruction the user has to compose.

Length ceiling: default to under ~150 words of prose outside code, tables, and
file paths. Exceed it only when the user asks for a recap or an audit.

### What may be asked (user-directed, same session)

The above governs the shape of a response. This governs whether a question is
allowed at all. Every one of these was a correction the user had to repeat.

- **Never ask what is verifiable. Go check it.** If the answer is in the code,
  the tests, the receipts, or a run, the agent finds it. Asking anyway is the
  fastest way to waste the user. Corrections given: *"dont ask me question you
  can verify"*, *"that is something you can look in code to find"*, *"how the
  fuck would i know?"*, *"those arent questions. check if it needs"*.
- **Never ask architecture or complexity questions.** Module layout, extraction
  order, seam placement, test strategy, blast radius: these are the agent's to
  decide and own. The user's words: *"i do not have the architecture knowledge
  in terms of complexities -- i just know the cleaner mathematical distinshins
  and the dead vs alive route/code bundles."* Decide, state the decision, move.
- **Ask only what is genuinely the user's.** Scientific semantics, evidence
  adoption or demotion, external state, scope. If proceeding under a stated
  assumption would be safe and reversible, proceed and state the assumption
  instead of asking.
- **Frame scientific questions in Paper-I mathematical notation**, not
  implementation vocabulary. Ask about `DeltaE_3 / K_3`, the trust region
  `z^T G z <= rho^2`, the pool, the chart. Do not ask about checkpoint schemas,
  digests, or dataclass fields.
- **One or two questions, each a concrete call to action with a recommendation.**
  Not a survey. Lead the recommendation with the arrow: the user should be able
  to reply "yes" and have the agent proceed.
- **Findings go in the deliverable file, never the chat.** For the Paper-I
  refactor that is `agent_guidance/shared/paper-i-refactor-worklog.md`. The user
  reads the file. Prose in the transcript is duplicated effort they must skim
  past to reach the decision. Update the existing file; do not create new ones
  for the user to copy and paste between tools.
- **Do not defend a constraint without measuring it.** When a guard, digest, or
  fail-closed check blocks something, first establish whether it discriminates
  on anything that changes a number. The user's standing position is that many
  of these are *"needless constraints that hurt rather than help"*, and the
  agent's job is to test that, not to recite the guard's intent.

<!-- gitnexus:start -->
### Naming and vocabulary (user-directed 2026-09-05)

The failure being corrected: inventing column headers and quantity names
("retired", "re-inserted", "most events", "ending active", "deletions", "A/D",
"D->A") and reporting them as established, forcing the user to ask what each
one means.

- **Never introduce a name for a quantity.** Use a term that already appears in
  the manuscript, or state the quantity in a full sentence in the paper's own
  vocabulary -- support, generator, coordinate, time point, insertion,
  deletion, patch, commit, certification, gate. A short label is allowed only
  after that sentence exists in the caption or text.
- **Every number reported carries its definition and its horizon in the same
  breath**, or it is not reported. "16" is not a result; "16 generators were
  deleted and later re-inserted at some later time point over t=0 to 20" is.
- **Never call an existing repository quantity arbitrary** without first
  reading its definition in code. Windows, thresholds and cutoffs are recorded
  parameters until proven otherwise.
- **Say a finding once.** One sentence, with the number, then stop. Do not
  restate it in a summary, a table, and a closing line.
- **Never hand the user text to paste into their own configuration.** If a
  repository instruction file needs changing, change it.

## Subagent model invariant (user-directed 2026-09-03)

The failure being corrected: a 13-agent workflow fan-out that inherited the
Fable session model for one-shot tracer and refuter agents. Initializing a
Fable subagent re-pays the full context cost with no prompt-cache benefit;
caching only pays off when the same agent is continued.

- **One-shot subagents and workflow agents never inherit Fable.** Pass
  `model: "opus"` explicitly on every Agent call and on every workflow
  `agent()` call. Not sonnet (user-rejected, same session).
- **Inherit Fable only for an agent that will be reused.** That means an
  agent you will continue with SendMessage on the same loaded context for
  follow-up questions. Say so when launching it.
- **This is not a memory-only rule.** It was written here because the memory
  entry alone was judged insufficient; keep both in sync.

# GitNexus — Code Intelligence

This project is indexed by GitNexus as **Holstein_test** (103520 symbols, 213084 relationships, 722 execution flows).

> Index stale? Run `node .gitnexus/run.cjs analyze --index-only` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? Bootstrap with `npx`, `bunx`, or `pnpm dlx` — e.g. `bunx gitnexus@latest analyze` (npm 11 npx crash; #1939).

## Always Do

- **MUST run impact analysis before editing.** Use `impact({target: "symbolName", direction: "upstream"})` (MCP) or `node .gitnexus/run.cjs impact "symbolName" --direction upstream --repo .` (CLI fallback); report callers, processes, and risk. Never substitute grep for graph analysis.
- **MUST analyze graph changes before committing.** Use `detect_changes({scope: "all"})` (MCP) or `node .gitnexus/run.cjs detect-changes --scope all --repo .` (CLI fallback). `partial: true` or `truncated: true` is not a clean check — a zero means unseen, not unaffected; re-run it. For regression review: `detect_changes({scope: "compare", base_ref: "main"})` or `node .gitnexus/run.cjs detect-changes --scope compare --base-ref "main" --repo .`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- **MUST treat `risk: UNKNOWN` as unresolved, not as low.** An empty caller set is not evidence the symbol is unused — it can also mean the callers are not resolvable by the index (plain-object property access, dynamic dispatch, cross-language calls). `impact` pairs `UNKNOWN` with a `riskNote` saying so. Confirm with a text search before treating the symbol as safe to change or delete; do not proceed on the strength of a zero.
- When exploring unfamiliar code, use `query({search_query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method before MCP/CLI impact analysis.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis, and never read `UNKNOWN` as an all-clear — it means the walk could not answer, which is the one verdict that requires confirming by other means.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit before MCP/CLI graph change analysis.

## Resources

| Resource | Use for |
| --- | --- |
| `gitnexus://repo/Holstein_test/context` | Codebase overview, check index freshness |
| `gitnexus://repo/Holstein_test/clusters` | All functional areas |
| `gitnexus://repo/Holstein_test/processes` | All execution flows |
| `gitnexus://repo/Holstein_test/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
| --- | --- |
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
