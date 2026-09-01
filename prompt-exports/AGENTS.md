# prompt-exports AGENTS.md

This subtree owns persistent GPT/Atlas/ChatGPT handoff markdown exports.

## Export Path Rule

- `prompt-exports/*` is intentionally ignored by git for ordinary generated
  handoffs.
- RepoPrompt/context-builder/raw context exports that need to be returned through
  tools must not be written directly to the ignored `prompt-exports/` root.
- If a tool allows choosing the path, use `prompt-exports/repoprompt-returnable/`
  for RepoPrompt `prompt export`, `workspace_context export`, context-builder/oracle
  exported responses, and any other tool-mediated raw export that another
  model/agent must read back.
- If RepoPrompt/context-builder/oracle auto-generates a path directly under
  `prompt-exports/`, its common auto-export patterns are intentionally unignored
  in `.gitignore` (`oracle-*.md`, `context-builder-*.md`, `repoprompt-*.md`) so
  the tool can return/read back the file.
- Final GPT/Atlas/ChatGPT-facing handoffs may still be written under the ignored
  `prompt-exports/` root when created directly by the agent; they must be
  standalone and must not be raw RepoPrompt/context-builder dumps.
- If both a raw RepoPrompt export and a final GPT-facing handoff are needed,
  write the raw export under `prompt-exports/repoprompt-returnable/`, then write
  the rewritten final handoff under `prompt-exports/`.

## Reveal Rule

- Exception: do not reveal Paper-I GPT-Pro review packets generated under
  `prompt-exports/paper-i-gpt-pro-review/`. That workflow is agent-submitted
  through Atlas and must not run `open -R` unless the user explicitly asks to
  locate a file.
- After writing a final user-facing handoff markdown file in `prompt-exports/`, reveal it in Finder with `open -R <absolute-path>`.
- Do this automatically without asking for extra confirmation.
- Reveal the final GPT-facing export, not temporary sender files.
- If both raw intermediate and final GPT-facing exports exist, reveal the final GPT-facing export unless the user explicitly asks for the raw file too.

## Length Rule

- Default user-facing handoffs should be substantially longer than the old compact style.
- Treat the first concise draft as incomplete unless the user explicitly asks for brevity.
- Expand final GPT-facing exports to about 2.5x the first short draft by adding useful math, notation, equations, quantitative comparisons, artifact evidence, commands, caveats, and unresolved tensions.
- For HH exports, prioritize `MATH/Math.md` and artifact evidence over repo-agent prose.
- If a broad HH GPT export lands only around a few thousand words, assume it is still too short and expand it further before finalizing.
