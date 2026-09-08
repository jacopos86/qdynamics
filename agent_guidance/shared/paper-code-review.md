# Paper–Code Review

**Paper–Code Review** is the generic manuscript → mathematical workflow →
implementation viewer. “Code-to-paper workflow” and “paper–code viewer” name
this same tool for any paper, including Papers I, II and III.

## Locate and operate

The implementation and operating instructions are in
`tools/paper_code_viewer/README.md`. If that directory is absent from the
current checkout, inspect `git worktree list --porcelain` and read the copy in
the viewer worktree. The current local installation is
`/Users/lordskyler/Holstein_test.worktrees/codex-paper-viewer-review-20260908`.

Resolve manuscripts through `agent_guidance/paper-registry.yaml` and their own
repository contracts. A viewer bundle pairs explicit manuscript and code
snapshots; its source links do not establish which settings a reported run
executed. Use GitNexus to trace calls, then verify the exact source and run
receipts behind scientific correspondence claims.

To refresh or expose a version, build a new bundle from its declared source and
code roots, register it in the server catalog, and restart that server. Retain
the version's Markdown review-file path. Verify catalog identity, PDF and source
assets through HTTP; interact with the graphical viewer only when requested.

## Versions and review ownership

The paper title opens the paper/version chooser. Each entry selects the entire
PDF, workflow, code snapshot and review queue. The local catalog is
`~/paper-code-bundles/paper-review-catalog-20260908.json`.

Paper II v1's preserved original is `versions/v1/` in its manuscript repository;
the author-requested SIAM appendix belongs to `versions/v1-working/`. The
`papers/paper-ii-v1/` mapping targets that working revision. Canonical Paper II
remains v2 and its refreshed workflow is deferred to its owning task.

Source-line comments accumulate in the catalog entry's durable Markdown file.
The author hands the collected file to an agent when ready for implementation.
Saving comments does not dispatch work. Preserve unresolved correspondence
notes until source and evidence checks settle them.
