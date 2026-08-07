# Domain Docs

This repository uses a single-context domain-document layout.

## Before exploring

Agents should read:

- root `CONTEXT.md`, when it exists;
- relevant ADRs under `docs/adr/`, when they exist;
- the repository's `AGENTS.md` routing surfaces and any nearer subtree
  contracts.

If `CONTEXT.md` or `docs/adr/` does not yet exist, proceed silently. Create or
extend them only when domain-modeling work resolves terminology or a durable
architectural decision.

## Vocabulary

Use the terms defined in `CONTEXT.md` consistently in specifications, tickets,
tests, and implementation. If a required concept is missing, first determine
whether the proposed term is unnecessary or whether a genuine glossary gap
should be recorded.

## ADR conflicts

Surface any conflict with an applicable ADR explicitly. Do not silently
override a recorded decision.
