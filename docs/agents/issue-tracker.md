# Issue tracker: GitHub

Issues and PRDs for this repository live as GitHub issues. Use the `gh` CLI for
issue operations and infer the repository from the configured Git remote.

## Conventions

- Create an issue with `gh issue create`.
- Read an issue and its discussion with `gh issue view <number> --comments`.
- List issues with `gh issue list`, requesting structured JSON when an agent
  needs to filter labels, bodies, or comments.
- Comment with `gh issue comment <number>`.
- Apply or remove labels with `gh issue edit`.
- Close with `gh issue close`.

## Pull requests as a triage surface

PRs as a request surface: no.

GitHub shares one number space across issues and pull requests. Resolve an
ambiguous number before mutating it.

## Skill routing

When a skill says to publish to the issue tracker, create a GitHub issue in the
repository identified by the current checkout's Git remote. When a skill says
to fetch a ticket, read the complete issue body, labels, and comments.

Publish implementation tickets in dependency order. Prefer GitHub's native
issue-dependency relationship for blocking edges. If that API is unavailable,
place a `Blocked by:` line near the top of each ticket body. A ticket is ready
only when all of its blockers are closed.
