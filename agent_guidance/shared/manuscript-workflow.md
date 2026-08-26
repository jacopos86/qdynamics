# Manuscript workflow (Paper I)

Read this before editing, building, or promoting results into the Paper I
manuscript. It governs where the manuscript lives and how edits reach the
author; it never overrides scientific or promotion authority.

## The manuscript is a separate repository

It is **not** in this checkout and **not** a branch of it.

| field | value |
|---|---|
| remote | `https://github.com/20jastrobel/ADAPT---Paper-I.git` (private) |
| local clone | `/Users/jakestrobel/local_repos/ADAPT---Paper-I` |
| branch | `main` |
| live source | `Paper_I_overleaf_20260818/Paper_I.tex` |
| figures | `Paper_I_overleaf_20260818/figures/` |
| build | `latexmk -pdf -interaction=nonstopmode Paper_I.tex`, run inside `Paper_I_overleaf_20260818/` |

The two repositories have unrelated histories. **Do not merge them, and do not
copy the manuscript into this checkout.** A copy inside the code repo diverges
from Overleaf the first time either side edits, after which two agents edit
different files with the same name. That failure has already produced wrong
claims in the paper once.

## Overleaf is the other end of `main`

`main` is bidirectionally synced with an Overleaf project the author edits
directly. Author pushes appear as `Updates from Overleaf` or
`Merge overleaf-<date> into main`.

Consequences that matter:

- **Pull before editing.** The author may have pushed since your last look.
- **Push after editing.** An unpushed commit is invisible to the author; they
  pull in Overleaf, not from your worktree.
- **Never force-push.** It would discard author edits made in Overleaf.
- Overleaf compiles its own PDF. Do not commit a built `Paper_I.pdf`.
- LaTeX build artifacts are gitignored in that repo; do not add them.

## Decoys — never treat as the source

| path | what it is |
|---|---|
| `ADAPT---Paper-I/Old archaic/Paper_I_RA_ADAPT_20260728/Paper_I.tex` | superseded July draft, tracked in the same repo |
| `~/Documents/Holstein_implementation/.../paper_packages/Paper_I_full_20260816/Paper_I.tex` | build mirror, not under git |

Only two `.tex` files are tracked in the manuscript repo; the live one is the
`Paper_I_overleaf_20260818/` copy.

## Promotion of results

Promoting run results into the manuscript happens **only after the runs
finish** (standing author directive). During a campaign, record results in run
artifacts and report them; do not write numbers into the manuscript
mid-campaign.

When promoting, record in the artifact which method configuration produced the
number — route variant, compile identity, cost weights — because appendix
evidence and main-table evidence can come from different configurations during
a transition.

## Claims about code must be verified against receipts

The manuscript describes what the runs did. Do not source such claims from
module constants or a grepped protocol: superseded campaigns share directories
with current ones, and inert configuration fields persist in artifacts. Read
the run receipts, and see
`chtc/paper_i_ra_adapt_repair_20260727/HANDOFF_MANUSCRIPT_cost_reporting_20260819.md`
for the verified compile and optimizer contract.
