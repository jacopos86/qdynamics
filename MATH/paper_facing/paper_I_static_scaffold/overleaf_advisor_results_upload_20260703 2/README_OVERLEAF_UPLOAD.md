# Paper I Advisor Overleaf Results Upload

Created: 2026-07-03

## Upload Target

Upload this directory, or the zip generated from it, to Overleaf.

Use `main.tex` as the Overleaf main file. It is the advisor/Overleaf pasted draft with the current Paper-I source-of-truth abstract and the complete current Paper-I source-of-truth `Results` section inserted.

## Files

- `main.tex`: advisor draft for Overleaf, with current source-of-truth abstract and current source-of-truth `Results`.
- `source_of_truth_current_paper_i.tex`: full current local Paper-I source from `MATH/paper_details/static_adapt_paper_I.tex`, path-normalized for Overleaf.
- `advisor_raw_snapshot_20260703.tex`: raw advisor snapshot from the pasted Overleaf text, path-normalized only.
- `RESULTS_SECTION_REPLACEMENT_DIFF_20260703.diff`: unified diff showing the old advisor `Results` section replaced by the current source-of-truth `Results` section.
- `overleaf_advisor_results_upload_manifest_20260703.json`: provenance, hashes, copied graphics, and missing-graphics audit.
- `figures/` and `output/pdf/`: graphics copied for the included figure paths.

## Source Contract

The local source of truth used for the inserted abstract and `Results` section is:

`/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/MATH/paper_details/static_adapt_paper_I.tex`

The advisor snapshot used as the base document is:

`/Users/jakestrobel/.codex/attachments/fe632410-cee4-4ce8-8611-b71de40fbd0f/pasted-text.txt`

## Verification Intent

The purpose of this payload is not to keep the current source-of-truth manuscript as a sidecar only. The file `main.tex` itself is the advisor Overleaf draft with the source-of-truth `Results` section already substituted into the document body.
