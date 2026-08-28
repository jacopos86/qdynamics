# Paper-III selected-QSE versus direct-Davidson HH candidate campaign

This package distributes the locked 12-case Hubbard--Holstein matrix over
CHTC. Each process runs both independent routes for one case so the problem
construction is shared without introducing a selected-QSE-to-Davidson handoff.
The run classification is `candidate`; collection does not promote evidence or
edit the manuscript.

From the remote repository root, validate the exact descriptor before submit:

```bash
mkdir -p chtc/paper_iii_selected_vs_direct_hh_20260828_v1/logs \
  chtc/paper_iii_selected_vs_direct_hh_20260828_v1/case_results
python3 chtc/paper_iii_selected_vs_direct_hh_20260828_v1/validate_package.py \
  --repo-root . --require-clean-source
python3 chtc/validate_condor_submit_lifecycle.py \
  chtc/paper_iii_selected_vs_direct_hh_20260828_v1/submit_cases.sub
condor_submit -dry-run \
  chtc/paper_iii_selected_vs_direct_hh_20260828_v1/dryrun.ad \
  chtc/paper_iii_selected_vs_direct_hh_20260828_v1/submit_cases.sub
condor_submit \
  chtc/paper_iii_selected_vs_direct_hh_20260828_v1/submit_cases.sub
```

Each successful process transfers one JSON file to `case_results/`. After all
12 are fetched locally, collect and independently revalidate the complete
matrix with:

```bash
python -m pipelines.exact_bench.paper_iii_selected_vs_direct_collect \
  --case-results-dir chtc/paper_iii_selected_vs_direct_hh_20260828_v1/case_results \
  --output-root output/diagnostics/paper_iii_selected_vs_direct_hh_20260828_v1
```

The collector refuses missing, duplicate, profile-drifted, source-drifted, or
tampered case results. Exact errors remain post-run audit data and cannot
rewrite either route's online termination status.
