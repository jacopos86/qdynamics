# CHTC rescue fetch

The committed receipts on `main` record **221 remote paths** on the CHTC
submit node that were never fully fetched to git — the raw run archives were
pulled only to the lost machine. Until CHTC purges them, they are recoverable.
Extracted 2026-08-31 from `chtc/`, `output/`, `MATH/`, `agent_guidance/`
receipts; grep pattern: `/home/jsstrobel` and `/staging/.*/jsstrobel`.

**Do this soon.** CHTC purges inactive home/staging data on its own schedule.

## Primary: fetch everything (recommended)

From any machine with your CHTC SSH access (substitute your submit host,
e.g. `ap2002.chtc.wisc.edu`):

```bash
rsync -avh --progress jsstrobel@<submit-host>:/home/jsstrobel/ ~/chtc_rescue/home/
```

```bash
rsync -avh --progress jsstrobel@<submit-host>:/staging/j/jsstrobel/ ~/chtc_rescue/staging/
```

Size warning: a 2026-08 receipt records staging at ~65–78 GB after cleanup.
Check first with `ssh jsstrobel@<submit-host> 'du -sh /home/jsstrobel /staging/j/jsstrobel'`.

## Highest-value items if fetching selectively

The 47 recorded top-level items under `/home/jsstrobel/`, most valuable first:

- `paperii_combined_cycle_20260830_*.tar.gz` + `.json` — the final Paper II
  cycle uploads from the day before the machine was lost
- `Holstein_paper_ii_prospective_*_2026082{9,30}/` (8 dirs) — HH/Hubbard APM
  and scientific prospective campaign working dirs incl. `transfer_archives/`
- `Holstein_siam_nb4_*_2026082{7,8}/` (6 dirs) + `siam_*_upload.tar.gz` —
  SIAM screening sources and frontier archives (r07 cluster 9715270 lineage)
- `Holstein_paper_ii_staged_accuracy_euler_v1_20260825/` and
  `..._resume_fix_20260826/` — the t20/t5 staged accuracy raw archives
  (e.g. `paper_ii_t20_resume_fix_cluster9698264_12.tar.gz`); the only backing
  beyond the 12-row committed receipt for the provisional t20 frontier
- `Holstein_paper_ii_resource_rescue_t5_frontier340_74da5166_20260828/`
- `hh_apm_*`/`hubbard_apm_*` upload tarballs + `hh_apm_9715900_failure_logs_20260830.tar.gz`
- `paper_i_l4_qiskit_proxy_extensions_20260825{,_v2,_v3}/` and
  `paper_i_ra_adapt_historical_..._salvage_v1.tar.gz` — Paper I L4 lane
- `/staging/j/jsstrobel/paper_i_ra_adapt_page12_insertion_comparators_20260812_v1/outputs/transfer/*.tar.gz`
  and any other staging transfer dirs — Paper I bundle-page raw evidence
- `Holstein_phase3_optuna_chtc/`, `paper_ii_validators/`,
  `Holstein_jr_l10_*`/`Holstein_wave11_*` (July-era, lower priority)

After fetching, keep `~/chtc_rescue/` backed up somewhere off-machine; a
follow-up agent task can then reconcile the archives against the receipts on
`main` (each receipt names its expected `transfer_archives/` contents).

The complete 221-path listing can be regenerated any time:

```bash
grep -rhoE '"/home/jsstrobel[^"]*"|/staging/[a-z]*/?jsstrobel[^"'"'"' ]*' --include='*.json' --include='*.md' --include='*.txt' chtc output MATH agent_guidance | tr -d '"' | sort -u
```
