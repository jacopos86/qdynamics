> **SUPERSEDED 2026-08-25** — see
> `HANDOFF_PAPER_I_B3_SUPERSEDES_B5_20260825.md`.
> Bundle 5 executes a macro-parent traversal that is split into Pauli children
> between Phase I and Phase II, which is not the flat-pool route Paper I
> describes. B5 moves to the appendix as an optional method; Bundle 3 becomes
> the HH headline evidence. The "do not reopen" instruction below no longer
> applies to the choice of bundle.

# Handoff — Paper I update: Bundle 5 as sole headline evidence, duplication-corrected S_alg

Written 2026-08-22 (~05:10Z) by the Claude session that produced the B5
corrections. Executor: the Paper-I manuscript agent. This document is
self-contained; the companion evidence file is
`FINDINGS_ESTIMATOR_LEDGER_DUPLICATION_20260821.md` in this same directory
(read §5, §13 before editing any estimator number).

## Task

Update Paper I so that:

1. The headline results (main-text tables/figures for the Hubbard–Holstein
   weak sector) come from **Bundle 5 only** — the measured-residual campaign
   (`b3mr`, CHTC cluster 9674089): arms `mr_append_ra`, `mr_always_insertion`,
   `mr_plateau_insertion`.
2. Every RA `S_alg` value printed from B5 is the **duplication-corrected**
   value, not the raw ledger occurrence count.
3. Strong-sector B5 numbers are **not typed yet** (runs still on CHTC; see
   "Pending" below).

## Settled decisions — do not reopen

- **B5-only** is a user decision (2026-08-21). B1–B4 material may survive only
  as appendix/reference; no B3-vs-B5 winner framing anywhere.
- **Correction rule** (both audit sessions converged, findings §5/§11):
  `E_corrected = N_H_outer(raw) + N_H_refit(raw) + N_grad(unique) + N_metric(unique)`.
  Repeated Powell/Hamiltonian callbacks stay charged. The rule is a no-op on
  AVQE, so comparisons stay symmetric. AVQE values are unchanged.
- **No runtime accounting code changes** as part of this task.

## Anchors

- Data of record: `results_tooling/kstar_tables.json` (this directory),
  key `b5`, entries `<arm>|<regime>`. Fields: `k_star`, `dE_star`, `dE_term`,
  `tuple_star`/`tuple_term` (compiled FakeMarrakesh, opt level 1, seed 7),
  `salg_star`, `salg_term` (raw, untouched), **`salg_star_corrected`,
  `salg_term_corrected`** (print these), `salg_correction` (per-scope removal
  receipts). AVQE comparators: key `aavqe` per regime.
- Manuscript repo: `/Users/jakestrobel/local_repos/ADAPT---Paper-I`, branch
  `main`, `Paper_I_overleaf_20260818/Paper_I.tex` (Overleaf-synced).
- B5 archives (verified sha256, local):
  `output/local_runs/paper_i_hh_b3_measured_residual_20260821/transfer/` in the
  `local_repos` checkout.
- Recount tool (for independent re-derivation):
  `chtc/paper_i_ra_adapt_repair_20260727/recount_estimator_phase_reuse.py` in
  the Documents checkout; invocation:
  `python3 recount_estimator_phase_reuse.py <archive.tar.gz> --prefix kstar=<salg_star> --prefix terminal=<salg_term>`.

## Weak-sector numbers of record (complete, corrected, final)

k\* matches AVQE per regime (W–W 11, I–W 9, S–W 10). Cutoff nph3.

| cell | k\* | dE at k\* | dE terminal (50) | tuple\* (N2q,D2q,Dc,W1q) | tuple_term | S\* corr (raw) | S_term corr (raw) |
|---|---|---|---|---|---|---|---|
| append W–W | 11 | 6.604e-04 | 1.123e-08 | 39,32,112,65 | 382,297,921,402 | **6,277** (6,409) | **103,493** (104,093) |
| append I–W | 9 | 5.052e-04 | 9.224e-09 | 35,25,83,55 | 265,194,660,370 | **5,599** (5,707) | **107,600** (108,200) |
| append S–W | 10 | 3.205e-06 | 1.380e-06 | 37,28,101,68 | 160,107,348,344 | **5,790** (5,910) | **87,578** (88,202) |
| plateau W–W | 11 | 6.604e-04 | 2.989e-07 | 39,32,112,65 | 242,203,625,348 | **8,736** (8,912) | **275,384** (281,194) |
| plateau I–W | 9 | 5.052e-04 | 1.648e-08 | 35,25,83,55 | 269,218,716,380 | **5,599** (5,707) | **277,200** (283,470) |
| plateau S–W | 10 | 5.988e-06 | 8.007e-07 | 36,24,87,64 | 230,214,732,624 | **10,294** (10,478) | **291,497** (300,377) |
| always W–W | 11 | 5.170e-04 | 5.218e-14 | 45,40,154,89 | 412,334,1097,460 | **17,776** (18,168) | **298,653** (306,248) |
| always I–W | 9 | 5.052e-04 | 3.630e-14 | 23,17,69,59 | 394,292,951,460 | **12,358** (12,632) | **309,779** (316,957) |
| always S–W | 10 | 2.078e-06 | 8.492e-11 | 38,27,101,72 | 298,225,813,504 | **14,031** (14,343) | **300,914** (307,868) |

AVQE comparators at k\* (`salg_star`): W–W 13,538; I–W 10,910; S–W 11,845.

Derived RA-vs-AVQE estimator ratios at k\* from the corrected values (recompute
from `kstar_tables.json` before typing; do not copy blindly): append
−53.6/−48.7/−51.1%, plateau −35.5/−48.7/−13.1%, always +31.3/+13.3/+18.5%
(W–W/I–W/S–W). Note the sign structure: B5 append/plateau now *undercut* AVQE
at matched k\*, unlike the B3-era numbers.

## Strong sector — all 9 cells promoted (complete 2026-08-22 18:32Z)

Cutoff nph7, horizon 75. S_alg columns are corrected (raw). Tuple is
N2q,D2q,Dc,W1q at k\*.

| cell | k\* | ΔE at k\* | tuple at k\* | S_alg at k\* | S_alg at 75 |
|---|---|---|---|---|---|
| append W–S | 75 | 4.556e-07 | 567,410,1353,675 | **389,622** (390,522) | same as k\* |
| append I–S | 75 | 2.798e-07 | 742,579,1777,685 | **427,501** (428,401) | same as k\* |
| append S–S | 27 | 2.284e-05 | 187,141,454,215 | **42,018** (42,342) | **307,738** (308,638) |
| plateau W–S | 75 | 2.831e-06 | 566,407,1376,743 | **772,024** (783,484) | same as k\* |
| plateau I–S | 75 | 2.258e-07 | 660,491,1612,731 | **747,963** (760,059) | same as k\* |
| plateau S–S | 27 | 2.283e-05 | 238,161,504,227 | **113,326** (115,264) | **834,902** (850,832) |
| always W–S | 75 | 6.314e-06 | 738,542,1706,745 | **1,046,764** (1,063,480) | same as k\* |
| always I–S | 75 | 2.172e-07 | 695,586,1939,811 | **1,006,016** (1,024,676) | same as k\* |
| always S–S | 27 | 1.340e-07 | 219,154,520,257 | **183,767** (185,889) | **881,404** (896,222) |

Duplication inflation in the strong sector runs 1.15–1.91%, the same
`gradient_surface` + `phase1_candidate_self_metric` signature as the weak cells.

**Bundle 5 is complete: 18 of 18 cells carry k\*, k_term, ΔE at both, circuit
tuples at both, raw and duplication-corrected S_alg at both, and
`run_self_certification`.** A field-level audit reports no gaps. Nothing in this
bundle is pending and no re-runs are outstanding. Every archive is local and
SHA-256 verified; CHTC staging for this campaign is empty.

Every cell's certification reads `ledger_closure_passed: true`,
`ledger_closure_status: complete`, `estimator_accounting_complete: true`,
`controller_validation_status: ok`, `paper_i_table_work_status: ok`,
`checkpoint_reason: iteration_done` — and in all 18 the checkpoint's own
`checkpoint_terminal_salg` equals the table's `salg_term`, which is independent
confirmation that the recovered values are what the runs recorded.

### Two provenance notes that affect printed numbers

**1. ΔE energy source is now harmonized; some values changed.** The two ingest
sessions used different energies: one took the stream's `hardcoded_adapt_iter`
energy at the reported depth, the other the checkpoint's terminal accepted
energy (`adapt_vqe.energy`). They differ by ~1e-8 because the checkpoint is
written after the round's refit while the stream event logs at selection. The
rule now applied to every strong cell, and recorded per-entry in
`dE_energy_source`: **use the most converged energy available at the reported
depth** — the checkpoint's accepted energy when k\* equals the terminal round,
the stream trace otherwise. Two values moved as a result:
`plateau|intermediate_strong` 2.871e-07 → **2.258e-07** and
`always|weak_strong` 6.905e-06 → **6.314e-06**. The append rows were already on
this convention and are unchanged. k\* is unaffected in every cell.

**2. Each cell now carries `run_self_certification`.** Because B5's worker
crashed after the science but before writing `result.json`, the natural referee
question is whether these numbers are reconstructions. They are not
uncertified: the checkpoint preserves the run's own receipts, and every ingested
strong cell reports `ledger_closure_passed: true`, `ledger_closure_status:
complete`, `estimator_accounting_complete: true`,
`controller_validation_status: ok`, `paper_i_table_work_status: ok`, and
`checkpoint_reason: iteration_done`, with `checkpoint_terminal_salg` matching
the ingested S_alg. Cite those fields rather than the recovery procedure.

### Tooling note

`recount_estimator_phase_reuse.py` lives in the `~/Documents/...` checkout,
which became unreadable mid-session (macOS TCC blocks reads on `~/Documents`;
files stat but will not open). The strong-sector corrections were produced by a
scratchpad reimplementation of the identical component-selective rule, gated on
reproducing published values before use: it returns 5,790 / 87,578 for append
S–W and 17,776 / 298,653 for always W–W, exact matches. If you need to re-derive
a number and hit the same permission error, that is the cause — not a missing
file.

## Traps

- **Scope names do not transfer across bundles.** B5's duplicates carry scopes
  `gradient_surface` + `phase1_candidate_self_metric` (plus small
  `phase3_scaffold_geometry` / `historical_..._active_gradient` terminal
  contributions). B3's scope was `full_candidate_self_metric_recompute`. Any
  recount filtered on B3's scope names removes zero occurrences from a B5
  ledger. Use the component-selective rule only.
- **`weak_report_45_b5` salg values are RAW.** If any round-45 report value
  enters the paper, run the recount with that prefix first
  (`--prefix r45=<salg>`); do not print raw r45 values next to corrected k\*
  values.
- **Findings §12 (B3-era table corrections) is superseded for the main text**
  by the B5-only decision; it remains valid for any surviving appendix B3
  reference. The pure-Hubbard controls and noise diagnostics in §12 are from
  non-B5 campaigns — those corrections stand as computed there.
- Weak sector is nph3, strong sector nph7 (binary-aligned cutoff policy);
  never compare across cutoffs in one table cell.
- The manuscript Appendix E, Eq. (E2) defect (findings §3: charges `g_r` at
  Phase I that the runs reuse from Phase 0) is a separate manuscript-side fix;
  if this task touches Appendix E, the corrected per-round expression is
  `R0 + R1 + R2 + n(n+1) + 2n·R3` — coordinate through findings §3 rather than
  re-deriving.

## Verification before finishing

1. For at least one cell, re-derive the corrected S\* independently with the
   recount command above and match `salg_star_corrected` exactly.
2. Every RA estimator number typed into the paper must equal a
   `*_corrected` field in `kstar_tables.json` (or a recount output for r45);
   every AVQE number must equal the raw `aavqe` field.
3. Percentages/ratios recomputed from those fields, not hand-carried.

## Report back

Per edited table/figure: which cells changed, source field for each value, the
one-cell recount verification result, and any B5 value the paper needs that
`kstar_tables.json` does not yet carry (stop and flag rather than improvise).
