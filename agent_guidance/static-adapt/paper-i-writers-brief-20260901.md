# Paper I writer's brief — 2026-09-01

For every agent doing Paper I manuscript or PDF work. Standalone; verify
nothing here by asking the user — the sources are cited.

## Canonical manuscript (user ruling 2026-09-01)

- Repo: `https://github.com/20jastrobel/ADAPT---Paper-I.git` (its own git —
  commit TeX work there, not in `Holstein_test`). Resolve a local sibling clone
  by repository name; no absolute machine path is authoritative.
- Source: `Paper_I_overleaf_20260818/Paper_I.tex`. This is the filled,
  parity-verified version.
- `MATH/paper_details/Paper_I.tex` in Holstein_test is the stale 08-17
  Dimer draft — historical only. Never sync numbers from it.

## What is verified (cite, don't re-derive)

Per `docs/reports/EVIDENCE_PARITY_AUDIT_20260831.md` §2.1 and
`docs/reports/PAPER_I_RERUN_PARITY_RESULTS_20260831.md`:

- Every locally checkable claim MATCHES stored evidence: Table I 24/24
  cells + 6/6 k\*, Table III 24/24, all quoted percentages, Fig 2/3/5
  annotations, Table II verbatim; golden manifest 10/10 (Gate G2), kstar
  anchor sha `060e7524…` intact.
- Re-execution (k=20, weak cells, evidence lineage): star physics
  reproduces 11/12; appends to 4.4e-16.
- L4 RA matched-k75: `./ra_l3_matched_k75/result.json` now sha-anchored
  (`e8662dd1…fc3de5`) from the fetched cluster-9697719 tarball.

## Open flags — report, never silently resolve in prose

1. **Fig 2 §6.5 formal closure** (`RECOVERY-AUDIT.md` §6.5): substantively
   addressed on main (B3-supersedes-B5 + rebind), formal closure is the
   user's.
2. **EST-08 / S_alg accounting**: AVQE `N_H_refit` (Powell callbacks)
   drifts machine-to-machine while `N_grad` is exact; the archive itself
   disagrees −48…+103 between lanes. Any printed S_alg claim edits go
   through the user.
3. **L4 AVQE arm**: still unverified anywhere local (rerun is
   user-scheduled from `53bba1c2`); RA arm is anchored (above).
4. **R21 finding** (`intermediate_weak`, always-insertion): round-3
   selection tie-flip vs the archived CHTC run, reproduces identically on
   both code lineages — user interpretation pending. Do not "fix" the
   manuscript around it.
5. **Table I estimator-ledger raw backing** (cluster-9664325 tarballs) is
   unrecoverable (empty remotely pre-loss); the printed values stand on
   the recomputed corrections files, which verify. Flag if a reviewer
   claim needs the raw ledgers.
6. **Fig 3 optional-controls raw** (`l3_phase0`/`promotion_k15`):
   unrecoverable; regeneration = reconstruction, user's call.

## Working rules (multi-agent)

- Repo-changing Holstein_test work: own worktree under
  `Holstein_test.worktrees/`, own branch, explicit-path staging only.
  Manuscript TeX work: commit in repository `ADAPT---Paper-I`.
- Never silently rewrite manuscript claims when code/settings change
  (root `AGENTS.md` hard invariant); evidence adoption/promotion is the
  user's.
- LaTeX availability is machine-specific. Prefer the manuscript repository's
  declared build command, fall back to an installed Tectonic workflow when the
  contract allows it, and report unavailable builders accurately.
- RAM: builds are light; heavy reruns follow
  `agent_guidance/shared/memory-budget.md` (18 GB aggregate, bounded
  concurrent jobs with per-tree `ram_guard` allocations).

## Update 2026-09-04 — flag 3 is closed; arXiv readiness audit

### Flag 3 (L4 AVQE arm) — RESOLVED, brief above is stale

The user-scheduled rerun from `53bba1c2` completed 2026-09-04 15:33 in
`Holstein_test.worktrees/codex-paper-i-l4-level1-regen-20260904`, at
`output/local_runs/paper_i_hubbard_l4_avqe_level1_regen_20260904/append/`.

```text
schema        paper_i_hubbard_l4_arm_summary_v1
arm           append          rounds 60/60 accepted
S_alg         141495          (n_grad 2880, n_h_outer 60, n_h_refit 138555, n_metric 0)
terminal      absolute_energy_error 9.926e-2, accepted -3.283504070421171
exact         same-sector -3.3827681604170525
protocol_sha256      a77f501f4ac27cdb...
git_diff_sha256      e3b0c44298fc1c14...b855   (empty diff — sealed tree)
git_status_porcelain (empty)
```

Manuscript commit `bdac723` (2026-09-04 15:37, four minutes later) adopted
exactly these values: printed $\mathcal E=141495$ and the $9.93\times10^{-2}$
plateau. **The L4 AVQE arm is now locally verified and adopted; treat flag 3 as
closed and stop reporting it as unverified.**

Note against flag 2: $S_{\rm alg}=141495$ is an `N_H_refit`-derived printed
number, which flag 2 reserves to the user. `bdac723` is the user's own commit,
so that call has been made for this value. Flag 2 still governs any *further*
$S_{\rm alg}$ edit.

### arXiv submission readiness — Paper I is essentially clean

Verified against `Paper_I_overleaf_20260818/` @ `bdac723`:

| check | result |
|---|---|
| pages | 21 |
| class | `revtex4-2`, `aps pra reprint superscriptaddress` |
| bibliography | inline `thebibliography`, 58 `\bibitem`s (no `.bbl` dependency) |
| overfull boxes | 0 |
| undefined references / citations | none |
| live `\placeholder` / `\tentative` | none — `\structlabel` is defined at line 29 and never used |
| referenced figures | 4/4 present and resolving under `figures/` |

Compare Paper II, which still carries a live acknowledgments `\placeholder` and
a three-way unversioned source split. **Paper I is the nearer submission.**

Two tarball mechanics remain, neither scientific:

1. `\graphicspath{{../../}{}}` (line 16) — the `{../../}` entry is dead. The
   build log resolves no graphic outside the source directory; all four
   figures sit under `figures/`. Drop it so an arXiv tarball cannot reach
   above its root.
2. Tarball hygiene — 5 of the 9 `figures/` subdirectories are unreferenced
   (`paper_i_hubbard_noise_common_prefix_k20_20260814`,
   `paper_i_macro_insertion_policies_20260813`,
   `paper_i_ra_vs_append_matched_singleton_plateau_20260812`,
   `ra_adapt_appendix_v5`, `ra_adapt_appendix_v9`), and the stale
   `Paper_I.bbl` must be excluded because the bibliography is inline.

### Still open and still the user's

Flags 1, 2, 4, 5, and 6 above are unchanged. None blocks a build; each is an
evidence or interpretation decision. Flag 1 (Fig 2 §6.5 formal closure) and
flag 4 (R21 tie-flip) are the two that a referee could raise.
