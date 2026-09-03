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
  `agent_guidance/shared/memory-budget.md` (18 GB aggregate, one heavy
  job, `ram_guard`).
