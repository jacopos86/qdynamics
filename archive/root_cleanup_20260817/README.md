# Root Cleanup Archive 2026-08-17

User-authorized relocation of non-control-surface files from the repository
root. Contents are unchanged; only the location moved from the repo root to
this folder. Companion storage record:
`agent_guidance/shared/storage-cleanup-20260817.md`.

Moved here from the repository root:

- `advisor_project_overview.{tex,pdf}`, `advisor_project_overview_assets/`,
  `advisor_project_overview_ping_alignment.tex` — advisor overview document
  source, build, and assets.
- `advisor_project_overview_full/`, `advisor_project_overview_full 2/` —
  export snapshots; the two differ (both kept).
- `advisor_project_overview_full_20260805/`,
  `advisor_project_overview_full_20260805 2/` — byte-identical duplicates
  (both kept per user decision), plus the `_20260802`/`_20260805` zips.
- `paper_v_archive_eom_divergence_advisor.{tex,pdf}`,
  `paper_v_archive_eom_divergence_compact.tex` — Paper-V advisor documents.
  `pipelines/open_dynamics/analyze_archive_correction_metric_ablation.py`
  names `paper_v_archive_eom_divergence_advisor.pdf` as a `visible_artifact`
  metadata label only; it does not read the file.

Added 2026-08-17 (second pass): `repo-architecture-plan.md` — root-level
"Repository Partition and ICM Desires" planning document (last modified
2026-07-28, referenced by nothing). Superseded by
`agent_guidance/paper-lane-refactor-plan.md` (which declares itself the
replacement contract) and `agent_guidance/shared/icm-gitnexus-pilot-plan.md`.

Deleted at the same time (regenerable build debris only): the
`paper_v_archive_eom_divergence_advisor` LaTeX aux files
(`.aux/.fdb_latexmk/.fls/.log/.out`), three `tmux-*.log` files, and
`.DS_Store`. The `.tex` sources above regenerate them.
