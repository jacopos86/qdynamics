# Notes for the results-pages agent — 2026-08-20

Hard-won specifics from the run/plot session that produced
`ra_adapt_variant_pages_20260820.pdf`. This is deliberately NOT a status
document — status, blockers, and campaign facts live in three committed
files in this directory:

- `CAMPAIGN_STATE_20260820.md` — clusters, bundles, defects, deliverable
- `AUDIT_FRESH_SESSION_20260820.md` — fresh-session risk audit (F1–F9)
- `SECOND_REVIEW_CODEX_20260820.md` — ranked blockers to the finished
  product, including three CRITICALs (two of them in the manuscript)

What follows is only the things that silently cost hours if rediscovered
the hard way. Working scripts referenced below are committed beside this
file in `results_tooling/` (they were living in a session scratchpad and
would otherwise have evaporated — the second review flagged exactly that
as a reproducibility blocker).

## 1. Data-extraction mechanics that bite

**AAVQE full per-round history is inside the result archives, not the
summaries.** `summary.json` carries `accepted_error_trace`, but the
continuation cells' complete r1–75 energies live in `run/result.json`
inside the tarball, which is ~385 MB for strong cells. A plain
`json.load` on that will thrash memory on a laptop. `extract_aavqe_hist.py`
slices only the top-level `"history"` array by bracket-matching the raw
bytes, which keeps it bounded. This is the difference between "the AAVQE
curve is a dot at k=75" and a real trajectory on every panel.

**The k=75 continuation streams are off by one.** Their live streams
carry an extra leading entry (the reference state at depth 1), so stream
round `r` is controller round `r−1`. The page-1 builder already knows
this (`K75_TO_MATRIX` block); anything new reading those streams must
apply the same shift or every k75 curve lands one round late.

**Live sampling of running cells** (how the variant pages showed
in-flight arms): `condor_ssh_to_job <cluster>.<proc>` then grep
`hardcoded_adapt_iter` out of `_condor_stdout`. Two traps in
`sample_live.sh`:
- `ssh` inside a `while read` loop eats the loop's stdin and the loop
  dies after one iteration. Use `ssh -n`. This failed silently — it
  sampled exactly one job of 29 and reported success.
- The job's `Arguments` do not reliably yield the execution id. Map
  `proc → execution_id` by line order in the package's `queue_fresh.tsv`
  (or `queue_resume.tsv`); that ordering is what `condor_submit`
  consumed, so it is authoritative.

**`load_ra_cells` contract** (page-1 builder): it globs
`x_*/run/summary/summary.json` and matches the arm by `__{arm}__`
appearing in the cell directory name, against the module's `FAMILIES`.
New arm ids (the Bundle 2/3/4 arms) must be added to `FAMILIES` or their
cells are silently skipped — no error, just missing series.

## 2. A validity check worth running on every continuation

The AAVQE r51–75 defect was found by eye and confirmed in one command:
**plot |ΔE| across the seam round and look for a discontinuity.** A
continuation that resumes correctly is smooth at the seam; the invalid
batch dropped ~3 decades between r50 and r51 (weak_weak 3.7e-04 →
1.3e-07). The cause was an optimizer-tolerance mismatch: the published
anchors ran the tolmatch overlay (ftol 1e-08 / xtol 1e-04, recorded in
each anchor payload's `optimizer_tolerance_overlay.json`) while the
continuation ran defaults (1e-12 / 1e-05).

Generalize it: **any time you extend or resume a published run, diff the
optimizer settings of the anchor against the continuation, and check the
seam round for a discontinuity.** Cheap, and it catches a class of
artifact that looks like a physics result. The corrected cluster
(9665004) is seam-smooth in all six regimes (weak_weak step −6.0e-10).

## 3. Matrix C (α\*) — measured engagement, and how to re-check

The second review asked for a warm-start acceptance gate. First two
completed cells of 9665210 (`c_ws_depth_append_ra`, intermediate_weak
and strong_weak_u8) both ran exact forced 50/50 rounds with **50 seeded
admissions and 0 cold fallbacks** — the rank-deficient fallback path
never fired.

Re-check per cell by counting telemetry in the cell's stdout:

```bash
grep -c '"status": "seeded"' <cell>.out; grep -c '"status": "unavailable"' <cell>.out
```

Event name is `static_adapt_schur_warm_start_v1`, stage
`session_kernel_transition_splice`; seeded rows carry `seed`,
`chart_index`, `effective_position`. If any cell shows fallbacks, that
cell's α\* claim is partial and must be disclosed as such (a run that
fell back on most rounds is Bundle 3, not Bundle 4).

Worth knowing when labeling page 4: the warm start was **inert on this
execution lane until commit `24e23efa`**. The historical machinery only
engaged on the legacy lane; the sealed knob alone produced zero
telemetry and byte-identical energies. So any older run claiming a warm
start on the session-kernel lane did not have one.

## 4. Empirical memory envelopes (the ladder profile)

Observed on this campaign — useful for sizing resubmissions:

| cell class | sealed request | what it actually needed |
|---|---|---|
| weak (nph3) append + depth cost | 16 G | 16 G — fine |
| weak (nph3) **costexp** arms | 16 G | **64 G** — all ten OOM'd |
| strong (nph7) matrix cells | 40 G | 64 G, several to 80 G |
| strong k75 continuations | 64 G | 80 G for the deepest rounds |

The driver is the costexp per-member transpile expansion: scoring every
member position of each equivalence class multiplies the compile work
held in memory. Ladder is 16 → 64 → 80 G; above 80 G leave held for
inspection rather than climbing.

**The automation that was doing this is now stopped.** `fleet_tender.sh`
(committed in `results_tooling/`) ran 30-minute cycles doing: snapshot,
auto-ladder OOM holds (code 34), auto-release quota holds (code 12)
after draining, and fetch-verify-clear of completed staging tarballs.
Nothing is tending the clusters now — held jobs will sit, and CHTC
staging (~96 GiB used of a 100 GiB quota, mostly Aug 7–17 archives) can
wedge transfers again. Either restart that script with the current
cluster ids, or watch holds manually.

## 5. Two things the reviews want that are not yet done

Repeating only because they gate the deliverable and are easy to lose:

- **Cost tuples are compiled from the final reordered circuit**, not
  each round's actual accepted ordering
  (`compile_routed_marrakesh_tuples.py`), and the manuscript admits the
  approximation. This undercuts matched-k\* insertion comparisons.
  Either reconstruct per-round orderings from checkpoints and compile
  those, or restrict claims to authenticated terminal costs. **This is a
  scientific decision, not a coding one** — it needs Jake.
- **The arXiv build has six `\bibitem`s after `\end{document}`**
  (`Paper_I.tex:2441`), so ED / Qiskit / Holstein / Powell / SciPy
  render as `[?]`.

## 6. Committed tooling

`results_tooling/` beside this file:

- `extract_aavqe_hist.py` — bounded-memory r1–75 extraction from the
  continuation archives (point it at the tolmatch dir, not the
  quarantined v1 one)
- `sample_live.sh` — live per-round sampling of running cells
  (`ssh -n`, queue-tsv proc→eid mapping)
- `build_variant_pages.py` — the variant-pages renderer that produced
  `ra_adapt_variant_pages_20260820.pdf`; imports the page-1 builder via
  `importlib` so panel layout and loaders stay shared rather than
  forked. Treat it as a starting point, not the authority: the second
  review's HIGH finding is that there is no single committed,
  manifest-driven generator with pinned inputs and output checksums, and
  that is still true.
- `fleet_tender.sh` — the cluster-tending loop described in §4.
