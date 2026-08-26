# Plan — staged Paper-II campaign (screen → tune → qualify → promote → report)

**For:** Codex · **Author:** Claude · **Base commit:** `7edf45e3`
**Scope:** orchestration and reporting only. The APM and AVQDS mathematics, the
selector, the guards, and the runners are **not** in scope; they stay as
adapters beneath the new module.

Read `agent_guidance/shared/agent-handoff-contract.md` and
`agent_guidance/time-dynamics/paper-ii-comparison-protocol.md` first. This plan
implements the protocol; the protocol defines what a valid number is.

---

## 1. The workflow being automated

```
t=5 screen  →  threshold escalation  →  accuracy qualification  →  t=20 promotion
```

Author decisions, already settled — do not reopen:

| # | decision |
|---|---|
| 1 | Accuracy target ε = **1e-4**. |
| 2 | Gated quantities: **mean \|ΔE\|**, **mean \|Δn_d\|**, and **mean over time of maxᵢ\|Δnᵢ\|** (site occupation, componentwise). Primary density is redundant on a two-site problem: plot it, do not gate on it. |
| 3 | Both methods must independently satisfy ε. Their errors need not be equal. Tune each threshold so the achieved error lands **within the band 1e-5 … 1e-4** — qualify, do not overshoot. |
| 4 | State infidelity is an **additional gate** when the statevector was recorded (`--record-statevector`), otherwise a reported diagnostic. |
| 5 | Screen **all six drives × all three step-control laws**. |
| 6 | Each method tunes **its own** threshold, seeded from the existing t=10 ladder. Note the naming: AVQDS's knob is an **append** threshold (`--avqds-l2-cut`); this route's is an **insertion** threshold (`--insertion-l2-cut`). They are not the same flag and must not be conflated in code or in the report. |
| 7 | Escalation ladder tightens by ~3×: `3e-6, 1e-6, 3e-7, 1e-7, …` until the target is met or a genuine numerical safeguard trips. |
| 8 | Promotion to t=20 requires **only** the accuracy target. Cost is reported, never gating. |
| 9 | Report the full cost tuple (Ĉ₂q, Ĉ_D, Ĉ₁q); the author judges it. No automatic cost comparison. |
| 10 | "Continue to t=20" means **resume** the qualified t=5 trajectory. |
| 11 | If a cell qualifies at t=5 but misses ε over [0,20], **tighten and rerun t=20 automatically**. |
| 12 | Report: one PNG per gated observable carrying exact + AVQDS + APM, with the cost table immediately below. |
| 13 | No PDF rendering. |

---

## 2. What exists (do not rebuild)

| module | lines | owns |
|---|---|---|
| `pipelines/time_dynamics/paper_ii_runs.py` | 817 | registry: arms, insertion gates, inner numerics, step controls, drives, horizons, regimes; `build_run`, `RunCommand.argv/shell`, CLI `list/regimes/show/matrix/params` |
| `pipelines/time_dynamics/campaign.py` | 990 | seed × drive × horizon × arm product, manifest, CHTC packaging |
| `pipelines/time_dynamics/resume.py` | 162 | `build_resume_state`, `runtime_input_from_resume_state`; runner flags `--resume-from-run-json`, `--t-initial` |
| `pipelines/time_dynamics/run_lock.py` | 169 | `physics_fingerprint`, `assert_comparable` |
| `pipelines/time_dynamics/accuracy_target_report.py` | 199 | fixed-accuracy report (recently revised by Codex: `EXTEND LADDER` + next rung) |
| `pipelines/time_dynamics/write_paper_ii_manifest.py` | 116 | generates `paper-ii-run-parameters.md` |

`HORIZONS` already contains `t2`, `t5`… check: it contains `smoke`, `t2`, `t10`,
`t20`. **`t5` must be added** (t_final 5.0, 126 checkpoints to match the 0.04
grid spacing used elsewhere).

## 3. The four gaps

1. **No staged concept.** `campaign.py` is one-shot; there is no screen → promote.
2. **Ladders are shell scripts in a scratchpad**, so they are unreproducible and
   invisible to version control. Everything in §1 must live in the repo.
3. **`resume.py` is unused by any campaign** — promotion is manual today.
4. **No compiled-cost extraction.** Every number in this lane is Nθ. The cost
   tuple in §1 item 9 does not exist yet; Paper I's oracle is the source
   (`pipelines/static_adapt/hh_backend_compile_oracle.py`,
   `pipelines/qiskit_backend_tools.py`) and must be **called, not modified**.

## 4. Proposed module

`pipelines/time_dynamics/staged_campaign.py` — one deep module owning the whole
workflow, with the runners as adapters beneath it.

```
StageSpec        target epsilon, band, gated quantities, ladder schedule
CellKey          (regime, drive, arm, step_control, numerics)
ScreenResult     cell -> {threshold, errors per gated quantity, Ntheta, qualified}
QualifyResult    cell -> threshold that lands the error inside the band
PromoteResult    cell -> t=20 run resumed from its qualified t=5 run
CampaignLedger   append-only JSON record of every rung attempted, with run paths
```

Required behaviours:

- **Idempotent.** Re-invoking skips completed cells by reading the ledger, so an
  interrupted campaign resumes rather than restarting. Today's shell scripts do
  this by checking for `run.json`; keep that property.
- **Fail-loud on a no-op.** `eval` of an empty command string returns 0. A
  command generator that fails silently must not be recorded as a successful
  cell — check the generator's exit status *and* that its output is non-empty.
- **Provenance per rung**, not only per qualified cell. The ladder's non-monotone
  behaviour is itself evidence (see the protocol doc) and the discarded rungs
  must remain inspectable.
- **Verify from artifacts, never from intent.** Read the achieved configuration
  from `summary.support_patch_config` in each emitted `run.json`. This lane has
  twice produced arms that silently ran another arm's configuration.

## 5. Increments

Each ends committable, with tests, and can be verified without running a long job.

**1 — `t5` horizon + gated-quantity extraction.** Add `t5` to `HORIZONS`. Add a
pure function mapping a `run.json` to the three gated quantities plus infidelity
when present. Tests use existing artifacts under `output/frontier/`; no new runs.

**2 — ladder escalation as library code.** Port the scratchpad ladders into the
module: given a cell and a starting rung, emit the next rung per the ~3×
schedule, stopping at target, at band, or at a safeguard. Pure and unit-tested.
The safeguard condition must be explicit — a rung that trips one is *not* the
same as a rung that missed the target, and the ledger must distinguish them.

**3 — screening driver.** Run the t=5 screen over 6 drives × 3 step controls ×
{avqds, append_only, exchange}. That is 54 cells; at t=5 the cheap arms are
~30 s and exchange is ~10 min, so budget roughly 4 h sequential. Respect the
shared-machine limits: 10 GB ceiling, one heavy job at a time.

**4 — qualification.** Escalate per cell until the error lands in [1e-5, 1e-4] or
a safeguard trips. Record every rung.

**5 — promotion by resume.** For qualified cells, resume to t=20 via
`resume.py`. Re-check the gated quantities over the full [0,20]; on a miss,
tighten and rerun automatically (item 11). Note in the ledger that continuation
is physically faithful but **not decision-identical** — controller history does
not cross the boundary.

**6 — cost tuple.** Extract (Ĉ₂q, Ĉ_D, Ĉ₁q) by calling Paper I's oracle. Report
only; never gate.

**7 — report generator.** Per qualifying pair: one PNG per gated observable with
exact + AVQDS + APM overlaid, cost table immediately beneath. Compact the layout
relative to `output/pdf/ap_mclachlan_weak_weak_snake_progress_diagnostic.pdf`
(28 pages) — more panels per row. Markdown only; no PDF.

## 6. Traps specific to this lane

- **AVQDS's knob is append, this route's is insertion.** Different flags,
  different semantics. Conflating them in the tuner would silently tune the
  wrong dial for one arm.
- **Screening is a filter, not a guarantee.** Error here is dominated by rare
  discrete events: measured, the three largest step-to-step jumps carry 72–81%
  of total error growth under state-motion control. A cell can pass at t=5 and
  blow up at t≈8.6. Item 11 exists because of this; expect it to fire.
- **The ladder is non-monotone.** Adjacent-rung ratio reaches 19.1× and exceeds
  1 in 11 of 42 pairs — tightening the threshold degrades accuracy about a
  quarter of the time. An escalation loop that assumes monotonicity will stop
  early on a lucky rung.
- **Arms may differ only in the structural rule.** rk4 beats Euler ~3.5× for both
  methods, so arms differing in integrator are not comparable. Enforce with
  `assert_comparable` before any aggregation.
- `output/` and `prompt-exports/` are gitignored. Durable artifacts go in tracked
  paths.
- Another agent's broad `git add` can sweep uncommitted work; commit per
  increment.

## 7. Definition of done

The finalized commit; per-increment test results; the campaign ledger for a full
t=5 screen; the qualification table (cell → threshold → achieved errors → Nθ);
the promoted t=20 set; and one example report page for a qualifying pair. Stop
before promoting anything into Paper II's results — the author decides that from
the ledger.
