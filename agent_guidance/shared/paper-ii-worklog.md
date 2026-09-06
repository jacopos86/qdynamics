# Paper II worklog — manuscript/code faithfulness

Manuscript: `AP-McLachlan---Paper-II` @ `daa7ed5`,
`Paper_II_overleaf_20260901/time_dynamics_paper_II.tex`.
Code: `Holstein_test` @ `58ffd09a`, `pipelines/time_dynamics/ap_mclachlan/`.

## 2026-09-05 — which residual opens insertion

Question asked: does APM decide to append through the Schur route or through
the AVQDS residual?

**APM opens insertion on the AVQDS quantity, not on a Schur quantity.**

The manuscript gate is `m(\dot\theta) > \tau_m` with
`m(v) = 2[\|\bar d\|_2^2 - \Gamma(v)]` (Eq. `realized_mclachlan_distance`,
tex:222), declared applied to the *unnormalized* `m` (tex:411), with
`\tau_m = 3e-6` (tex:516).

The code computes exactly that quantity in
`ap_mclachlan/avqds.py:mclachlan_distance_squared`
(`2.0 * (norm_b_sq - captured_drift)`) and thresholds it in
`ap_mclachlan/exchange_integration.py:330`
(`insertion_gate_open = checkpoint_l2 > l2_cut`) whenever
`insertion_gate_mode == "mclachlan_l2"`.

Receipts confirm the reported arms use that mode:
`chtc/siam_nb4_full_apm_continue_r07_exchangefix_source_36c40d52_20260830/input/resume/action_receipt.json:127`
carries `insertion_gate_mode: "mclachlan_l2"`,
`insertion_l2_cut: 3.33333333333e-06`. The paired comparator arm is
`dynamics_policy: "avqds"` with `avqds_l2_cut` set to the same cut, so the
matched pair holds the residual and the threshold fixed and varies only the
structural rule.

### Where Schur actually enters

`schur_identity.exact_deletion_loss` (the reverse-Schur form
`r_{D|R}^T S_{D|R}^+ r_{D|R}`) is imported by exactly one non-test module,
`ap_mclachlan/deletion_permission.py`. It is the measurement-free *deletion*
permission test, paired with the effective-angle ray bound; its normalized cut
is `1e-3` (tex:521). `schur_identity.py:35` states the operating reading:
Schur degeneracy may nominate a deletion, never authorize one.

On the insertion side `support_patch.schur_novelty` is computed for any patch
with `inserted_count > 0` and stored on `SupportPatchScore.schur_novelty`, but
it does not appear in `rank_score`
(`support_patch.py`: `rank_score = normalized - cost_weight * penalty`, where
`normalized` is the realized before/after captured-drift change over
`\|\bar d\|^2 + \epsilon`). Ranking is realized captured drift, not a Schur
projection.

### What actually separates APM from AVQDS

Not the residual. Both threshold the same `m`. The differences are the
candidate population and the ranking:

| | AVQDS (`avqds.py:select_avqds_appends`) | APM (`exchange_integration.select_generalized_exchange_patch`) |
|---|---|---|
| candidate | one pool operator appended at circuit end | complete patch `(D, A)` |
| ranking | largest reduction in `m` | `\widetilde{\Delta\Gamma}_{\mathcal P} / K(\mathcal P)` (tex Eq. `insertion_bearing_score`) |
| deletion | never | permitted, ray + reverse-Schur prepermission |
| cost | not weighted | population-relative compiled burden `K` |

### Open faithfulness items

1. **tex:337** lists "patch assembly, Schur projection, ranking, and the
   augmented solves" as the classical stage. "Ranking" does not consume the
   Schur projection in code; it consumes the realized before/after captured
   drift. The hedge at tex:318 already says the implementation "evaluates the
   physical-metric change" and keeps Eqs. `ideal_exchange_components` /
   `ideal_exchange_change` as the exact-projection reference, so tex:337 is
   the one sentence that overstates the Schur block's role.
2. **Default drift.** `SupportPatchControllerConfig.insertion_gate_mode`
   (`adaptive_trajectory.py`) and `--insertion-gate-mode`
   (`runners/ap_append_from_adapt_artifact.py:1766`) both default to
   `residual_ratio`, while every reported APM arm runs `mclachlan_l2`. The
   comment above `max_joint_patch_evaluations` in the same dataclass claims
   "defaults reproduce the reported configuration, so a flagless call is the
   paper's route rather than a diagnostic one." That claim is false for this
   field.

## 2026-09-05 — the ranking ordinate, cost ignored

Question asked: what equation ranks a candidate gain in APM, is it a Schur
quantity, and what does AVQDS use instead?

**Ignoring cost, the two rules rank on the same quantity.** They differ by the
checkpoint-constant factor `2\|\bar d\|_2^2`, which cannot reorder candidates
within one time point.

### Shared primitive

`inverse.py:235` computes, for any support `S` at the realized (regularized)
velocity `\dot\theta_S`, against the raw unridged Gram:

    Q(S) = 2 f_S^T \dot\theta_S - \dot\theta_S^T G_S \dot\theta_S

This is the manuscript's `\Gamma(\dot\theta)` (tex Eq. `explained_drift`), and
`L^2(S) = m(\dot\theta_S) = 2[\|\bar d\|_2^2 - Q(S)]`.

### APM

`exchange_structural.py:scored` via `structural_cache.structural_candidate_solve`,
which returns `q(S) = Q(S) / (\|\bar d\|_2^2 + \epsilon)`. For a patch
`\mathcal P = (D, A)` with `R = J \setminus D`:

    \widetilde{\Delta\Gamma}_{\mathcal P} = q(R \cup A) - q(J)
    insertion gain           = max(0, q(R \cup A) - q(R))
    deletion loss            = max(0, q(J \cup A) - q(R \cup A))

`\widetilde{\Delta\Gamma}_{\mathcal P}` is tex Eq. `realized_patch_gain`. The
insertion gain is referenced to `q(R)` — the post-deletion support — so the
deletion is priced on its own term rather than folded into the gain.

### AVQDS

`avqds.py:select_avqds_appends` takes, over single appends `a` to the current
support `R`:

    \argmin_a L^2(R \cup \{a\}) = \argmin_a 2[\|\bar d\|_2^2 - Q(R \cup \{a\})]

`\|\bar d\|_2^2` is built from the state and is support-independent, so this is
`\argmax_a Q(R \cup \{a\})`, and the realized reduction is

    L^2(R) - L^2(R \cup \{a\}) = 2[Q(R \cup \{a\}) - Q(R)]
                               = 2\|\bar d\|_2^2 \cdot \widetilde{\Delta\Gamma}

### Where the Schur form actually sits

`\Gamma` itself is not a Schur quantity. Under an *exact* solve
`\dot\theta^\star = G^+ f`, `\Gamma(\dot\theta^\star) = f^T G^+ f`, and the
block identity (tex Eq. `ideal_exchange_components`,
`schur_identity.py`) gives

    \Gamma_{R \cup X}(\dot\theta^\star) - \Gamma_R(\dot\theta^\star)
        = r_{X|R}^T S_{X|R}^+ r_{X|R},
    S_{X|R} = G_{XX} - G_{XR} G_{RR}^+ G_{RX},
    r_{X|R} = f_X - G_{XR} G_{RR}^+ f_R.

So the *difference* of `\Gamma`s equals a Schur quadratic form, but only under
exact projection. The implementation never evaluates that form to rank: it
subtracts two realized `Q`s from the regularized solve, which is not equal to
the Schur form once ridge, retained-mode truncation, or damping is active.
`test_ap_mclachlan_support_patch.py::test_score_support_patch_gains_use_captured_drift_not_gamma`
pins that choice.

### What is left when cost is ignored

At `D = \varnothing`, `|A| = 1`, APM and AVQDS produce the same ranking. The
remaining structural differences are:

| | AVQDS | APM |
|---|---|---|
| deletion allowed | no | yes, `D \ne \varnothing` |
| batch size | 1 | `|A| \ge 1`, scored jointly, singletons not summed |
| insertion position | circuit end only | commutation-reduced positions in the survivor word |
| gain reference | `Q(R)` | `Q(R)` for the insertion term, `Q(J)` for the complete patch |
| admission filter | any strict reduction | `\widetilde{\Delta\Gamma}_{\mathcal P} > 0` on the complete patch |

## 2026-09-05 — can ridge / damping / truncation live inside the Schur form?

Question asked: the ranking is a difference of realized `\Gamma`s rather than a
Schur form because the solve is regularized — but can the regularization be
carried inside the Schur form instead?

**Ridge and damping: yes. Truncation: no. Under the reported policy the
regularized Schur form is rank-identical to what the code computes.**

Reported policy, from
`chtc/siam_nb4_full_apm_continue_r07_exchangefix_source_36c40d52_20260830/input/resume/action_receipt.json`
(`solve_repair_config`): `pinv_rcond = 1e-10`, `ridge_lambda = 1e-6`,
`solve_damping = 0.0`, every ladder a single rung.

### Why ridge carries through

Ridging the candidate Gram by `\lambda I` leaves the retained block equal to
the ridged base Gram, `(G_{R\cup A} + \lambda I)_{RR} = G_{RR} + \lambda I`, so
the block-inverse identity applies verbatim:

    S_\lambda = (G_{AA} + \lambda I) - G_{AR}(G_{RR} + \lambda I)^{-1} G_{RA}
    r_\lambda = f_A                  - G_{AR}(G_{RR} + \lambda I)^{-1} f_R
    f^T (G + \lambda I)^{-1} f - f_R^T (G_{RR} + \lambda I)^{-1} f_R
        = r_\lambda^T S_\lambda^{-1} r_\lambda

Damping is the same object on a PSD Gram with no truncation: `supported_inverse`
builds `sign(\mu)/(|\mu| + d)`, which for `\mu \ge 0` is `1/(\mu + d)` — the
ridge spectrum. Ridge and damping produced bit-identical columns in the sweep.

### Why truncation does not

`supported_inverse` zeroes eigenmodes below `pinv_rcond * max|\mu|`. The
retained set of `G_{R\cup A}` is not the union of the retained sets of `G_{RR}`
and `S`, so nesting fails. Measured (40 seeds, base support with planted rank
deficiency 2, pool of 12 single appends):

| policy | max rel gap to realized `\Delta Q` | argmax agree | mean Kendall tau |
|---|---|---|---|
| exact (`rcond 1e-12`) | 1.7e-11 | 40/40 | 1.0000 |
| ridge 1e-6 | 1.1e-05 | 40/40 | 1.0000 |
| ridge 1e-4 | 1.1e-03 | 40/40 | 1.0000 |
| ridge 1e-2 | 1.2e-01 | 40/40 | 1.0000 |
| damp 1e-4 | 1.1e-03 | 40/40 | 1.0000 |
| damp 1e-2 | 1.2e-01 | 40/40 | 1.0000 |
| trunc 1e-6 | 1.7e-11 | 40/40 | 1.0000 |
| trunc 1e-3 | 1.7e-11 | 40/40 | 1.0000 |
| **trunc 1e-1** | **5.0e+03** | **36/40** | **0.7083** |

The reported `pinv_rcond = 1e-10` is ten orders below where truncation starts to
bite, so this never fires in the reported trajectories.

### The residual gap is the Q definition, not the Schur algebra

What the ridged Schur form equals exactly is the *surrogate*
`f^T (G + \lambda I)^{-1} f`. APM's `Q` is the **realized** drift against the raw
unridged Gram, `Q = 2 f^T \dot\theta - \dot\theta^T G \dot\theta` with
`\dot\theta = (G+\lambda I)^{-1} f`, which expands to

    Q = f^T [ (G+\lambda I)^{-1} + \lambda (G+\lambda I)^{-2} ] f

so the two differ by `\lambda f^T (G+\lambda I)^{-2} f`. Under the reported
policy, over 200 seeds:

| quantity | value |
|---|---|
| max rel `\|\Delta Q_{realized} - schur\|` | 4.681e-04 |
| argmax agreement | 200/200 |
| mean Kendall tau | 1.000000 (min 1.000000) |
| seeds with a negative Schur form | 0 |

Ordering is untouched; only the value carries the `O(\lambda)` offset.

### Conditioning cuts in favour of the Schur form

Against an exact rational-arithmetic reference (`fractions.Fraction` Gaussian
elimination), the Schur form is the *better*-conditioned route, because the
difference-of-two-solves subtracts two large nearly-equal numbers:

| `\kappa(G_{RR} + \lambda I)` | two-solve rel err | Schur-form rel err |
|---|---|---|
| 1.41e+01 | 2.68e-12 | 8.29e-15 |
| 4.82e+07 | 2.75e-06 | 3.74e-08 |
| 6.15e+07 | 3.40e-07 | 9.92e-09 |
| 6.01e+07 | 1.56e-06 | 5.32e-09 |

### Consequence for the manuscript

tex:318 currently demotes Eqs. `ideal_exchange_components` /
`ideal_exchange_change` to an "exact-projection reference" and says ridge,
truncation and damping "change the optimizing velocity." That is true of the
value and false of the ordering. The manuscript may state the regularized form
directly, with `(G_{RR} + \lambda I)^{-1}` in place of `G_{RR}^{\dagger}`,
provided it declares the retained-mode truncation as the one operation the
identity does not survive.

**Caveat before this goes in the manuscript:** measured on random Gram matrices
with planted rank deficiency, not on checkpoint geometry from a completed run.
Repeat on `(G, f)` pulled from a locked trajectory before adopting.

Scripts: `scratchpad/schur_reg2.py`, `schur_run.py`, `schur_cond.py`
(session scratch; reproduce from the equations above).

## 2026-09-05 — projected-geometry schematic added to the manuscript

Asset: `Paper_II_overleaf_20260901/figures/paper_ii_mclachlan_projected_geometry.pdf`,
placed as `figure*` after Eq. `mclachlan_metric_objective`, label
`fig:projected_mclachlan_geometry`, referenced from the paragraph above it.

Provenance: user-supplied `mclachlan_aligned.svg` (1448x1086, 126 paths, zero
`<text>` elements — all glyphs already outlined), converted to vector PDF with
`svglib`/`reportlab`. The conversion preserves all three `linearGradient`s as
PDF axial shadings (`/Sh0`--`/Sh2`) and carries 291 `m` / 908 `l` / 1363 `c`
path operators with no raster XObject. Page box 1086 x 814.5 pt. The manuscript
PDF is 335 KiB with this vector figure against 995 KiB with the 719 KiB PNG the
user first supplied.

Build: `tectonic --keep-logs --reruns 2` from `Paper_II_overleaf_20260901/`;
no undefined references, no errors, figure read twice by the engine.

### Notation resolved: the figure adopts the manuscript's `\bar d`

Decision (user, 2026-09-05): the figure uses `\bar d`, not `\bar b`. The
caption was rewritten to name `\bar d` and `-\ii H\ket{\psi}` directly; the
reconciliation sentence is gone.

Two labels must change in the SVG, not one. The lower label carries the
projector applied to the upper arrow, so leaving `H\ket{\psi}` while the
projected vector reads `-\ii\Pi H\ket{\psi}` would make the projection off by
`-\ii`:

| position | current | required |
|---|---|---|
| upper plane, orange arrow | `H\ket{\psi}` | `-\ii H\ket{\psi}` |
| lower plane, red arrow | `\Pi H\ket{\psi}=\bar b` | `-\ii\Pi H\ket{\psi}=\bar d` |

No other label changes; the geometry is untouched. The manuscript caption at
`fig:projected_mclachlan_geometry` already assumes the relabelled figure, so
the committed asset is one revision behind the caption until the new SVG lands.

Note for the record: the *code* keeps the `b` convention (`norm_b_sq` in
`ap_mclachlan/inverse.py`, `geometry.py`, `support_patch.py`). Manuscript and
figure now agree on `\bar d`; code and manuscript still differ in symbol only,
with `\|\bar b\|_2 = \|\bar d\|_2`, so no reported number is affected.

## 2026-09-05 — equivalence and measurement cost, on real locked-seed geometry

Two questions: is the Schur form mathematically equivalent to what APM ranks
by, and does it require the same measurements?

**Equivalent under exact projection; rank-equivalent under the reported
regularized policy; identical measurement requirement.**

### Real geometry

Seed `chtc/paper_ii_production_v2/input/seeds/hh_snake_nph1.json` — the
calibration seed the lane's measurements were taken on. Loaded through
`pipelines/scaffold/runtime_loader.py`, drive `(A, omega) = (1.2, 2)`, policy
`rcond 1e-10 / ridge 1e-6 / damping 0`. Candidate blocks come from the
production `build_structural_insertion_cache` / `assemble_candidate_geometry`
path, so they are the blocks the selector actually scores. `|R| = 24` runtime
coordinates, 64-dimensional state, 80 singleton candidates.

| `t` | `kappa(G)` | max rel gap, realized `\Delta Q` vs Schur | argmax agree | Kendall tau | Schur `< 0` |
|---|---|---|---|---|---|
| 0.5 | 6.99e+28 | 8.786e-04 | yes | 1.000000 | 0 |
| 1.5 | 6.99e+28 | 8.786e-04 | yes | 1.000000 | 0 |
| 3.0 | 6.99e+28 | 8.786e-04 | yes | 1.000000 | 0 |
| 5.0 | 6.99e+28 | 8.786e-04 | yes | 1.000000 | 0 |

The real Gram is far worse conditioned than the synthetic stress case
(`7.0e28` against `6.1e07`) and the ordering still matches exactly.

**Caveat, unresolved.** `\|f_R\| = 0` at this checkpoint, so
`r_{A|R} = f_A - G_{AR}G_{RR}^{+}f_R` reduces to `f_A` and the run exercises the
`S_{A|R}` block without the drive-residual correction. The conclusion is
therefore established for the novelty block and untested for the correction
term. A checkpoint with `f_R \ne 0` — propagate off the seed rather than
evaluating at the seed angles — is required before this goes in the manuscript.

### Measurement requirement

Identical. Both routes consume the same blocks:

| block | source |
|---|---|
| `G_{RR}`, `f_R` | already acquired for propagation |
| `G_{RA}`, `G_{AA}`, `f_A` | new tangent-overlap measurements |

The Schur form is a rearrangement of those same measured blocks, not additional
data, so the executed scalar-estimator count `\mathcal E` is unchanged. Only the
classical post-processing differs: the two-solve route eigendecomposes the full
`(|R|+|A|)` matrix per candidate, whereas the Schur form needs `G_{RR}^{+}` once
and an `|A| \times |A|` solve per candidate.

Measured cost on the seed above, 80 candidates: 0.066--0.120 ms per candidate
for the two-solve route against 0.080--0.186 ms for the Schur form **as written
in the probe**, which recomputes `G_{RR}^{+}` inside the per-candidate loop. The
reuse is the entire point of the Schur route and the probe does not exploit it;
the timing column is therefore not evidence either way and must be redone with
the inverse hoisted before any efficiency claim is made.

Note: `score_support_patch` already threads a `schur_available_inverse`
argument for exactly this reuse, but only along the `schur_novelty` diagnostic
path, which does not feed `rank_score`.

### Where measurement cost genuinely differs

Not between the two ways of computing an insertion gain — between insertion and
deletion. Deletion candidates are row and column selections of geometry already
acquired (tex:443), and the deletion permission test is measurement-free by
construction (`deletion_permission.py` docstring). APM's extra candidate class
therefore costs no additional measurement; only joint multi-insertion batches
add candidate--candidate cross terms beyond what AVQDS pays per candidate.

## 2026-09-05 — corrected figure installed; resolution regression

`mclachlan_drift_corrected.png` verified before installing: the upper orange
arrow now reads `-\ii H\ket{\psi}` and the lower red arrow
`-\ii\Pi H\ket{\psi}=\bar d`. Both required changes are present and nothing
else moved. Installed as
`figures/paper_ii_mclachlan_projected_geometry.png`; the earlier
svglib-converted vector PDF was removed.

**Resolution regression, unresolved.** The corrected raster is 1448 x 1086,
which is 263 dpi at the 5.5 in print width the `figure*` uses. The first
raster the user supplied was 4344 x 3258 (790 dpi), and the vector conversion
had no resolution at all. PRA asks for 600 dpi on combination line art, so this
asset is below the journal floor by more than a factor of two. The manuscript
PDF went from 335 KiB (vector) to 1.33 MiB (this raster).

The fix is the corrected SVG rather than a corrected PNG: the source is the
same 1448 x 1086 viewBox with 126 outlined paths, so the same
`svglib`/`reportlab` conversion recorded above applies unchanged and restores
resolution independence.

### tex:337 reworded

Was:

> After the candidate tangent blocks have been measured, patch assembly, Schur
> projection, ranking, and the augmented solves are classical linear algebra.

Now:

> After the candidate tangent blocks have been measured, patch assembly, the
> conditional Schur projection, the augmented solves, and the ranking are
> classical linear algebra. The ranking ordinate is the realized captured-drift
> change evaluated from the augmented solves on `C` and `C^{\mathcal P}`, not
> from the Schur block; Eq. `ideal_exchange_components` makes the two agree
> under exact projection.

This removes the reading in which the Schur projection feeds the ranking. The
tex:318 strengthening — stating the regularized Schur form directly, since the
ordering survives — remains gated on the `f_R \ne 0` checkpoint.

Build: `tectonic --keep-logs --reruns 2`, no undefined references, no errors.
Still uncommitted: `.git/index.lock` (zero bytes, 15:13) in the manuscript repo.

## 2026-09-05 — figure closed: corrected vector source installed

`mclachlan_drift_corrected.svg` supersedes both the 263 dpi raster and the
strip-and-typeset workaround. Converted with the same `svglib`/`reportlab`
path: 125 paths, 294 `m` / 919 `l` / 1410 `c` operators (against 291 / 908 /
1363 for the pre-correction drawing, the difference being the added `-\ii`
glyphs), all three axial shadings intact, no raster XObject, page box
1086 x 814.5 bp, 42,191 bytes.

Installed as `figures/paper_ii_mclachlan_projected_geometry.pdf` and included
with a plain `\includegraphics[width=0.78\textwidth]`. The interim
`tikzpicture` overlay that typeset the two labels from the manuscript is
reverted and `\usepackage{tikz}` removed, because that version's label
placement was a computed estimate that had never been looked at, whereas the
corrected source carries placement the author has seen.

Manuscript PDF 344.5 KiB. Build clean, no undefined references.

Recorded for the future: the labels are baked into the artwork again, so a
later change to `\bar d` or to the drive convention is a redraw, not a
one-line edit. The removal recipe is in the superseded entry below if that
trade ever needs reversing.

### superseded — LaTeX-typeset labels over stripped artwork

## 2026-09-05 — resolution regression removed: labels typeset, artwork vector

The 263 dpi raster is gone. The figure is vector again and the two drive labels
are set by LaTeX in the manuscript's own font and notation.

Method: located the two label glyph runs in the original
`mclachlan_aligned.svg` by fill colour and transform --
`translate(994.58968750,163.28000000) fill="#ff8900"` (the upper drive label)
and `translate(1026.74218750,706.90000000) fill="#ef1f27"` (the projected-drift
label) -- and removed those two `<path>` elements. The remaining 124 paths
convert to a 35,986-byte vector PDF with all three axial shadings intact.

The labels are then placed in a `tikzpicture` over the artwork, right-anchored
at the former right edges of the removed runs, in the drawing's own colours:

| label | former baseline origin (SVG) | placed at (bp, page box 1086 x 814.5) | colour |
|---|---|---|---|
| `-\ii H\ket{\psi}` | (994.59, 163.28) | base east (847.5, 692.04) | `#FF8900` |
| `-\ii\Pi H\ket{\psi}=\bar d` | (1026.74, 706.90) | base east (984, 284.32) | `#EF1F27` |

Mapping is `x_pdf = 0.75 x_svg`, `y_pdf = 814.5 - 0.75 y_svg`; font size 52.5 bp
in the figure's own coordinates, which the enclosing `\resizebox` carries to the
printed size. `\usepackage{tikz}` added to the preamble.

Consequences: the notation is now bound to the manuscript rather than to the
artwork, so a future change to `\bar d` or to the drive convention is a one-line
edit instead of a redraw. The user-supplied `mclachlan_drift_corrected.png` is
no longer the asset; it served as the confirmation of what the two labels should
say.

Manuscript PDF: 333.8 KiB. Build clean, no undefined references.

**Unverified: label placement.** The coordinates are derived from the removed
paths' own baseline origins, but the rendered result has not been looked at,
and the repository contract forbids rendering or viewing the generated PDF
without an explicit request. Baseline positions are exact; what is uncertain is
whether the Latin Modern metrics make the longer strings collide with the
orange arrow head or the dashed drop line.

# Candidate change list — 2026-09-05

Source: user pass through the manuscript, linear from Eq. 1. Nothing below is
implemented yet. Verdicts: **DO** (implement as stated), **DO*** (implement with
the stated deviation), **CODE** (verify against the source before writing),
**ASK** (needs a decision only the user can make), **NO** (disagree, reason
given).

## Equation-number snapshot (authoritative, from `.aux` at this revision)

The user's item numbers below refer to these. Re-run the snapshot after any
edit that adds or removes a numbered equation.

| Eq | label | tex line |
|---|---|---|
| 1 | `eq:hubbard_holstein_driven` | 127 |
| 2 | `eq:mclachlan_velocity` | 143 |
| 3 | `eq:dynamics_sampling` | 179 |
| 4 | `eq:projected_tangent_block` | 184 |
| 5 | `eq:projected_schrodinger_drift` | 192 |
| 6 | `eq:mclachlan_metric_objective` | 204 |
| 7 | `eq:explained_drift` | 243 |
| 8 | `eq:realized_mclachlan_distance` | 248 |
| 9 | `eq:support_patch` | 282 |
| 10 | `eq:conditional_block_normal_equations` | 307 |
| 11 | `eq:conditional_baseline_response` | 312 |
| 12 | `eq:conditional_schur_system` | 327 |
| 13 | `eq:ideal_exchange_components` | 343 |
| 14 | `eq:ideal_exchange_change` | 354 |
| 15 | `eq:deletion_angle_sum` | 367 |
| 16 | `eq:deletion_ray_bound` | 372 |
| 17 | `eq:deletion_prepermission` | 379 |
| 18 | `eq:patch_ray_refit` | 388 |
| 19 | `eq:patch_ray_distance` | 397 |
| 20 | `eq:patch_velocity_continuity` | 409 |
| 21 | `eq:realized_patch_gain` | 423 |
| 22 | `eq:shifted_compiled_score` | 432 |
| 23 | `eq:insertion_bearing_score` | 443 |
| 24 | `eq:operating_score` | 454 |
| 25 | `eq:raw_coordinate_step_control` | 476 |
| 26 | `eq:state_space_step` | 483 |
| 27 | `eq:metric_eigenmode_step_control` | 488 |
| A1–A2 | `eq:static_seed_circuit`, `eq:zero_angle_seed_handoff` | 863, 881 |
| B1–B14 | compiled-cost and estimator appendix | 899–1056 |

## Structure and framing

| # | change | verdict | note |
|---|---|---|---|
| 1 | Eq. 1: inline the drive term in prose rather than carrying it as a separate display | DO | |
| 2 | Eq. 2: point the reader at the geometry figure; cite `fig:projected_mclachlan_geometry` | DO | figure is already in and builds |
| 3 | Reframe limitations: finding useful tangent directions is already solved; say so and cite AVQDS as the solution | DO | |
| 4 | Limitations: QPU noise depends on ansatz construction, so the ansatz is one bottleneck; current formulations are monotonic in manifold rank, approximately monotonic in noise and support size; shots are the second bottleneck, tied to both the ansatz and the integration controller; segue to how APM addresses each | DO | this is the new contribution framing |
| 5 | Name the prepared seed, with `U(\theta)` appended to it; state deletion candidates live in `U(\theta)`, which grows ADAPT-like in ordinary McLachlan and is variable / non-monotonic in APM | ASK | the user said "Cyref" — confirm the symbol (`C_{\rm ref}`?) |
| 6 | Drop the numerical focus on ridge / eigenvalue cutoff / damped inverse — every method does this; the actual secondary contribution is the controller | DO | consistent with items 30, 34 |
| 7 | Move Sec. II A (time-point geometry) into the introduction — it defines standard McLachlan and nothing more | DO | |
| 8 | After it, consider a short subsection defining the AVQDS extension, which is where Eqs. 7–8 belong | DO* | fold into the same move as item 7 |
| 9 | Title "McLachlan time-point geometry and the two failure modes" is redundant | NO | no better candidate offered; leave until the reordering settles |
| 10 | Add: McLachlan needs a circuit, so the available circuit is NISQ-satisfying; but because McLachlan grows circuits, the dynamics is NISQ-*compatible* and need not actually be runnable on a QPU | DO | |
| 11 | "ADAPT-style state prep" is over-restrictive — any ground-state preparation qualifies | DO | |
| 12 | State the method as an interpolation between fixed-structure VQE and resource-aware ADAPT: precondition the initial state, then run resource-aware ADAPT on it | DO | |
| 13 | Define the numerical methods once, early, as a named group, so later text contrasts "the group" against the exact solve instead of re-listing | DO | fixes the repetition item 6 targets |
| 14 | "the numerical lane chooses how to realize motion on the accepted fixed support" — vague and archaic now that there is one lane | DO | |
| 15 | Reference the compiled-cost appendix earlier than it currently appears | DO | |

## Notation

| # | change | verdict | note |
|---|---|---|---|
| 16 | Ideal velocity becomes `v^\star`; realized velocity becomes `\dot\theta^\star`; the free argmin variable becomes `\dot\theta` | ASK | **blocking.** The dictation gives three partly conflicting mappings ("v star for the ideal", "the old theta dot star is v star", "the old theta dot is theta dot star", plus a "dot beta star"). One table of old symbol -> new symbol settles the whole manuscript; nothing else in this list should be written first. |
| 17 | Replace `\mathcal P` and eliminate `C^{\mathcal P}` | ASK | user reasoned both ways: patch is more general than exchange (`\|D\| \ne \|A\|` allowed), so "patch" survives as the word. Proposal to confirm: `\mathcal P = (D, A)` stays the *patch*, and the patched support becomes plain `P`, replacing `C^{\mathcal P}`. |
| 18 | Where `U(\theta)` is introduced, state `D \subset \vec\theta` and `A \cap \vec\theta = \varnothing`, noting an inserted generator may repeat an existing operator type | DO | depends on 5 and 17 |
| 19 | "pure append" -> "pure insertion" throughout | DO | |
| 20 | "component-wise velocity" -> "parameter velocity" | DO | |
| 21 | Retained metric eigenmodes: say they are whitened | DO | matches `eq:app_whitening` (B13) |
| 22 | `m(\dot\theta)` is used before it is defined in the AP-McLachlan opening paragraph | DO | resolved by the item-7 reordering |
| 23 | The opening structural paragraph introduces many symbols before they earn definition; most need not be symbolic yet | DO | |
| 24 | Audit and cut the repeated `A \ne \varnothing` / `D \ne \varnothing` statements of the pure-insertion and pure-deletion limits — once, at most twice | DO | currently stated at tex 151, 411, 437, 443, 787, 808 |
| 25 | Remove every "exact trajectories do not enter the enumeration/ranking/commit" sentence | DO | |
| 26 | Every reported numeric must cite a symbol or an equation number, not a prose description | DO | applies across Sec. V |

## Equations

| # | change | verdict | note |
|---|---|---|---|
| 27 | Eq. 12 line 1: state how it is obtained, and add a prose sentence bridging line 1 to line 2 (the stationarity substitution) | DO | |
| 28 | Eq. 13: drop `\ge 0` from the display; say "non-negative" in prose | DO | |
| 29 | "Eqs. 13–14 remain the exact-projection reference" must say what is used instead — `\Gamma` starred or unstarred, not the retired `Q` | DO | ties to item 16 |
| 30 | Eq. 15: drop `a_D` as a defined symbol; substitute it directly into Eq. 16 | DO | |
| 31 | Eq. 17 and elsewhere: drop the fractional form, display the numerator, state the `\|\bar d\|_2^2` normalization in prose | DO* | apply to 17, 20, 21; keeping the denominator in 21 may be worth it to simplify Eq. 22 — user raised both, see item 36 |
| 32 | Say Eq. 16 holds when the change is state-local and Eq. 17 when there is no tangent-plane loss; the first is the refit bound (shot cost), the second is the quantity of interest | DO | |
| 33 | Drop the `_{\rm refit}` subscript on `R`; describe it as a ball in parameter space restricting the search, i.e. a shot-cost restriction, not a state-space one | DO | |
| 34 | Move the `10^{-14}` out of the method section into results | DO | |
| 35 | Do not define `z_{\mathcal P}`; substitute the braket definition directly, optionally carrying the global phase in Eq. 19 | DO | |
| 36 | Eq. 21: drop the tilde on `\Delta\Gamma`, since `\Delta\Gamma` is never defined | DO | |
| 37 | Eq. 22: is there a form without two appearances of `\tau`? | ASK | see item 44; depends on whether `\tau` is an indicator |
| 38 | Eqs. 23 and 24: move the argmax subscript restrictions into prose | DO | |
| 39 | Eqs. 25–27: drop the inequality and the equality; display only `\Delta t\\|\dot\theta\\|_\infty`, `\Delta t\sqrt{\dot\theta^\top G\dot\theta}`, and the third norm | DO | |
| 40 | Make the relation of Eq. 26 to Eq. 27 explicit; consider one joined display | DO | |
| 41 | "reducing the time step", not "changing" | DO* | user allowed either; "reducing" is the accurate one |

## Needs the source checked first

| # | question | what to check |
|---|---|---|
| 42 | Is velocity continuity (Eq. 20) actually enforced, or only recorded? The user believes it never decided a trajectory | `exchange_selector.CertificationGates.smoothness_eta_max`, and whether a finalist is ever rejected on it in a locked run |
| 43 | Is AVQDS's parameter-motion cap genuinely not state motion? The relationship needs stating, not asserting | `avqds_stepping.py` against `eq:state_space_step`; derive the exact relation |
| 44 | Is `\tau` an indicator on `m(\dot\theta) > \tau_m`, or a continuous allowance? | `exchange_config`/`exchange_structural` shifted-score path |
| 45 | What is "the work budget"? Needs pseudocode before prose | `JointWorkGuard`, `max_joint_patch_evaluations` |
| 46 | What are "time-point finalist family", "warm a memo", "surviving angles are transported", "remap the insertion positions"? Each needs a definition the source supports | `exchange_structural.iter_structural_families`, `structural_cache.solve_memo`, `state_with_appended_terms`, appendix pseudocode |
| 47 | Why is a finiteness test applied to an angle? Suspected implementation clutter | `deletion_permission`, `exchange_certification` |
| 48 | When is the position cost compilation taken for a deletion branch? | `append_cost` / `prune_cost` call sites |
| 49 | The caps that bound the *search space* matter more than the two certification caps that bound failure, and are barely defined | `max_structural_pool_size`, `max_joint_patch_evaluations`, rung admission |
| 50 | Wall-clock of one checkpoint search (128 pool entries, <= 50,000 complete patches), excluding quantum measurement | measure on the calibration seed |
| 51 | What method propagated the state, stated exactly | integrator selection in `adaptive_trajectory` |
| 52 | "exact Schrödinger motion generally leaves a finite variational manifold" — source, and is it too strong (a ground state need not leave)? | citation audit |

## Results section

| # | change | verdict | note |
|---|---|---|---|
| 53 | Expand the Pauli-string notation; put the electronic register first and the phonon register second | DO | |
| 54 | 501 is the *minimum* number of reporting times; subdivision raises the realized count, and the actual count lies between | DO | affects tex 497-ish and every "501" mention |
| 55 | The combined controller bound no longer exists — delete that sentence | DO | |
| 56 | Explain why the insertion-activation thresholds differ per cell: they equalize APM and AVQDS accuracy so that cost is the only free axis, since arbitrary accuracy is reachable and a comparison at unmatched accuracy is meaningless | DO | this is the matched-accuracy rule's actual justification and it is currently missing |
| 57 | Justify the zero deletion-history weight and absent cooldown: a plausible implementation would carry a patience period or a history criterion; both are set to zero here | DO | |
| 58 | One prose sentence saying the Eq. 22 cost term is a marginal, population-relative compiled cost over two-qubit count, two-qubit depth, and total depth; put the weights in the appendix, not the text | DO | |
| 59 | State how long a checkpoint search takes | CODE | see item 50 |

## Deferred to a new subsection

| # | change | verdict | note |
|---|---|---|---|
| 60 | Move the multi-insertion batching block out of the method and into a short later subsection near the bounded-search discussion, stating that the formulation is generic over multi-generator supports while the reported results use singletons only | DO | |
| 61 | Time-integration convention: give it an appendix, or cite an external description | DO* | prefer the citation; an appendix only if no clean reference exists |

## Blocking

Items 16 and 17 gate everything else — a symbol swap applied after prose edits
means editing the same sentences twice. Nothing gets written until the old ->
new symbol table is fixed.

## 2026-09-05 — notation swap applied (candidate items 16, 17)

Decisions taken by the user: the star marks the argmin of its own problem, and
the letter carries exact-versus-realized. `\dot\theta^\star` is optimal under
the numerical constraints, so it keeps the star.

| role | before | after | count |
|---|---|---|---|
| free variable in the objective | `v` | `v` | unchanged |
| exact minimum-norm minimizer | `\dot\theta^\star` | `v^\star` | 9 |
| exact, conditioned on a block | `\dot\theta_{X\mid B}^\star` etc. | `v_{X\mid B}^\star` etc. | 17 |
| realized / selected velocity | `\dot\theta` | `\dot\theta^\star` | 43 |
| realized on the patched support | `\dot\theta_{\mathcal P}` | `\dot\theta_{P}^\star` | 5 |
| realized component | `\dot\theta_\mu` | `\dot\theta_\mu^\star` | 1 |
| patched support | `C^{\mathcal P}` | `P` | 6 |
| capture on the patched support | `\Gamma_{\mathcal P}` | `\Gamma_{P}` | 4 |
| mismatch on the patched support | `m_{\mathcal P}` | `m_{P}` | 1 |

The patch itself stays `\mathcal P = (D, A)`, so `\widetilde{\Delta\Gamma}_{\mathcal P}`
(15 occurrences), `S_\tau(\mathcal P)`, `K(\mathcal P)` and `c(\mathcal P)` are
untouched -- those take the patch as an argument rather than naming a support.

Every substitution asserted its expected occurrence count before running, so a
silent partial rename was not possible. The pass produced two double
superscripts, `\dot\theta^\star^\top` at tex 482 and 491, from transposes that
previously sat directly on `\dot\theta`; both are now `(\dot\theta^\star)^\top`.
Build clean afterwards, no undefined references, 345 KiB.

Pre-swap source preserved at
`scratchpad/tex_before_notation.tex` for this session.

### Collision to decide

The figure artwork labels the chain components `\dot\theta_1`, `\dot\theta_j`,
`\dot\theta_3` without stars, and it is now baked. The caption's one component
reference became `\dot\theta_\mu^\star` under the swap, so caption and artwork
differ by a star on the same quantity. Options: leave it (a reader reads them as
the same symbol), drop the star in the caption alone (inconsistent with the
text), or relabel the artwork. Nothing else in the manuscript is affected.

### Still open from item 5

The seed symbol is `\psi_{\rm ref}`, which the manuscript already carries once.
The change itself -- naming the seed where the route is introduced and stating
that deletion candidates live in `U(\theta)`, ADAPT-like and monotone in
ordinary McLachlan but variable and non-monotone in APM -- is prose and is not
written yet.

## 2026-09-05 — RETRACTED: "the score in the manuscript is not the score in the code"

**That entry was wrong and is retracted in full.** Eqs. 22-24 stand. The
manuscript's shifted score is exactly what the reported route computes.

`pipelines/time_dynamics/ap_mclachlan/population_scoring.py:578`:

    def shifted_score(merit, K, tau=0.0):
        return float((m + t) / k - t)          # S = (m + tau)/K - tau

with `RankRegime.LARGE_MISS` -> `tau = 0`, eligibility `m > 0` strict, and
`RankRegime.MAINTENANCE` -> `tau = tau_del`, eligibility `m >= -tau_del` and
`S > 0` (population_scoring.py:52-63, 571-576). `K` comes from
`PopulationCostNormalization().seal(...)` and `sealed.k_of(patch_id)` at
`exchange_integration.py:646-661`. The route bundle is
`EXCHANGE_ROUTE_BUNDLE_Q_COMPILED = "paper_ii_exchange_q_compiled_v1"`
(`exchange_config.py:29`). The docstring even states the property the shift
exists for: at `m = 0` a maintenance deletion scores
`S = tau_del (1/K - 1) > 0` and beats stay whenever `K < 1`.

The user reports verification on 904 stored ranked rows across two
reported-route runs: `max|S - ((m+tau)/K - tau)| = 0.00`, with `K` non-constant.

### Root cause, and it invalidates more than one entry

**This checkout is `main`. The reported Paper-II route does not live on `main`.**
`population_scoring.py` was added by `a0c27ad2` ("paper-ii: population cost
normalization and shifted Paper-I kernel") and exists on
`claude/paper-ii-corrected-campaign-20260905`,
`codex/metric-whitened-avqds-matrix-20260903`,
`codex/paper-ii-dual-selector-overnight-20260903`,
`codex/paper-ii-exact-estimator-telemetry-20260903`,
`codex/siam-repaired-proxy-20260903`, and seven worktrees under
`Holstein_test.worktrees/`. It is absent from `main`.

The three-term score recorded in the retracted entry is real, but it is `main`'s
**legacy proxy path**, `paper_i_proxy_denominator_v1`
(`append_cost.py:18`) -- superseded, and not what produced any reported number.

The failure was searching one checkout and reporting the result as absolute.
`agent_guidance/shared/` already records the same class of mistake for the SIAM
lineage (the r07 engine lives only on the search-profiles branch). The rule that
follows: **no claim about the reported route may be made from `main`.** Resolve
the branch first, then read.

### Consequence for the verification pass below

Every agent in workflow `wf_3b7cfab3-04d` read `main`. Its findings therefore
describe the legacy path wherever `main` and the reported-route branch diverge,
and the following must be re-checked against the correct worktree before any of
it reaches the manuscript: Q45 (work budget), Q46a (ranking key), Q49 (search
caps), Q43 (time-step lane), Q51 (integrator and reporting grid). The Q44 agent,
which this worklog accused of fabricating `population_scoring.py`, was reading
correctly; that accusation is withdrawn.

## 2026-09-05 — code verification pass: 12 questions, 24 agents

Workflow `wf_3b7cfab3-04d`, one investigator and one adversarial refuter per
question. 1 answer survived unrefuted, 11 were corrected. The corrections are
refinements to load-bearing details, not reversals of the central answers.

### One agent fabricated a source file — its citations are discarded

The Q44 refuter cited `population_scoring.py:714`, `:571-576` for the location
of a `tau` assignment and a `RankRegime` enum. **That file does not exist**
anywhere under `pipelines/`. Re-checked directly: `find pipelines -name
population_scoring.py` is empty, and `rg '\btau\b'` over
`pipelines/time_dynamics/ap_mclachlan/*.py` and
`pipelines/time_dynamics/generalized_exchange.py` returns nothing.

**The earlier finding stands: there is no `\tau` in the implementation.** It is
now confirmed twice, once by direct reading of `ExchangeScoreConfig` and
`exchange_structural.scored`, once by exhaustive grep. Nothing from the Q44
agent is carried forward.

Spot-checked the other agents' new structural claims before recording them:
`generalized_exchange.py` really does define `insertion_face_open`,
`accuracy_debt`, `EXCHANGE_RANKING_SIGNED_DRIFT` and `debt_policy` (lines 22,
53-59, 118-125), and the euler promotions manifest really carries 12
`--integrator` entries. Those agents are grounded.

### Findings that change a candidate item

**Q51 — "501" is exact, not a minimum (contradicts candidate item 54).** 501 is
the count of *reporting* times. Local subdivision changes the number of
*internal* substeps between reporting times, not the reporting grid. The two are
different quantities and the candidate item conflates them. The manuscript
sentence is correct as written; what is missing is a sentence distinguishing the
reporting grid from the realized internal substep count. All 12 promoted actions
pass `--integrator euler`.

**Q45 — the work budget truncates the search at almost every checkpoint (the
one unrefuted answer).** The unit is typed patches `(D, I)`. A whole family is
pre-charged at admission before any member is scored, and a single refusal sets
`rejected_family`, which makes `exhausted` permanently true: every higher
deletion cardinality is then never enumerated and the multi-child frontier stage
is skipped entirely. At the canonical `max_joint_patch_evaluations = 50000` the
recorded SIAM Nb=4 screen refused a rung at **98 of 100 recorded time points**,
most often `singleton_d6`. The manuscript says rungs are "admitted whole or not
at all against the joint-work budget"; it does not say that one refusal ends the
structural search for that time point, which is the operative behaviour.
Also: a rung's size is computed *before* `deletion_permission` is consulted, so
refused deletion sets are charged against the budget without ever being scored.

**Q46a — `debt_ranking` is not set by `debt_policy` alone.** The lexicographic
key (quantized `delta` primary, `tanh` of the utility sum secondary) applies only
when `accuracy_debt AND debt_policy == "drift_ranked"`, where `accuracy_debt`
requires `l2_debt_enabled` and `checkpoint_l2 > l2_cut`
(`generalized_exchange.py:118-125`, reached from `exchange_integration.py:657`).
Outside L2 debt the ordering is the scalar score. This refines the earlier
worklog entry on the implemented score.

**Q49 — the insertion gate, not the work guard, is the dominant restriction.**
`insertion_l2_cut` closed the entire insertion face at 46 of the 100 checkpoints
carrying a work-guard record, taking the candidate pool from 24 atoms to 0.

**Q43 — the raw-coordinate lane sets the committed internal grid.** In the cited
run the raw-coordinate cap set the local-subdivision depth in 375 of 375
subdivided intervals; the recorded severity matches
`dt*||theta_dot||_inf / 5e-5` to 4.5e-6 relative, never the state-motion lane.
Sum of `2^depth` over those intervals equals the recorded
`accepted_internal_substep_count` exactly.

Full agent output: `/private/tmp/.../tasks/wuskz6epm.output` (434 KB, this
session only). Per-agent transcripts under the workflow journal.

## 2026-09-05 — figure finished; first mechanical batch applied

**Figure.** Left clipping was in the source: content ran to `x = -43` against a
`viewBox` starting at `0`. Widened to `viewBox="-48 -5 1504 1096"`, page box
1128 x 822 bp, same 2623 path operators and three axial shadings.

The three chain labels are now typeset by LaTeX rather than baked, so they carry
the starred velocity. The three glyph runs at `translate(678.84,937.13)`,
`(862.84,849.13)` and `(995.84,766.13)` were removed from the drawing and
replaced by `tikzpicture` nodes at the mapped baseline origins
(`x_pdf = (x_svg+48)*0.75`, `y_pdf = 822-(y_svg+5)*0.75`), 36.5 bp,
`anchor=base west`. Verified by rasterizing page 4 with `pypdfium2` and reading
it: placement matches the reference image and nothing is clipped. This also
closes the caption/artwork star collision recorded earlier -- both now starred.

**Candidate items applied.**

| # | change | count |
|---|---|---|
| 19 | "pure append" -> "pure insertion" | 6 |
| 20 | "componentwise velocity" -> "parameter velocity" | 1 |
| 25 | "Exact trajectories do not enter the enumeration, ranking, or commit tests." removed | 1 |
| 25 | "Exact reference trajectories serve only the post-run comparisons." removed | 1 |
| 28 | `\geq 0` removed from Eq. 13; non-negativity stated in the following sentence | 1 |
| 41 | "changing the time-step acceptance norm" -> "reducing the time step under a different acceptance norm" | 1 |

Build clean throughout.
