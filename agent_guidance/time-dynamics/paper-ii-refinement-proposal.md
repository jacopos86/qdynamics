# Paper II refinement proposal (2026-09-05)

Candidate changes only. Nothing here is applied.

## Equations

| # | eq | proposal |
|---|---|---|
| 1 | Eq. 2 `mclachlan_velocity` | Drop the argmin half. Keep `\dot\theta^\star=G^{-1}f`, with $G$ the Gram matrix and $f$ the force vector defined on the spot. The least-squares statement is already carried by the miss functional later, so the pair is redundant. |
| 2 | Eq. 15 `deletion_prepermission` | Split the `aligned` block into three numbered equations: the accumulated angle $a_D$, the ray bound $\widehat\ell_D$, and the normalized loss. Put $\geq0$ on the same line as the loss expression rather than on its own row. |
| 3 | Eqs. 21/22 | Pick one convention and use it in both. Eq. 22 currently carries three conditions under the argmax ($\mathcal P=(D,\varnothing)$, $\widetilde{\Delta\Gamma}\ge-\tau$, $S_\tau>0$); the first is the population definition and belongs in the surrounding sentence, and the third is implied by taking a maximum over a set that the text already says is nonempty. Recommend: keep only the admissibility condition under the argmax in both, state the population in prose. Eq. 21 then reduces to a maximization of $\widetilde{\Delta\Gamma}_{\mathcal P}/K(\mathcal P)$ over the insertion-bearing population. |
| 4 | Eq. 24 `state_space_step` | State the $\delta\theta$ convention here: the two controllers cap the same $\Delta t$ against different norms, and the raw-parameter rule of Eq. 23 is the published AVQDS convention the comparators run. |
| 5 | Eqs. 23/25 | Confirmed as the user supposed: Eq. 23 is the standard AVQDS raw-parameter rule; Eq. 25 is the whitened (retained-metric-eigenmode) controller that produced the step reduction in the appendix ablation. |

## Prose

| # | proposal |
|---|---|
| 6 | Remove negative and contrasting constructions throughout (`does not`, `whereas`, `rather than`, `not free`, `fails`). State what holds. The two claim-boundary sentences in Sec. V B need rewriting as positive scope statements. |
| 7 | Results opening: replace the multi-clause seed description with "the seed is prepared by Resource-Aware ADAPT~\cite{Strobel2026StaticAdapt} on an electron--phonon-preconditioned Hartree--Fock reference" and stop there. |

## Tables

| # | proposal |
|---|---|
| 8 | Move $N_{\rm step}$ next to $N_\theta$, so the two counts sit together ahead of the compiled-resource block. |
| 9 | Add $(2.4,\,3)$ state motion as a pending row: locked in the matrix, never run. |
| 10 | Table II: replace $W_{\rm geom}$ with $\mathcal E$. There is no reason to carry two estimator quantities. |

## Estimator accounting

| # | proposal |
|---|---|
| 11 | **Retire $W_{\rm geom}$.** It is the propagation-only sum and predates the instrumented reruns. Appendix~A currently derives it, then says $\mathcal E$ is a future target that "will combine with $W_{\rm geom}$" -- that sentence is stale, because $\mathcal E$ is now reported exactly in Table I. Rewrite the appendix to define $\mathcal E$ directly as the sum of integration, checkpoint base, candidate acquisition, certification and refit, and delete the $W_{\rm geom}$ symbol from the paper. Three body sentences reference it and would be restated in $\mathcal E$. |

## Response-spectra figure

| # | proposal |
|---|---|
| 12 | Cut each panel's $y$-axis at the floor instead of shading below it. The axis currently runs to $10^{-11}$ so a third of every panel is empty gray. Clipping removes the gray entirely and the panel then shows only what the comparison scores. |

## Figures

Currently seven; the instruction was three. Candidate three:

| keep | why |
|---|---|
| workflow | orientation, not a result |
| $(2.4,3)$ $\delta\theta$ | widest support separation that passes the accuracy rule |
| response spectra | the only frequency-domain result |

That drops all three state-motion trajectory panels and the $(0.3,1)$ $\delta\theta$ panel. If the state-motion panels are the ones to keep instead, the workflow and one trajectory panel come out. Needs a decision.
