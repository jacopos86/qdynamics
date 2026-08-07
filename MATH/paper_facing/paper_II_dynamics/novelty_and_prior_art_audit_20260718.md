# Paper II Novelty and Prior-Art Audit

Status: support ledger, not manuscript prose  
Created: 2026-07-18  
Target: `MATH/paper_details/time_dynamics_paper_II.tex`  
External audit source SHA-256: `428c2674bbfa20350de7ee30b06cd92b934c51c0745cf1df6ee1f2e38cacd482`

## Evidence Boundary

This document records the conclusions and citation leads from an external Deep
Research audit. The audit is a research lead, not a primary source. Every
literature claim and every citation must be verified against the original
paper, patent, dissertation, or software record before it enters Paper II.

The audit searched for overlap with the complete AP-McLachlan route rather than
assuming that the supplied novelty hypotheses were correct. Negative-search
results are bounded by the reviewed literature through 2026-07-18 and must not
be rewritten as unqualified claims of being first, unique, or unprecedented.

## Executive Boundary

The established foundation comprises projective McLachlan propagation,
regularized metric inversion, residual-triggered adaptive ansatz growth, and
zero-initialized append operations. Paper II must cite these as prior art.

The strongest candidate contribution is the combination of:

- one checkpoint-level finalist family containing stay, pure append, pure
  prune, and true delete-plus-append exchange;
- conditional gain and deletion-loss evaluation on the patched support;
- grouped prune loss, persistence, conditioning-aware nomination, and
  same-checkpoint continuity certification for deletion-containing patches;
- separate structural and numerical miss diagnostics with bidirectional
  inverse-policy repair, local subdivision, and release hysteresis; and
- a driven Hubbard--Holstein gate-based demonstration with online support
  maintenance.

The application claim is narrow. The audit found extensive prior work on
electron--phonon simulation, Hubbard--Holstein simulation, and spin--boson
McLachlan dynamics. It did not find an earlier gate-based adaptive McLachlan
real-time study of driven Hubbard--Holstein with online support maintenance.

## Claim Ledger

| Paper-II element | Audit classification | Required positioning |
|---|---|---|
| Projective McLachlan propagation | Established | State that Paper II builds on standard projective McLachlan VQS. |
| Residual-triggered append | Established | Cite AVQDS, adaptive pVQD, and adaptive open-system VQS. |
| Zero-initialized append | Established | Do not claim state-continuous zero-angle append itself as new. |
| Drive-aligned zero-amplitude tangent | Possible secondary distinction | Treat cautiously; a targeted driven-response search is still required. |
| Cost-weighted combinatorial batched append | Possible combination novelty | Present as a refinement of adaptive growth, not the invention of batched growth. |
| Schur novelty plus augmented-solve confirmation | Possible algorithmic combination novelty | Compare explicitly against AVQDS, adaptive pVQD, and AVQDS(T). |
| Failed-append certificate reuse and secant retry | Possible controller-contract novelty | Verify against trust-region/model-management and adaptive-VQS records. |
| Grouped batched prune loss | Strong candidate setting-specific novelty | Contrast with Pruned-ADAPT-VQE and adaptive basis/rank methods. |
| History- and conditioning-aware prune nomination | Strong candidate combination novelty | Keep current full-support deletion loss as physical authority. |
| Atom-history persistence across support growth | Possible implementation/controller novelty | Claim only if implementation and run telemetry substantiate it. |
| Same-checkpoint prune smoothness and deferral | Strong candidate controller novelty | Distinguish prune-worthy from smoothly removable now. |
| Unified stay/append/prune/exchange selection | Strong candidate combination novelty | This is the main route-level framing candidate. |
| Conditional true exchange | Strong candidate algorithmic novelty | Requires empirical evidence that true exchange finalists are evaluated and, for a result claim, accepted. |
| Numerical-miss diagnostic | Possible combination novelty | Do not claim ridge, pseudoinverse, or eigencutoff as new. |
| Bidirectional inverse repair plus subdivision | Possible combination novelty | Frame as a state-space repair architecture built from established numerical ingredients. |
| State-space kink guard and release hysteresis | Possible controller novelty | Verify against adaptive integration and rank-adaptive dynamics. |
| Exact-reference-free online controller | Expected QPU-faithful practice | Novelty can lie only in extending the contract to prune/exchange decisions. |
| Driven Hubbard--Holstein application | Possible bounded application novelty | Never claim first Hubbard--Holstein quantum dynamics or first mixed fermion--boson McLachlan method. |

## Closest Prior-Art Families

### Direct adaptive variational dynamics

- McLachlan 1964; Li--Benjamin 2017; Yuan et al. 2019; Endo et al. 2020:
  projective variational dynamics and its quantum-circuit formulation.
- Yao et al., AVQDS: McLachlan-distance-triggered, zero-initialized,
  append-only adaptive dynamics.
- Barison--Vicentini--Carleo, pVQD, and Linteau et al., adaptive pVQD:
  projected dynamics and adaptive circuit growth, including driven examples.
- Zhang et al., AVQDS(T)/compressed AVQDS: disjoint-layer growth,
  eigentruncation, and reduced-measurement stabilization.
- Adaptive variational simulation for open systems: adaptive growth beyond
  closed-system dynamics.

### Quantum pruning precedent

- Pruned-ADAPT-VQE: operator deletion after ground-state optimization using
  parameter magnitude and position/history criteria. It is not online
  real-time McLachlan deletion, patched-support exchange, or a same-checkpoint
  continuity certificate, but it is mandatory prior art for quantum ansatz
  pruning.

### Cross-field support maintenance

- DP-MCTDH: online basis/configuration selection and pruning.
- Rank-adaptive dynamical low-rank approximation: enrichment, compression,
  rejection, and adaptive-rank integration.
- OMPR and stepwise subspace-pursuit methods: replacement-style add/remove
  sparse-support moves.

These records prevent claims that bidirectional support maintenance is new in
all of dynamics or optimization. The candidate distinction is its translation
and coupling to measured gate-based McLachlan tangent support.

### Mixed fermion--boson applications

- Macridin et al. 2018: electron--phonon and fermion--boson digital quantum
  simulation.
- Miessen--Ollitrault--Tavernelli 2021 and later spin--boson work:
  McLachlan-type real-time dynamics for mixed spin--boson systems.
- Denner et al. 2023: hybrid Hubbard--Holstein ground-state work.
- Kumar et al. 2025: digital-analog Hubbard--Holstein real-time dynamics.

## Claims That Must Not Appear

- APM invents adaptive McLachlan dynamics.
- APM invents residual-triggered or zero-angle append.
- APM is the first batched adaptive variational dynamics method.
- Ridge loading, Moore--Penrose inversion, eigencutoff, or local whitening is
  itself new.
- This is the first quantum simulation of Hubbard--Holstein dynamics.
- This is the first mixed fermion--boson McLachlan method.
- True exchange is empirically demonstrated before an accepted exchange event
  is present in the audited run matrix.

## Planned Runs and Results Framing

### Core ansatz-family and drive survey

The first result program compares AP-McLachlan propagation from the currently
displayed Paper-I Append-ADAPT and SNAKE ansatz prefixes. Use the four corner
Hubbard--Holstein regimes, skip the intermediate-Hubbard sectors, and apply both
weak and strong drives to each static seed:

| Hubbard--Holstein regime | Append-ADAPT prefix | SNAKE prefix | Drive amplitudes |
|---|---:|---:|---:|
| weak--weak | (k=23) | (k=30) | (A=0.2,0.6) |
| weak--strong | (k=23) | (k=50) | (A=0.2,0.6) |
| strong--weak, (U/t=8) | (k=6) | (k=9) | (A=0.2,0.6) |
| strong--strong, (U/t=8) | (k=8) | (k=33) | (A=0.2,0.6) |

This is a 16-job core matrix:

\[
4\ \text{regimes}
\times 2\ \text{static ansatz families}
\times 2\ \text{drive strengths}
=16\ \text{trajectories}.
\]

The method-specific prefix choices are the current visible Paper-I evaluation
points, not iteration-matched prefixes. Each Append-ADAPT/SNAKE pair must use
the same Hamiltonian, drive, time grid, AP-McLachlan controller, observables,
and reporting convention. Use

\[
t\in[0,5],\qquad \Delta t=0.005,
\qquad N_t=1001.
\]

Exact trajectories remain reporting-only.

For each trajectory, report:

- total energy, doublon, total site occupations, and spin-resolved site
  occupations against same-seed exact propagation and the ED-ground-state
  reference trajectory where available;
- structural and numerical residual behavior;
- stay, append, prune, and true-exchange finalist and acceptance counts;
- rejection reasons for deletion-containing and true-exchange patches;
- support size versus time, unsupported-checkpoint count, and solve-repair or
  local-subdivision activity; and
- final \(t=5\) compiled Qiskit ansatz cost under one common compilation
  contract.

### Factorial mechanism ablation

After the core survey, select one source-locked combination of static ansatz
family, interaction regime, and drive strength for a full mechanism ablation.
Prefer a completed case with true exchange; otherwise choose the case with the
clearest simultaneous append and prune scouting, the largest true-exchange
finalist or near-miss activity, and enough structural activity to discriminate
the controller branches. Do not select the ablation case by exact-trajectory
error alone.

Run the Cartesian product of three binary factors:

| Factor | Enabled | Disabled |
|---|---|---|
| Append | Pure append and the add side of true exchange are admissible. | No support additions; true exchange is unavailable. |
| Prune | Pure prune and the delete side of true exchange are admissible. | No support deletions; true exchange is unavailable. |
| Numerical stabilization | Canonical state-space solve repair, local subdivision, and release hysteresis intervene when triggered. | The adaptive stabilization controller is non-intervening; passive telemetry remains enabled. |

This gives eight runs:

| Ablation cell | Append | Prune | Stabilization | Route interpretation |
|---:|:---:|:---:|:---:|---|
| 1 | on | on | on | Canonical stay/append/prune/true-exchange controller |
| 2 | off | on | on | Prune-only support maintenance |
| 3 | on | off | on | Append-only support growth |
| 4 | off | off | on | Fixed support with canonical numerical stabilization |
| 5 | on | on | off | Full support patching without adaptive numerical intervention |
| 6 | off | on | off | Prune-only route without adaptive numerical intervention |
| 7 | on | off | off | Append-only route without adaptive numerical intervention |
| 8 | off | off | off | Fixed-support, non-intervening baseline |

The stabilization factor isolates the adaptive repair controller. Its disabled
level retains the same base pseudoinverse, cutoff, and nominal solve convention
as the enabled level; a separate ridge/eigencutoff sensitivity study is needed
if the base inverse itself is to be ablated. All other algorithmic settings,
the seed, drive, time grid, and reporting inputs remain fixed across the eight
cells.

The result discussion should use the factorial structure rather than treating
each run as an unrelated trajectory. Separate the main effects of append,
prune, and stabilization from their interactions, especially the append--prune
interaction that makes true exchange possible and the stabilization
interactions that determine whether structural patching remains smooth.

### Empirical gate for exchange language

Run the canonical support-patch controller on the four corner Hubbard--Holstein
regimes under both weak and strong drives for both static ansatz families.
Record, separately for each run:

- append, prune, and true-exchange finalist counts;
- accepted pure append, pure prune, and true-exchange counts;
- true-exchange rejection reasons and governing thresholds;
- final trajectory errors, support size, and compiled Qiskit cost.

If no true exchange commits across the 16-run core matrix, do not erase
exchange from the mathematical method. First determine whether:

1. append and prune scouts were simultaneously available;
2. true-exchange pairs were formed;
3. pair selection failed on utility, conditional gain/loss, Schur novelty,
   patched solve geometry, persistence/cooldown, or patch smoothness; or
4. the pure/stay finalist was genuinely superior.

Only after this audit can the manuscript decide whether exchange is an
implemented but inactive limiting branch, an under-parameterized route, or a
demonstrated empirical mechanism.

## Later Framing Handoff

Do not ask a framing agent to rewrite Paper II until the 16-run core matrix,
the selected eight-cell mechanism ablation, and any no-exchange investigation
are complete. The handoff should ask for a short structural plan and high-level
section changes, not line-level semantic polishing.

Use the Paper-I introduction sequence as the template:

1. modeled problem and physical importance;
2. current solution strategies;
3. limitations of current approaches;
4. the proposed solution and its bounded contribution.

For method subsections, use the Paper-I purpose-first opening rule: identify a
specific local failure, state the mathematical or algorithmic construction
used to address it, and name the score, reduced model, admissibility decision,
warm start, or controller object produced for the next stage. This is a
structural rule, not wording to repeat verbatim.

## Primary-Source Verification Queue

Before manuscript insertion, verify bibliographic metadata, chronology, and
the exact method overlap for:

- McLachlan 1964;
- Li--Benjamin 2017;
- Yuan et al. 2019 and Endo et al. 2020;
- AVQDS;
- pVQD and adaptive pVQD;
- AVQDS(T)/compressed AVQDS;
- adaptive open-system VQS;
- Pruned-ADAPT-VQE;
- TETRIS-ADAPT-VQE;
- DP-MCTDH;
- rank-adaptive dynamical low-rank approximation;
- OMPR/stepwise subspace pursuit;
- Macridin et al. 2018 electron--phonon and fermion--boson papers;
- Miessen--Ollitrault--Tavernelli spin--boson McLachlan work;
- Denner et al. 2023; and
- Kumar et al. 2025.

Update `papers_to_cite.md` and the bibliography only after this queue is checked
against primary sources.
