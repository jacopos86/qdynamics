# Paper III static manuscript status — 2026-08-29

## Authority split

Paper III now uses two deliberately different artifacts:

1. The **numerical source anchor** is the compact, reproducible ledger. It owns
   numerical values, route identities, first-crossing policy, hashes, failures,
   resource-accounting boundaries, comparator matrices, pool evidence, and
   static spectral audits.
2. The **static manuscript** is the expansive scientific framing layer. It owns
   the derivation, method presentation, relationship to Papers I and II,
   physical interpretation, limitations, and publication argument. It is
   intentionally over-complete so the author can prune it later without first
   reconstructing missing logic.

Neither artifact silently overrides the other. Numerical disagreements are
resolved in favor of the source anchor and its manifest; framing and exposition
are revised in the manuscript.

## Canonical artifacts

- Branch: `paper-ii-exchange-selector`
- Tested manuscript/source-anchor revision:
  `d1717ccdb11fbbda4548f9ada4065f5c9e170cf2`
- Static manuscript source:
  `MATH/paper_details/excited_spectra_static_paper_III.tex`
- Local compiled static manuscript:
  `MATH/paper_details/excited_spectra_static_paper_III.pdf`
- Numerical source anchor:
  `output/pdf/paper_iii_static_qse_source_anchor_20260829_v2/paper_iii_static_qse_source_anchor.pdf`
- Numerical manifest:
  `output/pdf/paper_iii_static_qse_source_anchor_20260829_v2/manifest.json`
- Source-anchor evidence contract:
  `agent_guidance/qse/EVIDENCE_paper_iii_static_source_anchor_20260829.md`

The generated PDFs and manifest remain in the ignored output/build trees. The
tracked LaTeX source, builder, tests, evidence record, and this status record are
the reproducible repository contract.

The older
`MATH/paper_details/excited_spectra_dynamics_paper_III.tex` is preserved as a
separate dirty/shared artifact and was not staged or modified for this static
manuscript increment. Frozen-QSE/McLachlan dynamics remain deferred to a later
paper.

## Integrated static evidence

- OS-QSE reaches the declared six-root `1e-6` task in 16/16 tested cases:
  twelve Hubbard--Holstein cells, three Peierls--Hubbard regimes, and one
  ten-qubit finite-box SIAM `N_b=4` transfer case.
- Independently initialized direct quantum Davidson reaches 16/16. OS-QSE has
  the lower summed Qiskit two-qubit structural annotation at each route's first
  crossing in all 16 cells.
- Single-seed RTK reaches 10/16 and fixed six-probe block RTK reaches 11/16.
  Their union is reported only as a post-run coverage diagnostic, never as an
  oracle comparator.
- The Hubbard--Holstein base physical-macro pool reaches 10/12 cells at
  `1e-4`; the encoded-boson completion pool reaches 12/12 under the same current
  selector contract.
- Terminal static spectra audit density, displacement, double-occupancy, and
  magnetization response independently of the six-root energy criterion.

## Manuscript contract

The new manuscript is static and no-fallback. It includes the ideal
minimum-`K` combinatorial objective and identifies the implemented greedy
surrogate explicitly. Its production selector is the full proposed-pencil
Ky--Fan gain per cost with novelty retained as an admissibility floor and no
additive novelty reward. It preserves:

- `r`, `\mathcal R`, `\Delta`, `K`, `\mathcal S=\Delta/K`, and `\mathcal B`
  as the cross-paper notation, with Roman `S` reserved for overlap;
- supported-overlap truncation rather than a ridge;
- retained-frame residualization with two Gram--Schmidt passes;
- the pre-six-root `h_{11}` contribution to acquisition and stopping;
- pool-relative stopping and structured terminal statuses, with no implicit
  Davidson completion;
- atomic singleton delete--insert exchange inherited from Paper II, including
  whole-patch re-solve and rejection semantics;
- direct Davidson, fixed-class QSE, single-seed RTK, and block RTK as separately
  declared comparators;
- earliest-successful-prefix accounting, one-to-one root assignment, and
  route-local resource boundaries;
- explicit separation between final-support structural cost and unmeasured
  cumulative acquisition work.

The manuscript intentionally includes extensive appendices and caveats. It is
ready for author-directed pruning, not yet declared submission-final.

## Verification

- Static manuscript: 1,996 source lines, 11,087 words, 20 letter-sized pages.
- Final manuscript log: no LaTeX/natbib warnings, unresolved references or
  citations, or overfull boxes.
- Page-by-page visual audit: no clipping, overlaps, broken tables, or missing
  figures. Only nonblocking narrow-column underfull boxes and generous white
  space remain.
- Scientific contract audit: all benchmark Hamiltonians, route identities,
  counts, costs, root conventions, and claim boundaries agree with the source
  anchor.
- Focused tests: 6 passed (five source-anchor contract tests plus one Paper-III
  reporting guard).
- Source anchor: two controlled consecutive builds at revision `d1717ccd`
  reproduced all output hashes byte for byte.

## Locked hashes

| Artifact | SHA-256 |
|---|---|
| `MATH/paper_details/excited_spectra_static_paper_III.tex` | `cb49ce165e04f8d1566bf2d0e4d913595c7e8071876bd8da6c1b68b5a6c77f60` |
| `MATH/paper_details/excited_spectra_static_paper_III.pdf` | `040a684d727161eb5280e36606900380d090a1c91b93ee96b0d58dbf0cfda383` |
| `output/pdf/paper_iii_static_qse_source_anchor_20260829_v2/paper_iii_static_qse_source_anchor.tex` | `2275e65b45678669457a8b1d3710e88c47d249ca8d390328e6cb9226de5ba713` |
| `output/pdf/paper_iii_static_qse_source_anchor_20260829_v2/paper_iii_static_qse_source_anchor.pdf` | `4a9d1224785484d34e448617a486bb912d61da7a93d76272cf77da9853e841a4` |
| `output/pdf/paper_iii_static_qse_source_anchor_20260829_v2/manifest.json` | `d7c274fd3e7387034a8227dc9d92cdba13a55d54037609dca0874864db3b74a7` |

## Explicitly open, nonblocking evidence

The demonstrated 16-case static result is complete under the current contract.
The following items remain valuable for stronger causal or broader claims, but
they are not silently promoted into the completed evidence set:

- current-policy cost-discount-off ablation;
- current-policy exchange-off ablation and accepted-patch ledger;
- cumulative acquisition-work accounting under a common comparator model;
- Hamiltonian-power QKSD;
- approximate-seed robustness;
- physical bosonic-cutoff convergence;
- numerical qEOM/q-sc-EOM, VQD, or SSVQE comparators if later retained in
  scope;
- finite-shot/hardware and time-dynamics evidence.

Until those are supplied, Paper III's supported claim is lower declared
structural construction cost for the tested static six-root task, not fewer
shots, lower total work, universal superiority over Krylov methods, or a
time-dynamics result.
