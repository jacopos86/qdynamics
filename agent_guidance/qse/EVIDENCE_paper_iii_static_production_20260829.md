# Paper III static-QSE production evidence — 2026-08-29

## Status

The static and frequency-domain Paper III result package is complete for the
locked 12-cell Hubbard--Holstein matrix. Time propagation is absent by design
and is deferred to a later paper.

The completed result is a pure fixed-pool selected-QSE method, not a
selected-QSE-to-Davidson hybrid:

- selection uses full proposed-pencil Ky Fan gain per incremental cost;
- the supported-image novelty floor remains active;
- no additive novelty reward is present;
- certified atomic exchange remains active;
- the fixed pool is the coverage-qualified encoded-boson completion;
- pool insufficiency remains an explicit terminal status;
- direct Davidson is an independently initialized energy/cost comparator;
- complete fixed linear-response QSE is the conventional static spectral
  baseline.

## Source locks

- Branch: `paper-ii-exchange-selector`
- Acquisition implementation: `4542a895b7670cfd4233f67b1ed4447f05eec510`
- Acquisition evidence promotion: `8cb31cb029098178ace4362e663bca36ee159d42`
- Static production reporter: `a4eaf578257dd24c9c1841ff7db47164bbc702c7`
- Selector policy:
  `full_pencil_ky_fan_cost_ratio_novelty_floor_no_reward_v1`
- Pool policy:
  `hh_full_meta_plus_encoded_boson_pauli_completion_v1`
- Acquisition source-lock digest:
  `096888afb3a211fbee75ab50f89d63d2d7b30bcf67320c6821d87b3aa3f64080`

## Production results

All 12 selected-QSE cells terminate as `TARGET_WINDOW_CERTIFIED` and reach
both reported exact-error targets in the post-run audit.

| Result | Outcome |
|---|---:|
| Online certified target windows | 12/12 |
| Reached maximum six-root error $10^{-4}$ | 12/12 |
| Reached maximum six-root error $10^{-6}$ | 12/12 |
| Selected $N_{2q}\leq$ direct Davidson at $10^{-4}$ | 10/12 |
| Selected $N_{2q}\leq$ direct Davidson at $10^{-6}$ | 12/12 |
| Maximum selected/direct $N_{2q}$ ratio at $10^{-4}$ | 1.407 |
| Maximum selected/direct $N_{2q}$ ratio at $10^{-6}$ | 0.929 |

The only $10^{-4}$ losses are `weak_weak_nph3` (242 versus direct 172) and
`intermediate_weak_nph3` (242 versus direct 240). The completion pool repairs
the two genuine base-pool failures:

- `weak_strong_nph3`: selected 428 versus direct 852 at $10^{-4}$;
- `intermediate_strong_nph3`: selected 428 versus direct 1124 at $10^{-4}$.

The terminal selected supports were re-solved with site-0 density and
displacement observables. Across 24 case/observable broadened spectra, the
selected-QSE normalized spectral $L^1$ error has median
$1.73\times10^{-9}$ and maximum $2.00\times10^{-5}$. The complete fixed
linear-response class has median $3.61\times10^{-3}$ and maximum 12.20. It
can reproduce bright peaks in some cells while badly misrepresenting the
six-root window or inserting spurious spectral weight in others; this is why
it remains a static baseline rather than the adaptive growth comparator.

## Evidence artifacts

All full numerical artifacts are in the canonical repository's ignored
`output/` tree. The hashes below reproduced byte-for-byte on an immediate
second execution of the committed reporter.

| Artifact | SHA-256 |
|---|---|
| `output/pdf/paper_iii_static_qse_production_20260829_v1/paper_iii_static_qse_production_evidence.json` | `72aa1ee34464bff63b389d8a762e0d9142424920cd6623d2a95b623cb07d8426` |
| `output/pdf/paper_iii_static_qse_production_20260829_v1/paper_iii_static_qse_production_evidence.md` | `f156c42af61a095de60b2fe910396c20791bbc85d85c887ec87c183e88935054` |
| `output/pdf/paper_iii_static_qse_production_20260829_v1/paper_iii_static_qse_production_report.pdf` | `d5aa0418ad0d37981426b511bdbbdec653b28e56200cdd67a47e5f3bee49471f` |
| `output/pdf/paper_iii_static_qse_production_20260829_v1/static_cost_ratios.png` | `48138b8647485e4db2c23e875e0a0de705c57f5e9aea74d66c9db3e112899a14` |
| `output/pdf/paper_iii_static_qse_production_20260829_v1/static_spectral_errors.png` | `c8c5d9d526e38327f123de9ce0854fed1111f78dd3633efae8042c39d9aaa42d` |
| `output/pdf/paper_iii_static_qse_production_20260829_v1/static_hard_case_spectra.png` | `71802a509d1c7cb6c35af47d1dfe966380d228d4556f1a08f7caf5b7e20e88a7` |

The three-page PDF was rendered to PNG and inspected page by page. It has no
clipped labels, overlaps, broken glyphs, or unreadable panels.

## Scientific boundary

These runs use the provider's exact sector ground state as a statevector seed
to isolate the QSE acquisition and diagonalization method. The manuscript
must describe the benchmark that way. A general method statement may accept
an optimized prepared ground state, but the paper must not claim that these
12 numerical rows used imported Paper-I variational seed artifacts.

Exact roots, exact transition strengths, and exact broadened curves are
post-run references. They select no record, stop no route, certify no online
window, and trigger no fallback.

## Verification

- New reporting contract: 4 passed.
- Reporter plus selector/pool/completion/exchange suite: 49 passed.
- Full committed-source report run: complete in approximately 14 seconds.
- Deterministic replay: all six reported artifact hashes identical.
- PDF: three pages, visually verified after final regeneration.
- No Paper III manuscript source was edited in this increment.
- No CHTC job was submitted.

The manuscript-finalization work is specified in
`agent_guidance/qse/HANDOFF_paper_iii_static_manuscript_finalization_20260829.md`.
