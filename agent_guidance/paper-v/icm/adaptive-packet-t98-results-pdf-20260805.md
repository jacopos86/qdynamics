# Paper V adaptive-packet `t=98` results-PDF flow

Status: `materialized; awaiting user review`.

This ICM workspace routes one completed Paper V evidence update. It does not
authorize another trajectory, threshold retuning, archive-closure rerun,
evidence promotion, commit, or push.

## Intent

Place the newest adaptive multi-coherent packet result at the top of
`output/pdf/paper_v_results_progression_20260804.pdf`. Show the contracted site
occupations and electronic, phononic, electron--phonon, and total internal
energies against the matched exact cutoff-16 Hamiltonian reference. Preserve
the existing one-column, plot-dump report structure and older pages.

## Lock

Physical protocol:

- exact-correlated central preparation;
- `t_hop=1`, `gamma=0.5`, `lambda=1.5`, and `g/t_hop=0.6123724357`;
- Gaussian double pulse with amplitude `1`, width `1`, and delays `0` and `8`;
- phonon cutoff `16`;
- adaptive packet checkpoints every `0.5`;
- native-residual trigger `relative >= 0.05` and `absolute >= 0.02`;
- unchanged three-objective admission test and state-preserving zero-weight
  packet admission;
- no exact-reference information in propagation or admission.

Trusted continuation source:

- `output/local_runs/paper_v_archive_gram_adaptive_packet_cutoff16_t40_20260805_v1/`.

Interrupted execution workspace:

- `output/local_runs/paper_v_archive_gram_adaptive_packet_cutoff16_t100_resume40_20260805_v1/`.

The user stopped the run after declaring the long-horizon evidence sufficient.
The last complete full-trajectory checkpoint is `t=98`; the isolated current
state reached `t=99` but is not used for trajectory scoring.

## Execute receipt

The stored `t=40` state resumed successfully. The previously terminal `t=40`
admission check did not trigger. The adaptive trajectory then admitted packets
at

```text
41, 48.5, 54.5, 67.5, 72, 75.5, 90, 90.5, 92.5, 93, 93.5,
94, 96.5, 97, 97.5, 98
```

in addition to the earlier admissions at `0, 0, 8, 19, 22, 26.5, 38`.
Capacity therefore rose from `K=11` at the continuation boundary to `K=27`
packets per electronic branch at `t=98`. Peak resident memory was `224.4 MB`.
The accelerated late admissions are evidence that the frozen residual trigger
is aggressive at long times; they are not a reason to alter this completed
trajectory retrospectively.

## Validate receipt

Offline score artifact:

- `output/local_runs/paper_v_archive_gram_adaptive_packet_cutoff16_t98_user_stopped_20260805_v1/`.

The score contracts the normalized packet ket into the archive moments and
compares them with exact cutoff-16 DOP853 wavefunction propagation. The
ordinary `K=6` packet trajectory is a matched preparation-and-drive baseline;
no failed archive-closure route is rerun or displayed as new evidence.

| score over `0 <= t <= 98` | adaptive | ordinary `K=6` |
|---|---:|---:|
| all-31 scaled RMS | `0.063649` | `0.140791` |
| site-0 occupation RMS | `0.013600` | `0.045221` |
| electronic-energy RMS | `0.023320` | `0.067278` |
| phonon-energy RMS | `0.028350` | `0.061007` |
| electron--phonon-energy RMS | `0.033785` | `0.088403` |
| total-internal-energy RMS | `0.000579` | `0.001188` |

Additional checks:

- correlation-block scaled RMS: `0.056529`;
- minimum and final exact-state fidelity: `0.923677`;
- minimum local-cutoff retained norm: `0.998472`;
- minimum electronic-density eigenvalue: `0.058149`;
- minimum bosonic and joint-Gram eigenvalues: `-1.41e-15` and `-1.72e-15`,
  respectively, consistent with positive semidefiniteness to floating-point
  roundoff;
- maximum correlation-trace residual: exactly zero on the stored samples;
- normalized physical-ket norm error: at most `3.33e-16`.

The multi-coherent route uses no online minimum-norm physicality correction.
Representability follows from propagating a normalized ket; the archive joint
Gram is an observer and admission metric. The shrinking raw coefficient-gauge
norm is not the physical-ket norm.

## Analyze

The adaptive model remains stable and representable almost to the requested
`t=100` horizon. Its contracted observables remain substantially closer to the
exact cutoff-16 reference than the matched ordinary packet baseline. Accuracy
degrades gradually relative to `t=40`: all-31 RMS grows from `0.037626` through
`t=40` to `0.063649` through `t=98`, while minimum fidelity declines from
`0.981390` to `0.923677`.

The result supports the packet ket as a useful representable parent model. It
also shows that the present residual trigger purchases that accuracy with
rapid late capacity growth. A future efficiency study may vary the admission
threshold or objective, but this PDF update reports the frozen rule rather
than retuning it after exact scoring.

## Materialize

Update flow:

1. Read the stopped score artifact only; perform no propagation.
2. Generate `adaptive_t98_observables.pdf` and `.png` in
   `output/plots/paper_v_results_progression_20260804/`.
3. Prepend one page to
   `paper_5/notes/paper_v_results_progression_20260804.tex` using the existing
   page structure: protocol line, observable figure, compact score table, and
   concise interpretation.
4. Rebuild `output/pdf/paper_v_results_progression_20260804.pdf`.
5. Verify compilation, page count, newest-page ordering, referenced files,
   and absence of overflow mechanically. Visual inspection is not requested.
6. Record the completed PDF and source hashes in this workspace or the
   generated plotting manifest.

## Review gate

The updated PDF remains an iterative exploratory result dump. Promotion into
an advisor report or manuscript is a separate user decision. The principal
review question is whether later work should retain the frozen aggressive gate
as a high-accuracy reference or test a less aggressive accuracy--capacity
tradeoff.

## Materialization receipt

The update completed without further propagation.

| artifact | SHA-256 |
|---|---|
| stopped trajectory arrays | `fff0424d74299b00aad96404d9a6a35b47a5f3d0ae5d31b2281938895d43fa7d` |
| stopped score summary | `6ce957889baf67891d05685c848d9c15d6c451a3b18bc0e01b6453071709db67` |
| new observable figure | `4d476dfec566c5f68cfdb19e3a381e2a2e95d5c9e97d86cc6a1a2c55291ee139` |
| progression TeX | `03cdf8a8ad54740d7a38bd2fca5dded26168f9b5a070dba57c975485f1f63432` |
| rebuilt six-page PDF | `6b19d5515f4e28b36bdd2863cab6c3ceb2d3ca0c2bd8f31278ccf59b33bb97d6` |

Mechanical verification found six letter-size pages, the new result on page
one, no overflow or undefined-reference warnings, and a PDF modification time
newer than both its TeX source and new figure.  The full Paper V suite passes:
`228 passed in 78.66s`.

## Acceptance criteria

- The first PDF page shows the `t=98` adaptive and exact observables.
- The page states that `K` means packets per electronic branch and reports
  `K=27` at the stop.
- The page distinguishes normalized-ket representability from the older
  archive-EOM physicality controller.
- Every reported number is read from the stopped score artifact.
- Existing report pages remain in their previous order after the new page.
- No archive-closure trajectory is rerun.

Files to edit:

- `pipelines/open_dynamics/plot_paper_v_results_progression.py`
- `paper_5/notes/paper_v_results_progression_20260804.tex`
- `paper_5/notes/electron_phonon_closure_worklog_20260803.md`
- `agent_guidance/paper-v/icm/adaptive-packet-t98-results-pdf-20260805.md`
