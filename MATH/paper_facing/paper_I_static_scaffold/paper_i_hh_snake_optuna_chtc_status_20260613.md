# Paper I HH SNAKE Optuna CHTC Status

Created: 2026-06-13T10:07:51-05:00  
Scope: Paper-I Hubbard--Holstein SNAKE/competitor CHTC batches for Table III / Figure II follow-up evidence.  
Use: agent-facing provenance and retrieval tracker. This is not a manuscript table and is not a promotion decision.

## Evidence Gate

Treat all rows here as evidence candidates until the result JSONs, manifests, logs, and plotting/table source maps are retrieved and validated. Promotion into Paper I requires user choice after objective validation of settings, result fields, and provenance.

Status sources used for this snapshot:

- `condor_q jsstrobel -nobatch -af ClusterId ProcId JobBatchName JobStatus HoldReason`
- `condor_history jsstrobel -limit 800 -af ClusterId ProcId JobBatchName JobStatus ExitCode`
- CHTC submit files under `~/Holstein_phase3_optuna_chtc/chtc/phase3_optuna/`
- CHTC record-id files under `chtc/phase3_optuna/input/...`
- Access-point output inspection under `~/Holstein_phase3_optuna_chtc/raw_outputs/`
- Local June-12 retrieval bundle under
  `raw_outputs/chtc_fetches/paper_i_hh_20260612_quota_retrieval/raw_outputs/`
- Live read-only CHTC job inspection for status/depth/trial only; no live energy
  value is used as a Paper-I HH table value unless a same-cutoff result JSON is retrieved.

Operational risk:

- The CHTC home area was over quota in an earlier login, but the 2026-06-13 10:02 login banner showed about `38G / 40G` (`94.99%`). Prefer targeted retrieval and remote cleanup over blind full-output copying.
- The current completed novelty-surface output directories contain multi-GB JSON/JSONL payloads. Fetch summaries/best-trial JSONs first, not every file.

## Intermediate Evidence Package Pass 1

Updated: 2026-06-13.

Purpose: agent-facing candidate/mock Paper-I Hubbard--Holstein SNAKE evidence package before any manuscript or canonical source-map promotion. The PDF includes the current Table-III HH SNAKE regimes and the requested `U/t=8` new-sector rows in the same artifact.

Metric contract:

- Paper-I Hubbard--Holstein rows must render and compare only same-cutoff error:
  `abs(E_alg(n_ph_work) - E_ED(n_ph_work))`.
- Non-same-cutoff reference metrics from raw run logs are not copied into the
  pass-1 package and are not Paper-I HH result evidence.
- Live rows keep same-cutoff error marked pending until a terminal/current result JSON
  with an audited same-cutoff result field is retrieved.

Generated artifacts:

The stable human-facing shortcut `output/pdf/hh_snake_pass1.pdf` now points to
the rebuilt Paper-I-style SNAKE snapshot below. The earlier pass-1 output names
were overwritten with the same cleaned same-cutoff-only snapshot to avoid stale
artifact confusion.

| Artifact | Path | SHA256 |
|---|---|---|
| pass-1 source JSON | `MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_snake_intermediate_evidence_pass1_sources_20260613.json` | `fb57f3676d34e09613cbff5033127cf88f59de8089ba26d4d48186150c6c6000` |
| human TeX snapshot builder | `pipelines/reporting/build_paper_i_hh_snake_results_snapshot_tex.py` | `302b3fddaafbd82c31e01ec7a1316e14f206ff2e1310f015209ba8a9281d7918` |
| legacy ReportLab pass-1 builder | `pipelines/reporting/build_paper_i_hh_intermediate_evidence_pdf.py` | `0ecacae5bcac54d0a749e81a49da3fe6156fc8f51318d69d8232e843db98d3ff` |
| current human TeX snapshot | `output/pdf/paper_i_hh_snake_results_snapshot_20260613.tex` | `11fe09d7866a27302a5da081a368266e445bbd7a1b0ae51c966d6eaf6dca217b` |
| current human PDF snapshot | `output/pdf/paper_i_hh_snake_results_snapshot_20260613.pdf` | `d92e1d29585ab8e88a5359661760fdb4a980b18e59b4d2a1ee0654563264672f` |
| stable short Finder copy | `output/pdf/hh_snake_pass1.pdf` | `d92e1d29585ab8e88a5359661760fdb4a980b18e59b4d2a1ee0654563264672f` |
| legacy-name TeX snapshot | `output/pdf/paper_i_hh_snake_intermediate_evidence_pass1_20260613.tex` | `11fe09d7866a27302a5da081a368266e445bbd7a1b0ae51c966d6eaf6dca217b` |
| legacy-name PDF snapshot | `output/pdf/paper_i_hh_snake_intermediate_evidence_pass1_20260613.pdf` | `d92e1d29585ab8e88a5359661760fdb4a980b18e59b4d2a1ee0654563264672f` |
| report sidecar JSON | `output/pdf/paper_i_hh_snake_intermediate_evidence_pass1_20260613.json` | `b65b61757ebd2332edd58e92c383fa8180ba72b6337da2829fa4555743f468a9` |
| report CSV | `output/pdf/paper_i_hh_snake_intermediate_evidence_pass1_20260613.csv` | `62cc91e6e51984e2a65653845d13ec651c99d4f61446a1ceca4b435eaf633c7d` |

Pass-1 selected-row status:

| Sector | Regime | Selected status | Same-cutoff error | Depth | Trial | Interpretation |
|---|---|---|---:|---:|---:|---|
| current Table III | weak-weak `(0.25,0.25)` | completed JSON | `3.7671790687321405e-04` | 26 | 4 | Lower than current visible same-cutoff SNAKE value `4.229364909101053e-04`. |
| current Table III | strong-weak `(1.25,0.25)` | completed JSON | `2.0797526934657196e-04` | 42 | 19 | Higher than current visible same-cutoff SNAKE value `1.7773193839498713e-04`; retain as candidate evidence, not replacement. |
| current Table III | weak-strong `(0.25,1.25)` | live status only | pending | 9 | 15 | Same-cutoff unavailable from current live telemetry. |
| current Table III | strong-strong `(1.25,1.25)` | live status only | pending | 12 | 10 | Same-cutoff unavailable from current live telemetry. |
| `U/t=8` new sector | strong-weak `(8,0.25)` | completed JSON | `5.3588643219804055e-05` | 8 | 7 | Separate new-sector result; not the current Table-III `U/t=1.25` strong-Hubbard contract. |
| `U/t=8` new sector | strong-strong `(8,1.25)` | live best-trial JSON snapshot | `1.5430884619094254e-04` | 26 | 9 | Retrieved without stopping the still-running job. |

Human/Machine split:

- The pass-2 human PDF is a TeX-backed Paper-I-style SNAKE snapshot. It renders same-cutoff error plots, first-plateau table cells where available, and a compact appendix of terminal/current SNAKE rows.
- The pass-2 human PDF renders only same-cutoff energy error values.
- The first pages are one block per regime, comparing only current Paper-I SNAKE against current CHTC SNAKE variants.
- Competitor rows remain only in machine-readable sidecar data for later pass-3 use; they are not plotted or rendered in the pass-2 human PDF.
- The appendix renders terminal/current SNAKE rows so current-running checkpoints are not confused with first-plateau table rows.
- Agent-facing provenance, hashes, source paths, and interpretation notes stay in this MD plus the source/sidecar JSON and CSV.
- The TeX source contains comments pointing to this MD and the machine-readable source JSON/sidecars.
- Visual render check passed after rendering `output/pdf/paper_i_hh_pass2_costs_plots_20260613.pdf` to PNG under `output/pdf/render_check_hh_pass2_20260613/`.

## Pass-2 Cost/Plot Report

Updated: 2026-06-13.

Human-facing artifacts:

| Artifact | Path | SHA256 |
|---|---|---|
| PDF | `output/pdf/paper_i_hh_pass2_costs_plots_20260613.pdf` | `131b3cf143d88a93af821e9a5afaebf789f7a2d2044f2e6a8737c806f6c696cc` |
| TeX | `output/pdf/paper_i_hh_pass2_costs_plots_20260613.tex` | `023b95662aae01b1a199f50974704a0b0a4ecf8cc9572b5f10b5a793eabd919e` |
| Sidecar JSON | `output/pdf/paper_i_hh_pass2_costs_plots_20260613.json` | `3c75ad69a2bef7914e528b58ac7df74781a9f449e22290cf04d90d5a83d940dc` |
| Live snapshot tarball | `raw_outputs/chtc_fetches/live_snapshots/paper_i_hh_snake_live_current_20260613T170623Z.tgz` | `146e2deb85b2711c646f9cd333bdb0b62277b8073c2c73be9f4cbc2870881a57` |
| U8 strong-strong live progress tarball | `raw_outputs/chtc_fetches/live_snapshots/paper_i_u8_strong_strong_live_20260613T133909/paper_i_u8_strong_strong_live_20260613T133909.tgz` | `21151407d2ec653cbb8d8ca6a85751b15a0b4b89543b35006d55c602c34c5721` |
| U8 strong-strong best-trial JSON tarball | `raw_outputs/chtc_fetches/live_snapshots/paper_i_u8_strong_strong_live_20260613T133909/paper_i_u8_strong_strong_best_trial_json_20260613T134140.tgz` | `47167b14c035001fbc5136aac328fba232a30a3c3cf45d7867e430eb97aa8f43` |

Rendered regimes:

| Regime | Plot included | Main table included | Notes |
|---|---|---|---|
| weak-weak `(U/t, lambda)=(0.25,0.25)`, `nph=2` | yes | current Paper-I SNAKE, SNAKE exponent, SNAKE flat | New SNAKE exponent/flat rows improve same-cutoff error relative to current Paper-I SNAKE at the displayed first-plateau prefix. |
| strong-weak/intermediate-weak `(1.25,0.25)`, `nph=2` | yes | current Paper-I SNAKE, SNAKE exponent, SNAKE flat | This is the Table-III intermediate-Hubbard weak-Holstein sector. |
| weak-strong `(0.25,1.25)`, `nph=4` | yes | current Paper-I SNAKE, live SNAKE exponent, live SNAKE flat | Live SNAKE current checkpoints are plotted as same-cutoff error from current JSON/history-tail snapshots; their plateau costs remain blank until a valid first plateau is identified. |
| strong-strong/intermediate-strong `(1.25,1.25)`, `nph=4` | yes | current Paper-I SNAKE, live SNAKE exponent, live SNAKE flat, live SNAKE novelty-surface | This is the Table-III intermediate-Hubbard sector, not the `U/t=8` true-strong sector. |
| U8 strong-weak `(8,0.25)`, `nph=2` | yes | SNAKE novelty-surface | Separate true-strong-Hubbard sector requested by the user. |
| U8 strong-strong `(8,1.25)`, `nph=4` | yes | SNAKE novelty-surface | Retrieved from a live CHTC best-trial JSON without stopping the still-running job. |

Cost-cell policy for this pass:

- Existing Paper-I rows use the current visible Paper-I plateau tuples.
- CHTC SNAKE rows use reconstructed historical ansatz-prefix costs under the Table-I Qiskit convention when a first plateau can be identified from the retrieved history.
- Current-running SNAKE rows with no accepted first plateau keep main-table plateau costs blank. Their terminal/current reconstructed costs appear only in the appendix terminal/current table.
- Human PDF renders no alternate-cutoff comparison fields. Same-cutoff means `abs(E_alg(nph) - E_ED(nph))` for the row's displayed phonon cutoff.

Live U8 strong-strong retrieval:

- CHTC job `7652257.5` was `JobStatus=2` before and after retrieval. The live snapshot used `condor_ssh_to_job` for read-only file access and did not evict or stop the running job.
- Full best-trial result JSON: `raw_outputs/chtc_fetches/live_snapshots/paper_i_u8_strong_strong_live_20260613T133909/raw_outputs/paper_i_hh_snake_novelty_surface_optuna_20260611_v2_u8_strong_strong/run/hh_L2_nph4_three_model_sym_u8_strong_strong/trial_0009/hh_L2_nph4_three_model_sym_u8_strong_strong/json/result.json`
- Result JSON SHA256: `424cc8b43973594ae9ec8da9ab143190217b337367c33131d3e7a9bd1d680be1`
- Retrieved best trial: `trial=9`, `adapt_depth=26`, `abs_delta_e_same_cutoff=1.5430884619094254e-04`, `energy=0.520730586556899`, `same_cutoff_exact_gs_energy=0.520576277710708`, `stop_reason=benchmark_abs_delta_e_target`.
- Reconstructed Table-I-style prefix costs at `k=26`: `N2q=2464`, `D2q=2282`, `Dc=12391`.

## U8 Strong-Hubbard Retrieval Update

Updated: 2026-06-14.

Purpose: retrieve the completed CHTC evidence for the `U/t=8` Hubbard--Holstein
strong-Hubbard sector and refresh the local replacement-candidate PDF without
changing `MATH/paper_details/static_adapt_paper_I.tex`.

Metric contract:

- All Hubbard--Holstein values below are same-cutoff errors:
  `abs(E_alg(n_ph_work)-E_ED(n_ph_work))`.
- `U/t=8, lambda=0.25` uses `n_ph_work=2`.
- `U/t=8, lambda=1.25` uses `n_ph_work=4`.
- Higher-cutoff reference diagnostics are not used in these tables or plots.

Retrieved local bundle:

| Artifact | Path | SHA256/status |
|---|---|---|
| CHTC retrieval bundle | `raw_outputs/chtc_fetches/u8_retrieval_20260614T1203Z/paper_i_hh_u8_retrieval_bundle_20260614T1207Z.tgz` | `a998a79a0c8384b6419b94adf47ecafedc1c2217753da99c4d065f0f0050956f` |
| Extracted retrieval root | `raw_outputs/chtc_fetches/u8_retrieval_20260614T1203Z/extracted/` | contains U8 strong-strong SNAKE best-trial JSON plus partial comparator output |
| Source-locked replay archive | `raw_outputs/chtc_fetches/u8_retrieval_20260614T1203Z/extracted/paper_i_hh_u8_snake_source_locked_replay_20260613_v1/` | input/preflight only; no result JSON evidence |
| Flat/no-cost replay archive | `raw_outputs/chtc_fetches/u8_retrieval_20260614T1203Z/extracted/paper_i_hh_u8_snake_flatnovelty_nocost_20260613_v1/` | input/preflight only; no result JSON evidence |

Important retrieval note:

- The comparator CHTC submit used `transfer_output_files = raw_outputs, logs`
  for every proc. Several completed procs reported successful output transfer
  in Condor logs, but their `raw_outputs/.../records/<record_id>` directories no
  longer exist on the access point. The available evidence is therefore the
  locally retrieved June-12 bundle plus the June-14 partial bundle. Still-running
  procs are left alone.

U8 strong-weak `(U/t,lambda)=(8,0.25)`, `n_ph_work=2`, same-cutoff best rows:

| Method | Status | Trial | k | same-cutoff error | N2q | D2q | Dc | S/source note |
|---|---|---:|---:|---:|---:|---:|---:|---|
| HEA VQE | completed | 14 | -- | `3.097221814463924e-02` | 14 | 9 | 39 | `S=1572962304` |
| family VQE | completed | 1 | 12 | `4.7354129920115706e-01` | 1782 | 1614 | 7003 | `S=524353536` |
| Append-ADAPT | completed | 13 | 5 | `4.31507390002972e-05` | 144 | 116 | 723 | `S=979304448` |
| TETRIS-ADAPT | completed | 16 | 13 | `1.1954807104586074e-05` | 348 | 197 | 1372 | `S=955777024` |
| Geo-ADAPT | completed | 10 | 9 | `1.2254429947899936e-05` | 296 | 264 | 1548 | `S=5369626624` |
| Qubit/QEB | running on CHTC | -- | -- | -- | -- | -- | -- | proc `7651653.9` |
| SNAKE | completed | 7 | 8 | `5.3588643219804055e-05` | 455 | 284 | 1079 | Qiskit compile-scout FakeMarrakesh cost; `S` missing |

U8 strong-strong `(U/t,lambda)=(8,1.25)`, `n_ph_work=4`, same-cutoff best rows:

| Method | Status | Trial | k | same-cutoff error | N2q | D2q | Dc | S/source note |
|---|---|---:|---:|---:|---:|---:|---:|---|
| HEA VQE | completed | 14 | -- | `2.2255541673980805e-02` | 18 | 11 | 41 | `S=3440855040` |
| family VQE | completed | 8 | 12 | `4.794237222901738e-01` | 11350 | 10423 | 35457 | `S=3441070080` |
| Append-ADAPT | running on CHTC | -- | -- | -- | -- | -- | -- | proc `7651653.2` |
| TETRIS-ADAPT | running on CHTC | -- | -- | -- | -- | -- | -- | proc `7651653.10` |
| Geo-ADAPT | completed | 15 | 15 | `1.1327391134974274e-04` | 840 | 711 | 3940 | `S=12952862720` |
| Qubit/QEB | running on CHTC | -- | -- | -- | -- | -- | -- | proc `7651653.8` |
| SNAKE | completed | 9 | 26 | `1.5430884619094254e-04` | 3199 | 3119 | 11313 | Qiskit compile-scout FakeMarrakesh cost; `S` missing |

SNAKE source hashes:

| Regime | Result JSON | Result SHA256 | Compile-scout JSON | Compile-scout SHA256 |
|---|---|---|---|---|
| U8 strong-weak | `raw_outputs/chtc_fetches/paper_i_hh_20260612_quota_retrieval/raw_outputs/paper_i_hh_snake_novelty_surface_optuna_20260611_v2_u8_strong_weak/run/hh_L2_nph2_three_model_sym_u8_strong_weak/trial_0007/hh_L2_nph2_three_model_sym_u8_strong_weak/json/result.json` | `13e5e1699704f2b8e12b87f04ed0e66852b1410728c8b2b58b79841cf0fcada1` | `raw_outputs/chtc_fetches/paper_i_hh_20260612_quota_retrieval/raw_outputs/paper_i_hh_snake_novelty_surface_optuna_20260611_v2_u8_strong_weak/run/hh_L2_nph2_three_model_sym_u8_strong_weak/trial_0007/hh_L2_nph2_three_model_sym_u8_strong_weak/json/compile_scout_fake_marrakesh.json` | `3d1142538936e477f1eca9de10ca111c7b863fcb0c5bdd13f6f6cdb80ee9734f` |
| U8 strong-strong | `raw_outputs/chtc_fetches/u8_retrieval_20260614T1203Z/extracted/paper_i_u8_strong_strong_best_trial_json_20260613T134140/raw_outputs/paper_i_hh_snake_novelty_surface_optuna_20260611_v2_u8_strong_strong/run/hh_L2_nph4_three_model_sym_u8_strong_strong/trial_0009/hh_L2_nph4_three_model_sym_u8_strong_strong/json/result.json` | `424cc8b43973594ae9ec8da9ab143190217b337367c33131d3e7a9bd1d680be1` | `raw_outputs/chtc_fetches/u8_retrieval_20260614T1203Z/extracted/paper_i_u8_strong_strong_best_trial_json_20260613T134140/raw_outputs/paper_i_hh_snake_novelty_surface_optuna_20260611_v2_u8_strong_strong/run/hh_L2_nph4_three_model_sym_u8_strong_strong/trial_0009/hh_L2_nph4_three_model_sym_u8_strong_strong/json/compile_scout_fake_marrakesh.json` | `89c0169d2f5437fb704bb6dfa79e2725e896fa5499a9576abcc289460e9c21ec` |

Regenerated human-facing report:

| Artifact | Path | SHA256 |
|---|---|---|
| PDF | `output/pdf/paper_i_true_strong_replacement_20260613.pdf` | `fd0e9218f98443f96c6d05a1f097bd13729cb91fcf2775469e0667c00673d596` |
| TeX | `output/pdf/paper_i_true_strong_replacement_20260613.tex` | `678fa34c284f553a0a3b5b4c2584fd848bd10c84464739781e036b23eb66f996` |
| Sidecar JSON | `output/pdf/paper_i_true_strong_replacement_20260613.json` | `72050c03ef1ada9a86b151acb7a66ac1fa269935be272ba3f3328af85a281b07` |
| U8 strong-weak plot PDF | `output/pdf/paper_i_true_strong_replacement_20260613_hh_u8_strong_weak.pdf` | `9ae908dbabce6f25463664bdfeecae8f86a8f34d674aa0d1884c31ea71d34a5a` |
| U8 strong-strong plot PDF | `output/pdf/paper_i_true_strong_replacement_20260613_hh_u8_strong_strong.pdf` | `130c3117d9833ccc35d5e1a345f2d577933900f34daef16e817f81757c732955` |

Render check:

- Rendered with `pdftoppm` to
  `tmp/pdfs/paper_i_true_strong_replacement_20260614_render/`.
- Pages 1--3 visually checked. The plot lines are solid, markers appear only on
  the displayed row, and tables fit within the page.

Paper-I manuscript update:

| Artifact | Path | SHA256/status |
|---|---|---|
| Manuscript TeX | `MATH/paper_details/static_adapt_paper_I.tex` | `eb86c233008c91607bbab5b1b0cbf45578aee56a514cfde2b5c9a61f5fee279b` |
| Manuscript PDF | `MATH/paper_details/static_adapt_paper_I.pdf` | `cc7e5afcccd9f5e406570ea4ed6db62553c14653fc742c993928530f716e1596` |
| Render check | `tmp/pdfs/static_adapt_paper_I_20260614_hh_u8_snake_render/` | pages 8--11 rendered; pages 9--10 visually checked |

Visible manuscript changes:

- Main Hubbard strong block now uses \(U/t=8\), with the corresponding U/t=8
  Hubbard error-vs-iteration plot and full competitor table.
- The main Hubbard--Holstein table remains the existing Paper-I
  weak/intermediate-Hubbard table. It was not replaced by U/t=8
  Hubbard--Holstein rows because the full U/t=8 HH comparator suite is still
  incomplete.
- A visible SNAKE-only \(U/t=8\) Hubbard--Holstein table was added as
  `tab:hh_u8_snake_current_best`: strong--weak \((8,0.25)\),
  `n_ph=2`, `k=8`, same-cutoff `5.36e-5`, `N2q=455`, `D2q=284`,
  `Dc=1079`; strong--strong \((8,1.25)\), `n_ph=4`, `k=26`,
  same-cutoff `1.54e-4`, `N2q=3199`, `D2q=3119`, `Dc=11313`.
- Hidden TeX provenance comment
  `BEGIN_MACHINE_READABLE_TABLE_I_HUBBARD_U8_STRONG_UPDATE_20260614` points to
  the standalone source report and records the incomplete HH replacement gate.
- Hidden TeX provenance comment
  `BEGIN_MACHINE_READABLE_HH_U8_SNAKE_CURRENT_BEST_20260614` points to the
  local SNAKE result JSONs and compile-scout sidecars for the visible SNAKE-only
  \(U/t=8\) table.

No edits were made to:

- `MATH/paper_facing/paper_I_static_scaffold/hh_tableiii_convergence_sources.json`

Additional failed SNAKE correction fetch:

| Artifact | Path | SHA256/status |
|---|---|---|
| U8 SNAKE Pareto incumbent fetch | `raw_outputs/chtc_fetches/u8_snake_pareto_incumbent_20260614T1734Z/paper_i_u8_hh_snake_pareto_incumbent_fetch_20260614T173430Z.tgz` | `f297571c8e52e042a7f68c7e4e63764f81d2f11cf8fd652e028eb6c06e87a99b` |
| Extracted root | `raw_outputs/chtc_fetches/u8_snake_pareto_incumbent_20260614T1734Z/extracted/` | all eight fetched arms failed before writing result JSON |

Failure reason:

- Every fetched U/t=8 SNAKE Pareto arm failed route-identity validation with
  missing score-formula fields and an amplitude-history prune mismatch:
  `missing:phase2_raw_score_formula`,
  `missing:canonical_score_formula`,
  `missing:primary_selector_score_key`,
  `missing:auxiliary_terms_primary_mode`, and
  `mismatch:phase1_prune_amplitude_witness_required:'false'!=True`.
- These failed arms do not replace the earlier novelty-surface SNAKE rows in
  the U/t=8 HH snapshot below.

## Current Best SNAKE Snapshot

Updated: 2026-06-13, live CHTC plus local June-12 retrieval.

Main Table-III HH regimes. `deltaE_same` is same-cutoff error when a terminal result JSON exists. For still-running rows, the same-cutoff value is pending.

| Regime | Best current SNAKE source | Status | Best error seen | Depth | Trial | Notes |
|---|---|---|---:|---:|---:|---|
| weak-weak `(0.25,0.25)` | exponent and flat/no-exponent, cluster `7646619` procs `0,1`; local retrieved best-trial JSONs | completed result JSON local | `deltaE_same=3.7671790687321405e-04` | 26 | 4 | Exponent and flat have identical terminal same-cutoff metrics but distinct source hashes. Stop reason `drop_plateau`; target-hit flag false under strict target-stop semantics. |
| strong-weak `(1.25,0.25)` | exponent and flat/no-exponent, cluster `7646619` procs `2,3`; local retrieved best-trial JSONs | completed result JSON local | `deltaE_same=2.0797526934657196e-04` | 42 | 19 | Exponent and flat have identical terminal same-cutoff metrics but distinct source hashes. |
| weak-strong `(0.25,1.25)` | exponent/flat, cluster `7646619` procs `4,5` | running; same-cutoff pending | pending | 9 | 15 | Same-cutoff value requires a terminal/current result JSON. Novelty-surface proc `7652257.2` timed out. |
| strong-strong `(1.25,1.25)` | exponent/flat, cluster `7646619` procs `6,7` | running; same-cutoff pending | pending | 12 | 10 | Same-cutoff value requires a terminal/current result JSON. |

Best-trial JSONs already local for completed Table-III weak-weak/strong-weak:

| Label | Source JSON | SHA256 |
|---|---|---|
| `weak_weak_exponent` | `raw_outputs/chtc_fetches/paper_i_hh_20260612_quota_retrieval/raw_outputs/paper_i_hh_tableiii_snake_novelty_optuna_20260611_v1_weak_weak_exponent/run/hh_L2_nph2_three_model_sym_weak_weak/trial_0004/hh_L2_nph2_three_model_sym_weak_weak/json/result.json` | `c210ca239c80e97bd66ee315b9d57b2749685590a217f6645485653e6b9dd273` |
| `weak_weak_flat` | `raw_outputs/chtc_fetches/paper_i_hh_20260612_quota_retrieval/raw_outputs/paper_i_hh_tableiii_snake_novelty_optuna_20260611_v1_weak_weak_flat/run/hh_L2_nph2_three_model_sym_weak_weak/trial_0004/hh_L2_nph2_three_model_sym_weak_weak/json/result.json` | `57462c6ea7c30b6f726244a69c7accadad26d7c91fd88334e5c166463060b2d6` |
| `strong_weak_exponent` | `raw_outputs/chtc_fetches/paper_i_hh_20260612_quota_retrieval/raw_outputs/paper_i_hh_tableiii_snake_novelty_optuna_20260611_v1_strong_weak_exponent/run/hh_L2_nph2_three_model_sym_strong_weak/trial_0019/hh_L2_nph2_three_model_sym_strong_weak/json/result.json` | `990039c86b34b0ec206d795ea925db1f1033ffabd16eec1a4a3f9b5e3bc747b4` |
| `strong_weak_flat` | `raw_outputs/chtc_fetches/paper_i_hh_20260612_quota_retrieval/raw_outputs/paper_i_hh_tableiii_snake_novelty_optuna_20260611_v1_strong_weak_flat/run/hh_L2_nph2_three_model_sym_strong_weak/trial_0019/hh_L2_nph2_three_model_sym_strong_weak/json/result.json` | `f4c507b56c607b102569fc82ee75f9e0d7184d5b705e1b7a37e4dae8f6797f5f` |

Extra `U/t=8` SNAKE sector:

| Sector | Source | Status | Best error seen | Depth | Trial | Notes |
|---|---|---|---:|---:|---:|---|
| U8 strong-weak | novelty-surface, cluster `7652257` proc `4`; local retrieved best-trial JSON | completed result JSON local | `deltaE_same=5.3588643219804055e-05` | 8 | 7 | Source JSON SHA256 `13e5e1699704f2b8e12b87f04ed0e66852b1410728c8b2b58b79841cf0fcada1`. |
| U8 strong-strong | novelty-surface, cluster `7652257` proc `5`; live best-trial JSON snapshot | running on CHTC; current result JSON retrieved locally | `deltaE_same=1.5430884619094254e-04` | 26 | 9 | Job remained running after read-only snapshot. Source JSON SHA256 `424cc8b43973594ae9ec8da9ab143190217b337367c33131d3e7a9bd1d680be1`. |

## Weak-Strong Depth-42 Reprobe Provenance Update

Updated: 2026-06-14.

Purpose: record the completed replayable JSON bundle for the weak-strong Table-III
SNAKE continuation without silently replacing the current rendered plot/table
source.

Result status:

| Field | Value |
|---|---|
| Batch | `paper-i-hh-ws-snake-depth42-reprobe-20260613-v1` |
| Cluster.proc | `7696570.0` |
| Local root | `raw_outputs/routeA_paper_i_hh_weak_strong_snake_depth42_reprobe_20260613_v1/weak_strong` |
| Regime | weak-strong `(U/t, lambda)=(0.25,1.25)` |
| Phonon cutoff | `n_ph_work=4`; `n_ph_ref=7` recorded only as a diagnostic/reference field |
| Same-cutoff metric | `abs(E_alg(n_ph_work)-E_ED(n_ph_work))` |
| Reprobe same-cutoff error | `1.897402690174932e-02` |
| Energy | `-1.1196051734575971` |
| Ansatz depth | `42` |
| History rows in recovered JSON | `24` resumed-admission rows |
| Stop reason | `max_depth` |

Comparison to current Paper-I plot/table provenance:

| Source | Same-cutoff error | Interpretation |
|---|---:|---|
| Current Table-III weak-strong SNAKE plateau value | `1.924606706563992e-02` | Reprobe is lower by `2.720401638906002e-04`. |
| Current rendered error-vs-iteration plot terminal point | `1.8593762719976814e-02` at plotted x `47` | Reprobe is higher by `3.80264181772505e-04` and is not an exact plot replacement. |
| Recovered depth-42 JSON bundle | `1.897402690174932e-02` | Valid companion/recovery provenance for the depth-42 continuation target. |

Updated provenance surfaces:

| Artifact | Path | SHA256/status |
|---|---|---|
| Source map | `MATH/paper_facing/paper_I_static_scaffold/hh_tableiii_convergence_sources.json` | Added `replayable_depth42_reprobe_20260614`; source-map SHA256 after update `646851c785941f0b81c88b739e9b5b9b56ab7989d0839cfefb14740c57f02042`. |
| Plot provenance sidecar | `output/pdf/paper_i_hh_table3_adapt_descent_all_regimes_paper_sources_combined_ww20_sw20_ws50_ss60_logy_notarget_strongy168.provenance.json` | Added replayable companion source; plotted values unchanged. |
| Summary JSON | `raw_outputs/routeA_paper_i_hh_weak_strong_snake_depth42_reprobe_20260613_v1/weak_strong/summary.json` | SHA256 `a9e082be4d3584aa39ae89e97f85ebeb61f1300f952414e33d49c1bf332da242`. |
| Manifest JSON | `raw_outputs/routeA_paper_i_hh_weak_strong_snake_depth42_reprobe_20260613_v1/weak_strong/manifest.json` | SHA256 `988bd7dae9930fb85d67ddb2dc8c81aaec483c4b0d389a36ec95f9e0ea2a5dd7`. |
| Result JSON | `raw_outputs/routeA_paper_i_hh_weak_strong_snake_depth42_reprobe_20260613_v1/weak_strong/json/result.json` | SHA256 `6e657ca07c67e7df55e06a50e785fffdf810d3aae8acc36e5805b61b91e99ea3`. |
| Current JSON | `raw_outputs/routeA_paper_i_hh_weak_strong_snake_depth42_reprobe_20260613_v1/weak_strong/json/current.json` | SHA256 `dd13e09c3434a4ee79aaa0f3374a15940375988403e836e049b380db64cca606`. |
| Stdout log | `raw_outputs/routeA_paper_i_hh_weak_strong_snake_depth42_reprobe_20260613_v1/weak_strong/logs/stdout.log` | SHA256 `b26e2d380ba237196321dc1a64db6d782c6da14e3c9987b439ba9aa1bbc4d9c4`. |

Use note:

- This bundle matches the weak-strong HH Table-III SNAKE continuation target and
  supplies replayable depth-42 provenance.
- It should not be used as a silent replacement for the current rendered
  47-point stdout-derived convergence curve or visible table value.
- A future plot/table update may use it only after an explicit decision about
  whether to prefer the replayable depth-42 JSON value or regenerate from a
  different approved replacement source.

## Human PDF Snapshot, 2026-06-13

Current report files:

- PDF: `output/pdf/paper_i_hh_pass2_costs_plots_20260613.pdf`
- TeX: `output/pdf/paper_i_hh_pass2_costs_plots_20260613.tex`
- Sidecar JSON: `output/pdf/paper_i_hh_pass2_costs_plots_20260613.json`

Report semantics:

- Main regime tables are plateau tables. For each row, `same-cutoff |Delta E|`, `compiled at k`, `N2q`, `D2q`, and `Dc` refer to the same displayed plateau prefix.
- CHTC SNAKE plateau costs are filled only when the historical ordered ansatz prefix can be reconstructed from run history and Qiskit-compiled under the Table-I convention.
- Terminal or still-running current costs are not mixed into plateau rows. They are rendered only in the appendix terminal/current table and preserved with full provenance in the sidecar JSON.
- Same-cutoff means `abs(E_alg(n_ph_work) - E_ED(n_ph_work))`. Higher-cutoff ED diagnostics are intentionally excluded from this report.
- The rendered PDF contains no source-path provenance; source paths and hashes stay in this MD file and the sidecar JSON.

## Active Batch Summary

| Batch | Cluster | Fresh CHTC state | Access-point evidence state | Notes |
|---|---:|---|---|---|
| Table-III HH SNAKE exponent-vs-flat Optuna | `7646619` | procs `0-3` completed exit `0`; procs `4-7` running | Completed procs are in Condor spool on CHTC and already exist in local June-12 retrieval for weak-weak/strong-weak. Live scratch telemetry was parsed for procs `4-7`. | Main batch for 4 HH regimes x 2 SNAKE variants. Need final transfer for running weak-strong/strong-strong rows. |
| HH novelty-surface Optuna, Table-III and U/t=8 | `7652257` | procs `0,1,4` completed exit `0`; proc `2` exit `124`; procs `3,5` running | v2 AP dirs found for Table-III weak-weak and strong-weak. U8 strong-weak best-trial JSON is local from the June-12 retrieval. U8 strong-strong best-trial JSON was retrieved from the live sandbox. | Current-best reporter has null error fields for completed Table-III rows, so parse summaries/trial payloads rather than trusting `current_best.json`. |
| U/t=8 HH comparator SPSA full suite | `7651653` | procs `0,1,3,4,5,6,7,11` completed exit `0`; procs `2,8,9,10` running | `raw_outputs/paper_i_hh_u8_comparator_spsa_v1` exists, about `318M`, with about `374` JSON/JSONL files. | Need map record IDs to methods/regimes before result interpretation. |
| U/t=8 HH SNAKE source-locked replay | `7696585` | submitted 2026-06-13; procs `0-3` idle with no hold at latest queue check | No result output yet. Local preflight passed for all four records before upload. Remote login-node preflight failed because the access-point Python lacks `numpy`; this was an environment issue on the submit host, not an apptainer-job validation failure. | Corrective batch for the bad novelty-surface warm-start interpretation. Keeps prior jobs alive. Uses Paper-I source SNAKE settings, Route A, `full_meta`, same-cutoff-only metrics, and explicit zero motif/path/cost-prior bonuses. |
| U/t=8 HH SNAKE flat-novelty no-cost replay | `7696589` | submitted 2026-06-13; procs `0-5` idle with no hold at latest queue check | Six-record batch generated, local preflight passed, uploaded to CHTC, extracted under `~/Holstein_phase3_optuna_chtc`, and submitted. | Adds the missing flat-novelty arm. Covers source/exponent novelty, flat novelty, and no novelty for U8 strong-weak and U8 strong-strong. Hard-zeroes legacy and current cost fields, forces batching on, forces commutation prefilter off, and sets amplitude-history witness off. |

Older held jobs still visible:

| Cluster | Batch | State | Interpretation |
|---:|---|---|---|
| `7466802` | `paper-i-hh-snake-structural-continue-recovery-20260604-v2` | held | Older recovery batch; at least one proc exceeded `65536 MB`; one was manually held. |
| `7582403` | `paper-i-hh-strong-weak-snake-continue-currentbest-20260608-v1` | held | Held after plateau-stop retrieval around iteration `129`. |
| `7593091` | `paper-i-hh-tableiii-competitor-history-extend-20260609-v1` | held | Transfer-output failure due missing scratch `logs` directory. Prior result JSONs were reported as transferred; treat held state as transfer-contract noise until validated. |

## Regime Matrix

Legend:

- `completed/local`: Condor history says exit `0`, and selected outputs exist in the local June-12 retrieval bundle.
- `spooled`: Condor history says exit `0`, and `TransferOutputStats` shows output was spooled by HTCondor, but it was not pulled into the AP working directory during this status pass.
- `completed/AP parse needed`: transferred output exists, but best-error extraction is not clean from `current_best.json`.
- `running`: still in Condor queue.
- `timeout`: Condor history exit `124`.

| SNAKE batch/method | weak-weak `(U/t, lambda)=(0.25,0.25)` | strong-weak `(1.25,0.25)` | weak-strong `(0.25,1.25)` | strong-strong `(1.25,1.25)` |
|---|---|---|---|---|
| exponent novelty, proc map from `7646619` | proc `0`: completed/local | proc `2`: completed/local | proc `4`: running | proc `6`: running |
| flat/no-exponent novelty, proc map from `7646619` | proc `1`: completed/local | proc `3`: completed/local | proc `5`: running | proc `7`: running |
| novelty-surface Optuna, proc map from `7652257` | proc `0`: completed/AP parse needed | proc `1`: completed/AP parse needed | proc `2`: timeout/no AP dir found | proc `3`: running |

Completed novelty-surface details currently known:

| Regime | AP summary path | Reporter state | Best-error caveat |
|---|---|---|---|
| Table-III weak-weak | `raw_outputs/paper_i_hh_snake_novelty_surface_optuna_20260611_v2_tableiii_weak_weak/run/hh_L2_nph2_three_model_sym_weak_weak/summary.json` | `current_best.json` reports `best_value=100.0` with null error fields. | Summary parser found a terminal same-cutoff error best around `1.1271233585e-03` for trial `0`, while active-objective best points at a failed/null trial. Needs trial-level parse before use. |
| Table-III strong-weak | `raw_outputs/paper_i_hh_snake_novelty_surface_optuna_20260611_v2_tableiii_strong_weak/run/hh_L2_nph2_three_model_sym_strong_weak/summary.json` | `current_best.json` reports `best_value=100.0` with null error fields. | Prior parsed/fetched artifact suggested trial-level best around `2.760871e-04`, but the fresh reporter fields are null. Re-parse trial JSON before citing. |

## U/t=8 Sector

The extra strong-Hubbard sector requested earlier was the Hubbard--Holstein sector with `U/t = 8`. This changes the strong-weak and strong-strong HH points relative to the Table-III convention.

| U/t=8 component | Status | Evidence caveat |
|---|---|---|
| SNAKE novelty-surface strong-weak, cluster `7652257` proc `4` | completed result JSON local | Same-cutoff error `5.3588643219804055e-05` at depth `8`; source JSON SHA256 `13e5e1699704f2b8e12b87f04ed0e66852b1410728c8b2b58b79841cf0fcada1`. |
| SNAKE novelty-surface strong-strong, cluster `7652257` proc `5` | running; live best-trial JSON retrieved | Same-cutoff error `1.5430884619094254e-04` at depth `26`; job remained running after read-only snapshot. |
| Comparator SPSA full suite, cluster `7651653` | partially completed, partially running | AP output exists, but method/regime mapping and best-result parsing are still pending. |

## Corrective U/t=8 Source-Locked Replay Batch

Batch id: `paper_i_hh_u8_snake_source_locked_replay_20260613_v1`  
Cluster id: `7696585`  
Purpose: replace the flawed inference that the earlier U/t=8 novelty-surface Optuna run was a valid warm start from the current Paper-I SNAKE settings.

Diagnosis recorded for future agents:

- The earlier novelty-surface Optuna path sampled `novelty_bonus`; in the current policy materializer that field also set `phase2_motif_bonus_weight`, so the U/t=8 SNAKE rows had a motif bonus active.
- Paper-I SNAKE source settings do not use that motif bonus. Trial-zero/source replay must therefore force motif/path priors off rather than sample them.
- The corrective batch is not a generic Optuna search. It is a source-locked replay of the actual current Paper-I SNAKE settings into the `U/t=8` HH strong-weak and strong-strong sectors, with a no-novelty companion for each sector.

Submitted records:

| Record id | Sector | Novelty arm | Phonon cutoff | Same-cutoff metric |
|---|---|---|---:|---|
| `paper_i_hh_u8_snake_source_locked_replay_20260613_v1_u8_strong_weak_source_novelty` | `(U/t, lambda)=(8,0.25)` | Paper-I exponent schedule | `n_ph=2` | `abs(E_alg(n_ph=2)-E_ED(n_ph=2))` |
| `paper_i_hh_u8_snake_source_locked_replay_20260613_v1_u8_strong_weak_no_novelty` | `(U/t, lambda)=(8,0.25)` | flat/no novelty | `n_ph=2` | `abs(E_alg(n_ph=2)-E_ED(n_ph=2))` |
| `paper_i_hh_u8_snake_source_locked_replay_20260613_v1_u8_strong_strong_source_novelty` | `(U/t, lambda)=(8,1.25)` | Paper-I exponent schedule | `n_ph=4` | `abs(E_alg(n_ph=4)-E_ED(n_ph=4))` |
| `paper_i_hh_u8_snake_source_locked_replay_20260613_v1_u8_strong_strong_no_novelty` | `(U/t, lambda)=(8,1.25)` | flat/no novelty | `n_ph=4` | `abs(E_alg(n_ph=4)-E_ED(n_ph=4))` |

Policy locks:

- `static_route_id=route_a`
- `static_meta_feature_profile=paper_i_production_v1`
- `pool_key=full_meta`
- `family_repeat_penalty=0.0`
- `novelty_bonus=0.0`
- `phase2_motif_bonus_weight=0.0`
- `compile_position_shift_weight=0.0`
- `lambda_1q=lambda_2q=lambda_d=lambda_shot=lambda_theta=0.0`
- hard maximum depth `64`
- same-cutoff primary metric only; no higher-phonon-cutoff ED comparison is part of this corrective evidence lane.

Local provenance:

- Generator: `chtc/phase3_optuna/generate_paper_i_hh_u8_snake_source_locked_replay_records.py`
- Submit file: `chtc/phase3_optuna/submit_paper_i_hh_u8_snake_source_locked_replay_20260613_v1.sub`
- Input manifest: `chtc/phase3_optuna/input/paper_i_hh_u8_snake_source_locked_replay_20260613_v1/paper_i_hh_u8_snake_source_locked_replay_manifest.json`
- Local preflight: `output/pdf/paper_i_hh_u8_snake_source_locked_replay_preflight_20260613.json`, status `pass` for all four records.

## Prepared U/t=8 Flat-Novelty No-Cost Replay Batch

Batch id: `paper_i_hh_u8_snake_flatnovelty_nocost_20260613_v1`  
Cluster id: `7696589`  
CHTC status: submitted 2026-06-13; six jobs idle with no holds at latest queue check.  
Local bundle: `/tmp/paper_i_hh_u8_snake_flatnovelty_nocost_20260613_v1.tgz`

Purpose: add the missing flat-novelty-enabled arm under the corrected U/t=8 source-lock constraints.

Submitted-on-auth records:

| Record id | Sector | Novelty arm | Phonon cutoff | Same-cutoff metric |
|---|---|---|---:|---|
| `paper_i_hh_u8_snake_flatnovelty_nocost_20260613_v1_u8_strong_weak_source_novelty` | `(U/t, lambda)=(8,0.25)` | Paper-I exponent schedule | `n_ph=2` | `abs(E_alg(n_ph=2)-E_ED(n_ph=2))` |
| `paper_i_hh_u8_snake_flatnovelty_nocost_20260613_v1_u8_strong_weak_flat_novelty` | `(U/t, lambda)=(8,0.25)` | flat novelty | `n_ph=2` | `abs(E_alg(n_ph=2)-E_ED(n_ph=2))` |
| `paper_i_hh_u8_snake_flatnovelty_nocost_20260613_v1_u8_strong_weak_no_novelty` | `(U/t, lambda)=(8,0.25)` | no novelty | `n_ph=2` | `abs(E_alg(n_ph=2)-E_ED(n_ph=2))` |
| `paper_i_hh_u8_snake_flatnovelty_nocost_20260613_v1_u8_strong_strong_source_novelty` | `(U/t, lambda)=(8,1.25)` | Paper-I exponent schedule | `n_ph=4` | `abs(E_alg(n_ph=4)-E_ED(n_ph=4))` |
| `paper_i_hh_u8_snake_flatnovelty_nocost_20260613_v1_u8_strong_strong_flat_novelty` | `(U/t, lambda)=(8,1.25)` | flat novelty | `n_ph=4` | `abs(E_alg(n_ph=4)-E_ED(n_ph=4))` |
| `paper_i_hh_u8_snake_flatnovelty_nocost_20260613_v1_u8_strong_strong_no_novelty` | `(U/t, lambda)=(8,1.25)` | no novelty | `n_ph=4` | `abs(E_alg(n_ph=4)-E_ED(n_ph=4))` |

Policy locks:

- source/exponent novelty: `phase2_gamma_N=1.0`, `phase2_gamma_N_schedule_mode=depth_linear_v1`, start `3.0`, end `0.35`, `phase3_novelty_ablation_mode=off`;
- flat novelty: `phase2_gamma_N=1.0`, `phase2_gamma_N_schedule_mode=fixed`, `phase3_novelty_ablation_mode=off`;
- no novelty: `phase2_gamma_N=0.0`, `phase2_gamma_N_schedule_mode=fixed`, `phase3_novelty_ablation_mode=all`;
- `phase1_prune_amplitude_witness_required=false`;
- `phase2_enable_batching=true`;
- `phase3_batch_selection_mode=reduced_plane`;
- `phase3_batch_prefilter_mode=off`;
- all legacy and current cost fields are zero: `lambda_compile`, `lambda_measure`, `lambda_leak`, `lambda_1q`, `lambda_2q`, `lambda_d`, `lambda_shot`, `lambda_theta`, compile weights, measurement weights, optimizer-dimension cost, and Phase-II cost weights.

Local provenance:

- Generator: `chtc/phase3_optuna/generate_paper_i_hh_u8_snake_flatnovelty_nocost_records.py`
- Submit file: `chtc/phase3_optuna/submit_paper_i_hh_u8_snake_flatnovelty_nocost_20260613_v1.sub`
- Input manifest: `chtc/phase3_optuna/input/paper_i_hh_u8_snake_flatnovelty_nocost_20260613_v1/paper_i_hh_u8_snake_flatnovelty_nocost_manifest.json`
- Local preflight: `output/pdf/paper_i_hh_u8_snake_flatnovelty_nocost_preflight_20260613.json`, status `pass` for all six records.
- CHTC submit: `condor_submit chtc/phase3_optuna/submit_paper_i_hh_u8_snake_flatnovelty_nocost_20260613_v1.sub`, cluster `7696589`.

## Retrieval Plan

Use a staged retrieval plan:

1. Fetch a small provenance bundle first: Condor queue/history snapshots, submit files, record-id files, submit logs, and small `summary.json`, `current_best.json`, `trial_events.jsonl`, `records.tsv/jsonl` files where present.
2. Parse completed AP summaries/trial payloads locally to identify the actual best trial JSONs. Do not trust null `current_best.json` fields as evidence.
3. Fetch only the best-trial result JSONs and their manifests/logs for each completed method/regime.
4. For still-running procs, prefer normal transfer on exit. If the user needs interim status, use `condor_ssh_to_job` to snapshot only `progress/current.json`, `progress/current_best.json`, and `trial_events.jsonl`.
5. After all rows terminate and disk pressure is controlled, optionally fetch the full JSON archive for reproducibility.
6. For cluster `7696585`, retrieve source-locked replay result JSONs before generating any replacement human PDF. Compare same-cutoff error and Qiskit compiled costs at the Paper-I plateau prefix and separately at terminal/current depth.
7. For cluster `7696589`, retrieve the six flat-novelty/no-cost replay result JSONs after transfer. Compare same-cutoff error and Qiskit compiled costs at the Paper-I plateau prefix and separately at terminal/current depth.

## Immediate Next Actions

- Parse `raw_outputs/paper_i_hh_u8_comparator_spsa_v1` record IDs and method/regime labels.
- Re-parse novelty-surface completed summaries to recover trial-level best `deltaE` and result JSON paths.
- Monitor cluster `7696585` and retrieve the four source-locked replay result JSONs as soon as they transfer.
- Monitor cluster `7696589` and retrieve the six flat-novelty no-cost replay result JSONs as soon as they transfer.
- Wait for normal final transfer of still-running CHTC jobs, then replace live snapshots with terminal transferred JSONs where applicable.
- Keep running jobs alive. Do not remove held older jobs until their result JSON retrieval is confirmed and the user approves queue cleanup.

## Weak-Strong SNAKE Depth-6 Pareto Manuscript Update

Updated: 2026-06-14.

This supersedes the earlier depth-42 displayed weak--strong SNAKE row for the
current Table III/table-and-plot update. The main visible row now reports the
cheap CHTC `7646619.5` trial0015 flat-novelty depth-6 Pareto plateau, while the
visible weak--strong iteration plot keeps the same trial trajectory through
depth 13 and crops the panel at iteration 20.

Current displayed row:

| Regime | Method | k_pl | Same-cutoff DeltaE | N2q | D2q | Dc | S_alg |
|---|---|---:|---:|---:|---:|---:|---:|
| weak-strong `(U/t, lambda)=(0.25,1.25)` | SNAKE | 6 | `3.9582082240634975e-02` | 320 | 179 | 1148 | `--` |

Primary source/provenance:

- Clean history source: `MATH/paper_facing/paper_I_static_scaffold/history_sources/hh_tableiii_weak_strong_snake_trial0015_flat_depth13_clean_20260614.json`
- Clean history source SHA256: `aef232a3cea093fe5280b9266903d4ec4c37a92a9b3d9ba0521bf7df3e22d656`
- Raw CHTC live snapshot: `raw_outputs/chtc_fetches/live_snapshots/raw_outputs/live_snapshots/paper_i_hh_snake_live_current_20260613T170623Z/7646619_5_trial0015_actual_current.json`
- Raw CHTC live snapshot SHA256: `1d90e29a8058593199d5c99eeee74628289fc714bff1bc9203e12e25d28e702e`
- Depth-6/depth-13 Qiskit cost sidecar: `output/pdf/paper_i_table_iii_snake_weak_strong_trial0015_depth6_depth13_qiskit_cost_20260614.json`
- Depth-6/depth-13 Qiskit cost sidecar SHA256: `c5297a8f4f65331cb3e54f8a9ea65f686402403d302d07c9736428df282cdc56`
- Updated source map: `MATH/paper_facing/paper_I_static_scaffold/hh_tableiii_convergence_sources.json`
- Updated weak-strong single-panel provenance: `output/pdf/paper_i_hh_convergence_weak_strong_single_20260610.provenance.json`

## Superseded Weak-Strong SNAKE Depth-42 Manuscript Update

Updated: 2026-06-14.

The main Paper-I Hubbard--Holstein weak--strong SNAKE row now uses a replayable
combined depth-42 source instead of the previous stdout-only depth-47 tail.
The displayed metric remains same-cutoff error:
`abs(E_alg(n_ph_work=4)-E_ED(n_ph_work=4))`.

Current displayed row:

| Regime | Method | k_pl | Same-cutoff DeltaE | N2q | D2q | Dc | S_alg |
|---|---|---:|---:|---:|---:|---:|---:|
| weak-strong `(U/t, lambda)=(0.25,1.25)` | SNAKE | 42 | `1.897402690174932e-02` | 4548 | 3903 | 21841 | `2.99e5` |

Primary source/provenance:

- Combined replayable source: `MATH/paper_facing/paper_I_static_scaffold/history_sources/hh_tableiii_weak_strong_snake_depth42_replayable_combined_20260614.json`
- Combined source SHA256: `bec3c5b14f6d563e559dd30e24932f4f8924de0c8749747b17236ea0f46971ca`
- Depth-42 result JSON: `raw_outputs/routeA_paper_i_hh_weak_strong_snake_depth42_reprobe_20260613_v1/weak_strong/json/result.json`
- Depth-42 result SHA256: `6e657ca07c67e7df55e06a50e785fffdf810d3aae8acc36e5805b61b91e99ea3`
- Depth-42 Qiskit cost sidecar: `output/pdf/paper_i_table_iii_snake_weak_strong_depth42_qiskit_cost_20260614.json`
- Depth-42 Qiskit cost SHA256: `d580ab6bac25ae9a7cdd99ebff9199ad4be18cc2621cb56cba824d0365e2ef58`
- Updated source map: `MATH/paper_facing/paper_I_static_scaffold/hh_tableiii_convergence_sources.json`
- Updated weak-strong single-panel provenance: `output/pdf/paper_i_hh_convergence_weak_strong_single_20260610.provenance.json`

Continuation status:

- The energy is worse than the older plotted depth-47 point because the
  replayable recovery stops at cumulative depth 42.
- The older depth-47 point was stdout-derived only and is retained as provenance,
  not as the current manuscript-facing source.
- Continue from the depth-42 result JSON above to recover or exceed the old
  depth-47 tail.

Prepared continuation:

- Batch name: `paper-i-hh-ws-snake-depth60-continue-20260614-v1`
- Submit file: `chtc/phase3_optuna/submit_routeA_paper_i_hh_weak_strong_snake_depth60_continue_20260614_v1.sub`
- Runner: `chtc/phase3_optuna/input/routeA_paper_i_hh_weak_strong_snake_depth60_continue_20260614_v1/run_weak_strong_depth60_continue.py`
- Staged resume source: `chtc/phase3_optuna/input/routeA_paper_i_hh_weak_strong_snake_depth60_continue_20260614_v1/sources/weak_strong_depth42_result.json`
- Staged resume source SHA256: `6e657ca07c67e7df55e06a50e785fffdf810d3aae8acc36e5805b61b91e99ea3`
- Dry-run manifest: `tmp/paper_i_hh_ws_depth60_continue_dryrun_20260614/manifest.json`
- Dry-run status: passed command construction.
- Target: cumulative depth `60`, `18` new admissions from the replayable depth-42 source.
- Settings policy: reuse preserved 2026-05-31 weak-strong command/settings; change only `adapt_resume_scaffold_json`, segment id, segment target depth, max new admissions, and output root.
- CHTC staging/submission status: submitted 2026-06-14 after retry. Remote `condor_submit -dry-run` passed, then one job was submitted as cluster `7704849`.
- Initial queue status: `7704849.0`, batch `paper-i-hh-ws-snake-depth60-continue-20260614-v1`, Condor `JobStatus=1` (idle), `HoldReason=undefined`.

## U/t=8 Strong-Strong SNAKE Live-Best Refresh and Geo Continuation

Updated: 2026-06-14.

This block is the current handoff point for the Paper-I U/t=8
Hubbard--Holstein strong-Hubbard sector. The displayed manuscript metric is
same-cutoff error only:
`abs(E_alg(n_ph_work)-E_ED(n_ph_work))`.

### Manuscript/PDF state

The active Paper-I source now contains a SNAKE-only U/t=8 table
`tab:hh_u8_snake_current_best` immediately after the main Hubbard--Holstein
plateau discussion. The strong--weak row remains unchanged. The strong--strong
row has been refreshed from the live SNAKE current-best snapshot.

| Regime | Method | n_ph_work | k | Same-cutoff DeltaE | N2q | D2q | Dc | S_alg | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| strong--weak `(U/t, lambda)=(8,0.25)` | SNAKE | 2 | 8 | `5.3588643219804055e-05` | 455 | 284 | 1079 | `--` | completed source row |
| strong--strong `(U/t, lambda)=(8,1.25)` | SNAKE | 4 | 24 | `5.5802037121632786e-05` | `--` | `--` | `--` | `--` | live current-json snapshot; final cost sidecar missing |

Current manuscript/render hashes:

- TeX: `MATH/paper_details/static_adapt_paper_I.tex`
  SHA256 `a427aa3b1081e27a9d5ed4058c3a6b0d3256a8899f3522993c3db06dc2d91954`
- PDF: `MATH/paper_details/static_adapt_paper_I.pdf`
  SHA256 `33009a2dc1f2c34d9995fed0f745b73331c2c3eb9961d7223b12dcc29a5d7937`
- Build status: `tectonic --keep-logs --reruns 2 static_adapt_paper_I.tex`
  passed from `MATH/paper_details/`; pages 9--10 were rendered and visually
  checked after the update.

Current replacement-report hashes:

- Report PDF: `output/pdf/paper_i_true_strong_replacement_20260613.pdf`
  SHA256 `e3886ae36666788c4db16e1d7e444c3ad9a1bc435d5c0da1c9ce70faa41de562`
- U/t=8 strong--strong plot:
  `output/pdf/paper_i_true_strong_replacement_20260613_hh_u8_strong_strong.pdf`
  SHA256 `93fa22207421f2535fd4a04bf35f40d7d0bf6a2e0a3ec9ca4ee367cc340f6f4c`
- Builder:
  `pipelines/reporting/build_paper_i_true_strong_replacement_pdf.py`
  SHA256 `4368e3e3f35c0f62fbaa13122b061c55bcc8fe67d6dc717f738f5a761512d222`

### SNAKE strong--strong live snapshot provenance

Source:
`raw_outputs/chtc_retrievals/paper_i_u8_hh_strong_strong_snake_current_best/paper_i_u8_hh_ss_v2_7702629_2_20260614T180758Z/trial_0001_current.json`

SHA256:
`602dc1acdccd4ca4463fa7130b762c755f1d62f388949864247ccb5493a16d02`

Observed fields:

- method: `hardcoded_adapt_vqe_phase3_v1_hh`
- history count / displayed k: `24`
- ansatz depth: `21`
- energy: `0.5206320797478297`
- same-cutoff ED reference: `0.520576277710708`
- same-cutoff DeltaE: `5.5802037121632786e-05`
- checkpoint reason: `beam_round_done`
- source kind: live current-json snapshot, not terminal transferred result

Companion snapshot files:

- `trial_events.jsonl`
  SHA256 `1cbbd76a1152dc52b310d954d3604bb6c390af66d0f79c298d2e7fef89be532e`
- `snapshot_summary.json`
  SHA256 `92fb3c06aa53005c45f8b79844f8b786b21cb0258d15ca9ca4c2d9fa5b1cb677`

Superseded visible strong--strong SNAKE row:

| k | Same-cutoff DeltaE | N2q | D2q | Dc | Source SHA256 |
|---:|---:|---:|---:|---:|---|
| 26 | `1.5430884619094254e-04` | 3199 | 3119 | 11313 | `424cc8b43973594ae9ec8da9ab143190217b337367c33131d3e7a9bd1d680be1` |

The new live row improves the same-cutoff error but does not yet carry Qiskit
compiled costs. Keep costs blank in the manuscript until a terminal result or
compiled-cost sidecar is available.

### Geo-ADAPT no-target continuation

The completed U/t=8 strong--strong Geo-ADAPT comparator did not crash or fail;
it stopped because it reached the benchmark same-cutoff target at iteration 15.
The continuation below intentionally disables target stopping and extends the
same Geo-ADAPT trajectory to cumulative depth 60.

Completed source row used for the continuation:

- method: `static_geo_adapt_vqe`
- regime: `(U/t, lambda, n_ph_work)=(8,1.25,4)`
- CHTC proc: `7651653.4`, completed exit `0`
- best trial: `trial_0015`
- source:
  `raw_outputs/chtc_fetches/paper_i_hh_u8_strong_strong_completed_20260613/raw_outputs/paper_i_hh_u8_comparator_spsa_v1/records/paper_i_hh_u8_comp_spsa__full__static_geo_adapt_vqe__hh_u8_strong_strong/trial_0015/cases/hh_L2_nph4_three_model_sym_u8_strong_strong/generic_static_single.json`
- source SHA256:
  `438f159bf008457d2f02582a42f1546ac6e6adafa98120ee7709cc93ee0bfd4f`
- same-cutoff DeltaE: `0.00011327391134974274`
- adapt depth / iterations: `15`
- stop reason: `benchmark_abs_delta_e_target`
- compiled costs: `N2q=840`, `D2q=711`, `Dc=3940`
- shot proxy: `12952862720`

Continuation artifacts:

- runner:
  `chtc/phase3_optuna/input/paper_i_hh_u8_strong_strong_geo_no_target_continue_20260614_v1/run_u8_strong_strong_geo_no_target_continue.py`
  SHA256 `05842896e95c69728db5597bf0358f70cc14d86fd99024a4345d3af804c6345b`
- apptainer wrapper:
  `chtc/phase3_optuna/run_paper_i_hh_u8_strong_strong_geo_no_target_continue_task_apptainer.sh`
  SHA256 `d3d2a6af7535c27bd1b4578d6b2e8b32c6e31aefadac186c4c5564b730046066`
- submit file:
  `chtc/phase3_optuna/submit_paper_i_hh_u8_strong_strong_geo_no_target_continue_20260614_v1.sub`
  SHA256 `3b986ad4759465485a3379719146b14d4fc1cf9020e62839e8fb30d7d55b9467`
- local upload bundle:
  `/tmp/paper_i_hh_u8_ss_geo_continue_20260614_v1.tgz`
  SHA256 `919f420e0b507680bd00e51c7f7aafdf61316a06a63d1e4bba58e919dbd4ecb0`

Local preflight:

```text
python3 chtc/phase3_optuna/input/paper_i_hh_u8_strong_strong_geo_no_target_continue_20260614_v1/run_u8_strong_strong_geo_no_target_continue.py geo --output-root tmp/paper_i_hh_u8_strong_strong_geo_no_target_continue_preflight --max-depth 15 --max-segments 0
```

Preflight result: passed baseline validation with same-cutoff DeltaE
`0.00011327391134974274` at depth 15 and status
`validated_no_segments_requested`.

CHTC submission:

- batch name:
  `paper-i-hh-u8-strong-strong-geo-no-target-continue-20260614-v1`
- cluster/proc: `7706274.0`
- submit status at launch: idle, `JobStatus=1`, `HoldReason=undefined`
- requested cumulative depth: `60`
- output root:
  `raw_outputs/paper_i_hh_u8_strong_strong_geo_no_target_continue_20260614_v1/geo`

Next retrieval target:

Fetch `7706274.0` after transfer or snapshot it if live progress is needed.
Compare the continued Geo curve against the current SNAKE-only U/t=8
strong--strong curve using same-cutoff error at `n_ph_work=4`.
