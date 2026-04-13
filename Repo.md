# Repo

This is the human guide for the **current HH workflow** in this repo.

- Scope: HH only.
- Use this file for runs and outputs.
- Use `MATH/Math.md` for derivations and controller math.
- Ignore historical and legacy paths unless you are reproducing an old artifact.
- Doc priority: `AGENTS.md -> run_guide.md -> README.md -> MATH/Math.md`.

## Quick start

1. Run direct HH ADAPT to make the main scaffold artifact.
2. Reuse that artifact for either:
   - realtime McLachlan controller dynamics, or
   - Suzuki-Trotter baseline dynamics.
3. Optionally prune an artifact.
4. Optionally compute FFT / spectra from a controller JSON.

---

## 1. Hamiltonian Builder

### Run

The builder is a library layer, so the practical way to use it is through the direct HH ADAPT CLI:

```bash
python pipelines/static_adapt/adapt_pipeline.py \
  --L 2 --problem hh \
  --t 1.0 --u 4.0 --dv 0.0 \
  --omega0 1.0 --g-ep 0.5 --n-ph-max 1 \
  --boundary open --ordering blocked --boson-encoding binary \
  --adapt-max-depth 30 --adapt-eps-grad 1e-5 --adapt-maxiter 800 \
  --output-json artifacts/json/adapt_hh_L2.json \
  --output-pdf output/pdf/adapt_hh_L2.pdf
```

### What it does

This resolves the HH physics point used by the rest of the repo.

Current builder path:

- `build_problem_hamiltonian(...)`
- `build_hubbard_holstein_hamiltonian(...)`
- `build_hh_sector_hamiltonian_ed(...)`

### Core inputs

| Flag | Meaning |
| --- | --- |
| `--L` | number of sites |
| `--t` | hopping |
| `--u` | onsite interaction |
| `--dv` | static onsite potential |
| `--omega0` | phonon frequency |
| `--g-ep` | electron-phonon coupling |
| `--n-ph-max` | phonon cutoff |
| `--boundary` | `open` or `periodic` |
| `--ordering` | `blocked` or `interleaved` |
| `--boson-encoding` | current direct HH path uses `binary` |

### Output / PDF

- No standalone builder artifact.
- In practice, you use the ADAPT artifact as the builder output.
- PDF path here is the ADAPT PDF, not a separate builder PDF.

---

## 2. Direct HH ADAPT

### Run

```bash
python pipelines/static_adapt/adapt_pipeline.py \
  --L 2 --problem hh \
  --omega0 1.0 --g-ep 0.5 --n-ph-max 1 \
  --adapt-max-depth 30 --adapt-eps-grad 1e-5 --adapt-maxiter 800 \
  --adapt-inner-optimizer SPSA \
  --initial-state-source adapt_vqe \
  --output-json artifacts/json/adapt_hh_L2_phase3_v1.json \
  --output-pdf output/pdf/adapt_hh_L2_phase3_v1.pdf
```

### What it does

This is the **main HH entrypoint**.

Use it to build the static HH problem, run the current ADAPT selection path, and save the scaffold/state artifact used by later stages.

Important current behavior:

- If you omit `--adapt-continuation-mode`, HH resolves to `phase3_v1`.
- If you omit `--adapt-pool`, HH resolves to the current default direct path rather than the old broad legacy path.

### Core inputs

| Flag | Meaning |
| --- | --- |
| `--adapt-max-depth` | ADAPT depth cap |
| `--adapt-eps-grad` | gradient stop threshold |
| `--adapt-maxiter` | inner reoptimization budget |
| `--adapt-inner-optimizer` | `SPSA`, `POWELL`, or `COBYLA` |
| `--initial-state-source` | `exact`, `adapt_vqe`, or `hf` |
| `--adapt-continuation-mode` | omit for current HH `phase3_v1` |
| `--adapt-pool` | override pool only if you mean to |
| `--output-json` | main artifact path |
| `--output-pdf` | built-in PDF path |
| `--skip-pdf` | disable PDF output |

### Output / PDF

Main output: ADAPT JSON.

Useful top-level blocks:

- `settings`
- `ground_state`
- `adapt_vqe`
- `initial_state`

PDF status:

- Yes. Use `--output-pdf`.
- This is the easiest current way to get a PDF from the static HH stage.

---

## 3. Time dynamics

## 3A. McLachlan checkpoint controller

### Run

```bash
python -m pipelines.time_dynamics.hh_realtime_from_adapt_artifact \
  --artifact-json artifacts/json/adapt_hh_L2_phase3_v1.json \
  --output-json artifacts/json/time_dynamics_from_adapt_L2.json \
  --progress-json artifacts/json/time_dynamics_from_adapt_L2.progress.json \
  --partial-payload-json artifacts/json/time_dynamics_from_adapt_L2.partial.json \
  --t-final 8.0 --num-times 161 \
  --drive-A 1.5 --drive-omega 1.2 --drive-tbar 4.0 --drive-pattern staggered \
  --exact-steps-multiplier 4 \
  --checkpoint-controller-mode exact_v1
```

### What it does

This reuses an ADAPT artifact and runs the current realtime checkpoint controller.

This is the path tied to `MATH/Math.md` Section `17A`.

### Core inputs

| Flag | Meaning |
| --- | --- |
| `--artifact-json` | input ADAPT artifact |
| `--output-json` | main controller output |
| `--progress-json` | rolling progress file |
| `--partial-payload-json` | partial recovery/debug file |
| `--t-final` | total time horizon |
| `--num-times` | checkpoint count |
| `--drive-A` | drive amplitude |
| `--drive-omega` | drive frequency |
| `--drive-tbar` | drive envelope width |
| `--drive-pattern` | drive spatial profile |
| `--exact-steps-multiplier` | exact-reference refinement |
| `--checkpoint-controller-mode` | current example uses `exact_v1` |

### Output / PDF

Main output: controller JSON.

Useful top-level blocks:

- `summary`
- `trajectory`
- `ledger`
- `reference`

PDF status:

- No generic built-in PDF writer for this CLI.
- Treat this stage as JSON-first.

## 3B. Suzuki-Trotter baseline

### Run

```bash
python pipelines/hardcoded/hubbard_pipeline.py \
  --L 2 --problem hh \
  --omega0 1.0 --g-ep 0.5 --n-ph-max 1 \
  --initial-state-source adapt_json \
  --adapt-input-json artifacts/json/adapt_hh_L2_phase3_v1.json \
  --propagator suzuki2 \
  --enable-drive --drive-A 1.5 --drive-omega 1.2 --drive-tbar 4.0 --drive-pattern staggered \
  --trotter-steps 64 --t-final 10.0 --num-times 201 \
  --skip-qpe \
  --output-json artifacts/json/hc_hh_L2_suzuki.json \
  --output-pdf output/pdf/hc_hh_L2_suzuki.pdf
```

### What it does

This is the maintained hardcoded baseline path.

Use it when you want a Suzuki trajectory starting from an imported ADAPT state.

### Core inputs

| Flag | Meaning |
| --- | --- |
| `--initial-state-source` | use `adapt_json` to hand off from ADAPT |
| `--adapt-input-json` | ADAPT artifact to import |
| `--propagator` | use `suzuki2` for the maintained approximate path |
| `--enable-drive` | enable drive |
| `--drive-A`, `--drive-omega`, `--drive-tbar`, `--drive-pattern` | drive settings |
| `--trotter-steps` | Suzuki macro-step count |
| `--t-final` | total time horizon |
| `--num-times` | number of reported samples |
| `--skip-qpe` | skip QPE tail stage |
| `--output-json` | main artifact path |
| `--output-pdf` | built-in PDF path |

### Output / PDF

Main output: trajectory JSON.

Useful top-level blocks:

- `settings`
- `ground_state`
- `vqe`
- `qpe`
- `initial_state`
- `trajectory`

PDF status:

- Yes. Use `--output-pdf`.

---

## 4. Post-processing pruning

### Run

```bash
python pipelines/static_adapt/hh_prune_nighthawk.py \
  --input-json <artifact.json> \
  --mode scaffold \
  --prune-threshold 1e-4 \
  --scaffold-method POWELL \
  --scaffold-maxiter 2000 \
  --output-json artifacts/json/hh_prune_summary.json \
  --output-md artifacts/reports/hh_prune_summary.md
```

### What it does

This is the current prune / scaffold-cleanup surface for an existing HH artifact.

Use it after an artifact already exists.

### Core inputs

| Flag | Meaning |
| --- | --- |
| `--input-json` | source artifact |
| `--mode` | `scaffold`, `readapt`, `both`, or `export_fixed_scaffolds` |
| `--prune-threshold` | prune cutoff |
| `--scaffold-method` | fixed-scaffold optimizer |
| `--scaffold-maxiter` | scaffold reopt budget |
| `--readapt-max-depth` | re-ADAPT depth cap |
| `--output-json` | summary JSON |
| `--output-md` | Markdown summary |

### Output / PDF

- Output is JSON plus optional Markdown / exported scaffold JSONs.
- No dedicated PDF path here.

---

## 5. FFT / spectra

### Run

```bash
python pipelines/time_dynamics/hh_time_dynamics_spectra.py \
  --input-json artifacts/json/time_dynamics_from_adapt_L2.json \
  --time-key time \
  --detrend constant \
  --window hann \
  --max-peaks 5 \
  --max-harmonic 3 \
  --output-json artifacts/json/time_dynamics_from_adapt_L2_spectra.json \
  --output-png artifacts/png/time_dynamics_from_adapt_L2_spectra.png
```

### What it does

This computes FFT-style post-processing from a controller trajectory JSON.

### Core inputs

| Flag | Meaning |
| --- | --- |
| `--input-json` | controller JSON |
| `--output-json` | spectra summary JSON |
| `--output-png` | spectra figure |
| `--time-key` | `time`, `physical_time`, or `auto` |
| `--pair` | optional pair-difference channel |
| `--detrend` | `constant` or `linear` |
| `--window` | `hann` or `none` |
| `--max-peaks` | number of reported peaks |
| `--max-harmonic` | harmonic summary depth |
| `--plot-max-omega` | x-axis cap |

### Output / PDF

- Output is JSON plus PNG.
- No dedicated PDF path here.

---

## PDF summary

For report-PDF commands from existing JSON artifacts, see the appendix.

| Stage | PDF available? |
| --- | --- |
| Builder | only through the ADAPT PDF |
| Direct HH ADAPT | yes |
| McLachlan controller | no generic built-in PDF |
| Suzuki-Trotter baseline | yes |
| Pruning | no |
| FFT / spectra | no |

---

## Appendix: Summary artifacts and report PDFs

Use this appendix only when you already have JSON artifacts and want a summary PDF.

### 1. Staged controller dynamics report

**Script:** `pipelines/reporting/hh_staged_controller_dynamics_report.py`

**Input JSON:** staged HH noise workflow JSON artifact

```bash
python pipelines/reporting/hh_staged_controller_dynamics_report.py \
  --input-json <staged_noise_workflow.json> \
  --output-pdf output/pdf/staged_controller_report.pdf
```

**Output:** physics-facing PDF for staged controller time-dynamics artifacts.

### 2. Fixed-manifold dynamics report

**Script:** `pipelines/reporting/hh_fixed_manifold_dynamics_report.py`

**Input JSON:** measured fixed-manifold dynamics JSON artifact

```bash
python pipelines/reporting/hh_fixed_manifold_dynamics_report.py \
  --input-json <fixed_manifold_measured.json> \
  --output-pdf output/pdf/fixed_manifold_report.pdf
```

**Output:** PDF summary for fixed-manifold HH time-dynamics runs.

### 3. Staged noiseless exact report

**Script:** `pipelines/reporting/hh_staged_noiseless_exact_report.py`

**Input JSON:** `hh_staged_noiseless` workflow JSON

```bash
python pipelines/reporting/hh_staged_noiseless_exact_report.py \
  --input-json <hh_staged_noiseless.json> \
  --output-pdf output/pdf/staged_noiseless_report.pdf
```

**Output:** focused PDF comparing staged noiseless dynamics against the exact reference.

### No dedicated report PDF

These paths do not currently have a dedicated generic report-PDF CLI:

- `python -m pipelines.time_dynamics.hh_realtime_from_adapt_artifact` -> JSON-first output
- `python pipelines/static_adapt/hh_prune_nighthawk.py` -> JSON / Markdown / exported JSONs
- `python pipelines/time_dynamics/hh_time_dynamics_spectra.py` -> JSON / PNG

## If you need more detail

- Run guide: `run_guide.md`
- Main repo orientation: `README.md`
- Math and controller derivations: `MATH/Math.md`
