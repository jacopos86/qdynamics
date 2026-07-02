# Simplified Hubbard Pipeline Repo

This repo has been simplified to keep only the runtime pipeline path and its direct dependencies.

## Main Entry Points

- `pipelines/hardcoded_hubbard_pipeline.py`
- `pipelines/qiskit_hubbard_baseline_pipeline.py`
- `pipelines/compare_hardcoded_vs_qiskit_pipeline.py`

## Run Guide

Full command usage and parameter options are documented in:

- `pipelines/PIPELINE_RUN_GUIDE.md`

Quick start:

```bash
cd "/Users/jakestrobel/Downloads/qdynamics-main/Fermi-Hamil-JW-VQE-TROTTER-PIPELINE"
/opt/anaconda3/bin/python pipelines/compare_hardcoded_vs_qiskit_pipeline.py --help
```

## Output Location

Generated JSON/PDF outputs go to:

- `artifacts/`

See:

- `artifacts/README.md`

## Archived Legacy Content

Legacy notebooks, old test harnesses, and prior report artifacts were moved (archive-first) to:

- `archive/pre_simplify_*`

They are preserved for rollback/reference but are no longer part of the active runtime path.
