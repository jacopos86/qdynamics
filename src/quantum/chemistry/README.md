# Chemistry prototype

Scope:
- electronic-only
- Psi4 front-end
- RHF closed-shell only
- blocked spin-orbital ordering only
- chemistry-folder-local ADAPT core
- no boson / phonon encoding yet

Local env:
- conda env path: `src/quantum/chemistry/conda-env`
- run commands from repo root with `PYTHONPATH` set to the repo root

H2 prototype:
```bash
PYTHONPATH=$(pwd) src/quantum/chemistry/conda-env/bin/python \
  -m src.quantum.chemistry.prototype_h2 \
  --bond-length 0.7414 \
  --basis sto-3g
```

Generic molecule prototype:
```bash
PYTHONPATH=$(pwd) src/quantum/chemistry/conda-env/bin/python \
  -m src.quantum.chemistry.prototype_molecule \
  --geometry 'H 0 0 0; H 0 0 0.7414' \
  --basis sto-3g
```

Chemistry-local smoke check:
```bash
PYTHONPATH=$(pwd) src/quantum/chemistry/conda-env/bin/python \
  -m src.quantum.chemistry.smoke_h2
```

Current outputs default into `src/quantum/chemistry/` unless `--output-json` overrides them.

Vibronic H2 prototype:
```bash
PYTHONPATH=$(pwd) src/quantum/chemistry/conda-env/bin/python \
  -m src.quantum.chemistry.prototype_h2_vibronic \
  --bond-length 0.7414 \
  --bond-step 0.01 \
  --basis sto-3g \
  --n-ph-max 3 \
  --boson-encoding binary
```

Vibronic H2 smoke check:
```bash
PYTHONPATH=$(pwd) src/quantum/chemistry/conda-env/bin/python \
  -m src.quantum.chemistry.smoke_h2_vibronic
```

Bosons now:
- electronic H2 path remains fermions-only
- vibronic H2 path reuses repo boson encoding from `src/quantum/hubbard_latex_python_pairs.py`
- current vibronic model is one quantized H-H stretch mode with linear coupling from finite-difference `dH/dR`
- current coupling derivative is overlap-aligned to the center MO basis before finite differencing
- this is still a prototype vibronic model, not a full nonadiabatic derivative-coupling treatment
