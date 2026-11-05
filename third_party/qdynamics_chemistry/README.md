# Doron Wu's chemistry qubitization, vendored read-only

Source: https://github.com/jacopos86/qdynamics.git, branch `doron-chemistry`, commit
`38f40805d6db3fa3c2cee8a003bd07cf7dd963b3` ("Add chemistry-specific qubitization workflow",
doronwu, 2026-10-27). License: GPL-3.0; `LICENSE` is the source repository's. Every file under
`src/`, `test/` and `dependencies/` is byte-identical to that commit: `MANIFEST.json` records each
file's git blob sha (checked against the upstream tree when the copy was made, 2026-11-04) and
sha256, and `test/test_qdynamics_vendored_crosscheck.py::test_vendored_files_match_manifest`
re-checks the sha256 values. Do not edit these files; anything we need lives in our own package.

What is here: his PySCF driver (`src/chemistry/pyscf/`: RHF, analytic Hessian, the PySCF `eph`
vertex, writing `<prefix>_ele.h5`, `<prefix>_vib.h5`, `<prefix>_eph.h5`), his qubitizer
(`src/chemistry_qubitization/`: Jordan--Wigner in blocked spin order, binary or unary bosons, the
JSON schema `qdynamics_qubit_hamiltonian_v1`) and his tests. Not here: the rest of qdynamics
(Psi4 route, dynamics, io_module). His H2O input deck is not in his repository.

Ours, next to it: `src/quantum/chemistry/qdynamics_interop.py` (load his JSON into our operator
and register layout, export ours in his schema, compare term by term, HF (x) vacuum energy, exact
sector spectrum) and `src/quantum/chemistry/qdynamics_vendored.py` (run this code from Python).

## Running his code

His modules import `src.chemistry...` and `src.chemistry_qubitization`, the same top-level name
as our `src` package, so run them with `PYTHONPATH` set to this directory and with `python -P`
(no working-directory entry on `sys.path`); pyscf and h5py are required (the isolated venv
`~/.venvs/doron-bridge-gpu` has them; the repository `.venv` does not).

```bash
PYTHONPATH=third_party/qdynamics_chemistry python -P third_party/qdynamics_chemistry/src/chemistry/pyscf/main_pyscf.py h2o.inp
```

```bash
PYTHONPATH=third_party/qdynamics_chemistry python -P -m src.chemistry_qubitization <dir with the .h5 files> -o h2o.json --cutoffs 1 --encoding binary
```

Input-deck keys are documented in `src/chemistry/pyscf/input_parser.py`; `write_eph = true` is
needed for the vibronic terms and `isotope_masses = <one value per atom>` for correct masses
(see below).

## What we measured against this commit (2026-11-04)

H2O, STO-3G, at the RHF optimum (our Paper IV geometry, r = 0.9896 A, theta = 100.0 deg),
his driver and qubitizer run here, compared with our builder set to his model
(`builder_model_in_qdynamics_conventions`) and exported in his schema; scripts and files in
`work/doron_interop_20261104/`.

- Electronic terms (words with identity on every boson qubit, excluding the constant): equal to
  4e-15. Same 1207 words at cutoff 1, same 1564 words at cutoff 3; mode order differs (his is
  descending in omega, ours ascending) and two mode signs differ, both of which are relabelings.
- With `isotope_masses = 15.99491461957, 1.00782503223, 1.00782503223`: oscillator terms equal to
  1e-10, coupling terms to 4e-6 absolute (7e-5 relative; residual: his Hessian keeps translations
  and rotations, ours projects them out), HF (x) vacuum energy to 5e-11 Ha, sector ground state at
  cutoff 1 to 3e-7 Ha.
- With his default masses: PySCF's `eph` uses integer mass numbers times MP_ME; his
  `EPH_FREQUENCY_SCALE = sqrt(MP_ME/AMU2AU)` undoes the MP_ME part only, so every omega comes out
  0.363 % too high (bend 2178.65 instead of 2170.79 cm^-1), the couplings move with the modes
  (max difference 3.5e-4 absolute, 0.5 % relative) and E_RHF + sum omega/2 is 8.9e-5 Ha too high.
  The frequencies on his 2026-11-02 slides (2170.78 / 4138.70 / 4389.26 cm^-1) are the correct
  ones; the code at this commit does not reproduce them unless `isotope_masses` is given.
- His slides' "direct classical ground state energy" is <HF x vac| H |HF x vac> = E_RHF +
  sum omega/2 (-74.941528 Ha here); the ground state of the same operator in the (5,5) sector is
  -75.1190 Ha at cutoff 1, -75.2645 at cutoff 3.
