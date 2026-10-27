# Chemistry qubitization

Read HDF5 matrix elements produced by the chemistry backends, map molecular
electronic and vibrational operators to a Pauli Hamiltonian, and save the
result as JSON. This package requires `numpy` and `h5py`.

## Usage

Run these commands from the worktree root:

```bash
# Electronic and molecular-vibrational data from one calculation directory
python -m src.chemistry_qubitization /path/to/files --cutoffs 2 -o hamiltonian.json

# Electronic data only
python -m src.chemistry_qubitization /path/job_ele.h5 -o electronic.json

# Explicitly select the electronic and electron-phonon files
python -m src.chemistry_qubitization /path/job_ele.h5 /path/job_eph.h5 \
  --cutoffs 2 3 --encoding unary -o hamiltonian.json
```

The direct script path `src/chemistry_qubitization/__main__.py` may be used in
place of `-m src.chemistry_qubitization`.

- `--cutoffs`: maximum vibrational occupation N, retaining levels 0 through N.
  Supply one shared value or one value per mode. This option is required when
  electron-phonon data are present.
- `--encoding`: `binary` (default) or `unary`.
- `--particles ALPHA BETA`: provide particle counts when `nelec` metadata are
  unavailable.
- `--tolerance`: coefficient truncation threshold after assembly; the default
  is `1e-12`.
- If a directory contains multiple calculations, select the files explicitly.
  Existing output files are never overwritten.

## Input and output

The electronic HDF5 file must contain `h1`, `h2`, and `nuclear_repulsion`. An
optional electron-phonon HDF5 file must contain `eph_mat` and `omega`. Files are
identified by their datasets, and frequencies and couplings must come from the
same electron-phonon file. Vibrational-analysis files such as `*_vib.h5` do not
participate directly in Hamiltonian assembly.

The current implementation supports real restricted closed-shell HF data in
the full MO space, interleaved spin-orbital input, non-antisymmetrized chemist
ERIs, and mean-field electron-phonon coupling in the MO basis. Source files
must belong to the same calculation and use the same MO gauge.

The JSON output preserves Pauli words, complex coefficients, source metadata,
particle counts, cutoffs, and encoding. Energies use Hartree units. Electronic
spin ordering is blocked, and Pauli words use `e/x/y/z` in
`q_(n-1)...q_0` order.

## Python API

```python
from src.chemistry_qubitization import (
    load_integrals, build_hamiltonian, save_hamiltonian, load_hamiltonian,
)

integrals = load_integrals(["/path/to/files"])
hamiltonian = build_hamiltonian(integrals, cutoffs=[2])
save_hamiltonian(hamiltonian, "/path/hamiltonian.json")
restored = load_hamiltonian("/path/hamiltonian.json")
operator = restored.operator  # PauliPolynomial owned by chemistry_qubitization
```

The package implements its own Pauli algebra and Jordan--Wigner mapping in
`pauli.py`; it does not depend on the legacy `src.qubitization` package.

Run the tests with:

```bash
python -m pytest -q test/test_chemistry_qubitization.py
```
