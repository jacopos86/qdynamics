# Chemistry prototype

Scope:
- electronic-only molecular fixtures plus the H2 one-mode vibronic fixture route
- Psi4 front-end for generation paths
- RHF closed-shell integral extraction, with H2 active-space exact/FCI references where feasible
- blocked spin-orbital ordering only
- chemistry-folder-local ADAPT core
- normal tests/static runs use checked JSON fixtures and do not import optional chemistry backends

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

Checked-in LiH/STO-3G static benchmark fixture:
- path: `test_support/molecular_problem_lih_sto3g.json`
- molecule: neutral LiH, RHF closed shell, STO-3G
- geometry: `Li 0.0 0.0 0.0`, `H 0.0 0.0 1.595`
- units: Angstrom
- generated dimensions: 6 spatial orbitals / 12 spin orbitals, `(n_alpha, n_beta) = (2, 2)`
- provenance/rationale: the fixed `1.595 Å` bond length is the repo's LiH/STO-3G benchmark geometry, chosen as the auditable checked-in fixture for canonical molecular ADAPT wiring rather than regenerated during normal tests.

Regenerate only when Psi4 is available locally; normal tests and static runs load
the checked-in JSON and do not import Psi4:
```bash
PYTHONPATH=$(pwd) src/quantum/chemistry/conda-env/bin/python - <<'PY'
import json
from pathlib import Path
from src.quantum.chemistry.psi4_adapter import load_restricted_closed_shell_problem_from_psi4

problem = load_restricted_closed_shell_problem_from_psi4(
    geometry_spec="Li 0.0 0.0 0.0\nH 0.0 0.0 1.595",
    basis="sto-3g",
    charge=0,
    multiplicity=1,
    units="angstrom",
    reference="rhf",
    scf_type="pk",
)
Path("test_support/molecular_problem_lih_sto3g.json").write_text(
    json.dumps(problem.to_jsonable(), indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
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

Checked-in molecular-vibronic H2 static benchmark fixture:
- canonical path: `test_support/molecular_vibronic_h2_sto3g_fd001.json`
- legacy fallback path: `test_support/molecular_vibronic_h2_sto3g_nph1_binary.json`
- family key: `molecular_vibronic_h2`
- molecule/mode: H2/STO-3G restricted closed shell plus one quantized stretch mode
- geometry: `R = 0.7414 Å`, finite-difference step `0.01 Å`
- encoding: 4 fermion qubits + 1 binary boson qubit (`n_ph_max=1`)
- use: Route A static ADAPT mixed-lane anchor alongside Hubbard-Holstein
- normal tests/static runs load the fixture and do not import Psi4

Checked-in molecular-vibronic H2O active-space prototype:
- canonical path: `test_support/molecular_vibronic_h2o_sto3g_active2_fd001.json`
- family key: `molecular_vibronic_h2o`
- molecule/mode: H2O/STO-3G restricted closed shell projected to the frontier two-spatial-orbital active space plus one quantized mode
- encoding: 4 active-space fermion qubits + 1 binary boson qubit (`n_ph_max=1`)
- derivative source: deterministic `frontier_gap_surrogate_dQ_v1`; replace with finite-difference normal-mode `dH/dQ` before using the row as paper evidence
- use: follow-on application-paper plumbing target after validating the existing H2 vibronic route
- normal tests/static runs load the fixture and do not import Psi4

Regenerate only when Psi4 is available locally:
```bash
PYTHONPATH=$(pwd) src/quantum/chemistry/conda-env/bin/python - <<'PY'
import json
from pathlib import Path
from src.quantum.chemistry.vibronic_h2 import (
    build_vibronic_h2_model,
    exact_ground_energy_physical_sector,
    vibronic_h2_fixture_to_jsonable,
)

model = build_vibronic_h2_model(
    bond_length_angstrom=0.7414,
    bond_step_angstrom=0.01,
    basis="sto-3g",
    n_ph_max=1,
    boson_encoding="binary",
    coupling_scale=1.0,
    ordering="blocked",
)
exact = exact_ground_energy_physical_sector(
    model.h_vibronic,
    n_spatial_orbitals=2,
    num_particles=(1, 1),
    n_ph_max=1,
    boson_encoding="binary",
)
Path("test_support/molecular_vibronic_h2_sto3g_nph1_binary.json").write_text(
    json.dumps(
        vibronic_h2_fixture_to_jsonable(
            model,
            exact_ground_energy=exact,
            provenance={
                "generated_by": "src.quantum.chemistry.vibronic_h2.build_vibronic_h2_model",
                "generator_env": "src/quantum/chemistry/conda-env",
                "psi4_required_to_regenerate": True,
            },
        ),
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
```

Vibronic H2 smoke check:
```bash
PYTHONPATH=$(pwd) src/quantum/chemistry/conda-env/bin/python \
  -m src.quantum.chemistry.smoke_h2_vibronic
```

Ab initio H2 curve fixture route:
```bash
PYTHONPATH=$(pwd) src/quantum/chemistry/conda-env/bin/python \
  -m src.quantum.chemistry.generate_h2_abinitio_fixture \
  --basis cc-pVDZ \
  --r-grid 0.5414:1.7414:0.01 \
  --center-r 0.7414 \
  --finite-diff-step 0.01 \
  --n-vibrational-levels 8 \
  --output-curve-json src/quantum/chemistry/h2_abinitio_curve_psi4_ccpvdz.json
```

The ab initio generator writes a full `h2_abinitio_curve_v1` artifact with the
R grid, units, backend/package versions, nuclear masses, separate nuclear
repulsion, array hashes, active-space metadata, mapped Pauli Hamiltonians, and a
1D finite-difference vibrational reference. `--electronic-reference fci` is the
default and asks Psi4 for FCI total energies on the curve; `jw-sector-dense` is
available for small active-space reproduction checks, and `scf` is diagnostic.
The generator imports Psi4 only on the generator path; normal tests and static
runs still load checked JSON and do not import Psi4 or PySCF. Add
`--pyscf-cross-check` to record an optional center-geometry PySCF FCI comparison
when PySCF is installed.

Runtime compatibility is deliberately conservative. The current
`molecular_vibronic_h2_fixture_v1` SNAKE/ADAPT fixture remains a 4-fermion-qubit
local Taylor runtime contract. Direct `--emit-runtime-v1` only works for native
two-spatial-orbital curves and still fails loudly for larger active spaces.
Larger-basis curves such as cc-pVDZ can feed SNAKE only through the explicit
projected runtime-active route:

```bash
PYTHONPATH=$(pwd) src/quantum/chemistry/conda-env/bin/python \
  -m src.quantum.chemistry.generate_h2_abinitio_fixture \
  --basis cc-pVDZ \
  --r-grid 0.5414:1.7414:0.01 \
  --center-r 0.7414 \
  --finite-diff-step 0.01 \
  --emit-downfolded-runtime-v1 \
  --runtime-active-space-policy frontier_occupied_virtual \
  --output-curve-json src/quantum/chemistry/h2_abinitio_curve_psi4_ccpvdz.json \
  --output-runtime-active-curve-json src/quantum/chemistry/h2_abinitio_curve_psi4_ccpvdz_projected_active2.json \
  --output-runtime-fixture-json src/quantum/chemistry/molecular_vibronic_h2_ccpvdz_projected_active2_runtime_v1.generated.json
```

This route slices the parent MO integrals to a two-spatial-orbital H2 active
space, recomputes projected active-space exact energies from the projected
4-qubit Hamiltonian, derives the local Taylor product from that projected curve,
and labels the emitted fixture with
`projected_two_spatial_orbital_active_space_not_full_parent_basis`. It is an
explicit MO-subspace projection (`mo_subspace_projection_no_core_correction_v1`),
not a correlated effective-Hamiltonian downfolding and not the full cc-pVDZ
Hamiltonian. Feed such a generated fixture into the static SNAKE path with
`--molecular-vibronic-h2-fixture-json <fixture.json>`; do not replace the checked
canonical STO-3G fixture for diagnostics.

Coupling convention: `--runtime-coupling-scale 1.0` is the physical
finite-difference coupling for the local Taylor fixture. Other values, including
weak comparison lanes such as `0.25`, are artificial diagnostic rescalings.

Bosons now:
- electronic H2 path remains fermions-only
- vibronic H2 path reuses repo boson encoding from `src/quantum/hubbard_latex_python_pairs.py`
- current vibronic model is one quantized H-H stretch mode with linear coupling from finite-difference `dH/dR`
- current coupling derivative is overlap-aligned to the center MO basis before finite differencing
- this is still a prototype vibronic model, not a full nonadiabatic derivative-coupling treatment
