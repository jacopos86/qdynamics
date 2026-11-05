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

## Molecule-agnostic linear vibronic builder

`src/quantum/chemistry/molecular_vibronic.py` generalizes the H2O linear-FD
generator to any closed-shell molecule, with PySCF as the electronic-structure
backend (imported lazily; the module and its PySCF-free tests load without it):

```bash
PYTHONPATH=$(pwd) python -m src.quantum.chemistry.molecular_vibronic \
  --geometry "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587" \
  --basis sto-3g --active-space frozen_core --cutoffs 3 7 3 \
  --coupling full_linear --exact 2 --output h2o_vibronic.json
```

Model: `H = H_el(R0) + sum_nu omega_nu (b†b + 1/2) + sum_nu Q_nu C_nu` with
`Q_nu = (b + b†)/sqrt(2 omega_nu)`. Steps: RHF, analytic Hessian, mass-weighted
normal modes with translations and rotations projected out, displaced SCFs at
`R0 ± Q e_nu/sqrt(M)` aligned to the R0 MO gauge by orthogonal Procrustes
inside each of the frozen-core, active and external blocks (`--alignment
block_diagonal`, the default, keeps the frozen core canonical; `full` rotates
the whole set as the Paper IV H2O generator did and adds a spurious curvature
of the size of the force constants to the frozen-reference curvature),
frozen-core/CAS slicing (`full`, `frozen_core`, `frozen_core:<n>`,
`cas(<ne>,<no>)`), central differences, Jordan-Wigner in blocked spin order,
binary or unary bosons, and a vectorized exact solver in the physical sector.
Coupling models: `full_linear` (derivative of the scalar, one-body and two-body
active tensors; the Paper IV model), `mean_field_fd` (Fock-matrix derivative)
and `mean_field_pyscf_eph` (PySCF's vertex with the AO basis fixed). The
mean-field models get the reference-force counter-term `C -> C - <HF|C|HF>` by
default; without it their residual force displaces every oscillator.

Oscillator frequencies (`--harmonic-curvature`): `pes_consistent` (the default,
decided 2026-11-02) sets `omega_nu^2 = K_nu^PES + chi_nu`, the curvature of the
frozen-core FCI surface along the mode plus the model's own response to the
assembled coupling, so the exactly solved model reproduces the diagonal of the
FCI Hessian in the normal-mode basis; it costs one active-space FCI per
displaced geometry and per coupling step. `rhf_hessian` is the Paper IV H2O
definition (the RHF Hessian already holds the orbital relaxation the coupling
regenerates) and `frozen_reference` is the Hartree-Fock analogue of the
default. The model file records all three frequency sets. Mode basis
(`--mode-basis`): `model_consistent` (the default, decided 2026-11-02) also
measures the off-diagonal FCI surface curvature (four extra displaced SCFs per
mode pair) and takes the modes as the eigenvectors of `K_PES + chi`, after
which the model's adiabatic Hessian is the full FCI Hessian; the modes come in
ascending frequency and can reorder against the RHF modes (H2O/STO-3G: bend
and symmetric stretch mix by 2.6 degrees and the B2 stretch becomes mode 1),
so per-mode cutoffs follow that order. `rhf_normal_modes` keeps the RHF normal
modes, which reproduces only the diagonal of the FCI Hessian; the model's
cross response `chi_{mu nu}` between modes of one irreducible representation
stays uncompensated there (recorded as a matrix in the provenance).

Paper IV production fixture from the builder:
`src/quantum/chemistry/paper_iv_h2o_fixture_from_builder.py` turns the
builder's decided model into the lane's `h2o_linear_fd_backend_record_v1`
(model modes labelled bend / symmetric_stretch / antisymmetric_stretch by
internal-coordinate character and kept in that order, their frequencies the
builder's, displaced SCFs along the model modes aligned block by block) and
hands it to `generate_h2o_linear_fd_fixture.build_h2o_linear_fd_fixture_from_record`,
so the fixture is produced, validated (production tier, drift tolerance 1e-5 as
the 2026-08-26 fixture) and attested exactly as before:

```bash
PYTHONPATH=$(pwd) python -m src.quantum.chemistry.paper_iv_h2o_fixture_from_builder \
  --output-dir test_support/paper_iv_h2o --tag h2o_builder_model_consistent_work333_ref444_20261103 \
  --mode-cutoffs 3,3,3 --reference-mode-cutoffs 4,4,4
```

The summary it writes next to the fixture cross-checks the fixture's exact
sector ground energy against the builder's own solution of the same model
(agreement to 1e-8 Ha) and records the fixture's byte SHA-256, the identity
the lane's checkpoints attest.

Interoperability with Doron Wu's qubitizer (`github.com/jacopos86/qdynamics`,
branch `doron-chemistry`, `src/chemistry_qubitization`):
`src/quantum/chemistry/qdynamics_interop.py` loads his
`qdynamics_qubit_hamiltonian_v1` JSON into our `PauliPolynomial` and register
layout (same blocked spin order, same `q_(n-1)...q_0` word order, same boson
registers; his coupling `g_I (b + b+)` is our `C_I Q_I` with `C_I = sqrt(2
omega_I) g_I`, the `mean_field_pyscf_eph` model without the counter-term),
writes our models in his schema, evaluates the Hartree-Fock (x) vacuum energy
and the exact physical-sector spectrum of either, and compares two
Hamiltonians term by term. `test/test_qdynamics_interop.py` reproduces the
reference matrices of his own test suite with our assembly to 1e-12 (binary and
unary, cutoffs 1 and 2) without running his code.

His code itself is vendored read-only under `third_party/qdynamics_chemistry`
(GPL-3.0, byte-identical to `jacopos86/qdynamics@38f40805`, manifest with git
blob and sha256 per file, its README for the provenance and the measurements).
`src/quantum/chemistry/qdynamics_vendored.py` runs it as subprocesses
(`write_input_deck` -> `run_pyscf_driver` -> `run_qubitizer`, or
`his_hamiltonian_json` for the chain; pyscf + h5py needed, e.g.
`~/.venvs/doron-bridge-gpu`), and `qdynamics_interop.py` matches his mode order
and mode signs to ours (`match_modes_to_reference`) and reports the largest
difference per class of Pauli word (`classify_term_differences`: identity,
electronic, oscillator, coupling). `test/test_qdynamics_vendored_crosscheck.py`
runs his driver and qubitizer on H2O/STO-3G at the RHF optimum and compares
with `builder_model_in_qdynamics_conventions`: electronic terms equal to 1e-12;
with `isotope_masses` in his input deck the oscillator terms agree to 1e-9 and
the coupling terms to 2e-5 (his Hessian keeps translations and rotations, ours
projects them out) and the sector ground state to 1e-6 Ha; with his default
masses every frequency is `sqrt(MP_ME/AMU2AU)` = 1.00363 times ours (PySCF's
`eph` uses integer mass numbers times MP_ME, his scale factor undoes the MP_ME
part only), so his reference energy E_RHF + sum omega/2 sits 8.9e-5 Ha above
ours. The vendored tree has its own `conftest.py` so pytest never collects his
tests, whose `src.` imports would resolve to our package.

Acceptance: assembling from the Paper IV fixture's tensors reproduces its
operators exactly, and building H2O from scratch at the fixture geometry with
`--harmonic-curvature rhf_hessian --alignment full` reproduces its exact sector
ground energies to 5e-8 Ha at cutoffs (3,3,3), (4,4,4) and (3,7,3); with the
defaults the exact ground energy at cutoffs (3,3,3) is -74.9999118 Ha (18
qubits; -75.0002755 Ha with `--mode-basis rhf_normal_modes`). Output schema
`molecular_vibronic_qubit_hamiltonian_v1` states units, spin order, Pauli-word
order and the ladder convention.
