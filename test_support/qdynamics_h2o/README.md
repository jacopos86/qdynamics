# Doron Wu's H2O driver outputs (qdynamics, `doron-chemistry` at 38f40805)

Origin: his own code, vendored byte for byte under `third_party/qdynamics_chemistry`
(`jacopos86/qdynamics`, branch `doron-chemistry`, commit `38f40805`), run here on
2026-11-04 with `src/quantum/chemistry/qdynamics_vendored.py` in the isolated
`~/.venvs/doron-bridge-gpu` environment (pyscf 2.14, h5py).  These are not bytes
from Doron; their origin is "simulated" in the bridge's vocabulary (his code, our
run, our input deck).

| file | what |
|---|---|
| `h2o_optiso.inp` | his input deck: H2O/STO-3G RHF at our Paper IV geometry (the RHF/STO-3G optimum, Bohr), analytic e-ph vertex, and the `isotope_masses` line that makes his files self-consistent (without it every frequency is `sqrt(MP_ME/AMU2AU)` = 1.00363 times too high; see `third_party/qdynamics_chemistry/README.md`) |
| `h2o_optiso_ele.h5` | `h1` (14 x 14, interleaved spin orbitals), `h2` (chemists' `(pq|rs)` spin-orbital Coulomb, not antisymmetrized), `nuclear_repulsion`, MO data, metadata (`e_scf` = -74.96590106193605 Ha) |
| `h2o_optiso_vib.h5` | `norm_mode` (3 modes x 3 atoms x 3, Cartesian, normalized as sum_A m_A d_A^2 = 1 in amu), `freq_wavenumber`, `reduced_mass` |
| `h2o_optiso_eph.h5` | `omega` (Hartree, descending) and `eph_mat` (3 x 7 x 7, MO basis, `g_I` with the zero-point factor `1/sqrt(2 omega_I)` included) |
| `his_optiso_c1.json`, `his_optiso_c3.json` | his qubitizer's `qdynamics_qubit_hamiltonian_v1` output from the three files at every mode cutoff 1 (17 qubits) and 3 (20 qubits), binary encoding |

`src/quantum/chemistry/qdynamics_vendored.py::vibronic_model_from_qdynamics_outputs` turns
the three HDF5 files into our `VibronicQubitModel` (full MO space, `C_I = sqrt(2 omega_I) g_I`,
modes mass-weighted with the deck's isotope masses, checks on the tensor layout and the HF
energy), and `paper_iv_h2o_fixture_from_vibronic_model.py` turns that model into the Paper IV
lane's RA-ADAPT fixture; `test/test_paper_iv_h2o_fixture_from_vibronic_model.py` checks that
the fixture's Hamiltonian equals `his_optiso_c1.json` term by term and pins these files' sha256.

sha256:

```
862c925076c6cdc0e773319214a6b4eefa0d4dd6de4a30a85cd211ed39b0bb79  h2o_optiso.inp
c7c3d5b8c77cd7c8087a92421359c6f329a425e20a981cd2be416816de359b8f  h2o_optiso_ele.h5
6407d0e441321bb758a3957f75ccdb5c1a0c727d017945522c6e43db4d653ff8  h2o_optiso_eph.h5
fe5ed93fdc4d06b051f27a81f5c4128585bcc1f869ff32703b17860e953da350  h2o_optiso_vib.h5
06f7e8f4afad31a249b0f6051630579db68577339f21d3897e9e667212d662c4  his_optiso_c1.json
c7866448184c545a236151ded668558e18b1ba52e9a55d7cc4549633711d00f6  his_optiso_c3.json
```
