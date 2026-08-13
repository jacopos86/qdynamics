# Psi4 EPH derivative helper

Psi4 1.9 exposes total four-centre integral derivatives to Python, but not the
single-AO-leg derivative needed by the molecular finite-difference EPH Pulay
correction. This plugin contracts that missing derivative with the reference
RHF density and returns three AO matrices through wavefunction array variables.

`FiniteDifferenceElectronPhononSolver` builds the plugin lazily in the system
temporary directory. Building requires the Psi4 development headers, CMake,
pybind11, Boost headers, and Eigen headers. A prebuilt library can instead be
selected with:

```bash
export QDYNAMICS_PSI4_EPH_PLUGIN=/absolute/path/qdynamics_eph_deriv.so
```

The current contraction supports Cartesian shells. This covers the s and p
shells used by the present molecular STO-3G workflow; pure d and higher shells
are rejected rather than silently producing an incorrectly transformed tensor.
