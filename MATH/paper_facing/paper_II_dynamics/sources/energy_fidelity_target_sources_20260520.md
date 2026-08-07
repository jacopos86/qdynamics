# Energy and Fidelity Target Sources for Paper-II Reuse

Purpose: preserve the source trail for energy-density and fidelity targets that may affect Paper-II reusable ground-state seeds, spectra, observables, and downstream dynamics claims.

## Local source dossier

- `energy_density_targets_adapt_vqe_lattice_many_body_20260520.pdf`
  - Local copy of the policy dossier originally stored at `/Users/jakestrobel/Downloads/Energy-Density Targets for ADAPT and VQE Benchmarks in Lattice Many-Body Systems.pdf`.
  - Use as internal synthesis only. Cite primary literature in manuscripts.

## Target policy under discussion

- Molecular energy target:
  - `tau_E^mol = 1.5936e-3 E_h`.
  - Interpretation: chemical accuracy, 1 kcal/mol scale.

- Lattice/control energy-density target:
  - `delta e = |E_alg - E_ref| / (N_phys E0)`.
  - Primary reviewer-defensible target from recent Hubbard/VQE scale: `delta e <= 3e-4`.
  - Stricter SNAKE/reuse target when data support it: `delta e <= 1e-4`.

- Reusable ground-state fidelity target:
  - `1 - F <= 1e-4`.
  - Interpretation: co-primary reuse diagnostic for downstream spectra, observables, and dynamics, not a replacement for observable reporting.

## Primary sources to cite

1. Qubit coupled-cluster / chemical accuracy
   - URL: https://arxiv.org/abs/1809.03827
   - Use: supports molecular chemical-accuracy target near 1 kcal/mol.

2. Qubit-ADAPT implementation for H2
   - URL: https://arxiv.org/abs/2308.07259
   - Use: supports molecular/H2 chemical-accuracy framing near 1.6 mHa.

3. Classical Benchmarks for VQE Simulations of the Hubbard Model
   - URL: https://arxiv.org/abs/2408.00836
   - Use: supports per-site lattice energy error as the reporting currency and motivates the `3e-4` to `1e-3` high-performing range.

4. Variational quantum eigensolvers for sparse Hamiltonians
   - URL: https://arxiv.org/abs/2012.07171
   - Use: supports sampling burden scaling like `epsilon^-2`; useful for explaining why `1e-6` to `1e-8` are not main-table targets.

5. Accelerated Variational Quantum Eigensolver
   - URL: https://arxiv.org/abs/1802.00171
   - Use: additional support for VQE expectation-estimation sample scaling.

6. Adaptive variational preparation of Fermi-Hubbard eigenstates
   - URL: https://arxiv.org/abs/2109.12126
   - Use: supports tracking fidelity alongside energy for Hubbard-state preparation.

7. Fuchs--van de Graaf distinguishability source
   - URL: https://arxiv.org/abs/quant-ph/9712042
   - Use: supports relating fidelity to distinguishability and worst-case bounded-observable stability.

8. Variational Hamiltonian Ansatz Hubbard-chain fidelity caveat
   - URL: https://arxiv.org/abs/2111.11996
   - Use: supports the caveat that fidelity should not replace observable reporting.

## Classical Hubbard benchmark support from dossier

- LeBlanc et al., `Solutions of the Two-Dimensional Hubbard Model: Benchmarks and Results from a Wide Range of Numerical Algorithms`, Phys. Rev. X 5, 041041 (2015).
  - Use: supports `1e-4`-scale favorable-case Hubbard reference precision and `1e-3`-scale harder-regime caution.

- Qin, Shi, and Zhang, `Benchmark study of the two-dimensional Hubbard model with auxiliary-field quantum Monte Carlo method`, Phys. Rev. B 94, 085103 (2016).
  - Use: supports high-accuracy AFQMC reference construction and benchmark-scale precision.

## Paper-II relevance

- If Paper-II uses Paper-I ansatzes as dynamics seeds, energy convergence alone is not enough.
- Paper-II seed admission should retain energy target status, fidelity status, spectra/observable diagnostics, and cutoff/reference provenance.
- Exact-reference energy/fidelity/observable values may remain diagnostic/reporting-only; they must not enter QPU-faithful controller decisions.
