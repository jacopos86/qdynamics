# Weak-weak FM-SNAKE versus projected-singleton Append+FM

This is an isolated, source-locked CHTC bundle for one sequential matched pair.
The single Condor process runs `fm_snake` first and
`projected_singleton_append_fm` second through
`pipelines.exact_bench.paper_i_hh_fm_vs_append_fm_first_hit run-pair`.

Both rows use the same supported-FS-whitened inverse-RBFGS accepted-ansatz
optimizer, Armijo backtracking capped at 15 trials, qBroyden/qBANG disabled,
accepted-ansatz optimizer budget 200, seed 7, and the same-cutoff weak-weak NPH=3
Hamiltonian. Each row terminates at the first completed outer iteration with
`abs(Delta E) <= 2e-4`, or after 30 controller rounds. The comparison reports
only the winning-lineage `S_alg` and Paper-I Qiskit basis-gate opt0/seed7 cost.

The pair is intentionally sequential inside one process so it cannot exceed one
FM scientific slot. The submit contract requests 4 CPUs, 24 GiB RAM, and 40 GiB
disk. Output/error streaming is disabled. Only a compressed, terminal
results/Qiskit/query/provenance allowlist is transferred on success. On failure
or eviction, the same narrow archive additionally preserves FM `current.json`
and the Append partial-result/progress files when present; caches and the rest of
the worker tree remain outside the transfer archive.

Preparation and validation do not submit or authenticate to CHTC. A later
authorized operator may submit `submit.sub` after checking the bundle manifest,
source archive hash, image hash, and local preflight receipt.
