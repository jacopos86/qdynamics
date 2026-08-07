# Open-dynamics evidence pipeline

This package supports a bounded scientific result: on the stored 12-point
Holstein-dimer grid, it compares the independent 31-coordinate closure with a
matched coherent-only limit and tests whether joint-Gram loss tracks observable
error. It also normalizes solver-native electron--phonon outputs into a common
reduced-trajectory contract without changing the underlying equations.

Current scope:

- a required electronic 1-RDM with declared basis, units, method identity, and
  provenance;
- capability-declared coherent, normal, anomalous, and connected
  electron--phonon moments;
- structural validation that raises on uninterpretable data;
- non-mutating physicality diagnostics;
- an explicit offline-only policy for truncated-exact references;
- a Paper V dimer adapter over the existing tested implementation; and
- declared dimer polarization and Hann-spectrum conventions.

The first fixture is
`test/open_dynamics/fixtures/riva_2026_dimer_protocol_v1.json`. It deliberately
separates statements in Riva--Simoni--Ping v1 from repository numerical choices.
The resulting short run is a source-anchored independent invariant benchmark,
not an ab initio result or reproduction of published curves.

The evidence-first extension in `closure_evidence.py` consumes the immutable,
hash-recorded Paper V run
`output/local_runs/paper_v_electron_phonon_analysis_20260801_v3/`. It adds the
previously missing matched coherent-only trajectory to every stored grid point,
compares only shared observables, and tests whether joint-Gram certificate loss
tracks electronic error. It does not modify or rerun the source evidence and
does not fabricate a Gram certificate for the coherent-only method.

Run the bounded extension with the separate Paper V source tree on the import
path:

```bash
PYTHONPATH=paper_5/src:. python -m pipelines.open_dynamics.closure_evidence \
  --source-run output/local_runs/paper_v_electron_phonon_analysis_20260801_v3 \
  --output-directory output/local_runs/ping_group_closure_evidence_YYYYMMDD_v1
```

The writer is deliberately no-overwrite: use a fresh output directory for a
new run.

Run the focused validation from the repository root:

```bash
python -m pytest -q test/open_dynamics
python -m pytest -q \
  paper_5/tests/test_matrix_scalar_parity.py \
  paper_5/tests/test_exact_driven_reference.py
```

The separate `paper5` src-layout package is loaded lazily. For direct scripts,
install `paper_5` or include `paper_5/src` in `PYTHONPATH`; importing the common
contracts themselves does not require Paper V.

Not implemented here:

- a periodic first-principles material bundle;
- a portable FeynWann producer-side exporter;
- QE/EPW or INQ ingestion;
- a residual-bath closure;
- a quantum-advantage claim; or
- any collaborator-reviewed compatibility result.

Those boundaries are intentional. Optional physics must be omitted when it is
unavailable, never represented by undocumented zeros.
