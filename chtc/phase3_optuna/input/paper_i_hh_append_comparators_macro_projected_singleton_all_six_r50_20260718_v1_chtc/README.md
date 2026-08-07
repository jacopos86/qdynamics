# Paper-I Append-ADAPT completion comparator bundle

Prepared for authenticated submission. This immutable bundle contains twelve fresh
Hubbard--Holstein jobs: Append-ADAPT with unsplit `full_meta` macro generators
and Append-ADAPT with projected singleton children, each over the six Paper-I
completion regimes through exactly 50 controller iterations.

The source of truth is `bundle_manifest.json` plus the per-job normalized
manifests. The authenticated image/Qiskit gate passed and `submit.sub` now
requires `TARGET.HasSIF`.

The approved comparator changes relative to the six visible Append source
locks are recorded in `settings_difference_audit.json`. No result in this
bundle is a continuation: every row begins at round zero with seed 7, no HH
preseed, and the same working/reference phonon cutoff (3/3 or 7/7).
