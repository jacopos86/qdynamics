# Paper-I Geo-ADAPT completion comparator bundle (v3 operational repair)

This immutable successor preserves the v1 source archive and scientific job
contracts. It fixes only the Condor worker path and removes explicit empty
continuation keyword arguments so the generic runner records a true fresh
round-zero invocation. The superseded v1 bundle remains unchanged; remote v2
proved the worker-path repair but retained the continuation-classification bug.

Prepared for authenticated submission. This immutable bundle contains twelve fresh
Hubbard--Holstein jobs: Geo-ADAPT with unsplit `full_meta` macro generators and
Geo-ADAPT with symmetry/padding-valid projected singleton children, each over
the six Paper-I completion regimes through exactly 50 controller iterations.

The source of truth is `bundle_manifest.json` plus the per-job normalized
manifests. The authenticated image/Qiskit gate passed and `submit.sub` now
requires `TARGET.HasSIF`.

Every row uses seed 7, Powell with the strict 200-iteration budget, no HH
preseed, fixed-horizon stopping, and the same working/reference phonon cutoff
(3/3 or 7/7). The selector uses the exact real tangent span with an SVD
Moore--Penrose solve; true null/alias directions are removed without a
coordinate-scale-dependent regularized inverse. Geo candidates are scored with
replacement, while an immediate repeat winner skips the append for that round.

The visible-baseline resolver outputs are under `visible_source_locks/`.
`visible_source_map_resolved.json` changes only obsolete local paths to
hash-identical source copies; `settings_difference_audit.json` records the
approved completion-tracker differences and requires zero unapproved drift.
