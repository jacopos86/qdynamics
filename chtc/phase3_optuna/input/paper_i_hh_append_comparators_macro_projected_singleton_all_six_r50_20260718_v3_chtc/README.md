# Paper-I Append-ADAPT completion comparator bundle (v3 queue repair)

This immutable v3 successor preserves v2's true-fresh and fail-closed Powell
cap repairs and changes only the Condor queue surface: `queue.tsv` is headerless
because `queue <vars> from <file>` treats every nonempty line as a process. The
submission therefore expands to exactly twelve real comparator jobs.

This immutable successor preserves the v1 source archive and scientific
comparator contract. It removes explicit empty continuation keyword arguments
so the runner records a true fresh round-zero invocation and selects the
already source-defined `accept_finite_nonincreasing_v1` Powell cap policy from
the normalized job manifest. That policy accepts only SciPy Powell's exact
status-2/maxiter termination with finite parameters and objective plus a finite,
non-increasing exact refit energy; every other optimizer failure still aborts.

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
