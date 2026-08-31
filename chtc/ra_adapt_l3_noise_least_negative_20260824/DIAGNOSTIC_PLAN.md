# Paper-I L=3 noise least-negative diagnostic

- **Run class:** diagnostic; not manuscript evidence and not eligible for automatic promotion.
- **Question:** when synthetic noise makes every feasible Phase-III predicted decrease nonpositive, does retaining the maximum signed score (the least-negative candidate) restore a meaningful RA trajectory?
- **Scientific problem:** half-filled open (L=3) Hubbard chain at (U/t=1.25), using the existing Paper-I appendix noise adapter and exact same-cutoff reference energy.
- **Route held fixed:** repaired Paper-I (L=3) RA-Append route with adaptive Phase-I--III shortlists, declared Phase-0 gradient screen with cap 24, adaptive frontier ratio (0.9), exact Qiskit compiled marginal costs, append-only insertion, and the route's matched Powell configuration.
- **Only intended selector change:** when Phase III has no strictly positive feasible score, retain exactly one candidate: the deterministic signed argmax. If all scores are negative, this is the least-negative candidate. Ordinary positive-score inverse-Simpson shortlisting is unchanged.
- **Noise smoke cell:** `low`, horizon 10, seed and noise contract inherited unchanged from the typed appendix noise application.
- **Environment:** one CPU thread; candidate-record and Hubbard-pool caches disabled so noise draws are not skipped.
- **Outputs:** `output/local_runs/ra_adapt_l3_noise_least_negative_20260824/low_h10_smoke/`.
- **Acceptance test:** the run completes ten accepted rounds without the `no positive feasible candidate` exception; its Phase-III receipts explicitly record the least-negative fallback whenever it activates; the exact diagnostic energy trajectory is non-flat and improves materially from the reference-state error.
- **Failure interpretation:** completion alone is insufficient. A flat or degrading trajectory means the fallback is not scientifically useful and must not be promoted.

## Result

The diagnostic failed its scientific acceptance test.

- The controller completed all ten requested rounds, so the typed fallback removed the Phase-III exception.
- The fallback activated in rounds 2--10. In every activation the retained Phase-III score was exactly zero, not negative; consequently, "least negative" reduced to a deterministic tie break.
- The exact diagnostic energy remained near (0.25) throughout. The terminal same-cutoff absolute error was (2.391493570746068), essentially unchanged from round 1 ((2.391493564667719)).
- The recorded coordinate models show why: after round 1, the candidate incremental gain was zero in all 18 inspected projections because the full joint gain equaled the active-only gain. The retained directions therefore did not contribute predicted candidate response.
- The live selections repeatedly favored onsite diagonal generators. Under this noise surface, the adaptive funnel had already collapsed to ineffective candidates before the Phase-III fallback was invoked.

**Decision:** do not promote this policy and do not run the high or extreme levels with it. The prior fixed-width noise route remains the valid control. A genuinely adaptive noisy route would require a separately specified noise-aware acquisition rule (for example, uncertainty-aware shortlist floors or replicated score estimates), not a sign fallback at Phase III.
