# Fixed-source greedy Phase-III batch-3 appendix bundle

This immutable six-regime bundle derives from the repaired v6 combinatorial
bundle.  The only scientific mechanism changed is the Phase-III post-shortlist
selector: `combinatorial_reduced_plane` becomes `greedy_reduced_plane`.  The greedy selector adds up to
three candidates using fixed-source marginal gains, then performs one joint
supported-FS trust solve and one full accepted-ansatz supported-FS-whitened
Powell refit.  Phase II remains singleton.

- batch: `paper-i-hh-sr-appendix-phase3-greedy-batch3-fullresp-symcost-noprune-nobeam-nonovelty-six-r50-20260720-v1`
- source archive SHA-256: `ef3fb0ec04b5fc0242fe6b640ec4dff57c857d440cf948fa40e2196078e939cd`
- route-contract SHA-256: `7554bb2488a26573039eb94a74e2697b38d883a53698515a9b3ed0e5ea0fef9f`
- parent source SHA-256: `f11607321e426d73627910a1da76a22a96f4d4bd82f66708b5b202b2e5a61453`
- parent route SHA-256: `27df701ab280c02422e7030ec60a77d37ff20b73132ae4824cc41017f93fa050`
- horizon: 50 controller rounds in all six regimes
- cutoff: n_ph=3 for weak-Holstein, n_ph=7 for strong-Holstein, same-cutoff references
