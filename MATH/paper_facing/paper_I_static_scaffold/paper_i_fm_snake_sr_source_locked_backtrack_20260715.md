# FM-SNAKE accepted-reoptimizer backtrack from the recovered SR source

Status: diagnostic source-locked comparison contract. This note does not
authorize manuscript edits or evidence promotion.

## Immutable weak--weak source closure

- Historical command:
  `raw_outputs/paper_i_hh_weak_weak_route4_whitened_adaptive_geometry_expansion_repair_20260712/full/command.json`
- Historical command SHA-256:
  `37751de2805875337cb8a0034a7394b02344c893e1b0a583439b1954c7c8061e`
- Historical result:
  `raw_outputs/paper_i_hh_weak_weak_route4_whitened_adaptive_geometry_expansion_repair_20260712/full/json/result.json`
- Historical absolute same-cutoff energy error:
  `4.472864776339236e-7`
- Self-contained source archive:
  `raw_outputs/paper_i_hh_sr_snake_historical_source_recovery_20260714/source_lock_revision_v2_self_contained_20260715/source_lock/paper_i_hh_sr_snake_0caf2834_self_contained_source_tree_v2.tar.gz`
- Archive SHA-256:
  `c290d9ee1b31cd211e41faad174cd2e311ca65cf351c46bbb84fbaaea9504c6c`

## Locked selector semantics

The selector is the recovered Paper-I singleton-response controller. Preserve:

- Phase 0 disabled;
- Phase-II and Phase-III batching disabled;
- singleton child forwarding only;
- windowed selector geometry with window size 3, top-k 0, periodic full refit
  every 8 rounds, and a final full refit with Powell budget 200;
- supported-metric whitening in Phase III only;
- displacement-calibrated adaptive trust;
- collective Phase-II novelty and ordinary Gram novelty;
- Phase-III novelty ablation off, including the historical N2/N3 multipliers
  and legacy pairwise novelty fallback;
- finite-angle fallback enabled at angle 0.1 with minimum improvement `1e-12`;
- the historical 3-by-2 speculative beam;
- recoverability-ladder pruning with Hessian-coupling nomination;
- structural rollback disabled.

Finite-angle fallback and novelty fallback are distinct mechanisms. Neither is
disabled in this backtrack.

## Single permitted mechanism change

The weak--weak FM variant changes only accepted-ansatz reoptimization:

- `adapt_reoptimization_route=formal_manifold_warm_start_v1`;
- route profile
  `sr_source_locked_supported_whitened_adaptive_trust_v1`;
- FM config with `qbroyd_epsilon0=0` and `line_search_max_steps=15`.

The SR selector continues to use its historical windowed geometry and cost
model. After admission, FM reoptimizes the complete accepted coordinate
registry in supported whitened manifold coordinates. These are separate typed
scopes and must not be conflated by changing `adapt_reopt_policy` to `full`.
Finite-angle selector probes remain chargeable; an admitted new coordinate is
materialized at exact zero and its supported geometry is recomputed by FM.

## Ordered gate

1. Execute the immutable archived weak--weak SR anchor and validate it against
   the recorded source result.
2. Execute current-code SR with an argv that differs only in output paths and
   require trajectory, operator-sequence, and terminal-energy parity.
3. Only after both anchors pass, execute weak--weak FM with qBroyden shadow
   epsilon zero and line-search depth 15.
4. Only after the weak--weak FM result validates, transfer the identical FM
   policy to intermediate--weak. This fourth cell is a cross-regime transfer,
   not a same-regime source-lock claim.

All stages are serial. Any selector, novelty, batching, pruning, trust,
whitening, optimizer-budget, or finite-angle drift blocks the comparison.
The live Python implementation surface used by the current-code stages is
tree-hashed at planning time and revalidated before and after every stage, so
concurrent route edits cannot silently enter later cells.
