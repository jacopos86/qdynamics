# Paper-II positioned-insertion geometry audit

Date: 2026-08-24  
Branch: `codex/paper-ii-positioned-tangent-audit-20260824`  
Seed: `chtc/paper_ii_production_v2/input/seeds/hh_snake_nph1.json`

## Result

The proposed scoring/realization mismatch is not present. The structural cache
constructs the correct suffix-conjugated tangent at every cut, and the
selector's cached captured drift agrees with a fresh solve after directly
materializing the same patch. Position also changes the score substantially on
the locked calibration seed, so the commutation-reduced cut set is not
numerically degenerate.

No production selector change was made. The observed parity with AVQDS must be
explained downstream of positioned geometry: by the candidate pool, score
ordering, certification/attempt budgets, accepted edit sequence, or the
benchmark regime.

## Increment 1: independent positioned-tangent oracle

Commit: `700e7b2c`

The oracle inserts a candidate at zero angle, central-differences only the new
coordinate, and horizontalizes the derivative at the materialized ray. It does
not call the structural cache, the analytic all-tangents evaluator, or any
suffix-propagation helper.

| cut | suffix length | max absolute error | tangent L2 error |
|---:|---:|---:|---:|
| 0 | 3 | 3.053959560556e-11 | 3.734909368365e-11 |
| 1 | 2 | 3.053959560556e-11 | 3.734909368365e-11 |
| 2 | 1 | 3.053956764439e-11 | 3.734906585794e-11 |
| 3 | 0 | 1.584149896249e-11 | 2.583782734271e-11 |

All four cuts agree well below the requested `1e-8` tolerance. Agreement is
not restricted to the tail cut, so a missing suffix conjugation is ruled out.

## Increment 2: selector score versus fresh realization

Commit: `01263b1f`

The new test compares every cached `q(D,I)` in a 24-candidate small-system
enumeration with a fresh geometry evaluation and dense solve after directly
materializing the same deletion/insertion patch.

| population | count | median absolute discrepancy | p95 | maximum |
|---|---:|---:|---:|---:|
| all stay/delete/insert/exchange candidates | 24 | 0.000e+00 | 1.110e-16 | 2.220e-16 |

| cut | suffix length | candidates | median discrepancy | maximum discrepancy |
|---:|---:|---:|---:|---:|
| 0 | 3 | 8 | 0.000e+00 | 1.110e-16 |
| 1 | 2 | 4 | 1.110e-16 | 2.220e-16 |
| 3 | 0 | 8 | 0.000e+00 | 0.000e+00 |

There is no discrepancy above floating-point roundoff and no error growth with
suffix length.

## Increment 3: score spread on the locked calibration seed

The literal initial checkpoint was used: strong-fast drive contract
`A=2.4`, `omega=3.0`, drive-aligned support enabled, `t=0`, ridge `1e-7`,
`pinv_rcond=1e-10`, and zero solve damping. The active support has 28 runtime
coordinates. The nonbinding cap admits all 125 deduplicated candidate atoms,
which produce 2,029 retained positioned columns.

Across all 125 atoms, the per-atom normalized captured-drift spread across
retained cuts is:

| statistic | min | median | p95 | max |
|---|---:|---:|---:|---:|
| `max_p q(a,p) - min_p q(a,p)` | 0.000000e+00 | 2.175120e-05 | 1.488161e-02 | 9.997005e-01 |
| best-position advantage over tail representative | 0.000000e+00 | 2.117582e-22 | 1.341763e-03 | 1.731145e-02 |

For 48 of 125 atoms, a non-tail retained cut beats the tail representative by
more than `1e-12`. Thus positioning is an active scoring degree of freedom at
the checkpoint.

The atom with the largest advantage over its tail placement is
`pauli:paop_lf_full:paop_hop2(0,1)::r3::yyyyee`. The base support has
`Q=1.8042642636735991e-10` and `q=8.0592246926096700e-07`.

| cut | captured drift `Q(a,p)` | normalized `q(a,p)` | `q(a,p)-q_base` |
|---:|---:|---:|---:|
| 0 | 2.5103727222344348e-10 | 1.1213245331087223e-06 | 3.1540206384775531e-07 |
| 1 | 1.9118311541566407e-10 | 8.5397007278236533e-07 | 4.8047603521398328e-08 |
| 4 | 1.8044179151180643e-10 | 8.0599110175234981e-07 | 6.8632491382805406e-11 |
| 5 | 4.2955039341577818e-05 | 1.9187007175368537e-01 | 1.9186926583121611e-01 |
| 6 | 7.0657225609224158e-05 | 3.1560923131167595e-01 | 3.1560842538920669e-01 |
| 7 | 5.6965973170023137e-05 | 2.5445362237327129e-01 | 2.5445281645080203e-01 |
| 8 | 1.1544560075163446e-04 | 5.1566838348633637e-01 | 5.1566757756386705e-01 |
| 9 | 1.1466671583180311e-04 | 5.1218928748859971e-01 | 5.1218848156613039e-01 |
| 10 | 1.1554089230797291e-04 | 5.1609402848707275e-01 | 5.1609322256460344e-01 |
| 13 | 1.6198653623435418e-04 | 7.2355581107180178e-01 | 7.2355500514933246e-01 |
| 14 | 2.0055543645113299e-04 | 8.9583402954129643e-01 | 8.9583322361882711e-01 |
| 15 | 2.1477696647263892e-04 | 9.5935826389189405e-01 | 9.5935745796942473e-01 |
| 16 | 2.2380879063311170e-04 | 9.9970130108378896e-01 | 9.9970049516131965e-01 |
| 18 | 2.2218972530901882e-04 | 9.9246931655602910e-01 | 9.9246851063355979e-01 |
| 20 | 2.2060566274748885e-04 | 9.8539368114742898e-01 | 9.8539287522495966e-01 |
| 22 | 2.2031550046457401e-04 | 9.8409759438097610e-01 | 9.8409678845850679e-01 |
| 24 | 2.1993317764945115e-04 | 9.8238984816318387e-01 | 9.8238904224071455e-01 |
| 27 | 2.1993317764945115e-04 | 9.8238984816318387e-01 | 9.8238904224071455e-01 |
| 28 | 2.1993317764945115e-04 | 9.8238984816318387e-01 | 9.8238904224071455e-01 |

The captured-drift spread is `2.2380861019132020e-04`. Cut 16 is best; its
captured drift exceeds the ordinary tail cut 28 by
`3.8756129836605476e-06`, corresponding to a normalized advantage of
`1.7311452920605097e-02`.

## Verification

- Focused positioned-tangent test: 1 passed.
- Exchange oracle, structural selector, structural cache, and insertion-state
  files: 31 passed.
- No production source or manuscript file changed.
