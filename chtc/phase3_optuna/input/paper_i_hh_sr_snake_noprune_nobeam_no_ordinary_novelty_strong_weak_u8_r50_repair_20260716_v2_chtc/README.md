# Strong-weak SR-SNAKE round-50 source repair v2

Status: submitted as CHTC cluster `8811308` on 2026-07-16.

This single-row bundle resumes the exact authenticated round-30 strong-weak
prefix used by cluster 8811168. The only source change is the verified
no-batching telemetry initialization in `pipelines/static_adapt/adapt_pipeline.py`.
No scientific setting, checkpoint, estimator ledger, route policy, optimizer
budget, horizon, or reference changed.

- Parent archive SHA-256: `070febcf91a31fc1249afd24f59b9a68e57c6ed547315cabed461933b51b1c2a`
- Repaired archive SHA-256: `e682e79d4c9218794c94822ebce99df427f0840287e5212a45e528931cb2efc5`
- Repair patch SHA-256: `33fce0f0eb608437b6d329eccd598b9252336d4b13126435b067f54475d00b0b`
- Job validation: pass
