# Stale bundle — do not submit

This bundle predates the 2026-07-16 user override requiring the SR-SNAKE v4
finite-angle fallback/guard to be disabled. Its manifests resolve
`adapt_finite_angle_fallback=true`, so its source digest, route digest, smoke
evidence, and preflight are obsolete. Rebuild the complete bundle from the
revised source-locked v4 profile before submission.
