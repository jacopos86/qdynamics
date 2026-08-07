# Stale preflight bundle

Do not submit v4.  Archive-only validation correctly preserved the inherited
v5/v6 operational-repair receipts, but the worker validator compared their
historical route field against the new greedy route.  No scientific work was
run.  V5 compares those inherited receipts against their authenticated parent.
