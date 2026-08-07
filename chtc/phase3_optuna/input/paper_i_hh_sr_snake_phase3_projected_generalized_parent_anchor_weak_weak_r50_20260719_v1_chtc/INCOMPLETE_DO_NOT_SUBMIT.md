# Incomplete source-lock revision

This unsubmitted v1 anchor stopped before archive-only validation. The local
preflight exposed an incomplete dependency closure: the overlaid CLI expected
a newer formal-manifold route-profile module than the frozen parent archive
contained. Preserve this directory as failed build evidence; use the immutable
v2 successor instead.
