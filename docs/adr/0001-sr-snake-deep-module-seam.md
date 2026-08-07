# Keep SR-SNAKE behind one deep run seam

SR-SNAKE will expose one run operation that accepts a resolved physical problem
and one optional typed SR-SNAKE request, then returns one typed result.
Omitting the request selects singleton admission, pruning off, beam off, a
50-controller-round maximum, no exact-ED target, a fresh start, and default
observation. Pool resolution, estimator execution, candidate phases, trust
solving, accepted refitting, accounting, checkpoint events, profile derivation,
and legacy translation stay behind that seam because exposing them would
recreate the current wide configuration surface.

The caller chooses only method policy, execution policy, and observation
policy. The runner derives the route family, profile, digest, optimizer, seed,
trust/refit policy, and numerical guards, then reports those choices as
receipts. Selection and accepted transition may form deep internal seams, but
they are not public extension points; historical routes remain separate
compatibility entry points.
