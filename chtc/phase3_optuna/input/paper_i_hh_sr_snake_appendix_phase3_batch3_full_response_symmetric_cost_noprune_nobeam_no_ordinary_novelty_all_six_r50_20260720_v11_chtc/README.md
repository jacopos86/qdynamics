# Phase-III batch3 selected-workspace receipt repair

This immutable v6 bundle derives only from the untouched v5 archive. It
preserves the six fresh round-0 to round-50 regimes, route digest
`27df701ab280c02422e7030ec60a77d37ff20b73132ae4824cc41017f93fa050`,
and every scientific setting.

The only v5-to-v6 source change makes the greedy and combinatorial non-beam
selector wrappers return their complete selector summary. The selected records,
response model, scores, and ordering are unchanged; the repair preserves the
already-computed `geometry_workspace` required by the accepted-path trust
transaction. The v5 archive remains unchanged at SHA-256
`d7ce13820d6b59bbe01a4dade40b304daa4e9a0c8e705f5ed19457561c524bd1`.

The v6 source archive was built, completely inventoried, and locally tested.
It was **not submitted** by the builder; exact remote archive/image validation
remains pending parent review.
