# Phase-III batch3 serialized empty-matrix receipt repair

This immutable v5 bundle derives only from the untouched hysteresis-disabled
v4 bundle. It preserves the six fresh round-0 to round-50 regimes, route digest
`27df701ab280c02422e7030ec60a77d37ff20b73132ae4824cc41017f93fa050`,
and every scientific setting.

The only source repair restores JSON payloads such as `[]` to the matrix shape
declared by the typed Phase-III batch response receipt. This is required for
valid depth-zero blocks including `G_AA/H_AA` with shape `(0,0)` and
`G_AB/H_AB` with shape `(0,b)`. Nonempty shape mismatches and nonfinite values
still fail closed. The v4 archive remains unchanged at SHA-256
`e93b75e0cf9961d78f1ec9b41108a93deb9f5c039e15ee5371187ff7b103f299`.

The v5 source archive was built, inventoried, and locally preflighted. It was
**not submitted** by the builder.
