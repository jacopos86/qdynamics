# Paper V Agent Contract

Scope: Paper V, high-`U` regularization / GKBA exploration, stability
diagnostics, and quantum-computable encoding planning under `paper_5/`.

Follow this authority chain:

1. Root `AGENTS.md`.
2. `MATH/AGENTS.md` when paper-program, manuscript, math-default, or
   paper-support scope is triggered.
3. `paper_5/pyproject.toml` and only the target `paper_5/notes/`, `src/`,
   `tests/`, or `references/` files named by the task.

No `paper_5/AGENTS.md` or `paper_5/README.md` exists in this checkout. Do not
invent either contract during unrelated work.

Paper V does not yet have a dedicated run skill or results skill. The active
draft manuscript is `MATH/paper_details/paper_V_high_u_gkba.tex`; code, tests,
references, and exploratory notes stay in `paper_5/` unless the user explicitly
asks to move them. If Paper V reuses parent-repo quantum primitives, preserve
the root Pauli, JW, and number-operator conventions.
