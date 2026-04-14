# Plan to Update AGENTS.md and run_guide.md

## Objective
Update the core repository documentation (`AGENTS.md` and `run_guide.md`) to establish the new `phase3_v1` adapt pipeline, reduced winning pools for $L=2, L=3$, and the `secant` time dynamics as the primary recommended paths.

## 1. `run_guide.md` Updates
- **Update Section 0 (Current surface map):**
  - Clarify the "canonical direct HH ADAPT" entry to explicitly state that it uses the `phase3_v1` methodology out of `adaptive_selection_staged_continuation.tex`.
  - Add explicit shorthand commands for running the `secant` time dynamics path as the definitive standard for time propagation.
  - Detail the usage of the reduced winning pools for $L=2$ and $L=3$ (referencing `Math.md`) rather than the full meta pools, to save computational resources and target physical relevance.

## 2. `AGENTS.md` Updates
- **Update Section 3 (VQE implementation rules) / Section 4 (Time-dynamics readiness):**
  - Add explicit rules regarding `phase3_v1` and the related equations from the TEX files. Make it the default for new agents checking the repository.
  - Formally document the reduced pools for $L=2, 3$ as the required pools for ADAPT/time-dynamics, explicitly banning the "full meta" pool for standard runs to prevent excessive QPU depths or memory usage.
  - Define the specific `secant` time dynamics path as the "winning" standard and default time-dynamics solver/controller for agents.

## Next Steps
Once this plan is approved, I will use `replace_string_in_file` to execute these edits smoothly and strictly adhere to your formatting rules.
