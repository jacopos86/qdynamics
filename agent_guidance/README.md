# Agent Guidance

This folder is the organized escalation surface for repository agents. Root
`AGENTS.md` is the shared global policy/router. Codex discovers it directly;
Claude Code enters through the root `CLAUDE.md` import.

Use this folder after root `AGENTS.md` when the task has a paper/workflow identity.

Internal paper shorthand for repo-agent routing:

- **Paper I** / **paper one** means the static-ADAPT/SNAKE lane:
  `agent_guidance/static-adapt/`, `pipelines/static_adapt/`, and
  `MATH/paper_facing/paper_I_static_scaffold/`.
- **Paper II** / **paper two** means the time-dynamics/AP-McLachlan lane:
  `agent_guidance/time-dynamics/`, `pipelines/time_dynamics/`, and
  `MATH/paper_facing/paper_II_dynamics/`.
- **Paper III** / **paper three** means the QSE/excited-dynamics lane:
  `agent_guidance/qse/` and `MATH/paper_facing/paper_III_spectra/`.
- **Paper IV** / **paper four** means the molecular-vibronic water lane:
  `agent_guidance/molecular-vibronic/`,
  `MATH/paper_facing/paper_IV_molecular_vibronic_h2o/`, and
  `MATH/paper_details/molecular_vibronic_h2o_paper_IV.tex`.
- **Paper V** / **paper five** means the high-`U` regularization / GKBA lane:
  `agent_guidance/paper-v/` and `paper_5/`.

This shorthand is for internal routing and repo work. Manuscript-facing prose
should use method names rather than local paper numbers.

## Static-construction convergence

Choose the paper lane before choosing the shared method:

| User workflow | Resolve first | Shared method path | Return ownership |
|---|---|---|---|
| Paper-I static construction | Paper-I Hubbard--Holstein problem and source lock | static ADAPT/SNAKE | Paper-I validation, reporting, provenance, and evidence |
| Paper-IV static construction | Paper-IV molecular-vibronic problem and source lock | static ADAPT/SNAKE | Paper-IV validation, reporting, provenance, and evidence |

The two lanes converge only after each has produced a complete resolved
physical problem. They diverge again when the static result is interpreted or
recorded. Do not route a Paper-IV request through Paper-I Hubbard--Holstein
defaults merely because the current static method implementation lives under
`pipelines/static_adapt/`.

The current typed `run_ra_adapt` facade enforces the Paper-I
Hubbard--Holstein `L=2` scope. Treat Paper-IV use of that public seam as planned,
not executable, until family-specific admissibility and route-faithful tests
exist.

## Escalation order

1. Root `AGENTS.md` for global invariants, safety, and routing.
2. `agent_guidance/shared/repository-work.md` for repository-changing work.
3. `MATH/AGENTS.md` and
   `agent_guidance/shared/scientific-invariants.md` when scientific scope is
   triggered.
4. The relevant lane below and its `AGENTS.md` contract.
5. A lane `run-guide.md`, `CONTEXT.md`, or `ubiquitous-language.md` only when
   that file exists and the lane contract routes the task to it.
6. `agent_guidance/skills.md` for lane-to-skill routing.
7. More specific workflows, paper-support docs, source files, or artifacts
   named by that lane.

`ubiquitous-language.md` files should be dendrite dictionaries: user phrase -> math/paper anchor -> symbolic object -> run-facing leaves -> code-verification leaves. They should not force repo prose into user replies; repo leaves are for running, debugging, provenance, and verification.

`skills.md` is the skill index. Actual repo-local executable skills now live under `agent_guidance/skills/`; keep those folders intact so relative `scripts/`, `references/`, and `agents/` resources continue to resolve.

Use `agent_guidance/skills.md` as an existence-checked index. Do not infer a
skill from a directory name, an orphan script, or an old lane reference. Skills
trigger by their concrete workflow, not merely because a task belongs to a
paper lane.

For repeated Paper-I Hubbard--Holstein SNAKE candidate replay reports with
Qiskit costs and Geo/SNAKE/Append overlays, follow the Paper-I run contract and
the explicit target replay/report scripts. No repo-local replay-overlay or
Paper-I results skill exists in this checkout.

For an ordinary Paper-I RA-ADAPT request, read
`static-adapt/AGENTS.md` and then `static-adapt/run-guide.md`. Do not load the
historical route registry, refactor plans, or handoffs. Read
`static-adapt/route-identities.md` only when the user explicitly asks to
interpret or replay a preserved route/profile identity such as the legacy
`route_a` field.

## Lanes

| Lane | Scope |
|---|---|
| `static-adapt/` | Paper-I RA-ADAPT and Append-ADAPT; explicit JR-SNAKE, FM-SNAKE, SR-SNAKE, and Route-A compatibility/provenance; HH Table III and static-ansatz evidence. |
| `time-dynamics/` | Paper II, AP-McLachlan dynamics, comparator evidence, dynamics tables. |
| `qse/` | Paper III, geometry-selected QSE, excited spectra/dynamics evidence. |
| `molecular-vibronic/` | Paper IV, molecular-vibronic water application and H2O static benchmark framing. |
| `paper-v/` | Paper V, high-`U` regularization / GKBA exploratory workspace routing. |
| `shared/` | Shared run guide plus temporary holding area for guidance not yet split into lane-specific versions. |

## Current skill status

Five repo-local skills currently have a valid `SKILL.md`: Paper-I run,
Paper-I noise-model primer, Paper-II run, Paper-II results, and source-locked
sensitivity. Paper-I results and Paper-III run/results are not active skills.
Papers III-V follow their lane/root/MATH contracts and fail closed only when a
paper-facing run or evidence-transfer operation requires a missing contract.
