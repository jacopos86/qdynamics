# Repository Partition and ICM Desires

## Purpose

The repository should be organized around a stable shared kernel and explicit
paper-owned lanes. Paper identity should determine ownership, defaults,
evidence, and workflow routing without duplicating genuinely shared methods.

The partition is resolved progressively. The high-level architecture below is
fixed first; finer structure is then derived in successive passes from
GitNexus structural evidence rather than declared up front. Each pass adds one
level of granularity to a dendritic ownership tree and must close before the
next begins.

This document expresses architectural desires and the refinement method only.
It does not authorize code edits, file moves, migrations, scientific runs, or
evidence changes.

## Authority and relationship to other contracts

This document is subordinate to `AGENTS.md` and, where scientific scope
applies, to `MATH/AGENTS.md` and the selected lane contract. It defines target
structure and refinement method; it never overrides route identity, typed
defaults, source locks, or evidence contracts.

`CONTEXT-MAP.md` supplies the ubiquitous ownership language used here. Where
the two overlap, `CONTEXT-MAP.md` owns terminology and this file owns target
structure and migration sequencing.

## Desired root structure

```text
kernel/
papers/
  paper_i/
  paper_ii/
  paper_iii/
  paper_iv/
  paper_v/
  paper_vi/
icm/
```

Existing directories may remain during migration, but their ownership should
eventually resolve into this model.

## Kernel

The kernel is the stable substrate shared by multiple paper lanes.

Kernel responsibilities include:

- neutral mathematical and quantum primitives;
- shared typed contracts;
- problem, register, sector, and state interfaces;
- accepted-ansatz and dynamics-export contracts;
- validation and provenance primitives;
- genuinely paper-neutral algorithmic infrastructure.

Kernel code must not contain:

- paper-specific scientific defaults;
- manuscript claims or evidence;
- paper-owned source locks;
- paper-specific campaign settings;
- imports from `papers/paper_*`.

Shared code is a candidate for the kernel, not automatically kernel code. Any
shared module containing paper-specific behavior must be separated or adapted.

### Worked example: the staged energy law

This is the reference case for how a kernel boundary is identified. It is
recorded as architectural direction only; it changes no scientific behavior and
authorizes no edit. Scope is the energy models. Cost shaping, insertion,
batching, pruning, and beam are out of scope here.

The three RA-ADAPT scoring phases evaluate one energy law:

```text
E(g, h, F, rho) = max over 0 <= alpha <= rho/sqrt(F)
                    [ g*alpha - (1/2)*h_+*alpha^2 ]
```

It exists in source as `trust_region_drop(g_lcb, h_eff, F, rho)` at
`pipelines/scaffold/hh_continuation_scoring.py:2314`, returning the interior
Newton optimum `g^2/(2 h_+)` when `g/h_+ <= rho/sqrt(F)` and the trust-boundary
value otherwise.

The phases do not use different laws. They supply progressively better
geometry to the same law, with an identical gradient argument
(`_selector_gradient_lcb`) throughout:

| Phase | Curvature `h` | Metric `F` | Call site |
|---|---|---|---|
| I, canonical first-order | `0` | measured `F` | closed form, `:2163-2170` |
| I, legacy compatibility | `lambda_F * F` proxy | legacy `F` | `:2178` |
| II | measured directional `h_raw` | `F_raw` | `:2766` |
| III | reduced active-plus-candidate `h_eff` | reduced `F_red` | `:3578` |

Consequences for the partition:

- The law is paper-neutral scalar mathematics and is a **kernel** candidate.
- The geometry suppliers that produce `(g, h, F)` are method- and lane-owned
  and stay outside the kernel.
- Phase I canonical is the `h -> 0` limit of the law: `rho*g/sqrt(F)` equals
  the kernel evaluated at zero curvature. It currently reaches that value
  through a separate closed form rather than through the shared function.
  Phase II and Phase III are not limits of each other; they are the same law
  evaluated on refined measured geometry.
- The law therefore has one implementation and one duplicate. Recording the
  duplicate is the point of this example: it is the small, provable kind of
  boundary defect that refinement is meant to surface.

Constraint on any future unification: the Phase-I closed form and the kernel
are algebraically equal but not guaranteed bit-identical, because they
associate `rho`, `g`, and `sqrt(F)` differently. Phase-I rankings can therefore
move at tie boundaries. Unification is route-identity-sensitive, requires a
numerical-equivalence regression and typed-settings synchronization, and is not
a free cleanup.

## Paper lanes

Each paper lane owns:

- its physical problem adapters;
- scientific defaults and source locks;
- workflow orchestration;
- paper-specific validation;
- evidence and provenance;
- manuscript support;
- lane-specific tests.

Paper lanes may depend on the kernel. They must not import another paper's
campaigns, evidence, defaults, or artifact schemas.

Cross-paper consumption must use explicit immutable exports.

```text
Paper I or Paper IV
  -> accepted ansatz export
  -> Paper II dynamics
  -> dynamics export
  -> Paper III QSE/excited-state workflow
```

Paper VI should remain a future declaration until its scientific workflow and
ownership contract exist. Empty executable scaffolds should not be created
merely to reserve a paper number.

## ICM

ICM is the progressive-disclosure control surface for the repository.

It should route:

```text
user intent
  -> domain or paper lane
  -> applicable contract
  -> workflow
  -> typed result or export
  -> owning evidence
```

ICM may contain:

- lane manifests;
- workflow declarations;
- stage definitions;
- ownership metadata;
- context and handoff receipts;
- links to the next required guidance surface.

ICM must not:

- become a universal `run_paper(...)` API;
- contain scientific defaults;
- launch scientific work directly;
- search artifacts and silently choose a result;
- provide cross-paper fallback behavior;
- become a runtime dependency of the kernel.

### Navigation versus campaign receipts

Two distinct things currently share the ICM name and must not be merged:

| Concept | Content | Owner | Target location |
|---|---|---|---|
| ICM navigation | lane manifests, workflow declarations, stage definitions, ownership metadata, routing links | repository | `icm/` |
| Campaign stage receipts | one campaign's `lock`/`refactor`/`verify`/`materialize`/`analyze`/`user-review` receipts, hash-linked to its inputs and outputs | producing paper lane | the lane, e.g. `papers/paper_i/icm/<campaign_id>/` |

A campaign receipt records one lane's provenance and therefore stays with that
lane by the ownership principles below. Hoisting campaign receipts into the
neutral `icm/` surface is an ownership violation, not a consolidation. The
existing `agent_guidance/static-adapt/icm/ra-adapt-repair-20260727/` chain is
lane-owned campaign provenance and migrates with Paper I.

## Current-directory ownership resolution

Every top-level directory resolves to exactly one target owner. `stays in
place` means the physical path does not move; only ownership metadata is
recorded.

| Current path | Target owner | Disposition |
|---|---|---|
| `src/quantum/` | kernel | primary promotion candidate; already paper-neutral |
| `pipelines/contracts/` | kernel | typed problem/provenance contracts; classify per module |
| `pipelines/scaffold/` | contended | contains paper-neutral geometry and Paper-I scoring policy in one tree; separation is a named work item (R2 below) |
| `pipelines/static_adapt/` | `papers/paper_i/` | includes `ra_adapt/`, route profiles, and Paper-I source locks |
| `pipelines/time_dynamics/` | `papers/paper_ii/` | currently imports Paper-I modules; see R2 |
| `pipelines/qse_spectra/`, `pipelines/excited_dynamics/` | `papers/paper_iii/` | classify before moving |
| `paper_5/` | `papers/paper_v/` | resolve duplication with `agent_guidance/paper-v/` |
| `pipelines/exact_bench/`, `pipelines/reporting/`, `pipelines/hardcoded/`, `pipelines/pareto_offline/`, `pipelines/error_protected/` | mixed | per-module classification required; expect a kernel/lane split |
| `MATH/paper_details/`, `MATH/paper_facing/` | per-paper | manuscript and paper-support ownership is already per-paper; physical split is deferred and low priority |
| `test/` | mixed | flat suite; lane-specific tests follow their lane, contract tests follow the kernel |
| `chtc/` | producing lane | bundles and submissions are lane-owned operational provenance |
| `agent_guidance/` | mixed | shared contracts to `icm/`; lane contracts to their lane |
| `archive/`, `raw_outputs/`, `output/`, `artifacts/`, `logs/`, `tmp/`, `docs/` | preserved | stays in place; immutable provenance and generated output are never reorganized for tidiness |

An unassigned directory blocks the level at which it appears. Do not migrate
around an unresolved owner.

## Progressive disclosure rules

- If the request is repository inspection, reveal the repository map and
  nearest ownership contract.
- If the request names a paper, reveal only that paper's lane contract and
  relevant kernel interfaces.
- If the request concerns a shared method, first resolve the owning physical
  problem, then reveal the neutral method interface.
- If the request consumes another paper's result, require an explicit typed
  export and import receipt.
- If the request involves a run, evidence, manuscript, or external state,
  reveal the appropriate operational gate only at that point.
- Never expose every historical route, compatibility identity, artifact tree,
  or policy file by default.
- Never silently retry another paper lane or compatibility route after failure.

## Ownership principles

- Paper numbers identify evidence and manuscript ownership.
- Shared methods do not transfer evidence ownership.
- A producer owns the source result and immutable provenance.
- A consumer owns its import receipt and downstream interpretation.
- Scientific defaults remain with the physical problem or paper adapter.
- The kernel owns interfaces and invariants, not paper claims.
- ICM owns navigation, not scientific authority.

## Dendritic refinement with GitNexus

### GitNexus role and hard constraints

GitNexus is a read-only, index-only structural sidecar used to derive the
ownership tree from observed dependency structure. It is never a runtime
dependency, scientific authority, evidence selector, or execution surface.

Binding constraints:

- never run `gitnexus setup`;
- generate no agent files, hooks, skills, or context files;
- maintain the strict `.gitnexusignore` surface; generated outputs, artifacts,
  PDFs, caches, archives, environments, and editor state stay excluded;
- confirm every consequential graph claim directly in source and focused tests
  before it changes a boundary, a move, or a contract;
- an index answer proposes a partition; source and tests ratify it.

### Current index state

The index is current and usable as the starting evidence surface:

```text
schemaVersion   5
lastCommit      a42de3b64b094a9920a131f6396fd41cff56a692
indexedAt       2026-07-28T03:54:46Z
files           1592
nodes           52682
edges           95921
communities     1776
processes       300
capabilities    graph, full-text search, vector search
```

Reindex before each refinement pass and record the commit and `indexedAt` in
that pass's receipt. A pass may not consume an index older than the tree state
it is refining.

### Granularity levels

The ownership tree is refined one level per pass. Each level is a strict
subdivision of its parent; a node is never reassigned across parents by a
later pass without an explicit supersession record.

| Level | Unit | Question the level answers | Primary evidence |
|---|---|---|---|
| L0 | root partition | kernel, papers, or ICM? | the fixed structure above |
| L1 | lane or kernel subtree | which paper owns this subtree? | import direction across candidate boundaries |
| L2 | subsystem | which coherent subsystem does this cluster form? | graph communities, restricted to one L1 owner |
| L3 | module family | which modules move together and which split? | intra-community edges, call processes, fan-in/fan-out |
| L4 | module and public seam | what is the stable exported interface? | node-level qualified names and their external callers |

Community and process counts are structural proposals, not an ownership
answer. A community that spans two L1 owners is a boundary defect to resolve,
not a new shared directory to create.

### Refinement loop

Each pass runs the same closed loop and emits one receipt:

```text
1. reindex and record commit, indexedAt, and candidate count
2. query the graph at the current level
3. classify every node: kernel, one lane, contended, or unresolved
4. confirm each consequential classification in source and focused tests
5. resolve contended nodes by separation or adapter, not by duplication
6. add boundary tests at the new level
7. record the level in the tree manifest with its evidence
8. close the pass; unresolved nodes block only their own branch
```

Exit criteria for a level:

- every node at that level has exactly one owner or an explicit `unresolved`
  record naming what would resolve it;
- no cross-lane import crosses the level's boundary except through a typed
  export;
- boundary tests exist and pass at that level;
- the tree manifest and receipt are written and hash-linked to the index used.

### Known first targets

These are already visible in the current structure and should be the first
work items when refinement begins.

- **R1 — kernel promotion candidate.** `src/quantum/` is the cleanest
  paper-neutral subtree and should be classified first to establish the kernel
  boundary and its boundary tests.
- **R2 — cross-lane scoring coupling.** Eighteen modules under
  `pipelines/time_dynamics/` import from `pipelines.scaffold` or
  `pipelines.static_adapt`. The largest contended module is
  `pipelines/scaffold/hh_continuation_scoring.py` (~18,800 lines), with
  fan-in from fifteen `pipelines/static_adapt/` modules, three
  `pipelines/time_dynamics/` modules, and twenty test files. Separating its
  paper-neutral geometry from Paper-I scoring policy is the gating
  prerequisite for the acceptance criterion that lanes cannot import one
  another's defaults. Resolve it behind adapters with import paths preserved.
- **R3 — Paper-V duplication.** `paper_5/` and `agent_guidance/paper-v/`
  describe one lane in two places; resolve ownership before either moves.

## Migration principles

- Begin with architecture and dependency classification.
- Preserve existing import paths while ownership moves behind adapters.
- Separate paper-specific behavior from shared implementations before promoting
  modules into the kernel.
- Keep historical compatibility code quarantined and provenance-preserving.
- Add boundary tests before changing physical locations.
- Do not introduce a universal settings union or generic cross-paper runner.
- Do not modify scientific behavior as a side effect of reorganization.
- Emit a move record for every relocation: old path and qualified name, new
  path and qualified name, and content SHA-256 at move time. Move records keep
  archive provenance, source locks, and external semantic bindings resolvable
  across the migration.

## Sequencing preconditions

Refinement passes L0 through L2 are read-only classification and may proceed at
any time. Physical relocation is gated.

Do not move a file while a materialized run bundle pins it. The current
Paper-I Study-1 materialization verifies 144 implementation files by exact path
and SHA-256 and pins the resulting inventory digest; the pinned set extends
beyond `pipelines/` and includes report utilities under `docs/reports/`. Any
relocation invalidates that verification and forces rematerialization of
validated, unsubmitted bundles.

Physical relocation therefore begins only after the active Paper-I campaign
reaches its evidence decisions, and each relocation carries its move record.
The same rule applies to any later lane with pinned bundles.

## Desired acceptance state

The architecture is successful when:

- every active paper has an obvious owner directory;
- kernel dependencies are paper-neutral;
- paper lanes cannot accidentally import one another's evidence or defaults;
- ICM resolves user intent through progressive disclosure;
- cross-paper handoffs use typed immutable exports;
- historical paths remain traceable without being ordinary execution choices;
- a higher-level agent can locate the correct files without scanning the whole
  repository.

The refinement method is successful when:

- each granularity level closed with a receipt naming the index commit,
  the classification, and its source confirmation;
- no graph claim entered a boundary decision without source ratification;
- contended modules were separated or adapted rather than duplicated;
- every relocation carries a move record and every unresolved node names what
  would resolve it.

Files to edit:

- `repo-architecture-plan.md` (this file).
- Code, tests, configs, manuscripts, bundles, and evidence: none.
