# Cross-Paper Scientific Flow and ICM/GitNexus Pilot

Status: architecture and implementation-sequencing plan only.

This file does not authorize a scientific run, repository index, scheduler
action, manuscript edit, evidence replacement, compatibility deletion, commit,
or push.

## Governing purpose

Paper numbers identify evidence and manuscript ownership. They do not define a
software inheritance hierarchy.

Paper I and Paper IV are peer **static-construction producer lanes**:

- Paper I resolves Hubbard--Holstein problems and owns its static
  ADAPT/SNAKE evidence.
- Paper IV resolves molecular-vibronic problems and owns its static
  ADAPT/SNAKE evidence.
- Both may invoke the same paper-neutral static ansatz-construction method.
- Swapping Paper I for Paper IV changes the resolved physical problem and its
  family-owned capabilities; it does not create a second SNAKE algorithm.

Paper II is a **dynamics consumer lane**. It may begin from a completed static
ansatz produced by Paper I or Paper IV. It consumes a neutral accepted-ansatz
export rather than importing another paper's campaign or artifact schema.

Paper III is a **QSE/excited-dynamics consumer lane**. It may consume a static
accepted-ansatz export, a Paper-II dynamics export, or an explicitly defined
combination. Paper III owns the later adapters and scientific validation for
those inputs; this plan does not prematurely define that interface.

## Scientific flow

```text
Paper-I HH problem adapter ───────┐
                                  ├─> static ansatz construction
Paper-IV vibronic problem adapter ┘        │
                                           ▼
                                accepted ansatz export
                                  │                 │
                                  ▼                 ▼
                         Paper-I/IV evidence   Paper-II dynamics
                                                      │
                                                      ▼
                                            dynamics export
                                               │
                     accepted ansatz export ───┴──> Paper III
```

Evidence never changes owners:

- a Paper-I export still points to immutable Paper-I evidence;
- a Paper-IV export still points to immutable Paper-IV evidence;
- Paper II owns the import receipt and every dynamics artifact it produces;
- Paper III owns its import receipts and downstream results.

## Stable concepts

**Static ansatz construction** is the paper-neutral application of SNAKE or an
explicit comparator to one resolved physical problem.

**Accepted ansatz export** is an immutable, problem-bound accepted prefix from
a completed static-construction result. It records construction method,
inclusive controller-round cutoff, ordered generators, parameter identities
and values, reference-state and prepared-state identities, problem/sector/
cutoff binding, and source hashes.

**Dynamics export** is a Paper-II-owned, immutable trajectory result suitable
for an explicitly defined downstream consumer.

**Paper-lane adapter** translates a lane-owned source into a neutral input or a
neutral result into lane-owned evidence. It does not transfer evidence
ownership.

Avoid:

- “Paper II runs Paper I”;
- “Paper IV is a Paper-I route”;
- “Paper III inherits the earlier papers”;
- a generic cross-paper artifact mapping with optional fields for every lane.

## Deep interfaces

The reusable static method is:

```python
result = run_ra_adapt(problem, request=None)
```

`problem` owns the Hamiltonian, register, sector, reference state, cutoff,
exact-comparison space, pool capabilities, and hard guards. `request` owns
method, execution, and observation policy. Paper identity is not an argument.

The cross-paper static handoff is conceptually:

```python
export = export_accepted_ansatz(
    result,
    through_controller_round=10,
)
```

The selected controller round is inclusive. The export is method-neutral
enough to represent SNAKE or append-ADAPT while retaining the exact construction
method in provenance. Geo-ADAPT is not an ordinary downstream source.

Paper II consumes the export through its own adapter:

```python
dynamics_seed = load_dynamics_seed(export)
```

The adapter validates problem identity, encoding, cutoff, sector, ordered
generators, parameters, replay, and prepared-state parity. It never searches
artifact trees or silently chooses another accepted prefix.

No universal `run_paper(...)` or cross-paper settings union is planned.
Paper-specific campaigns remain thin orchestration around the neutral method
and handoff interfaces.

## Current executable facts

- `ResolvedProblemContext` already represents multiple Hamiltonian families.
- The problem registry already resolves Hubbard--Holstein and molecular-
  vibronic H2/H2O families.
- The typed `run_ra_adapt` facade currently rejects every problem except
  Hubbard--Holstein `L=2`. That restriction protects Paper-I behavior today but
  prevents Paper IV from using the same public method seam.
- AP-McLachlan already has a neutral scaffold-state adapter, but current
  Paper-II launch surfaces still depend on artifact-oriented source routing
  rather than one accepted-ansatz export.
- The canonical Paper-I run summary already owns accepted-prefix selection and
  reporting. A cross-paper exporter should consume its typed run/result
  contracts rather than normalize historical result schemas.

The future Paper-IV generalization must move the Hubbard--Holstein `L=2` lock
into the Paper-I problem adapter while preserving route-faithful tests for every
newly admitted problem family. It must not assume that an HH pool or symmetry
guard applies unchanged to molecular-vibronic physics.

## ICM purpose

ICM is the progressive-disclosure control surface for choosing the correct
workflow, not a Paper-I campaign launcher.

```text
user intent
  -> inspect or analyze completed evidence?
       -> owning paper's provenance/reporting adapter
  -> export an accepted static ansatz?
       -> source paper -> accepted ansatz export
  -> run remaining Paper-I pruning or batching study?
       -> source-locked Paper-I study workflow
  -> run molecular-vibronic static construction?
       -> Paper-IV problem adapter -> shared static method
  -> run time dynamics?
       -> Paper-II import of a Paper-I or Paper-IV accepted ansatz export
  -> run QSE or excited dynamics?
       -> Paper-III-owned import of declared static and/or dynamics exports
```

Each actual campaign may use:

```text
intent -> lock -> execute -> validate -> analyze -> review
```

These stages contain references, receipts, and decisions. They do not duplicate
scientific defaults, source artifacts, or paper evidence.

Folder presence never determines completion. Execution always requires a
separate current authorization record.

## Immediate Paper-I scope

Do not build a six-regime canonical Paper-I launcher merely to create an ICM
pilot. The user expects few or no further ordinary canonical Paper-I runs.

Remaining Paper-I pruning and batching runs are explicit source-locked studies
using the existing typed SR request and Paper-I run gate. They may demonstrate
ICM stage receipts, but they do not create a permanent generic campaign
interface.

Preserve completed Paper-I SNAKE and append artifacts through publication.
Their later neutral export adapter must reference them immutably and must not
rewrite, relabel, or normalize the source evidence.

Detailed Paper-II, Paper-III, and Paper-IV implementation remains deferred
until its lane becomes active. The relationships above prevent the Paper-I
cleanup from blocking those later consumers.

## GitNexus pilot

GitNexus remains a read-only, index-only structural sidecar. It is not part of
execution, scientific authority, or evidence selection.

Before indexing:

- add a strict `.gitnexusignore`;
- exclude generated outputs, artifacts, PDFs, caches, archives, environments,
  and editor state;
- include executable code, tests, consumed schemas/configs, and importable
  compatibility code needed for reachability audits;
- never run `gitnexus setup`;
- generate no `AGENTS.md`, hooks, skills, or context files.

The first useful questions are:

1. Which callers reach `run_ra_adapt`, and which remaining callers use the
   historical `run_sr_snake` compatibility facade?
2. Where is the Hubbard--Holstein `L=2` restriction enforced?
3. Which family-specific capabilities already exist for molecular-vibronic
   problem resolution, pools, sectors, and guards?
4. How do current Paper-II runners ingest static ansätze?
5. Which callers still require `paper_i_runner.py` and other quarantined
   compatibility code?

Every consequential graph answer requires direct source and focused-test
confirmation.

## Compatibility and cleanup register

| Candidate | Current status | Handling |
|---|---|---|
| `agent_guidance/skills/paper-i-run/SKILL.md` | compact operational gate | retain |
| `agent_guidance/shared/run-guide.md` | thin cross-paper run router | retain |
| `pipelines/static_adapt/paper_i_runner.py` | author-retired after caller migration and scheduler verification | inert snapshot recorded in `archive/paper_i_static_adapt_legacy_20260727/MANIFEST.json` |
| `pipelines/exact_bench/paper_i_hh_powell_pareto.py` | author-retired after bundle replacement and scheduler verification | inert snapshot recorded in `archive/paper_i_static_adapt_legacy_20260727/MANIFEST.json` |
| `pipelines/run_guide.md` | redundant compatibility pointer | hide now; review after publication |
| detailed SR-SNAKE handoffs and refactor plan | implementation/provenance history | hide through publication; review afterward |
| `agent_guidance/qse/ubiquitous-language.md` | honest self-contained placeholder; missing shared-glossary edge removed | retain; defer the Paper-III semantic pass |

Compatibility and historical identities remain preservation-only. Retired
executable source is represented by hash-locked inert snapshots; current
ordinary navigation does not expose it as an execution choice.

## Implementation order

1. Completed: contract the root/Paper-I routing and run guidance.
2. Record the peer-producer and cross-paper-export model in the Paper-I
   glossary and paper-lane plan.
3. Finish only the explicitly requested source-locked Paper-I prune/batch
   studies and Paper-I publication work.
4. When Paper II becomes active, implement the accepted-ansatz export/import
   seam against both SNAKE and append fixtures.
5. When Paper IV becomes active, generalize the static-construction seam using
   explicit molecular-vibronic problem-family capabilities and route-faithful
   tests.
6. When Paper III becomes active, define its owned import contracts from static
   and/or dynamics exports.
7. Add ICM workspaces around the first real workflow that benefits from them;
   do not create empty paper scaffolds.
8. Run the GitNexus index-only pilot when a concrete reachability question is
   being implemented.

## Acceptance criteria

- Paper I and Paper IV remain equal-footing owners of their static evidence.
- SNAKE behavior is reusable without importing Paper-I evidence ownership.
- Paper II can consume an accepted ansatz from either producer through one
  neutral export contract.
- Paper III can later declare static and dynamics inputs without importing
  upstream campaign schemas.
- Paper-specific physical defaults remain in problem adapters and source locks.
- Optional method policy remains progressively disclosed.
- No generic cross-paper flag union, artifact search, fallback route, or
  duplicated settings authority is introduced.
- ICM and GitNexus remain control/navigation tools rather than scientific
  runtime dependencies.

## Unresolved questions

Paper-III input composition and Paper-IV family-specific SNAKE admissibility
remain intentionally deferred scientific contracts. They are not Paper-I
cleanup blockers.

Files to edit:

- `agent_guidance/shared/icm-gitnexus-pilot-plan.md`
- `agent_guidance/paper-lane-refactor-plan.md`
- `agent_guidance/static-adapt/CONTEXT.md`
- Code: none in this planning step.
