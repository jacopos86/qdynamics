# Paper I Submission Package Notes

Created: 2026-05-09  
Companion strategy memo: `MATH/paper_facing/two_paper_strategy.md`

## Core identity

Static scaffold-design paper for mixed fermion--boson variational simulation. Do not sell this as merely another ADAPT pool or batching tweak.

Autonomous terminal claim:

> Scaffold construction can be improved by local geometric and resource-aware record scoring in mixed fermion--boson settings.

## Title options

| Title option | Best venue family | Why it works |
|---|---|---|
| Geometry-Aware Adaptive Scaffolds for Mixed Fermion--Boson Simulation | PRA / Quantum / npj | Puts scaffold design before ADAPT lineage. |
| Adaptive Scaffold Construction for Mixed Fermion--Boson Variational States | PRA / Quantum | Clean and technical. |
| Hardware-Efficient Adaptive Scaffolds for Mixed Fermion--Boson Hamiltonians | PRA / PRR / npj | Best if resource--accuracy Pareto plots are strongest. |
| Local-Geometric Scaffold Construction for Mixed Fermion--Boson Simulation | Quantum | Best if local refit/novelty criterion is central. |
| Compact Adaptive Scaffolds for Mixed Fermion--Boson Simulation | npj / Nature-family style | Short, low-jargon title. |

Keep ADAPT in the abstract/keywords for discoverability, but do not lead with it if the target venue should read the paper as scaffold design.

## Abstract components

Opener:

> Adaptive variational ansätze can reduce circuit depth, but current growth rules do not directly optimize scaffold structure under local geometric gain and hardware cost.

Novelty sentence:

> We introduce a scaffold-construction procedure that scores generator-position records using local geometric diagnostics, refit-aware gains, and compiled-resource proxies for mixed fermion--boson systems.

Results sentence template:

> Across held-out fermionic, bosonic, and mixed benchmarks, the resulting scaffolds reach comparable or lower error with fewer parameters, lower compiled depth, or both.

## Cover-letter sentence

> This manuscript introduces a geometry- and cost-aware adaptive scaffold construction procedure for mixed fermion--boson simulation, addressing a gap left by current ADAPT-style methods, which optimize ansatz growth without directly optimizing scaffold structure under local geometric and compiled-resource evidence.

## Suggested keywords

- adaptive variational quantum algorithms;
- ADAPT-VQE;
- mixed fermion--boson systems;
- electron--phonon simulation;
- hardware-efficient ansatz design;
- operator-pool methods;
- encoding-aware quantum simulation;
- compact variational scaffolds.

## Reviewer anxiety checklist

- Incrementalism: emphasize scaffold decision rule over operator-position records, not a new pool alone.
- Mixed fermion--boson specificity: make bosonic truncation and cross-sector operators part of the geometry/resource claim.
- Proxy metrics: validate proxy scores against compiled resource accounting on held-out instances.
- Baselines: include strongest relevant ADAPT descendants, especially qubit-ADAPT, TETRIS, CEO-ADAPT, and nearest pool/batching method.
