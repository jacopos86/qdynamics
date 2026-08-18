# Canonical Paper-I RA Runtime Identity (2026-08-18)

Date: 2026-08-18
Scope: agent-facing method identity and routing for the canonical Paper-I
static route. This document declares identity; it does not edit,
reinterpret, or promote a manuscript result.

Supersedes `paper_i_sr_snake_canonical_runtime_settings_20260716.md`
(SR-SNAKE v3.1) as the canonical declaration. That document remains the
preserved identity of the pre-unification runs it describes.

## Stable identity

The canonical Paper-I static route is:

```text
route_id = ra_singleton_always
seam = pipelines.static_adapt.ra_adapt.run_ra_adapt(problem, request=None)
candidate_representation = single_pauli_word_v1
insertion = AlwaysCommutationReducedInsertion  (executes full_commutation_reduced)
```

Authoritative constants: `pipelines/static_adapt/ra_adapt/bundles.py`
(`ROUTE_RA_SINGLETON_ALWAYS`) and
`pipelines/static_adapt/ra_adapt/contracts.py`. Executable settings
resolve from the typed protocol contracts; agents must not reconstruct
them from prose, including this file.

## Scientific rationale (user, 2026-08-18)

The paper's primary singleton result is that, when route provenance is
tracked, insertion always reduces commutation. Always-commutation-reduced
insertion is therefore the canonical route, and append-only behavior is
essentially its specialization.

## Declared design intent (stage-3 work item)

Append-ADAPT is currently a separate insertion adapter
(`AppendOnlyInsertion`). It should become a **true algorithmic
specialization** of the always-insertion route — the end-position-restricted
case of the same code path — so the comparator differs from the canonical
route by a constraint, not by an implementation. Until that lands, the
existing adapter remains the executable comparator; evidence produced by
either surface keeps its recorded identity.

## Boundary directive (user, 2026-08-18)

Algorithmic metadata (route contracts, protocol digests, accepted
trajectories, ledgers) must remain strictly separated from reporting
artifacts. Reporting consumes typed results read-only; nothing in
reporting may feed back into route identity, scoring, or controller
decisions. This constraint governs the stage-3 module split.
