"""Benchmark-local external ADAPT competitor reference support.

This package contains provenance, fetch, and adapter scaffolding for public or
request-only competitor implementations.  It is intentionally isolated under
``pipelines.exact_bench`` so external code never becomes a production
``static_adapt`` dependency.
"""

from pipelines.exact_bench.external_adapt.provenance import (
    EXTERNAL_ADAPT_ALGORITHM_IDS,
    ExternalReferenceSpec,
    external_algorithm_manifest_metadata,
    get_external_reference_spec,
    reference_catalog,
    reference_specs_for_algorithm,
)

__all__ = [
    "EXTERNAL_ADAPT_ALGORITHM_IDS",
    "ExternalReferenceSpec",
    "external_algorithm_manifest_metadata",
    "get_external_reference_spec",
    "reference_catalog",
    "reference_specs_for_algorithm",
]
