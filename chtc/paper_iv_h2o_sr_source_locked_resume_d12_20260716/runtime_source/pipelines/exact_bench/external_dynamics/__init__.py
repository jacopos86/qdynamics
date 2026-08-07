"""Diagnostic-only external dynamics parity helpers.

This package is intentionally outside the production dynamics benchmark path.
It may inspect or execute optional local reference checkouts, but importing it
must never import third-party reference packages or paper-facing run artifacts.
"""

from __future__ import annotations

from pipelines.exact_bench.external_dynamics.adapter import (
    EXTERNAL_DYNAMICS_PARITY_SCHEMA,
    run_dynamics_parity_checks,
)
from pipelines.exact_bench.external_dynamics.provenance import (
    ExternalDynamicsReferenceSpec,
    external_dynamics_reference_catalog,
    get_external_dynamics_reference_spec,
)

__all__ = [
    "EXTERNAL_DYNAMICS_PARITY_SCHEMA",
    "ExternalDynamicsReferenceSpec",
    "external_dynamics_reference_catalog",
    "get_external_dynamics_reference_spec",
    "run_dynamics_parity_checks",
]
