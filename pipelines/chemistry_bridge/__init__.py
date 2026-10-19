"""Opt-in bridge from Doron Wu's ab initio exports to this repository's quantum methods.

Nothing in this package is imported by a canonical route.  Every entry point is
explicit: an export adapter (``qdynamics_export``), an array backend with
honest device reporting (``array_backend``), a matrix-free Pauli-polynomial
action on that backend (``polynomial_action``), and method runners that call
the repository's own QSE core, AP-McLachlan controller and static pool
resolvers (``methods``).  The strict molecular diagnostic intake
(``pipelines.static_adapt.builders.molecular_diagnostic_intake``) remains the
sole intake boundary; this package only prepares its inputs and consumes its
``ResolvedProblemContext``.
"""

from __future__ import annotations

BRIDGE_SCHEMA_VERSION = "doron_chemistry_quantum_bridge_v1"

__all__ = ["BRIDGE_SCHEMA_VERSION"]
