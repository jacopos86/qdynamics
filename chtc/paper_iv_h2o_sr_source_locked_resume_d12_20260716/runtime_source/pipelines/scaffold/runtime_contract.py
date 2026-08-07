"""Compatibility re-export for scaffold runtime contracts.

Canonical definitions live in :mod:`pipelines.contracts.scaffold` so other
pipeline lanes can depend on stable contract shapes without importing scaffold
implementation internals.
"""

from __future__ import annotations

from pipelines.contracts.scaffold import (
    CandidatePoolSource,
    PoolCompleteness,
    PoolSourceKind,
    ScaffoldRuntimeInput,
)


__all__ = [
    "CandidatePoolSource",
    "PoolCompleteness",
    "PoolSourceKind",
    "ScaffoldRuntimeInput",
]
