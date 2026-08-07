from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from pipelines.static_adapt import exact_geometry_backend
from pipelines.static_adapt import exact_state_backend
from pipelines.static_adapt import geometry_fingerprints
from pipelines.static_adapt import selector_exact_query_geometry


_ROOT = Path(__file__).resolve().parents[1]
_FORMAL_MODULE = "pipelines.static_adapt.formal_manifold_warm_start"


def _imported_modules(relative_path: str) -> set[str]:
    tree = ast.parse((_ROOT / relative_path).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(str(node.module))
        elif isinstance(node, ast.Import):
            imported.update(str(alias.name) for alias in node.names)
    return imported


def test_neutral_consumers_do_not_import_formal_manifold_route() -> None:
    for path in (
        "pipelines/static_adapt/exact_geometry_backend.py",
        "pipelines/static_adapt/selector_exact_query_geometry.py",
    ):
        assert _FORMAL_MODULE not in _imported_modules(path)


def test_exact_geometry_backend_reexports_exact_state_types_with_identity() -> None:
    for name in (
        "ExactEnergyEvaluation",
        "ExactGradientEvaluation",
        "ExactStateEvaluation",
        "ExactStateBackend",
    ):
        neutral_type = getattr(exact_state_backend, name)
        assert getattr(exact_geometry_backend, name) is neutral_type

def test_selector_reexports_neutral_fingerprints_with_identity() -> None:
    for name in (
        "candidate_generator_fingerprint",
        "candidate_coordinate_fingerprint",
        "compiled_hamiltonian_fingerprint",
        "ordered_scaffold_fingerprint",
    ):
        assert getattr(selector_exact_query_geometry, name) is getattr(
            geometry_fingerprints, name
        )


def test_exact_state_backend_preserves_validation_and_horizontalization() -> None:
    raw_tangents = np.asarray([[1.0 + 1.0j], [2.0 - 1.0j]], dtype=complex)
    backend = exact_state_backend.ExactStateBackend(
        evaluate_fn=lambda _theta: exact_state_backend.ExactStateEvaluation(
            energy=-1.25,
            gradient=np.asarray([0.5], dtype=float),
            statevector=np.asarray([1.0, 0.0], dtype=complex),
            tangents=raw_tangents,
            metadata={"source": "focused_extraction_test"},
        ),
        coordinate_registry=("theta:0",),
        manifold_id="focused-neutral-backend",
        parameterization_mode="logical_shared",
    )

    evaluated = backend.evaluate(np.asarray([0.0], dtype=float))

    assert evaluated.energy == -1.25
    assert np.allclose(evaluated.gradient, [0.5])
    assert np.allclose(np.conjugate(evaluated.statevector) @ evaluated.tangents, 0.0)
    assert evaluated.metadata["horizontalization"] == "state_projector_v1"
