"""Regression coverage for the neutral exact-geometry backend extraction."""

from pipelines.static_adapt import exact_geometry_backend


def test_exact_geometry_backend_owns_public_implementation() -> None:
    assert (
        exact_geometry_backend.CompiledExactManifoldAdapter.__module__
        == "pipelines.static_adapt.exact_geometry_backend"
    )
    assert (
        exact_geometry_backend.build_compiled_exact_manifold_adapter.__module__
        == "pipelines.static_adapt.exact_geometry_backend"
    )
