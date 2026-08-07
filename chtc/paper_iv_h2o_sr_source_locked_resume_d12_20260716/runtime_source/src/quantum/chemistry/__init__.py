from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "RestrictedClosedShellMolecularProblem": "src.quantum.chemistry.psi4_adapter",
    "build_h2_geometry": "src.quantum.chemistry.psi4_adapter",
    "build_h2_problem_from_psi4": "src.quantum.chemistry.psi4_adapter",
    "load_restricted_closed_shell_problem_from_psi4": "src.quantum.chemistry.psi4_adapter",
    "run_local_adapt_vqe_with_pool": "src.quantum.chemistry.molecular_adapt_core",
    "run_pipeline_local_adapt_vqe_with_pool": "src.quantum.chemistry.molecular_adapt_core",
    "run_local_molecular_adapt_vqe": "src.quantum.chemistry.molecular_adapt_core",
    "build_restricted_closed_shell_molecular_hamiltonian": "src.quantum.chemistry.molecular_hamiltonian",
    "build_molecular_uccsd_pool": "src.quantum.chemistry.molecular_uccsd",
    "CachedVibronicH2Fixture": "src.quantum.chemistry.vibronic_h2",
    "build_vibronic_h2_model": "src.quantum.chemistry.vibronic_h2",
    "default_vibronic_h2_fixture_path": "src.quantum.chemistry.vibronic_h2",
    "exact_ground_energy_dense": "src.quantum.chemistry.vibronic_h2",
    "exact_ground_energy_physical_sector": "src.quantum.chemistry.vibronic_h2",
    "load_cached_vibronic_h2_fixture": "src.quantum.chemistry.vibronic_h2",
}

__all__ = tuple(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORT_MODULES[str(name)]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), str(name))
    globals()[str(name)] = value
    return value
