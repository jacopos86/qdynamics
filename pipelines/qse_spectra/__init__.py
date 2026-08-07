"""Isolated Quantum Subspace Expansion spectra sidecar.

The package root exposes the historical public symbols lazily.  Importing
``pipelines.qse_spectra`` should not eagerly import all response/conductivity/QSE
I/O modules; those modules can be heavy and some consumers only need a submodule.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_SYMBOL_MODULES: dict[str, str] = {
    "QSEBasisElement": "pipelines.qse_spectra.core",
    "QSEBasisVectorDiagnostics": "pipelines.qse_spectra.core",
    "QSEBasisVectorPolicy": "pipelines.qse_spectra.core",
    "QSEMatrices": "pipelines.qse_spectra.core",
    "QSEObservable": "pipelines.qse_spectra.core",
    "QSEPruningConfig": "pipelines.qse_spectra.core",
    "QSEResult": "pipelines.qse_spectra.core",
    "QSETransitionObservableResult": "pipelines.qse_spectra.core",
    "normalize_statevector": "pipelines.qse_spectra.core",
    "computational_basis_state": "pipelines.qse_spectra.core",
    "pauli_string_basis_element": "pipelines.qse_spectra.core",
    "polynomial_basis_element": "pipelines.qse_spectra.core",
    "pauli_string_observable": "pipelines.qse_spectra.core",
    "polynomial_observable": "pipelines.qse_spectra.core",
    "apply_qse_observable": "pipelines.qse_spectra.core",
    "expect_qse_observable": "pipelines.qse_spectra.core",
    "build_qse_matrices": "pipelines.qse_spectra.core",
    "solve_qse_generalized_eigenproblem": "pipelines.qse_spectra.core",
    "compute_transition_observables": "pipelines.qse_spectra.core",
    "compute_qse_spectra": "pipelines.qse_spectra.core",
    "statevector_from_manifest": "pipelines.qse_spectra.io",
    "load_state_json": "pipelines.qse_spectra.io",
    "polynomial_from_serialized_terms": "pipelines.qse_spectra.io",
    "load_polynomial_json": "pipelines.qse_spectra.io",
    "basis_elements_from_artifact_source": "pipelines.qse_spectra.io",
    "basis_elements_from_labels": "pipelines.qse_spectra.io",
    "load_operator_basis_json": "pipelines.qse_spectra.io",
    "transition_observables_from_labels": "pipelines.qse_spectra.io",
    "load_transition_observables_json": "pipelines.qse_spectra.io",
    "qse_result_to_manifest": "pipelines.qse_spectra.io",
    "write_manifest_json": "pipelines.qse_spectra.io",
    "CONDUCTIVITY_RESPONSE_SCHEMA_VERSION": "pipelines.qse_spectra.conductivity",
    "ConductivityChannel": "pipelines.qse_spectra.conductivity",
    "build_conductivity_response_payload": "pipelines.qse_spectra.conductivity",
    "parse_conductivity_channel_spec": "pipelines.qse_spectra.conductivity",
    "parse_conductivity_channel_specs": "pipelines.qse_spectra.conductivity",
    "GREEN_FUNCTION_SCHEMA_VERSION": "pipelines.qse_spectra.green_functions",
    "GreenFunctionMode": "pipelines.qse_spectra.green_functions",
    "GreenFunctionSourceState": "pipelines.qse_spectra.green_functions",
    "build_green_function_payload": "pipelines.qse_spectra.green_functions",
    "jw_ladder_source_state": "pipelines.qse_spectra.green_functions",
    "parse_green_function_mode_spec": "pipelines.qse_spectra.green_functions",
    "parse_green_function_mode_specs": "pipelines.qse_spectra.green_functions",
    "HH_CURRENT_CONTACT_POLICY": "pipelines.qse_spectra.hh_current_observables",
    "HH_CURRENT_EDGE_ORIENTATIONS": "pipelines.qse_spectra.hh_current_observables",
    "HH_CURRENT_OBSERVABLES_SCHEMA_VERSION": "pipelines.qse_spectra.hh_current_observables",
    "HH_CURRENT_PEIERLS_POLICY": "pipelines.qse_spectra.hh_current_observables",
    "HHCurrentHoppingResolution": "pipelines.qse_spectra.hh_current_observables",
    "HHCurrentObservableBundle": "pipelines.qse_spectra.hh_current_observables",
    "HHCurrentObservableError": "pipelines.qse_spectra.hh_current_observables",
    "build_hh_current_observable_bundle": "pipelines.qse_spectra.hh_current_observables",
    "directed_hh_current_edges": "pipelines.qse_spectra.hh_current_observables",
    "resolve_hh_current_hopping_from_sources": "pipelines.qse_spectra.hh_current_observables",
    "spin_resolved_hh_edge_contact_operator": "pipelines.qse_spectra.hh_current_observables",
    "spin_resolved_hh_edge_current_operator": "pipelines.qse_spectra.hh_current_observables",
    "total_hh_charge_contact_operator": "pipelines.qse_spectra.hh_current_observables",
    "total_hh_charge_current_operator": "pipelines.qse_spectra.hh_current_observables",
    "HHFormFactor": "pipelines.qse_spectra.hh_response_observables",
    "HHResponseLayout": "pipelines.qse_spectra.hh_response_observables",
    "HHResponseObservableBundle": "pipelines.qse_spectra.hh_response_observables",
    "HHResponseObservableError": "pipelines.qse_spectra.hh_response_observables",
    "build_hh_neutral_response_observable_bundle": "pipelines.qse_spectra.hh_response_observables",
    "density_baseline_from_state": "pipelines.qse_spectra.hh_response_observables",
    "mixed_density_displacement_operator": "pipelines.qse_spectra.hh_response_observables",
    "normalize_hh_neutral_response_channels": "pipelines.qse_spectra.hh_response_observables",
    "parse_hh_form_factor": "pipelines.qse_spectra.hh_response_observables",
    "phonon_displacement_operator": "pipelines.qse_spectra.hh_response_observables",
    "phonon_momentum_operator": "pipelines.qse_spectra.hh_response_observables",
    "resolve_hh_response_layout_from_sources": "pipelines.qse_spectra.hh_response_observables",
    "site_density_operator": "pipelines.qse_spectra.hh_response_observables",
    "weighted_density_fluctuation_operator": "pipelines.qse_spectra.hh_response_observables",
    "weighted_phonon_displacement_operator": "pipelines.qse_spectra.hh_response_observables",
    "weighted_phonon_momentum_operator": "pipelines.qse_spectra.hh_response_observables",
    "StaticRecordSelectionConfig": "pipelines.qse_spectra.record_selection",
    "StaticRecordCandidate": "pipelines.qse_spectra.record_selection",
    "StaticRecordSelectionResult": "pipelines.qse_spectra.record_selection",
    "select_static_qse_records": "pipelines.qse_spectra.record_selection",
    "finalize_static_record_selection_payload": "pipelines.qse_spectra.record_selection",
    "RESPONSE_FUNCTIONS_SCHEMA_VERSION": "pipelines.qse_spectra.response_functions",
    "ResponseChannel": "pipelines.qse_spectra.response_functions",
    "ResponseTimeGrid": "pipelines.qse_spectra.response_functions",
    "build_response_functions_payload": "pipelines.qse_spectra.response_functions",
    "parse_response_channel_spec": "pipelines.qse_spectra.response_functions",
    "parse_response_channel_specs": "pipelines.qse_spectra.response_functions",
    "SpectralGrid": "pipelines.qse_spectra.spectral_functions",
    "BroadeningKernelConfig": "pipelines.qse_spectra.spectral_functions",
    "SpectralWindow": "pipelines.qse_spectra.spectral_functions",
    "SpectralReference": "pipelines.qse_spectra.spectral_functions",
    "CutoffBoundaryLayout": "pipelines.qse_spectra.spectral_functions",
    "lorentzian_kernel": "pipelines.qse_spectra.spectral_functions",
    "gaussian_kernel": "pipelines.qse_spectra.spectral_functions",
    "evaluate_broadening_kernel": "pipelines.qse_spectra.spectral_functions",
    "parse_spectral_window_spec": "pipelines.qse_spectra.spectral_functions",
    "load_spectral_references_json": "pipelines.qse_spectra.spectral_functions",
    "build_spectral_function_payload": "pipelines.qse_spectra.spectral_functions",
    "build_spectral_window_metrics_payload": "pipelines.qse_spectra.spectral_functions",
    "build_spectral_postprocessing_payloads": "pipelines.qse_spectra.spectral_functions",
    "build_cutoff_boundary_diagnostics": "pipelines.qse_spectra.spectral_functions",
}

__all__ = list(_SYMBOL_MODULES)


def __getattr__(name: str) -> Any:
    try:
        module_name = _SYMBOL_MODULES[str(name)]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    value = getattr(module, str(name))
    globals()[str(name)] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
