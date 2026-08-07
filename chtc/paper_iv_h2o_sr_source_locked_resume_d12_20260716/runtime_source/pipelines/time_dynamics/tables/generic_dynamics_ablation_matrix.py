#!/usr/bin/env python3
"""Compatibility shim for the generic controller ablation benchmark matrix."""

from __future__ import annotations

from typing import Any, Sequence

from pipelines.time_dynamics.benchmarks import legacy_native as _impl
from pipelines.time_dynamics.benchmarks.legacy_native import (
    GENERIC_CONTROLLER_ABLATION_MATRIX_ALGORITHM,
    GENERIC_CONTROLLER_ABLATION_MATRIX_SCHEMA,
    GenericControllerAblationVariant,
    controller_ablation_variants,
    default_controller_ablation_variants,
    get_controller_ablation_variant,
    run_generic_controller_ablation_matrix,
    run_generic_controller_ablation_row,
    validate_ablation_decision_data_flow,
    validate_ablation_variant_runtime,
)

# Historical monkeypatch target.
realtime = _impl.realtime


def _sync_compat_overrides() -> None:
    _impl.realtime = realtime


def run_generic_controller_ablation_row(*args, **kwargs):
    _sync_compat_overrides()
    return _impl.run_generic_controller_ablation_row(*args, **kwargs)


def run_generic_controller_ablation_matrix(*args, **kwargs):
    _sync_compat_overrides()
    return _impl.run_generic_controller_ablation_matrix(*args, **kwargs)


def build_parser():
    return _impl.build_parser()


def main(argv: Sequence[str] | None = None) -> int:
    return _impl.main(argv)


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)


__all__ = [
    "GENERIC_CONTROLLER_ABLATION_MATRIX_ALGORITHM",
    "GENERIC_CONTROLLER_ABLATION_MATRIX_SCHEMA",
    "GenericControllerAblationVariant",
    "controller_ablation_variants",
    "default_controller_ablation_variants",
    "get_controller_ablation_variant",
    "run_generic_controller_ablation_matrix",
    "run_generic_controller_ablation_row",
    "realtime",
    "validate_ablation_decision_data_flow",
    "validate_ablation_variant_runtime",
    "with_class_settings_lock_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
