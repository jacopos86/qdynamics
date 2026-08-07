"""HH adapter boundary for the generic realtime entrypoint.

The generic realtime façade uses this module as its only direct dependency on
HH-specific controller/parser/audit implementation.  HH implementation modules
are imported lazily so importing the generic façade remains lightweight.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any


_HH_CONTROLLER_MODULE = "pipelines.time_dynamics.legacy.checkpoint_controller"
_HH_COMPILE_AUDIT_MODULE = "pipelines.time_dynamics.legacy.checkpoint_compile_audit"
_HH_EXACT_AUDIT_MODULE = "pipelines.time_dynamics.legacy.checkpoint_exact_audit"
_HH_ENTRYPOINT_MODULE = "pipelines.time_dynamics.runners.hh_from_adapt_artifact"


def _module(name: str) -> Any:
    return importlib.import_module(str(name))


def _controller_module() -> Any:
    return _module(_HH_CONTROLLER_MODULE)


def _compile_audit_module() -> Any:
    return _module(_HH_COMPILE_AUDIT_MODULE)


def _exact_audit_module() -> Any:
    return _module(_HH_EXACT_AUDIT_MODULE)


def _entrypoint_module() -> Any:
    return _module(_HH_ENTRYPOINT_MODULE)


@dataclass(frozen=True)
class HHRealtimeAdapter:
    """Small explicit adapter over HH-specific realtime implementation."""

    def build_parser(self) -> Any:
        return _entrypoint_module().build_parser()

    def build_controller_config(self, args: Any) -> Any:
        return _entrypoint_module().build_controller_config(args)

    def build_drive_config(self, args: Any, *, n_sites: int, ordering: str) -> Any:
        return _entrypoint_module().build_drive_config(
            args,
            n_sites=int(n_sites),
            ordering=str(ordering),
        )

    def build_oracle_config(self, args: Any) -> Any:
        return _entrypoint_module().build_oracle_config(args)

    def create_controller(self, **kwargs: Any) -> Any:
        return _controller_module().RealtimeCheckpointController(**kwargs)

    def build_exact_audit_helper_for_controller(
        self,
        controller: Any,
        *,
        exact_reference_cache: dict[str, object] | None = None,
    ) -> Any:
        return _exact_audit_module().build_exact_audit_helper_for_controller(
            controller,
            exact_reference_cache=exact_reference_cache,
        )

    def run_controller_with_exact_audit(self, controller: Any, exact_helper: Any, **kwargs: Any) -> Any:
        return _exact_audit_module().run_controller_with_exact_audit(controller, exact_helper, **kwargs)

    def attach_diagnostic_exact_reference(self, *, args: Any, controller: Any, result: Any) -> Any:
        return _entrypoint_module()._attach_diagnostic_exact_reference(
            args=args,
            controller=controller,
            result=result,
        )

    def build_compile_audit_config_from_args(self, args: Any) -> Any:
        return _compile_audit_module().build_compile_audit_config_from_args(args)

    def run_final_scaffold_compile_audit(self, *, controller: Any, config: Any) -> Any:
        return _compile_audit_module().run_final_scaffold_compile_audit(
            controller=controller,
            config=config,
        )

    def run_prune_event_compile_audit(self, *, controller: Any, config: Any) -> Any:
        return _compile_audit_module().run_prune_event_compile_audit(
            controller=controller,
            config=config,
        )

    def build_output_payload(self, **kwargs: Any) -> dict[str, Any]:
        return _entrypoint_module().build_output_payload(**kwargs)


HH_REALTIME_ADAPTER = HHRealtimeAdapter()


__all__ = ["HHRealtimeAdapter", "HH_REALTIME_ADAPTER"]
