#!/usr/bin/env python3
"""Probe sealed Page-16 source isolation and exact FakeMarrakesh support."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
import tempfile


ALGORITHM_ID = (
    "paper_i_ra_adapt_macro_gradient_phase0_macro_phase1_phase2_phase3_"
    "qiskit_phase2_phase3_plateau_no_lanes_v1"
)
CANDIDATE_ADAPTER_ID = (
    "paper_i_ra_adapt_macro_gradient_phase0_candidate_adapter_v1"
)
STRUCTURAL_PROXY_MODE = "marrakesh_graph_span_v1"
BACKEND_COMPILE_SCOPE = (
    "phase_i_proxy_phase_ii_phase_iii_qiskit_transpile_v1"
)
SOURCE_ACTIVATION_POLICY = (
    "chdir_source_then_purge_ambient_modules_and_paths_before_sealed_import_v2"
)
SEALED_MODULE_NAMES = (
    "pipelines.scaffold.hh_continuation_scoring",
    "pipelines.static_adapt.hh_backend_compile_oracle",
    "pipelines.static_adapt.ra_adapt.adapters",
    "pipelines.static_adapt.ra_adapt.engine",
)


def _safe_path(value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise RuntimeError(f"unsafe archive member: {value!r}")
    return Path(*pure.parts)


def _extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    observed: set[str] = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            relative = _safe_path(member.name).as_posix()
            if (
                relative in observed
                or not member.isfile()
                or member.issym()
                or member.islnk()
            ):
                raise RuntimeError(f"unsafe archive member: {relative}")
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"unreadable archive member: {relative}")
            target = destination / Path(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as output:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(block)
            observed.add(relative)


def _activate_source_root(source_root: Path) -> None:
    root = source_root.resolve()
    for name in list(sys.modules):
        if (
            name == "pipelines"
            or name.startswith("pipelines.")
            or name == "src"
            or name.startswith("src.")
        ):
            del sys.modules[name]
    sys.path[:] = [
        item
        for item in sys.path
        if not (
            (Path(item or ".").resolve() / "pipelines").exists()
            or (Path(item or ".").resolve() / "src").exists()
        )
    ]
    sys.path.insert(0, root.as_posix())
    importlib.invalidate_caches()


def _sealed_module_paths(source_root: Path) -> dict[str, object]:
    root = source_root.resolve()
    paths: dict[str, object] = {}
    for name, module in sorted(sys.modules.items()):
        if not (
            name == "pipelines"
            or name.startswith("pipelines.")
            or name == "src"
            or name.startswith("src.")
        ):
            continue
        if module is None:
            raise RuntimeError(f"loaded source module is unavailable: {name}")
        module_file = getattr(module, "__file__", None)
        if isinstance(module_file, str):
            try:
                relative = Path(module_file).resolve().relative_to(root)
            except ValueError as exc:
                raise RuntimeError(
                    f"sealed module escaped source root: {name}"
                ) from exc
            paths[name] = relative.as_posix()
            continue
        module_search_path = getattr(module, "__path__", None)
        if module_search_path is None:
            raise RuntimeError(f"sealed module has no source path: {name}")
        relative_paths: list[str] = []
        for entry in module_search_path:
            try:
                relative = Path(entry).resolve().relative_to(root)
            except ValueError as exc:
                raise RuntimeError(
                    f"sealed namespace escaped source root: {name}"
                ) from exc
            relative_paths.append(relative.as_posix())
        if not relative_paths:
            raise RuntimeError(f"sealed namespace has no source path: {name}")
        paths[name] = relative_paths
    return paths


def probe(source_archive: Path) -> dict[str, object]:
    temporary = tempfile.TemporaryDirectory(prefix="page16-macro-qiskit-image-probe.")
    original_cwd = Path.cwd()
    try:
        source_root = Path(temporary.name) / "source"
        _extract(source_archive, source_root)
        os.chdir(source_root)
        _activate_source_root(source_root)

        import numpy
        import qiskit
        import qiskit_ibm_runtime
        import scipy
        from qiskit import QuantumCircuit, transpile
        modules = {
            name: importlib.import_module(name) for name in SEALED_MODULE_NAMES
        }
        continuation = modules[
            "pipelines.scaffold.hh_continuation_scoring"
        ]
        oracle = modules[
            "pipelines.static_adapt.hh_backend_compile_oracle"
        ]
        adapters = modules[
            "pipelines.static_adapt.ra_adapt.adapters"
        ]
        engine = modules["pipelines.static_adapt.ra_adapt.engine"]
        module_paths = _sealed_module_paths(source_root)

        adapter = adapters.MacroGradientPhase0CandidateAdapter()
        if (
            not hasattr(
                continuation,
                "BATCH_ADDITIVITY_HARD_GATE_LEGACY_V1",
            )
            or engine.RA_ADAPT_MACRO_GRADIENT_PHASE0_MACRO_PHASE23_QISKIT_ALGORITHM_ID
            != ALGORITHM_ID
            or adapters.MACRO_GRADIENT_PHASE0_ADAPTER_ID
            != CANDIDATE_ADAPTER_ID
            or adapter.adapter_id != CANDIDATE_ADAPTER_ID
            or oracle.MARRAKESH_GRAPH_SPAN_MODE != STRUCTURAL_PROXY_MODE
        ):
            raise RuntimeError("sealed Page-16 macro route identity drifted")

        config = oracle.BackendCompileConfig(
            mode="transpile_single_v1",
            requested_backend_name="FakeMarrakesh",
            seed_transpiler=7,
            optimization_level=1,
            structure_theta_value=1.0,
            reward_negative_deltas=True,
            allow_preferred_fallback=False,
            one_qubit_coordinate_policy=(
                oracle.ONE_QUBIT_COORDINATE_COMPILED_POSITIVE_DELTA_V1
            ),
        )
        compile_oracle = oracle.BackendCompileOracle(
            config=config,
            num_qubits=2,
            ref_state=None,
        )
        if len(compile_oracle.targets) != 1:
            raise RuntimeError("oracle did not resolve one exact target")
        target = compile_oracle.targets[0]
        if (
            str(target.resolved_name) != "FakeMarrakesh"
            or str(target.resolution_kind) != "fake_exact"
            or not bool(target.using_fake_backend)
        ):
            raise RuntimeError("oracle did not resolve exact FakeMarrakesh")
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        compiled = transpile(
            circuit,
            backend=target.backend_obj,
            optimization_level=config.optimization_level,
            seed_transpiler=config.seed_transpiler,
        )
        if compiled.num_qubits < 2 or compiled.depth() <= 0:
            raise RuntimeError("FakeMarrakesh transpilation returned no circuit")
        return {
            "status": "passed",
            "python_version": sys.version.split()[0],
            "numpy_version": str(numpy.__version__),
            "qiskit_version": str(qiskit.__version__),
            "qiskit_ibm_runtime_version": str(qiskit_ibm_runtime.__version__),
            "scipy_version": str(scipy.__version__),
            "algorithm_id": ALGORITHM_ID,
            "candidate_adapter_id": CANDIDATE_ADAPTER_ID,
            "structural_proxy_mode": STRUCTURAL_PROXY_MODE,
            "selector_qiskit_compile_cost_active": True,
            "selector_compile_cost_scope": BACKEND_COMPILE_SCOPE,
            "resolved_backend_name": str(target.resolved_name),
            "backend_resolution_kind": str(target.resolution_kind),
            "optimization_level": int(config.optimization_level),
            "seed_transpiler": int(config.seed_transpiler),
            "structure_theta_value": float(config.structure_theta_value),
            "allow_preferred_fallback": bool(config.allow_preferred_fallback),
            "negative_delta_reward_enabled": bool(
                config.reward_negative_deltas
            ),
            "compiled_depth": int(compiled.depth()),
            "compiled_operations": {
                str(key): int(value)
                for key, value in compiled.count_ops().items()
            },
            "source_activation_policy": SOURCE_ACTIVATION_POLICY,
            "source_cwd_isolated": Path.cwd().resolve() == source_root.resolve(),
            "sealed_module_paths": module_paths,
            "sealed_module_paths_verified": True,
            "loaded_source_module_count": len(module_paths),
            "sealed_source_imported": True,
        }
    finally:
        os.chdir(original_cwd)
        temporary.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-archive", type=Path, required=True)
    args = parser.parse_args()
    payload = probe(args.source_archive.resolve())
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
