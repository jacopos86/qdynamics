#!/usr/bin/env python3
"""Probe the pinned execution image against the sealed Qiskit oracle source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import tempfile


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


def probe(source_archive: Path) -> dict[str, object]:
    temporary = tempfile.TemporaryDirectory(prefix="phase3-qiskit-image-probe.")
    try:
        source_root = Path(temporary.name) / "source"
        _extract(source_archive, source_root)
        sys.path.insert(0, source_root.as_posix())

        import qiskit
        import qiskit_ibm_runtime
        from qiskit import QuantumCircuit, transpile
        from pipelines.static_adapt.hh_backend_compile_oracle import (
            BackendCompileConfig,
            BackendCompileOracle,
            ONE_QUBIT_COORDINATE_COMPILED_POSITIVE_DELTA_V1,
        )

        config = BackendCompileConfig(
            mode="transpile_single_v1",
            requested_backend_name="FakeMarrakesh",
            requested_backend_shortlist=("FakeMarrakesh",),
            seed_transpiler=7,
            optimization_level=1,
            structure_theta_value=1.0,
            preferred_fake_backends=("FakeMarrakesh",),
            reward_negative_deltas=False,
            allow_preferred_fallback=False,
            one_qubit_coordinate_policy=(
                ONE_QUBIT_COORDINATE_COMPILED_POSITIVE_DELTA_V1
            ),
        )
        oracle = BackendCompileOracle(
            config=config,
            num_qubits=2,
            ref_state=None,
        )
        if len(oracle.targets) != 1:
            raise RuntimeError("oracle did not resolve one exact target")
        target = oracle.targets[0]
        if str(target.resolved_name) != "FakeMarrakesh":
            raise RuntimeError("oracle did not resolve FakeMarrakesh")
        if any(
            not isinstance(row, dict)
            or row.get("success") is not True
            or row.get("resolved_name") != "FakeMarrakesh"
            for row in oracle.resolution_audit
        ):
            raise RuntimeError("oracle backend resolution audit drifted")

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
            "qiskit_version": str(qiskit.__version__),
            "qiskit_ibm_runtime_version": str(qiskit_ibm_runtime.__version__),
            "resolved_backend_name": str(target.resolved_name),
            "backend_resolution_kind": str(target.resolution_kind),
            "optimization_level": int(config.optimization_level),
            "seed_transpiler": int(config.seed_transpiler),
            "structure_theta_value": float(config.structure_theta_value),
            "allow_preferred_fallback": bool(config.allow_preferred_fallback),
            "reward_negative_deltas": bool(config.reward_negative_deltas),
            "one_qubit_coordinate_policy": str(
                config.one_qubit_coordinate_policy
            ),
            "compiled_depth": int(compiled.depth()),
            "compiled_operations": {
                str(key): int(value)
                for key, value in compiled.count_ops().items()
            },
            "sealed_source_imported": True,
        }
    finally:
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
