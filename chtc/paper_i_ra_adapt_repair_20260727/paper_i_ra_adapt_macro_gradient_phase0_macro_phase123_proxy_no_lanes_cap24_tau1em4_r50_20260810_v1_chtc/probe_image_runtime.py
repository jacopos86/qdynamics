#!/usr/bin/env python3
"""Probe the pinned image without invoking Qiskit selector compilation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import tempfile


ALGORITHM_ID = (
    "paper_i_ra_adapt_macro_gradient_phase0_macro_phase1_phase2_phase3_"
    "proxy_plateau_no_lanes_v1"
)
CANDIDATE_ADAPTER_ID = (
    "paper_i_ra_adapt_macro_gradient_phase0_candidate_adapter_v1"
)
STRUCTURAL_PROXY_MODE = "marrakesh_graph_span_v1"


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
    temporary = tempfile.TemporaryDirectory(prefix="macro-phase0-proxy-image-probe.")
    try:
        source_root = Path(temporary.name) / "source"
        _extract(source_archive, source_root)
        sys.path.insert(0, source_root.as_posix())

        import numpy
        import scipy
        from pipelines.static_adapt.hh_backend_compile_oracle import (
            MARRAKESH_GRAPH_SPAN_MODE,
        )
        from pipelines.static_adapt.ra_adapt.adapters import (
            MACRO_GRADIENT_PHASE0_ADAPTER_ID,
            MacroGradientPhase0CandidateAdapter,
        )
        from pipelines.static_adapt.ra_adapt.engine import (
            RA_ADAPT_MACRO_GRADIENT_PHASE0_PROXY_NO_LANES_ALGORITHM_ID,
        )

        adapter = MacroGradientPhase0CandidateAdapter()
        if (
            RA_ADAPT_MACRO_GRADIENT_PHASE0_PROXY_NO_LANES_ALGORITHM_ID
            != ALGORITHM_ID
            or MACRO_GRADIENT_PHASE0_ADAPTER_ID != CANDIDATE_ADAPTER_ID
            or adapter.adapter_id != CANDIDATE_ADAPTER_ID
            or MARRAKESH_GRAPH_SPAN_MODE != STRUCTURAL_PROXY_MODE
        ):
            raise RuntimeError("sealed macro Phase-0 route identity drifted")
        return {
            "status": "passed",
            "python_version": sys.version.split()[0],
            "numpy_version": str(numpy.__version__),
            "scipy_version": str(scipy.__version__),
            "algorithm_id": ALGORITHM_ID,
            "candidate_adapter_id": CANDIDATE_ADAPTER_ID,
            "structural_proxy_mode": STRUCTURAL_PROXY_MODE,
            "selector_qiskit_compile_cost_active": False,
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
