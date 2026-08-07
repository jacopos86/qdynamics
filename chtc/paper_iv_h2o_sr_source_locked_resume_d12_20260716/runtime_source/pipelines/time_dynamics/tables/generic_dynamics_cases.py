#!/usr/bin/env python3
"""Explicit generic time-dynamics benchmark case catalog.

The catalog intentionally points at existing small realtime fixtures.  Table
class is stored per case rather than inferred from the family key so downstream
reporting cannot silently reclassify manuscript rows.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.time_dynamics.tables.dynamics_benchmark_contract import DynamicsBenchmarkCase


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _fixture(name: str) -> str:
    return str(_repo_root() / "test_support" / "fixtures" / name)


_BUILTIN_GENERIC_CASES: tuple[DynamicsBenchmarkCase, ...] = (
    DynamicsBenchmarkCase(
        case_id="hubbard_dynamics_default",
        family="hubbard",
        table_class="fermionic_lattice",
        tuning_class="fermionic",
        artifact_json=_fixture("hubbard_realtime_seed.json"),
        description="L=2 Hubbard realtime smoke fixture",
    ),
    DynamicsBenchmarkCase(
        case_id="ionic_hubbard_dynamics_default",
        family="ionic_hubbard",
        table_class="fermionic_lattice",
        tuning_class="fermionic",
        artifact_json=_fixture("ionic_hubbard_realtime_seed.json"),
        description="L=2 ionic Hubbard realtime smoke fixture",
    ),
    DynamicsBenchmarkCase(
        case_id="extended_hubbard_dynamics_default",
        family="extended_hubbard",
        table_class="fermionic_lattice",
        tuning_class="fermionic",
        artifact_json=_fixture("extended_hubbard_realtime_seed.json"),
        description="L=2 extended Hubbard realtime smoke fixture",
    ),
    DynamicsBenchmarkCase(
        case_id="ttprime_hubbard_dynamics_default",
        family="ttprime_hubbard",
        table_class="fermionic_lattice",
        tuning_class="fermionic",
        artifact_json=_fixture("ttprime_hubbard_realtime_seed.json"),
        description="L=3 t-tprime Hubbard realtime smoke fixture",
    ),
    DynamicsBenchmarkCase(
        case_id="spinless_tv_dynamics_default",
        family="spinless_tv",
        table_class="spinless_fermion_lattice",
        tuning_class="fermionic",
        artifact_json=_fixture("spinless_tv_realtime_seed.json"),
        description="Spinless t-V realtime smoke fixture",
    ),
    DynamicsBenchmarkCase(
        case_id="spin_boson_dynamics_default",
        family="spin_boson",
        table_class="spin_boson",
        tuning_class="hybrid",
        artifact_json=_fixture("spin_boson_realtime_seed.json"),
        description="One-emitter spin-boson realtime smoke fixture",
    ),
    DynamicsBenchmarkCase(
        case_id="bose_hubbard_dynamics_default",
        family="bose_hubbard",
        table_class="boson_chain",
        tuning_class="bosonic",
        artifact_json=_fixture("bose_hubbard_realtime_seed.json"),
        description="L=2 Bose-Hubbard realtime smoke fixture",
    ),
    DynamicsBenchmarkCase(
        case_id="harmonic_kerr_chain_dynamics_default",
        family="harmonic_kerr_chain",
        table_class="boson_chain",
        tuning_class="bosonic",
        artifact_json=_fixture("harmonic_kerr_chain_realtime_seed.json"),
        description="L=2 harmonic/Kerr chain realtime smoke fixture",
        metadata={
            "aggregate_role": "stress_nonconverged_seed",
            "exclude_from_main_dynamics_aggregate": True,
            "aggregate_exclusion_reason": (
                "Current fixture is preserved as a stress/nonconverged-seed row; "
                "exclude it from main class medians/means until the seed converges."
            ),
        },
    ),
)

GENERIC_DYNAMICS_IMPLEMENTED_FAMILIES: tuple[str, ...] = tuple(
    case.family for case in _BUILTIN_GENERIC_CASES
)


def builtin_generic_dynamics_cases() -> tuple[DynamicsBenchmarkCase, ...]:
    return tuple(_BUILTIN_GENERIC_CASES)


def _read_case_manifest(path: Path) -> list[Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        raw_cases = payload.get("cases", [])
    else:
        raw_cases = payload
    if not isinstance(raw_cases, list):
        raise ValueError(f"generic dynamics case manifest {path} must contain a cases list")
    return [dict(item) for item in raw_cases if isinstance(item, Mapping)]


def load_generic_dynamics_cases(
    case_manifest: str | Path | None = None,
    *,
    families: Sequence[str] | None = None,
) -> tuple[DynamicsBenchmarkCase, ...]:
    if case_manifest is None:
        cases = list(builtin_generic_dynamics_cases())
    else:
        manifest_path = Path(case_manifest).expanduser().resolve()
        cases = []
        for raw in _read_case_manifest(manifest_path):
            case = DynamicsBenchmarkCase.from_mapping(raw)
            artifact = Path(case.artifact_json).expanduser()
            if not artifact.is_absolute():
                artifact = manifest_path.parent / artifact
            cases.append(replace(case, artifact_json=str(artifact.resolve())))
    if families is not None:
        wanted = {str(family) for family in families}
        cases = [case for case in cases if case.family in wanted]
    return tuple(cases)


def cases_for_family(
    family: str,
    case_manifest: str | Path | None = None,
) -> tuple[DynamicsBenchmarkCase, ...]:
    return tuple(
        case
        for case in load_generic_dynamics_cases(case_manifest, families=(str(family),))
        if case.family == str(family)
    )


def get_generic_dynamics_case(
    case_id: str,
    *,
    family: str | None = None,
    case_manifest: str | Path | None = None,
) -> DynamicsBenchmarkCase:
    cases = load_generic_dynamics_cases(
        case_manifest,
        families=None if family is None else (str(family),),
    )
    for case in cases:
        if case.case_id == str(case_id) and (family is None or case.family == str(family)):
            return case
    known = ", ".join(case.case_id for case in cases)
    suffix = f" for family {family!r}" if family is not None else ""
    raise ValueError(f"unknown generic dynamics case_id={case_id!r}{suffix}; known cases: {known}")


__all__ = [
    "GENERIC_DYNAMICS_IMPLEMENTED_FAMILIES",
    "builtin_generic_dynamics_cases",
    "cases_for_family",
    "get_generic_dynamics_case",
    "load_generic_dynamics_cases",
]
