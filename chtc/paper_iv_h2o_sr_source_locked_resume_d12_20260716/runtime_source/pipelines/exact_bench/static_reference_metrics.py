"""Shared exact/reference-cutoff energy helpers for static Paper-I benchmark rows."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.static_adapt.builders.problem_setup import (
    _exact_gs_energy_for_problem,
    build_problem_hamiltonian,
)
from src.quantum.vqe_latex_python_pairs import half_filled_num_particles


def _pipeline_arg_value(args: Sequence[str], name: str) -> str | None:
    tokens = tuple(str(x) for x in args)
    for idx, token in enumerate(tokens[:-1]):
        if token == name:
            return tokens[idx + 1]
    return None


def _exact_sector_identity(problem: str, L: int) -> tuple[tuple[int, int] | None, str | None, str]:
    problem_key = str(problem).strip().lower()
    if problem_key == "spin_boson":
        return None, None, "single_emitter_truncated_boson_register"
    if problem_key in {"bose_hubbard", "harmonic_kerr_chain"}:
        return None, None, "unrestricted_truncated_boson_register"
    particles = tuple(int(value) for value in half_filled_num_particles(int(L)))
    return (
        particles,
        "src.quantum.vqe_latex_python_pairs.half_filled_num_particles",
        "canonical_half_filled_fermionic_sector",
    )


def _spec_values(spec: Any, *, n_ph_max: int | None = None) -> dict[str, Any]:
    args = tuple(str(x) for x in getattr(spec, "base_pipeline_args", ()))
    features = getattr(spec, "features", None)
    problem = str(_pipeline_arg_value(args, "--problem") or getattr(spec, "family", ""))
    L = int(_pipeline_arg_value(args, "--L") or getattr(features, "L", 1))
    work_nph = _pipeline_arg_value(args, "--n-ph-max")
    num_particles, num_particles_source, exact_sector_policy = _exact_sector_identity(problem, L)
    return {
        "problem": problem,
        "L": L,
        "num_particles": num_particles,
        "num_particles_source": num_particles_source,
        "exact_sector_policy": exact_sector_policy,
        "t": float(_pipeline_arg_value(args, "--t") or 1.0),
        "u": float(_pipeline_arg_value(args, "--u") or 0.0),
        "dv": float(_pipeline_arg_value(args, "--dv") or 0.0),
        "omega0": float(_pipeline_arg_value(args, "--omega0") or 1.0),
        "g_ep": float(_pipeline_arg_value(args, "--g-ep") or 0.0),
        "v_nn": float(_pipeline_arg_value(args, "--v-nn") or 0.0),
        "t_prime": float(_pipeline_arg_value(args, "--t-prime") or 0.0),
        "n_ph_max": int(n_ph_max if n_ph_max is not None else (work_nph or 1)),
        "boson_encoding": str(_pipeline_arg_value(args, "--boson-encoding") or "binary"),
        "ordering": str(_pipeline_arg_value(args, "--ordering") or "blocked"),
        "boundary": str(_pipeline_arg_value(args, "--boundary") or "open"),
        "include_zero_point": True,
    }


def reference_energy_key(spec: Any, *, n_ph_max: int) -> dict[str, Any]:
    values = _spec_values(spec, n_ph_max=int(n_ph_max))
    return {
        key: values[key]
        for key in (
            "problem",
            "L",
            "num_particles",
            "num_particles_source",
            "exact_sector_policy",
            "t",
            "u",
            "dv",
            "omega0",
            "g_ep",
            "v_nn",
            "t_prime",
            "n_ph_max",
            "boson_encoding",
            "ordering",
            "boundary",
            "include_zero_point",
        )
    }


def reference_energy_key_hash(key: Mapping[str, Any]) -> str:
    blob = json.dumps(dict(key), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


@lru_cache(maxsize=256)
def _exact_energy_cached(
    problem: str,
    L: int,
    num_particles: tuple[int, int] | None,
    exact_sector_policy: str,
    t: float,
    u: float,
    dv: float,
    omega0: float,
    g_ep: float,
    v_nn: float,
    t_prime: float,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    include_zero_point: bool,
) -> float:
    del exact_sector_policy  # Part of the cache identity; dispatch below is keyed by ``problem``.
    solver_num_particles = (
        tuple(int(value) for value in num_particles)
        if num_particles is not None
        else tuple(int(value) for value in half_filled_num_particles(int(L)))
    )
    h_poly = build_problem_hamiltonian(
        problem_key=str(problem),
        num_sites=int(L),
        t=float(t),
        u=float(u),
        dv=float(dv),
        omega0=float(omega0),
        g_ep=float(g_ep),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        ordering=str(ordering),
        boundary=str(boundary),
        include_zero_point=bool(include_zero_point),
        v_nn=float(v_nn),
        t_prime=float(t_prime),
    )
    return float(
        _exact_gs_energy_for_problem(
            h_poly,
            problem=str(problem),
            num_sites=int(L),
            num_particles=solver_num_particles,
            indexing=str(ordering),
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            t=float(t),
            u=float(u),
            dv=float(dv),
            v_nn=float(v_nn),
            t_prime=float(t_prime),
            omega0=float(omega0),
            g_ep=float(g_ep),
            boundary=str(boundary),
            include_zero_point=bool(include_zero_point),
        )
    )


def exact_energy_for_spec(spec: Any, *, n_ph_max: int) -> tuple[float, str, dict[str, Any]]:
    key = reference_energy_key(spec, n_ph_max=int(n_ph_max))
    key_hash = reference_energy_key_hash(key)
    energy = _exact_energy_cached(
        str(key["problem"]),
        int(key["L"]),
        (
            None
            if key["num_particles"] is None
            else tuple(int(value) for value in key["num_particles"])
        ),
        str(key["exact_sector_policy"]),
        float(key["t"]),
        float(key["u"]),
        float(key["dv"]),
        float(key["omega0"]),
        float(key["g_ep"]),
        float(key["v_nn"]),
        float(key["t_prime"]),
        int(key["n_ph_max"]),
        str(key["boson_encoding"]),
        str(key["ordering"]),
        str(key["boundary"]),
        bool(key["include_zero_point"]),
    )
    return float(energy), key_hash, key


def materialize_reference_energy_cache(
    specs: Sequence[Any],
    *,
    output_json: Path,
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for spec in specs:
        features = getattr(spec, "features", None)
        if not bool(getattr(features, "bosonic", False)):
            continue
        args = tuple(str(x) for x in getattr(spec, "base_pipeline_args", ()))
        work_raw = _pipeline_arg_value(args, "--n-ph-max")
        ref_raw = getattr(spec, "exact_reference_n_ph_max", None)
        for label, raw in (("same_cutoff", work_raw), ("reference_cutoff", ref_raw)):
            if raw in {None, ""}:
                continue
            energy, key_hash, key = exact_energy_for_spec(spec, n_ph_max=int(raw))
            records[key_hash] = {
                "schema": "static_reference_energy_record_v1",
                "label": label,
                "key_hash": key_hash,
                "key": key,
                "exact_energy": float(energy),
                "source": "pipelines.exact_bench.static_reference_metrics",
                "status": "ok",
            }
    payload = {
        "schema": "static_reference_energy_cache_v1",
        "record_count": len(records),
        "records": records,
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
