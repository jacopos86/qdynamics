#!/usr/bin/env python3
"""Runtime fixture override helper for molecular_vibronic_h2 exact-bench rows."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

MOLECULAR_VIBRONIC_H2_FIXTURE_JSON_ENV = "GENERIC_STATIC_TABLE_MOLECULAR_VIBRONIC_H2_FIXTURE_JSON"
MOLECULAR_VIBRONIC_H2_FIXTURE_JSON_FLAG = "--molecular-vibronic-h2-fixture-json"


def molecular_vibronic_h2_fixture_json_from_env() -> str | None:
    """Return the optional H2 runtime fixture override path from the benchmark env."""
    raw = os.environ.get(MOLECULAR_VIBRONIC_H2_FIXTURE_JSON_ENV, "")
    text = str(raw).strip()
    return text or None


def _set_cli_option(args: Sequence[str], flag: str, value: object) -> tuple[str, ...]:
    out: list[str] = []
    idx = 0
    while idx < len(args):
        token = str(args[idx])
        if token == str(flag):
            idx += 2
            continue
        out.append(token)
        idx += 1
    out.extend([str(flag), str(value)])
    return tuple(out)


def with_molecular_vibronic_h2_fixture_override(
    spec: Any,
    *,
    family: str,
    fixture_json: str | Path | None = None,
) -> Any:
    """Append the H2 fixture override to a spec's base args when requested.

    This is benchmark-runner plumbing only. It lets exact-bench comparators use
    the same runtime-compatible H2 fixture as direct static_adapt/SNAKE runs
    without changing the canonical Table-I case definitions.
    """
    if str(family).strip() != "molecular_vibronic_h2":
        return spec
    raw_path = fixture_json if fixture_json not in {None, ""} else molecular_vibronic_h2_fixture_json_from_env()
    if raw_path in {None, ""}:
        return spec
    args = tuple(str(x) for x in getattr(spec, "base_pipeline_args", ()))
    updated_args = _set_cli_option(
        args,
        MOLECULAR_VIBRONIC_H2_FIXTURE_JSON_FLAG,
        str(Path(str(raw_path))),
    )
    if updated_args == args:
        return spec
    return replace(spec, base_pipeline_args=updated_args)
