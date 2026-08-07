"""Shared defaults and gate-list helpers for expectation-oracle noise modes."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

SYNTHETIC_DEPOLARIZING_1Q_GATES_DEFAULT: tuple[str, ...] = ("x", "sx", "rx", "ry", "h")
SYNTHETIC_DEPOLARIZING_2Q_GATES_DEFAULT: tuple[str, ...] = ("cx", "cz", "ecr")

SYNTHETIC_COHERENT_1Q_GATES_DEFAULT: tuple[str, ...] = ("x", "sx", "rx", "ry", "h")
SYNTHETIC_COHERENT_2Q_GATES_DEFAULT: tuple[str, ...] = ("cx", "cz", "ecr")
SYNTHETIC_COHERENT_GENERATOR_MODE_DEFAULT = "random_pauli_frozen_v1"

_GATE_LIST_SEPARATOR_RE = re.compile(r"[\s,;]+")
_GATE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _gate_fragments(raw: Any) -> tuple[Any, ...]:
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        return tuple(raw)
    return (raw,)


def normalize_gate_name_tuple(
    raw: Any,
    *,
    default: Sequence[str] | None = None,
    field_name: str = "gate list",
) -> tuple[str, ...]:
    """Normalize user/config gate lists while preserving legacy separators.

    Accepted separators are ASCII whitespace, commas, and semicolons.  Output is
    lower-case, de-duplicated in first-seen order, and validated with a
    conservative Qiskit-style instruction-name pattern.
    """

    use_default = raw is None or (isinstance(raw, str) and str(raw).strip() == "")
    fragments: tuple[Any, ...]
    if use_default:
        if default is None:
            return ()
        fragments = tuple(default)
    else:
        fragments = _gate_fragments(raw)

    names: list[str] = []
    saw_fragment = False
    for item in fragments:
        text = str(item).strip().lower()
        if not text:
            continue
        saw_fragment = True
        for token in _GATE_LIST_SEPARATOR_RE.split(text):
            name = token.strip().lower()
            if not name:
                continue
            if not _GATE_NAME_RE.fullmatch(name):
                raise ValueError(
                    f"{field_name} contains invalid gate name {name!r}; gate names must match "
                    "^[a-z][a-z0-9_]*$ and be separated by commas, semicolons, or whitespace."
                )
            if name not in names:
                names.append(name)

    if not names:
        qualifier = "configured " if not use_default and saw_fragment else ""
        raise ValueError(f"{qualifier}{field_name} must contain at least one gate name.")
    return tuple(names)


def gate_tuple_to_cli_value(gates: Sequence[str], *, field_name: str = "gate list") -> str:
    """Serialize a normalized gate tuple for new CLI/TSV records."""

    return ",".join(normalize_gate_name_tuple(gates, default=None, field_name=field_name))
