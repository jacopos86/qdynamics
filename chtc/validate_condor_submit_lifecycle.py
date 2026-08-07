#!/usr/bin/env python3
"""Validate bounded factories and explicit ordinary-held release modes."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable


_ASSIGNMENT_RE = re.compile(
    r"^\s*(?P<name>[+A-Za-z_][+.A-Za-z0-9_]*)\s*=\s*(?P<value>.*?)\s*$"
)
_ORDINARY_HELD_MODE = "ordinary_held_exact_proc_release_v1"
_LIFECYCLE_MODE_ATTRIBUTE = "+holsteinlifecyclemode"


class SubmitLifecycleError(ValueError):
    """Raised when a submit description can retain every factory slot."""


def parse_submit_assignments(text: str) -> dict[str, tuple[str, ...]]:
    """Return case-normalized submit assignments without interpreting ClassAds."""

    values: dict[str, list[str]] = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _ASSIGNMENT_RE.match(line)
        if match is None:
            continue
        name = match.group("name").lower()
        values.setdefault(name, []).append(match.group("value").strip())
    return {name: tuple(items) for name, items in values.items()}


def _strip_balanced_outer_parentheses(value: str) -> str:
    candidate = value.strip()
    while candidate.startswith("(") and candidate.endswith(")"):
        depth = 0
        encloses_entire_value = True
        for index, character in enumerate(candidate):
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth < 0:
                    return candidate
                if depth == 0 and index != len(candidate) - 1:
                    encloses_entire_value = False
                    break
        if depth != 0 or not encloses_entire_value:
            break
        candidate = candidate[1:-1].strip()
    return candidate


def _positive_integer(value: str) -> int | None:
    candidate = _strip_balanced_outer_parentheses(value)
    if not candidate.isdecimal():
        return None
    parsed = int(candidate)
    return parsed if parsed > 0 else None


def _nonnegative_integer(value: str) -> int | None:
    candidate = _strip_balanced_outer_parentheses(value)
    if not candidate.isdecimal():
        return None
    return int(candidate)


def _constant_true(value: str) -> bool:
    return _strip_balanced_outer_parentheses(value).casefold() == "true"


def _constant_false(value: str) -> bool:
    return _strip_balanced_outer_parentheses(value).casefold() == "false"


def _constant_string(value: str) -> str | None:
    candidate = _strip_balanced_outer_parentheses(value)
    if (
        len(candidate) >= 2
        and candidate[0] == candidate[-1]
        and candidate[0] in {'"', "'"}
    ):
        return candidate[1:-1]
    return None


def factory_lifecycle_blockers(text: str) -> tuple[str, ...]:
    """Describe unsafe factory or reserved ordinary-held lifecycle policies."""

    assignments = parse_submit_assignments(text)
    limits = assignments.get("max_materialize", ())
    idle_limits = assignments.get("max_idle", ())
    retention = assignments.get("leave_in_queue", ())
    lifecycle_modes = assignments.get(_LIFECYCLE_MODE_ATTRIBUTE, ())
    holds = assignments.get("hold", ())
    periodic_releases = assignments.get("periodic_release", ())
    blockers: list[str] = []
    if len(limits) > 1:
        blockers.append("max_materialize is assigned more than once")
    if len(idle_limits) > 1:
        blockers.append("max_idle is assigned more than once")
    if len(retention) > 1:
        blockers.append("leave_in_queue is assigned more than once")
    if len(lifecycle_modes) > 1:
        blockers.append("HolsteinLifecycleMode is assigned more than once")

    lifecycle_mode = (
        _constant_string(lifecycle_modes[0])
        if len(lifecycle_modes) == 1
        else None
    )
    if lifecycle_modes and lifecycle_mode != _ORDINARY_HELD_MODE:
        blockers.append("unsupported HolsteinLifecycleMode")
    elif lifecycle_mode == _ORDINARY_HELD_MODE:
        if limits or idle_limits:
            blockers.append(
                "ordinary-held lifecycle mode must not use "
                "max_materialize or max_idle"
            )
        if len(holds) != 1 or not _constant_true(holds[0]):
            blockers.append(
                "ordinary-held lifecycle mode requires exactly one "
                "hold=True assignment"
            )
        if (
            len(periodic_releases) != 1
            or not _constant_false(periodic_releases[0])
        ):
            blockers.append(
                "ordinary-held lifecycle mode requires "
                "periodic_release=False"
            )
        if any(_constant_true(value) for value in retention):
            blockers.append(
                "ordinary-held lifecycle mode must not retain "
                "successful released jobs"
            )

    factory_requested = bool(limits or idle_limits)
    positive_limits = tuple(
        limit
        for value in limits
        if (limit := _positive_integer(value)) is not None
    )
    if factory_requested and (
        len(limits) != 1 or len(positive_limits) != 1
    ):
        blockers.append(
            "factory mode requires exactly one positive constant "
            "max_materialize"
        )
    if idle_limits and (
        len(idle_limits) != 1
        or _nonnegative_integer(idle_limits[0]) is None
    ):
        blockers.append(
            "factory mode requires max_idle, when present, to be one "
            "nonnegative constant integer"
        )
    if positive_limits and any(_constant_true(value) for value in retention):
        blockers.append(
            "positive max_materialize is incompatible with unconditional "
            "leave_in_queue=True because successful jobs never free factory slots"
        )
    return tuple(blockers)


def validate_submit_lifecycle(text: str) -> None:
    """Raise when a declared lifecycle mode is unsafe or unprovable."""

    blockers = factory_lifecycle_blockers(text)
    if blockers:
        raise SubmitLifecycleError("; ".join(blockers))


def _validate_paths(paths: Iterable[Path]) -> int:
    failed = False
    for path in paths:
        try:
            validate_submit_lifecycle(path.read_text(encoding="utf-8"))
        except (OSError, SubmitLifecycleError) as exc:
            failed = True
            print(f"{path}: {exc}")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate bounded factories and reserved ordinary-held "
            "release modes."
        )
    )
    parser.add_argument("submit_files", nargs="+", type=Path)
    args = parser.parse_args()
    return _validate_paths(args.submit_files)


if __name__ == "__main__":
    raise SystemExit(main())
