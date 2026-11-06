"""Engine surface of a trajectory runner, and revision equivalence proved against it.

Two runs are produced by the same engine when every first-party module the runner
can import is byte-identical, whatever the git revision recorded beside them.  A
revision identifier is a coarser proxy: it also moves when a report generator, a
campaign lock, a submit package, or a worklog changes, none of which a running
trajectory can observe.

This module computes the surface that matters --- the transitive first-party
import closure of a declared entry point --- and turns "these two revisions differ
only outside that surface" into a checkable record.  Comparability checks accept
the record instead of comparing revision strings, and the record names what was
compared so the claim can be re-derived later.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

SOURCE_ENGINE_SURFACE_V1 = "source_engine_surface_v1"
REVISION_EQUIVALENCE_V1 = "source_revision_equivalence_v1"

#: Import roots treated as first party.  Anything else is a dependency pinned by
#: the container image, not by this repository's revision.
FIRST_PARTY_ROOTS = ("pipelines", "src")

#: The entry point every Paper II trajectory is launched through.
TRAJECTORY_ENTRY_POINT = "pipelines/time_dynamics/runners/ap_append_from_adapt_artifact.py"

SourceReader = Callable[[str], str | None]
"""Return the text of a repository-relative path, or ``None`` when absent."""


def _module_candidates(module: str) -> tuple[str, ...]:
    base = module.replace(".", "/")
    return (f"{base}.py", f"{base}/__init__.py")


def _imported_modules(tree: ast.AST) -> list[str]:
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module:
                # Relative imports do not occur in this tree; skip rather than guess.
                continue
            modules.append(node.module)
            # ``from pkg import mod`` may name a module rather than an attribute.
            modules.extend(f"{node.module}.{alias.name}" for alias in node.names)
    return modules


def python_import_closure(
    entry_point: str,
    read_source: SourceReader,
    *,
    first_party_roots: Sequence[str] = FIRST_PARTY_ROOTS,
) -> frozenset[str]:
    """Repository-relative paths reachable from ``entry_point`` by import.

    ``read_source`` supplies file text, so the closure can be taken at a git
    revision, in a worktree, or inside an extracted package.  A module that
    cannot be read is dropped: it is a third-party or generated import, and the
    caller's ``first_party_roots`` filter already excludes those by name.
    """

    roots = tuple(first_party_roots)
    seen: set[str] = set()
    pending = [entry_point]
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        source = read_source(path)
        if source is None:
            continue
        seen.add(path)
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for module in _imported_modules(tree):
            if not any(module == root or module.startswith(f"{root}.") for root in roots):
                continue
            for candidate in _module_candidates(module):
                if candidate not in seen:
                    pending.append(candidate)
    return frozenset(seen)


def _digest(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def declare_revision_equivalence(
    *,
    left_revision: str,
    right_revision: str,
    differing_paths: Iterable[str],
    engine_paths: Iterable[str],
    entry_point: str = TRAJECTORY_ENTRY_POINT,
    note: str | None = None,
) -> dict[str, Any]:
    """Record that two revisions differ only outside the engine surface.

    ``engine_paths`` is the union of the entry point's import closure taken at
    both revisions, so a module added or removed between them still counts.
    Raises when any differing path lies inside that surface: the revisions are
    then genuinely incomparable and no record is produced.
    """

    left = str(left_revision).strip()
    right = str(right_revision).strip()
    if not left or not right:
        raise ValueError("revision equivalence needs both revisions")
    if left == right:
        raise ValueError("revision equivalence is only meaningful for distinct revisions")
    differing = tuple(sorted({str(path) for path in differing_paths}))
    surface = frozenset(str(path) for path in engine_paths)
    if not surface:
        raise ValueError("engine surface is empty; the closure was not computed")
    intruding = tuple(sorted(surface.intersection(differing)))
    if intruding:
        raise ValueError(
            "revisions differ inside the engine surface: " + ", ".join(intruding[:8])
        )
    return {
        "schema": REVISION_EQUIVALENCE_V1,
        "surface_schema": SOURCE_ENGINE_SURFACE_V1,
        "entry_point": str(entry_point),
        "revisions": [left, right],
        "engine_path_count": len(surface),
        "engine_paths_sha256": _digest(surface),
        "differing_path_count": len(differing),
        "differing_paths_sha256": _digest(differing),
        "differing_paths": list(differing),
        "note": note or "",
    }


def revision_equivalence_covers(
    equivalence: Mapping[str, Any] | None,
    left_revision: str,
    right_revision: str,
) -> bool:
    """True when ``equivalence`` is a valid record for exactly these revisions."""

    if not equivalence:
        return False
    if equivalence.get("schema") != REVISION_EQUIVALENCE_V1:
        raise ValueError("unsupported revision-equivalence schema")
    declared = {str(value) for value in equivalence.get("revisions", ())}
    if declared != {str(left_revision), str(right_revision)}:
        return False
    if int(equivalence.get("engine_path_count", 0)) <= 0:
        raise ValueError("revision-equivalence record carries no engine surface")
    stated = tuple(equivalence.get("differing_paths", ()))
    if _digest(stated) != str(equivalence.get("differing_paths_sha256")):
        raise ValueError("revision-equivalence record fails its own digest")
    return True


def load_revision_equivalence(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        record = json.load(handle)
    if record.get("schema") != REVISION_EQUIVALENCE_V1:
        raise ValueError("unsupported revision-equivalence schema")
    return dict(record)


__all__ = [
    "FIRST_PARTY_ROOTS",
    "REVISION_EQUIVALENCE_V1",
    "SOURCE_ENGINE_SURFACE_V1",
    "TRAJECTORY_ENTRY_POINT",
    "declare_revision_equivalence",
    "load_revision_equivalence",
    "python_import_closure",
    "revision_equivalence_covers",
]
