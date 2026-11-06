"""Declare, in every manuscript file, whether it is the active source.

The failure this prevents: a tool or agent that indexes this repository finds
one of the ~35 historical, upload, backup, or draft manuscript copies under
``MATH/`` and links a reader to it instead of the active manuscript, which for
Papers I--III lives in a separate repository.  On 2026-09-08 a source-bundle
viewer resolved ``eq:siam_hamiltonian`` to
``MATH/paper_details/excited_spectra_static_paper_III.tex``, a 2026-08-29
snapshot 2,993 lines away from the manuscript actually being edited.

``agent_guidance/paper-registry.yaml`` already says which file is active, but a
tool has to know to consult it.  A one-line banner on line 1 of every
manuscript-class file states the answer where any reader -- human, agent, or
indexer -- is already looking.  ``--check`` is a regression gate; ``--apply``
writes or refreshes the banners.

Nothing is deleted: a superseded copy stays where it is, labelled.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = Path("agent_guidance/paper-registry.yaml")
SEARCH_ROOTS = ("MATH",)
BANNER_PREFIX = "% MANUSCRIPT-IDENTITY:"
STATUS_ACTIVE = "active"
STATUS_NOT_ACTIVE = "not-active"
UNASSIGNED = "unassigned"

# A manuscript-class file declares a document class and either an abstract or a
# title block.  Figure sources, standalone tikz pages and macro fragments do not
# and are out of scope: no reader mistakes them for the paper.
_DOCUMENT_CLASS = re.compile(r"\\documentclass")
_MANUSCRIPT_BODY = re.compile(r"\\begin\{abstract\}|\\maketitle")
_HEAD_LINES = 400

# Most specific first: "paper_III" contains "paper_I" as a substring.
PAPER_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("paper_iii", ("paper_iii", "paper_III", "excited_spectra")),
    ("paper_iv", ("paper_iv", "paper_IV", "molecular_vibronic")),
    ("paper_v", ("paper_v", "paper_V", "high_u_gkba")),
    ("paper_ii", ("paper_ii", "paper_II", "time_dynamics")),
    ("paper_i", ("paper_i", "paper_I", "static_adapt", "snake")),
)


class ManuscriptIdentityError(RuntimeError):
    """A manuscript file's declared identity disagrees with the registry."""


def load_registry(root: Path = REPO_ROOT) -> Mapping[str, object]:
    return yaml.safe_load((root / REGISTRY_PATH).read_text(encoding="utf-8"))


def active_sources(registry: Mapping[str, object]) -> dict[str, str]:
    """Repository-relative path -> paper alias, for papers this repo owns.

    A paper whose manuscript repository is not this one has no active source
    here, so every local copy of it is superseded by construction.
    """

    out: dict[str, str] = {}
    for alias, entry in dict(registry["papers"]).items():  # type: ignore[index]
        manuscript = dict(entry["manuscript"])  # type: ignore[index]
        if str(manuscript.get("repository_name")) != "Holstein_test":
            continue
        source = manuscript.get("source")
        if source:
            out[str(source)] = str(alias)
    return out


def active_reference(registry: Mapping[str, object], alias: str) -> str:
    """``repository_name:source`` of the paper's active manuscript."""

    manuscript = dict(dict(registry["papers"])[alias]["manuscript"])  # type: ignore[index]
    return f"{manuscript['repository_name']}:{manuscript['source']}"


def classify(rel_path: str, registry: Mapping[str, object]) -> tuple[str, str]:
    """(status, paper alias) for one manuscript-class file."""

    active = active_sources(registry)
    if rel_path in active:
        return STATUS_ACTIVE, active[rel_path]
    for alias, markers in PAPER_MARKERS:
        if any(marker in rel_path for marker in markers):
            return STATUS_NOT_ACTIVE, alias
    return STATUS_NOT_ACTIVE, UNASSIGNED


def banner_for(rel_path: str, registry: Mapping[str, object]) -> str:
    status, alias = classify(rel_path, registry)
    if status == STATUS_ACTIVE:
        tail = "this file is the authoritative source"
    elif alias == UNASSIGNED:
        tail = "no paper in the registry claims this file"
    else:
        tail = f"active source: {active_reference(registry, alias)}"
    return f"{BANNER_PREFIX} {status} | {alias} | {tail} | {REGISTRY_PATH}"


def is_manuscript(path: Path) -> bool:
    try:
        head = "".join(path.open(encoding="utf-8", errors="replace").readlines()[:_HEAD_LINES])
    except OSError:
        return False
    return bool(_DOCUMENT_CLASS.search(head) and _MANUSCRIPT_BODY.search(head))


def manuscript_files(root: Path = REPO_ROOT, roots: Sequence[str] = SEARCH_ROOTS) -> Iterator[Path]:
    for search_root in roots:
        for path in sorted((root / search_root).rglob("*.tex")):
            if path.is_file() and is_manuscript(path):
                yield path


def declared_banner(path: Path) -> str | None:
    with path.open(encoding="utf-8", errors="replace") as handle:
        first = handle.readline().rstrip("\n")
    return first if first.startswith(BANNER_PREFIX) else None


def parse_banner(line: str) -> tuple[str, str]:
    """(status, paper alias) declared by a banner line.

    ``not-active`` ends in ``active``, so a substring test on the whole field
    silently reads every file as active; parse the fields instead.
    """

    if not line.startswith(BANNER_PREFIX):
        raise ManuscriptIdentityError(f"not an identity banner: {line!r}")
    fields = [field.strip() for field in line[len(BANNER_PREFIX) :].split("|")]
    if len(fields) < 2 or fields[0] not in (STATUS_ACTIVE, STATUS_NOT_ACTIVE):
        raise ManuscriptIdentityError(f"malformed identity banner: {line!r}")
    return fields[0], fields[1]


def violations(root: Path = REPO_ROOT, roots: Sequence[str] = SEARCH_ROOTS) -> list[str]:
    registry = load_registry(root)
    out: list[str] = []
    for path in manuscript_files(root, roots):
        rel = path.relative_to(root).as_posix()
        expected = banner_for(rel, registry)
        found = declared_banner(path)
        if found is None:
            out.append(f"{rel}: no identity banner on line 1; expected {expected!r}")
        elif found != expected:
            out.append(f"{rel}: banner is {found!r}; registry says {expected!r}")
    return out


def apply_banners(
    root: Path = REPO_ROOT, roots: Sequence[str] = SEARCH_ROOTS, paths: Iterable[Path] | None = None
) -> list[str]:
    registry = load_registry(root)
    changed: list[str] = []
    for path in sorted(paths) if paths is not None else manuscript_files(root, roots):
        rel = path.relative_to(root).as_posix()
        expected = banner_for(rel, registry)
        text = path.read_text(encoding="utf-8")
        lines = text.split("\n")
        if lines and lines[0].startswith(BANNER_PREFIX):
            if lines[0] == expected:
                continue
            lines[0] = expected
        else:
            lines.insert(0, expected)
        path.write_text("\n".join(lines), encoding="utf-8")
        changed.append(rel)
    return changed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="exit 1 listing files whose banner is missing or wrong")
    mode.add_argument("--apply", action="store_true", help="write or refresh every banner")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.apply:
        changed = apply_banners(root)
        print(f"banners written: {len(changed)}")
        for rel in changed:
            print(f"  {rel}")
        return 0
    problems = violations(root)
    for problem in problems:
        print(problem)
    print(f"{len(problems)} manuscript files misdeclare their identity")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
