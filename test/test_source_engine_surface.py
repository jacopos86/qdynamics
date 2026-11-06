"""Engine surface and revision equivalence."""

from __future__ import annotations

import pytest

from pipelines.time_dynamics.diagnostics.source_engine_surface import (
    REVISION_EQUIVALENCE_V1,
    declare_revision_equivalence,
    python_import_closure,
    revision_equivalence_covers,
)

SOURCES = {
    "pipelines/run.py": "import os\nfrom pipelines.engine import step\nfrom src.model import H\n",
    "pipelines/engine.py": "from pipelines.solver import solve\n",
    "pipelines/solver.py": "import numpy as np\n",
    "src/model.py": "x = 1\n",
    "pipelines/report.py": "from pipelines.engine import step\n",
}


def _read(path: str) -> str | None:
    return SOURCES.get(path)


def test_closure_is_transitive_and_first_party_only() -> None:
    closure = python_import_closure("pipelines/run.py", _read)
    assert closure == {
        "pipelines/run.py",
        "pipelines/engine.py",
        "pipelines/solver.py",
        "src/model.py",
    }
    # A module that imports the engine is not itself reachable from the entry point.
    assert "pipelines/report.py" not in closure


def test_missing_and_unparsable_modules_are_dropped() -> None:
    closure = python_import_closure(
        "pipelines/run.py",
        lambda path: "from pipelines.absent import thing\n(" if path == "pipelines/run.py" else None,
    )
    assert closure == {"pipelines/run.py"}


def test_equivalence_accepts_a_difference_outside_the_surface() -> None:
    closure = python_import_closure("pipelines/run.py", _read)
    record = declare_revision_equivalence(
        left_revision="aaaa",
        right_revision="bbbb",
        differing_paths=["pipelines/report.py", "docs/notes.md"],
        engine_paths=closure,
    )
    assert record["schema"] == REVISION_EQUIVALENCE_V1
    assert record["engine_path_count"] == 4
    assert revision_equivalence_covers(record, "aaaa", "bbbb")
    assert revision_equivalence_covers(record, "bbbb", "aaaa")


def test_equivalence_refuses_a_difference_inside_the_surface() -> None:
    closure = python_import_closure("pipelines/run.py", _read)
    with pytest.raises(ValueError, match="inside the engine surface"):
        declare_revision_equivalence(
            left_revision="aaaa",
            right_revision="bbbb",
            differing_paths=["pipelines/solver.py"],
            engine_paths=closure,
        )


def test_equivalence_does_not_cover_other_revisions_or_a_tampered_record() -> None:
    closure = python_import_closure("pipelines/run.py", _read)
    record = declare_revision_equivalence(
        left_revision="aaaa",
        right_revision="bbbb",
        differing_paths=["pipelines/report.py"],
        engine_paths=closure,
    )
    assert not revision_equivalence_covers(record, "aaaa", "cccc")
    assert not revision_equivalence_covers(None, "aaaa", "bbbb")
    tampered = dict(record, differing_paths=["pipelines/solver.py"])
    with pytest.raises(ValueError, match="fails its own digest"):
        revision_equivalence_covers(tampered, "aaaa", "bbbb")


def test_equivalence_requires_distinct_revisions_and_a_surface() -> None:
    with pytest.raises(ValueError, match="distinct revisions"):
        declare_revision_equivalence(
            left_revision="aaaa",
            right_revision="aaaa",
            differing_paths=[],
            engine_paths=["pipelines/run.py"],
        )
    with pytest.raises(ValueError, match="surface is empty"):
        declare_revision_equivalence(
            left_revision="aaaa",
            right_revision="bbbb",
            differing_paths=[],
            engine_paths=[],
        )
