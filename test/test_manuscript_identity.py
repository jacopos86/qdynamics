"""Every manuscript-class file states whether it is the active source."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.shell.manuscript_identity import (  # noqa: E402
    BANNER_PREFIX,
    STATUS_ACTIVE,
    STATUS_NOT_ACTIVE,
    active_sources,
    apply_banners,
    banner_for,
    classify,
    declared_banner,
    is_manuscript,
    load_registry,
    manuscript_files,
    parse_banner,
    violations,
)


def test_every_manuscript_declares_its_identity_on_line_one() -> None:
    assert violations(ROOT) == []


def test_only_the_registry_active_sources_are_labelled_active() -> None:
    registry = load_registry(ROOT)
    active = set(active_sources(registry))
    labelled = {
        path.relative_to(ROOT).as_posix()
        for path in manuscript_files(ROOT)
        if parse_banner(declared_banner(path) or "")[0] == STATUS_ACTIVE
    }
    assert labelled == active
    # Papers I--III are edited in their own repositories, so no copy here may
    # claim to be active for them.
    for alias in ("paper_i", "paper_ii", "paper_iii"):
        assert alias not in set(active_sources(registry).values())


def test_a_copy_of_a_separate_repository_paper_is_never_active() -> None:
    registry = load_registry(ROOT)
    status, alias = classify(
        "MATH/paper_details/excited_spectra_static_paper_III.tex", registry
    )
    assert (status, alias) == (STATUS_NOT_ACTIVE, "paper_iii")
    banner = banner_for("MATH/paper_details/excited_spectra_static_paper_III.tex", registry)
    assert "QSE---Paper-III:Paper_III_overleaf_20260901" in banner
    # The most specific marker wins: paper_III contains paper_I as a substring.
    assert classify("MATH/notes/primer_paper_III_qse.tex", registry)[1] == "paper_iii"
    assert classify("MATH/paper_details/static_adapt_paper_I.tex", registry)[1] == "paper_i"


def test_an_unbannered_manuscript_copy_is_reported_and_repaired(tmp_path: Path) -> None:
    # A new copy dropped into MATH/ fails the gate until it declares itself.
    registry_dir = tmp_path / "agent_guidance"
    registry_dir.mkdir()
    (registry_dir / "paper-registry.yaml").write_text(
        (ROOT / "agent_guidance/paper-registry.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    target = tmp_path / "MATH" / "paper_details" / "excited_spectra_static_paper_III.tex"
    target.parent.mkdir(parents=True)
    target.write_text(
        "\\documentclass[reprint]{revtex4-2}\n\\begin{document}\n"
        "\\begin{abstract}copy\\end{abstract}\n\\end{document}\n",
        encoding="utf-8",
    )
    assert is_manuscript(target)
    assert len(violations(tmp_path)) == 1

    assert apply_banners(tmp_path) == [
        "MATH/paper_details/excited_spectra_static_paper_III.tex"
    ]
    assert violations(tmp_path) == []
    first = target.read_text(encoding="utf-8").split("\n")[0]
    assert first.startswith(BANNER_PREFIX) and STATUS_NOT_ACTIVE in first
    # Idempotent, and the document itself is untouched below line 1.
    assert apply_banners(tmp_path) == []
    assert target.read_text(encoding="utf-8").split("\n")[1].startswith("\\documentclass")


def test_a_wrong_banner_is_reported_not_silently_accepted(tmp_path: Path) -> None:
    registry_dir = tmp_path / "agent_guidance"
    registry_dir.mkdir()
    (registry_dir / "paper-registry.yaml").write_text(
        (ROOT / "agent_guidance/paper-registry.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    target = tmp_path / "MATH" / "paper_details" / "time_dynamics_paper_II.tex"
    target.parent.mkdir(parents=True)
    target.write_text(
        f"{BANNER_PREFIX} {STATUS_ACTIVE} | paper_ii | this file is the authoritative source"
        " | agent_guidance/paper-registry.yaml\n"
        "\\documentclass[reprint]{revtex4-2}\n\\maketitle\n",
        encoding="utf-8",
    )
    problems = violations(tmp_path)
    assert len(problems) == 1 and "registry says" in problems[0]


def test_figure_and_fragment_sources_are_out_of_scope(tmp_path: Path) -> None:
    fragment = tmp_path / "figure.tex"
    fragment.write_text("\\documentclass{standalone}\n\\begin{document}\\end{document}\n", encoding="utf-8")
    assert not is_manuscript(fragment)


@pytest.mark.parametrize("alias", ["paper_iv", "paper_v"])
def test_in_repository_papers_keep_an_active_source(alias: str) -> None:
    registry = load_registry(ROOT)
    sources = active_sources(registry)
    assert alias in sources.values()
    path = next(rel for rel, name in sources.items() if name == alias)
    assert (ROOT / path).is_file()
    assert classify(path, registry) == (STATUS_ACTIVE, alias)


def test_the_status_field_is_parsed_not_substring_matched() -> None:
    # "not-active" ends in "active"; a substring test reads every file as active.
    assert parse_banner(f"{BANNER_PREFIX} not-active | paper_iii | x | y") == (
        STATUS_NOT_ACTIVE,
        "paper_iii",
    )
    assert parse_banner(f"{BANNER_PREFIX} active | paper_v | x | y") == (STATUS_ACTIVE, "paper_v")
    with pytest.raises(Exception):
        parse_banner("% something else")
    with pytest.raises(Exception):
        parse_banner(f"{BANNER_PREFIX} maybe | paper_v | x | y")
