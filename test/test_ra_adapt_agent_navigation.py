from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_GUIDANCE = REPO_ROOT / "agent_guidance"
STATIC_LANE = REPO_ROOT / "agent_guidance" / "static-adapt"
ARCHIVE_MANIFEST = (
    REPO_ROOT
    / "archive"
    / "paper_i_static_adapt_legacy_20260727"
    / "MANIFEST.json"
)

ORDINARY_SURFACES = (
    REPO_ROOT / "AGENTS.md",
    REPO_ROOT / "MATH" / "AGENTS.md",
    AGENT_GUIDANCE / "README.md",
    AGENT_GUIDANCE / "molecular-vibronic" / "AGENTS.md",
    AGENT_GUIDANCE / "skills" / "paper-i-run" / "SKILL.md",
    STATIC_LANE / "AGENTS.md",
    STATIC_LANE / "CONTEXT.md",
    STATIC_LANE / "run-guide.md",
    STATIC_LANE / "reporting" / "run-summary.md",
    AGENT_GUIDANCE / "shared" / "run-guide.md",
    *(sorted((STATIC_LANE / "policies").glob("*.md"))),
)

ROUTER_TARGETS = (
    REPO_ROOT / "agent_guidance" / "README.md",
    REPO_ROOT / "agent_guidance" / "shared" / "repository-work.md",
    REPO_ROOT / "agent_guidance" / "shared" / "scientific-invariants.md",
    REPO_ROOT / "agent_guidance" / "shared" / "run-guide.md",
    STATIC_LANE / "AGENTS.md",
    STATIC_LANE / "run-guide.md",
    STATIC_LANE / "policies" / "batching.md",
    STATIC_LANE / "policies" / "pruning.md",
    STATIC_LANE / "policies" / "beam.md",
    STATIC_LANE / "policies" / "resume.md",
    STATIC_LANE / "policies" / "insertion.md",
    STATIC_LANE / "policies" / "stopping.md",
    STATIC_LANE / "reporting" / "run-summary.md",
    REPO_ROOT / "agent_guidance" / "skills" / "paper-i-run" / "SKILL.md",
)

COMPATIBILITY_TARGETS = (
    STATIC_LANE / "route-identities.md",
    STATIC_LANE / "route-a-language.md",
    STATIC_LANE / "sr-snake-refactor-plan.md",
    STATIC_LANE / "paper-i-sr-snake-current-run-map.md",
    STATIC_LANE / "post-refactor-paper-i-evidence-queue.md",
    AGENT_GUIDANCE / "paper-lane-refactor-plan.md",
    AGENT_GUIDANCE / "shared" / "icm-gitnexus-pilot-plan.md",
    ARCHIVE_MANIFEST,
)

CONDITIONAL_ROUTES = {
    "greedy or combinatorial batching": "policies/batching.md",
    "metric or trust-region pruning": "policies/pruning.md",
    "beam or multiple accepted lineages": "policies/beam.md",
    "accepted-state checkpoint resume": "policies/resume.md",
    "insertion or append-only ablation": "policies/insertion.md",
    "round limit or exact-ED stop": "policies/stopping.md",
    (
        "summary, plateau, common accuracy, Qiskit resources, or `S_alg`"
    ): "reporting/run-summary.md",
}


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _is_quarantined_compatibility_doc(path: Path) -> bool:
    relative = path.relative_to(STATIC_LANE)
    name = relative.name
    return bool(
        "handoffs" in relative.parts
        or name.startswith("sr-snake-issue-")
        or name.startswith("ra-adapt-unification-")
        or name
        in {
            "route-identities.md",
            "route-a-language.md",
            "sr-snake-refactor-plan.md",
            "paper-i-sr-snake-current-run-map.md",
            "post-refactor-paper-i-evidence-queue.md",
        }
    )


def _is_allowed_historical_reference(path: Path) -> bool:
    if path.is_relative_to(STATIC_LANE):
        return _is_quarantined_compatibility_doc(path)
    return path in {
        AGENT_GUIDANCE / "paper-lane-refactor-plan.md",
        AGENT_GUIDANCE / "shared" / "icm-gitnexus-pilot-plan.md",
    }


def test_ra_adapt_router_targets_exist() -> None:
    missing = [
        str(path.relative_to(REPO_ROOT))
        for path in (*ROUTER_TARGETS, *COMPATIBILITY_TARGETS)
        if not path.is_file()
    ]
    assert missing == []
    assert tuple(STATIC_LANE.glob("sr-snake-issue-*-handoff.md"))
    assert (STATIC_LANE / "handoffs").is_dir()


def test_ordinary_navigation_names_the_ra_adapt_seam() -> None:
    root = _text(REPO_ROOT / "AGENTS.md")
    math = _text(REPO_ROOT / "MATH" / "AGENTS.md")
    lane = _text(STATIC_LANE / "AGENTS.md")
    guide = _text(STATIC_LANE / "run-guide.md")
    reporting = _text(STATIC_LANE / "reporting" / "run-summary.md")
    shared = _text(AGENT_GUIDANCE / "shared" / "run-guide.md")
    guidance_index = _text(AGENT_GUIDANCE / "README.md")
    run_skill = _text(
        AGENT_GUIDANCE / "skills" / "paper-i-run" / "SKILL.md"
    )
    molecular_lane = _text(
        AGENT_GUIDANCE / "molecular-vibronic" / "AGENTS.md"
    )

    assert "`run_ra_adapt` facade" in root
    assert "`run_ra_adapt(problem, request=None)`" in math
    assert (
        "`pipelines.static_adapt.ra_adapt.run_ra_adapt("
        "problem, request=None)`"
    ) in lane
    assert (
        "from pipelines.static_adapt.ra_adapt import run_ra_adapt"
    ) in guide
    assert "run_ra_adapt(problem, request=None)" in guide
    request_shape = guide.split("```text", maxsplit=1)[1].split(
        "```",
        maxsplit=1,
    )[0]
    assert request_shape.split() == [
        "adapter",
        "method",
        "execution",
        "observation",
    ]
    assert "run_ra_adapt(problem, request=None)" in shared
    assert "current typed `run_ra_adapt` facade" in guidance_index
    assert (
        "`pipelines.static_adapt.ra_adapt.run_ra_adapt("
        "problem, request=None)`"
    ) in run_skill
    assert "current typed `run_ra_adapt` facade" in molecular_lane
    assert "`result.run.paper_i_summary`" in reporting

    stale = [
        str(path.relative_to(REPO_ROOT))
        for path in ORDINARY_SURFACES
        if "run_sr_snake" in _text(path)
    ]
    assert stale == []


def test_policy_and_reporting_guides_remain_intent_gated() -> None:
    lane = _text(STATIC_LANE / "AGENTS.md")
    for intent, relative_target in CONDITIONAL_ROUTES.items():
        expected_row = (
            f"| {intent} | "
            f"`agent_guidance/static-adapt/{relative_target}` |"
        )
        assert expected_row in lane

    policy_keywords = {
        "batching.md": "batching",
        "pruning.md": "pruning",
        "beam.md": "beam",
        "resume.md": "resume",
        "insertion.md": "insertion",
        "stopping.md": "stop",
    }
    for name, keyword in policy_keywords.items():
        opening = "\n".join(
            _text(STATIC_LANE / "policies" / name).splitlines()[:5]
        ).lower()
        assert "read this file" in opening
        assert keyword in opening

    assert (
        "Read `agent_guidance/static-adapt/reporting/run-summary.md` only "
        "for run"
    ) in lane
    assert "accepted-prefix resources" in lane
    assert "summary, plateau, common accuracy" in lane


def test_agent_guides_do_not_duplicate_executable_scientific_defaults() -> None:
    guide = _text(STATIC_LANE / "run-guide.md")
    policies = {
        path.name: _text(path)
        for path in sorted((STATIC_LANE / "policies").glob("*.md"))
    }

    assert "build_resolved_ra_protocol" in guide
    assert "sole executable source for scientific defaults" in guide
    assert "## Silent canonical contract" not in guide
    assert "| pool |" not in guide
    assert "| horizon |" not in guide

    combined = "\n".join(policies.values())
    assert "maximum_size=3" not in combined
    assert "search_window_size=6" not in combined
    assert "fork-local S_alg weight = 0.01" not in combined
    assert "strictly below `1e-8`" not in combined
    assert "defaults to `50`" not in combined
    assert "maximum_controller_rounds=50" not in combined


def test_historical_facade_references_stay_quarantined() -> None:
    offenders: list[str] = []
    retained: list[str] = []
    for path in sorted(AGENT_GUIDANCE.rglob("*.md")):
        if "run_sr_snake" not in _text(path):
            continue
        relative = str(path.relative_to(REPO_ROOT))
        retained.append(relative)
        if not _is_allowed_historical_reference(path):
            offenders.append(relative)

    assert retained
    assert offenders == []

    lane = _text(STATIC_LANE / "AGENTS.md")
    ordinary_lane, quarantine = lane.split(
        "## Compatibility quarantine",
        maxsplit=1,
    )
    assert "route-identities.md" not in ordinary_lane
    assert "route-identities.md" in quarantine

    math = _text(REPO_ROOT / "MATH" / "AGENTS.md")
    assert "Only for an explicit historical provenance request" in math
    assert "historical provenance workflow" in math


def test_materialization_is_not_execution_authorization() -> None:
    shared = _text(AGENT_GUIDANCE / "shared" / "run-guide.md")
    guide = _text(STATIC_LANE / "run-guide.md")

    assert "execution_authorized=false" in shared
    assert "obtain current authorization" in shared
    assert "materialized bundle does not authorize execution" in guide
    assert "icm-gitnexus-pilot-plan.md" not in shared
