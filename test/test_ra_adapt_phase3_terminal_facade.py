from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXPECTED_OUTCOME = "phase_iii_no_positive_feasible_candidate_v1"


@pytest.mark.parametrize(
    "import_lines",
    (
        (
            "from pipelines.static_adapt import adaptive_phase_contracts as contracts",
            "from pipelines.static_adapt.ra_adapt import semantic_closure_routes as semantic",
            "import pipelines.static_adapt.ra_adapt as facade",
        ),
        (
            "import pipelines.static_adapt.ra_adapt as facade",
            "from pipelines.static_adapt import adaptive_phase_contracts as contracts",
            "from pipelines.static_adapt.ra_adapt import semantic_closure_routes as semantic",
        ),
    ),
)
def test_phase3_terminal_contract_is_stable_through_public_facade_import_order(
    import_lines: tuple[str, ...],
) -> None:
    """Fresh interpreters expose one cycle-neutral outcome and validator."""

    script = "\n".join(
        (
            *import_lines,
            f"assert contracts.ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1 == {_EXPECTED_OUTCOME!r}",
            "assert facade.ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1 == contracts.ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1",
            "assert facade.ADAPTIVE_PHASE3_NO_POSITIVE_POLICY_RAISE_V1 == contracts.ADAPTIVE_PHASE3_NO_POSITIVE_POLICY_RAISE_V1",
            "assert facade.ADAPTIVE_PHASE3_NO_POSITIVE_POLICY_TYPED_TERMINAL_V1 == contracts.ADAPTIVE_PHASE3_NO_POSITIVE_POLICY_TYPED_TERMINAL_V1",
            "assert facade.ADAPTIVE_HORIZON_POLICY_EXACT_TARGET_V1 == contracts.ADAPTIVE_HORIZON_POLICY_EXACT_TARGET_V1",
            "assert facade.ADAPTIVE_HORIZON_POLICY_MAXIMUM_V1 == contracts.ADAPTIVE_HORIZON_POLICY_MAXIMUM_V1",
            "assert facade.validate_semantic_phase3_no_positive_terminal_receipt is semantic.validate_semantic_phase3_no_positive_terminal_receipt",
            "assert facade.validate_semantic_phase3_natural_terminal_route_contract is semantic.validate_semantic_phase3_natural_terminal_route_contract",
            "assert facade.build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request is semantic.build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request",
            "assert facade.PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2 == semantic.PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2",
            "assert 'ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1' in facade.__all__",
            "assert 'validate_semantic_phase3_no_positive_terminal_receipt' in facade.__all__",
            "assert 'validate_semantic_phase3_natural_terminal_route_contract' in facade.__all__",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(_REPO_ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
