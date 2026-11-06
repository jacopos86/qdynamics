"""One-cell locks that declare the pre-permission ray threshold.

The corrected lock stays undeclared (the historical coupling) and its argv is
unchanged; a sweep lock declares the threshold once, in ``extra_flags``, and
every action it yields carries that literal and nothing else new.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.time_dynamics.campaigns import (
    paper_ii_final_estimator_telemetry_v1 as telemetry,
    paper_ii_final_selector_comparison_v1 as selector,
)
from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import (
    _build_parser,
)

CAMPAIGNS = Path(selector.__file__).parent
CORRECTED = CAMPAIGNS / "paper_ii_corrected_exchange_cells_v1.json"
SWEEP = {
    "2e-2": CAMPAIGNS / "paper_ii_permission_threshold_2em02_cells_v1.json",
    "5e-2": CAMPAIGNS / "paper_ii_permission_threshold_5em02_cells_v1.json",
    "1e-1": CAMPAIGNS / "paper_ii_permission_threshold_1em01_cells_v1.json",
    "5e-3": CAMPAIGNS / "paper_ii_permission_threshold_5em03_cells_v1.json",
    "1e-2": CAMPAIGNS / "paper_ii_permission_threshold_1em02_cells_v1.json",
}
FLAG = selector.PERMISSION_RAY_FLAG


def _load(path: Path) -> selector.MatchedCellLock:
    return selector.load_lock(path, **selector.lock_parameters(path))


def _pairs(argv: list[str]) -> set[tuple[str, str]]:
    return set(zip(argv, argv[1:]))


def test_the_corrected_lock_stays_undeclared_and_flagless(tmp_path: Path) -> None:
    lock = _load(CORRECTED)
    assert lock.permission_ray_distance_max is None
    rows = telemetry.build_actions(
        lock, output_root=tmp_path, comparator_mode=telemetry.COMPARATOR_MODE_REUSE
    )
    assert len(rows) == 11
    assert all(FLAG not in row["runner_argv"] for row in rows)


@pytest.mark.parametrize("literal", sorted(SWEEP))
def test_each_sweep_lock_is_one_cell_carrying_its_literal(
    literal: str, tmp_path: Path
) -> None:
    lock = _load(SWEEP[literal])
    assert lock.campaign_id.startswith("paper_ii_permission_threshold_")
    assert lock.permission_ray_distance_max == literal
    assert [cell.cell_id for cell in lock.cells] == ["strongfast__delta_theta_5e-5"]
    assert lock.exclusions == ()
    rows = telemetry.build_actions(
        lock, output_root=tmp_path, comparator_mode=telemetry.COMPARATOR_MODE_REUSE
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["campaign_id"] == lock.campaign_id
    argv = list(row["runner_argv"])
    assert (FLAG, literal) in _pairs(argv)
    assert ("--exchange-insertion-frontier", "8") in _pairs(argv)
    parsed = _build_parser().parse_args(argv)
    assert parsed.exchange_permission_ray_distance_max == pytest.approx(float(literal))
    assert parsed.prune_ray_distance_tol == pytest.approx(2.0e-3)


def test_sweep_argv_differs_from_the_corrected_cell_only_by_the_flag_pair(
    tmp_path: Path,
) -> None:
    corrected = telemetry.build_actions(
        _load(CORRECTED), output_root=tmp_path,
        comparator_mode=telemetry.COMPARATOR_MODE_REUSE,
    )
    base = next(r for r in corrected if r["cell_id"] == "strongfast__delta_theta_5e-5")
    sweep = telemetry.build_actions(
        _load(SWEEP["5e-2"]), output_root=tmp_path,
        comparator_mode=telemetry.COMPARATOR_MODE_REUSE,
    )[0]
    strip = lambda argv: [  # noqa: E731
        t for i, t in enumerate(argv)
        if t not in (FLAG, "5e-2") and not (i > 0 and argv[i - 1] == "--output-json")
        and t != "--output-json"
    ]
    assert strip(list(sweep["runner_argv"])) == strip(list(base["runner_argv"]))


def test_a_non_positive_or_malformed_literal_fails_closed(tmp_path: Path) -> None:
    payload = json.loads(SWEEP["5e-2"].read_text(encoding="utf-8"))
    for bad in ("0", "-1e-3", "nan", "abc"):
        payload["registry_bindings"]["extra_flags"] = ["--record-statevector", FLAG, bad]
        path = tmp_path / f"lock_{bad}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(selector.MatchedCellLockError, match="positive finite"):
            _load(path)


def test_an_unrelated_extra_flag_is_still_refused(tmp_path: Path) -> None:
    payload = json.loads(SWEEP["5e-2"].read_text(encoding="utf-8"))
    payload["registry_bindings"]["extra_flags"] = ["--record-statevector", "--ridge-lambda", "1e-5"]
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(selector.MatchedCellLockError, match="extra flags"):
        _load(path)
