from __future__ import annotations

import math

from pipelines.reporting import append_paper_i_l3_weak_holstein_page19 as report


def test_l3_page19_adapter_closes_completed_append_and_pending_ra() -> None:
    adapter = report.build_adapter()

    assert adapter["status"] == "append_completed_3_of_3_ra_pending_3_of_3"
    assert adapter["paper_evidence_adopted"] is False
    assert [cell["regime_id"] for cell in adapter["cells"]] == list(
        report.REGIME_ORDER
    )
    assert adapter["sha256"] == report._canonical_sha256(
        {key: value for key, value in adapter.items() if key != "sha256"}
    )
    expected = {
        "weak_weak": (2.7241585398972252, 92, 90, 1230, 342, 43029),
        "intermediate_weak": (3.174244327566621, 4, 2, 1152, 254, 45276),
        "strong_weak_u8": (8.459890821267408, 92, 86, 1176, 338, 45909),
    }
    for cell in adapter["cells"]:
        terminal = cell["append"]["terminal"]
        observed = (
            terminal["delta_e"],
            terminal["N2q"],
            terminal["D2q"],
            terminal["Dc"],
            terminal["W1q"],
            terminal["S_alg"],
        )
        target = expected[cell["regime_id"]]
        assert math.isclose(observed[0], target[0], rel_tol=0.0, abs_tol=1.0e-14)
        assert observed[1:] == target[1:]
        assert [point["k"] for point in cell["append"]["points"]] == list(
            range(51)
        )
        assert cell["page12_ra"]["status"] == "idle_zero_starts"
        assert cell["page12_ra"]["trajectory_available"] is False
