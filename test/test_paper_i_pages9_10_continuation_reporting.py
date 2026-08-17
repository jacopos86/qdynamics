from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pipelines.reporting import (
    build_paper_i_stationary_vs_paper_i_route_comparison_pdf as comparison,
)
from pipelines.reporting import (
    paper_i_mixed_horizon_continuation as mixed,
)
from pipelines.reporting import (
    update_paper_i_macro_then_singleton_phase23_qiskit_no_lanes_page10 as page10,
)
from pipelines.reporting import (
    update_paper_i_phase3_qiskit_no_lanes_page9 as page9,
)


def _route() -> dict[str, object]:
    return {
        "status": "complete",
        "points": [{"k": k, "error": 1.0 / k} for k in range(1, 51)],
        "marker": {"k": 40, "error": 1.0 / 40},
        "terminal": {
            "k": 50,
            "error": 1.0 / 50,
            "N2q": 10,
            "D2q": 11,
            "Dc": 12,
            "W1q": 13,
            "S_alg": 14,
        },
        "exact_same_cutoff_energy": 0.0,
        "source_bindings": {},
    }


def test_mixed_horizon_keeps_fixed_round_costs_and_deduplicates_full_trace() -> None:
    base = _route()
    full_resumed_trace = [
        {"k": k, "error": 1.0 / k} for k in range(1, 57)
    ]

    decorated = mixed.decorate_route(
        base,
        regime_id="weak_strong",
        continuation_points=full_resumed_trace,
    )

    assert len(decorated["points"]) == 50
    assert len(decorated["trajectory_points"]) == 56
    assert decorated["trajectory_terminal"] == {"k": 56, "error": 1.0 / 56}
    assert decorated["terminal"] == base["terminal"]
    assert decorated["paper_facing_fixed_round_50"] == base["terminal"]
    assert decorated["continuation"] == {
        "status": "recoverable_prefix_incomplete",
        "selected": True,
        "base_round": 50,
        "target_round": 70,
        "observed_through_round": 56,
        "continuation_point_count": 6,
        "source": None,
    }


def test_mixed_horizon_rejects_disagreeing_overlap() -> None:
    continuation = [{"k": k, "error": 1.0 / k} for k in range(1, 52)]
    continuation[49]["error"] = 999.0
    with pytest.raises(
        mixed.MixedHorizonContinuationError,
        match="disagrees at k=50",
    ):
        mixed.decorate_route(
            _route(),
            regime_id="weak_strong",
            continuation_points=continuation,
        )


def test_continuation_adapter_schema_is_digested_and_route_bound() -> None:
    cell = {
        "regime_id": "strong_strong_u8",
        "status": "complete",
        "observed_through_round": 70,
        "trajectory_points": [
            {"k": k, "error": 1.0 / k} for k in range(51, 71)
        ],
        "source_bindings": {"summary": {"sha256": "a" * 64}},
    }
    adapter = mixed.digested_continuation_adapter(
        route_contract_sha256="b" * 64,
        cells=[cell],
        status="partial_1_of_3_complete",
    )

    validated = mixed.validate_continuation_adapter(
        adapter,
        expected_route_contract_sha256="b" * 64,
    )
    assert validated["strong_strong_u8"]["status"] == "complete"
    assert validated["strong_strong_u8"]["points"][-1]["k"] == 70

    drifted = copy.deepcopy(adapter)
    drifted["cells"][0]["trajectory_points"][-1]["error"] = 1.0
    with pytest.raises(
        mixed.MixedHorizonContinuationError,
        match="identity drifted",
    ):
        mixed.validate_continuation_adapter(
            drifted,
            expected_route_contract_sha256="b" * 64,
        )


def test_page10_adapter_absorbs_recoverable_prefixes_without_replacing_k50(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provenance = {
        "phase3_on_plateau_singleton_sixregime_r50": {
            "cells": [
                {
                    "regime_id": regime,
                    "append_adapt": copy.deepcopy(_route()),
                }
                for regime in page10.REGIME_ORDER
            ]
        },
        "included_sources": [],
    }
    recoverable = {
        "weak_strong": {
            "points": [
                {"k": k, "energy": -1.0 / k, "error": 1.0 / k}
                for k in range(51, 57)
            ],
            "status": "recoverable_prefix_incomplete",
            "exact_same_cutoff_energy": 0.0,
            "source": {"kind": "fixture"},
        },
        "intermediate_strong": {
            "points": [{"k": 51, "energy": -1.0 / 51, "error": 1.0 / 51}],
            "status": "recoverable_prefix_incomplete",
            "exact_same_cutoff_energy": 0.0,
            "source": {"kind": "fixture"},
        },
    }
    monkeypatch.setattr(page10, "ADAPTER_PATH", tmp_path / "adapter.json")
    monkeypatch.setattr(
        page10,
        "load_current",
        lambda regime, spec: copy.deepcopy(_route()),
    )
    monkeypatch.setattr(page10, "load_recoverable_continuations", lambda: recoverable)
    monkeypatch.setattr(page10, "load_continuation_adapter", lambda: {})

    adapter = page10.build_adapter(provenance)
    by_regime = {row["regime_id"]: row for row in adapter["cells"]}
    weak_strong = by_regime["weak_strong"]["macro_then_singleton"]
    intermediate = by_regime["intermediate_strong"]["macro_then_singleton"]
    strong_strong = by_regime["strong_strong_u8"]["macro_then_singleton"]

    assert len(weak_strong["points"]) == 50
    assert [row["k"] for row in weak_strong["trajectory_points"]] == list(
        range(1, 57)
    )
    assert intermediate["trajectory_terminal"]["k"] == 51
    assert strong_strong["trajectory_terminal"]["k"] == 50
    assert strong_strong["continuation"]["status"] == "pending"
    assert weak_strong["paper_facing_fixed_round_50"] == weak_strong["terminal"]
    assert adapter["horizon_policy"]["paper_facing_cost_round"] == 50


def test_stationary_comparison_does_not_append_adapter_prefix_twice() -> None:
    base = [{"k": k, "error": 1.0 / k} for k in range(1, 51)]
    adapter = [{"k": k, "error": 1.0 / k} for k in range(1, 57)]
    retained = [{"k": k, "error": 1.0 / k} for k in range(51, 57)]

    merged = comparison._merge_page10_trajectory(
        base_points=base,
        adapter_trajectory=adapter,
        retained_continuation=retained,
        label="weak_strong Page-10",
    )

    assert [row["k"] for row in merged] == list(range(1, 57))


@pytest.mark.parametrize(
    ("module", "page_number", "layout_key"),
    ((page9, 9, "page_9"), (page10, 10, "page_10")),
)
def test_page_replacement_preserves_pages_11_and_12(
    module: object,
    page_number: int,
    layout_key: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pypdf = pytest.importorskip("pypdf")
    target_pdf = tmp_path / f"target-page-{page_number}.pdf"
    page_pdf = tmp_path / f"replacement-page-{page_number}.pdf"
    target_provenance = tmp_path / "provenance.json"
    adapter_path = tmp_path / "adapter.json"
    page_png = tmp_path / "page.png"

    writer = pypdf.PdfWriter()
    for index in range(12):
        writer.add_blank_page(width=600 + index, height=800)
    with target_pdf.open("wb") as stream:
        writer.write(stream)
    replacement = pypdf.PdfWriter()
    replacement.add_blank_page(width=999, height=800)
    with page_pdf.open("wb") as stream:
        replacement.write(stream)
    adapter_path.write_text("{}\n", encoding="utf-8")
    page_png.write_bytes(b"fixture")

    monkeypatch.setattr(module, "TARGET_PDF", target_pdf)
    monkeypatch.setattr(module, "TARGET_PROVENANCE", target_provenance)
    monkeypatch.setattr(module, "PAGE_PDF", page_pdf)
    monkeypatch.setattr(module, "PAGE_PNG", page_png)
    monkeypatch.setattr(module, "ADAPTER_PATH", adapter_path)
    provenance = {
        "layout": {
            "page_count": 12,
            layout_key: module.PAGE_ID,
            "page_11": "fixture-page-11",
            "page_12": "fixture-page-12",
        },
        "outputs": {"partial_progress_pdf": module.binding(target_pdf)},
    }
    adapter = {
        "status": "fixture",
        "sha256": "f" * 64,
        "cells": [],
        "horizon_policy": mixed.horizon_policy(),
    }

    result = module.append_page(adapter, provenance)
    updated = json.loads(target_provenance.read_text(encoding="utf-8"))

    assert result["page_count"] == 12
    assert len(pypdf.PdfReader(str(target_pdf)).pages) == 12
    assert updated["layout"]["page_11"] == "fixture-page-11"
    assert updated["layout"]["page_12"] == "fixture-page-12"
    assert updated["layout"]["page_count"] == 12
