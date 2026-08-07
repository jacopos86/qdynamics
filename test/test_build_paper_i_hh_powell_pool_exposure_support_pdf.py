from __future__ import annotations

from pathlib import Path

from pipelines.reporting.build_paper_i_hh_powell_pool_exposure_support_pdf import (
    DerivedRow,
    ROLE_SPECS,
    generic_prefix_s_alg,
    selected_prefix_payload,
    write_tex,
)
from pipelines.reporting.paper_i_run_summary import EFFECTIVE_PLATEAU_POLICY


def _role(method: str):
    return next(spec for spec in ROLE_SPECS if spec.method == method)


def _manifest_row() -> DerivedRow:
    return DerivedRow(
        role_key="snake_native_a1",
        role_display="SNAKE",
        role_description="native singleton response",
        matrix_label="A_native",
        regime="weak-weak",
        method="snake",
        optimizer="POWELL",
        status="complete",
        plotted=True,
        missing_reason="",
        selection_policy=EFFECTIVE_PLATEAU_POLICY,
        selection_status="ok",
        selected_prefix_k=2,
        record_id="weak-weak-snake",
        input_report_json="records/input-report.json",
        input_report_json_sha256="inputsha256",
        iteration=2,
        depth=2,
        abs_delta_e=0.1,
        fidelity=None,
        fidelity_status="blocked_selected_prefix_state_not_serialized",
        fidelity_source="not_computed",
        fidelity_status_detail="selected prefix state unavailable",
        n2q=12,
        d2q=10,
        dc=18,
        cost_source="selected_prefix_compile",
        cost_status="ok",
        s_grad=3,
        s_refit=4,
        s_outer=2,
        s_h=6,
        s_metric=1,
        s_alg=10,
        s_work_status="ok",
        s_work_source="canonical_receipt",
        s_work_status_detail="",
        trajectory_points=[[1.0, 0.4], [2.0, 0.1]],
        source_json="records/source.json",
        source_sha256="sourcesha256",
        source_dir="records",
        note="source-bound test row",
    )


def test_write_tex_renders_a_visible_final_parameter_manifest(
    tmp_path: Path,
) -> None:
    tex_path = tmp_path / "report.tex"

    write_tex(tex_path, [_manifest_row()], [], "powell-support")

    tex = tex_path.read_text(encoding="utf-8")
    visible_tex = "\n".join(
        line for line in tex.splitlines() if not line.lstrip().startswith("%")
    )
    manifest_heading = r"\section*{Parameter and provenance manifest}"
    assert manifest_heading in visible_tex
    assert visible_tex.rfind(manifest_heading) > visible_tex.rfind(
        r"\subsection*{Strong--strong}"
    )
    assert visible_tex.rfind(manifest_heading) < visible_tex.rfind(
        r"\end{document}"
    )
    for expected in (
        "Report scope",
        "Model scope",
        "drive\\_enabled=false",
        "snake\\_native\\_a1",
        "POWELL",
        "paper\\_i\\_effective\\_plateau\\_v1",
        "selected\\_prefix\\_k=2",
        "records/source.json",
        "sourcesha256",
        "records/input-report.json",
        "inputsha256",
        "source-bound",
        "unavailable",
    ):
        assert expected in visible_tex


def test_selected_prefix_uses_the_shared_paper_i_effective_plateau_policy() -> None:
    selected = selected_prefix_payload(
        {"source_json": ""},
        _role("snake"),
        "weak-weak",
        [[1.0, 0.40], [2.0, 0.10], [3.0, 0.11]],
    )

    assert selected["selection_policy"] == EFFECTIVE_PLATEAU_POLICY
    assert selected["selected_prefix_k"] == 2
    assert selected["abs_delta_e"] == 0.10


def test_generic_prefix_accounting_blocks_a_selected_depth_inside_a_batch() -> None:
    s_alg, status, meta = generic_prefix_s_alg(
        {"result": {"adapt_history": [{"batch_size": 2}]}},
        selected_k=1,
    )

    assert s_alg is None
    assert status == "generic_prefix_s_alg_blocked_batch_cut"
    assert "cuts through adaptive batch 0->2" in meta["reason"]


def test_generic_prefix_accounting_blocks_ambiguous_geo_immediate_repeat_depth() -> None:
    s_alg, status, meta = generic_prefix_s_alg(
        {
            "result": {
                "adapt_history": [
                    {"depth_after": 1},
                    {"depth_after": 1},
                ]
            }
        },
        selected_k=1,
    )

    assert s_alg is None
    assert status == "generic_prefix_s_alg_blocked_batch_cut"
    assert "multiple history rows" in meta["reason"]


def test_generic_prefix_accounting_blocks_missing_qngd_gradient_ledger() -> None:
    s_alg, status, meta = generic_prefix_s_alg(
        {
            "result": {
                "adapt_history": [
                    {
                        "depth_after": 1,
                        "qngd_metric_eval_count": 1,
                    }
                ]
            }
        },
        selected_k=1,
    )

    assert s_alg is None
    assert status == "generic_prefix_s_alg_blocked_missing_qngd_gradient_ledger"
    assert meta["qngd_metric_eval_count"] == 1.0


def test_generic_prefix_accounting_rejects_noncanonical_extra_work_component() -> None:
    s_alg, status, meta = generic_prefix_s_alg(
        {
            "result": {
                "adapt_history": [
                    {
                        "depth_after": 1,
                        "N_other_quantum": 1,
                    }
                ]
            }
        },
        selected_k=1,
    )

    assert s_alg is None
    assert status == "generic_prefix_s_alg_blocked_noncanonical_other_work"
    assert meta["N_other_quantum"] == 1.0
