#!/usr/bin/env python3
"""Assemble the Paper III results-consolidation Markdown from evidence JSONs.

Reads the committed evidence sets (frontier arms, oracle cross-check,
comparator arms, regime sweeps, child-granularity study, exchange evidence
and repair) and emits one deterministic Markdown document with the central
regime-by-arm table and all supporting tables. No numbers are typed by
hand: every cell is read from its evidence JSON, so the document can be
regenerated whenever a driver reruns. Reporting only; never feeds
controller decisions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DIAG = REPO_ROOT / "output/diagnostics"
DEFAULT_OUTPUT = (
    DIAG / "paper_iii_results_consolidation_20260818_v1/paper_iii_results_consolidation.md"
)

EVIDENCE = {
    "convention_sweep": DIAG
    / "paper_iii_paper_i_convention_sweep_20260818_v1/paper_i_convention_sweep.json",
    "exchange_repair": DIAG
    / "paper_iii_exchange_repair_20260818_v1/exchange_repair_summary.json",
    "comparators": DIAG / "paper_iii_cost_frontier_arms_20260818_v1/comparator_arms_summary.json",
    "oracle_agreement": DIAG / "paper_iii_cost_frontier_arms_20260818_v1/oracle_agreement_2q.json",
    "exchange_dimer": DIAG
    / "paper_iii_cost_frontier_arms_20260818_v1/exchange_maintenance_evidence.json",
    "nph3_sweep": DIAG / "paper_iii_regime_frontier_sweep_20260818_v1/regime_frontier_sweep.json",
    "child_granularity": DIAG
    / "paper_iii_child_granularity_20260818_v1/child_granularity_summary.json",
}


def _load(key: str) -> dict[str, Any]:
    path = EVIDENCE[key]
    if not path.is_file():
        raise FileNotFoundError(f"evidence file for {key!r} missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sci(value: Any) -> str:
    if value is None:
        return "--"
    return f"{float(value):.1e}"


def _cell(err: Any, cost: Any) -> str:
    if err is None:
        return "--"
    return f"{_sci(err)} @ {float(cost):.0f}"


def _frontier_endpoint(arm: dict[str, Any]) -> tuple[Any, Any]:
    rows = [row for row in arm.get("frontier", []) if row.get("abs_err_E1") is not None]
    if not rows:
        return None, None
    last = rows[-1]
    return last["abs_err_E1"], last["cum_2q"]


def build_markdown() -> str:
    convention = _load("convention_sweep")
    repair = _load("exchange_repair")
    comparators = _load("comparators")
    oracle = _load("oracle_agreement")
    exchange_dimer = _load("exchange_dimer")
    nph3 = _load("nph3_sweep")
    child = _load("child_granularity")

    lines: list[str] = []
    add = lines.append
    add("# Paper III results consolidation (2026-08-18)")
    add("")
    add(
        "Statevector diagnostics on the L=2 Hubbard--Holstein model at the Paper I "
        "regime conventions. Accuracy is the absolute first-excited-energy error "
        "against exact sector-restricted (1,1) references; cost is compiled "
        "two-qubit gate count (`two_qubit_only_v1`, Marrakesh graph-span oracle, "
        "cross-checked against full transpilation below). Cells read "
        "`error @ 2Q`. Every number is generated from the committed evidence "
        "JSONs by `pipelines/reporting/build_paper_iii_results_consolidation.py`."
    )
    add("")

    add("## Central table: six Paper I regimes, selection and exchange arms (budget 40)")
    add("")
    add(
        "| regime | u | g | nph | gap | manifold limit | input order | fixed class "
        "(complete) | geometry a=1 | + exchange (dominance) | + exchange (budgeted) |"
    )
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    for regime, record in convention["regimes"].items():
        arms = record["arms"]
        io_err, io_cost = _frontier_endpoint(arms["input_order"])
        a1_err, a1_cost = _frontier_endpoint(arms["geometry_alpha1"])
        fixed = arms["fixed_linear_response_complete"]
        repair_record = repair["regimes"].get(regime, {}).get("arms", {})
        dom = repair_record.get("dominance", {})
        bud = repair_record.get("budgeted_linear_class_parity", {})
        add(
            f"| {regime} | {record['u']} | {record['g_ep']:.4f} | {record['n_ph_max']} "
            f"| {record['exact_gap']:.4f} | {_sci(record['manifold_limit_abs_err_E1'])} "
            f"| {_cell(io_err, io_cost)} "
            f"| {_cell(fixed['abs_err_E1'], fixed['total_2q'])} "
            f"| {_cell(a1_err, a1_cost)} "
            f"| {_cell(dom.get('final_abs_err_E1'), dom.get('final_2q', 0))} "
            f"| {_cell(bud.get('final_abs_err_E1'), bud.get('final_2q', 0))} |"
        )
    add("")
    add(
        "The budgeted exchange envelope is the complete fixed-class cost of the "
        "same regime (cost parity); dominance-mode exchange never increases cost."
    )
    add("")

    add("## Dimer benchmark and cross-family comparators (weak-coupling demo point)")
    add("")
    ref = comparators["reference_full_basis_root0_energy"]
    add(f"Reference: full-158-basis q0 root-0 energy {ref:.9f} (stored 20260802 inputs).")
    add("")
    add("| arm | error @ 2Q | note |")
    add("|---|---|---|")
    dimer_runs = exchange_dimer["runs"]
    geo = dimer_runs["from_geometry_alpha1"]
    add(
        f"| geometry a=1 (k=20) | {_cell(abs(geo['initial']['root0_energy']-exchange_dimer['reference_root0']), geo['initial']['total_compiled_cost'])} | selection only |"
    )
    add(
        f"| geometry a=1 + exchange | {_cell(abs(geo['final']['root0_energy']-exchange_dimer['reference_root0']), geo['final']['total_compiled_cost'])} | {geo['committed_patch_count']} certified patches, fixed size |"
    )
    rescue = dimer_runs.get("from_cheapest_first_budget48")
    if rescue:
        add(
            f"| cheapest-first + budgeted exchange | {_cell(abs(rescue['final']['root0_energy']-exchange_dimer['reference_root0']), rescue['final']['total_compiled_cost'])} | budget 48; {rescue['committed_patch_count']} patch |"
        )
    for name, arm in comparators["fixed_class_arms"].items():
        add(
            f"| fixed class: {name} ({arm['class_size']} ops) | {_cell(arm['abs_err_vs_reference'], arm['total_2q_graph_span'])} | deterministic, complete |"
        )
    envelope = comparators["krylov_arm"]["best_per_cost_envelope"]
    if envelope:
        best = envelope[-1]
        add(
            f"| real-time Krylov (best) | {_cell(best['abs_err_vs_reference'], best['cum_2q_graph_span'])} | K={best['krylov_dimension']}, dt={best['dt']}; kicked source, Krylov-favoring costing |"
        )
    add("")

    add("## Compiled-cost oracle cross-check")
    add("")
    ratio = oracle["transpile_over_span_2q_ratio"]
    add(
        f"Graph-span vs full-transpile 2Q over the 158-element pool: Spearman "
        f"{oracle['spearman_rank_correlation']:.4f}, Pearson "
        f"{oracle['pearson_correlation']:.4f}, transpile/span ratio median "
        f"{ratio['median']:.2f} (range {ratio['min']:.2f}-{ratio['max']:.2f}), "
        f"zero-cost agreement {oracle['zero_cost_agreement']}/{oracle['elements']}. "
        "All five selection arms produce identical selections under either oracle."
    )
    add("")

    add("## Alpha sensitivity (nph3 shared-pool sweep, k=40 endpoints)")
    add("")
    add("| regime | a=0.5 | a=1 | a=2 | subtractive |")
    add("|---|---|---|---|---|")
    for regime, record in nph3["regimes"].items():
        cells = []
        for arm_name in (
            "geometry_alpha0.5",
            "geometry_alpha1",
            "geometry_alpha2",
            "geometry_subtractive",
        ):
            err, cost = _frontier_endpoint(record["arms"][arm_name])
            cells.append(_cell(err, cost))
        add(f"| {regime} | " + " | ".join(cells) + " |")
    add("")

    add("## Child-singleton granularity (Paper II atom coordinate)")
    add("")
    add("| regime | macro manifold limit | child manifold limit | child a=1 k=80 |")
    add("|---|---|---|---|")
    for regime, record in child["regimes"].items():
        k80 = next(
            (row for row in record["child_frontier_alpha1"] if row["budget"] == 80), None
        )
        add(
            f"| {regime} | {_sci(record['macro_manifold_abs_err_E1'])} "
            f"| {_sci(record['child_manifold_abs_err_E1'])} "
            f"| {_cell(k80['abs_err_E1'], k80['cum_2q']) if k80 else '--'} |"
        )
    add("")
    add(
        "The child span is numerically exact in every studied regime, so the "
        "macro-level manifold limits are macro-span limitations, not phonon "
        "truncation. Children break fermion-number conservation and therefore "
        "require the exact sector projector in the pencil. Macro granularity "
        "wins at low budget; child granularity wins beyond roughly 200 2Q."
    )
    add("")

    add("## Methodology corrections locked during results gathering")
    add("")
    add(
        "1. Sector references must come from the exact sector-restricted "
        "eigenproblem: expectation-based filtering silently dropped the u=8 "
        "spin triplet (degenerate with its S_z=+-1 partners across number "
        "sectors) and produced a spurious pool-failure reading."
    )
    add(
        "2. The weak_strong `nph3` manifold limit is macro-span limitation, "
        "not phonon-truncation error (child span is exact in the same "
        "encoded space)."
    )
    add(
        "3. Dominance vs budgeted exchange are one algorithm with one "
        "acceptance functional; the budget is an optional declared envelope "
        "and budgeted numbers must always be labeled with it."
    )
    add("")

    add("## Evidence inventory")
    add("")
    for key, path in EVIDENCE.items():
        add(f"- {key}: `{path.relative_to(REPO_ROOT)}`")
    add(
        "- drivers: `pipelines/exact_bench/paper_iii_qse_paper_i_convention_sweep.py`, "
        "`paper_iii_qse_exchange_repair.py`, `paper_iii_qse_comparator_arms.py`, "
        "`paper_iii_qse_regime_frontier_sweep.py`, `paper_iii_qse_child_granularity.py`; "
        "machinery in `pipelines/qse_spectra/` (`compiled_costs`, `record_selection`, "
        "`exchange_maintenance`)."
    )
    add("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    markdown = build_markdown()
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown + "\n", encoding="utf-8")
    print(f"output_md: {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
