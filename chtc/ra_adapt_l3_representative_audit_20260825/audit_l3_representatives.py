#!/usr/bin/env python3
"""Recount pure-Hubbard L=3 historical prefixes under one graph proxy.

This is an evidence-audit tool.  It does not alter selector behavior and it
does not treat the analytic proxy as a substitute for compiled Qiskit counts.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipelines.static_adapt.hh_backend_compile_oracle import (
    BackendCompileConfig,
    MarrakeshGraphSpanCostOracle,
)
from pipelines.static_adapt.ra_adapt.appendix_demos import (
    build_ra_adapt_appendix_pure_hubbard_problem,
)
from pipelines.static_adapt.ra_adapt.pools import (
    build_ra_adapt_appendix_pure_hubbard_sector_guarded_macro_pool,
)


MAIN_REPO = Path("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3")
OUT_DIR = Path(__file__).resolve().parent
EXACT_ENERGY = -2.1414935642714075
TARGET_ERROR = 1.0e-6
REFERENCE_PREPARATION_DEPTH = 1


@dataclass(frozen=True)
class SourceSpec:
    run_id: str
    family: str
    arm: str
    path: Path
    parser: str
    selector_cost: str
    refit_chart: str
    shortlist: str
    active_response: str
    powell_match: str
    evidence_status: str
    notes: str = ""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def corrected_estimator(receipt: dict[str, Any] | None) -> int | None:
    if not receipt:
        return None
    raw = receipt.get("cumulative_raw_occurrences", {}).get("components", {})
    unique = receipt.get("cumulative_unique_primitives", {}).get("components", {})
    required_raw = ("N_H_outer", "N_H_refit")
    required_unique = ("N_grad", "N_metric")
    if all(key in raw for key in required_raw) and all(
        key in unique for key in required_unique
    ):
        return int(
            raw["N_H_outer"]
            + raw["N_H_refit"]
            + unique["N_grad"]
            + unique["N_metric"]
        )
    executed = receipt.get("cumulative_executed_queries", {}).get("S_alg")
    return None if executed is None else int(executed)


def raw_estimator(receipt: dict[str, Any] | None) -> int | None:
    if not receipt:
        return None
    raw = receipt.get("cumulative_raw_occurrences", {}).get("total")
    if raw is None:
        raw = receipt.get("cumulative_executed_queries", {}).get("S_alg")
    return None if raw is None else int(raw)


def corrected_estimator_delta(receipt: dict[str, Any] | None) -> int | None:
    if not receipt:
        return None
    raw = receipt.get("raw_occurrence_delta", {}).get("components", {})
    unique = receipt.get("unique_primitive_delta", {}).get("components", {})
    if all(key in raw for key in ("N_H_outer", "N_H_refit")) and all(
        key in unique for key in ("N_grad", "N_metric")
    ):
        return int(
            raw["N_H_outer"]
            + raw["N_H_refit"]
            + unique["N_grad"]
            + unique["N_metric"]
        )
    return None


def raw_estimator_delta(receipt: dict[str, Any] | None) -> int | None:
    if not receipt:
        return None
    raw = receipt.get("raw_occurrence_delta", {}).get("total")
    if raw is None:
        raw = receipt.get("executed_query_delta", {}).get("S_alg")
    return None if raw is None else int(raw)


def parse_ra_current(spec: SourceSpec) -> list[dict[str, Any]]:
    payload = json.loads(spec.path.read_text())
    rows: list[dict[str, Any]] = []
    beam_occurrences: list[dict[str, Any]] | None = None
    beam_branch_ids: set[str] = set()
    if "beam" in spec.arm:
        ledger_path = spec.path.parent / "estimator_ledger.json"
        ledger_payload = json.loads(ledger_path.read_text())
        beam_occurrences = list(ledger_payload["ledger"]["occurrences"])
    for ordinal, item in enumerate(payload["adapt_vqe"]["history"], start=1):
        checkpoint = item["active_prefix_checkpoint"]
        labels = checkpoint.get("ordered_active_operator_labels")
        if labels is None:
            labels = checkpoint.get("accepted_operator_labels")
        if labels is None:
            raise KeyError(f"missing active labels in {spec.path} round {ordinal}")
        receipt = checkpoint.get("estimator_ledger_receipt")
        if receipt is None:
            receipt = checkpoint.get("estimator_prefix")
        if beam_occurrences is not None:
            branch_id = item.get("branch_id")
            if branch_id is None:
                raise KeyError(f"beam branch id missing in {spec.path} round {ordinal}")
            beam_branch_ids.add(str(branch_id))
            selected_occurrences = [
                occurrence
                for occurrence in beam_occurrences
                if occurrence.get("branch_id") is None
                or str(occurrence.get("branch_id")) in beam_branch_ids
            ]
            estimator_raw = len(selected_occurrences)
            hamiltonian_count = sum(
                1
                for occurrence in selected_occurrences
                if occurrence["component"] in {"N_H_outer", "N_H_refit"}
            )
            response_primitives = {
                (occurrence["component"], occurrence["primitive_id"])
                for occurrence in selected_occurrences
                if occurrence["component"] in {"N_grad", "N_metric"}
            }
            estimator_corrected = hamiltonian_count + len(response_primitives)
        else:
            estimator_corrected = corrected_estimator(receipt)
            estimator_raw = raw_estimator(receipt)
        energy = item.get("energy_after_opt")
        if energy is None:
            energy = checkpoint.get("accepted_energy")
        k = checkpoint.get("outer_iteration")
        if k is None:
            k = checkpoint.get("controller_round", ordinal)
        rows.append(
            {
                "k": int(k),
                "labels": [str(label) for label in labels],
                "energy": float(energy),
                "error": abs(float(energy) - EXACT_ENERGY),
                "estimator_corrected": estimator_corrected,
                "estimator_raw": estimator_raw,
                "estimator_scope": (
                    "selected_lineage" if "beam" in spec.arm else "all_executed"
                ),
            }
        )
    return rows


def parse_avqe_result(spec: SourceSpec) -> list[dict[str, Any]]:
    payload = json.loads(spec.path.read_text())
    rows: list[dict[str, Any]] = []
    for ordinal, item in enumerate(payload["result_payload"]["history"], start=1):
        checkpoint = item["active_prefix_checkpoint"]
        labels = checkpoint["accepted_operator_labels"]
        receipt = checkpoint.get("estimator_prefix")
        energy = item.get("energy_after", checkpoint.get("accepted_energy"))
        rows.append(
            {
                "k": int(item.get("controller_round", ordinal)),
                "labels": [str(label) for label in labels],
                "energy": float(energy),
                "error": abs(float(energy) - EXACT_ENERGY),
                "estimator_corrected": corrected_estimator(receipt),
                "estimator_raw": raw_estimator(receipt),
                "estimator_scope": "all_executed",
            }
        )
    return rows


def parse_ra_result(spec: SourceSpec) -> list[dict[str, Any]]:
    payload = json.loads(spec.path.read_text())
    trajectory = payload["run"]["accepted_trajectory"]
    replay = payload["scientific_receipts"]["controller_replay_evidence"]
    signed = replay["signed_controller_round_prefixes"]
    receipts = {
        int(row["controller_round"]): row["active_prefix_checkpoint"].get(
            "estimator_ledger_receipt"
        )
        for row in signed
    }
    beam_occurrences: list[dict[str, Any]] | None = None
    beam_branch_ids: set[str] = set()
    if "beam" in spec.arm:
        ledger_payload = json.loads((spec.path.parent / "estimator_ledger.json").read_text())
        beam_occurrences = list(ledger_payload["ledger"]["occurrences"])
    rows: list[dict[str, Any]] = []
    for item in trajectory:
        k = int(item["controller_round"])
        receipt = receipts.get(k)
        if beam_occurrences is not None:
            branch_id = None if receipt is None else receipt.get("branch_id")
            if branch_id is None:
                raise KeyError(f"beam branch id missing in {spec.path} round {k}")
            beam_branch_ids.add(str(branch_id))
            selected_occurrences = [
                occurrence
                for occurrence in beam_occurrences
                if occurrence.get("branch_id") is None
                or str(occurrence.get("branch_id")) in beam_branch_ids
            ]
            estimator_raw = len(selected_occurrences)
            hamiltonian_count = sum(
                1
                for occurrence in selected_occurrences
                if occurrence["component"] in {"N_H_outer", "N_H_refit"}
            )
            response_primitives = {
                (occurrence["component"], occurrence["primitive_id"])
                for occurrence in selected_occurrences
                if occurrence["component"] in {"N_grad", "N_metric"}
            }
            estimator_corrected = hamiltonian_count + len(response_primitives)
        else:
            estimator_corrected = corrected_estimator(receipt)
            estimator_raw = raw_estimator(receipt)
        energy = float(item["energy"])
        rows.append(
            {
                "k": k,
                "labels": [str(label) for label in item["operators"]],
                "energy": energy,
                "error": abs(energy - EXACT_ENERGY),
                "estimator_corrected": estimator_corrected,
                "estimator_raw": estimator_raw,
                "estimator_scope": (
                    "selected_lineage" if beam_occurrences is not None else "all_executed"
                ),
            }
        )
    return rows


def source_specs() -> list[SourceSpec]:
    specs: list[SourceSpec] = []

    historical = {
        "v6": ("native", "historical_unmatched_powell"),
        "v7": ("native", "historical_unmatched_powell"),
        "v8": ("native", "historical_unmatched_powell"),
        "v9": ("native", "historical_unmatched_powell"),
        "v10_whitened": ("fubini_study_whitened", "historical_unmatched_powell"),
    }
    for version, (chart, status) in historical.items():
        campaign = ROOT / "output" / "local_runs" / (
            f"ra_adapt_appendix_demos_l3_20260820_{version}"
        )
        if not campaign.exists():
            continue
        for current in sorted(campaign.glob("cells/*/observations/current.json")):
            cell = current.parents[1].name
            arm = cell.removeprefix("l3_").rsplit("_h", 1)[0]
            specs.append(
                SourceSpec(
                    run_id=f"historical_{version}_{arm}",
                    family=f"historical_{version}",
                    arm=arm,
                    path=current,
                    parser="ra_current",
                    selector_cost="marrakesh_graph_span_five_coordinate",
                    refit_chart=chart,
                    shortlist="fixed_24_12_1_phase0_off",
                    active_response="measured_residual",
                    powell_match="unmatched_RA_loose_vs_AVQE_tight",
                    evidence_status=status,
                    notes="Historical route; useful diagnostically, not a matched AVQE claim.",
                )
            )

    current_root = ROOT / "chtc" / "ra_adapt_l3_manuscript_cost_20260824" / "results"
    current_runs = {
        "baseline_k12": ("append", "claim_ready"),
        "batching_k12": ("batch", "claim_ready"),
        "beam_escalated_k12": ("beam", "claim_ready"),
        "prune_confirmation/prune_canonical_current_k12": (
            "prune_protect1",
            "confirmed_alternative",
        ),
        "prune_confirmation/prune_protect0_confirm_k12": (
            "prune_protect0",
            "claim_ready",
        ),
    }
    for relative, (arm, status) in current_runs.items():
        current = current_root / relative / "current.json"
        if current.exists():
            specs.append(
                SourceSpec(
                    run_id=f"current_qiskit_{arm}",
                    family="current_qiskit_tight",
                    arm=arm,
                    path=current,
                    parser="ra_current",
                    selector_cost="exact_qiskit_marginal",
                    refit_chart="fubini_study_whitened",
                    shortlist="phase0_cap24_then_adaptive_gamma0.9",
                    active_response="measured_residual_marginal_gain",
                    powell_match="matched_tight",
                    evidence_status=status,
                )
            )

    for current in sorted((current_root / "prune_sweep").glob("*/current.json")):
        arm = current.parent.name
        specs.append(
            SourceSpec(
                run_id=f"prune_sweep_{arm}",
                family="current_qiskit_tight_prune_sweep",
                arm=arm,
                path=current,
                parser="ra_current",
                selector_cost="exact_qiskit_marginal",
                refit_chart="fubini_study_whitened",
                shortlist="phase0_cap24_then_adaptive_gamma0.9",
                active_response="measured_residual_marginal_gain",
                powell_match="matched_tight",
                evidence_status="exploratory_posthoc_sweep",
                notes="Exploratory pruning sweep; cannot be selected as headline evidence without locked confirmation.",
            )
        )

    avqe = current_root / "avqe_k12" / "result.json"
    specs.append(
        SourceSpec(
            run_id="current_avqe_tight",
            family="current_qiskit_tight",
            arm="avqe",
            path=avqe,
            parser="avqe_result",
            selector_cost="none_gradient_only",
            refit_chart="native",
            shortlist="flat_pool_all24",
            active_response="not_applicable",
            powell_match="matched_tight",
            evidence_status="claim_ready",
        )
    )

    repaired_root = ROOT / "chtc" / "ra_adapt_l3_repaired_20260823" / "results"
    repaired_runs = {
        "baseline_k12": (
            "append_native_adaptive",
            "phase0_cap24_then_adaptive_gamma0.9",
            "matched_tight",
            "matched_archived_needs_source_lock",
        ),
        "ra_fixed_tight_k12": (
            "append_native_fixed",
            "fixed_24_12_1_phase0_off",
            "matched_tight",
            "matched_diagnostic",
        ),
        "ra_adaptive_tight_k12": (
            "append_native_adaptive_repeat",
            "phase0_cap24_then_adaptive_gamma0.9",
            "matched_tight",
            "matched_diagnostic",
        ),
        "ra_fixed_loose_k12": (
            "append_native_fixed_loose",
            "fixed_24_12_1_phase0_off",
            "matched_loose",
            "matched_diagnostic",
        ),
        "ra_adaptive_loose_k12": (
            "append_native_adaptive_loose",
            "phase0_cap24_then_adaptive_gamma0.9",
            "matched_loose",
            "matched_diagnostic",
        ),
        "batching_k12": (
            "batch_native_adaptive",
            "phase0_cap24_then_adaptive_gamma0.9",
            "matched_tight",
            "matched_archived_needs_source_lock",
        ),
        "beam_escalated_k12": (
            "beam_native_adaptive",
            "phase0_cap24_then_adaptive_gamma0.9",
            "matched_tight",
            "matched_archived_needs_source_lock",
        ),
        "metric_ablation_gated_k12": (
            "prune_native_adaptive",
            "phase0_cap24_then_adaptive_gamma0.9",
            "matched_tight",
            "matched_archived_needs_source_lock",
        ),
        "prune_no_screen_h15": (
            "prune_native_no_screen",
            "phase0_cap24_then_adaptive_gamma0.9",
            "matched_tight",
            "exploratory_posthoc_sweep",
        ),
    }
    for directory, (arm, shortlist, match, status) in repaired_runs.items():
        result = repaired_root / directory / "result.json"
        if not result.exists():
            continue
        specs.append(
            SourceSpec(
                run_id=f"repaired_{arm}",
                family="repaired_exact_qiskit_native",
                arm=arm,
                path=result,
                parser="ra_result",
                selector_cost="exact_qiskit_marginal",
                refit_chart="native_with_metric_informed_directions",
                shortlist=shortlist,
                active_response="measured_residual_marginal_gain",
                powell_match=match,
                evidence_status=status,
                notes=(
                    "Archived repaired route; result receipts are complete, but the source map "
                    "states that its three supporting code fixes were uncommitted."
                ),
            )
        )

    proxy = (
        MAIN_REPO
        / "output/local_runs/paper_i_l3_proxy_adaptive_whitened_20260824"
        / "proxy_adaptive_whitened_k12/current.json"
    )
    if proxy.exists():
        specs.append(
            SourceSpec(
                run_id="matched_proxy_adaptive_whitened_append",
                family="matched_proxy_tight",
                arm="append",
                path=proxy,
                parser="ra_current",
                selector_cost="marrakesh_graph_span_three_coordinate",
                refit_chart="fubini_study_whitened",
                shortlist="phase0_cap24_then_adaptive_gamma0.9",
                active_response="measured_residual_marginal_gain",
                powell_match="matched_tight",
                evidence_status="diagnostic_needs_locked_promotion",
            )
        )

    response_root = (
        MAIN_REPO / "output/local_runs/paper_i_l3_response_gain_matrix_20260824"
    )
    response_cells = {
        "measured_joint_k12": (
            "append_measured_joint",
            "measured_residual_joint_total",
        ),
        "stationary_joint_equiv_marginal_k12": (
            "append_stationary",
            "stationary_source_joint_equals_marginal",
        ),
    }
    for directory, (arm, response) in response_cells.items():
        current = response_root / directory / "current.json"
        if current.exists():
            specs.append(
                SourceSpec(
                    run_id=f"response_gain_{arm}",
                    family="response_gain_diagnostic_tight",
                    arm=arm,
                    path=current,
                    parser="ra_current",
                    selector_cost="exact_qiskit_marginal",
                    refit_chart="fubini_study_whitened",
                    shortlist="phase0_cap24_then_adaptive_gamma0.9",
                    active_response=response,
                    powell_match="matched_tight",
                    evidence_status="diagnostic_needs_locked_promotion",
                )
            )

    joint_root = (
        MAIN_REPO
        / "output/local_runs/paper_i_l3_b9_joint_controls_20260825"
    )
    joint_cells = {
        "baseline_joint_k12": "append",
        "batch_joint_k12": "batch",
        "beam_joint_k12": "beam",
        "prune_joint_tight_nomination_k12/metric_ablation_gated_k12": "prune",
    }
    for directory, arm in joint_cells.items():
        result = joint_root / directory / "result.json"
        if not result.exists():
            continue
        specs.append(
            SourceSpec(
                run_id=f"b9_joint_{arm}",
                family="b9_matched_full_joint_tight",
                arm=arm,
                path=result,
                parser="ra_result",
                selector_cost="exact_qiskit_marginal",
                refit_chart="native_with_metric_informed_directions",
                shortlist="phase0_cap24_then_adaptive_gamma0.9",
                active_response="measured_residual_joint_total",
                powell_match="matched_tight",
                evidence_status="claim_ready",
                notes=(
                    "B9-matched full joint Phase-III gain; all other L=3 "
                    "route coordinates are held fixed."
                ),
            )
        )

    stationary_joint_root = (
        MAIN_REPO
        / "output/local_runs/paper_i_l3_b9_stationary_joint_controls_20260825"
    )
    stationary_joint_cells = {
        "baseline_stationary_joint_k12": "append",
        "batch_stationary_joint_k12": "batch",
        "beam_stationary_joint_k12": "beam",
        "prune_stationary_joint_tight_k12": "prune",
    }
    for directory, arm in stationary_joint_cells.items():
        result = stationary_joint_root / directory / "result.json"
        if not result.exists():
            continue
        specs.append(
            SourceSpec(
                run_id=f"b9_stationary_joint_{arm}",
                family="b9_matched_stationary_full_joint_tight",
                arm=arm,
                path=result,
                parser="ra_result",
                selector_cost="exact_qiskit_marginal",
                refit_chart="native_with_metric_informed_directions",
                shortlist="phase0_cap24_then_adaptive_gamma0.9",
                active_response="stationary_source_joint_total",
                powell_match="matched_tight",
                evidence_status="claim_ready",
                notes=(
                    "B9-matched stationary-source full joint Phase-III gain; "
                    "all other L=3 route coordinates are held fixed."
                ),
            )
        )

    return specs


def dominates(left: dict[str, Any], right: dict[str, Any], axes: Iterable[str]) -> bool:
    values = [(left[axis], right[axis]) for axis in axes]
    if any(a is None or b is None for a, b in values):
        return False
    return all(a <= b for a, b in values) and any(a < b for a, b in values)


def pareto(rows: list[dict[str, Any]], axes: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if not any(other is not row and dominates(other, row, axes) for other in rows)
    ]


def main() -> None:
    problem = build_ra_adapt_appendix_pure_hubbard_problem(num_sites=3)
    inventory = build_ra_adapt_appendix_pure_hubbard_sector_guarded_macro_pool(problem)
    candidates = {candidate.label: candidate for candidate in inventory.candidates}
    if len(candidates) != 24:
        raise AssertionError(f"expected 24 candidates, found {len(candidates)}")
    oracle = MarrakeshGraphSpanCostOracle(
        config=BackendCompileConfig(requested_backend_name="FakeMarrakesh"),
        num_qubits=6,
    )

    def proxy(labels: list[str]) -> dict[str, float | int]:
        missing = [label for label in labels if label not in candidates]
        if missing:
            raise KeyError(f"labels absent from canonical L=3 pool: {missing}")
        terms = [candidates[label].term for label in labels]
        absolute = oracle._absolute_cost_for_ops(terms)
        total_depth = REFERENCE_PREPARATION_DEPTH
        term_count = 0
        for term in terms:
            for row in oracle._candidate_cost_rows(term):
                if int(row.get("support_size", 0)) <= 0:
                    continue
                label = str(row["label"]).lower()
                basis_depth = 2 if "y" in label else (1 if "x" in label else 0)
                total_depth += int(2 * int(row["span_edges"]) + 2 * basis_depth + 1)
                term_count += 1
        c2q = float(absolute["absolute_c_hat_2q"])
        d2q = float(absolute["absolute_c_hat_d"])
        dc = float(total_depth)
        return {
            "proxy_N2q": c2q,
            "proxy_D2q_span": d2q,
            "proxy_Dc_template": dc,
            "proxy_N1q_template": float(absolute["absolute_c_hat_1q"]),
            "proxy_theta": float(absolute["absolute_theta_count"]),
            "proxy_pauli_term_count": int(term_count),
            "proxy_K_030_030_025": float(0.30 * c2q + 0.30 * d2q + 0.25 * dc),
        }

    specs = source_specs()
    all_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    for spec in specs:
        if not spec.path.exists():
            raise FileNotFoundError(spec.path)
        if spec.parser == "ra_current":
            parser = parse_ra_current
        elif spec.parser == "ra_result":
            parser = parse_ra_result
        else:
            parser = parse_avqe_result
        prefixes = parser(spec)
        trajectory = [
            {"k": row["k"], "labels": row["labels"], "energy": row["energy"]}
            for row in prefixes
        ]
        first_target = next(
            (row["k"] for row in prefixes if row["error"] <= TARGET_ERROR), None
        )
        inventory_rows.append(
            {
                **asdict(spec),
                "path": str(spec.path),
                "source_sha256": sha256_path(spec.path),
                "trajectory_sha256": canonical_sha256(trajectory),
                "prefix_count": len(prefixes),
                "terminal_k": prefixes[-1]["k"],
                "first_target_k": first_target,
            }
        )
        for row in prefixes:
            all_rows.append(
                {
                    "run_id": spec.run_id,
                    "family": spec.family,
                    "arm": spec.arm,
                    "selector_cost": spec.selector_cost,
                    "refit_chart": spec.refit_chart,
                    "shortlist": spec.shortlist,
                    "active_response": spec.active_response,
                    "powell_match": spec.powell_match,
                    "evidence_status": spec.evidence_status,
                    "source_path": str(spec.path),
                    **row,
                    **proxy(row["labels"]),
                }
            )

    by_key = {(row["run_id"], row["k"]): row for row in all_rows}
    expected = {
        ("historical_v10_whitened_baseline", 12): (114.0, 118.0, 247.0),
        ("current_avqe_tight", 12): (344.0, 344.0, 615.0),
    }
    for key, values in expected.items():
        row = by_key[key]
        observed = (
            row["proxy_N2q"],
            row["proxy_D2q_span"],
            row["proxy_Dc_template"],
        )
        if observed != values:
            raise AssertionError(f"proxy validation failed for {key}: {observed}")

    target_rows = [row for row in all_rows if row["error"] <= TARGET_ERROR]
    matched_rows = [
        row
        for row in target_rows
        if row["powell_match"] == "matched_tight"
        and row["evidence_status"] != "exploratory_posthoc_sweep"
    ]
    first_target_rows: list[dict[str, Any]] = []
    for run_id in sorted({row["run_id"] for row in matched_rows}):
        candidates_for_run = [row for row in matched_rows if row["run_id"] == run_id]
        first_target_rows.append(min(candidates_for_run, key=lambda row: row["k"]))

    axes = (
        "proxy_N2q",
        "proxy_D2q_span",
        "proxy_Dc_template",
        "estimator_corrected",
    )
    frontier_first_target = pareto(first_target_rows, axes)
    frontier_all_target = pareto(matched_rows, axes)
    k9_rows = [row for row in matched_rows if row["k"] == 9]
    frontier_k9 = pareto(k9_rows, axes)
    k12_rows = [row for row in matched_rows if row["k"] == 12]
    frontier_k12 = pareto(k12_rows, axes)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "l3_run_inventory.json").write_text(
        json.dumps(inventory_rows, indent=2, sort_keys=True) + "\n"
    )
    columns = [
        "run_id",
        "family",
        "arm",
        "k",
        "energy",
        "error",
        "proxy_N2q",
        "proxy_D2q_span",
        "proxy_Dc_template",
        "proxy_N1q_template",
        "proxy_theta",
        "proxy_pauli_term_count",
        "proxy_K_030_030_025",
        "estimator_corrected",
        "estimator_raw",
        "estimator_scope",
        "selector_cost",
        "refit_chart",
        "shortlist",
        "active_response",
        "powell_match",
        "evidence_status",
        "source_path",
    ]
    with (OUT_DIR / "l3_prefix_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    def compact(row: dict[str, Any]) -> dict[str, Any]:
        return {key: row.get(key) for key in columns if key != "source_path"}

    frontiers = {
        "schema": "paper_i_l3_graph_proxy_frontiers_v1",
        "target_error": TARGET_ERROR,
        "axes": list(axes),
        "eligibility": (
            "matched-tight runs, excluding unconfirmed post-hoc pruning sweeps; "
            "diagnostic protocol ablations remain labeled as needing promotion"
        ),
        "first_target_prefix_frontier": [compact(row) for row in frontier_first_target],
        "all_target_satisfying_prefix_frontier": [compact(row) for row in frontier_all_target],
        "k9_frontier": [compact(row) for row in frontier_k9],
        "k12_frontier": [compact(row) for row in frontier_k12],
    }
    (OUT_DIR / "l3_pareto_frontiers.json").write_text(
        json.dumps(frontiers, indent=2, sort_keys=True) + "\n"
    )

    print(
        json.dumps(
            {
                "sources": len(specs),
                "prefixes": len(all_rows),
                "matched_target_prefixes": len(matched_rows),
                "first_target_frontier": [
                    [row["run_id"], row["k"]] for row in frontier_first_target
                ],
                "all_target_frontier": [
                    [row["run_id"], row["k"]] for row in frontier_all_target
                ],
                "k9_frontier": [[row["run_id"], row["k"]] for row in frontier_k9],
                "k12_frontier": [[row["run_id"], row["k"]] for row in frontier_k12],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
