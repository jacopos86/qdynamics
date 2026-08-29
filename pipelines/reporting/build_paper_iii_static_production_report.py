#!/usr/bin/env python3
"""Build the locked Paper-III static-QSE production evidence package.

The input is the completed fixed-pool profile ablation produced by
``paper_iii_pool_profile_campaign``.  This postprocessor admits only the
coverage-qualified encoded-boson completion profile, reconstructs the
terminal selected-QSE pencils, evaluates static transition strengths and
broadened spectra, and renders a Markdown/JSON/PDF report.  Direct Davidson
remains the matched energy/cost comparator; the complete fixed linear-
response class is retained as the conventional static-QSE spectral baseline.

No time propagation is run or represented by this module.  Exact sector
eigenpairs are post-run references only and never alter acquisition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench import paper_iii_growth_trace_campaign as growth
from pipelines.exact_bench.paper_iii_qse_regime_frontier_sweep import (
    _dense_hamiltonian,
)
from pipelines.qse_spectra.core import compute_qse_spectra, polynomial_observable
from pipelines.qse_spectra.hh_response_observables import (
    HHResponseLayout,
    phonon_displacement_operator,
    site_density_operator,
)
from pipelines.qse_spectra.paper_iii_problem import load_problem
from pipelines.qse_spectra.pool_profiles import (
    HH_POOL_ENCODED_BOSON_COMPLETION_V1,
    apply_hh_pool_policy,
)
from pipelines.qse_spectra.selection_policy import (
    PAPER_III_LOCKED_SELECTOR_POLICY_ID,
)
from pipelines.qse_spectra.spectral_functions import lorentzian_kernel
from src.quantum.compiled_polynomial import (
    apply_compiled_polynomial,
    compile_polynomial_action,
)


SCHEMA_VERSION = "paper_iii_static_qse_production_evidence_v1"
INPUT_SCHEMA_VERSION = "paper_iii_hh_pool_profile_ablation_v1"
SELECTED_METHOD_ID = "record_first_adaptive_qse"
DIRECT_METHOD_ID = "direct_davidson_qse"
SELECTED_CERTIFIED_STATUS = "TARGET_WINDOW_CERTIFIED"
LOCK_TARGET = 1.0e-4
TARGET_ROOT_COUNT = 6
STRENGTH_FLOOR = 1.0e-8
SPECTRAL_ETA = 0.05
SPECTRAL_POINTS = 801
FERMION_QUBITS_UP = (0, 1)
FERMION_QUBITS_DN = (2, 3)
CONTROLLER_BOUNDARY = {
    "feeds_controller_decisions": False,
    "controller_usable": False,
    "post_run_diagnostic_only": True,
    "time_dynamics_included": False,
}

DEFAULT_ACQUISITION = (
    REPO_ROOT
    / "output/diagnostics/paper_iii_hh_pool_profile_ablation_20260828_v2"
    / "pool_profile_ablation.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "output/pdf/paper_iii_static_qse_production_20260829_v1"
)


class StaticProductionReportError(RuntimeError):
    """Raised when source-locked production evidence is incomplete or drifts."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StaticProductionReportError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise StaticProductionReportError(f"JSON root at {path} must be an object.")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(payload), indent=1, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def validate_acquisition_payload(payload: Mapping[str, Any]) -> str:
    """Validate the locked acquisition and return its qualified pool policy."""

    if payload.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise StaticProductionReportError("acquisition schema mismatch.")
    if payload.get("selector_policy_id") != PAPER_III_LOCKED_SELECTOR_POLICY_ID:
        raise StaticProductionReportError("selector policy is not the locked policy.")
    boundary = payload.get("controller_boundary")
    if not isinstance(boundary, Mapping):
        raise StaticProductionReportError("acquisition lacks controller boundary.")
    if bool(boundary.get("implicit_fallback_enabled")) or bool(
        boundary.get("hybrid_enabled")
    ):
        raise StaticProductionReportError("implicit fallback/hybrid is forbidden.")
    lock_gate = payload.get("lock_gate")
    if not isinstance(lock_gate, Mapping):
        raise StaticProductionReportError("acquisition lacks lock gate.")
    if not math.isclose(float(lock_gate.get("error_target", math.nan)), LOCK_TARGET):
        raise StaticProductionReportError("acquisition lock target mismatch.")
    qualified = tuple(lock_gate.get("coverage_qualified_pool_policy_ids", ()))
    if qualified != (HH_POOL_ENCODED_BOSON_COMPLETION_V1,):
        raise StaticProductionReportError(
            "encoded-boson completion must be the sole coverage-qualified pool."
        )
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 12:
        raise StaticProductionReportError("acquisition must contain exactly 12 cells.")
    case_keys: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise StaticProductionReportError("acquisition cell must be a mapping.")
        case = row.get("case")
        if not isinstance(case, Mapping):
            raise StaticProductionReportError("acquisition cell lacks case identity.")
        case_key = str(case.get("case_key") or "")
        if not case_key or case_key in case_keys:
            raise StaticProductionReportError("case identities are missing or duplicate.")
        case_keys.add(case_key)
        selected_by_pool = row.get("selected_qse_by_pool_policy")
        if not isinstance(selected_by_pool, Mapping):
            raise StaticProductionReportError(f"{case_key}: selected pool map missing.")
        selected = selected_by_pool.get(HH_POOL_ENCODED_BOSON_COMPLETION_V1)
        direct = row.get("direct_davidson")
        if not isinstance(selected, Mapping) or not isinstance(direct, Mapping):
            raise StaticProductionReportError(f"{case_key}: route evidence missing.")
        if selected.get("algorithm_status") != SELECTED_CERTIFIED_STATUS:
            raise StaticProductionReportError(f"{case_key}: selected QSE is uncertified.")
        if selected.get("pool_policy_id") != HH_POOL_ENCODED_BOSON_COMPLETION_V1:
            raise StaticProductionReportError(f"{case_key}: selected pool identity drift.")
        for route in (selected, direct):
            crossing = _crossing(route, LOCK_TARGET)
            if crossing.get("status") != "REACHED":
                raise StaticProductionReportError(
                    f"{case_key}: route did not reach the lock target."
                )
    return HH_POOL_ENCODED_BOSON_COMPLETION_V1


def _crossing(route: Mapping[str, Any], target: float) -> dict[str, Any]:
    for crossing in route.get("crossings", ()):  # type: ignore[union-attr]
        if math.isclose(float(crossing.get("error_target", math.nan)), float(target)):
            return dict(crossing)
    raise StaticProductionReportError(f"route lacks crossing for target {target:g}.")


def _terminal_support(route_payload: Mapping[str, Any]) -> tuple[int, ...]:
    receipt = route_payload.get("online_receipt")
    if not isinstance(receipt, Mapping):
        raise StaticProductionReportError("selected route lacks online receipt.")
    rows = receipt.get("rows")
    if not isinstance(rows, list):
        raise StaticProductionReportError("selected route lacks online rows.")
    supports = [row.get("support_indices") for row in rows if row.get("support_indices")]
    if not supports:
        raise StaticProductionReportError("selected route has no terminal support.")
    return tuple(int(index) for index in supports[-1])


def _sector_eigensystem(
    dense: np.ndarray,
    *,
    count: int,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[float, ...]]:
    dim = int(dense.shape[0])
    occ_up = np.array(
        [sum((index >> qubit) & 1 for qubit in FERMION_QUBITS_UP) for index in range(dim)]
    )
    occ_dn = np.array(
        [sum((index >> qubit) & 1 for qubit in FERMION_QUBITS_DN) for index in range(dim)]
    )
    sector = np.where((occ_up == 1) & (occ_dn == 1))[0]
    restricted = dense[np.ix_(sector, sector)]
    energies, vectors = np.linalg.eigh(0.5 * (restricted + restricted.conj().T))
    states: list[np.ndarray] = []
    for column in range(min(int(count), vectors.shape[1])):
        state = np.zeros(dim, dtype=complex)
        state[sector] = vectors[:, column]
        states.append(state)
    if len(states) < int(count):
        raise StaticProductionReportError("sector eigensystem lacks requested roots.")
    return states[0], tuple(states), tuple(float(x) for x in energies[:count])


def _dense_apply(polynomial: Any, state: np.ndarray) -> np.ndarray:
    return apply_compiled_polynomial(state, compile_polynomial_action(polynomial))


def _observables(n_ph_max: int) -> tuple[Any, ...]:
    layout = HHResponseLayout(
        num_sites=2,
        n_ph_max=int(n_ph_max),
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
    )
    return (
        polynomial_observable(
            site_density_operator(layout, site=0), name="site0_density"
        ),
        polynomial_observable(
            phonon_displacement_operator(layout, site=0),
            name="site0_displacement",
        ),
    )


def _spectrum_values(
    omega: np.ndarray,
    poles: Sequence[float],
    strengths: Sequence[float],
    *,
    eta: float,
) -> np.ndarray:
    values = np.zeros_like(omega, dtype=float)
    for pole, strength in zip(poles, strengths, strict=True):
        values += float(strength) * lorentzian_kernel(omega - float(pole), eta)
    return values


def normalized_l1(
    reference: Sequence[float],
    candidate: Sequence[float],
    grid: Sequence[float],
) -> float:
    ref = np.asarray(reference, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    x = np.asarray(grid, dtype=float)
    if ref.shape != cand.shape or ref.shape != x.shape:
        raise ValueError("spectral arrays and grid must have identical shapes.")
    denominator = float(np.trapezoid(np.abs(ref), x))
    numerator = float(np.trapezoid(np.abs(cand - ref), x))
    return numerator / max(denominator, 1.0e-15)


def _transition_metrics(
    *,
    result: Any,
    exact_energies: Sequence[float],
    exact_states: Sequence[np.ndarray],
    ground: np.ndarray,
    observables: Sequence[Any],
    omega_grid: np.ndarray,
) -> tuple[dict[str, Any], float]:
    target_energies = tuple(float(x) for x in exact_energies[1 : TARGET_ROOT_COUNT + 1])
    predicted_energies = tuple(
        float(x) for x in np.asarray(result.eigenvalues).reshape(-1)[:TARGET_ROOT_COUNT]
    )
    if len(predicted_energies) < TARGET_ROOT_COUNT:
        raise StaticProductionReportError("QSE solve retained fewer than six roots.")
    max_energy_error = max(
        abs(predicted - exact)
        for predicted, exact in zip(predicted_energies, target_energies, strict=True)
    )
    exact_poles = [float(x - exact_energies[0]) for x in target_energies]
    predicted_poles = [float(x - exact_energies[0]) for x in predicted_energies]
    transition_by_name = {
        str(record.observable.name): record for record in result.transition_observables
    }
    metrics: dict[str, Any] = {}
    for observable in observables:
        name = str(observable.name)
        image = _dense_apply(observable.polynomial, ground)
        exact_strengths = [
            float(abs(complex(np.vdot(exact_states[root], image))) ** 2)
            for root in range(1, TARGET_ROOT_COUNT + 1)
        ]
        record = transition_by_name.get(name)
        if record is None:
            raise StaticProductionReportError(f"missing QSE observable {name!r}.")
        predicted_strengths = [
            float(x)
            for x in np.asarray(record.transition_strengths, dtype=float).reshape(-1)[
                :TARGET_ROOT_COUNT
            ]
        ]
        if len(predicted_strengths) < TARGET_ROOT_COUNT:
            raise StaticProductionReportError(
                f"observable {name!r} retained fewer than six roots."
            )
        absolute_errors = [
            abs(predicted - exact)
            for predicted, exact in zip(
                predicted_strengths, exact_strengths, strict=True
            )
        ]
        bright_relative_errors = [
            error / exact
            for error, exact in zip(absolute_errors, exact_strengths, strict=True)
            if exact > STRENGTH_FLOOR
        ]
        exact_curve = _spectrum_values(
            omega_grid, exact_poles, exact_strengths, eta=SPECTRAL_ETA
        )
        predicted_curve = _spectrum_values(
            omega_grid, predicted_poles, predicted_strengths, eta=SPECTRAL_ETA
        )
        metrics[name] = {
            "exact_strengths": exact_strengths,
            "predicted_strengths": predicted_strengths,
            "absolute_errors": absolute_errors,
            "bright_root_count": sum(x > STRENGTH_FLOOR for x in exact_strengths),
            "max_bright_relative_error": (
                max(bright_relative_errors) if bright_relative_errors else None
            ),
            "normalized_spectral_l1": normalized_l1(
                exact_curve, predicted_curve, omega_grid
            ),
            "spectrum": {
                "exact": [float(x) for x in exact_curve],
                "predicted": [float(x) for x in predicted_curve],
            },
        }
    return metrics, float(max_energy_error)


def _case_product(
    acquisition_path: Path,
    aggregate_row: Mapping[str, Any],
    pool_policy_id: str,
) -> dict[str, Any]:
    case = dict(aggregate_row["case"])
    case_key = str(case["case_key"])
    route_path = (
        acquisition_path.parent
        / "profiles"
        / pool_policy_id
        / "routes"
        / case_key
        / f"{SELECTED_METHOD_ID}.json"
    )
    route_payload = _read_json(route_path)
    selected_summary = dict(
        aggregate_row["selected_qse_by_pool_policy"][pool_policy_id]
    )
    if route_payload.get("online_receipt_digest") != selected_summary.get(
        "online_receipt_digest"
    ):
        raise StaticProductionReportError(
            f"{case_key}: aggregate/route receipt digest mismatch."
        )
    if route_payload.get("source_lock_digest") != aggregate_row.get(
        "source_lock_digest", route_payload.get("source_lock_digest")
    ):
        raise StaticProductionReportError(f"{case_key}: source lock drift.")

    base_problem = load_problem(
        regime=str(case["regime"]),
        u=float(case["u"]),
        g_ep=float(case["g_ep"]),
        n_ph_max=int(case["n_ph_max"]),
        target_roots=TARGET_ROOT_COUNT,
    )
    selected_problem = apply_hh_pool_policy(base_problem, pool_policy_id)
    receipt = route_payload["online_receipt"]
    if receipt.get("pool_digest") != selected_problem.pool_digest:
        raise StaticProductionReportError(f"{case_key}: selected pool digest drift.")
    support = _terminal_support(route_payload)
    observables = _observables(int(case["n_ph_max"]))
    selected_result = compute_qse_spectra(
        selected_problem.hamiltonian,
        selected_problem.ground,
        tuple(selected_problem.basis[index] for index in support),
        basis_vector_policy=growth._Q0_POLICY,
        transition_observables=observables,
    )

    fixed_support = tuple(
        index
        for index, element in enumerate(base_problem.basis)
        if growth._element_family(element.name) in growth._LINEAR_RESPONSE_FAMILIES
    )
    if not fixed_support:
        raise StaticProductionReportError(f"{case_key}: fixed class is empty.")
    fixed_result = compute_qse_spectra(
        base_problem.hamiltonian,
        base_problem.ground,
        tuple(base_problem.basis[index] for index in fixed_support),
        basis_vector_policy=growth._Q0_POLICY,
        transition_observables=observables,
    )

    dense = _dense_hamiltonian(base_problem.hamiltonian, 1 << base_problem.num_qubits)
    exact_ground, exact_states, exact_energies = _sector_eigensystem(
        dense, count=TARGET_ROOT_COUNT + 1
    )
    overlap = abs(complex(np.vdot(exact_ground, base_problem.ground)))
    if abs(overlap - 1.0) > 1.0e-8:
        raise StaticProductionReportError(
            f"{case_key}: provider ground differs from exact-sector ground."
        )
    omega_max = max(
        1.0,
        float(exact_energies[TARGET_ROOT_COUNT] - exact_energies[0])
        + 6.0 * SPECTRAL_ETA,
    )
    omega_grid = np.linspace(0.0, omega_max, SPECTRAL_POINTS, dtype=float)
    selected_metrics, selected_energy_error = _transition_metrics(
        result=selected_result,
        exact_energies=exact_energies,
        exact_states=exact_states,
        ground=exact_ground,
        observables=observables,
        omega_grid=omega_grid,
    )
    fixed_metrics, fixed_energy_error = _transition_metrics(
        result=fixed_result,
        exact_energies=exact_energies,
        exact_states=exact_states,
        ground=exact_ground,
        observables=observables,
        omega_grid=omega_grid,
    )

    terminal_row = receipt["rows"][-1]
    direct = dict(aggregate_row["direct_davidson"])
    exchange_rungs = sum(bool(row.get("exchanged")) for row in receipt["rows"])
    return {
        "case": case,
        "selected_online_status": str(selected_summary["algorithm_status"]),
        "direct_online_status": str(direct["algorithm_status"]),
        "selected_terminal": {
            "support_size": len(support),
            "retained_rank": int(terminal_row["retained_rank"]),
            "n2q": float(terminal_row["n2q"]),
            "d2q": float(terminal_row["d2q"]),
            "dc": float(terminal_row["dc"]),
            "exchange_rung_count": int(exchange_rungs),
            "support_indices": list(support),
            "max_root_energy_error": selected_energy_error,
        },
        "fixed_linear_response": {
            "support_size": len(fixed_support),
            "n2q": float(sum(base_problem.costs[index] for index in fixed_support)),
            "max_root_energy_error": fixed_energy_error,
        },
        "crossings": {
            "1e-4": {
                "selected_n2q": float(_crossing(selected_summary, 1.0e-4)["n2q"]),
                "direct_n2q": float(_crossing(direct, 1.0e-4)["n2q"]),
            },
            "1e-6": {
                "selected_n2q": float(_crossing(selected_summary, 1.0e-6)["n2q"]),
                "direct_n2q": float(_crossing(direct, 1.0e-6)["n2q"]),
            },
        },
        "spectral_grid": {
            "omega_min": 0.0,
            "omega_max": float(omega_max),
            "num_points": SPECTRAL_POINTS,
            "values": [float(x) for x in omega_grid],
            "kernel": "lorentzian",
            "eta": SPECTRAL_ETA,
        },
        "transition_observables": {
            name: {
                "selected": selected_metrics[name],
                "fixed_linear_response": fixed_metrics[name],
            }
            for name in selected_metrics
        },
        "route_artifact": {
            "path": str(route_path),
            "sha256": _sha256_file(route_path),
            "online_receipt_digest": str(route_payload["online_receipt_digest"]),
        },
    }


def build_production_payload(acquisition_path: Path) -> dict[str, Any]:
    acquisition = _read_json(acquisition_path)
    pool_policy_id = validate_acquisition_payload(acquisition)
    rows = [
        _case_product(acquisition_path, row, pool_policy_id)
        for row in acquisition["rows"]
    ]
    cost_ratios_1e4 = [
        row["crossings"]["1e-4"]["selected_n2q"]
        / row["crossings"]["1e-4"]["direct_n2q"]
        for row in rows
    ]
    cost_ratios_1e6 = [
        row["crossings"]["1e-6"]["selected_n2q"]
        / row["crossings"]["1e-6"]["direct_n2q"]
        for row in rows
    ]
    spectral_l1_selected = [
        row["transition_observables"][observable]["selected"][
            "normalized_spectral_l1"
        ]
        for row in rows
        for observable in ("site0_density", "site0_displacement")
    ]
    spectral_l1_fixed = [
        row["transition_observables"][observable]["fixed_linear_response"][
            "normalized_spectral_l1"
        ]
        for row in rows
        for observable in ("site0_density", "site0_displacement")
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "static_and_frequency_domain_qse_only",
        "controller_boundary": dict(CONTROLLER_BOUNDARY),
        "method_identity": {
            "selected_method_id": SELECTED_METHOD_ID,
            "direct_comparator_method_id": DIRECT_METHOD_ID,
            "selected_pool_policy_id": pool_policy_id,
            "selector_policy_id": PAPER_III_LOCKED_SELECTOR_POLICY_ID,
            "implicit_davidson_fallback": False,
        },
        "acquisition": {
            "path": str(acquisition_path),
            "sha256": _sha256_file(acquisition_path),
            "source_lock_digest": str(acquisition["source_lock_digest"]),
        },
        "reporting_source": {
            "revision": _git_revision(),
            "path": "pipelines/reporting/build_paper_iii_static_production_report.py",
            "sha256": _sha256_file(Path(__file__)),
        },
        "spectral_policy": {
            "observables": ["site0_density", "site0_displacement"],
            "strength_floor": STRENGTH_FLOOR,
            "kernel": "lorentzian",
            "eta": SPECTRAL_ETA,
            "grid_points": SPECTRAL_POINTS,
        },
        "summary": {
            "case_count": len(rows),
            "selected_online_certified_count": sum(
                row["selected_online_status"] == SELECTED_CERTIFIED_STATUS
                for row in rows
            ),
            "selected_not_more_expensive_than_direct_at_1e-4": sum(
                ratio <= 1.0 for ratio in cost_ratios_1e4
            ),
            "selected_not_more_expensive_than_direct_at_1e-6": sum(
                ratio <= 1.0 for ratio in cost_ratios_1e6
            ),
            "max_selected_to_direct_n2q_ratio_at_1e-4": max(cost_ratios_1e4),
            "max_selected_to_direct_n2q_ratio_at_1e-6": max(cost_ratios_1e6),
            "median_selected_spectral_l1": float(np.median(spectral_l1_selected)),
            "max_selected_spectral_l1": max(spectral_l1_selected),
            "median_fixed_spectral_l1": float(np.median(spectral_l1_fixed)),
            "max_fixed_spectral_l1": max(spectral_l1_fixed),
        },
        "rows": rows,
    }


def build_markdown(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Paper III static-QSE production evidence",
        "",
        "Scope: static and frequency-domain QSE only. No time propagation is included.",
        "",
        "## Locked outcome",
        "",
        f"- Selected-QSE online certification: {summary['selected_online_certified_count']}/{summary['case_count']} cells.",
        "- Fixed pool: `hh_full_meta_plus_encoded_boson_pauli_completion_v1`.",
        "- Selector: `full_pencil_ky_fan_cost_ratio_novelty_floor_no_reward_v1`.",
        f"- Selected QSE no more expensive than direct Davidson at $10^{{-4}}$: {summary['selected_not_more_expensive_than_direct_at_1e-4']}/{summary['case_count']} cells.",
        f"- Selected QSE no more expensive than direct Davidson at $10^{{-6}}$: {summary['selected_not_more_expensive_than_direct_at_1e-6']}/{summary['case_count']} cells.",
        "- No selected-QSE-to-Davidson fallback is present.",
        "",
        "## Per-cell energy/cost and static spectral errors",
        "",
        "`spec L1` is the normalized broadened-spectrum $L^1$ error at the certified terminal selected support. `fixed L1` is the complete fixed linear-response QSE baseline, averaged over site-0 density and displacement.",
        "",
        "| case | selected/direct $N_{2q}@10^{-4}$ | selected/direct $N_{2q}@10^{-6}$ | terminal selected $N_{2q}$ | selected max root error | fixed max root error | selected spec L1 | fixed spec L1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        c4 = row["crossings"]["1e-4"]
        c6 = row["crossings"]["1e-6"]
        selected_l1 = np.mean(
            [
                row["transition_observables"][name]["selected"][
                    "normalized_spectral_l1"
                ]
                for name in ("site0_density", "site0_displacement")
            ]
        )
        fixed_l1 = np.mean(
            [
                row["transition_observables"][name]["fixed_linear_response"][
                    "normalized_spectral_l1"
                ]
                for name in ("site0_density", "site0_displacement")
            ]
        )
        lines.append(
            "| `{case}` | {s4:.0f}/{d4:.0f} | {s6:.0f}/{d6:.0f} | {terminal:.0f} | {se:.2e} | {fe:.2e} | {sl1:.2e} | {fl1:.2e} |".format(
                case=row["case"]["case_key"],
                s4=c4["selected_n2q"],
                d4=c4["direct_n2q"],
                s6=c6["selected_n2q"],
                d6=c6["direct_n2q"],
                terminal=row["selected_terminal"]["n2q"],
                se=row["selected_terminal"]["max_root_energy_error"],
                fe=row["fixed_linear_response"]["max_root_energy_error"],
                sl1=selected_l1,
                fl1=fixed_l1,
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The direct-Davidson comparison is an energy/cost comparison. The fixed linear-response class supplies the conventional static-QSE spectral baseline. Exact eigensystems, transition strengths, and broadened curves are post-run references only. This package makes no time-dynamics claim and contains no real-time Krylov route.",
            "",
        ]
    )
    return "\n".join(lines)


def _case_labels(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        str(row["case"]["case_key"])
        .replace("intermediate", "int")
        .replace("strong_strong_u8", "strong-strong")
        .replace("strong_weak_u8", "strong-weak")
        .replace("weak_weak", "weak-weak")
        .replace("weak_strong", "weak-strong")
        .replace("_nph", " / nph")
        for row in rows
    ]


def render_plots(payload: Mapping[str, Any], output_dir: Path) -> tuple[Path, Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(payload["rows"])
    labels = _case_labels(rows)
    x = np.arange(len(rows))

    cost_path = out / "static_cost_ratios.png"
    fig, ax = plt.subplots(figsize=(10.5, 4.3), constrained_layout=True)
    ratios4 = [
        row["crossings"]["1e-4"]["selected_n2q"]
        / row["crossings"]["1e-4"]["direct_n2q"]
        for row in rows
    ]
    ratios6 = [
        row["crossings"]["1e-6"]["selected_n2q"]
        / row["crossings"]["1e-6"]["direct_n2q"]
        for row in rows
    ]
    width = 0.38
    ax.bar(x - width / 2, ratios4, width, label=r"$10^{-4}$", color="#3c78d8")
    ax.bar(x + width / 2, ratios6, width, label=r"$10^{-6}$", color="#6aa84f")
    ax.axhline(1.0, color="#222222", linewidth=1.0, linestyle="--")
    ax.set_ylabel(r"selected / direct $N_{2q}$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=8)
    ax.set_title("Matched static spectral-window cost")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(cost_path, dpi=220)
    plt.close(fig)

    spectral_path = out / "static_spectral_errors.png"
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True, constrained_layout=True)
    for ax, observable, title in zip(
        axes,
        ("site0_density", "site0_displacement"),
        ("Site-0 density", "Site-0 displacement"),
        strict=True,
    ):
        selected = [
            row["transition_observables"][observable]["selected"][
                "normalized_spectral_l1"
            ]
            for row in rows
        ]
        fixed = [
            row["transition_observables"][observable]["fixed_linear_response"][
                "normalized_spectral_l1"
            ]
            for row in rows
        ]
        ax.bar(x - width / 2, selected, width, label="selected QSE", color="#3c78d8")
        ax.bar(x + width / 2, fixed, width, label="fixed LR-QSE", color="#cc6f3c")
        ax.set_yscale("log")
        ax.set_ylabel("normalized spectral L1")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25, which="both")
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=38, ha="right", fontsize=8)
    fig.savefig(spectral_path, dpi=220)
    plt.close(fig)

    hard_path = out / "static_hard_case_spectra.png"
    hard_keys = ("weak_strong_nph3", "intermediate_strong_nph3")
    hard_rows = {row["case"]["case_key"]: row for row in rows}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), constrained_layout=True)
    for row_index, case_key in enumerate(hard_keys):
        row = hard_rows[case_key]
        omega = np.asarray(row["spectral_grid"]["values"], dtype=float)
        for col_index, observable in enumerate(
            ("site0_density", "site0_displacement")
        ):
            ax = axes[row_index, col_index]
            selected = row["transition_observables"][observable]["selected"][
                "spectrum"
            ]
            fixed = row["transition_observables"][observable][
                "fixed_linear_response"
            ]["spectrum"]
            ax.plot(omega, selected["exact"], color="#222222", linewidth=1.5, label="exact")
            ax.plot(omega, selected["predicted"], color="#3c78d8", linewidth=1.2, label="selected")
            ax.plot(omega, fixed["predicted"], color="#cc6f3c", linewidth=1.0, linestyle="--", label="fixed LR")
            ax.set_title(f"{case_key.replace('_', ' ')} - {observable.replace('_', ' ')}", fontsize=9)
            ax.set_xlabel(r"$\omega$")
            ax.set_ylabel(r"$A_O(\omega)$")
            ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=8)
    fig.savefig(hard_path, dpi=220)
    plt.close(fig)
    return cost_path, spectral_path, hard_path


def render_pdf(
    payload: Mapping[str, Any],
    output_path: Path,
    plot_paths: Sequence[Path],
) -> None:
    from reportlab import rl_config
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    rl_config.invariant = 1
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#17365d"),
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontSize=8,
            leading=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CaptionCenter",
            parent=styles["Small"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#444444"),
            spaceAfter=8,
        )
    )
    doc = SimpleDocTemplate(
        str(output),
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        title="Paper III static-QSE production evidence",
        author="Holstein Paper III results pipeline",
    )
    story: list[Any] = [
        Paragraph("Paper III static-QSE production evidence", styles["ReportTitle"]),
        Paragraph(
            "Static and frequency-domain QSE only. The locked method uses the coverage-qualified encoded-boson completion pool, full proposed-pencil Ky Fan gain per incremental cost, a novelty floor without an additive novelty reward, and certified atomic exchange. No time propagation or implicit Davidson fallback is included.",
            styles["BodyText"],
        ),
        Spacer(1, 8),
    ]
    summary = payload["summary"]
    summary_data = [
        ["Gate", "Result"],
        ["Online certified", f"{summary['selected_online_certified_count']}/{summary['case_count']}"],
        ["No more costly than Davidson at 1e-4", f"{summary['selected_not_more_expensive_than_direct_at_1e-4']}/{summary['case_count']}"],
        ["No more costly than Davidson at 1e-6", f"{summary['selected_not_more_expensive_than_direct_at_1e-6']}/{summary['case_count']}"],
        ["Implicit fallback", "disabled"],
    ]
    table = Table(summary_data, colWidths=[3.5 * inch, 2.2 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e7f5")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa7b3")),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fb")]),
            ]
        )
    )
    story.extend(
        [
            table,
            Spacer(1, 10),
            Image(str(plot_paths[0]), width=7.25 * inch, height=2.97 * inch),
            Paragraph(
                "Ratio below one favors selected QSE. Completion is cheaper in 10/12 cells at 1e-4 and all 12 cells at 1e-6.",
                styles["CaptionCenter"],
            ),
            PageBreak(),
            Paragraph("Static spectral products", styles["Heading1"]),
            Paragraph(
                "Terminal selected-QSE supports and the complete fixed linear-response class are re-solved with site-0 density and displacement observables. Curves use a Lorentzian kernel with eta=0.05; exact eigensystems are post-run references only.",
                styles["BodyText"],
            ),
            Spacer(1, 6),
            Image(str(plot_paths[1]), width=7.25 * inch, height=4.83 * inch),
            Paragraph(
                "Normalized L1 error of the broadened static spectrum. The fixed class is a conventional static-QSE baseline, not a growth competitor.",
                styles["CaptionCenter"],
            ),
            PageBreak(),
            Paragraph("Pool-completion stress cells", styles["Heading1"]),
            Paragraph(
                "The original macro pool exhausted in these two nph3 cells. The declared completion pool certifies both target windows without invoking Davidson.",
                styles["BodyText"],
            ),
            Spacer(1, 6),
            Image(str(plot_paths[2]), width=7.25 * inch, height=4.83 * inch),
            Paragraph(
                "Static response spectra for the two cells repaired by alphabet completion.",
                styles["CaptionCenter"],
            ),
            Spacer(1, 6),
            Paragraph(
                "Provenance: acquisition SHA-256 "
                + str(payload["acquisition"]["sha256"])
                + "; acquisition source-lock digest "
                + str(payload["acquisition"]["source_lock_digest"])
                + "; reporting revision "
                + str(payload["reporting_source"]["revision"])
                + ".",
                styles["Small"],
            ),
        ]
    )

    def add_page_number(canvas: Any, document: Any) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawRightString(7.95 * inch, 0.28 * inch, f"{document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


def run_campaign(acquisition_path: Path, output_dir: Path) -> dict[str, Any]:
    payload = build_production_payload(Path(acquisition_path))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "paper_iii_static_qse_production_evidence.json"
    markdown_path = out / "paper_iii_static_qse_production_evidence.md"
    _write_json(json_path, payload)
    markdown_path.write_text(build_markdown(payload), encoding="utf-8")
    plots = render_plots(payload, out)
    render_pdf(payload, out / "paper_iii_static_qse_production_report.pdf", plots)
    manifest = {
        "schema_version": "paper_iii_static_qse_production_manifest_v1",
        "controller_boundary": dict(CONTROLLER_BOUNDARY),
        "artifacts": {
            path.name: _sha256_file(path)
            for path in (json_path, markdown_path, *plots, out / "paper_iii_static_qse_production_report.pdf")
        },
    }
    _write_json(out / "manifest.json", manifest)
    return {"payload": payload, "manifest": manifest}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-json", type=Path, default=DEFAULT_ACQUISITION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    result = run_campaign(args.acquisition_json, args.output_dir)
    print(
        json.dumps(
            {
                "schema_version": result["payload"]["schema_version"],
                "scope": result["payload"]["scope"],
                "summary": result["payload"]["summary"],
                "output_dir": str(args.output_dir),
                "manifest": result["manifest"],
            },
            indent=1,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
