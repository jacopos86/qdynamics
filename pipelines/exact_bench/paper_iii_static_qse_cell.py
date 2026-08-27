#!/usr/bin/env python3
"""One-action producer for the Paper III static-QSE campaign.

```text
python -m pipelines.exact_bench.paper_iii_static_qse_cell \
  --action-json ACTION.json --output-dir OUTPUT_DIR
```

Executes exactly one declared :class:`StaticQSEAction` and writes its stage
payload plus a self-describing result envelope (action id, achieved
configuration, input/output digests, source revision, payload paths,
disposition, telemetry). All scientific work is delegated to the existing
implementations: the production record selector, certified exchange,
Davidson residual completion, the shared HH problem provider, the SIAM
static adapter, and the Green-function/spectral/resource modules. Nothing
here redefines the online policy: exact references qualify and report; they
never select a record, trigger the Davidson fallback, or stop a phase.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.qse_spectra.static_campaign import (
    METHOD_DIRECT_DAVIDSON,
    METHOD_FIXED_CLASS,
    METHOD_KRYLOV,
    METHOD_RECORD_FIRST,
    STAGE_GREEN_FUNCTION,
    STAGE_GROUND_STATE_IMPORT,
    STAGE_QUALIFICATION,
    STAGE_RESOURCE_EVALUATION,
    STAGE_SPECTRAL_POSTPROCESSING,
    STAGE_STATIC_QSE_TRACE,
    STAGE_TRANSITIONS,
    STATUS_CANDIDATE,
    STATUS_QUALIFIED,
    StaticCampaignError,
    StaticQSEAction,
    action_from_payload,
    build_cell_actions,
    canonical_digest,
)
from pipelines.qse_spectra.static_campaign_artifacts import (
    build_result_envelope,
    sha256_file,
    write_result_envelope,
)

TRACE_SCHEMA_VERSION = "paper_iii_static_qse_trace_v1"
_CONTROLLER_BOUNDARY = {
    "feeds_controller_decisions": False,
    "controller_usable": False,
    "post_run_diagnostic_only": True,
}
_DEFAULT_QUALIFICATION_GATES = {
    "window_error_gate": 1.0e-6,
    "generalized_residual_gate": 1.0e-6,
    "sum_rule_deficit_gate": 1.0e-8,
    "spectral_negativity_gate": 1.0e-10,
    "overlap_condition_gate": 1.0e14,
}


class StaticQSECellError(RuntimeError):
    """Raised when one action cannot be executed as declared."""


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Problem construction
# ---------------------------------------------------------------------------


class _ProblemView:
    """A shared problem with the prepared state swapped for the imported one."""

    def __init__(self, base: Any, ground: np.ndarray) -> None:
        self._base = base
        self.ground = np.asarray(ground, dtype=complex).reshape(-1)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def resource_triple(self, indices: Sequence[int]) -> dict[str, float]:
        return self._base.resource_triple(indices)

    def arm_receipt(self) -> dict[str, Any]:
        return self._base.arm_receipt()


def _load_hh_problem(context: Mapping[str, Any], *, target_roots: int) -> Any:
    from pipelines.qse_spectra.paper_iii_problem import load_problem

    spec = context["problem"]
    return load_problem(
        regime=str(spec["regime"]),
        u=float(spec["u"]),
        g_ep=float(spec["g_ep"]),
        n_ph_max=int(spec["n_ph_max"]),
        target_roots=int(target_roots),
    )


def _load_siam_problem(context: Mapping[str, Any], *, target_roots: int) -> Any:
    from pipelines.qse_spectra.siam_static_adapter import load_siam_problem

    spec = context["problem"]
    ground = context["ground_state"]
    kind = str(ground.get("kind"))
    if kind == "artifact":
        return load_siam_problem(
            str(spec["case_id"]),
            ground_state_manifest=Path(str(ground["manifest_path"])),
            target_roots=int(target_roots),
        )
    if kind == "exact_diagnostic":
        return load_siam_problem(
            str(spec["case_id"]),
            allow_exact_diagnostic_fixture=True,
            target_roots=int(target_roots),
        )
    raise StaticQSECellError(f"unknown SIAM ground_state kind {kind!r}.")


def _resolve_problem(action: StaticQSEAction, context: Mapping[str, Any]) -> Any:
    """Build the declared problem with the declared prepared state."""

    cell = action.cell
    target_roots = len(cell.target_roots)
    kind = str(context["problem"].get("kind"))
    if kind == "siam":
        problem = _load_siam_problem(context, target_roots=target_roots)
        ground_provenance = dict(problem.ground_provenance)
        psi = problem.ground
    elif kind == "hh":
        problem = _load_hh_problem(context, target_roots=target_roots)
        psi, ground_provenance = _resolve_hh_ground(problem, context)
        problem = _ProblemView(problem, psi)
    else:
        raise StaticQSECellError(f"unknown problem kind {kind!r}.")

    if str(problem.pool_digest) != str(cell.pool_policy_digest):
        raise StaticQSECellError(
            f"pool digest {problem.pool_digest} does not match the cell's "
            f"declared pool policy digest {cell.pool_policy_digest}."
        )
    return problem, ground_provenance


def _resolve_hh_ground(
    problem: Any, context: Mapping[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    from pipelines.qse_spectra.siam_static_adapter import state_payload_sha256

    ground = context["ground_state"]
    kind = str(ground.get("kind"))
    if kind == "exact_diagnostic":
        psi = np.asarray(problem.ground, dtype=complex).reshape(-1)
        return psi, {
            "ground_source": "provider_exact_sector_diagnostic_fixture",
            "diagnostic_only": True,
            "paper_facing": False,
            "state_payload_sha256": state_payload_sha256(psi),
            "ground_energy": float(problem.ground_energy),
        }
    if kind == "artifact":
        from pipelines.qse_spectra.io import load_state_json

        path = Path(str(ground["manifest_path"]))
        psi, io_provenance = load_state_json(
            path,
            expected_nq=int(problem.num_qubits),
            state_key=str(ground.get("state_key", "auto")),
        )
        norm = float(np.linalg.norm(psi))
        if abs(norm - 1.0) > 1.0e-9:
            raise StaticQSECellError(f"imported HH state norm {norm!r} is not 1.")
        import hashlib

        return np.asarray(psi, dtype=complex).reshape(-1), {
            "ground_source": "imported_prepared_state_artifact",
            "diagnostic_only": False,
            "manifest_path": str(path),
            "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "state_payload_sha256": state_payload_sha256(psi),
            "io_provenance": dict(io_provenance),
        }
    raise StaticQSECellError(f"unknown HH ground_state kind {kind!r}.")


# ---------------------------------------------------------------------------
# Spectral request resolution
# ---------------------------------------------------------------------------


def _resolve_transition_observables(
    context: Mapping[str, Any], problem: Any
) -> tuple[Any, ...]:
    from pipelines.qse_spectra.core import (
        pauli_string_observable,
        polynomial_observable,
    )

    specs = context["spectral_request"].get("transition_observables", [])
    out: list[Any] = []
    for spec in specs:
        builder = str(spec.get("builder"))
        name = str(spec.get("name", builder))
        if builder == "pauli_string":
            out.append(
                pauli_string_observable(
                    str(spec["label"]), nq=int(problem.num_qubits), name=name
                )
            )
            continue
        if builder.startswith("siam_"):
            from src.quantum.siam_model import SIAMLayout, UP, DOWN, number_operator
            from src.quantum.siam_observables import (
                impurity_double_occupancy_operator,
                impurity_magnetization_operator,
            )

            layout = SIAMLayout(int(problem.key.n_bath))
            if builder == "siam_impurity_occupation":
                poly = number_operator(layout, layout.impurity(UP)) + number_operator(
                    layout, layout.impurity(DOWN)
                )
            elif builder == "siam_impurity_double_occupancy":
                poly = impurity_double_occupancy_operator(layout)
            elif builder == "siam_impurity_magnetization":
                poly = impurity_magnetization_operator(layout)
            else:
                raise StaticQSECellError(f"unknown SIAM observable builder {builder!r}.")
            out.append(polynomial_observable(poly, name=name))
            continue
        raise StaticQSECellError(f"unknown transition observable builder {builder!r}.")
    return tuple(out)


def _spectral_grid_and_kernel(context: Mapping[str, Any]):
    from pipelines.qse_spectra.spectral_functions import (
        BroadeningKernelConfig,
        SpectralGrid,
    )

    request = context["spectral_request"]
    grid = SpectralGrid(**{k: request["grid"][k] for k in ("omega_min", "omega_max", "num_points")})
    kernel = BroadeningKernelConfig(
        kernel=str(request["kernel"]["kernel"]), eta=float(request["kernel"]["eta"])
    )
    return grid, kernel


def _q0_policy():
    from pipelines.qse_spectra.core import QSEBasisVectorPolicy

    return QSEBasisVectorPolicy(
        reference_projection="q0", basis_vector_normalization="raw_projected"
    )


# ---------------------------------------------------------------------------
# Dependency payload access
# ---------------------------------------------------------------------------


def _stage_payload_path(action: StaticQSEAction, stage: str, output_dir: Path) -> Path:
    actions = build_cell_actions(action.cell)
    stage_action = actions[stage]
    return Path(output_dir).parent / stage_action.action_id / f"{stage}.json"


def _read_stage_payload(
    action: StaticQSEAction, stage: str, output_dir: Path
) -> tuple[dict[str, Any], str, Path]:
    path = _stage_payload_path(action, stage, output_dir)
    if not path.is_file():
        raise StaticQSECellError(
            f"dependency payload for stage {stage!r} is missing at {path}; the "
            "planner schedules this action only after its dependencies complete."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, sha256_file(path), path


def _terminal_record_support(trace_payload: Mapping[str, Any]) -> list[int]:
    rows = trace_payload.get("rows", [])
    supports = [row for row in rows if row.get("support_indices") is not None]
    if not supports:
        raise StaticQSECellError(
            "trace has no record support; QSE product stages require a record-"
            "supported terminal basis in this increment."
        )
    return [int(i) for i in supports[-1]["support_indices"]]


def _solve_terminal_pencil(
    problem: Any,
    support: Sequence[int],
    *,
    transition_observables: Sequence[Any] = (),
) -> Any:
    from pipelines.qse_spectra.core import compute_qse_spectra

    return compute_qse_spectra(
        problem.hamiltonian,
        problem.ground,
        tuple(problem.basis[int(i)] for i in support),
        basis_vector_policy=_q0_policy(),
        transition_observables=tuple(transition_observables) or None,
    )


def _pencil_estimator_cost(problem: Any, indices: Sequence[int]) -> dict[str, int]:
    """Generic (S,H)-pencil Pauli-work counter for any record support."""

    estimator = getattr(problem, "estimator_cost", None)
    if callable(estimator):
        return dict(estimator(indices))
    from pipelines.exact_bench.paper_iii_qse_measurement_cost import (
        _element_words,
        _polynomial_words,
        _qwc_cover,
        _word_product,
    )

    nq = int(problem.num_qubits)
    ham_words = _polynomial_words(problem.hamiltonian)
    words = {int(i): _element_words(problem.basis[int(i)], nq=nq) for i in indices}
    ordered = [int(i) for i in indices]
    overlap: set[str] = set()
    hamiltonian_words: set[str] = set()
    naive = 0
    for pos, a in enumerate(ordered):
        for b in ordered[pos:]:
            products = {_word_product(x, y) for x in words[a] for y in words[b]}
            overlap |= products
            naive += len(products)
            ham_products = {_word_product(p, h) for p in products for h in ham_words}
            hamiltonian_words |= ham_products
            naive += len(ham_products)
    allw = overlap | hamiltonian_words
    return {
        "pair_count": len(ordered) * (len(ordered) + 1) // 2,
        "naive_terms": int(naive),
        "distinct_words": len(allw),
        "qwc_groups": _qwc_cover(sorted(allw), nq=nq),
    }


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def _stage_ground_state_import(
    action: StaticQSEAction, context: Mapping[str, Any], output_dir: Path
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    problem, provenance = _resolve_problem(action, context)
    payload = {
        "schema_version": "paper_iii_static_qse_ground_state_import_v1",
        "ground_source": str(provenance.get("ground_source")),
        "diagnostic_only": bool(provenance.get("diagnostic_only")),
        "state_payload_sha256": str(provenance.get("state_payload_sha256")),
        "num_qubits": int(problem.num_qubits),
        "provenance": _json_safe(provenance),
        "controller_boundary": dict(_CONTROLLER_BOUNDARY),
    }
    inputs: dict[str, str] = {}
    manifest_sha = provenance.get("manifest_sha256")
    if manifest_sha:
        inputs["ground_state_manifest"] = str(manifest_sha)
    settings = {
        "ground_state_kind": str(context["ground_state"].get("kind")),
        "problem_kind": str(context["problem"].get("kind")),
    }
    return payload, inputs, settings


def _stage_static_qse_trace(
    action: StaticQSEAction, context: Mapping[str, Any], output_dir: Path
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    import pipelines.exact_bench.paper_iii_growth_trace_campaign as growth

    problem, provenance = _resolve_problem(action, context)
    cell = action.cell
    method = str(cell.method_id)
    cap = int(cell.dimension_cap)
    stride = int(context.get("trace", {}).get("stride", 2))
    trigger: dict[str, Any] | None = None
    selection_receipt: dict[str, Any] | None = None

    if method == METHOD_RECORD_FIRST:
        from pipelines.qse_spectra.record_selection import (
            StaticRecordSelectionConfig,
            select_static_qse_records,
        )

        marker = select_static_qse_records(
            problem.basis,
            config=StaticRecordSelectionConfig(
                mode="geometry_selected",
                max_records=len(problem.basis),
                geometry_target_roots=len(cell.target_roots),
                geometry_cost_discount_alpha=1.0,
                geometry_residual_stop=growth.RESIDUAL_MARKER,
            ),
            hamiltonian=problem.hamiltonian,
            prepared_state=problem.ground,
            basis_vector_policy=_q0_policy(),
            compiled_costs=problem.costs,
        )
        order = [int(i) for i in marker.selected_original_indices]
        rows, trigger = growth.trace_record_first_adaptive_qse(
            problem,
            order,
            k_max=cap,
            stride=stride,
            record_stop=dict(marker.geometry_stop or {"stop_reason": "missing"}),
        )
        selection_receipt = {
            "acquisition_order": order,
            "record_stop": dict(marker.geometry_stop or {}),
            "residual_marker": float(growth.RESIDUAL_MARKER),
        }
    elif method == METHOD_DIRECT_DAVIDSON:
        rows = growth.trace_adaptive_qse(problem, k_max=cap)
    elif method == METHOD_KRYLOV:
        rows = growth.trace_krylov(problem, k_max=cap)
    elif method == METHOD_FIXED_CLASS:
        indices = [
            i
            for i, element in enumerate(problem.basis)
            if growth._element_family(element.name) in growth._LINEAR_RESPONSE_FAMILIES
        ]
        if not indices:
            raise StaticQSECellError(
                "no declared fixed linear-response class exists for this model "
                "family; the campaign marks that slot not_planned."
            )
        rows = [growth._row(problem, indices, len(indices), exchanged=False)]
    else:  # ablations are internal-campaign work, not this producer's surface
        raise StaticQSECellError(f"method {method!r} is not runnable by this producer.")

    terminal_support: list[int] | None = None
    supports = [row for row in rows if row.get("support_indices") is not None]
    if supports:
        terminal_support = [int(i) for i in supports[-1]["support_indices"]]

    payload = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "method_id": method,
        "rows": _json_safe(rows),
        "trigger": _json_safe(trigger),
        "selection_receipt": _json_safe(selection_receipt),
        "terminal_support_indices": terminal_support,
        "shared_problem_receipt": _json_safe(problem.arm_receipt()),
        "pool_digest": str(problem.pool_digest),
        "target_roots": [int(r) for r in cell.target_roots],
        "dimension_cap": cap,
        "stride": stride,
        "reference_ground_energy": float(problem.ground_energy),
        "reference_excitations": [float(x) for x in problem.references],
        "ground_state_payload_sha256": str(provenance.get("state_payload_sha256")),
        "controller_boundary": dict(_CONTROLLER_BOUNDARY),
    }
    inputs = {"ground_state_payload": str(provenance.get("state_payload_sha256"))}
    settings = {
        "method_id": method,
        "dimension_cap": cap,
        "stride": stride,
        "pool_digest": str(problem.pool_digest),
    }
    return payload, inputs, settings


def _product_stage_common(
    action: StaticQSEAction, context: Mapping[str, Any], output_dir: Path
):
    trace_payload, trace_sha, _ = _read_stage_payload(
        action, STAGE_STATIC_QSE_TRACE, output_dir
    )
    problem, _prov = _resolve_problem(action, context)
    support = _terminal_record_support(trace_payload)
    return problem, support, trace_payload, {"static_qse_trace_payload": trace_sha}


def _stage_transitions(
    action: StaticQSEAction, context: Mapping[str, Any], output_dir: Path
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    problem, support, _trace, inputs = _product_stage_common(action, context, output_dir)
    observables = _resolve_transition_observables(context, problem)
    if not observables:
        raise StaticQSECellError("transitions stage requires declared observables.")
    result = _solve_terminal_pencil(
        problem, support, transition_observables=observables
    )
    records = []
    for transition in result.transition_observables:
        records.append(
            {
                "name": str(transition.observable.name),
                "kind": str(transition.observable.kind),
                "transition_amplitudes": [
                    [float(a.real), float(a.imag)]
                    for a in np.asarray(transition.transition_amplitudes).reshape(-1)
                ],
                "transition_strengths": [
                    float(x)
                    for x in np.asarray(transition.transition_strengths).reshape(-1)
                ],
                "observable_hermitian_residual_max_abs": (
                    None
                    if transition.observable_matrix_hermitian_residual_max_abs is None
                    else float(transition.observable_matrix_hermitian_residual_max_abs)
                ),
            }
        )
    payload = {
        "schema_version": "paper_iii_static_qse_transitions_v1",
        "support_indices": [int(i) for i in support],
        "root_energies": [float(x) for x in np.asarray(result.eigenvalues).reshape(-1)],
        "retained_rank": int(result.retained_rank),
        "transition_observables": records,
        "controller_boundary": dict(_CONTROLLER_BOUNDARY),
    }
    settings = {"observable_names": [str(o.name) for o in observables]}
    return payload, inputs, settings


def _stage_green_function(
    action: StaticQSEAction, context: Mapping[str, Any], output_dir: Path
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    from pipelines.qse_spectra.green_functions import (
        build_green_function_payload,
        parse_green_function_mode_specs,
    )

    problem, support, _trace, inputs = _product_stage_common(action, context, output_dir)
    modes = parse_green_function_mode_specs(
        [str(x) for x in context["spectral_request"].get("green_function_modes", [])]
    )
    if not modes:
        raise StaticQSECellError("green_function stage requires declared modes.")
    grid, kernel = _spectral_grid_and_kernel(context)
    result = _solve_terminal_pencil(problem, support)
    payload = build_green_function_payload(
        result,
        hamiltonian=problem.hamiltonian,
        prepared_state=problem.ground,
        modes=modes,
        grid=grid,
        kernel_config=kernel,
        fermion_mode_count=int(problem.num_qubits),
    )
    payload = {
        "schema_version": str(payload.get("schema_version")),
        "support_indices": [int(i) for i in support],
        **{k: v for k, v in payload.items() if k != "schema_version"},
    }
    settings = {
        "mode_labels": [str(m.label) for m in modes],
        "eta": float(kernel.eta),
        "grid": grid.to_manifest(include_values=False),
    }
    return payload, inputs, settings


def _stage_resource_evaluation(
    action: StaticQSEAction, context: Mapping[str, Any], output_dir: Path
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    trace_payload, trace_sha, _ = _read_stage_payload(
        action, STAGE_STATIC_QSE_TRACE, output_dir
    )
    problem, _prov = _resolve_problem(action, context)
    inputs = {"static_qse_trace_payload": trace_sha}
    rows = trace_payload.get("rows", [])
    if not rows:
        raise StaticQSECellError("resource evaluation requires a nonempty trace.")
    terminal = rows[-1]
    graph_span = {
        "n2q": float(terminal["n2q"]),
        "d2q": float(terminal["d2q"]),
        "dc": float(terminal["dc"]),
    }
    support = trace_payload.get("terminal_support_indices")
    proxy = None
    estimator = None
    if support is not None:
        indices = [int(i) for i in support]
        proxy = {
            "scalarized_canonical_cost_sum": float(
                sum(float(problem.costs[i]) for i in indices)
            ),
            "support_size": len(indices),
        }
        estimator = dict(_pencil_estimator_cost(problem, indices))
        estimator["queries"] = int(estimator["qwc_groups"])
        estimator["convention"] = "record_pencil_qwc_basis_cover"
    elif isinstance(terminal.get("estimator"), Mapping):
        estimator = dict(terminal["estimator"])
    include_compile = bool(context.get("resources", {}).get("include_qiskit_compile"))
    compiled = None
    if include_compile and support is not None:
        from pipelines.qse_spectra.compiled_costs import (
            ORACLE_KIND_BACKEND_TRANSPILE,
            annotate_basis_with_compiled_costs,
            resolve_cost_weights_preset,
        )

        rows_compiled = annotate_basis_with_compiled_costs(
            tuple(problem.basis[int(i)] for i in support),
            num_qubits=int(problem.num_qubits),
            oracle_kind=ORACLE_KIND_BACKEND_TRANSPILE,
            cost_weights=resolve_cost_weights_preset("two_qubit_only_v1"),
        )
        compiled = {
            "two_qubit_count": float(sum(r.estimate.c_hat_2q for r in rows_compiled)),
            "depth": float(sum(r.estimate.c_hat_d for r in rows_compiled)),
        }
    payload = {
        "schema_version": "paper_iii_static_qse_resources_v1",
        "resource_policy_id": str(action.cell.resource_policy_id),
        "axes": {
            "proxy": proxy,
            "graph_span": graph_span,
            "qiskit_compiled": compiled
            if include_compile
            else {"status": "not_requested"},
            "estimator": estimator,
        },
        "terminal_support_indices": support,
        "controller_boundary": dict(_CONTROLLER_BOUNDARY),
    }
    settings = {"include_qiskit_compile": include_compile}
    return payload, inputs, settings


def _stage_spectral_postprocessing(
    action: StaticQSEAction, context: Mapping[str, Any], output_dir: Path
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    from pipelines.qse_spectra.spectral_functions import (
        SpectralWindow,
        build_spectral_postprocessing_payloads,
    )

    transitions_payload, transitions_sha, _ = _read_stage_payload(
        action, STAGE_TRANSITIONS, output_dir
    )
    green_payload, green_sha, _ = _read_stage_payload(
        action, STAGE_GREEN_FUNCTION, output_dir
    )
    problem, support, _trace, inputs = _product_stage_common(action, context, output_dir)
    inputs["transitions_payload"] = transitions_sha
    inputs["green_function_payload"] = green_sha
    observables = _resolve_transition_observables(context, problem)
    grid, kernel = _spectral_grid_and_kernel(context)
    windows = tuple(
        SpectralWindow(**{k: w[k] for k in ("name", "omega_min", "omega_max")})
        for w in context["spectral_request"].get("windows", [])
    )
    result = _solve_terminal_pencil(
        problem, support, transition_observables=observables
    )
    spectral_payload, window_payload = build_spectral_postprocessing_payloads(
        result, grid=grid, kernel_config=kernel, windows=windows
    )
    payload = {
        "schema_version": "paper_iii_static_qse_spectral_postprocessing_v1",
        "support_indices": [int(i) for i in support],
        "spectral": spectral_payload,
        "window_metrics": window_payload,
        "green_function_sha256": green_sha,
        "controller_boundary": dict(_CONTROLLER_BOUNDARY),
    }
    settings = {"window_count": len(windows), "kernel": kernel.to_manifest()}
    return payload, inputs, settings


def _stage_qualification(
    action: StaticQSEAction, context: Mapping[str, Any], output_dir: Path
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    ground_payload, ground_sha, _ = _read_stage_payload(
        action, STAGE_GROUND_STATE_IMPORT, output_dir
    )
    trace_payload, trace_sha, _ = _read_stage_payload(
        action, STAGE_STATIC_QSE_TRACE, output_dir
    )
    transitions_payload, transitions_sha, _ = _read_stage_payload(
        action, STAGE_TRANSITIONS, output_dir
    )
    green_payload, green_sha, _ = _read_stage_payload(
        action, STAGE_GREEN_FUNCTION, output_dir
    )
    spectral_payload, spectral_sha, _ = _read_stage_payload(
        action, STAGE_SPECTRAL_POSTPROCESSING, output_dir
    )
    resources_payload, resources_sha, _ = _read_stage_payload(
        action, STAGE_RESOURCE_EVALUATION, output_dir
    )
    inputs = {
        "ground_state_import_payload": ground_sha,
        "static_qse_trace_payload": trace_sha,
        "transitions_payload": transitions_sha,
        "green_function_payload": green_sha,
        "spectral_postprocessing_payload": spectral_sha,
        "resource_evaluation_payload": resources_sha,
    }
    gates = dict(_DEFAULT_QUALIFICATION_GATES)
    gates.update(
        {
            str(k): float(v)
            for k, v in context.get("qualification_policy", {}).items()
        }
    )

    problem, support, _trace, _in = _product_stage_common(action, context, output_dir)
    result = _solve_terminal_pencil(problem, support)

    rows = trace_payload.get("rows", [])
    finite_errors = [
        float(row["max_root_abs_error"])
        for row in rows
        if row.get("max_root_abs_error") is not None
    ]
    terminal_window_error = min(finite_errors) if finite_errors else None
    max_residual = (
        max(float(x) for x in result.generalized_residual_norms)
        if result.generalized_residual_norms
        else None
    )
    sum_rule_deficits = [
        float(
            mode["diagonal_sum_rule_diagnostics"]["source_norm_canonical_deficit_abs"]
        )
        for mode in green_payload.get("modes", [])
    ]
    spectral_negativity = 0.0
    for mode in green_payload.get("modes", []):
        values = np.asarray(
            mode["diagonal_spectral_function"]["values"], dtype=float
        )
        spectral_negativity = max(
            spectral_negativity, float(max(0.0, -(values.min() if values.size else 0.0)))
        )
    condition = result.overlap_condition_estimate

    failed: list[str] = []
    if terminal_window_error is None or terminal_window_error > gates["window_error_gate"]:
        failed.append("window_error_gate")
    if max_residual is None or max_residual > gates["generalized_residual_gate"]:
        failed.append("generalized_residual_gate")
    if sum_rule_deficits and max(sum_rule_deficits) > gates["sum_rule_deficit_gate"]:
        failed.append("sum_rule_deficit_gate")
    if spectral_negativity > gates["spectral_negativity_gate"]:
        failed.append("spectral_negativity_gate")
    if condition is not None and float(condition) > gates["overlap_condition_gate"]:
        failed.append("overlap_condition_gate")

    status = STATUS_QUALIFIED if not failed else STATUS_CANDIDATE
    record = {
        "schema_version": "paper_iii_static_qse_qualification_v1",
        "status": status,
        "failed_gates": failed,
        "gates": gates,
        "ground_state": {
            "ground_source": ground_payload.get("ground_source"),
            "diagnostic_only": ground_payload.get("diagnostic_only"),
            "state_payload_sha256": ground_payload.get("state_payload_sha256"),
            "num_qubits": ground_payload.get("num_qubits"),
        },
        "selection": {
            "support_indices": [int(i) for i in support],
            "pool_digest": trace_payload.get("pool_digest"),
            "retained_rank": int(result.retained_rank),
            "discarded_rank": int(result.discarded_rank),
            "overlap_min_eigenvalue_raw": float(result.overlap_min_eigenvalue_raw),
            "overlap_max_eigenvalue_raw": float(result.overlap_max_eigenvalue_raw),
            "overlap_condition_estimate": (
                None if condition is None else float(condition)
            ),
            "exchange_trigger": trace_payload.get("trigger"),
        },
        "roots": {
            "root_energies": [
                float(x) for x in np.asarray(result.eigenvalues).reshape(-1)
            ],
            "generalized_residual_norms": [
                float(x) for x in result.generalized_residual_norms
            ],
            "terminal_window_error": terminal_window_error,
            "reference_excitations": trace_payload.get("reference_excitations"),
        },
        "transitions": {
            "observable_names": [
                str(rec.get("name"))
                for rec in transitions_payload.get("transition_observables", [])
            ],
        },
        "green_function": {
            "mode_count": green_payload.get("summary", {}).get("mode_count"),
            "zero_source_sector_count": green_payload.get("summary", {}).get(
                "zero_source_sector_count"
            ),
            "sum_rule_deficits": sum_rule_deficits,
            "eta": green_payload.get("kernel", {}).get("eta"),
        },
        "spectral": {
            "max_negativity": spectral_negativity,
            "observable_count": len(
                spectral_payload.get("spectral", {}).get("observables", [])
            ),
        },
        "resources": resources_payload.get("axes"),
        "controller_boundary": dict(_CONTROLLER_BOUNDARY),
    }
    settings = {"gates": gates}
    return record, inputs, settings


_STAGE_DISPATCH = {
    STAGE_GROUND_STATE_IMPORT: _stage_ground_state_import,
    STAGE_STATIC_QSE_TRACE: _stage_static_qse_trace,
    STAGE_TRANSITIONS: _stage_transitions,
    STAGE_GREEN_FUNCTION: _stage_green_function,
    STAGE_RESOURCE_EVALUATION: _stage_resource_evaluation,
    STAGE_SPECTRAL_POSTPROCESSING: _stage_spectral_postprocessing,
    STAGE_QUALIFICATION: _stage_qualification,
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run_action(action_json: Path, output_dir: Path) -> dict[str, Any]:
    """Execute one declared action and write payload + result envelope."""

    started = time.time()
    action_file = json.loads(Path(action_json).read_text(encoding="utf-8"))
    if action_file.get("schema_version") != "paper_iii_static_qse_action_file_v1":
        raise StaticQSECellError(
            f"unknown action-file schema {action_file.get('schema_version')!r}."
        )
    action = action_from_payload(action_file["action"])
    if str(action_file.get("action_id")) != action.action_id:
        raise StaticQSECellError(
            "action file id does not match the reconstructed action identity."
        )
    context = action_file.get("context")
    if not isinstance(context, Mapping):
        raise StaticQSECellError("action file is missing its execution context.")
    declared = str(action.cell.spectral_request_digest)
    actual = canonical_digest(dict(context["spectral_request"]))
    if actual != declared:
        raise StaticQSECellError(
            f"spectral request digest {actual} does not match the declared {declared}."
        )

    stage = str(action.stage)
    handler = _STAGE_DISPATCH.get(stage)
    if handler is None:
        raise StaticQSECellError(f"no handler for stage {stage!r}.")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload, inputs, settings = handler(action, context, out_dir)
    payload_path = out_dir / f"{stage}.json"
    payload_path.write_text(
        json.dumps(_json_safe(payload), indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    inputs = dict(inputs)
    inputs["action_context"] = canonical_digest(dict(context))
    telemetry = {"elapsed_seconds": round(time.time() - started, 3)}
    envelope = build_result_envelope(
        action=action,
        achieved_settings=_json_safe(settings),
        source_revision=_git_revision(),
        input_digests=inputs,
        payload_files={f"{stage}.json": payload_path},
        disposition="completed",
        telemetry=telemetry,
    )
    if stage == STAGE_QUALIFICATION:
        envelope["qualification"] = {
            "status": str(payload["status"]),
            "failed_gates": list(payload["failed_gates"]),
        }
    write_result_envelope(envelope, out_dir)
    return envelope


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    envelope = run_action(args.action_json, args.output_dir)
    print(
        f"action {envelope['action_id'][:12]} stage {envelope['achieved']['stage']} "
        f"disposition {envelope['disposition']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
