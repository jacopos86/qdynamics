#!/usr/bin/env python3
"""Strict-JSON subprocess worker for the public CEO-ADAPT-VQE checkout.

This file is intentionally self-contained: the parent exact-bench adapter owns
provenance checks and normalized artifact writing, while this worker imports and
executes the pinned third-party public code under the selected external Python
interpreter and prints exactly one JSON object to stdout.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import traceback
from collections.abc import Iterable
from numbers import Real
from pathlib import Path
from typing import Any, Sequence

SCHEMA_VERSION = "ceo_public_code_worker_v1"
_WORKER_MODES = ("ceo", "tetris")


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    key = str(value).strip().lower()
    if key in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if key in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


def _tail(text: str, *, max_lines: int = 40) -> str:
    lines = str(text or "").splitlines()
    return "\n".join(lines[-max_lines:])


def _emit(payload: dict[str, Any]) -> int:
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    sys.stdout.flush()
    return 0


def _sum_nested(values: Any) -> int | None:
    if values is None:
        return None
    if isinstance(values, bool):
        return int(values)
    if isinstance(values, Real):
        return int(values)
    if isinstance(values, (str, bytes, bytearray)):
        return None
    if isinstance(values, Iterable):
        total = 0
        saw = False
        for item in values:
            part = _sum_nested(item)
            if part is None:
                continue
            total += int(part)
            saw = True
        return total if saw else None
    return None


def _float_tuple(values: Any) -> tuple[float, ...]:
    if values is None or isinstance(values, (str, bytes, bytearray)):
        return ()
    if isinstance(values, Real):
        return (float(values),)
    try:
        iterator = iter(values)
    except TypeError:
        try:
            return (float(values),)
        except Exception:
            return ()
    out: list[float] = []
    for item in iterator:
        out.extend(_float_tuple(item))
    return tuple(out)


def _int_tuple(values: Any) -> tuple[int, ...]:
    if values is None or isinstance(values, (str, bytes, bytearray)):
        return ()
    if isinstance(values, Real):
        return (int(values),)
    try:
        iterator = iter(values)
    except TypeError:
        try:
            return (int(values),)
        except Exception:
            return ()
    out: list[int] = []
    for item in iterator:
        try:
            out.append(int(item))
        except Exception:
            pass
    return tuple(out)


def _int_lists(values: Any) -> list[list[int]]:
    if values is None or isinstance(values, (str, bytes, bytearray)):
        return []
    if isinstance(values, Real):
        return [[int(values)]]
    try:
        iterator = iter(values)
    except TypeError:
        return []
    out: list[list[int]] = []
    for item in iterator:
        if item is None or isinstance(item, (str, bytes, bytearray)):
            out.append([])
        elif isinstance(item, Real):
            out.append([int(item)])
        else:
            out.append(list(_int_tuple(item)))
    return out


def _float_lists(values: Any) -> list[list[float]]:
    if values is None or isinstance(values, (str, bytes, bytearray)):
        return []
    if isinstance(values, Real):
        return [[float(values)]]
    try:
        iterator = iter(values)
    except TypeError:
        return []
    out: list[list[float]] = []
    for item in iterator:
        if item is None or isinstance(item, (str, bytes, bytearray)):
            out.append([])
        elif isinstance(item, Real):
            out.append([float(item)])
        else:
            out.append(list(_float_tuple(item)))
    return out


def _additions_per_iteration(indices_by_iteration: list[list[int]], sizes: Any) -> list[int]:
    if indices_by_iteration:
        ansatz_sizes = [len(indices) for indices in indices_by_iteration]
    else:
        ansatz_sizes = list(_int_tuple(sizes))
    additions: list[int] = []
    previous_size = 0
    for size in ansatz_sizes:
        additions.append(max(0, int(size) - int(previous_size)))
        previous_size = int(size)
    return additions


def _adapt_history(
    *,
    energies: Any,
    exact_energy: Any,
    initial_energy: Any,
    selected_indices_by_iteration: list[list[int]],
    coefficients_by_iteration: list[list[float]],
    operators_added_per_iteration: list[int],
    gradient_norms: tuple[float, ...],
    selected_gradients: tuple[float, ...],
) -> list[dict[str, Any]]:
    energy_values = list(_float_tuple(energies))
    exact = float(exact_energy) if exact_energy is not None else None
    initial = float(initial_energy) if initial_energy is not None else None
    out: list[dict[str, Any]] = []
    for idx, energy in enumerate(energy_values):
        cumulative_indices = (
            selected_indices_by_iteration[idx]
            if idx < len(selected_indices_by_iteration)
            else []
        )
        additions = (
            int(operators_added_per_iteration[idx])
            if idx < len(operators_added_per_iteration)
            else None
        )
        gradient_norm = float(gradient_norms[idx]) if idx < len(gradient_norms) else None
        selected_slice_start = sum(int(x) for x in operators_added_per_iteration[:idx])
        selected_slice_end = selected_slice_start + (int(additions) if additions is not None else 0)
        selected_gradients_this_iteration = list(selected_gradients[selected_slice_start:selected_slice_end])
        entry: dict[str, Any] = {
            "iteration": idx,
            "adapt_selection_round": idx + 1,
            "energy_after": float(energy),
            "delta_E_abs_after": None if exact is None else abs(float(energy) - exact),
            "abs_delta_e_after": None if exact is None else abs(float(energy) - exact),
            "abs_delta_e_same_cutoff_after": None if exact is None else abs(float(energy) - exact),
            "energy_change_from_initial": None if initial is None else float(energy) - initial,
            "selected_indices_after": list(cumulative_indices),
            "coefficients_after": (
                list(coefficients_by_iteration[idx])
                if idx < len(coefficients_by_iteration)
                else []
            ),
            "operators_added_this_iteration": additions,
            "gradient_norm_before": gradient_norm,
            "selected_gradients_this_iteration": selected_gradients_this_iteration,
        }
        out.append(entry)
    return out


def _import_public_code(checkout_dir: Path, stdout_buffer: io.StringIO) -> dict[str, Any]:
    checkout_str = str(checkout_dir)
    if checkout_str not in sys.path:
        sys.path.insert(0, checkout_str)
    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stdout_buffer):
            from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt
            from adaptvqe.hamiltonians import HubbardHamiltonian
            from adaptvqe.pools import OVP_CEO
    except Exception as exc:
        raise RuntimeError(
            "CEO public-code dependencies are not importable from the pinned checkout "
            f"({type(exc).__name__}: {exc})"
        ) from exc
    return {
        "LinAlgAdapt": LinAlgAdapt,
        "HubbardHamiltonian": HubbardHamiltonian,
        "OVP_CEO": OVP_CEO,
    }


def _run_hubbard_l2_public_code(
    checkout_dir: Path,
    *,
    worker_mode: str,
    case_profile: str,
    x_dim: int,
    y_dim: int,
    t: float,
    u: float,
    periodic: bool,
    particle_hole_symmetry: bool,
    threshold: float,
    max_adapt_iter: int,
    max_opt_iter: int,
) -> dict[str, Any]:
    stdout_buffer = io.StringIO()
    mode = str(worker_mode).strip().lower()
    if mode not in _WORKER_MODES:
        return {
            "schema": SCHEMA_VERSION,
            "status": "failed",
            "reason": f"unsupported worker_mode={worker_mode!r}",
            "exception_type": "ValueError",
        }
    try:
        modules = _import_public_code(checkout_dir, stdout_buffer)
    except Exception as exc:
        return {
            "schema": SCHEMA_VERSION,
            "status": "skipped_optional_dependency",
            "reason": str(exc),
            "exception_type": type(exc.__cause__ or exc).__name__,
            "raw_stdout_tail": _tail(stdout_buffer.getvalue()),
        }

    LinAlgAdapt = modules["LinAlgAdapt"]
    HubbardHamiltonian = modules["HubbardHamiltonian"]
    OVP_CEO = modules["OVP_CEO"]
    tetris_enabled = mode == "tetris"
    tetris_progressive_opt = False

    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stdout_buffer):
            hamiltonian = HubbardHamiltonian(
                int(x_dim),
                int(y_dim),
                float(t),
                float(u),
                bool(periodic),
                bool(particle_hole_symmetry),
            )
            pool = OVP_CEO(n=int(hamiltonian.n))
            adapt = LinAlgAdapt(
                pool=pool,
                custom_hamiltonian=hamiltonian,
                verbose=False,
                threshold=float(threshold),
                max_adapt_iter=int(max_adapt_iter),
                max_opt_iter=int(max_opt_iter),
                full_opt=True,
                convergence_criterion="total_g_norm",
                tetris=tetris_enabled,
                progressive_opt=tetris_progressive_opt,
                candidates=1,
                sel_criterion="gradient",
                recycle_hessian=False,
                rand_degenerate=False,
            )
            adapt.run()
    except Exception as exc:
        return {
            "schema": SCHEMA_VERSION,
            "status": "failed",
            "reason": str(exc),
            "exception_type": type(exc).__name__,
            "raw_stdout_tail": _tail(stdout_buffer.getvalue()),
            "traceback_tail": _tail(traceback.format_exc(), max_lines=20),
        }

    data = adapt.data
    evolution = data.evolution
    selected_indices = _int_tuple(getattr(adapt, "indices", ()))
    coefficients = _float_tuple(getattr(adapt, "coefficients", ()))
    gradient_norms = _float_tuple(getattr(evolution, "gradient_norms", ()))
    selected_gradients = _float_tuple(getattr(evolution, "sel_gradients", ()))
    selected_indices_by_iteration = _int_lists(getattr(evolution, "indices", ()))
    coefficients_by_iteration = _float_lists(getattr(evolution, "coefficients", ()))
    operators_added_per_iteration = _additions_per_iteration(
        selected_indices_by_iteration,
        getattr(evolution, "sizes", ()),
    )
    batch_iterations = sum(1 for count in operators_added_per_iteration if int(count) > 1)
    max_operators_added_per_iteration = max(operators_added_per_iteration) if operators_added_per_iteration else 0
    adapt_success = bool(getattr(data, "success", False))
    max_iter = int(getattr(adapt, "max_adapt_iter", 0))
    iteration_counter = int(getattr(data, "iteration_counter", len(selected_indices)))
    stop_reason = "converged" if adapt_success else f"max_adapt_iter_{max_iter}"

    exact_energy = getattr(adapt, "exact_energy", None)
    initial_energy = getattr(data, "initial_energy", None)
    energy_history = _float_tuple(getattr(evolution, "energies", ()))
    nfevs_by_iteration = _int_lists(getattr(evolution, "nfevs", ()))
    ngevs_by_iteration = _int_lists(getattr(evolution, "ngevs", ()))
    nits_by_iteration = _int_lists(getattr(evolution, "nits", ()))
    adapt_history = _adapt_history(
        energies=energy_history,
        exact_energy=exact_energy,
        initial_energy=initial_energy,
        selected_indices_by_iteration=selected_indices_by_iteration,
        coefficients_by_iteration=coefficients_by_iteration,
        operators_added_per_iteration=operators_added_per_iteration,
        gradient_norms=gradient_norms,
        selected_gradients=selected_gradients,
    )
    result = {
        "energy": float(getattr(adapt, "energy")),
        "exact_energy": float(exact_energy) if exact_energy is not None else None,
        "initial_energy": float(initial_energy) if initial_energy is not None else None,
        "energy_history": list(energy_history),
        "adapt_history": adapt_history,
        "selected_operator_count": int(len(selected_indices)),
        "num_parameters": int(len(coefficients)),
        "nfev": _sum_nested(getattr(evolution, "nfevs", None)),
        "ngev": _sum_nested(getattr(evolution, "ngevs", None)),
        "nit": _sum_nested(getattr(evolution, "nits", None)),
        "nfevs_by_iteration": nfevs_by_iteration,
        "ngevs_by_iteration": ngevs_by_iteration,
        "nits_by_iteration": nits_by_iteration,
        "adapt_iterations": iteration_counter,
        "adapt_success": adapt_success,
        "adapt_stop_reason": stop_reason,
        "pool_name": str(getattr(pool, "name", "OVP_CEO")),
        "pool_size": int(getattr(pool, "size")) if getattr(pool, "size", None) is not None else None,
        "selected_indices": list(selected_indices),
        "coefficients": list(coefficients),
        "gradient_norms": list(gradient_norms),
        "selected_gradients": list(selected_gradients),
        "selected_indices_by_iteration": selected_indices_by_iteration,
        "coefficients_by_iteration": coefficients_by_iteration,
        "operators_added_per_iteration": operators_added_per_iteration,
        "max_operators_added_per_iteration": int(max_operators_added_per_iteration),
        "batch_iterations": int(batch_iterations),
        "worker_mode": mode,
        "tetris_enabled": bool(tetris_enabled),
        "tetris_batching_enabled": bool(tetris_enabled),
        "tetris_progressive_opt": bool(tetris_progressive_opt),
        "tetris_candidate_window": "full_pool_nonzero_gradient_window" if tetris_enabled else None,
        "tetris_screening_rule": "disjoint_qubit_support_via_pool_get_qubits" if tetris_enabled else None,
        "external_case_profile": str(case_profile),
        "hubbard_x_dim": int(x_dim),
        "hubbard_y_dim": int(y_dim),
        "hubbard_t": float(t),
        "hubbard_u": float(u),
        "hubbard_periodic": bool(periodic),
        "hubbard_particle_hole_symmetry": bool(particle_hole_symmetry),
        "adapt_threshold": float(threshold),
        "adapt_max_adapt_iter": int(max_adapt_iter),
        "adapt_max_opt_iter": int(max_opt_iter),
        "raw_stdout_tail": _tail(stdout_buffer.getvalue()),
    }
    return {"schema": SCHEMA_VERSION, "status": "completed", "result": result}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the pinned CEO public-code Hubbard L2 smoke and emit strict JSON.")
    parser.add_argument("--checkout-dir", type=Path, required=True)
    parser.add_argument("--worker-mode", choices=_WORKER_MODES, default="ceo")
    parser.add_argument("--case-profile", default="external_hubbard_L2_public_code_default")
    parser.add_argument("--x-dim", type=int, default=2)
    parser.add_argument("--y-dim", type=int, default=1)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--u", type=float, default=4.0)
    parser.add_argument("--periodic", type=_parse_bool, default=True)
    parser.add_argument("--particle-hole-symmetry", type=_parse_bool, default=False)
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument("--max-adapt-iter", type=int, default=6)
    parser.add_argument("--max-opt-iter", type=int, default=300)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkout_dir = Path(args.checkout_dir).expanduser()
    if not checkout_dir.exists():
        return _emit(
            {
                "schema": SCHEMA_VERSION,
                "status": "skipped_optional_dependency",
                "reason": f"CEO reference checkout is missing at {checkout_dir}",
                "exception_type": "FileNotFoundError",
            }
        )
    return _emit(
        _run_hubbard_l2_public_code(
            checkout_dir,
            worker_mode=str(args.worker_mode),
            case_profile=str(args.case_profile),
            x_dim=int(args.x_dim),
            y_dim=int(args.y_dim),
            t=float(args.t),
            u=float(args.u),
            periodic=bool(args.periodic),
            particle_hole_symmetry=bool(args.particle_hole_symmetry),
            threshold=float(args.threshold),
            max_adapt_iter=int(args.max_adapt_iter),
            max_opt_iter=int(args.max_opt_iter),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
