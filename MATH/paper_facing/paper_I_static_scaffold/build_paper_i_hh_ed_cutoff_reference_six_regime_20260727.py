#!/usr/bin/env python3
"""Recompute the six-regime Paper-I Hubbard--Holstein ED cutoff check."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.quantum.ed_hubbard_holstein import (  # noqa: E402
    build_hh_sector_hamiltonian_ed,
    matrix_to_dense,
)


REGIMES = (
    ("weak-weak", 0.25, 0.25),
    ("intermediate-weak", 1.25, 0.25),
    ("strong-weak", 8.0, 0.25),
    ("weak-strong", 0.25, 1.25),
    ("intermediate-strong", 1.25, 1.25),
    ("strong-strong", 8.0, 1.25),
)
CUTOFFS = (3, 7, 9, 10)
PLAN = (
    ROOT
    / "MATH"
    / "paper_facing"
    / "paper_I_static_scaffold"
    / "paper_i_hh_ed_cutoff_reference_six_regime_20260727_plan.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "MATH"
    / "paper_facing"
    / "paper_I_static_scaffold"
    / "paper_i_hh_ed_cutoff_reference_six_regime_20260727.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ed_cell(*, u_over_t: float, lambda_value: float, cutoff: int) -> dict[str, Any]:
    matrix, basis = build_hh_sector_hamiltonian_ed(
        dims=2,
        J=1.0,
        U=float(u_over_t),
        omega0=1.0,
        g=math.sqrt(float(lambda_value) / 2.0),
        n_ph_max=int(cutoff),
        num_particles=(1, 1),
        indexing="blocked",
        boson_encoding="binary",
        pbc=False,
        include_zero_point=True,
        return_basis=True,
        sparse=True,
    )
    dense = matrix_to_dense(matrix)
    hermiticity_residual = float(np.max(np.abs(dense - dense.conj().T)))
    energy = float(np.real(np.linalg.eigvalsh(dense)[0]))
    expected_dimension = 4 * (int(cutoff) + 1) ** 2
    return {
        "M": int(cutoff),
        "E_ED": energy,
        "basis_dimension": int(basis.dimension),
        "expected_basis_dimension": int(expected_dimension),
        "basis_dimension_matches": bool(int(basis.dimension) == expected_dimension),
        "hermiticity_residual_max_abs": hermiticity_residual,
    }


def load_historical(path: Path | None) -> tuple[dict[tuple[float, float, int], float], dict[str, Any]]:
    if path is None:
        return {}, {"status": "not_requested"}
    payload = json.loads(path.read_text())
    values: dict[tuple[float, float, int], float] = {}
    for regime in payload["regimes"]:
        u_value = float(regime["u_over_t"])
        lambda_value = float(regime["lambda"])
        for cutoff in CUTOFFS:
            values[(u_value, lambda_value, cutoff)] = float(
                regime["energies"][str(cutoff)]
            )
    return values, {
        "status": "loaded",
        "basename": path.name,
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--historical-json", type=Path)
    args = parser.parse_args()

    plan = json.loads(PLAN.read_text())
    if not bool(plan.get("execution_authorized")):
        raise RuntimeError("execution_authorized is not true in the plan")

    historical, historical_meta = load_historical(args.historical_json)
    regimes: list[dict[str, Any]] = []
    historical_differences: list[dict[str, Any]] = []

    for name, u_over_t, lambda_value in REGIMES:
        cells = [
            ed_cell(
                u_over_t=u_over_t,
                lambda_value=lambda_value,
                cutoff=cutoff,
            )
            for cutoff in CUTOFFS
        ]
        energies = {int(cell["M"]): float(cell["E_ED"]) for cell in cells}
        working_cutoff = 3 if lambda_value == 0.25 else 7
        regimes.append(
            {
                "name": name,
                "U_over_t": u_over_t,
                "lambda": lambda_value,
                "g_over_t": math.sqrt(lambda_value / 2.0),
                "working_cutoff": working_cutoff,
                "cells": cells,
                "working_vs_M10_abs": abs(
                    energies[working_cutoff] - energies[10]
                ),
                "M9_vs_M10_abs": abs(energies[9] - energies[10]),
            }
        )
        for cutoff in CUTOFFS:
            key = (u_over_t, lambda_value, cutoff)
            if key in historical:
                historical_differences.append(
                    {
                        "regime": name,
                        "M": cutoff,
                        "absolute_difference": abs(
                            energies[cutoff] - historical[key]
                        ),
                    }
                )

    weak_rows = [row for row in regimes if row["lambda"] == 0.25]
    strong_rows = [row for row in regimes if row["lambda"] == 1.25]
    weak_max_row = max(weak_rows, key=lambda row: row["working_vs_M10_abs"])
    strong_max_row = max(strong_rows, key=lambda row: row["working_vs_M10_abs"])
    tail_max_row = max(regimes, key=lambda row: row["M9_vs_M10_abs"])

    all_cells = [cell for row in regimes for cell in row["cells"]]
    max_historical_difference = (
        max(item["absolute_difference"] for item in historical_differences)
        if historical_differences
        else None
    )
    validation = {
        "all_basis_dimensions_match": all(
            bool(cell["basis_dimension_matches"]) for cell in all_cells
        ),
        "max_hermiticity_residual": max(
            float(cell["hermiticity_residual_max_abs"]) for cell in all_cells
        ),
        "historical_comparison": {
            **historical_meta,
            "matched_cells": len(historical_differences),
            "max_absolute_difference": max_historical_difference,
        },
    }
    validation["status"] = (
        "pass"
        if validation["all_basis_dimensions_match"]
        and validation["max_hermiticity_residual"] <= 1.0e-12
        and (
            max_historical_difference is None
            or max_historical_difference <= 1.0e-11
        )
        else "fail"
    )

    payload = {
        "schema": "paper_i_hh_ed_cutoff_reference_six_regime_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": plan["campaign_id"],
        "run_class": plan["run_class"],
        "execution_authorized": True,
        "plan": {
            "path": str(PLAN.relative_to(ROOT)),
            "sha256": sha256(PLAN),
        },
        "source": {
            "repository_commit": plan["source_lock"]["repository_commit"],
            "solver": plan["source_lock"]["solver"],
            "solver_sha256": sha256(ROOT / plan["source_lock"]["solver"]),
            "builder": str(Path(__file__).resolve().relative_to(ROOT)),
            "builder_sha256": sha256(Path(__file__).resolve()),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "physics": plan["physics"],
        "cutoffs": list(CUTOFFS),
        "regimes": regimes,
        "summary": {
            "X_weak_M3_vs_M10_max_abs": weak_max_row["working_vs_M10_abs"],
            "X_regime": weak_max_row["name"],
            "Y_strong_M7_vs_M10_max_abs": strong_max_row["working_vs_M10_abs"],
            "Y_regime": strong_max_row["name"],
            "Z_M9_vs_M10_max_abs": tail_max_row["M9_vs_M10_abs"],
            "Z_regime": tail_max_row["name"],
            "X_per_site": weak_max_row["working_vs_M10_abs"] / 2.0,
            "Y_per_site": strong_max_row["working_vs_M10_abs"] / 2.0,
            "Z_per_site": tail_max_row["M9_vs_M10_abs"] / 2.0,
        },
        "validation": validation,
    }

    if validation["status"] != "pass":
        raise RuntimeError(f"validation failed: {json.dumps(validation, indent=2)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"validation={validation['status']}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
