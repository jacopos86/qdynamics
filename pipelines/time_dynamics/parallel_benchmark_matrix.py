"""Load and expand the frozen Paper-II parallel benchmark matrix."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping


MATRIX_SCHEMA_V1 = "paper_ii_parallel_benchmark_matrix_v1"
DEFAULT_MATRIX_PATH = Path(__file__).with_name("campaigns") / (
    "paper_ii_parallel_benchmark_matrix_v1.json"
)

_FROZEN_MATRIX_ID = "paper_ii_parallel_benchmarks_20260828"
_FROZEN_PAYLOAD_SHA256 = (
    "aaab5e8c025870f972d9170acec8820ecd0092a0ca3e9c48ae39401b282aa4ed"
)
_FROZEN_CELL_COUNTS = {
    "paper_ii_hh_product_v1": 36,
    "siam_nb4_pilot_v1": 1,
    "paper_ii_hubbard_l4_scale_v1": 2,
}
_FROZEN_CELL_COUNT = 39
_FROZEN_SEED_IDENTITY_COUNT = 8
_FROZEN_EXACT_REFERENCE_IDENTITY_COUNT = 39

_FROZEN_ACCURACY_POLICY = {
    "policy_id": "response_normalized_matched_pair_v2",
    "observable_relative_target": 0.05,
    "observable_relative_guard": 0.15,
    "infidelity_scale": 1.0e-4,
    "aggregate": "rms_of_normalized_q_components",
    "matched_factor": 2.0,
    "component_deadband": 0.10,
    "response_scale_floor": 1.0e-12,
    "componentwise_match_required": True,
    "retain_model_absolute_guards": True,
    "absolute_guards_by_lane": {
        "paper_ii_hh_product_v1": {
            "mean_abs_energy_error": 1.0e-2,
            "mean_abs_doublon_error": 5.0e-3,
            "mean_max_abs_site_occupation_error": 5.0e-3,
        },
        "paper_ii_hubbard_l4_scale_v1": {
            "mean_abs_energy_error": 1.0e-2,
            "mean_abs_doublon_error": 5.0e-3,
            "mean_max_abs_site_occupation_error": 5.0e-3,
        },
    },
}


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class BenchmarkCell:
    lane_id: str
    cell_id: str
    model: str
    drive: str
    controller: str
    horizon: float
    physics: Mapping[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "cell_id": self.cell_id,
            "model": self.model,
            "drive": self.drive,
            "controller": self.controller,
            "horizon": self.horizon,
            "physics": dict(self.physics),
        }


@dataclass(frozen=True)
class ParallelBenchmarkMatrix:
    matrix_id: str
    payload: Mapping[str, Any]
    cells: tuple[BenchmarkCell, ...]
    raw_sha256: str

    @classmethod
    def read(cls, path: str | Path = DEFAULT_MATRIX_PATH) -> "ParallelBenchmarkMatrix":
        target = Path(path)
        source_bytes = target.read_bytes()
        payload = json.loads(source_bytes.decode("utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") != MATRIX_SCHEMA_V1:
            raise ValueError("unsupported parallel benchmark matrix schema")
        matrix_id = str(payload.get("matrix_id") or "")
        if not matrix_id:
            raise ValueError("parallel benchmark matrix has no matrix_id")
        common = payload.get("common")
        if not isinstance(common, Mapping):
            raise ValueError("parallel benchmark matrix has no common contract")
        controller = str(common.get("controller") or "")
        methods = tuple(str(value) for value in common.get("methods") or ())
        if controller != "state_motion_5e-5" or methods != ("apm", "avqds"):
            raise ValueError("parallel matrix changed its frozen controller/methods")
        cls._validate_accuracy_policy(payload)
        if matrix_id != _FROZEN_MATRIX_ID or _digest(payload) != _FROZEN_PAYLOAD_SHA256:
            raise ValueError("parallel benchmark frozen matrix payload diverged")

        cells = (
            *cls._hh_cells(payload, controller),
            *cls._siam_cells(payload, controller),
            *cls._hubbard_cells(payload, controller),
        )
        ids = [cell.cell_id for cell in cells]
        if len(set(ids)) != len(ids):
            raise ValueError("parallel benchmark matrix repeats a cell_id")
        cls._validate_frozen_identity_counts(cells)
        result = cls(
            matrix_id=matrix_id,
            payload=payload,
            cells=tuple(cells),
            raw_sha256=hashlib.sha256(source_bytes).hexdigest(),
        )
        result.validate_frozen()
        return result

    def validate_frozen(self) -> None:
        """Revalidate an in-memory matrix against the frozen campaign authority."""

        common = self.payload.get("common")
        if not isinstance(common, Mapping):
            raise ValueError("parallel benchmark matrix has no common contract")
        if (
            self.matrix_id != _FROZEN_MATRIX_ID
            or _digest(self.payload) != _FROZEN_PAYLOAD_SHA256
            or str(common.get("controller") or "") != "state_motion_5e-5"
            or tuple(str(value) for value in common.get("methods") or ())
            != ("apm", "avqds")
        ):
            raise ValueError("parallel benchmark frozen matrix payload diverged")
        self._validate_accuracy_policy(self.payload)
        expected_cells = (
            *self._hh_cells(self.payload, "state_motion_5e-5"),
            *self._siam_cells(self.payload, "state_motion_5e-5"),
            *self._hubbard_cells(self.payload, "state_motion_5e-5"),
        )
        if tuple(self.cells) != tuple(expected_cells):
            raise ValueError("parallel benchmark expanded cells diverged from payload")
        self._validate_frozen_identity_counts(tuple(self.cells))

    @staticmethod
    def _validate_frozen_identity_counts(
        cells: tuple[BenchmarkCell, ...],
    ) -> None:
        lane_counts = {
            lane_id: sum(cell.lane_id == lane_id for cell in cells)
            for lane_id in _FROZEN_CELL_COUNTS
        }
        if lane_counts != _FROZEN_CELL_COUNTS or len(cells) != _FROZEN_CELL_COUNT:
            raise ValueError("parallel benchmark matrix must contain exactly 39 cells")

        seed_keys: set[tuple[str, str, str]] = set()
        for cell in cells:
            if cell.lane_id == "paper_ii_hh_product_v1":
                seed_keys.add(
                    (
                        cell.lane_id,
                        str(cell.physics["n_ph_max"]),
                        str(cell.physics["regime"]),
                    )
                )
            elif cell.lane_id == "siam_nb4_pilot_v1":
                seed_keys.add((cell.lane_id, "nb4", "givens_prequench"))
            elif cell.lane_id == "paper_ii_hubbard_l4_scale_v1":
                seed_keys.add((cell.lane_id, "l4", "u4_half_filled"))
            else:  # pragma: no cover - guarded by the exact lane counts above
                raise ValueError("parallel benchmark matrix contains an unknown lane")
        if len(seed_keys) != _FROZEN_SEED_IDENTITY_COUNT:
            raise ValueError(
                "parallel benchmark matrix must contain exactly 8 seed identities"
            )
        if len(cells) != _FROZEN_EXACT_REFERENCE_IDENTITY_COUNT:
            raise ValueError(
                "parallel benchmark matrix must contain exactly 39 exact-reference "
                "identities"
            )

    @staticmethod
    def _validate_accuracy_policy(payload: Mapping[str, Any]) -> None:
        policy = payload.get("accuracy_policy")
        if not isinstance(policy, Mapping):
            raise ValueError("parallel benchmark matrix has no accuracy policy")
        if dict(policy) != _FROZEN_ACCURACY_POLICY:
            raise ValueError(
                "parallel benchmark matrix changed its frozen complete accuracy policy"
            )

    @staticmethod
    def _hh_cells(
        payload: Mapping[str, Any], controller: str
    ) -> tuple[BenchmarkCell, ...]:
        row = payload.get("hh_product")
        if not isinstance(row, Mapping):
            raise ValueError("parallel benchmark matrix has no hh_product")
        cells = []
        for cutoff, regime, drive in product(
            row.get("phonon_cutoffs") or (),
            row.get("regimes") or (),
            row.get("drives") or (),
        ):
            if not isinstance(regime, Mapping):
                raise ValueError("HH regime rows must be objects")
            physics = {
                "lattice_sites": int(row["lattice_sites"]),
                "t_hop": float(row["t_hop"]),
                "omega0": float(row["omega0"]),
                "boundary": str(row["boundary"]),
                "ordering": str(row["ordering"]),
                "boson_encoding": str(row["boson_encoding"]),
                "n_up": int(row["n_up"]),
                "n_down": int(row["n_down"]),
                "staggered_potential": float(row["staggered_potential"]),
                "n_ph_max": int(cutoff),
                "regime": str(regime["id"]),
                "U": float(regime["U"]),
                "g": float(regime["g"]),
            }
            cell_id = (
                f"hh__nph{int(cutoff)}__{regime['id']}__{str(drive)}"
            )
            cells.append(
                BenchmarkCell(
                    lane_id="paper_ii_hh_product_v1",
                    cell_id=cell_id,
                    model=str(row["model"]),
                    drive=str(drive),
                    controller=controller,
                    horizon=float(row["horizon"]),
                    physics=physics,
                )
            )
        expected = int(row["physical_cell_count"])
        if len(cells) != expected:
            raise ValueError(f"HH product expands to {len(cells)}, expected {expected}")
        return tuple(cells)

    @staticmethod
    def _siam_cells(
        payload: Mapping[str, Any], controller: str
    ) -> tuple[BenchmarkCell, ...]:
        row = payload.get("siam_nb4")
        if not isinstance(row, Mapping):
            raise ValueError("parallel benchmark matrix has no siam_nb4")
        return (
            BenchmarkCell(
                lane_id="siam_nb4_pilot_v1",
                cell_id="siam__nb4__screening_quench",
                model=str(row["model"]),
                drive="hybridization_quench",
                controller=controller,
                horizon=float(row["horizon"]),
                physics={
                    "n_bath": int(row["n_bath"]),
                    "U": float(row["U"]),
                    "V": float(row["V"]),
                    "t_b": float(row["t_b"]),
                    "V_pre_quench": float(row["V_pre_quench"]),
                    "ordering": str(row["ordering"]),
                    "n_up": int(row["n_up"]),
                    "n_down": int(row["n_down"]),
                    "pool_families": tuple(
                        str(value) for value in row["pool_families"]
                    ),
                    "source_epoch": str(row["source_epoch"]),
                    "source_revision": str(row["source_revision"]),
                },
            ),
        )

    @staticmethod
    def _hubbard_cells(
        payload: Mapping[str, Any], controller: str
    ) -> tuple[BenchmarkCell, ...]:
        row = payload.get("hubbard_l4")
        if not isinstance(row, Mapping):
            raise ValueError("parallel benchmark matrix has no hubbard_l4")
        cells = tuple(
            BenchmarkCell(
                lane_id="paper_ii_hubbard_l4_scale_v1",
                cell_id=f"hubbard__l4__u4__{str(drive)}",
                model=str(row["model"]),
                drive=str(drive),
                controller=controller,
                horizon=float(row["horizon"]),
                physics={
                    "lattice_sites": int(row["lattice_sites"]),
                    "boundary": str(row["boundary"]),
                    "n_up": int(row["n_up"]),
                    "n_down": int(row["n_down"]),
                    "U_over_t": float(row["U_over_t"]),
                    "t_hop": float(row["t_hop"]),
                    "ordering": str(row["ordering"]),
                },
            )
            for drive in row.get("drives") or ()
        )
        if len(cells) != int(row["physical_cell_count"]):
            raise ValueError("Hubbard L4 cell count disagrees with its contract")
        return cells

    @property
    def digest(self) -> str:
        """Canonical digest of the complete parsed matrix payload.

        This is intentionally broader than the legacy semantic projection:
        metadata, exclusions, source handoff binding, and every future frozen
        field are part of the campaign identity.
        """

        return self.payload_sha256

    @property
    def payload_sha256(self) -> str:
        return _digest(self.payload)

    @property
    def expanded_semantic_sha256(self) -> str:
        """Digest of the fully expanded execution-relevant projection."""

        return _digest(
            {
                "matrix_id": self.matrix_id,
                "common": self.payload["common"],
                "cells": [cell.to_json_dict() for cell in self.cells],
                "accuracy_policy": self.payload["accuracy_policy"],
                "activation_ladders": self.payload["activation_ladders"],
                "hh_product": self.payload["hh_product"],
                "siam_nb4": self.payload["siam_nb4"],
                "hubbard_l4": self.payload["hubbard_l4"],
                "excluded": self.payload["excluded"],
            }
        )

    @property
    def accuracy_policy_spec_sha256(self) -> str:
        """Digest of the scale-independent frozen admission specification."""

        return _digest(self.payload["accuracy_policy"])

    def canonical_accuracy_policy(
        self,
        lane_id: str,
        response_scales: Mapping[str, Any],
    ) -> "AccuracyPolicy":
        """Build the only admissible policy for a prospective matrix lane.

        Exact-response scales are materialized per reference, while every
        threshold and absolute guard is frozen in this matrix.  Keeping those
        inputs separate prevents a caller from weakening the policy when it
        joins a newly materialized exact reference to the scientific planner.
        """

        from pipelines.time_dynamics.staged_campaign import AccuracyPolicy

        self._validate_accuracy_policy(self.payload)
        policy = self.payload["accuracy_policy"]
        guards = policy["absolute_guards_by_lane"].get(lane_id)
        if not isinstance(guards, Mapping):
            raise ValueError(
                f"accuracy policy has no absolute guards for lane {lane_id!r}"
            )
        return AccuracyPolicy.response_normalized_matched_pair(
            response_scales=dict(response_scales),
            response_relative_target=float(policy["observable_relative_target"]),
            response_relative_guard=float(policy["observable_relative_guard"]),
            matched_accuracy_factor=float(policy["matched_factor"]),
            matched_component_deadband=float(policy["component_deadband"]),
            infidelity_ceiling=float(policy["infidelity_scale"]),
            absolute_energy_error_ceiling=float(
                guards["mean_abs_energy_error"]
            ),
            absolute_doublon_error_ceiling=float(
                guards["mean_abs_doublon_error"]
            ),
            absolute_site_occupation_error_ceiling=float(
                guards["mean_max_abs_site_occupation_error"]
            ),
            response_scale_floor=float(policy["response_scale_floor"]),
        )

    @property
    def source_file_sha256(self) -> str:
        """Compatibility alias; serialized bindings must use ``raw_sha256``."""

        return self.raw_sha256

    @property
    def semantic_digest(self) -> str:
        """Compatibility alias for ``expanded_semantic_sha256``."""

        return self.expanded_semantic_sha256

    def cells_for(self, lane_id: str) -> tuple[BenchmarkCell, ...]:
        return tuple(cell for cell in self.cells if cell.lane_id == lane_id)


__all__ = [
    "BenchmarkCell",
    "DEFAULT_MATRIX_PATH",
    "MATRIX_SCHEMA_V1",
    "ParallelBenchmarkMatrix",
]
