#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.time_propagation.cfqm_propagator import CFQMConfig, cfqm_step
from src.quantum.time_propagation.cfqm_schemes import get_cfqm_scheme


class TestCFQMPropagator(unittest.TestCase):
    def test_cfqm_propagator_has_no_pipeline_imports(self) -> None:
        src = (REPO_ROOT / "src/quantum/time_propagation/cfqm_propagator.py").read_text()
        self.assertNotIn("from pipelines.", src)

    def test_cfqm_step_static_single_pauli_matches_exact(self) -> None:
        psi0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex)
        dt = 0.37
        hz = 0.9

        scheme = get_cfqm_scheme("cfqm4")
        cfg = CFQMConfig(backend="dense_expm", normalize=False)
        psi1 = cfqm_step(
            psi=psi0,
            t_abs=0.0,
            dt=dt,
            static_coeff_map={"z": hz},
            drive_coeff_provider=None,
            ordered_labels=["z"],
            scheme=scheme,
            config=cfg,
        )

        expected = np.array([np.exp(-1j * dt * hz), 0.0 + 0.0j], dtype=complex)
        self.assertTrue(np.allclose(psi1, expected, atol=1e-12, rtol=0.0))

    def test_a0_invariance_none_vs_zero_drive(self) -> None:
        psi0 = np.array([0.3 + 0.2j, -0.1 + 0.4j], dtype=complex)
        psi0 = psi0 / np.linalg.norm(psi0)
        dt = 0.23

        scheme = get_cfqm_scheme("cfqm6")
        cfg = CFQMConfig(backend="dense_expm", normalize=False)

        def zero_drive(_t: float) -> dict[str, float]:
            # Includes one known and one unknown label, all zero.
            return {"z": 0.0, "x": 0.0}

        psi_none = cfqm_step(
            psi=psi0,
            t_abs=0.5,
            dt=dt,
            static_coeff_map={"z": 0.7},
            drive_coeff_provider=None,
            ordered_labels=["z"],
            scheme=scheme,
            config=cfg,
        )
        psi_zero = cfqm_step(
            psi=psi0,
            t_abs=0.5,
            dt=dt,
            static_coeff_map={"z": 0.7},
            drive_coeff_provider=zero_drive,
            ordered_labels=["z"],
            scheme=scheme,
            config=cfg,
        )
        self.assertTrue(np.allclose(psi_none, psi_zero, atol=1e-12, rtol=0.0))

    def test_stage_application_is_descending_index(self) -> None:
        psi0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex)
        scheme = {
            "name": "toy",
            "order": 1,
            "c": [0.0],
            "a": [[0.0], [0.0], [0.0]],
            "s_static": [1.0, 2.0, 3.0],
        }
        cfg = CFQMConfig(backend="dense_expm", normalize=False)

        seen: list[float] = []

        def _fake_apply_stage_exponential(**kwargs):
            stage_coeff_map = kwargs["stage_coeff_map"]
            seen.append(float(np.real(stage_coeff_map.get("z", 0.0 + 0.0j))))
            return np.array(kwargs["psi"], copy=True)

        with patch(
            "src.quantum.time_propagation.cfqm_propagator.apply_stage_exponential",
            side_effect=_fake_apply_stage_exponential,
        ):
            _ = cfqm_step(
                psi=psi0,
                t_abs=0.0,
                dt=0.1,
                static_coeff_map={"z": 1.0},
                drive_coeff_provider=None,
                ordered_labels=["z"],
                scheme=scheme,
                config=cfg,
            )

        # Stages k = 2, 1, 0 => static scaling 3, 2, 1.
        self.assertEqual(seen, [3.0, 2.0, 1.0])

    def test_sparse_backend_norm_drift_small_over_many_steps(self) -> None:
        """Diagnostic: sparse expm_multiply keeps norm drift near machine precision."""
        psi = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex)
        scheme = get_cfqm_scheme("cfqm4")
        cfg = CFQMConfig(backend="expm_multiply_sparse", normalize=False)
        ordered_labels = ["x", "y", "z"]
        static_coeff_map = {"x": 0.9}
        total_time = 4.0
        n_steps = 128
        dt = total_time / n_steps

        def drive(t_abs: float) -> dict[str, float]:
            t = float(t_abs)
            return {
                "y": 0.35 * math.sin(2.1 * t),
                "z": 0.27 * math.cos(1.3 * t),
            }

        drifts: list[float] = []
        for k in range(n_steps):
            psi = cfqm_step(
                psi=psi,
                t_abs=float(k) * dt,
                dt=dt,
                static_coeff_map=static_coeff_map,
                drive_coeff_provider=drive,
                ordered_labels=ordered_labels,
                scheme=scheme,
                config=cfg,
            )
            drifts.append(abs(float(np.linalg.norm(psi)) - 1.0))

        self.assertLessEqual(max(drifts), 1e-12)

    def test_drive_edge_cases_empty_zero_and_unknown_labels(self) -> None:
        """Drive provider edge cases are handled deterministically."""
        psi0 = np.array([0.3 + 0.2j, -0.1 + 0.4j], dtype=complex)
        psi0 = psi0 / np.linalg.norm(psi0)
        scheme = get_cfqm_scheme("cfqm4")
        cfg = CFQMConfig(backend="dense_expm", normalize=False)
        ordered_labels = ["z"]
        static_coeff_map = {"z": 0.73}
        t_abs = 0.7
        dt = 0.21

        psi_none = cfqm_step(
            psi=psi0,
            t_abs=t_abs,
            dt=dt,
            static_coeff_map=static_coeff_map,
            drive_coeff_provider=None,
            ordered_labels=ordered_labels,
            scheme=scheme,
            config=cfg,
        )

        psi_empty = cfqm_step(
            psi=psi0,
            t_abs=t_abs,
            dt=dt,
            static_coeff_map=static_coeff_map,
            drive_coeff_provider=lambda _t: {},
            ordered_labels=ordered_labels,
            scheme=scheme,
            config=cfg,
        )
        self.assertTrue(np.allclose(psi_none, psi_empty, atol=1e-12, rtol=0.0))

        psi_zero = cfqm_step(
            psi=psi0,
            t_abs=t_abs,
            dt=dt,
            static_coeff_map=static_coeff_map,
            drive_coeff_provider=lambda _t: {"z": 0.0},
            ordered_labels=ordered_labels,
            scheme=scheme,
            config=cfg,
        )
        self.assertTrue(np.allclose(psi_none, psi_zero, atol=1e-12, rtol=0.0))

        # Unknown labels are ignored (do not fail; do not change evolution).
        psi_unknown = cfqm_step(
            psi=psi0,
            t_abs=t_abs,
            dt=dt,
            static_coeff_map=static_coeff_map,
            drive_coeff_provider=lambda _t: {"x": 0.6, "zz": 1.0, "z": 0.0},
            ordered_labels=ordered_labels,
            scheme=scheme,
            config={
                "backend": "dense_expm",
                "normalize": False,
                "unknown_label_policy": "ignore",
            },
        )
        self.assertTrue(np.allclose(psi_none, psi_unknown, atol=1e-12, rtol=0.0))

    def test_unknown_nontrivial_label_warns_once_and_ignored(self) -> None:
        """Default warn_ignore policy warns once per label and preserves dynamics."""
        psi0 = np.array([0.3 + 0.2j, -0.1 + 0.4j], dtype=complex)
        psi0 = psi0 / np.linalg.norm(psi0)
        scheme = get_cfqm_scheme("cfqm4")
        ordered_labels = ["z"]
        static_coeff_map = {"z": 0.73}
        warned: set[str] = set()

        cfg_warn = {
            "backend": "dense_expm",
            "normalize": False,
            "unknown_label_policy": "warn_ignore",
            "unknown_label_warn_abs_tol": 1e-14,
            "unknown_label_warned_labels": warned,
        }
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            psi_unknown_1 = cfqm_step(
                psi=psi0,
                t_abs=0.70,
                dt=0.21,
                static_coeff_map=static_coeff_map,
                drive_coeff_provider=lambda _t: {"bad_label": 0.3, "z": 0.0},
                ordered_labels=ordered_labels,
                scheme=scheme,
                config=cfg_warn,
            )
            _ = cfqm_step(
                psi=psi0,
                t_abs=0.91,
                dt=0.21,
                static_coeff_map=static_coeff_map,
                drive_coeff_provider=lambda _t: {"bad_label": 0.4, "z": 0.0},
                ordered_labels=ordered_labels,
                scheme=scheme,
                config=cfg_warn,
            )
        warn_msgs = [
            str(w.message)
            for w in rec
            if "Ignoring unknown drive label absent from ordered_labels" in str(w.message)
        ]
        self.assertEqual(len(warn_msgs), 1)

        psi_filtered = cfqm_step(
            psi=psi0,
            t_abs=0.70,
            dt=0.21,
            static_coeff_map=static_coeff_map,
            drive_coeff_provider=lambda _t: {"z": 0.0},
            ordered_labels=ordered_labels,
            scheme=scheme,
            config={"backend": "dense_expm", "normalize": False, "unknown_label_policy": "ignore"},
        )
        self.assertTrue(np.allclose(psi_unknown_1, psi_filtered, atol=1e-12, rtol=0.0))

    def test_unknown_label_strict_mode_raises(self) -> None:
        """Strict policy fails fast when drive emits unknown nontrivial labels."""
        psi0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex)
        scheme = get_cfqm_scheme("cfqm4")
        with self.assertRaises(ValueError):
            _ = cfqm_step(
                psi=psi0,
                t_abs=0.25,
                dt=0.2,
                static_coeff_map={"z": 0.7},
                drive_coeff_provider=lambda _t: {"x": 0.15},
                ordered_labels=["z"],
                scheme=scheme,
                config={
                    "backend": "dense_expm",
                    "normalize": False,
                    "unknown_label_policy": "strict",
                },
            )

    def test_coeff_drop_changes_result_and_preserves_a0_invariance(self) -> None:
        """Nonzero drop tolerance can alter dynamics but should keep A=0 invariance."""
        scheme = get_cfqm_scheme("cfqm4")
        psi = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex)

        ordered_labels = ["x", "z"]
        static_coeff_map = {"x": 1.0, "z": 2.5e-3}
        cfg_no_drop = CFQMConfig(backend="dense_expm", coeff_drop_abs_tol=0.0, normalize=False)
        cfg_drop = CFQMConfig(backend="dense_expm", coeff_drop_abs_tol=1e-2, normalize=False)

        psi_no_drop = cfqm_step(
            psi=psi,
            t_abs=0.0,
            dt=0.9,
            static_coeff_map=static_coeff_map,
            drive_coeff_provider=None,
            ordered_labels=ordered_labels,
            scheme=scheme,
            config=cfg_no_drop,
        )
        psi_drop = cfqm_step(
            psi=psi,
            t_abs=0.0,
            dt=0.9,
            static_coeff_map=static_coeff_map,
            drive_coeff_provider=None,
            ordered_labels=ordered_labels,
            scheme=scheme,
            config=cfg_drop,
        )
        self.assertGreater(float(np.linalg.norm(psi_no_drop - psi_drop)), 1e-6)

        # A=0 invariance with nonzero drop tolerance.
        psi0 = np.array([0.3 + 0.2j, -0.1 + 0.4j], dtype=complex)
        psi0 = psi0 / np.linalg.norm(psi0)
        cfg_drop_small = CFQMConfig(backend="dense_expm", coeff_drop_abs_tol=1e-3, normalize=False)
        ordered_labels_a0 = ["z"]
        static_a0 = {"z": 0.73}
        t_abs = 0.7
        dt = 0.21

        psi_none = cfqm_step(
            psi=psi0,
            t_abs=t_abs,
            dt=dt,
            static_coeff_map=static_a0,
            drive_coeff_provider=None,
            ordered_labels=ordered_labels_a0,
            scheme=scheme,
            config=cfg_drop_small,
        )
        psi_a0 = cfqm_step(
            psi=psi0,
            t_abs=t_abs,
            dt=dt,
            static_coeff_map=static_a0,
            drive_coeff_provider=lambda _t: {"z": 0.0},
            ordered_labels=ordered_labels_a0,
            scheme=scheme,
            config=cfg_drop_small,
        )
        self.assertTrue(np.allclose(psi_none, psi_a0, atol=1e-12, rtol=0.0))


if __name__ == "__main__":
    unittest.main()
