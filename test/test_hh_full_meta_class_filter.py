"""Tests for HH full_meta class filtering and fidelity emission."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.hubbard_latex_python_pairs import build_hubbard_holstein_hamiltonian
from src.quantum.vqe_latex_python_pairs import half_filled_num_particles

_spec = importlib.util.spec_from_file_location(
    "hardcoded_adapt_pipeline_class_filter",
    str(REPO_ROOT / "pipelines" / "hardcoded" / "adapt_pipeline.py"),
)
_adapt_mod = importlib.util.module_from_spec(_spec)
sys.modules["hardcoded_adapt_pipeline_class_filter"] = _adapt_mod
_spec.loader.exec_module(_adapt_mod)

_build_hh_full_meta_pool = _adapt_mod._build_hh_full_meta_pool
_classify_hh_full_meta_label = _adapt_mod._classify_hh_full_meta_label
_filter_hh_full_meta_pool_by_class = _adapt_mod._filter_hh_full_meta_pool_by_class
_load_hh_full_meta_class_filter_spec = _adapt_mod._load_hh_full_meta_class_filter_spec
_run_hardcoded_adapt_vqe = _adapt_mod._run_hardcoded_adapt_vqe


def _hh_h():
    return build_hubbard_holstein_hamiltonian(
        dims=2,
        J=1.0,
        U=0.5,
        omega0=1.0,
        g=0.2,
        n_ph_max=1,
        boson_encoding="binary",
        repr_mode="JW",
        indexing="blocked",
        pbc=False,
        include_zero_point=True,
    )


def _keep_spec_payload(*keep_classes: str) -> dict[str, object]:
    return {
        "classifier_version": _adapt_mod._HH_FULL_META_CLASSIFIER_VERSION,
        "source_pool": "full_meta",
        "source_problem": "hh",
        "source_num_sites": 2,
        "source_n_ph_max": 1,
        "keep_classes": list(keep_classes),
    }


def _full_meta_pool():
    return _build_hh_full_meta_pool(
        h_poly=_hh_h(),
        num_sites=2,
        t=1.0,
        u=0.5,
        omega0=1.0,
        g_ep=0.2,
        dv=0.0,
        n_ph_max=1,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        paop_r=1,
        paop_split_paulis=False,
        paop_prune_eps=0.0,
        paop_normalization="none",
        num_particles=half_filled_num_particles(2),
    )


class TestHHFullMetaClassFilter:
    def test_parse_accepts_full_meta_class_filter_json(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "adapt_pipeline.py",
                "--problem",
                "hh",
                "--adapt-pool",
                "full_meta",
                "--adapt-pool-class-filter-json",
                "keep_spec.json",
            ],
        )
        args = _adapt_mod.parse_args()
        assert str(args.adapt_pool) == "full_meta"
        assert str(args.adapt_pool_class_filter_json) == "keep_spec.json"

    def test_classifier_covers_every_l2_open_full_meta_label(self):
        pool, _meta = _full_meta_pool()
        families = {_classify_hh_full_meta_label(str(term.label)) for term in pool}
        assert None not in families
        assert families == set(_adapt_mod._HH_FULL_META_ALLOWED_CLASSES)

    def test_filter_keeps_requested_classes_and_reports_counts(self, tmp_path: Path):
        keep_spec_path = tmp_path / "keep_spec.json"
        keep_spec_path.write_text(
            json.dumps(
                _keep_spec_payload(
                    "hh_termwise_quadrature",
                    "uccsd_sing",
                    "paop_cloud_p",
                    "paop_hopdrag",
                    "paop_dbl_p",
                )
            ),
            encoding="utf-8",
        )
        spec = _load_hh_full_meta_class_filter_spec(keep_spec_path)
        pool, _meta = _full_meta_pool()
        filtered_pool, filter_meta = _filter_hh_full_meta_pool_by_class(pool, spec)

        assert filtered_pool
        assert filter_meta["source_json"] == str(keep_spec_path)
        assert set(filter_meta["class_counts_after"].keys()) == set(spec.keep_classes)
        assert filter_meta["dedup_total_after"] < filter_meta["dedup_total_before"]
        assert all(
            _classify_hh_full_meta_label(str(term.label)) in set(spec.keep_classes)
            for term in filtered_pool
        )

    def test_run_emits_filter_metadata_and_exact_state_fidelity(self, tmp_path: Path):
        keep_spec_path = tmp_path / "keep_spec.json"
        keep_spec_path.write_text(
            json.dumps(
                _keep_spec_payload(
                    "hh_termwise_quadrature",
                    "uccsd_sing",
                    "paop_cloud_p",
                    "paop_hopdrag",
                    "paop_dbl_p",
                )
            ),
            encoding="utf-8",
        )
        payload, _psi = _run_hardcoded_adapt_vqe(
            h_poly=_hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="full_meta",
            t=1.0,
            u=0.5,
            dv=0.0,
            boundary="open",
            omega0=1.0,
            g_ep=0.2,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=1,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=10,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            disable_hh_seed=True,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            adapt_pool_class_filter_json=keep_spec_path,
            phase3_symmetry_mitigation_mode="verify_only",
            phase3_enable_rescue=True,
            phase3_lifetime_cost_mode="phase3_v1",
        )

        assert payload["success"] is True
        assert payload["pool_type"] == "phase3_v1"
        assert payload["adapt_pool_class_filter_json"] == str(keep_spec_path)
        assert payload["adapt_pool_class_filter_classifier_version"] == _adapt_mod._HH_FULL_META_CLASSIFIER_VERSION
        assert payload["adapt_pool_class_filter_class_counts_before"] is not None
        assert payload["adapt_pool_class_filter_class_counts_after"] is not None
        assert int(payload["pool_size"]) == sum(payload["adapt_pool_class_filter_class_counts_after"].values())
        assert payload["exact_state_fidelity"] is not None
        assert 0.0 <= float(payload["exact_state_fidelity"]) <= 1.0
        assert payload["exact_state_fidelity_source"] == "phase3_rescue_exact_state"
        kept = set(payload["adapt_pool_class_filter_keep_classes"])
        assert payload["operators"]
        assert all(
            _classify_hh_full_meta_label(str(label)) in kept
            for label in payload["operators"]
        )
        shortlist_rows = payload["continuation"]["phase2_shortlist_rows"]
        assert shortlist_rows
        assert all(
            _classify_hh_full_meta_label(str(row["candidate_label"])) in kept
            for row in shortlist_rows
        )

    def test_main_persists_filter_settings_to_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        keep_spec_path = tmp_path / "keep_spec.json"
        output_json = tmp_path / "run.json"
        keep_spec_path.write_text(
            json.dumps(
                _keep_spec_payload(
                    "hh_termwise_quadrature",
                    "uccsd_sing",
                    "paop_cloud_p",
                    "paop_hopdrag",
                    "paop_dbl_p",
                )
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "adapt_pipeline.py",
                "--problem",
                "hh",
                "--L",
                "2",
                "--t",
                "1.0",
                "--u",
                "0.5",
                "--omega0",
                "1.0",
                "--g-ep",
                "0.2",
                "--n-ph-max",
                "1",
                "--boundary",
                "open",
                "--ordering",
                "blocked",
                "--adapt-pool",
                "full_meta",
                "--adapt-pool-class-filter-json",
                str(keep_spec_path),
                "--adapt-continuation-mode",
                "phase3_v1",
                "--adapt-max-depth",
                "0",
                "--adapt-maxiter",
                "5",
                "--adapt-disable-hh-seed",
                "--phase3-enable-rescue",
                "--skip-pdf",
                "--output-json",
                str(output_json),
                "--t-final",
                "0.0",
                "--num-times",
                "1",
                "--trotter-steps",
                "1",
            ],
        )
        _adapt_mod.main()
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        settings = payload["settings"]

        assert settings["adapt_pool"] == "full_meta"
        assert settings["adapt_pool_class_filter_json"] == str(keep_spec_path)
        assert settings["adapt_pool_class_filter_classifier_version"] == _adapt_mod._HH_FULL_META_CLASSIFIER_VERSION
        assert settings["adapt_pool_class_filter_keep_classes"] == [
            "hh_termwise_quadrature",
            "uccsd_sing",
            "paop_cloud_p",
            "paop_hopdrag",
            "paop_dbl_p",
        ]

    def test_run_rejects_class_filter_for_non_full_meta_pool(self, tmp_path: Path):
        keep_spec_path = tmp_path / "keep_spec.json"
        keep_spec_path.write_text(
            json.dumps(_keep_spec_payload("paop_dbl_p")),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="adapt_pool='full_meta'"):
            _run_hardcoded_adapt_vqe(
                h_poly=_hh_h(),
                num_sites=2,
                ordering="blocked",
                problem="hh",
                adapt_pool="paop_lf_std",
                t=1.0,
                u=0.5,
                dv=0.0,
                boundary="open",
                omega0=1.0,
                g_ep=0.2,
                n_ph_max=1,
                boson_encoding="binary",
                max_depth=0,
                eps_grad=1e-3,
                eps_energy=1e-8,
                maxiter=5,
                seed=7,
                allow_repeats=True,
                finite_angle_fallback=False,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
                disable_hh_seed=True,
                adapt_continuation_mode="phase3_v1",
                adapt_pool_class_filter_json=keep_spec_path,
            )
