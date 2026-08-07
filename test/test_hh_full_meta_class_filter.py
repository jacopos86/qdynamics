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
from pipelines.static_adapt.builders.hh_pool_presets import clear_hh_pool_cache_memory

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
_load_hh_full_meta_label_filter_spec = _adapt_mod._load_hh_full_meta_label_filter_spec
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
    def test_classifier_recognizes_pure_phonon_families(self):
        assert _classify_hh_full_meta_label("hh_phonon::x(site=0)") == "hh_phonon_linear"
        assert _classify_hh_full_meta_label("hh_phonon::p(site=1)") == "hh_phonon_linear"
        assert _classify_hh_full_meta_label("hh_phonon::n(site=1)") == "hh_phonon_linear"
        assert _classify_hh_full_meta_label("hh_phonon::s(site=0)") == "hh_phonon_quadratic"
        assert _classify_hh_full_meta_label("hh_phonon::x_sq(site=0)") == "hh_phonon_quadratic"
        assert _classify_hh_full_meta_label("hh_phonon::xp_sym(site=1)") == "hh_phonon_quadratic"

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

    def test_class_filter_json_requires_explicit_classifier_version(self, tmp_path: Path):
        keep_spec_path = tmp_path / "keep_spec.json"
        keep_spec_path.write_text(
            json.dumps(
                {
                    "source_pool": "full_meta",
                    "source_problem": "hh",
                    "keep_classes": ["paop_dbl_p"],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="classifier_version"):
            _load_hh_full_meta_class_filter_spec(keep_spec_path)

    def test_label_filter_json_requires_explicit_classifier_version(self, tmp_path: Path):
        drop_spec_path = tmp_path / "drop_spec.json"
        drop_spec_path.write_text(
            json.dumps(
                {
                    "source_pool": "full_meta",
                    "source_problem": "hh",
                    "drop_prefixes": ["hh_fermionic_reusable::"],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="classifier_version"):
            _load_hh_full_meta_label_filter_spec(drop_spec_path)

    def test_class_filter_json_requires_explicit_full_meta_source_pool(self, tmp_path: Path):
        keep_spec_path = tmp_path / "keep_spec.json"
        payload = _keep_spec_payload("paop_dbl_p")
        payload.pop("source_pool")
        keep_spec_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="source_pool='full_meta'"):
            _load_hh_full_meta_class_filter_spec(keep_spec_path)

    def test_class_filter_json_rejects_alias_source_pool(self, tmp_path: Path):
        keep_spec_path = tmp_path / "keep_spec.json"
        payload = _keep_spec_payload("paop_dbl_p")
        payload["source_pool"] = "math_md_full_meta_v1"
        keep_spec_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="source_pool='full_meta'"):
            _load_hh_full_meta_class_filter_spec(keep_spec_path)

    def test_classifier_covers_every_l2_open_full_meta_label(self):
        pool, _meta = _full_meta_pool()
        families = {_classify_hh_full_meta_label(str(term.label)) for term in pool}
        assert None not in families
        assert families.issubset(set(_adapt_mod._HH_FULL_META_ALLOWED_CLASSES))
        assert {
            "hh_termwise_unit",
            "hh_termwise_quadrature",
            "uccsd_sing",
            "uccsd_dbl",
            "hva_layer",
            "hh_hamiltonian_block",
            "hh_fermionic_reusable",
            "paop_disp",
            "paop_hopdrag",
            "paop_dbl_p",
        }.issubset(families)

    def test_builder_reports_broader_full_meta_component_surface(self):
        _pool, meta = _full_meta_pool()
        built = set(meta["built_component_keys"])
        skipped = set(meta["skipped_component_keys"])
        assert {
            "paop_min",
            "paop_std",
            "paop_lf_std",
            "paop_lf2_std",
            "hamiltonian_blocks",
            "hh_fermionic_reusable",
        }.issubset(built)
        assert int(meta["raw_paop_min"]) > 0
        assert int(meta["raw_paop_std"]) > 0
        assert int(meta["raw_paop_lf_std"]) > 0
        assert int(meta["raw_paop_lf2_std"]) > 0
        assert int(meta["raw_hamiltonian_blocks"]) > 0
        assert int(meta["raw_hh_fermionic_reusable"]) > 0
        assert skipped == set()
        assert set(meta["optional_component_keys"]).issubset(built)
        assert int(meta["raw_paop_lf3_std"]) > 0
        assert int(meta["raw_paop_lf4_std"]) > 0
        assert int(meta["raw_paop_bond_disp_std"]) > 0
        assert int(meta["raw_paop_hop_sq_std"]) > 0
        assert int(meta["raw_paop_pair_sq_std"]) > 0
        assert int(meta["raw_vlf_only"]) > 0
        assert int(meta["raw_uccsd_otimes_paop_lf_std"]) > 0
        assert int(meta["raw_uccsd_otimes_paop_bond_disp_std"]) > 0

    def test_filter_keeps_requested_classes_and_reports_counts(self, tmp_path: Path):
        keep_spec_path = tmp_path / "keep_spec.json"
        keep_spec_path.write_text(
            json.dumps(
                _keep_spec_payload(
                    "hh_hamiltonian_block",
                    "hh_fermionic_reusable",
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

    def test_full_meta_dispatch_skips_hva_component_when_class_filter_excludes_hva(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE_DIR", str(tmp_path / "hh_pool_cache"))
        monkeypatch.delenv("STATIC_ADAPT_HH_POOL_CACHE", raising=False)
        clear_hh_pool_cache_memory()
        keep_spec_path = tmp_path / "keep_spec.json"
        keep_spec_path.write_text(
            json.dumps(
                _keep_spec_payload(
                    "hh_hamiltonian_block",
                    "hh_fermionic_reusable",
                    "paop_cloud_p",
                    "paop_hopdrag",
                    "paop_dbl_p",
                )
            ),
            encoding="utf-8",
        )
        spec = _load_hh_full_meta_class_filter_spec(keep_spec_path)
        events: list[tuple[str, dict[str, object]]] = []
        pool, _method, class_meta, _label_meta, _legal_meta = _adapt_mod.build_hh_pool_by_key(
            pool_key_hh="full_meta",
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
            full_meta_class_filter_spec=spec,
            include_legal_subspace_filter_meta=True,
            ai_log=lambda event, **fields: events.append((event, fields)),
        )

        assert pool
        assert all(_classify_hh_full_meta_label(str(term.label)) in set(spec.keep_classes) for term in pool)
        assert class_meta is not None
        assert class_meta["prebuild_skipped_classes"] == ["hva_layer"]
        assert class_meta["prebuild_skipped_component_keys"] == ["hva"]
        assert "hva_layer" in set(class_meta["dropped_classes"])
        assert any(event == "hardcoded_adapt_full_meta_subpool_class_filter_skipped" for event, _fields in events)

    def test_run_emits_filter_metadata_and_exact_state_fidelity(self, tmp_path: Path):
        keep_spec_path = tmp_path / "keep_spec.json"
        keep_spec_path.write_text(
            json.dumps(
                _keep_spec_payload(
                    "hh_hamiltonian_block",
                    "hh_fermionic_reusable",
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
                    "hh_hamiltonian_block",
                    "hh_fermionic_reusable",
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
            "hh_hamiltonian_block",
            "hh_fermionic_reusable",
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
