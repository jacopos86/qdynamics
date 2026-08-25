from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = REPO_ROOT / "archive/paper_i_static_adapt_legacy_20260727"
MANIFEST_PATH = ARCHIVE_ROOT / "MANIFEST.json"
RETIRED_TETRIS_ALGORITHM_ID = "static_tetris_qubit_adapt_vqe"
RETIRED_TETRIS_SELECTION_MODE = "tetris_disjoint_benchmark"

EXPECTED_ENTRIES = {
    "pipelines/static_adapt/adapt_pipeline_legacy_20260322.py": {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/code/"
            "legacy_adapt_executors/adapt_pipeline_legacy_20260322.py.txt"
        ),
        "family": "legacy_adapt_executors",
        "sha256": "ece9a4e9b5c0e7eeea0bd0fe56b62a77301e20e9128fccf03f5b6ed14eae7279",
    },
    "pipelines/static_adapt/compare_adapt_current_vs_legacy_20260322.py": {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/code/"
            "legacy_adapt_executors/compare_adapt_current_vs_legacy_20260322.py.txt"
        ),
        "family": "legacy_adapt_executors",
        "sha256": "e353583b51242d0169b452c29f52c0116747edadd735c84878680c18b6369c36",
    },
    "pipelines/static_adapt/sr_snake/_legacy_adapter.py": {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/code/"
            "legacy_adapt_executors/_legacy_adapter.py.txt"
        ),
        "family": "legacy_adapt_executors",
        "sha256": "e00a6a985cccb5a63c5b7dc9cda17bbf1ac269f30161390fcf7982ce5f2c6af3",
    },
    "pipelines/static_adapt/hh_continuation_scoring_legacy_bridge.py": {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/code/"
            "legacy_adapt_executors/hh_continuation_scoring_legacy_bridge.py.txt"
        ),
        "family": "legacy_adapt_executors",
        "sha256": "a07657b5ba9f5bbe3e0e149a6eec14862df24b7426ef4b38336262975ef0bc72",
    },
    "test/test_legacy_phase3_geometry_bridge.py": {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/tests/"
            "legacy_adapt_executors/test_legacy_phase3_geometry_bridge.py.txt"
        ),
        "family": "legacy_adapt_executors",
        "sha256": "2663e9b927c8ccd8ab4a97c8015626d0e9ff468eeeb88231bad3aa9918a80eea",
    },
    "pipelines/hardcoded/adapt_pipeline.py": {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/code/"
            "paper_i_hardcoded_aliases/adapt_pipeline.py.txt"
        ),
        "family": "paper_i_hardcoded_aliases",
        "sha256": "93c0e91cd01981f5bfa1e9d1434b74296395ca0837f2901ca66ad18ac63dd42f",
    },
    "pipelines/hardcoded/adapt_circuit_cost.py": {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/code/"
            "paper_i_hardcoded_aliases/adapt_circuit_cost.py.txt"
        ),
        "family": "paper_i_hardcoded_aliases",
        "sha256": "bfa8994f68caed6266718cb14bb81409a7a4bc0d9f27da1a27b32cc48b305df2",
    },
    "pipelines/hardcoded/hh_continuation_generators.py": {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/code/"
            "paper_i_hardcoded_aliases/hh_continuation_generators.py.txt"
        ),
        "family": "paper_i_hardcoded_aliases",
        "sha256": "8c6292c5c71f67312bdc32afc8d3908a83cb550b8f2d8871ed7f7183824e6570",
    },
    "pipelines/hardcoded/hh_continuation_scoring.py": {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/code/"
            "paper_i_hardcoded_aliases/hh_continuation_scoring.py.txt"
        ),
        "family": "paper_i_hardcoded_aliases",
        "sha256": "f25b2ae3f4037758c5f1942e6e3e0c75df04f9c5c7008d8b8dacfa9f150aa492",
    },
    "pipelines/hardcoded/hh_continuation_symmetry.py": {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/code/"
            "paper_i_hardcoded_aliases/hh_continuation_symmetry.py.txt"
        ),
        "family": "paper_i_hardcoded_aliases",
        "sha256": "5f61b9c43c253fb81bc354aace4e015f0c4f06a1e8aa8a48b24a43a11b341e01",
    },
    "pipelines/hardcoded/hh_continuation_types.py": {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/code/"
            "paper_i_hardcoded_aliases/hh_continuation_types.py.txt"
        ),
        "family": "paper_i_hardcoded_aliases",
        "sha256": "f24b1a670179ec17c05132b3b65f9541db54ffd888429951f7e17d6aaaf41f4c",
    },
    "pipelines/hardcoded/imported_artifact_resolution.py": {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/code/"
            "paper_i_hardcoded_aliases/imported_artifact_resolution.py.txt"
        ),
        "family": "paper_i_hardcoded_aliases",
        "sha256": "7e8adacf65fb7be8da6ad19ddded10f5fff5c547ace3e289d4cf7b75003396de",
    },
}

for _original_path, _archive_suffix, _sha256_value in (
    ("pipelines/static_adapt/formal_manifold_exact_backend.py", "code/fm_snake/formal_manifold_exact_backend.py.txt", "4375f59163d65b0eb977d030aad5866df2ff5775f9bf61137e9236ad576c2bfb"),
    ("pipelines/static_adapt/formal_manifold_local_campaign.py", "code/fm_snake/formal_manifold_local_campaign.py.txt", "5aef13b365c3af2d7dab08ef471f7373aaadb62ebde8a391e5e2fb83016990ef"),
    ("pipelines/static_adapt/formal_manifold_pareto_campaign.py", "code/fm_snake/formal_manifold_pareto_campaign.py.txt", "a12a63ee14186879a7f562e71ee146f14617d73741068897b15117a62b6713de"),
    ("pipelines/static_adapt/formal_manifold_sr_source_locked_campaign.py", "code/fm_snake/formal_manifold_sr_source_locked_campaign.py.txt", "e5354059faf958db5ac1f7c29e1eebe93f9131891a19f3637ff1fd208a5c55af"),
    ("pipelines/static_adapt/formal_manifold_outer_information.py", "code/fm_snake/formal_manifold_outer_information.py.txt", "b1b000fba8b3a6b615d820a3dfca5d00bb65be2fa40380d3e124a987771360be"),
    ("pipelines/static_adapt/formal_manifold_sr_v3_outer_bridge.py", "code/fm_snake/formal_manifold_sr_v3_outer_bridge.py.txt", "fb8f18d159e19ce3b46fdabf7bcbab3a76611dadf65fbf027837ab7e551c2c5d"),
    ("pipelines/static_adapt/formal_manifold_warm_start.py", "code/fm_snake/formal_manifold_warm_start.py.txt", "0d6cfcd35cb1a67236c63a3f0c89ac432c178a5c7518d4db5244759d9fa258d8"),
    ("pipelines/static_adapt/formal_manifold_route_profile.py", "code/fm_snake/formal_manifold_route_profile.py.txt", "9cf2a8e3d72a8a0d45c17b75e8fd7edc20cb2823c7a69707c785cf4be17de694"),
    ("pipelines/static_adapt/reclose_formal_manifold_query_accounting.py", "code/fm_snake/reclose_formal_manifold_query_accounting.py.txt", "334c3cdb87c151a3a72192e113248b7ca9e7298f940116dd08c7e319e5c07937"),
    ("pipelines/exact_bench/paper_i_hh_fm_vs_append_fm_first_hit.py", "code/fm_snake/paper_i_hh_fm_vs_append_fm_first_hit.py.txt", "f1b34cce3fb1c6bf4e9c04c695a8848f494d0036a4d4b675d5b3e0b9ebadeb05"),
    ("pipelines/exact_bench/paper_i_hh_append_fm_first_hit_campaign.py", "code/fm_snake/paper_i_hh_append_fm_first_hit_campaign.py.txt", "6f354562fe376391fc49554fb6192e937a95ae53ed74e6ec537ec1f2967ce673"),
    ("pipelines/exact_bench/configs/paper_i_hh_fm_historical_qbroyd_off_config_20260712.json", "code/fm_snake/paper_i_hh_fm_historical_qbroyd_off_config_20260712.json.txt", "76e7533ad534e5ff1cffdae4df41e1cdb79b6e0061d113b8cf07cf067ad01fe0"),
    ("test/test_static_adapt_formal_manifold_local_campaign.py", "tests/fm_snake/test_static_adapt_formal_manifold_local_campaign.py.txt", "86e047ce671cda4ba197fdd2490be558d97e7a657ea942ab5fa0df134bb06716"),
    ("test/test_static_adapt_formal_manifold_outer_active_integration.py", "tests/fm_snake/test_static_adapt_formal_manifold_outer_active_integration.py.txt", "2a3901d20c835ea00c8e12001ea8819f71e9d8d477a10f90dd2be667406d5371"),
    ("test/test_static_adapt_formal_manifold_outer_information.py", "tests/fm_snake/test_static_adapt_formal_manifold_outer_information.py.txt", "6e31d4045ac18bc74b55718bb6bb7d1bb32477977a26ac68215512fd353a64e8"),
    ("test/test_static_adapt_formal_manifold_outer_shadow_integration.py", "tests/fm_snake/test_static_adapt_formal_manifold_outer_shadow_integration.py.txt", "c3d0c8035d41eb587aba82c00bada3a931a252ddb137c072495956dadaed3824"),
    ("test/test_static_adapt_formal_manifold_pareto_campaign.py", "tests/fm_snake/test_static_adapt_formal_manifold_pareto_campaign.py.txt", "00ca3da430f30c9ffca9072594ee1b0144be9b599e0b2b8ff1c65934b70a5469"),
    ("test/test_static_adapt_formal_manifold_pure_fm_beam.py", "tests/fm_snake/test_static_adapt_formal_manifold_pure_fm_beam.py.txt", "a9587f9f63119e1c1e210650eacd5b8da168504394dbfaa8b5ff8e8702d8ffd2"),
    ("test/test_static_adapt_formal_manifold_route_cli.py", "tests/fm_snake/test_static_adapt_formal_manifold_route_cli.py.txt", "63ea6d2a32521b93dd503cb442bbedc23d104eaf429975337472022c87cbbd87"),
    ("test/test_static_adapt_formal_manifold_route_integration.py", "tests/fm_snake/test_static_adapt_formal_manifold_route_integration.py.txt", "fede0ebb9a8fa5fdd51e24a9f0460eb59d8a98c940c53cdf9e127408456e8040"),
    ("test/test_static_adapt_formal_manifold_route_profile.py", "tests/fm_snake/test_static_adapt_formal_manifold_route_profile.py.txt", "461cd6d7abacfa390bac8808bf58cc46bf077768f7e1bb1847930cc4011764d0"),
    ("test/test_static_adapt_formal_manifold_sr_selector.py", "tests/fm_snake/test_static_adapt_formal_manifold_sr_selector.py.txt", "de4daa9d2c7bad8e0de9e1c8809f6fe6c627d3b614ebe08eabc9da2fa9fb8a13"),
    ("test/test_static_adapt_formal_manifold_sr_source_locked_campaign.py", "tests/fm_snake/test_static_adapt_formal_manifold_sr_source_locked_campaign.py.txt", "2b7782ad9930a95f095f3ca2980738988e9550c6770e5d9f2687f04abf197c37"),
    ("test/test_static_adapt_formal_manifold_sr_v3_outer_bridge.py", "tests/fm_snake/test_static_adapt_formal_manifold_sr_v3_outer_bridge.py.txt", "0da20a4875036bbeca09880f0c5c3208779bbe9b3f0263de692e46cb702e0ed2"),
    ("test/test_static_adapt_formal_manifold_warm_start.py", "tests/fm_snake/test_static_adapt_formal_manifold_warm_start.py.txt", "24e11289c4b4da4abeef3d7e8d15d51b67f9042c7c7757a549387b202263eb09"),
    ("test/test_paper_i_hh_fm_vs_append_fm_first_hit.py", "tests/fm_snake/test_paper_i_hh_fm_vs_append_fm_first_hit.py.txt", "a0b85a9d971d3b557f716ffb36e5ffdf5e9210bcbbff00726930621399c0b0b9"),
    ("test/test_paper_i_hh_append_fm_first_hit_campaign.py", "tests/fm_snake/test_paper_i_hh_append_fm_first_hit_campaign.py.txt", "1f55b72ba3966e83cd874caa873cb41955fb111322728d69f549dc237ee99025"),
    ("test/test_static_adapt_reclose_formal_manifold_query_accounting.py", "tests/fm_snake/test_static_adapt_reclose_formal_manifold_query_accounting.py.txt", "bd731c6acc22da55dc761d91451770e606b40936e784d6d4719e4ff729da96f5"),
    ("test/test_static_adapt_sr_outer_information_overlay_integration.py", "tests/fm_snake/test_static_adapt_sr_outer_information_overlay_integration.py.txt", "0a3756af1a5e25253f0352113fd5fd69a6a9fe2862b94a5821d394e51177fa42"),
):
    EXPECTED_ENTRIES[_original_path] = {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/" + _archive_suffix
        ),
        "family": "fm_snake",
        "sha256": _sha256_value,
    }

for _original_path, _archive_suffix, _sha256_value in (
    ("pipelines/static_adapt/optimization/__init__.py", "code/optuna_calibration/__init__.py.txt", "93b41687f3bdde82f779b5a4d841ffac32e4e5bb2a3933ae68a2a05e3b562fa3"),
    ("pipelines/static_adapt/optimization/hh_optuna_evidence_ledger.py", "code/optuna_calibration/hh_optuna_evidence_ledger.py.txt", "fae57c453c1eda631f64b8003f1e12ac1e5dbad87607832e46d7cd7004f9e9d2"),
    ("pipelines/static_adapt/optimization/hh_snake_interpretable_ml_analysis.py", "code/optuna_calibration/hh_snake_interpretable_ml_analysis.py.txt", "518302e22f26a99ed9af806aa56013d159208ac752c3f48aef0de9a3a17d2fd3"),
    ("pipelines/static_adapt/optimization/hh_snake_shallow_feature_extract.py", "code/optuna_calibration/hh_snake_shallow_feature_extract.py.txt", "dacdf2dea7af9dc7393930bacb6f52ab6c2499d06b0fa3232a1369566d5ca01a"),
    ("pipelines/static_adapt/optimization/phase3_policy_optuna.py", "code/optuna_calibration/phase3_policy_optuna.py.txt", "29866bcbef39054f5eaf5a6f33752f0a56615571493bd2b7815d7e6d5116b32d"),
    ("pipelines/static_adapt/optimization/phase3_robustness_gate.py", "code/optuna_calibration/phase3_robustness_gate.py.txt", "2edda356a2b037bd6c374a9dabe4ae7f8d47b5570aa4f2ffdee84bcef96b3101"),
    ("pipelines/static_adapt/optimization/staged_adapt_optuna.py", "code/optuna_calibration/staged_adapt_optuna.py.txt", "6f2ee0316e8c953289f4de133349a3f01074645d16cbd7d458a37869ab2462aa"),
    ("pipelines/exact_bench/paper_i_hh_full_policy_warm_start.py", "code/optuna_calibration/paper_i_hh_full_policy_warm_start.py.txt", "24f4d0519628a3a9b5f082b112c1e8b16c5452406f6de2b2ec37fe1a22acc8de"),
    ("pipelines/exact_bench/paper_i_hh_live_optuna_overlay_refresh.py", "code/optuna_calibration/paper_i_hh_live_optuna_overlay_refresh.py.txt", "c11d88d62a82eb5d48823e8aa89ff5e6bb2f535d5fece324750642090b8fbd54"),
    ("pipelines/exact_bench/paper_i_hh_local_optuna_status.py", "code/optuna_calibration/paper_i_hh_local_optuna_status.py.txt", "8b86fbda87e4059c1c58134a3415cb95d5afdee8ebc0f7077d9a8e1c0defc903"),
    ("pipelines/exact_bench/paper_i_hh_local_optuna_supervisor.py", "code/optuna_calibration/paper_i_hh_local_optuna_supervisor.py.txt", "8854bb45e5a05344eb7fe996e4c2ff9011f83a49b4f827a625c31e0e0fd9f8ee"),
    ("pipelines/exact_bench/paper_i_hh_optuna_artifact_offload.py", "code/optuna_calibration/paper_i_hh_optuna_artifact_offload.py.txt", "4dcb2c11b34352bf7a3c77f56f72488eee0ff63e24e437263fae3de3d5fcf92a"),
    ("pipelines/exact_bench/paper_i_hh_review_source_locked_rerun.py", "code/optuna_calibration/paper_i_hh_review_source_locked_rerun.py.txt", "7d24c40ce46140863003446c61adfea2bd260e4a95cd858e3171f720325e5eae"),
    ("pipelines/exact_bench/paper_i_hh_route_a_optuna.py", "code/optuna_calibration/paper_i_hh_route_a_optuna.py.txt", "7d845096e1574fe8bddd30aeac59959b04a1722316c687983ed4047de0baf136"),
    ("pipelines/exact_bench/paper_i_hh_snake_fullpolicy_warm_start.py", "code/optuna_calibration/paper_i_hh_snake_fullpolicy_warm_start.py.txt", "a6b0ad46021ad08858c4eb5bac74be4dce5844c5907686000ca8d6bd26d37049"),
    ("pipelines/exact_bench/paper_i_hh_snake_global_policy_optuna.py", "code/optuna_calibration/paper_i_hh_snake_global_policy_optuna.py.txt", "47ef5ff91a2e07de3ad2d6b0982838a1fb25607b4ebfd6cf4dfc5e355c0a0426"),
    ("pipelines/exact_bench/paper_i_hh_u8_comparator_spsa_optuna.py", "code/optuna_calibration/paper_i_hh_u8_comparator_spsa_optuna.py.txt", "55a8599ff749b55eda692dc9e4f3e3910c06772124cc118fe13e275a9a45ea9c"),
    ("pipelines/exact_bench/hh_cost_energy_optuna.py", "code/optuna_calibration/hh_cost_energy_optuna.py.txt", "45c288ebe721e038968a6e1dfd2fb870316b3894c81c0bcf2699851e5b413a8e"),
    ("pipelines/exact_bench/paper_i_hh_speed_optuna.py", "code/optuna_calibration/paper_i_hh_speed_optuna.py.txt", "d35065ac188b9b614248d3c70355bc08b6628493da4ce829ef6fa61d6b610fd7"),
    ("pipelines/exact_bench/hh_local_seed_admission.py", "code/optuna_calibration/hh_local_seed_admission.py.txt", "c34bf8577e7b09bb626ca4ad763aabb620b3eaf38c637e899860f48df6e45dac"),
    ("test/test_hh_same_cutoff_overnight.py", "tests/optuna_calibration/test_hh_same_cutoff_overnight.py.txt", "ad4d43b53ab2cb8917de46c3fcdeb0ab06f5ad9836d368afa034f74758ede8ee"),
    ("test/test_hh_snake_interpretable_ml_analysis.py", "tests/optuna_calibration/test_hh_snake_interpretable_ml_analysis.py.txt", "4d06e44fde7bf323b81bbdd61407ea17caa487f1625235a43ab02a88cff9edcc"),
    ("test/test_hh_snake_shallow_feature_extract.py", "tests/optuna_calibration/test_hh_snake_shallow_feature_extract.py.txt", "34f84733a4bcfcb40d596be6980f970b8352ae25b28eee0b43de349f554fb543"),
    ("test/test_paper_i_hh_full_policy_warm_start.py", "tests/optuna_calibration/test_paper_i_hh_full_policy_warm_start.py.txt", "effdad36b544e6cd9ed3fcda2a8c38335994e291b9cc55f2d5433362e92ab006"),
    ("test/test_paper_i_hh_route_a_optuna.py", "tests/optuna_calibration/test_paper_i_hh_route_a_optuna.py.txt", "ff2983809e9b64ab6065099629e4a8e41a2735d7ddbb2962881b4f7b432c0b3e"),
    ("test/test_paper_i_hh_same_cutoff_contract.py", "tests/optuna_calibration/test_paper_i_hh_same_cutoff_contract.py.txt", "4a0811872e368f2ad9f51a9b25378bcccf62f8740b8d4bc7ebe2f5d7f1d9d901"),
    ("test/test_paper_i_hh_snake_fullpolicy_warm_start.py", "tests/optuna_calibration/test_paper_i_hh_snake_fullpolicy_warm_start.py.txt", "d52b84726a123894d5accfa8cbed484fe7432752ed43b1273dc5ad08f93aa71b"),
    ("test/test_paper_i_snake_sector_contract_sentinels.py", "tests/optuna_calibration/test_paper_i_snake_sector_contract_sentinels.py.txt", "aadb66fcdc35bf9bb9db454b2e0c10ffefb28e03092d2a1f2b7ed91bbc39feb3"),
    ("test/test_phase3_clean_phonon_ladder_records.py", "tests/optuna_calibration/test_phase3_clean_phonon_ladder_records.py.txt", "0d813cc77dd210f0d3bc0a75df7e5395caa91606fd0ed65a76426bbc67e408e9"),
    ("test/test_phase3_policy_optuna.py", "tests/optuna_calibration/test_phase3_policy_optuna.py.txt", "aabc2a602a63b4261f3ef9258f78c0021e5e15d88aa5bffb39ce065c0a8a90bd"),
    ("test/test_phase3_reset_records.py", "tests/optuna_calibration/test_phase3_reset_records.py.txt", "c591dadd7bcad6d920aab35c4b6d681dbf4e2b3878d4cfc06b82bbb9b462cc56"),
    ("test/test_staged_adapt_optuna.py", "tests/optuna_calibration/test_staged_adapt_optuna.py.txt", "e8505a1cc88291b1d7e9a4c1ec2498192e4f7a1304db72a24d91a520ed8eab6f"),
    (
        "test/chtc/test_paper_i_hh_strong_weak_anchor_ablation_records.py",
        (
            "tests/optuna_calibration/"
            "test_paper_i_hh_strong_weak_anchor_ablation_records.py.txt"
        ),
        "ba892318c258f86fa3f75882d76736b3cae706aefd9f34971aeb0ea5184ee850",
    ),
):
    EXPECTED_ENTRIES[_original_path] = {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/" + _archive_suffix
        ),
        "family": "optuna_calibration",
        "sha256": _sha256_value,
    }

for _original_path, _archive_suffix, _sha256_value in (
    (
        "pipelines/static_adapt/paper_i_runner.py",
        "code/old_paper_i_runners/paper_i_runner.py.txt",
        "508464a14a5ac61323af3b0c88362849576d43f78f2268b532819826c82126b6",
    ),
    (
        "pipelines/exact_bench/paper_i_hh_powell_pareto.py",
        "code/old_paper_i_runners/paper_i_hh_powell_pareto.py.txt",
        "0edcd3c2a2818a98f47e22301b3ff89ff610e2fedc0b31202701fd36cc64e343",
    ),
    (
        "test/test_static_adapt_paper_i_runner.py",
        "tests/old_paper_i_runners/test_static_adapt_paper_i_runner.py.txt",
        "0d75aa421f174a41dda8fbb38530bc715ff0f18fa7f6fb0d9a6c496d7bb32d9b",
    ),
    (
        "test/test_paper_i_hh_powell_pareto.py",
        "tests/old_paper_i_runners/test_paper_i_hh_powell_pareto.py.txt",
        "d5374e199945a0fcc23a55debfbad2efed6bd2219b1c86299645377cf16a7771",
    ),
):
    EXPECTED_ENTRIES[_original_path] = {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/" + _archive_suffix
        ),
        "family": "old_paper_i_runners",
        "sha256": _sha256_value,
    }

for _original_path, _archive_suffix, _sha256_value in (
    (
        "pipelines/exact_bench/hh_path_a_autopilot.py",
        "code/optuna_calibration/hh_path_a_autopilot.py.txt",
        "44275cd6a5672da8b8834d0110bd0edb960f6bf5e9a3dc34c73ee44284706164",
    ),
    (
        "pipelines/exact_bench/hh_path_a_ledger.py",
        "code/optuna_calibration/hh_path_a_ledger.py.txt",
        "3cbfac3e7de0c6ff1f6446a3501c5de062bc366377ec08bcc73c28d835e59d22",
    ),
    (
        "pipelines/exact_bench/hh_path_a_tmux_wrapper.py",
        "code/optuna_calibration/hh_path_a_tmux_wrapper.py.txt",
        "109e0070405d5c4214d78ab8bb7759c1b323ad4f8cd2bc36da26dc7d6ee1cfd8",
    ),
    (
        "test/test_hh_path_a_ledger.py",
        "tests/optuna_calibration/test_hh_path_a_ledger.py.txt",
        "e0ba9b55dc84d77edd49fe1d30378c55460825c5751c6a55d5c2949bef29673a",
    ),
    (
        "test/test_hh_path_a_tmux_wrapper.py",
        "tests/optuna_calibration/test_hh_path_a_tmux_wrapper.py.txt",
        "2a6270d20d6327e6b4944dae808d780863b8072e1f6e7dbd76d7c9f62488147a",
    ),
):
    EXPECTED_ENTRIES[_original_path] = {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/" + _archive_suffix
        ),
        "family": "optuna_calibration",
        "sha256": _sha256_value,
    }

PATH_A_FRAGMENT_ORIGINAL = "test/test_path_autopilots.py#path_a_fragment"
PATH_A_FRAGMENT_ARCHIVE = (
    "archive/paper_i_static_adapt_legacy_20260727/tests/"
    "optuna_calibration/test_path_autopilots_path_a_fragment.py.txt"
)
PATH_A_FRAGMENT_SHA256 = (
    "8d4eaabfd44ae0e49a60f6e672bf4e15810386bebe4b2b7b69dd4ad146752d7b"
)
PATH_B_RETAINED_CONTENT_SHA256 = (
    "8ba510ea0c688bda23a925ec493b97ebc203bcc89006cbfe38d6d0ca4d67f3fe"
)

PHASE_LIVE_EXPECTED_ENTRIES = {
    (
        "pipelines/scaffold/hh_continuation_stage_control.py"
        "#phase_live_hysteresis_fragments"
    ): {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/code/"
            "phase_live_hysteresis/"
            "stage_controller_phase_live_hysteresis.py.txt"
        ),
        "sha256": (
            "f803fadbaae12aece7e660600e0f0ee15f2f07506709d5f5a3008948eee62eb3"
        ),
        "source_file_sha256": (
            "a7f384b70d664b773d929cf506613e9272be9843646126d1e4451c2df3c21d61"
        ),
    },
    (
        "pipelines/static_adapt/controller_phase_state.py"
        "#phase_live_helpers"
    ): {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/code/"
            "phase_live_hysteresis/controller_phase_live_helpers.py.txt"
        ),
        "sha256": (
            "f4b3585e95b9aacd89fb4ddd332f8627b7858c52a09191406658848b2af1f05d"
        ),
        "source_file_sha256": (
            "b5bc45d2d3e40cefd891f21fe69a84f9d6e8b9cd24b53a8357997fb5176c81c3"
        ),
    },
    "test/test_static_adapt_no_batch_terminal_phase.py": {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/tests/"
            "phase_live_hysteresis/"
            "test_static_adapt_no_batch_terminal_phase.py.txt"
        ),
        "sha256": (
            "d2ce47ebae18fd257cce676de9eec1e6a072372debfbd882e1573a6b531f8c19"
        ),
    },
}

AMPLITUDE_FRAGMENT_EXPECTATIONS = {}
for (
    _source_path,
    _snapshot_name,
    _snapshot_sha256,
    _source_sha256,
) in (
    (
        "pipelines/scaffold/hh_continuation_pruning.py",
        "hh_continuation_pruning.py.txt",
        "207012f213b5c09cd76f62434408aa5bee3bac5df6947a2b73fe4eea098d2520",
        "3b8be9adce5e52d7beab8fc66bcb4e2252327821c50eb4a2e82c5a1aee0f7ada",
    ),
    (
        "pipelines/scaffold/hh_continuation_types.py",
        "hh_continuation_types.py.txt",
        "15db32b5bd17c6128591623457300d77f7f4dbdc9cd4fa7d454e8d6ed14bccb1",
        "8e46df8d859d54695f98e4b8f9157d38878c8c805c326cd88e42fc7c0ab851fb",
    ),
    (
        "pipelines/static_adapt/cli_config.py",
        "cli_config.py.txt",
        "f395637a65e7bac0fee697215153ce3f3eddf0bbc2e67d706090029a54751550",
        "e632a654952449505fae9dea16e896bda72a079d78a99312f6c8857ab80369c7",
    ),
    (
        "pipelines/static_adapt/adapt_pipeline.py",
        "adapt_pipeline.py.txt",
        "31f1d9786175e2b493ca529c4dd4c2563893676cbee77bd466531882fe8b3ab4",
        "e74d1df68c25921b84e4d22e367bfd7c6b54feb03644f1aacba9a3063339a21c",
    ),
    (
        "pipelines/static_adapt/prune_risk_dataset.py",
        "prune_risk_dataset.py.txt",
        "1c344e7a8e1f42a92834f300a0748b11550144f5dd384dcb5b7fbcf40b20e54e",
        "554ee2daa98c45022142e8fcb07bb97d4bc2887ab637e55ff9c2fbb56c5c601f",
    ),
    (
        "pipelines/static_adapt/prune_schur_payloads.py",
        "prune_schur_payloads.py.txt",
        "ab066af4135fb221f8f73a1558d2f4d2f9e1d84653240924243ec655d3c77191",
        "948d916427999ff3c7fa482d178819150e9c99631e9ea9889af05e0d20b199c5",
    ),
    (
        "pipelines/static_adapt/checkpoint_telemetry.py",
        "checkpoint_telemetry.py.txt",
        "8c2f763890dfc67d8598d8bf3ff66272290983bca95a0d920e516fbac796d7e2",
        "0e218cca94fd608cb1f6c7aefce5bc3ac295b9e2b89209152c341828484ca683",
    ),
    (
        "pipelines/static_adapt/output_artifacts.py",
        "output_artifacts.py.txt",
        "be54c839924a15e231aea3b6a9a523947dfdd0d26d2b69374fdf8067f561c188",
        "76ef2b4672acf7a0172b3ccebcba4c47b571c2e7f2f7d0664ad41a983a17f34b",
    ),
    (
        "pipelines/static_adapt/route_identity.py",
        "route_identity.py.txt",
        "ea8fb04e9ff43c0e96537220bbbc282ce1f9f47a3815c6b38de738ccb0882a53",
        "e629fb699f976299f08d883448dcadd6b30d35de6c247fae08d47a60d733689e",
    ),
    (
        "pipelines/exact_bench/hh_cost_energy_optuna.py",
        "hh_cost_energy_optuna.py.txt",
        "7395936151553438494f45dde4e41a23bc7e97126eacd17a3b82c3d6530ef38a",
        "6db406605455142235912196ba8cfcb0c6c00dc36337cbf1d5f71356f5a99277",
    ),
):
    AMPLITUDE_FRAGMENT_EXPECTATIONS[
        _source_path + "#historical_amplitude_pruning_fragments"
    ] = {
        "source_path": _source_path,
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/code/"
            "historical_amplitude_pruning/"
            + _snapshot_name
        ),
        "sha256": _snapshot_sha256,
        "source_file_sha256": _source_sha256,
    }

NOVELTY_POLICY_PRE_RETIREMENT_SOURCE_SHA256 = (
    "d433e8976775a3bd8497819d34298028775bba2158735f5582c569df12551276"
)
NOVELTY_POLICY_TEST_FRAGMENT_EXPECTATIONS = {
    (
        "test/test_static_adapt_historical_singleton_overlays.py"
        "#legacy_novelty_fallback_telemetry_tests"
    ): {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/tests/"
            "ordinary_novelty_scoring/legacy_fallback_telemetry_tests.py.txt"
        ),
        "sha256": (
            "e8f73ee15f9e241242d276f6786890fbc48b0a1246fd6dc35432041e4afd34c3"
        ),
    },
    (
        "test/test_static_adapt_historical_singleton_overlays.py"
        "#legacy_novelty_route_identity_tests"
    ): {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/tests/"
            "ordinary_novelty_scoring/legacy_route_identity_tests.py.txt"
        ),
        "sha256": (
            "1a44a4d13bf953e6ff9f09ebaf5aa8ac96e464e4154591ff84d2b21d913c9b8b"
        ),
    },
    (
        "test/test_static_adapt_historical_singleton_overlays.py"
        "#ordinary_novelty_score_assumptions"
    ): {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/tests/"
            "ordinary_novelty_scoring/ordinary_novelty_score_assumptions.py.txt"
        ),
        "sha256": (
            "c8dce3faa83bc749b8040f3cf224feac2f0747063c70d25f720336cf8728bdfd"
        ),
    },
    (
        "test/test_static_adapt_historical_singleton_overlays.py"
        "#legacy_novelty_controller_ablation_contract_tests"
    ): {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/tests/"
            "ordinary_novelty_scoring/"
            "legacy_controller_ablation_contract_tests.py.txt"
        ),
        "sha256": (
            "211f04dc3cc36344b6b83c987b2b5928a2a1448f2778b49233c7b8e64b44a9e4"
        ),
    },
}

NOVELTY_RUNTIME_FRAGMENT_EXPECTATIONS = {
    (
        "pipelines/static_adapt/adapt_pipeline.py"
        "#retired_ordinary_novelty_signature_fragments"
    ): {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/code/"
            "ordinary_novelty_scoring/"
            "adapt_pipeline_retired_ordinary_novelty_signature_fragments.py.txt"
        ),
        "sha256": (
            "a7f49c3f514580232e022f72af8830da2d14b93a7f984b0ab162cf0486b3c580"
        ),
    },
    (
        "test/test_adapt_vqe_integration.py"
        "#retired_ordinary_novelty_gamma_kwargs"
    ): {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/tests/"
            "ordinary_novelty_scoring/"
            "test_adapt_vqe_integration_retired_gamma_kwargs.py.txt"
        ),
        "sha256": (
            "ecc3b6ab50462a527d23619c7e621e3bb23aa86f7dfb0f707072e7ebe525b94e"
        ),
    },
}

HISTORICAL_PROFILE_TEST_FRAGMENT_EXPECTATIONS = {
    (
        "test/test_static_adapt_resume_scaffold.py"
        "#retired_sr_controller_ablation_cli_fragment"
    ): {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/tests/"
            "historical_profiles_and_cli_controls/"
            "test_resume_scaffold_retired_sr_controller_ablation_cli_fragment.py.txt"
        ),
        "sha256": (
            "7d38443311df53c57660eeba1fee9a57caf3479f7bc649301b048ea9d5ad1ba9"
        ),
    },
    (
        "test/test_static_adapt_sr_route_profile.py"
        "#candidate_v4_execution_authority_fragment"
    ): {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/tests/"
            "historical_profiles_and_cli_controls/"
            "test_sr_route_profile_candidate_v4_execution_authority_fragment.py.txt"
        ),
        "sha256": (
            "710a570b35ecb661f9b364fcefeaff88a2ecdcd7f30e3b9905c97b94d5082454"
        ),
    },
}

FM_MONOLITH_FRAGMENT_EXPECTATIONS = {
    "pipelines/static_adapt/adapt_pipeline.py#fm_route_profile_selector_fragments": {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/code/fm_snake/"
            "adapt_pipeline_fm_integration_fragments.py.txt"
        ),
        "sha256": (
            "3f7ebfe05a376a904fb0f8405754ec3b459f644b79affd30dd817501580db2e2"
        ),
    },
    "pipelines/static_adapt/adapt_pipeline.py#fm_integration_followup_fragments": {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/code/fm_snake/"
            "adapt_pipeline_fm_integration_followup_fragments.py.txt"
        ),
        "sha256": (
            "bd1bb864c54ce1de2ad95b66e8feeac1ec176edd6e7c7b4924eac5e5ca5f2c39"
        ),
    },
    "pipelines/static_adapt/adapt_pipeline.py#fm_selector_followup_fragments": {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/code/fm_snake/"
            "adapt_pipeline_fm_selector_followup_fragments.py.txt"
        ),
        "sha256": (
            "c5fbab8afb98e89e9d252e1bc6a8dee175bbc008b12a416fe6c35534585795db"
        ),
    },
    "pipelines/static_adapt/adapt_pipeline.py#fm_prune_accounting_fragments": {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/code/fm_snake/"
            "adapt_pipeline_fm_prune_accounting_fragments.py.txt"
        ),
        "sha256": (
            "493ac9da6bd5b5224ab18b211dfd2ec75f5b2ae20275ca2535fbbd7225d0e2f7"
        ),
    },
    "pipelines/static_adapt/adapt_pipeline.py#fm_pinned_gate_fragments": {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/code/fm_snake/"
            "adapt_pipeline_fm_pinned_gate_fragments.py.txt"
        ),
        "sha256": (
            "cfbde909241773afb6f59edde1fd08ed79f470738a1446378303a7676bb5c2c0"
        ),
    },
    (
        "pipelines/static_adapt/adapt_pipeline.py"
        "#remaining_fm_named_plumbing_fragments"
    ): {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/code/fm_snake/"
            "adapt_pipeline_remaining_fm_named_plumbing_fragments.py.txt"
        ),
        "sha256": (
            "60c5e5b0a2a08fd486fe451687048f52a8797c251181e43f8e9d5b21ed687800"
        ),
    },
    (
        "test/test_static_adapt_resume_checkpoint_parameterization.py"
        "#pure_fm_generator_guard_fragment"
    ): {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/tests/fm_snake/"
            "test_resume_scaffold_pure_fm_generator_guard_fragment.py.txt"
        ),
        "sha256": (
            "5e4b79f118b5933848d6b566a2c7f402dc8c162a51585d158219a49815cf8988"
        ),
    },
    "test/test_static_adapt_accepted_refit.py#fm_outer_anchor_guard_fragment": {
        "archive_path": (
            "archive/paper_i_static_adapt_legacy_20260727/tests/fm_snake/"
            "test_accepted_refit_fm_outer_anchor_guard_fragment.py.txt"
        ),
        "sha256": (
            "9ce4215e8d6450915b8f654d732e84cef16f42f593642bf7160698197f1d479c"
        ),
    },
}

RETIRED_MODULES = (
    "pipelines.static_adapt.adapt_pipeline_legacy_20260322",
    "pipelines.static_adapt.compare_adapt_current_vs_legacy_20260322",
    "pipelines.static_adapt.sr_snake._legacy_adapter",
    "pipelines.static_adapt.hh_continuation_scoring_legacy_bridge",
    "pipelines.hardcoded.adapt_pipeline",
    "pipelines.hardcoded.adapt_circuit_cost",
    "pipelines.hardcoded.hh_continuation_generators",
    "pipelines.hardcoded.hh_continuation_scoring",
    "pipelines.hardcoded.hh_continuation_symmetry",
    "pipelines.hardcoded.hh_continuation_types",
    "pipelines.hardcoded.imported_artifact_resolution",
    "pipelines.static_adapt.formal_manifold_exact_backend",
    "pipelines.static_adapt.formal_manifold_local_campaign",
    "pipelines.static_adapt.formal_manifold_pareto_campaign",
    "pipelines.static_adapt.formal_manifold_sr_source_locked_campaign",
    "pipelines.static_adapt.formal_manifold_outer_information",
    "pipelines.static_adapt.formal_manifold_sr_v3_outer_bridge",
    "pipelines.static_adapt.formal_manifold_warm_start",
    "pipelines.static_adapt.formal_manifold_route_profile",
    "pipelines.static_adapt.reclose_formal_manifold_query_accounting",
    "pipelines.exact_bench.paper_i_hh_fm_vs_append_fm_first_hit",
    "pipelines.exact_bench.paper_i_hh_append_fm_first_hit_campaign",
    "pipelines.static_adapt.optimization.hh_optuna_evidence_ledger",
    "pipelines.static_adapt.optimization.hh_snake_interpretable_ml_analysis",
    "pipelines.static_adapt.optimization.hh_snake_shallow_feature_extract",
    "pipelines.static_adapt.optimization.phase3_policy_optuna",
    "pipelines.static_adapt.optimization.phase3_robustness_gate",
    "pipelines.static_adapt.optimization.staged_adapt_optuna",
    "pipelines.exact_bench.paper_i_hh_full_policy_warm_start",
    "pipelines.exact_bench.paper_i_hh_live_optuna_overlay_refresh",
    "pipelines.exact_bench.paper_i_hh_local_optuna_status",
    "pipelines.exact_bench.paper_i_hh_local_optuna_supervisor",
    "pipelines.exact_bench.paper_i_hh_optuna_artifact_offload",
    "pipelines.exact_bench.paper_i_hh_review_source_locked_rerun",
    "pipelines.exact_bench.paper_i_hh_route_a_optuna",
    "pipelines.exact_bench.paper_i_hh_snake_fullpolicy_warm_start",
    "pipelines.exact_bench.paper_i_hh_snake_global_policy_optuna",
    "pipelines.exact_bench.paper_i_hh_u8_comparator_spsa_optuna",
    "pipelines.static_adapt.paper_i_runner",
    "pipelines.exact_bench.paper_i_hh_powell_pareto",
    "pipelines.exact_bench.hh_path_a_autopilot",
    "pipelines.exact_bench.hh_path_a_ledger",
    "pipelines.exact_bench.hh_path_a_tmux_wrapper",
)

FORBIDDEN_ACTIVE_SNIPPETS = RETIRED_MODULES + (
    "from pipelines.hardcoded import adapt_pipeline",
    "from pipelines.hardcoded import adapt_circuit_cost",
    "from pipelines.hardcoded import hh_continuation_generators",
    "from pipelines.hardcoded import hh_continuation_scoring",
    "from pipelines.hardcoded import hh_continuation_symmetry",
    "from pipelines.hardcoded import hh_continuation_types",
    "from pipelines.hardcoded import imported_artifact_resolution",
    "pipelines/hardcoded/adapt_pipeline.py",
    "pipelines/hardcoded/adapt_circuit_cost.py",
    "pipelines/hardcoded/hh_continuation_generators.py",
    "pipelines/hardcoded/hh_continuation_scoring.py",
    "pipelines/hardcoded/hh_continuation_symmetry.py",
    "pipelines/hardcoded/hh_continuation_types.py",
    "pipelines/hardcoded/imported_artifact_resolution.py",
    "pipelines/static_adapt/adapt_pipeline_legacy_20260322.py",
    "pipelines/static_adapt/compare_adapt_current_vs_legacy_20260322.py",
    "pipelines/static_adapt/hh_continuation_scoring_legacy_bridge.py",
    "run_legacy_sr_snake",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_entries() -> dict[str, dict[str, object]]:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert payload["schema"] == "paper_i_legacy_archive_manifest_v1"
    entries = payload["entries"]
    assert len({entry["original_path"] for entry in entries}) == len(entries)
    return {entry["original_path"]: entry for entry in entries}


def test_retired_sources_are_inert_and_hash_preserved() -> None:
    entries = _manifest_entries()

    for original_path, expected in EXPECTED_ENTRIES.items():
        entry = entries[original_path]
        assert {
            "original_path",
            "sha256",
            "family",
            "retirement_decision",
            "reachability_proof",
            "neutral_extractions",
            "movement_receipt",
        } <= entry.keys()
        assert entry["archive_path"] == expected["archive_path"]
        assert entry["family"] == expected["family"]
        assert entry["sha256"] == expected["sha256"]

        original = REPO_ROOT / original_path
        snapshot = REPO_ROOT / str(entry["archive_path"])
        assert not original.exists()
        assert snapshot.is_file()
        assert snapshot.name.endswith((".py.txt", ".json.txt"))
        assert _sha256(snapshot) == expected["sha256"]

        movement = entry["movement_receipt"]
        assert movement["source_sha256"] == expected["sha256"]
        assert movement["snapshot_sha256"] == expected["sha256"]
        assert movement["byte_identity"] == "verified"
        assert movement["original_removed"] is True

    assert not list(ARCHIVE_ROOT.rglob("*.py"))


def test_every_manifest_snapshot_matches_its_recorded_hash() -> None:
    for entry in _manifest_entries().values():
        snapshot = REPO_ROOT / str(entry["archive_path"])
        assert snapshot.is_file()
        assert _sha256(snapshot) == entry["sha256"]


def test_archive_layout_is_normalized_and_inert() -> None:
    for entry in _manifest_entries().values():
        snapshot = REPO_ROOT / str(entry["archive_path"])
        relative_snapshot = snapshot.relative_to(ARCHIVE_ROOT)
        assert relative_snapshot.parts[0] in {"code", "tests", "guidance"}
        assert relative_snapshot.parts[1] == entry["family"]
        assert snapshot.is_file()

    assert not list(ARCHIVE_ROOT.rglob("*.py"))


def test_retired_monolith_fragments_are_absent_or_manifest_allowlisted() -> None:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = _manifest_entries()

    for original_path, expected in FM_MONOLITH_FRAGMENT_EXPECTATIONS.items():
        entry = entries[original_path]
        snapshot = REPO_ROOT / str(entry["archive_path"])
        assert entry["family"] == "fm_snake"
        assert entry["archive_path"] == expected["archive_path"]
        assert entry["sha256"] == expected["sha256"]
        assert snapshot.is_file()
        assert snapshot.name.endswith(".py.txt")
        assert _sha256(snapshot) == expected["sha256"]
        assert entry["retirement_decision"]["implementation_spec_section"] == (
            "5.1"
        )
        assert entry["reachability_proof"]["source_sweep"] == (
            "source_index_proof.post_archive_active_source_sweep.fm_snake"
        )
        movement = entry["movement_receipt"]
        assert movement["source_sha256"] == expected["sha256"]
        assert movement["snapshot_sha256"] == expected["sha256"]
        assert movement["byte_identity"] == "verified"
        assert movement["active_surface_removed"] is True
        assert movement["active_source_file_preserved"] is True

    adapt_path = REPO_ROOT / "pipelines/static_adapt/adapt_pipeline.py"
    adapt_source = adapt_path.read_text(encoding="utf-8")
    adapt_tree = ast.parse(adapt_source)
    allowlist = payload["source_index_proof"][
        "retained_compatibility_allowlist"
    ]["fm_snake"]

    fm_python_names = sorted(
        {
            node.id
            for node in ast.walk(adapt_tree)
            if isinstance(node, ast.Name)
            and (
                "formal" in node.id.lower()
                or node.id.lower().startswith("fm_")
            )
        }
    )
    fm_string_tokens = sorted(
        {
            node.value
            for node in ast.walk(adapt_tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and (
                "formal" in node.value.lower()
                or "fm_" in node.value.lower()
            )
        }
    )
    assert fm_python_names == allowlist["python_name_identifiers"]
    assert fm_string_tokens == allowlist["serialized_string_tokens"]
    assert allowlist["test_id"] == (
        "test/test_ra_adapt_retired_reachability.py::"
        "test_retired_monolith_fragments_are_absent_or_manifest_allowlisted"
    )
    assert "formal_manifold_finite_angle_fallback_bridge" in fm_string_tokens

    retired_executable_snippets = (
        "formal_manifold_route_enabled",
        "formal_manifold_selector_enabled",
        "formal_singleton_refeed_selector_enabled",
        "singleton_response_selector_enabled",
        "_build_formal_manifold_executor",
        "FormalManifoldSession",
        "FormalManifoldBranchRuntime",
        "formal_query_ledger",
        "formal_resume_candidate",
        "formal_final_candidate",
        "pure_formal_manifold_expected_generator_fingerprints",
        "build_formal_manifold_query_closure_from_estimator_ledger",
        "_rescore_formal_singleton_phase3_population",
        "nfev_formal_manifold_prune_trials",
        "nfev_formal_frozen_prune_energy_probes",
    )
    assert all(
        snippet not in adapt_source
        for snippet in retired_executable_snippets
    )

    constant_false_gates = [
        node
        for node in ast.walk(adapt_tree)
        if isinstance(node, (ast.If, ast.IfExp))
        and isinstance(node.test, ast.Constant)
        and node.test.value is False
    ]
    assert constant_false_gates == []
    assert " and False" not in adapt_source
    assert " or False" not in adapt_source

    main_definitions = [
        node
        for node in adapt_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
    ]
    assert len(main_definitions) == 1
    main_definition = main_definitions[0]
    assert len(main_definition.body) == 1
    assert isinstance(main_definition.body[0], ast.Raise)
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "parse_args"
        for node in ast.walk(main_definition)
    )

    resume_test_source = (
        REPO_ROOT
        / "test/test_static_adapt_resume_checkpoint_parameterization.py"
    ).read_text(encoding="utf-8")
    assert (
        "pure_formal_manifold_expected_generator_fingerprints"
        not in resume_test_source
    )

    for original_path, expected in (
        HISTORICAL_PROFILE_TEST_FRAGMENT_EXPECTATIONS.items()
    ):
        entry = entries[original_path]
        snapshot = REPO_ROOT / str(entry["archive_path"])
        assert entry["family"] == "historical_profiles_and_cli_controls"
        assert entry["archive_path"] == expected["archive_path"]
        assert entry["sha256"] == expected["sha256"]
        assert snapshot.is_file()
        assert _sha256(snapshot) == expected["sha256"]
        movement = entry["movement_receipt"]
        assert movement["source_sha256"] == expected["sha256"]
        assert movement["snapshot_sha256"] == expected["sha256"]
        assert movement["byte_identity"] == "verified"
        assert movement["active_surface_removed"] is True
        assert movement["active_source_file_preserved"] is True

    resume_scaffold_source = (
        REPO_ROOT / "test/test_static_adapt_resume_scaffold.py"
    ).read_text(encoding="utf-8")
    route_profile_test_source = (
        REPO_ROOT / "test/test_static_adapt_sr_route_profile.py"
    ).read_text(encoding="utf-8")
    assert '"--sr-controller-ablation-contract"' not in resume_scaffold_source
    assert "test_candidate_v4_complete_runtime_profile_is_registered" not in (
        route_profile_test_source
    )
    assert "test_candidate_v4_identity_is_readable_but_not_execution_authority" in (
        route_profile_test_source
    )


def test_phase_live_hysteresis_has_no_active_runtime_reachability() -> None:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = _manifest_entries()
    phase_entries = {
        original_path: entries[original_path]
        for original_path in PHASE_LIVE_EXPECTED_ENTRIES
    }

    assert {
        entry["family"] for entry in phase_entries.values()
    } == {"phase_live_hysteresis"}
    for original_path, expected in PHASE_LIVE_EXPECTED_ENTRIES.items():
        entry = phase_entries[original_path]
        snapshot = REPO_ROOT / str(entry["archive_path"])
        assert entry["archive_path"] == expected["archive_path"]
        assert entry["sha256"] == expected["sha256"]
        assert snapshot.is_file()
        assert snapshot.name.endswith(".py.txt")
        assert _sha256(snapshot) == expected["sha256"]
        assert entry["retirement_decision"]["implementation_spec_section"] == (
            "5.4"
        )
        assert entry["reachability_proof"]["source_sweep"] == (
            "source_index_proof.post_archive_active_source_sweep."
            "phase_live_hysteresis"
        )
        movement = entry["movement_receipt"]
        assert movement["snapshot_sha256"] == expected["sha256"]
        source_file_sha256 = expected.get("source_file_sha256")
        if source_file_sha256 is None:
            assert movement["source_sha256"] == expected["sha256"]
            assert movement["byte_identity"] == "verified"
            assert movement["original_removed"] is True
        else:
            assert movement["source_file_sha256_before_retirement"] == (
                source_file_sha256
            )
            assert movement["fragment_identity"] == (
                "verified_selected_fragments"
            )
            assert movement["active_surface_removed"] is True
            assert movement["active_source_file_preserved"] is True

    proofs = payload["source_index_proof"]
    assert "phase_live_hysteresis" in (
        proofs["pre_archive_family_internal_source_sweep"]
    )
    assert "phase_live_hysteresis" in (
        proofs["post_archive_active_source_sweep"]
    )
    assert not (
        REPO_ROOT / "test/test_static_adapt_no_batch_terminal_phase.py"
    ).exists()

    stage_source = (
        REPO_ROOT / "pipelines/scaffold/hh_continuation_stage_control.py"
    ).read_text(encoding="utf-8")
    controller_source = (
        REPO_ROOT / "pipelines/static_adapt/controller_phase_state.py"
    ).read_text(encoding="utf-8")
    cli_source = (
        REPO_ROOT / "pipelines/static_adapt/cli_config.py"
    ).read_text(encoding="utf-8")
    adapt_source = (
        REPO_ROOT / "pipelines/static_adapt/adapt_pipeline.py"
    ).read_text(encoding="utf-8")

    retired_stage_controls = (
        "phase_live_hysteresis_enabled",
        "phase2_null_nrem_high_threshold",
        "phase2_live_nrem_low_threshold",
        "phase3_null_nrem_high_threshold",
        "phase3_live_nrem_low_threshold",
        "phase2_hysteresis_steps",
        "phase3_hysteresis_steps",
        "def _update_phase_live",
        "def _terminal_phase",
    )
    assert all(
        control not in stage_source for control in retired_stage_controls
    )
    assert "def _controller_phase_live" not in controller_source
    assert "def _controller_terminal_phase" not in controller_source
    assert "_controller_phase_live(" not in controller_source
    assert "--phase-live-hysteresis-enabled" not in cli_source
    assert "--phase-live-hysteresis-disabled" not in cli_source
    assert "--phase2-null-nrem-high-threshold" not in cli_source
    assert "--phase3-null-nrem-high-threshold" not in cli_source
    assert "_controller_phase_live" not in adapt_source
    assert "_controller_terminal_phase" not in adapt_source

    plateau_start = adapt_source.index(
        "def _insertion_commutation_plateau_round_policy("
    )
    plateau_end = adapt_source.index("\ndef ", plateau_start + 4)
    plateau_source = adapt_source[plateau_start:plateau_end]
    assert '"hysteresis_active": False' in plateau_source


def test_historical_amplitude_pruning_has_no_active_runtime_reachability() -> None:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = _manifest_entries()

    assert "historical_amplitude_pruning" in (
        payload["source_index_proof"]["pre_archive_family_internal_source_sweep"]
    )
    assert "historical_amplitude_pruning" in (
        payload["source_index_proof"]["post_archive_active_source_sweep"]
    )

    for original_path, expected in AMPLITUDE_FRAGMENT_EXPECTATIONS.items():
        entry = entries[original_path]
        snapshot = REPO_ROOT / str(entry["archive_path"])
        assert entry["family"] == "historical_amplitude_pruning"
        assert entry["archive_path"] == expected["archive_path"]
        assert entry["sha256"] == expected["sha256"]
        assert snapshot.is_file()
        assert snapshot.name.endswith(".py.txt")
        assert _sha256(snapshot) == expected["sha256"]
        assert entry["retirement_decision"]["implementation_spec_section"] == (
            "5.6"
        )
        assert entry["reachability_proof"]["source_sweep"] == (
            "source_index_proof.post_archive_active_source_sweep."
            "historical_amplitude_pruning"
        )

        fragment = entry["fragment"]
        assert fragment["source_file"] == expected["source_path"]
        assert fragment["source_file_pre_retirement_sha256"] == (
            expected["source_file_sha256"]
        )
        completed = subprocess.run(
            [
                "git",
                "show",
                (
                    f"{fragment['source_revision']}:"
                    f"{fragment['source_file']}"
                ),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
        )
        assert hashlib.sha256(completed.stdout).hexdigest() == (
            expected["source_file_sha256"]
        )
        source_lines = completed.stdout.splitlines(keepends=True)
        selected = b"".join(
            b"".join(
                source_lines[
                    int(span["start_line"]) - 1 : int(span["end_line"])
                ]
            )
            for span in fragment["source_spans_pre_retirement"]
        )
        assert selected == snapshot.read_bytes()

        movement = entry["movement_receipt"]
        assert movement["source_file_sha256_before_retirement"] == (
            expected["source_file_sha256"]
        )
        assert movement["snapshot_sha256"] == expected["sha256"]
        assert movement["fragment_identity"] == "verified_selected_fragments"
        assert movement["active_surface_removed"] is True
        assert movement["active_source_file_preserved"] is True

    retired_snippets = (
        "legacy_small_angle_v1",
        "PRUNE_POLICY_LEGACY_SMALL_ANGLE_V1",
        "amplitude_collapse_witness",
        "amplitude_witness",
        "phase1_prune_collapse_",
        "phase1_prune_small_theta_",
        "phase1_prune_stale_age",
        "phase1_prune_stagnation_threshold",
        "small_angle_pool_indices",
        "current_abs_theta",
        "previous_abs_theta",
        "peak_abs_theta",
        "amplitude_observation_count",
        "apply_pruning(",
        "post_prune_refit(",
    )
    passive_historical_sources = {
        REPO_ROOT / "pipelines/static_adapt/historical_route_identity.py",
        REPO_ROOT / "pipelines/static_adapt/sr_snake_route_profile.py",
    }
    active_hits: list[str] = []
    for source_root in (REPO_ROOT / "pipelines", REPO_ROOT / "src"):
        for path in source_root.rglob("*.py"):
            if path in passive_historical_sources:
                continue
            text = path.read_text(encoding="utf-8")
            for snippet in retired_snippets:
                if snippet in text:
                    active_hits.append(
                        f"{path.relative_to(REPO_ROOT)}: {snippet}"
                    )
    assert active_hits == []

    pruning_source = (
        REPO_ROOT / "pipelines/scaffold/hh_continuation_pruning.py"
    ).read_text(encoding="utf-8")
    assert "PRUNE_POLICY_RECOVERABILITY_LADDER_V1" in pruning_source
    assert "def recoverability_prune_ladder(" in pruning_source
    assert '"acceptance_source": "remove_refit_energy_safety"' in pruning_source


def test_ordinary_novelty_policy_tests_are_archived_and_passive_diagnostics_remain(
) -> None:
    from pipelines.scaffold.hh_continuation_scoring import FullScoreConfig

    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = _manifest_entries()
    proofs = payload["source_index_proof"]
    assert "ordinary_novelty_scoring_tests" in (
        proofs["pre_archive_family_internal_source_sweep"]
    )
    assert "ordinary_novelty_scoring_tests" in (
        proofs["post_archive_active_source_sweep"]
    )

    for original_path, expected in (
        NOVELTY_POLICY_TEST_FRAGMENT_EXPECTATIONS.items()
    ):
        entry = entries[original_path]
        snapshot = REPO_ROOT / str(entry["archive_path"])
        assert entry["family"] == "ordinary_novelty_scoring"
        assert entry["archive_path"] == expected["archive_path"]
        assert entry["sha256"] == expected["sha256"]
        assert snapshot.is_file()
        assert snapshot.name.endswith(".py.txt")
        assert _sha256(snapshot) == expected["sha256"]
        assert entry["retirement_decision"]["implementation_spec_section"] == (
            "5.5"
        )
        assert entry["reachability_proof"]["source_sweep"] == (
            "source_index_proof.post_archive_active_source_sweep."
            "ordinary_novelty_scoring_tests"
        )
        fragment = entry["fragment"]
        assert fragment["source_file"] == (
            "test/test_static_adapt_historical_singleton_overlays.py"
        )
        assert fragment["source_file_pre_retirement_sha256"] == (
            NOVELTY_POLICY_PRE_RETIREMENT_SOURCE_SHA256
        )
        movement = entry["movement_receipt"]
        assert movement["source_file_sha256_before_retirement"] == (
            NOVELTY_POLICY_PRE_RETIREMENT_SOURCE_SHA256
        )
        assert movement["snapshot_sha256"] == expected["sha256"]
        assert movement["fragment_identity"] == "verified_selected_fragments"
        assert movement["active_surface_removed"] is True
        assert movement["active_source_file_preserved"] is True

    for original_path, expected in (
        NOVELTY_RUNTIME_FRAGMENT_EXPECTATIONS.items()
    ):
        entry = entries[original_path]
        snapshot = REPO_ROOT / str(entry["archive_path"])
        assert entry["family"] == "ordinary_novelty_scoring"
        assert entry["archive_path"] == expected["archive_path"]
        assert entry["sha256"] == expected["sha256"]
        assert snapshot.is_file()
        assert _sha256(snapshot) == expected["sha256"]
        assert entry["retirement_decision"]["implementation_spec_section"] == (
            "5.5"
        )
        movement = entry["movement_receipt"]
        assert movement["source_sha256"] == expected["sha256"]
        assert movement["snapshot_sha256"] == expected["sha256"]
        assert movement["byte_identity"] == "verified"
        assert movement["active_surface_removed"] is True
        assert movement["active_source_file_preserved"] is True

    overlay_source = (
        REPO_ROOT / "test/test_static_adapt_historical_singleton_overlays.py"
    ).read_text(encoding="utf-8")
    retired_test_snippets = (
        "GRAM_NOVELTY_POLICY_FALLBACK_ONLY_V1",
        "HISTORICAL_SINGLETON_PHASE2_WHITENED_SCORE_FORMULA",
        "phase2_gram_novelty_policy=",
        "phase3_gram_novelty_policy=",
        "novelty_eps=",
        "ordinary_multiplier_v1",
        "SR_CONTROLLER_ABLATION_CONTRACT_NOVELTY",
        "_all_energy_models_infeasible_novelty_fallback_telemetry",
        "_selected_admission_novelty_fallback_receipt",
        "novelty_fallback_ablation",
    )
    assert all(
        snippet not in overlay_source for snippet in retired_test_snippets
    )

    config_fields = set(FullScoreConfig.__dataclass_fields__)
    retired_config_fields = {
        "gamma_N",
        "gamma_N_schedule_mode",
        "gamma_N_schedule_start",
        "gamma_N_schedule_end",
        "novelty_eps",
        "novelty_ablation_mode",
        "phase2_gram_novelty_policy",
        "phase3_gram_novelty_policy",
        "phase2_novelty_multiplier_policy",
        "phase3_novelty_multiplier_policy",
        "phase2_novelty_mode",
    }
    assert config_fields.isdisjoint(retired_config_fields)
    assert {
        "deferred_gram_fallback_enabled",
        "deferred_gram_fallback_ridge",
    } <= config_fields
    assert "HISTORICAL_SINGLETON_PHASE2_WHITENED_NO_N2_SCORE_FORMULA" in (
        overlay_source
    )
    assert "phase3_measured_novelty" in overlay_source
    assert "deferred_gram_fallback_enabled=True" in overlay_source

    adapt_source = (
        REPO_ROOT / "pipelines/static_adapt/adapt_pipeline.py"
    ).read_text(encoding="utf-8")
    adapt_tree = ast.parse(adapt_source)
    # The legacy executor is deleted, so the retired-parameter guarantee it
    # used to carry is now structural: the parameters cannot exist because the
    # function that declared them does not.
    assert not any(
        isinstance(node, ast.FunctionDef)
        and node.name == "_run_hardcoded_adapt_vqe"
        for node in adapt_tree.body
    )

    novelty_symbols: set[str] = set()
    for node in ast.walk(adapt_tree):
        symbol: str | None = None
        if isinstance(node, ast.Name):
            symbol = node.id
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbol = node.name
        elif isinstance(node, ast.arg):
            symbol = node.arg
        elif isinstance(node, ast.Attribute):
            symbol = node.attr
        if symbol is not None and (
            "novelty" in symbol.lower()
            or "gamma_n" in symbol.lower()
        ):
            novelty_symbols.add(symbol)
    allowlist = proofs["retained_compatibility_allowlist"][
        "ordinary_novelty_scoring"
    ]
    assert sorted(novelty_symbols) == allowlist["python_symbol_identifiers"]
    assert allowlist["test_id"] == (
        "test/test_ra_adapt_retired_reachability.py::"
        "test_ordinary_novelty_policy_tests_are_archived_and_passive_diagnostics_remain"
    )


def test_path_a_mixed_test_fragment_is_archived_and_path_b_is_unchanged() -> None:
    entry = _manifest_entries()[PATH_A_FRAGMENT_ORIGINAL]
    assert entry["archive_path"] == PATH_A_FRAGMENT_ARCHIVE
    assert entry["family"] == "optuna_calibration"
    assert entry["sha256"] == PATH_A_FRAGMENT_SHA256

    snapshot = REPO_ROOT / PATH_A_FRAGMENT_ARCHIVE
    assert snapshot.is_file()
    assert _sha256(snapshot) == PATH_A_FRAGMENT_SHA256
    movement = entry["movement_receipt"]
    assert movement["operation"] == (
        "inert_fragment_snapshot_then_source_spans_removed_v1"
    )
    assert movement["source_sha256"] == PATH_A_FRAGMENT_SHA256
    assert movement["snapshot_sha256"] == PATH_A_FRAGMENT_SHA256
    assert movement["byte_identity"] == "verified"
    assert movement["original_removed"] is True
    assert movement["original_source_file_preserved"] is True

    mixed_test = REPO_ROOT / "test/test_path_autopilots.py"
    text = mixed_test.read_text(encoding="utf-8")
    assert "hh_path_a_autopilot" not in text
    assert "def test_path_a_" not in text
    path_b_import = (
        "from pipelines.pareto_offline import proposal_cycle_autopilot\n"
    )
    path_b_test_marker = (
        "def test_path_b_autopilot_once_records_summary("
    )
    assert path_b_import in text
    assert path_b_test_marker in text
    retained_path_b = (
        path_b_import + text[text.index(path_b_test_marker) :]
    ).encode("utf-8")
    assert hashlib.sha256(retained_path_b).hexdigest() == (
        PATH_B_RETAINED_CONTENT_SHA256
    )
    assert entry["fragment"]["retained_path_b_content_sha256"] == (
        PATH_B_RETAINED_CONTENT_SHA256
    )


def test_retired_modules_are_not_importable() -> None:
    probe_code = f"""
import importlib.util
import json

names = {RETIRED_MODULES!r}
answers = {{}}
for name in names:
    try:
        answers[name] = importlib.util.find_spec(name) is not None
    except (AttributeError, ImportError, ModuleNotFoundError):
        answers[name] = False
print(json.dumps(answers, sort_keys=True))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", probe_code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    answers = json.loads(completed.stdout)
    assert answers == {module: False for module in RETIRED_MODULES}


def test_active_pipeline_and_src_have_no_retired_imports() -> None:
    hits: list[str] = []
    for source_root in (REPO_ROOT / "pipelines", REPO_ROOT / "src"):
        for path in source_root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for snippet in FORBIDDEN_ACTIVE_SNIPPETS:
                if snippet in text:
                    hits.append(f"{path.relative_to(REPO_ROOT)}: {snippet}")
    assert hits == []


def test_exact_bench_cross_check_uses_canonical_hva_pool_owner() -> None:
    path = REPO_ROOT / "pipelines/exact_bench/cross_check_suite.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    assert "from adapt_pipeline import _build_hva_pool" not in source
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module == "adapt_pipeline"
        and any(alias.name == "_build_hva_pool" for alias in node.names)
        for node in ast.walk(tree)
    )
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module
        == "pipelines.static_adapt.builders.primitive_pools"
        and any(alias.name == "_build_hva_pool" for alias in node.names)
        for node in ast.walk(tree)
    )


def test_tetris_disjoint_batching_has_no_active_executor_or_registry_route() -> None:
    from pipelines.exact_bench import benchmark_algorithm_registry as registry
    from pipelines.exact_bench import generic_static_adapt_variants as variants
    from pipelines.exact_bench import generic_static_metric_enrichment as enrichment
    from pipelines.exact_bench import hh_static_ground_state_benchmark as hh_benchmark
    from pipelines.exact_bench import paper_i_comparator_spsa_calibration as calibration
    from pipelines.exact_bench import paper_i_main_tables_spsa_profile as profile
    from pipelines.exact_bench import table_i_static_benchmark as table_i
    from pipelines.scaffold import hh_continuation_scoring as scoring
    from pipelines.static_adapt import cli_config

    assert RETIRED_TETRIS_ALGORITHM_ID not in (
        variants.GENERIC_STATIC_ADAPT_VARIANT_ALGORITHM_IDS
    )
    assert RETIRED_TETRIS_ALGORITHM_ID not in variants._VARIANTS
    assert not hasattr(variants, "STATIC_TETRIS_QUBIT_ADAPT_VQE")

    registered_ids = {
        algorithm.algorithm_id
        for algorithm in registry.default_benchmark_algorithms(domain="static")
    }
    assert RETIRED_TETRIS_ALGORITHM_ID not in registered_ids
    assert "static_tetris_adapt_phase3" in registered_ids
    assert RETIRED_TETRIS_ALGORITHM_ID not in table_i.TABLE_I_STATIC_ALGORITHM_IDS
    assert RETIRED_TETRIS_ALGORITHM_ID not in (
        profile.PAPER_I_MAIN_TABLES_SPSA_COMPARATOR_ALGORITHM_IDS
    )
    assert RETIRED_TETRIS_ALGORITHM_ID not in (
        calibration.PAPER_I_COMPARATOR_SPSA_CALIBRATION_ALLOWED_METHOD_IDS
    )
    assert RETIRED_TETRIS_ALGORITHM_ID not in enrichment._ADAPT_VARIANT_IDS
    assert all(
        algorithm.algorithm_id != "hh_adapt_tetris_paop_lf_std_phase3"
        for algorithm in hh_benchmark.default_hh_benchmark_algorithms()
    )

    assert not hasattr(scoring, "tetris_disjoint_batch_select")
    with pytest.raises(ValueError, match="phase2_batch_selection_mode"):
        scoring.normalize_phase2_batch_selection_mode(
            RETIRED_TETRIS_SELECTION_MODE
        )
    assert RETIRED_TETRIS_SELECTION_MODE not in (
        cli_config._PHASE_BATCH_SELECTION_MODE_CHOICES
    )

    generic_benchmark_source = (
        REPO_ROOT / "pipelines/exact_bench/generic_static_benchmark.py"
    ).read_text(encoding="utf-8")
    assert RETIRED_TETRIS_ALGORITHM_ID not in generic_benchmark_source
    historical_chtc_sources = (
        "chtc/phase3_optuna/generate_generic_static_table_records.py",
        "chtc/phase3_optuna/preflight_submit.py",
        (
            "chtc/phase3_optuna/"
            "generate_paper_i_hubbard_spinboson_repeat_policy_suite_records.py"
        ),
    )
    for relative_path in historical_chtc_sources:
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert RETIRED_TETRIS_ALGORITHM_ID in source


def test_retired_phase0_overlap_context_assembly_is_unreachable() -> None:
    source = (
        REPO_ROOT / "pipelines/static_adapt/adapt_pipeline.py"
    ).read_text(encoding="utf-8")
    assert "def _phase0_context_labels_for_candidate(" not in source
    assert "candidate_support.isdisjoint(other_support)" not in source
