from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path


BUNDLE = Path(__file__).resolve().parent


def test_archive_resolves_l4_same_cutoff_cases_and_projected_pool(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    with tarfile.open(BUNDLE / "source_locked.tar.gz", "r:gz") as archive:
        archive.extractall(source, filter="data")

    probe = r'''
import json
from pipelines.exact_bench import generic_static_adapt_variants as variants
from pipelines.exact_bench.static_reference_metrics import exact_energy_for_spec
from pipelines.exact_bench.table_i_canonical_cases import table_i_canonical_spec_by_case_id

case_ids = [
    "hh_L4_nph1_scaling_weak_weak",
    "hh_L4_nph1_scaling_intermediate_weak",
    "hh_L4_nph1_scaling_strong_weak",
    "hh_L4_nph1_scaling_weak_strong",
    "hh_L4_nph1_scaling_intermediate_strong",
    "hh_L4_nph1_scaling_strong_strong",
]
references = {}
for case_id in case_ids:
    spec = table_i_canonical_spec_by_case_id("hh", case_id)
    energy, reference_hash, _key = exact_energy_for_spec(spec, n_ph_max=1)
    references[case_id] = {"energy": energy, "reference_hash": reference_hash}

spec = table_i_canonical_spec_by_case_id("hh", case_ids[0])
context = variants._resolve_context_from_spec(spec)
parents = variants.build_full_meta_candidate_pool(context, max_terms=None)
children, meta = variants._expand_pool_with_shared_pauli_children(
    pool=parents,
    context=context,
    config=variants._get_config("static_full_meta_append_adapt_vqe"),
    mode="projected_singleton_children_only_v1",
    symmetry_policy="hard_guard",
    max_subset_size=1,
    max_terms=9000,
)
qubit_cap = variants._resource_cap_from_env(
    variants._RESOURCE_QUBIT_CAP_ENV,
    variants._QUBIT_CAP,
)
fidelity_cap = variants._exact_fidelity_max_qubits_from_env()
guard = variants._resource_guard_for_context(
    context,
    children,
    qubit_cap=qubit_cap,
    pool_cap=9000,
    pool_name="projected_singleton_children_only_v1",
)
print(json.dumps({
    "total_qubits": context.layout.total_qubits,
    "resource_qubit_cap": qubit_cap,
    "exact_fidelity_max_qubits": fidelity_cap,
    "resource_guard": guard,
    "parent_count": len(parents),
    "raw_term_count": meta["projected_singleton_source_term_count"],
    "child_count": len(children),
    "null_count": meta["projected_singleton_null_count"],
    "references": references,
}))
'''
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(source),
            "PYTHONDONTWRITEBYTECODE": "1",
            "STATIC_ADAPT_HH_POOL_CACHE": "OFF",
            "TABLE_I_STATIC_SUITE_PROFILE": "paper_i_scaling_matrix_20260710_v1",
            "GENERIC_STATIC_TABLE_RESOURCE_QUBIT_CAP": "12",
            "GENERIC_STATIC_TABLE_EXACT_FIDELITY_MAX_QUBITS": "12",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    payload = json.loads(completed.stdout)
    assert payload["total_qubits"] == 12
    assert payload["resource_qubit_cap"] == 12
    assert payload["exact_fidelity_max_qubits"] == 12
    assert payload["resource_guard"] is None
    assert payload["parent_count"] == 474
    assert payload["raw_term_count"] == 3712
    assert payload["child_count"] == 647
    assert payload["null_count"] == 17

    expected = {
        "hh_L4_nph1_scaling_weak_weak": (-2.3048285705420595, "6afec225c633ee4f6eda41ae"),
        "hh_L4_nph1_scaling_intermediate_weak": (-1.4337263224699544, "51427f68812dba5c5799afa5"),
        "hh_L4_nph1_scaling_strong_weak": (0.8794946903883553, "0a5ae9b243ee5e620ed3ac56"),
        "hh_L4_nph1_scaling_weak_strong": (-2.634636761963219, "72ca58b50fdd3dc00d562246"),
        "hh_L4_nph1_scaling_intermediate_strong": (-1.655459805225902, "2b05b5daedbe05ede60a12bc"),
        "hh_L4_nph1_scaling_strong_strong": (0.8657815466399882, "9a3fa7ec3fd5f7213c30b83b"),
    }
    for case_id, (energy, reference_hash) in expected.items():
        observed = payload["references"][case_id]
        assert abs(observed["energy"] - energy) <= 1.0e-12
        assert observed["reference_hash"] == reference_hash
