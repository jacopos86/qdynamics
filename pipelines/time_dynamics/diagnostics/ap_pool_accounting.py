#!/usr/bin/env python3
"""Diagnostic accounting for AP-McLachlan append support atoms.

This module is diagnostic-only.  It normalizes AP runtime support atoms and
Paper-I sidecar pool labels into comparable Pauli-child records so run reports
do not confuse parent-pool counts, child-atom counts, and sidecar label counts.
It does not alter append, prune, scoring, or loader behavior.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.scaffold.runtime_loader import load_scaffold_runtime_input_from_payload
from pipelines.time_dynamics.ap_mclachlan.state import (
    AP_PARAMETERIZATION_PER_PAULI_TERM,
    APMcLachlanState,
    normalize_parameterization_mode,
    runtime_coordinate_records,
    state_from_scaffold_runtime_input,
)
from pipelines.time_dynamics.ap_mclachlan.support_atoms import (
    APPEND_OCCURRENCE_POLICY_LAYER_REUSE,
    candidate_append_atoms,
    no_pauli_split_parent_labels,
)
from src.quantum.ansatz_parameterization import iter_runtime_rotation_terms


SCHEMA_V1 = "ap_mclachlan_pool_support_atom_accounting_v1"


def build_pool_accounting_audit(
    state: APMcLachlanState,
    *,
    paper_i_pool_payload: Mapping[str, Any] | None = None,
    artifact_json: str | Path | None = None,
    allow_incomplete_candidate_pool: bool = False,
) -> dict[str, Any]:
    """Return labeled pool/support-atom counts and normalized digests."""

    mode = normalize_parameterization_mode(state.parameterization_mode)
    parent_pool_terms = tuple(state.candidate_pool_terms or ())
    all_child_records = _candidate_parent_child_records(state)
    active_child_records = _active_child_records(state)
    available_atoms = candidate_append_atoms(
        state,
        allow_incomplete_candidate_pool=bool(allow_incomplete_candidate_pool),
    )
    reusable_atoms = candidate_append_atoms(
        state,
        allow_incomplete_candidate_pool=bool(allow_incomplete_candidate_pool),
        occurrence_policy=APPEND_OCCURRENCE_POLICY_LAYER_REUSE,
    )
    available_child_records = tuple(
        _record_from_support_atom(atom, source_kind="ap_available_append_atom")
        for atom in available_atoms
    )
    reusable_child_records = tuple(
        _record_from_support_atom(atom, source_kind="ap_reusable_append_frontier_atom")
        for atom in reusable_atoms
    )
    sidecar = _paper_i_sidecar_accounting(paper_i_pool_payload)

    counts = {
        "selected_seed_terms": int(len(state.terms)),
        "runtime_parameter_count": int(state.runtime_parameter_count),
        "candidate_parent_pool_terms_after_loader": int(len(parent_pool_terms)),
        "no_pauli_split_parent_terms": int(len(no_pauli_split_parent_labels(state))),
        "all_pauli_child_atoms": int(len(all_child_records)),
        "active_pauli_child_atoms": int(len(active_child_records)),
        "available_append_atoms": int(len(available_child_records)),
        "reusable_append_frontier_atoms": int(len(reusable_child_records)),
        "paper_i_sidecar_pool_labels_raw": sidecar["counts"][
            "paper_i_sidecar_pool_labels_raw"
        ],
        "paper_i_sidecar_single_child_labels": sidecar["counts"][
            "paper_i_sidecar_single_child_labels"
        ],
        "paper_i_sidecar_multi_child_labels": sidecar["counts"][
            "paper_i_sidecar_multi_child_labels"
        ],
    }
    digests = {
        "ap_all_pauli_child_atoms_label_pauli_coeff_sha256": _digest_records(
            all_child_records,
            fields=("atom_label", "parent_label", "pauli_exyz", "coeff_real"),
        ),
        "ap_all_pauli_child_atoms_pauli_multiset_sha256": _digest_pauli_multiset(
            all_child_records
        ),
        "ap_all_pauli_child_atoms_unique_pauli_set_sha256": _digest_unique_pauli_set(
            all_child_records
        ),
        "ap_available_append_atoms_pauli_multiset_sha256": _digest_pauli_multiset(
            available_child_records
        ),
        "ap_available_append_atoms_unique_pauli_set_sha256": _digest_unique_pauli_set(
            available_child_records
        ),
        "ap_reusable_append_frontier_pauli_multiset_sha256": _digest_pauli_multiset(
            reusable_child_records
        ),
        "ap_reusable_append_frontier_unique_pauli_set_sha256": _digest_unique_pauli_set(
            reusable_child_records
        ),
        "paper_i_sidecar_single_child_pauli_multiset_sha256": _digest_pauli_multiset(
            sidecar["single_child_records"]
        ),
        "paper_i_sidecar_single_child_unique_pauli_set_sha256": _digest_unique_pauli_set(
            sidecar["single_child_records"]
        ),
    }
    comparisons = {
        "ap_all_children_vs_paper_i_single_child_pauli_multiset": _compare_pauli_multisets(
            all_child_records,
            sidecar["single_child_records"],
        ),
        "ap_available_children_vs_paper_i_single_child_pauli_multiset": _compare_pauli_multisets(
            available_child_records,
            sidecar["single_child_records"],
        ),
        "ap_reusable_frontier_vs_paper_i_single_child_pauli_multiset": _compare_pauli_multisets(
            reusable_child_records,
            sidecar["single_child_records"],
        ),
    }
    return {
        "schema": SCHEMA_V1,
        "diagnostic_only": True,
        "decision_data_flow": "post_load_accounting_only_not_controller_input",
        "artifact_json": None if artifact_json is None else str(artifact_json),
        "parameterization_mode": str(mode),
        "accounting_note": (
            "Raw Paper-I sidecar labels are not the same object type as AP "
            "runtime support atoms.  Equivalence is checked only after "
            "normalizing both sides to a common Pauli/poly-child representation."
        ),
        "counts": counts,
        "digests": digests,
        "comparisons": comparisons,
    }


def _candidate_parent_child_records(
    state: APMcLachlanState,
) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for candidate_index, term in enumerate(tuple(state.candidate_pool_terms or ())):
        parent_label = str(getattr(term, "label", f"candidate_{candidate_index}"))
        for local_index, spec in enumerate(_rotation_specs_for_term(term, state=state)):
            records.append(
                {
                    "source_kind": "ap_candidate_parent_child",
                    "candidate_index": int(candidate_index),
                    "local_child_index": int(local_index),
                    "atom_label": f"{parent_label}::r{int(local_index)}::{spec.pauli_exyz}",
                    "parent_label": parent_label,
                    "pauli_exyz": str(spec.pauli_exyz),
                    "coeff_real": float(spec.coeff_real),
                    "nq": int(spec.nq),
                }
            )
    return tuple(records)


def _active_child_records(state: APMcLachlanState) -> tuple[dict[str, Any], ...]:
    if normalize_parameterization_mode(state.parameterization_mode) != AP_PARAMETERIZATION_PER_PAULI_TERM:
        return ()
    records: list[dict[str, Any]] = []
    for record in runtime_coordinate_records(state):
        metadata = dict(record.metadata or {})
        if "pauli_exyz" not in metadata:
            continue
        records.append(
            {
                "source_kind": "ap_active_runtime_child",
                "runtime_index": int(record.runtime_index),
                "logical_index": int(record.logical_index),
                "atom_label": str(record.runtime_label),
                "parent_label": str(record.parent_label),
                "pauli_exyz": str(metadata["pauli_exyz"]),
                "coeff_real": float(metadata.get("coeff_real", 0.0)),
                "nq": int(metadata.get("nq", 0)),
            }
        )
    return tuple(records)


def _record_from_support_atom(atom: Any, *, source_kind: str) -> dict[str, Any]:
    metadata = dict(getattr(atom, "metadata", {}) or {})
    return {
        "source_kind": str(source_kind),
        "atom_label": str(getattr(atom, "atom_label", "")),
        "parent_label": str(getattr(atom, "parent_label", "")),
        "pauli_exyz": str(metadata.get("pauli_exyz", "")),
        "coeff_real": float(metadata.get("coeff_real", 0.0)),
        "nq": int(metadata.get("nq", 0)),
    }


def _paper_i_sidecar_accounting(
    payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    pool_labels = _extract_pool_pauli_labels(payload)
    single_child_records: list[dict[str, Any]] = []
    multi_child_labels: list[str] = []
    raw_pauli_occurrences = 0
    for label, paulis_in in sorted(pool_labels.items()):
        if not isinstance(paulis_in, Sequence) or isinstance(paulis_in, (str, bytes)):
            continue
        paulis = tuple(str(pauli) for pauli in paulis_in)
        raw_pauli_occurrences += int(len(paulis))
        if len(paulis) == 1:
            single_child_records.append(
                {
                    "source_kind": "paper_i_sidecar_single_child_label",
                    "sidecar_label": str(label),
                    "pauli_exyz": str(paulis[0]),
                }
            )
        elif len(paulis) > 1:
            multi_child_labels.append(str(label))
    return {
        "counts": {
            "paper_i_sidecar_pool_labels_raw": int(len(pool_labels)),
            "paper_i_sidecar_single_child_labels": int(len(single_child_records)),
            "paper_i_sidecar_multi_child_labels": int(len(multi_child_labels)),
            "paper_i_sidecar_pauli_occurrences_raw": int(raw_pauli_occurrences),
        },
        "single_child_records": tuple(single_child_records),
        "multi_child_label_sample": tuple(multi_child_labels[:10]),
    }


def _extract_pool_pauli_labels(
    payload: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    if isinstance(payload.get("pool_pauli_labels_exyz"), Mapping):
        return payload["pool_pauli_labels_exyz"]
    result = payload.get("result")
    if isinstance(result, Mapping) and isinstance(result.get("pool_pauli_labels_exyz"), Mapping):
        return result["pool_pauli_labels_exyz"]
    return {}


def _rotation_specs_for_term(term: Any, *, state: APMcLachlanState) -> tuple[Any, ...]:
    return iter_runtime_rotation_terms(
        getattr(term, "polynomial"),
        ignore_identity=bool(state.layout.ignore_identity),
        coefficient_tolerance=float(state.layout.coefficient_tolerance),
        sort_terms=(str(state.layout.term_order).strip().lower() == "sorted"),
    )


def _compare_pauli_multisets(
    left_records: Sequence[Mapping[str, Any]],
    right_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    left = _pauli_counter(left_records)
    right = _pauli_counter(right_records)
    only_left = left - right
    only_right = right - left
    return {
        "common_representation": "pauli_exyz_multiset_without_coefficients",
        "left_count": int(sum(left.values())),
        "right_count": int(sum(right.values())),
        "count_match": bool(sum(left.values()) == sum(right.values())),
        "digest_match": bool(
            _digest_counter(left) == _digest_counter(right)
        ),
        "unique_pauli_set_match": bool(set(left) == set(right)),
        "unique_pauli_set_digest_match": bool(
            _digest_unique_keys(left) == _digest_unique_keys(right)
        ),
        "only_left_occurrences": int(sum(only_left.values())),
        "only_right_occurrences": int(sum(only_right.values())),
        "only_left_sample": sorted(only_left)[:10],
        "only_right_sample": sorted(only_right)[:10],
    }


def _pauli_counter(records: Sequence[Mapping[str, Any]]) -> Counter[str]:
    return Counter(
        str(record.get("pauli_exyz", ""))
        for record in records
        if str(record.get("pauli_exyz", ""))
    )


def _digest_pauli_multiset(records: Sequence[Mapping[str, Any]]) -> str:
    return _digest_counter(_pauli_counter(records))


def _digest_unique_pauli_set(records: Sequence[Mapping[str, Any]]) -> str:
    return _digest_unique_keys(_pauli_counter(records))


def _digest_counter(counter: Counter[str]) -> str:
    values: list[str] = []
    for key in sorted(counter):
        values.extend([str(key)] * int(counter[key]))
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _digest_unique_keys(counter: Counter[str]) -> str:
    return hashlib.sha256(
        json.dumps(sorted(str(key) for key in counter), separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _digest_records(
    records: Sequence[Mapping[str, Any]],
    *,
    fields: Sequence[str],
) -> str:
    payload = [
        {str(field): record.get(str(field)) for field in fields}
        for record in records
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path!s}.")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnostic AP-McLachlan pool/support-atom accounting audit."
    )
    parser.add_argument("--artifact-json", required=True)
    parser.add_argument("--paper-i-pool-json", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--loader-mode", default="replay_family")
    parser.add_argument("--generator-family", default="match_adapt")
    parser.add_argument("--fallback-family", default="full_meta")
    parser.add_argument(
        "--parameterization-mode",
        default=AP_PARAMETERIZATION_PER_PAULI_TERM,
    )
    parser.add_argument(
        "--diagnostic-replay-family-pool",
        action="store_true",
        help=(
            "Diagnostic-only override that requests the replay family pool as "
            "the AP append candidate parent pool."
        ),
    )
    parser.add_argument(
        "--allow-incomplete-candidate-pool",
        action="store_true",
        help="Allow diagnostic accounting when the loader exposes an incomplete pool.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact_json = Path(args.artifact_json).expanduser()
    payload = _read_json(artifact_json)
    if bool(args.diagnostic_replay_family_pool):
        payload = dict(payload)
        payload["replay_candidate_pool_mode"] = "diagnostic_replay_family_pool"
    runtime_input = load_scaffold_runtime_input_from_payload(
        payload,
        artifact_json=artifact_json,
        loader_mode=str(args.loader_mode),
        generator_family=str(args.generator_family),
        fallback_family=str(args.fallback_family),
    )
    state = state_from_scaffold_runtime_input(
        runtime_input,
        parameterization_mode=str(args.parameterization_mode),
    )
    sidecar_payload = (
        None
        if args.paper_i_pool_json is None
        else _read_json(Path(args.paper_i_pool_json).expanduser())
    )
    audit = build_pool_accounting_audit(
        state,
        paper_i_pool_payload=sidecar_payload,
        artifact_json=artifact_json,
        allow_incomplete_candidate_pool=bool(args.allow_incomplete_candidate_pool),
    )
    text = json.dumps(audit, indent=2, sort_keys=True)
    if args.output_json:
        output_json = Path(args.output_json).expanduser()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
