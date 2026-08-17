#!/usr/bin/env python3
"""Build the Paper-I RA/Append singleton-support prose evidence receipt.

The receipt attributes realized accepted-step energy descent to the exact qubit
support of each admitted Pauli word.  Shares are normalized independently for
every method--regime pair; they are path credits, not connected correlators or
causal ablations.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import tarfile
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
RA_SOURCE = (
    REPO_ROOT
    / "output/pdf/paper_i_ra_singleton_qubit_support_pauli_diagnostic_provenance.json"
)
APPEND_ADAPTER = (
    REPO_ROOT
    / "output/pdf/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving/"
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "ra_append_singleton_r70_page6_adapter.json"
)
OUTPUT = (
    REPO_ROOT
    / "MATH/paper_facing/paper_I_static_scaffold/provenance/"
    "paper_i_ra_append_singleton_support_share_prose_20260811.json"
)

REGIME_ORDER = (
    "weak_weak",
    "intermediate_weak",
    "strong_weak_u8",
    "weak_strong",
    "intermediate_strong",
    "strong_strong_u8",
)
DISPLAY_LABELS = {
    "weak_weak": "W--W",
    "intermediate_weak": "I--W",
    "strong_weak_u8": "S--W",
    "weak_strong": "W--S",
    "intermediate_strong": "I--S",
    "strong_strong_u8": "S--S",
}


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def write_receipt(path: Path, payload: dict[str, Any]) -> None:
    unsigned = dict(payload)
    unsigned.pop("sha256", None)
    output = dict(unsigned)
    output["sha256"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    encoded = (
        json.dumps(
            output,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def support_from_word(word: str) -> tuple[int, ...]:
    normalized = str(word).strip().lower().replace("i", "e")
    invalid = sorted(set(normalized).difference({"e", "x", "y", "z"}))
    if invalid:
        raise ValueError(f"invalid Pauli word {word!r}: {invalid}")
    support = tuple(
        qubit
        for qubit, letter in enumerate(reversed(normalized))
        if letter != "e"
    )
    if not support:
        raise ValueError("identity Pauli word cannot define an accepted support")
    return support


def support_label(support: Iterable[int]) -> str:
    return "{" + ",".join(f"q{qubit}" for qubit in support) + "}"


def phonon_site(qubit: int, *, n_ph_max: int) -> int:
    if n_ph_max == 3:
        if qubit in (4, 5):
            return 0
        if qubit in (6, 7):
            return 1
    elif n_ph_max == 7:
        if qubit in (4, 5, 6):
            return 0
        if qubit in (7, 8, 9):
            return 1
    raise ValueError(f"qubit q{qubit} is not a phonon bit for n_ph_max={n_ph_max}")


def physical_support_label(support: tuple[int, ...], *, n_ph_max: int) -> str:
    if support == (2, 3):
        return "intersite down-spin fermionic rotation"
    if support == (0, 1):
        return "intersite up-spin fermionic rotation"
    if support == (0, 1, 2, 3):
        return "four-fermion support"
    fermion_qubits = tuple(qubit for qubit in support if qubit < 4)
    phonon_qubits = tuple(qubit for qubit in support if qubit >= 4)
    if fermion_qubits and not phonon_qubits:
        return f"{len(fermion_qubits)}-qubit fermionic support"
    fermion_sites = {qubit % 2 for qubit in fermion_qubits}
    phonon_sites = {
        phonon_site(qubit, n_ph_max=n_ph_max) for qubit in phonon_qubits
    }
    if phonon_qubits and not fermion_qubits:
        if len(phonon_sites) == 1:
            site = next(iter(phonon_sites))
            return f"site-{site} pure-phonon support"
        return "two-site pure-phonon support"
    if len(fermion_sites | phonon_sites) == 1:
        site = next(iter(fermion_sites | phonon_sites))
        return f"site-{site} local mixed fermion--phonon support"
    return "cross-site mixed fermion--phonon support"


def support_sector(support: tuple[int, ...]) -> str:
    if all(qubit < 4 for qubit in support):
        return "fermion_only"
    if all(qubit >= 4 for qubit in support):
        return "phonon_only"
    return "mixed_fermion_phonon"


def finalize_supports(
    metrics: dict[tuple[int, ...], dict[str, Any]],
    *,
    denominator: float,
    n_ph_max: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if denominator <= 0.0:
        raise ValueError("realized path-drop denominator must be positive")
    rows: list[dict[str, Any]] = []
    sector_shares = defaultdict(float)
    for support, raw in metrics.items():
        raw_drop = float(raw["raw_drop"])
        share = 100.0 * raw_drop / denominator
        sector = support_sector(support)
        sector_shares[sector] += share
        rows.append(
            {
                "support": list(support),
                "support_label": support_label(support),
                "physical_label": physical_support_label(
                    support, n_ph_max=n_ph_max
                ),
                "sector": sector,
                "accepted_count": int(raw["count"]),
                "raw_drop_numerator": raw_drop,
                "within_regime_share_percent": share,
                "pauli_words": dict(sorted(raw["words"].items())),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["raw_drop_numerator"]),
            tuple(row["support"]),
        )
    )
    if not math.isclose(
        sum(float(row["within_regime_share_percent"]) for row in rows),
        100.0,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError("within-regime support shares do not sum to 100 percent")
    for key in ("fermion_only", "mixed_fermion_phonon", "phonon_only"):
        sector_shares.setdefault(key, 0.0)
    return rows, dict(sorted(sector_shares.items()))


def build_ra_records(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if payload.get("schema") != "paper_i_ra_singleton_qubit_support_pauli_diagnostic_v2":
        raise ValueError("unexpected RA support-evidence schema")
    if payload.get("status") != "passed":
        raise ValueError("RA support evidence did not pass")
    scope = payload.get("scope", {})
    if int(scope.get("accepted_horizon", -1)) != 50:
        raise ValueError("RA evidence is not the completed k=50 horizon")
    if bool(scope.get("causal_claim")) or bool(scope.get("connected_correlation_claim")):
        raise ValueError("RA evidence has unsafe causal/correlation semantics")

    records: dict[str, dict[str, Any]] = {}
    for raw_regime in payload.get("regimes", []):
        regime = str(raw_regime["regime_id"])
        metrics: dict[tuple[int, ...], dict[str, Any]] = {}
        for raw_support in raw_regime["exact_supports"]:
            support = tuple(int(qubit) for qubit in raw_support["support"])
            metrics[support] = {
                "raw_drop": float(raw_support["raw_drop"]),
                "count": int(raw_support["count"]),
                "words": Counter(
                    {
                        str(word): int(count)
                        for word, count in raw_support["words"].items()
                    }
                ),
            }
        denominator = float(raw_regime["total_raw_drop"])
        support_rows, sector_shares = finalize_supports(
            metrics,
            denominator=denominator,
            n_ph_max=int(raw_regime["n_ph_max"]),
        )
        records[regime] = {
            "method": "RA-ADAPT",
            "route_id": str(scope["route_id"]),
            "accepted_horizon": int(raw_regime["accepted_rounds"]),
            "n_ph_max": int(raw_regime["n_ph_max"]),
            "total_qubits": int(raw_regime["total_qubits"]),
            "initial_energy": float(raw_regime["initial_energy"]),
            "terminal_energy": float(raw_regime["terminal_energy"]),
            "realized_path_drop_denominator": denominator,
            "sector_shares_percent": sector_shares,
            "supports": support_rows,
            "top_three": support_rows[:3],
        }
    return records


def extract_append_word(label: str, *, total_qubits: int) -> str:
    prefix = "guarded_singleton::"
    if not str(label).startswith(prefix):
        raise ValueError(f"unexpected Append-ADAPT operator label: {label}")
    word = str(label)[len(prefix) :].strip().lower().replace("i", "e")
    if len(word) != total_qubits:
        raise ValueError(
            f"Append-ADAPT Pauli width mismatch: {len(word)} != {total_qubits}"
        )
    support_from_word(word)
    return word


def build_append_records(
    payload: dict[str, Any],
    *,
    ra_records: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if payload.get("schema") != "paper_i_ra_append_singleton_r70_page6_adapter_v1":
        raise ValueError("unexpected completed Append-ADAPT adapter schema")
    if payload.get("status") != "passed_with_explicit_ra_terminal_limitation":
        raise ValueError("completed Append-ADAPT adapter did not pass")
    if list(payload.get("regime_order", [])) != list(REGIME_ORDER):
        raise ValueError("completed Append-ADAPT adapter regime order changed")
    route_id = "append_adapt_largest_absolute_commutator_gradient_v1"

    records: dict[str, dict[str, Any]] = {}
    for raw_cell in payload.get("cells", []):
        regime = str(raw_cell["regime_id"])
        if regime not in ra_records:
            raise ValueError(f"Append-ADAPT regime has no RA match: {regime}")
        raw_regime = raw_cell["append"]
        archive_record = raw_regime["source"]["archive"]
        archive_path = Path(str(archive_record["path"]))
        if not archive_path.is_absolute():
            archive_path = REPO_ROOT / archive_path
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        archive_sha = sha256_file(archive_path)
        if archive_sha != str(archive_record["sha256"]):
            raise ValueError(f"Append-ADAPT archive SHA mismatch: {archive_path}")
        with tarfile.open(archive_path, "r:gz") as archive:
            member = archive.getmember("worker_outputs/payload/summary.json")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"could not read Append summary from {archive_path}")
            summary = json.load(stream)
        if summary.get("schema") != "paper_i_append_run_summary_v1":
            raise ValueError(f"unexpected Append summary schema: {archive_path}")
        if summary.get("candidate_representation") != "single_pauli_word_v1":
            raise ValueError(f"Append representation mismatch: {archive_path}")
        if summary.get("selector_identity") != route_id:
            raise ValueError(f"Append selector mismatch: {archive_path}")
        if summary.get("optimizer") != "powell" or int(
            summary.get("optimizer_maxiter", -1)
        ) != 200:
            raise ValueError(f"Append optimizer mismatch: {archive_path}")
        if not bool(summary.get("append_position_only")):
            raise ValueError(f"Append placement contract mismatch: {archive_path}")
        prefixes = list(summary.get("accepted_history", []))[:50]
        if [int(row["controller_round"]) for row in prefixes] != list(range(1, 51)):
            raise ValueError(f"Append-ADAPT {regime} is not a consecutive k=50 path")
        first_label = str(prefixes[0]["selected_label"])
        total_qubits = len(first_label.removeprefix("guarded_singleton::"))
        n_ph_max = int(ra_records[regime]["n_ph_max"])
        if int(raw_regime["nph"]) != n_ph_max:
            raise ValueError(f"Append/RA cutoff mismatch in {regime}")
        metrics: dict[tuple[int, ...], dict[str, Any]] = {}
        initial_energy = float(prefixes[0]["energy_before"])
        if not math.isclose(
            initial_energy,
            float(ra_records[regime]["initial_energy"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"Append/RA initial-energy mismatch in {regime}")
        previous_energy = initial_energy
        for expected_round, row in enumerate(prefixes, start=1):
            before = float(row["energy_before"])
            if not math.isclose(
                previous_energy, before, rel_tol=0.0, abs_tol=1.0e-9
            ):
                raise ValueError(f"Append energy chain broke in {regime}")
            if int(row["insertion_position"]) != expected_round - 1:
                raise ValueError(f"Append position changed in {regime}")
            word = extract_append_word(
                str(row["selected_label"]), total_qubits=total_qubits
            )
            support = support_from_word(word)
            energy = float(row["energy_after"])
            drop = max(0.0, before - energy)
            metric = metrics.setdefault(
                support,
                {"raw_drop": 0.0, "count": 0, "words": Counter()},
            )
            metric["raw_drop"] += drop
            metric["count"] += 1
            metric["words"][word] += 1
            previous_energy = energy
        denominator = sum(float(row["raw_drop"]) for row in metrics.values())
        expected_drop = initial_energy - previous_energy
        if not math.isclose(denominator, expected_drop, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(f"Append-ADAPT {regime} energy chain is not monotone")
        support_rows, sector_shares = finalize_supports(
            metrics,
            denominator=denominator,
            n_ph_max=n_ph_max,
        )
        records[regime] = {
            "method": "Append-ADAPT",
            "route_id": route_id,
            "accepted_horizon": 50,
            "n_ph_max": n_ph_max,
            "total_qubits": total_qubits,
            "initial_energy": initial_energy,
            "terminal_energy": previous_energy,
            "realized_path_drop_denominator": denominator,
            "sector_shares_percent": sector_shares,
            "supports": support_rows,
            "top_three": support_rows[:3],
            "source_archive": {
                "path": repo_relative(archive_path),
                "sha256": archive_sha,
                "size_bytes": int(archive_record["size_bytes"]),
                "summary_member": "worker_outputs/payload/summary.json",
                "summary_source_payload_sha256": str(
                    summary["source_result_payload_sha256"]
                ),
            },
            "summary_contract": {
                "protocol_sha256": str(summary["protocol_sha256"]),
                "candidate_representation": str(
                    summary["candidate_representation"]
                ),
                "optimizer": str(summary["optimizer"]),
                "optimizer_maxiter": int(summary["optimizer_maxiter"]),
                "adapt_seed": int(summary["seeds"]["adapt"]),
                "transpiler_seed": int(summary["seeds"]["transpiler"]),
                "active_gradient_policy": str(summary["active_gradient_policy"]),
                "accepted_refit_scope": str(summary["accepted_refit_scope"]),
                "accepted_refit_coordinate_chart": str(
                    summary["accepted_refit_coordinate_chart"]
                ),
            },
        }
    return records


def main() -> None:
    ra_payload = json.loads(RA_SOURCE.read_text(encoding="utf-8"))
    append_payload = json.loads(APPEND_ADAPTER.read_text(encoding="utf-8"))
    ra_records = build_ra_records(ra_payload)
    append_records = build_append_records(append_payload, ra_records=ra_records)
    if set(ra_records) != set(REGIME_ORDER) or set(append_records) != set(REGIME_ORDER):
        raise ValueError("RA/Append evidence does not cover the canonical six regimes")

    regimes = []
    for regime in REGIME_ORDER:
        regimes.append(
            {
                "regime_id": regime,
                "display_label": DISPLAY_LABELS[regime],
                "RA-ADAPT": ra_records[regime],
                "Append-ADAPT": append_records[regime],
            }
        )

    payload = {
        "schema": "paper_i_ra_append_singleton_support_share_prose_v1",
        "status": "passed",
        "paper_scope": "Paper I Hubbard--Holstein Results support-composition prose",
        "metric_contract": {
            "transition_drop": "d_k=max(0,E_{k-1}-E_k)",
            "support_drop": "D_{S,r}^{(m)}=sum_{k:supp(P_k)=S} d_k",
            "within_method_regime_share": (
                "p_{S,r}^{(m)}=100*D_{S,r}^{(m)}/sum_S D_{S,r}^{(m)}"
            ),
            "normalization": "independent within each method--regime pair",
            "accepted_horizon": 50,
            "rounding_for_manuscript": "nearest 0.1 percentage point",
            "causal_claim": False,
            "connected_correlation_claim": False,
            "interpretation": (
                "path-dependent realized accepted-step descent attributed to exact "
                "Pauli-word support"
            ),
        },
        "register_contract": {
            "indexing": "zero-based; Pauli strings are q_(Q-1)...q_0",
            "fermions": {
                "q0": "site 0 spin up",
                "q1": "site 1 spin up",
                "q2": "site 0 spin down",
                "q3": "site 1 spin down",
            },
            "n_ph_max_3": {
                "q4-q5": "site 0 phonon binary register",
                "q6-q7": "site 1 phonon binary register",
            },
            "n_ph_max_7": {
                "q4-q6": "site 0 phonon binary register",
                "q7-q9": "site 1 phonon binary register",
            },
        },
        "comparison_contract": {
            "candidate_representation": "single Pauli word",
            "common_horizon": 50,
            "common_reference": "HF",
            "source_methods": ["RA-ADAPT", "Append-ADAPT"],
        },
        "sources": {
            "RA-ADAPT": {
                "path": repo_relative(RA_SOURCE),
                "sha256": sha256_file(RA_SOURCE),
                "embedded_receipt_sha256": str(ra_payload.get("sha256", "")),
                "route_id": str(ra_payload["scope"]["route_id"]),
            },
            "Append-ADAPT": {
                "path": repo_relative(APPEND_ADAPTER),
                "sha256": sha256_file(APPEND_ADAPTER),
                "embedded_receipt_sha256": str(append_payload.get("sha256", "")),
                "route_id": "append_adapt_largest_absolute_commutator_gradient_v1",
                "classification_in_source": str(append_payload["classification"]),
                "adoption_authority": (
                    "user-directed Paper-I Results adoption on 2026-08-11 for "
                    "the k=50 support-share comparison"
                ),
            },
        },
        "builder": repo_relative(Path(__file__)),
        "regimes": regimes,
    }
    write_receipt(OUTPUT, payload)
    print(f"wrote {repo_relative(OUTPUT)}")
    print(f"sha256={sha256_file(OUTPUT)}")


if __name__ == "__main__":
    main()
