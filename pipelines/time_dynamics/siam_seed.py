r"""Immutable prepared-state contract for SIAM screening trajectories.

The seed records only physics and preparation identity.  Reporting horizons,
controllers, activation cuts, and exact trajectories belong to later run
contracts and must not alter these bytes.  Loading regenerates every bound
object from the declared physical parameters and fails closed on any mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Real
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from pipelines.time_dynamics.siam_screening import SIAMScreeningConfig
from src.quantum.siam_model import (
    SIAMLayout,
    build_initial_state,
    build_siam_hamiltonian,
    sector_indices,
)
from src.quantum.siam_pool import SIAM_POOL_V1, build_siam_pool


SIAM_SEED_SCHEMA_V1 = "siam_screening_seed_v1"
SIAM_SEED_SCHEMA_V2 = "siam_screening_seed_v2"
SIAM_PREPARATION_SCHEMA_V1 = "siam_decoupled_closed_shell_seed_v1"
SIAM_STATE_IDENTITY_SCHEMA_V1 = "phase_canonical_quantized_complex_v1"
SIAM_STATE_IDENTITY_QUANTUM = 1.0e-12


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_float(value: float) -> str:
    number = float(value)
    if number == 0.0:
        number = 0.0
    return number.hex()


def _polynomial_payload(polynomial: Any) -> dict[str, Any]:
    terms = []
    for term in tuple(polynomial.return_polynomial()):
        coefficient = complex(term.p_coeff)
        terms.append(
            {
                "word": str(term.pw2strng()),
                "coefficient_real_hex": _canonical_float(coefficient.real),
                "coefficient_imag_hex": _canonical_float(coefficient.imag),
            }
        )
    terms.sort(key=lambda row: str(row["word"]))
    return {
        "n_qubits": int(polynomial.get_nq()),
        "terms": terms,
    }


def _statevector_sha256(vector: np.ndarray) -> str:
    """Legacy schema-v1 identity over raw platform-dependent bytes."""

    canonical = np.asarray(vector, dtype=np.dtype("<c16")).reshape(-1)
    digest = hashlib.sha256()
    digest.update(_canonical_json({"shape": [int(canonical.size)]}).encode("utf-8"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def portable_statevector_sha256(vector: np.ndarray) -> str:
    """Return a phase-invariant, roundoff-stable state identity.

    Schema-v1 hashed raw ``complex128`` bytes produced through LAPACK.  The
    same closed-shell state therefore acquired different identities across
    NumPy/BLAS builds at roughly machine precision.  Schema-v2 first
    normalizes the vector, fixes global phase using the first reliably nonzero
    amplitude, and quantizes real and imaginary parts before hashing explicit
    little-endian signed integers.  The quantum is eight orders tighter than
    any declared SIAM accuracy guard while comfortably above backend roundoff.
    """

    state = np.asarray(vector, dtype=np.complex128).reshape(-1).copy()
    if state.size == 0 or not np.all(np.isfinite(state)):
        raise ValueError("portable state identity requires a nonempty finite vector.")
    norm = float(np.linalg.norm(state))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("portable state identity requires positive finite norm.")
    state /= norm

    nonzero = np.flatnonzero(np.abs(state) > SIAM_STATE_IDENTITY_QUANTUM)
    if nonzero.size == 0:
        raise ValueError(
            "portable state identity has no amplitude above its quantization floor."
        )
    pivot = complex(state[int(nonzero[0])])
    state *= np.conjugate(pivot) / abs(pivot)

    quantum = float(SIAM_STATE_IDENTITY_QUANTUM)
    real = np.asarray(np.rint(state.real / quantum), dtype=np.dtype("<i8"))
    imag = np.asarray(np.rint(state.imag / quantum), dtype=np.dtype("<i8"))
    identity = {
        "schema": SIAM_STATE_IDENTITY_SCHEMA_V1,
        "shape": [int(state.size)],
        "quantum_hex": quantum.hex(),
        "encoding": "little_endian_int64_real_then_imag",
    }
    digest = hashlib.sha256(_canonical_json(identity).encode("utf-8"))
    digest.update(real.tobytes(order="C"))
    digest.update(imag.tobytes(order="C"))
    return digest.hexdigest()


def _manifest_bytes(manifest: "SIAMSeedManifest") -> bytes:
    return (
        json.dumps(
            manifest.to_json_dict(),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True)
class SIAMSeedManifest:
    """Canonical physical/preparation identity for one SIAM seed."""

    physical: Mapping[str, Any]
    layout: Mapping[str, Any]
    preparation: Mapping[str, Any]
    pool: Mapping[str, Any]
    digests: Mapping[str, str]
    schema: str = SIAM_SEED_SCHEMA_V2

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "physical": dict(self.physical),
            "layout": dict(self.layout),
            "preparation": dict(self.preparation),
            "pool": dict(self.pool),
            "digests": dict(self.digests),
        }

    def physical_config(self) -> SIAMScreeningConfig:
        """Reconstruct a horizon-neutral config for deterministic loading."""

        return SIAMScreeningConfig(
            # Do not coerce raw JSON values here.  SIAMScreeningConfig is the
            # type/finite validator; coercion would let JSON booleans masquerade
            # as 0/1 because Python considers True == 1.0.
            n_bath=self.physical["n_bath"],
            u=self.physical["u"],
            v_final=self.physical["v_post_quench"],
            hopping=self.physical["hopping"],
            t_final=1.0,
            num_times=2,
            label="siam_seed_loader",
        )


def build_siam_seed_manifest(
    config: SIAMScreeningConfig | None = None,
    *,
    schema: str = SIAM_SEED_SCHEMA_V2,
) -> SIAMSeedManifest:
    """Regenerate the complete immutable seed identity from physics."""

    schema_name = str(schema)
    if schema_name not in {SIAM_SEED_SCHEMA_V1, SIAM_SEED_SCHEMA_V2}:
        raise ValueError(f"unsupported SIAM seed schema: {schema_name!r}")
    cfg = config or SIAMScreeningConfig()
    layout = SIAMLayout(int(cfg.n_bath))
    n_up, n_down = layout.electron_counts()
    initial_state = build_initial_state(int(cfg.n_bath), hopping=float(cfg.hopping))
    h_pre = build_siam_hamiltonian(
        int(cfg.n_bath), u=float(cfg.u), v=0.0, hopping=float(cfg.hopping)
    )
    h_post = build_siam_hamiltonian(
        int(cfg.n_bath),
        u=float(cfg.u),
        v=float(cfg.v_final),
        hopping=float(cfg.hopping),
    )
    pool = tuple(build_siam_pool(int(cfg.n_bath)))
    pool_payload = [
        {
            "label": str(block.label),
            "family": str(block.family),
            "operator": _polynomial_payload(block.operator),
        }
        for block in pool
    ]
    physical = dict(cfg.physics_provenance())
    preparation: dict[str, Any] = {
        "schema": SIAM_PREPARATION_SCHEMA_V1,
        "deterministic": True,
        "rng_seed": None,
    }
    if schema_name == SIAM_SEED_SCHEMA_V2:
        preparation["state_identity"] = {
            "schema": SIAM_STATE_IDENTITY_SCHEMA_V1,
            "phase_rule": "first_amplitude_above_quantum_real_nonnegative",
            "quantum_hex": float(SIAM_STATE_IDENTITY_QUANTUM).hex(),
            "encoding": "little_endian_int64_real_then_imag",
        }
        initial_state_digests = {
            "initial_state_semantic_sha256": portable_statevector_sha256(
                initial_state
            )
        }
    else:
        initial_state_digests = {
            "initial_state_sha256": _statevector_sha256(initial_state)
        }

    return SIAMSeedManifest(
        physical=physical,
        layout={
            "mode_order": "spin_blocked_jordan_wigner",
            "n_qubits": int(layout.n_qubits),
            "electron_counts": {"up": int(n_up), "down": int(n_down)},
            "sector_dimension": int(len(sector_indices(int(cfg.n_bath)))),
        },
        preparation=preparation,
        pool={
            "schema": SIAM_POOL_V1,
            "block_count": int(len(pool)),
            "ordered_labels": [str(block.label) for block in pool],
        },
        digests={
            **initial_state_digests,
            "pre_quench_hamiltonian_sha256": _sha256_json(
                _polynomial_payload(h_pre)
            ),
            "post_quench_hamiltonian_sha256": _sha256_json(
                _polynomial_payload(h_post)
            ),
            "ordered_logical_pool_sha256": _sha256_json(pool_payload),
        },
        schema=schema_name,
    )


def write_siam_seed_manifest(
    path: str | Path,
    config: SIAMScreeningConfig | None = None,
    *,
    schema: str = SIAM_SEED_SCHEMA_V2,
) -> SIAMSeedManifest:
    """Atomically create an immutable deterministic seed manifest.

    Repeating the same write is a no-op.  Reusing the path for different seed
    bytes is rejected so a path referenced by an earlier run can never be
    silently rebound to different physics.
    """

    manifest = build_siam_seed_manifest(config, schema=schema)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = _manifest_bytes(manifest)
    if destination.exists():
        if destination.read_bytes() == content:
            return manifest
        raise FileExistsError(
            f"Refusing to replace immutable SIAM seed manifest: {destination}"
        )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # A hard-link install is atomic and never replaces an existing
            # destination, unlike os.replace.  The temporary lives beside the
            # destination, so both names are necessarily on one filesystem.
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() != content:
                raise FileExistsError(
                    "Refusing to replace immutable SIAM seed manifest: "
                    f"{destination}"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def _manifest_from_bytes(content: bytes) -> SIAMSeedManifest:
    """Verify exactly the bytes whose digest is bound to a runtime input."""

    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("SIAM seed manifest is not valid UTF-8 JSON.") from exc
    schema = payload.get("schema") if isinstance(payload, dict) else None
    if schema not in {SIAM_SEED_SCHEMA_V1, SIAM_SEED_SCHEMA_V2}:
        raise ValueError(
            "SIAM seed manifest must use a supported schema: "
            f"{SIAM_SEED_SCHEMA_V1!r} or {SIAM_SEED_SCHEMA_V2!r}."
        )
    physical = payload.get("physical")
    if not isinstance(physical, dict):
        raise ValueError("SIAM seed manifest is missing its physical contract.")
    v_pre_quench = physical.get("v_pre_quench")
    if (
        isinstance(v_pre_quench, (bool, np.bool_))
        or not isinstance(v_pre_quench, Real)
        or not np.isfinite(v_pre_quench)
        or float(v_pre_quench) != 0.0
    ):
        raise ValueError("SIAM seed requires the declared decoupled V_pre=0 state.")
    declared = SIAMSeedManifest(
        physical=physical,
        layout=dict(payload.get("layout") or {}),
        preparation=dict(payload.get("preparation") or {}),
        pool=dict(payload.get("pool") or {}),
        digests=dict(payload.get("digests") or {}),
        schema=str(schema),
    )
    expected = build_siam_seed_manifest(
        declared.physical_config(),
        schema=declared.schema,
    )
    if payload != expected.to_json_dict():
        raise ValueError(
            "SIAM seed manifest failed deterministic regeneration; physical, "
            "layout, preparation, pool, or digest content was altered."
        )
    return expected


def _load_verified_manifest(path: str | Path) -> tuple[SIAMSeedManifest, str]:
    source = Path(path)
    content = source.read_bytes()
    manifest = _manifest_from_bytes(content)
    return manifest, hashlib.sha256(content).hexdigest()


def load_siam_seed_manifest(path: str | Path) -> SIAMSeedManifest:
    """Load by regeneration; edited parameters or digests are rejected."""

    manifest, _digest = _load_verified_manifest(path)
    return manifest


def load_siam_runtime_input(path: str | Path):
    """Verify ``path`` and return the generic runtime-input contract."""

    source = Path(path).resolve()
    manifest, verified_seed_sha256 = _load_verified_manifest(source)
    from pipelines.time_dynamics.siam_runtime import build_siam_runtime_input

    return build_siam_runtime_input(
        manifest.physical_config(),
        artifact_json=str(source),
        seed_manifest=manifest,
        verified_seed_sha256=verified_seed_sha256,
    )


__all__ = [
    "SIAM_PREPARATION_SCHEMA_V1",
    "SIAM_SEED_SCHEMA_V1",
    "SIAM_SEED_SCHEMA_V2",
    "SIAM_STATE_IDENTITY_QUANTUM",
    "SIAM_STATE_IDENTITY_SCHEMA_V1",
    "SIAMSeedManifest",
    "build_siam_seed_manifest",
    "load_siam_runtime_input",
    "load_siam_seed_manifest",
    "portable_statevector_sha256",
    "write_siam_seed_manifest",
]
