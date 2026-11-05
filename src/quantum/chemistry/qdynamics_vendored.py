"""Run Doron Wu's vendored qdynamics chemistry pipeline (``third_party/qdynamics_chemistry``) from Python.

The vendored tree is byte-identical to jacopos86/qdynamics@38f40805 (GPL-3.0; see its MANIFEST.json and
README.md). His modules import ``src.chemistry...`` and ``src.chemistry_qubitization``, the same
top-level name as our package, so they run as subprocesses with ``PYTHONPATH`` set to the vendored
root and ``python -P`` (no working-directory entry on ``sys.path``). pyscf and h5py must be
importable by the interpreter used (``HOLSTEIN_QDYNAMICS_PYTHON`` overrides ``sys.executable``).

Pipeline: an input deck (his ``.inp`` format) -> ``main_pyscf.py`` writes ``<prefix>_ele.h5``,
``<prefix>_vib.h5``, ``<prefix>_eph.h5`` next to the deck -> ``python -m src.chemistry_qubitization``
turns that directory into ``qdynamics_qubit_hamiltonian_v1`` JSON, which
``qdynamics_interop.load_qdynamics_qubit_hamiltonian`` reads.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from src.quantum.chemistry.molecular_vibronic import AMU_TO_ELECTRON_MASS, MolecularGeometry, isotope_masses_me

REPO_ROOT = Path(__file__).resolve().parents[3]
VENDORED_ROOT = REPO_ROOT / "third_party" / "qdynamics_chemistry"
MANIFEST_PATH = VENDORED_ROOT / "MANIFEST.json"
DRIVER_SCRIPT = VENDORED_ROOT / "src" / "chemistry" / "pyscf" / "main_pyscf.py"
QUBITIZER_MODULE = "src.chemistry_qubitization"
UPSTREAM_COMMIT = "38f40805d6db3fa3c2cee8a003bd07cf7dd963b3"
UPSTREAM_REPOSITORY = "https://github.com/jacopos86/qdynamics.git"
UPSTREAM_BRANCH = "doron-chemistry"


# ------------------------------------------------------------------ integrity


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def verify_vendored_files(root: Path = VENDORED_ROOT) -> dict[str, str]:
    """sha256 of every vendored file against MANIFEST.json; raises RuntimeError naming each mismatch."""
    manifest = load_manifest(Path(root) / "MANIFEST.json")
    if manifest.get("commit") != UPSTREAM_COMMIT:
        raise RuntimeError(f"manifest commit {manifest.get('commit')!r} is not the pinned {UPSTREAM_COMMIT}")
    digests: dict[str, str] = {}
    problems = []
    for relative, expected in sorted(manifest["files"].items()):
        path = Path(root) / relative
        if not path.is_file():
            problems.append(f"missing {relative}")
            continue
        data = path.read_bytes()
        digests[relative] = hashlib.sha256(data).hexdigest()
        if digests[relative] != expected["sha256"] or len(data) != int(expected["bytes"]):
            problems.append(f"modified {relative}")
    if problems:
        raise RuntimeError("vendored qdynamics tree differs from MANIFEST.json: " + "; ".join(problems))
    return digests


# --------------------------------------------------------------- input decks


def default_isotope_masses_amu(symbols: Sequence[str]) -> tuple[float, ...]:
    """Our most-abundant-isotope masses, one per atom, in amu (what his ``isotope_masses`` key takes)."""
    return tuple(float(v) for v in isotope_masses_me(symbols) / AMU_TO_ELECTRON_MASS)


@dataclass(frozen=True)
class InputDeck:
    path: Path
    output_prefix: Path
    isotope_masses_amu: tuple[float, ...] | None

    @property
    def ele_h5(self) -> Path:
        return self.output_prefix.with_name(self.output_prefix.name + "_ele.h5")

    @property
    def vib_h5(self) -> Path:
        return self.output_prefix.with_name(self.output_prefix.name + "_vib.h5")

    @property
    def eph_h5(self) -> Path:
        return self.output_prefix.with_name(self.output_prefix.name + "_eph.h5")


def write_input_deck(
    geometry: MolecularGeometry,
    directory: Path,
    *,
    name: str = "job",
    basis: str = "sto-3g",
    isotope_masses_amu: Sequence[float] | str | None = "ours",
    eph_method: str = "analytic",
    eph_cutoff_frequency_cm1: float = 80.0,
    write_vibration: bool = True,
    e_converg: float = 1.0e-12,
    d_converg: float = 1.0e-8,
    max_iter: int = 100,
) -> InputDeck:
    """His ``.inp`` for ``geometry`` (coordinates written in Bohr), electronic + e-ph output on.

    ``isotope_masses_amu``: ``"ours"`` writes our most-abundant-isotope masses (his default, no key,
    makes PySCF use integer mass numbers; see the vendored README), ``None`` writes no key, a
    sequence is written as given.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if isinstance(isotope_masses_amu, str):
        if isotope_masses_amu != "ours":
            raise ValueError("isotope_masses_amu must be 'ours', None, or one mass per atom")
        masses: tuple[float, ...] | None = default_isotope_masses_amu(geometry.symbols)
    elif isotope_masses_amu is None:
        masses = None
    else:
        masses = tuple(float(v) for v in isotope_masses_amu)
        if len(masses) != len(geometry.symbols):
            raise ValueError("isotope_masses_amu needs one value per atom")
    if eph_method not in ("analytic", "finite_difference"):
        raise ValueError("eph_method must be 'analytic' or 'finite_difference'")
    lines = [
        f"# written by src/quantum/chemistry/qdynamics_vendored.py for Doron Wu's main_pyscf.py ({UPSTREAM_COMMIT[:8]})",
        "method = RHF", f"basis = {basis}", "unit = Bohr", f"charge = {int(geometry.charge)}",
        f"spin = {int(geometry.multiplicity) - 1}", f"output = {name}", "write_h5 = true", "write_fcidump = false",
        f"write_vibration = {'true' if write_vibration else 'false'}", "write_eph = true", f"eph_method = {eph_method}",
        f"eph_cutoff_frequency = {float(eph_cutoff_frequency_cm1)!r}", f"e_converg = {float(e_converg)!r}",
        f"d_converg = {float(d_converg)!r}", f"max_iter = {int(max_iter)}",
    ]
    if masses is not None:
        lines.append("isotope_masses = " + ", ".join(repr(m) for m in masses))
    lines.append("geometry")
    for symbol, xyz in zip(geometry.symbols, np.asarray(geometry.coordinates_bohr, dtype=float)):
        lines.append(f"{symbol} {xyz[0]!r} {xyz[1]!r} {xyz[2]!r}")
    lines.append("end")
    path = directory / f"{name}.inp"
    path.write_text("\n".join(lines) + "\n")
    return InputDeck(path=path, output_prefix=directory / name, isotope_masses_amu=masses)


# ----------------------------------------------------------------- subprocess


def interpreter(python: str | os.PathLike[str] | None = None) -> str:
    return str(python or os.environ.get("HOLSTEIN_QDYNAMICS_PYTHON") or sys.executable)


def _run(args: Sequence[str], *, cwd: Path, timeout: float) -> subprocess.CompletedProcess[str]:
    if not DRIVER_SCRIPT.is_file():
        raise FileNotFoundError(f"vendored qdynamics tree missing: {DRIVER_SCRIPT}")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(VENDORED_ROOT)
    env.pop("PYTHONSAFEPATH", None)
    completed = subprocess.run(list(args), cwd=str(cwd), env=env, capture_output=True, text=True, timeout=timeout, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{' '.join(args)!s} failed ({completed.returncode}):\n{completed.stdout[-2000:]}\n{completed.stderr[-4000:]}")
    return completed


@dataclass(frozen=True)
class DriverRun:
    deck: InputDeck
    stdout: str
    stderr: str

    @property
    def outputs(self) -> tuple[Path, ...]:
        return tuple(p for p in (self.deck.ele_h5, self.deck.vib_h5, self.deck.eph_h5) if p.is_file())


def run_pyscf_driver(deck: InputDeck, *, python: str | None = None, timeout: float = 3600.0) -> DriverRun:
    """``python -P main_pyscf.py <deck>``: RHF, Hessian, PySCF eph vertex; HDF5 files land next to the deck."""
    completed = _run([interpreter(python), "-P", str(DRIVER_SCRIPT), deck.path.name], cwd=deck.path.parent, timeout=timeout)
    if not deck.ele_h5.is_file() or not deck.eph_h5.is_file():
        raise RuntimeError(f"his driver finished without writing {deck.ele_h5.name} and {deck.eph_h5.name}:\n{completed.stderr[-2000:]}")
    return DriverRun(deck=deck, stdout=completed.stdout, stderr=completed.stderr)


def run_qubitizer(
    inputs: Sequence[Path] | Path,
    output_json: Path,
    *,
    cutoffs: Sequence[int] | int,
    encoding: str = "binary",
    particles: tuple[int, int] | None = None,
    tolerance: float | None = None,
    python: str | None = None,
    timeout: float = 3600.0,
) -> Path:
    """``python -P -m src.chemistry_qubitization <inputs> -o <json> --cutoffs ... --encoding ...`` (refuses to overwrite)."""
    output_json = Path(output_json)
    paths = [Path(inputs)] if isinstance(inputs, (str, os.PathLike)) else [Path(p) for p in inputs]
    cut = [int(cutoffs)] if isinstance(cutoffs, int) else [int(c) for c in cutoffs]
    args = [interpreter(python), "-P", "-m", QUBITIZER_MODULE, *map(str, paths), "-o", str(output_json), "--cutoffs", *map(str, cut), "--encoding", str(encoding)]
    if particles is not None:
        args += ["--particles", str(int(particles[0])), str(int(particles[1]))]
    if tolerance is not None:
        args += ["--tolerance", repr(float(tolerance))]
    _run(args, cwd=output_json.resolve().parent, timeout=timeout)
    if not output_json.is_file():
        raise RuntimeError(f"his qubitizer finished without writing {output_json}")
    return output_json


def his_hamiltonian_json(
    geometry: MolecularGeometry,
    directory: Path,
    *,
    cutoffs: Sequence[int] | int,
    encoding: str = "binary",
    name: str = "job",
    python: str | None = None,
    **deck_kwargs: Any,
) -> tuple[Path, DriverRun]:
    """Deck -> his driver -> his qubitizer; returns the JSON path and the driver run."""
    deck = write_input_deck(geometry, directory, name=name, **deck_kwargs)
    run = run_pyscf_driver(deck, python=python)
    tag = "_".join(str(int(c)) for c in ([cutoffs] if isinstance(cutoffs, int) else cutoffs))
    json_path = run_qubitizer([deck.ele_h5, deck.eph_h5], Path(directory) / f"{name}_{encoding}_c{tag}.json",
                              cutoffs=cutoffs, encoding=encoding, python=python)
    return json_path, run


def read_h5_metadata(path: Path) -> dict[str, Any]:
    """The ``metadata`` attributes of one of his HDF5 files, as plain Python values."""
    import h5py  # type: ignore

    with h5py.File(path, "r") as handle:
        raw = dict(handle["metadata"].attrs) if "metadata" in handle else {}
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, bytes):
            out[key] = value.decode()
        elif isinstance(value, np.generic):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def read_h5_datasets(path: Path, names: Sequence[str]) -> dict[str, np.ndarray]:
    import h5py  # type: ignore

    with h5py.File(path, "r") as handle:
        return {name: np.asarray(handle[name][()]) for name in names if name in handle}


# --------------------------------------------- his driver's outputs as our model


def isotope_masses_amu_from_metadata(metadata: Mapping[str, Any]) -> tuple[float, ...]:
    """The ``isotope_masses`` his driver records in every HDF5 file's metadata, as numbers.

    His deck's ``isotope_masses`` line is what makes his files self-consistent (see README:
    with his default masses every frequency is over-corrected by sqrt(MP_ME/AMU2AU)), so a
    file written without it is refused here rather than mass-weighted with masses it did not use.
    """
    raw = metadata.get("isotope_masses")
    if raw is None or str(raw).strip() in {"", "None", "null"}:
        raise ValueError(
            "his HDF5 files were written without isotope_masses; add "
            "'isotope_masses = 15.99491461957, 1.00782503223, 1.00782503223' (H2O) to the deck and "
            "rerun his driver, or build the model with our builder in his conventions instead."
        )
    values = json.loads(str(raw)) if not isinstance(raw, (list, tuple, np.ndarray)) else list(raw)
    masses = tuple(float(v) for v in values)
    if not masses or any(not m > 0.0 for m in masses):
        raise ValueError(f"isotope_masses must be positive; got {masses}.")
    return masses


def vibronic_model_from_qdynamics_outputs(
    ele_h5: str | Path,
    vib_h5: str | Path,
    eph_h5: str | Path,
    *,
    cutoffs: Sequence[int] | int = 1,
    encoding: str = "binary",
    reference_force_counter_term: bool = False,
    hf_energy_atol: float = 1.0e-8,
    frequency_rtol: float = 1.0e-6,
) -> Any:
    """His driver's three HDF5 files as our ``VibronicQubitModel``: his conventions, his numbers.

    Full MO space, H = E_nuc + h_pq a+a + 1/2 (pq|rs) a+a+aa + sum_I omega_I (n_I + 1/2) + sum_I C_I Q_I
    with C_I = sqrt(2 omega_I) g_I, g_I the ``eph_mat`` of ``<prefix>_eph.h5`` (MO basis, zero-point
    factor included) and omega_I its ``omega``; that is the ``mean_field_pyscf_eph`` model without
    the counter-term, which his qubitizer writes from the same files.  The modes are
    ``<prefix>_vib.h5``'s ``norm_mode`` (Cartesian, unit norm), mass-weighted with the deck's
    isotope masses so that sqrt(m_A) d_A has unit norm as our ``NormalMode`` requires, and matched to
    the e-ph file's frequencies one to one.  The spin-orbital tensors are checked to be the
    interleaved RHF layout and chemists' (pq|rs) Coulomb integrals, and the closed-shell
    determinant energy of the tensors is checked against the file's ``e_scf``.  Modes come out in
    ascending frequency (the builder's order); each coupling's ``diagnostics`` record the index
    the mode has in his files.  ``reference_force_counter_term`` applies C_I -> C_I - <HF|C_I|HF>,
    which his model does not; the default is his model as it is.
    """
    import h5py  # type: ignore

    from src.quantum.chemistry.molecular_vibronic import (
        HARTREE_TO_CM_INV,
        MODEL_SCHEMA,
        ActiveSpace,
        ActiveTensors,
        LinearCoupling,
        NormalMode,
        assemble_vibronic_model,
        parse_geometry,
        subtract_reference_force,
    )
    from src.quantum.chemistry.qdynamics_interop import OUR_EQUIVALENT_COUPLING_MODEL

    ele_path, vib_path, eph_path = Path(ele_h5), Path(vib_h5), Path(eph_h5)
    meta = read_h5_metadata(ele_path)
    meta_vib = read_h5_metadata(vib_path)
    meta_eph = read_h5_metadata(eph_path)
    for name, other in (("vibration", meta_vib), ("electron-phonon", meta_eph)):
        for key in ("mol_str", "unit", "basis", "isotope_masses", "nelec", "nmo", "e_scf"):
            if str(other.get(key)) != str(meta.get(key)):
                raise ValueError(f"his {name} file disagrees with the electronic file on {key!r}.")
    if str(meta.get("method", "RHF")).upper() != "RHF" or int(meta.get("spin", 0)) != 0:
        raise ValueError("only closed-shell RHF outputs of his driver are supported.")
    if str(meta.get("h1_convention")) != "spin_orbital_interleaved":
        raise ValueError(f"unexpected h1_convention {meta.get('h1_convention')!r}.")
    if str(meta.get("h2_convention")) != "chemist_spin_orbital_coulomb" or bool(meta.get("h2_antisymmetrized", False)):
        raise ValueError(f"unexpected h2 convention {meta.get('h2_convention')!r} (antisymmetrized={meta.get('h2_antisymmetrized')!r}).")
    masses_amu = isotope_masses_amu_from_metadata(meta)
    geometry = parse_geometry(str(meta["mol_str"]), units=str(meta.get("unit", "Bohr")).strip().lower())
    if len(masses_amu) != geometry.n_atoms:
        raise ValueError(f"{len(masses_amu)} isotope masses for {geometry.n_atoms} atoms.")
    masses_me = np.asarray(masses_amu, dtype=float) * AMU_TO_ELECTRON_MASS

    with h5py.File(ele_path, "r") as handle:
        h1 = np.asarray(handle["h1"][...], dtype=float)
        h2 = np.asarray(handle["h2"][...], dtype=float)
        e_nuc = float(np.asarray(handle["nuclear_repulsion"][()]))
    with h5py.File(eph_path, "r") as handle:
        omega = np.asarray(handle["omega"][...], dtype=float).reshape(-1)
        eph = np.asarray(handle["eph_mat"][...], dtype=float)
    with h5py.File(vib_path, "r") as handle:
        norm_mode = np.asarray(handle["norm_mode"][...], dtype=float)
        freq_cm1 = np.asarray(handle["freq_wavenumber"][...], dtype=float).reshape(-1)

    n = int(meta["nmo"])
    n_elec = int(meta["nelec"])
    if n_elec % 2:
        raise ValueError("odd electron count in a closed-shell file.")
    n_occ = n_elec // 2
    if h1.shape != (2 * n, 2 * n) or h2.shape != (2 * n,) * 4:
        raise ValueError(f"tensor shapes {h1.shape}, {h2.shape} do not match nmo={n}.")
    alpha, beta = slice(0, None, 2), slice(1, None, 2)
    h = h1[alpha, alpha]
    if not (np.allclose(h1[beta, beta], h, atol=1.0e-12) and np.allclose(h1[alpha, beta], 0.0, atol=1.0e-12)):
        raise ValueError("h1 is not the spin-diagonal interleaved RHF layout.")
    g = h2[alpha, alpha, alpha, alpha]
    if not (
        np.allclose(h2[beta, beta, beta, beta], g, atol=1.0e-12)
        and np.allclose(h2[alpha, alpha, beta, beta], g, atol=1.0e-12)
        and np.allclose(h2[alpha, beta, alpha, beta], 0.0, atol=1.0e-12)
    ):
        raise ValueError("h2 is not the chemists' (pq|rs) spin-orbital Coulomb layout.")
    h = 0.5 * (h + h.T)
    occ = slice(0, n_occ)
    e_hf = (
        e_nuc
        + 2.0 * float(np.trace(h[occ, occ]))
        + 2.0 * float(np.einsum("iijj->", g[occ, occ, occ, occ]))
        - float(np.einsum("ijji->", g[occ, occ, occ, occ]))
    )
    e_scf = float(meta["e_scf"])
    if abs(e_hf - e_scf) > float(hf_energy_atol):
        raise ValueError(f"closed-shell determinant energy of his tensors {e_hf!r} != his e_scf {e_scf!r}.")

    n_modes = len(omega)
    if norm_mode.shape != (n_modes, geometry.n_atoms, 3) or eph.shape != (n_modes, n, n) or freq_cm1.shape != (n_modes,):
        raise ValueError("mode arrays of his files have inconsistent shapes.")
    vib_hartree = freq_cm1 / HARTREE_TO_CM_INV
    assignment: list[int] = []
    for k in range(n_modes):
        free = [j for j in range(n_modes) if j not in assignment]
        j = min(free, key=lambda jj: abs(float(vib_hartree[jj]) - float(omega[k])))
        if abs(float(vib_hartree[j]) - float(omega[k])) > float(frequency_rtol) * abs(float(omega[k])):
            raise ValueError(
                f"e-ph mode {k} (omega {omega[k]!r} Ha) has no vibration-file frequency within rtol "
                f"{frequency_rtol}; nearest {vib_hartree[j]!r} Ha."
            )
        assignment.append(j)
    order = sorted(range(n_modes), key=lambda k: float(omega[k]))
    modes: list[NormalMode] = []
    couplings: list[LinearCoupling] = []
    for a, k in enumerate(order):
        # PySCF's norm_mode d_A is normalized as sum_A m_A |d_A|^2 = 1 with the masses in amu, so the
        # mass-weighted eigenvector sqrt(m_A) d_A is unit-normalized exactly when the deck's masses are
        # the ones his driver used; the mass unit cancels in the direction.
        vector = np.sqrt(np.asarray(masses_amu, dtype=float))[:, None] * norm_mode[assignment[k]]
        norm = float(np.linalg.norm(vector))
        if abs(norm - 1.0) > 1.0e-6:
            raise ValueError(
                f"mass-weighted mode {k} has norm {norm!r} with the deck's isotope masses; his "
                "vibration file and its masses are not consistent."
            )
        vector = vector / norm
        modes.append(NormalMode(int(a), float(omega[k]), vector, f"qdynamics_mode_{k}"))
        c = float(np.sqrt(2.0 * float(omega[k]))) * eph[k]
        if not np.allclose(c, c.T, atol=1.0e-10):
            raise ValueError(f"e-ph vertex of mode {k} is not symmetric.")
        couplings.append(
            LinearCoupling(
                int(a), OUR_EQUIVALENT_COUPLING_MODEL, 0.0, 0.5 * (c + c.T), None, 0.0,
                {
                    "qdynamics_eph_mode_index": int(k),
                    "qdynamics_vib_mode_index": int(assignment[k]),
                    "mass_weighted_norm_before_renormalization": norm,
                    "eph_method": meta_eph.get("eph_method"),
                    "eph_basis": meta_eph.get("eph_basis"),
                    "vib_frequency_cm1": float(freq_cm1[assignment[k]]),
                },
            )
        )
    coupling_rows: tuple[LinearCoupling, ...] = tuple(couplings)
    if bool(reference_force_counter_term):
        coupling_rows = subtract_reference_force(coupling_rows, n_occ)
    space = ActiveSpace((), tuple(range(n)), (), int(n_occ), int(n_occ), "full")
    files = {}
    for label, path in (("ele_h5", ele_path), ("vib_h5", vib_path), ("eph_h5", eph_path)):
        files[label] = {"path": str(path), "sha256": _sha256_file(path)}
    provenance = {
        "schema": MODEL_SCHEMA,
        "source": "qdynamics_driver_outputs",
        "files": files,
        "metadata": {k: (v.item() if isinstance(v, np.generic) else v) for k, v in meta.items()},
        "coupling_model": OUR_EQUIVALENT_COUPLING_MODEL,
        "harmonic_curvature": "rhf_hessian",
        "mode_basis": "rhf_normal_modes",
        "alignment": None,
        "reference_force_counter_term": bool(reference_force_counter_term),
        "rhf_hessian_frequencies_cm1": [float(omega[k]) * HARTREE_TO_CM_INV for k in order],
        "hf_energy_hartree": e_scf,
        "hf_energy_from_tensors_hartree": e_hf,
        "isotope_masses_amu": list(masses_amu),
        "masses_me": masses_me.tolist(),
        "qdynamics_eph_mode_order": [int(k) for k in order],
        "qdynamics_vendored_commit": _vendored_commit(),
    }
    return assemble_vibronic_model(
        geometry=geometry, basis=str(meta["basis"]), active_space=space, active_tensors=ActiveTensors(e_nuc, h, g),
        modes=modes, couplings=coupling_rows, cutoffs=cutoffs, encoding=encoding,
        coupling_model=OUR_EQUIVALENT_COUPLING_MODEL, provenance=provenance,
    )


def _vendored_commit() -> str | None:
    """The upstream commit the vendored module was copied from, as MANIFEST.json records it."""
    try:
        return str(json.loads((VENDORED_ROOT / "MANIFEST.json").read_text(encoding="utf-8")).get("commit"))
    except (OSError, ValueError):
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
