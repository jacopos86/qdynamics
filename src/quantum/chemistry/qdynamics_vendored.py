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
