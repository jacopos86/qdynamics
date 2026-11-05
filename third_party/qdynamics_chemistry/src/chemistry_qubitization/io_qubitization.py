"""Read and validate chemistry HDF5 inputs for qubitization."""
from pathlib import Path
import h5py
import numpy as np

from .data import MolecularIntegrals


def _read(path):
    with h5py.File(path, "r") as handle:
        meta = dict(handle["metadata"].attrs) if "metadata" in handle else {}
        meta = {k: v.decode() if isinstance(v, bytes) else v.item() if isinstance(v, np.generic) else v
                for k, v in meta.items()}
        arrays = {k: np.asarray(v[()]) for k, v in handle.items() if isinstance(v, h5py.Dataset)}
    return meta, arrays


def discover_files(paths):
    """Recognize inputs by datasets; reject ambiguous multi-calculation folders."""
    candidates = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            candidates.extend(p for p in sorted(path.iterdir()) if p.suffix.lower() in {".h5", ".hdf5"})
        elif path.is_file():
            candidates.append(path)
        else:
            raise FileNotFoundError(path)
    electronic, eph = [], []
    for path in dict.fromkeys(p.resolve() for p in candidates):
        with h5py.File(path, "r") as handle:
            keys = set(handle.keys())
        if {"h1", "h2", "nuclear_repulsion"} <= keys:
            electronic.append(path)
        elif {"eph_mat", "omega"} <= keys:
            eph.append(path)
        elif keys & {"h1", "h2", "eph_mat"}:
            raise ValueError(f"Incomplete matrix-element file: {path}")
    if len(electronic) != 1 or len(eph) > 1:
        raise ValueError(f"Expected exactly one electronic file and at most one e-ph file; found {len(electronic)}, {len(eph)}. Select files explicitly.")
    return electronic[0], eph[0] if eph else None


def _real(array, name):
    value = np.asarray(array)
    if not np.all(np.isfinite(value)) or np.any(np.abs(value.imag) > 1e-10):
        raise ValueError(f"{name} must contain finite real values")
    return np.asarray(value.real, dtype=float)


def load_integrals(paths, *, particles=None):
    """Load real restricted full-space integrals; optional MO e-ph coupling."""
    electronic, eph = discover_files(paths)
    meta, arrays = _read(electronic)
    if meta.get("h1_convention") != "spin_orbital_interleaved" or meta.get("h2_convention") != "chemist_spin_orbital_coulomb" or meta.get("h2_antisymmetrized", False):
        raise ValueError("Expected interleaved, non-antisymmetrized chemist spin-orbital integrals")
    if str(meta.get("method", "rhf")).lower() not in {"rhf", "hf", "scf"} or not meta.get("converged", True):
        raise ValueError("Converged restricted HF data required")
    h1, h2 = _real(arrays["h1"], "h1"), _real(arrays["h2"], "h2")
    if h1.ndim != 2 or h1.shape[0] != h1.shape[1] or h1.shape[0] == 0 or h1.shape[0] % 2:
        raise ValueError("h1 must be a nonempty even-dimensional square matrix")
    nso = len(h1)
    if h2.shape != (nso,) * 4:
        raise ValueError("h2 dimensions must agree with h1")
    spatial1, spatial2 = h1[::2, ::2], h2[::2, ::2, ::2, ::2]
    if not np.allclose(h1[1::2, 1::2], spatial1, atol=1e-10, rtol=0) or not np.allclose(h1[::2, 1::2], 0, atol=1e-10) or not np.allclose(h1[1::2, ::2], 0, atol=1e-10):
        raise ValueError("h1 violates restricted spin structure")
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                for d in (0, 1):
                    expected = spatial2 if a == b and c == d else 0
                    if not np.allclose(h2[a::2, b::2, c::2, d::2], expected, atol=1e-10, rtol=0):
                        raise ValueError("h2 violates restricted Coulomb spin structure")
    if not np.allclose(spatial1, spatial1.T, atol=1e-10, rtol=0):
        raise ValueError("h1 must be Hermitian")
    if any(not np.allclose(spatial2, transformed, atol=1e-10, rtol=0) for transformed in
           (spatial2.swapaxes(0, 1), spatial2.swapaxes(2, 3), spatial2.transpose(2, 3, 0, 1))):
        raise ValueError("h2 violates real chemist ERI symmetry")
    if particles is None:
        nelec = meta.get("nelec")
        if nelec is None or int(nelec) != nelec or nelec < 0 or nelec % 2:
            raise ValueError("Even nelec metadata or explicit --particles ALPHA BETA required")
        particles = (int(nelec) // 2,) * 2
    if len(particles) != 2 or any(int(n) != n or not 0 <= n <= len(spatial1) for n in particles) or particles[0] != particles[1] or meta.get("spin", meta.get("ms2", 0)) != 0:
        raise ValueError("Full-space closed-shell particle counts required")
    if "nelec" in meta and sum(particles) != meta["nelec"]:
        raise ValueError("Particle counts disagree with source nelec")
    coupling = frequencies = None
    sources = [str(electronic)]
    if eph is not None:
        eph_meta, eph_arrays = _read(eph)
        if str(eph_meta.get("eph_basis", "")).upper() != "MO":
            raise ValueError("eph_basis must be MO")
        for key in ("mol_str", "basis", "charge", "spin", "nelec", "method"):
            if key in meta and key in eph_meta and meta[key] != eph_meta[key]:
                raise ValueError(f"Electronic/e-ph metadata mismatch: {key}")
        coupling, frequencies = _real(eph_arrays["eph_mat"], "eph_mat"), _real(eph_arrays["omega"], "omega")
        if frequencies.ndim != 1 or len(frequencies) == 0 or np.any(frequencies <= 0):
            raise ValueError("omega must be a nonempty positive frequency vector")
        if coupling.shape != (len(frequencies), len(spatial1), len(spatial1)) or not np.allclose(coupling, coupling.swapaxes(1, 2), atol=1e-10, rtol=0):
            raise ValueError("eph_mat must contain one Hermitian spatial-MO matrix per mode")
        sources.append(str(eph))
        meta = {**meta, "electron_phonon": eph_meta}
    nuclear = _real(arrays["nuclear_repulsion"], "nuclear_repulsion")
    if nuclear.ndim != 0:
        raise ValueError("nuclear_repulsion must be scalar")
    return MolecularIntegrals(spatial1, spatial2, float(nuclear), tuple(map(int, particles)), coupling, frequencies, meta, tuple(sources))
