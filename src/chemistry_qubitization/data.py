"""Explicit contracts between reading, mapping and persistence."""
from dataclasses import dataclass
from typing import Any

import numpy as np

from .pauli import PauliPolynomial


@dataclass(frozen=True)
class MolecularIntegrals:
    h1: np.ndarray  # spatial MO, chemist convention
    h2: np.ndarray
    nuclear_repulsion: float
    particles: tuple[int, int]
    coupling: np.ndarray | None
    frequencies: np.ndarray | None
    metadata: dict[str, Any]
    source_files: tuple[str, ...]


@dataclass(frozen=True)
class QubitHamiltonian:
    operator: PauliPolynomial
    n_qubits: int
    n_fermion_qubits: int
    particles: tuple[int, int]
    cutoffs: tuple[int, ...]
    encoding: str | None
    frequencies: tuple[float, ...]
    source_files: tuple[str, ...]
    metadata: dict[str, Any]
