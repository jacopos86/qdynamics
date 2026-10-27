"""
Small HDF5 reader for qdynamics output files.

Examples
--------
List the contents of a file:

    python -m src.io_module.h5_reader results/psi4/h2o_psi4_matrix.h5

Read selected matrix elements:

    python -m src.io_module.h5_reader results/psi4/h2o_psi4_matrix.h5 --dataset h1 --index 0,0
    python -m src.io_module.h5_reader results/psi4/h2o_psi4_matrix.h5 --dataset h2 --index 0,0,1,1

Compare the same matrix element in two files:

    python -m src.io_module.h5_reader results/psi4/h2o_psi4_matrix.h5 \
        --compare results/pyscf/h2o_matrix.h5 --dataset h1 --index 0,0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def _h5py():
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "h5py is required to read HDF5 files. Install the project environment "
            "with `make configure install`, or install it with `python -m pip install h5py`."
        ) from exc
    return h5py


def _np():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "numpy is required to read HDF5 matrix data. Install the project "
            "environment with `make configure install`, or install it with "
            "`python -m pip install numpy`."
        ) from exc
    return np


def _decode_attr(value: Any) -> Any:
    np = _np()
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _parse_index(text: str | None) -> tuple[int, ...] | None:
    if text is None:
        return None
    parts = [part.strip() for part in text.split(",")]
    if not parts or any(part == "" for part in parts):
        raise ValueError("index must look like '0,1' or '0,0,1,1'")
    return tuple(int(part) for part in parts)


def summarize_h5(path: str | Path) -> list[str]:
    """Return human-readable lines describing groups, attrs, and datasets."""
    lines: list[str] = []
    h5py = _h5py()
    with h5py.File(path, "r") as h5_file:
        if "metadata" in h5_file:
            lines.append("metadata:")
            for key, value in sorted(h5_file["metadata"].attrs.items()):
                lines.append(f"  {key}: {_decode_attr(value)}")

        lines.append("datasets:")

        def visit(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            if isinstance(obj, h5py.Dataset):
                lines.append(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
            elif name != "metadata":
                lines.append(f"  {name}/")

        h5_file.visititems(visit)
    return lines


def read_dataset(path: str | Path, dataset: str) -> np.ndarray | np.generic:
    """Read a full dataset from an HDF5 file."""
    h5py = _h5py()
    with h5py.File(path, "r") as h5_file:
        return h5_file[dataset][()]


def read_element(path: str | Path, dataset: str, index: tuple[int, ...]) -> Any:
    """Read one element or slice from a dataset without loading all of it."""
    h5py = _h5py()
    np = _np()
    with h5py.File(path, "r") as h5_file:
        value = h5_file[dataset][index]
    if isinstance(value, np.generic):
        return value.item()
    return value


def compare_element(
    left_path: str | Path,
    right_path: str | Path,
    dataset: str,
    index: tuple[int, ...],
) -> tuple[Any, Any, Any]:
    """Return left value, right value, and left-right difference."""
    left = read_element(left_path, dataset, index)
    right = read_element(right_path, dataset, index)
    np = _np()
    return left, right, np.asarray(left) - np.asarray(right)


def _format_value(value: Any) -> str:
    np = _np()
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    return np.array2string(array, precision=12, suppress_small=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read or compare qdynamics HDF5 matrix-element files."
    )
    parser.add_argument("path", help="HDF5 file to inspect")
    parser.add_argument(
        "--dataset",
        help="Dataset name, for example h1, h2, eph_mat, mo_coeff, mo_energy",
    )
    parser.add_argument(
        "--index",
        help="Comma-separated zero-based index, for example 0,0 or 0,0,1,1",
    )
    parser.add_argument(
        "--compare",
        help="Second HDF5 file. Requires --dataset and --index.",
    )
    args = parser.parse_args(argv)

    index = _parse_index(args.index)

    if args.compare:
        if args.dataset is None or index is None:
            parser.error("--compare requires --dataset and --index")
        left, right, diff = compare_element(args.path, args.compare, args.dataset, index)
        print(f"{args.path}:{args.dataset}{index} = {_format_value(left)}")
        print(f"{args.compare}:{args.dataset}{index} = {_format_value(right)}")
        print(f"difference = {_format_value(diff)}")
        return 0

    if args.dataset and index is not None:
        value = read_element(args.path, args.dataset, index)
        print(f"{args.path}:{args.dataset}{index} = {_format_value(value)}")
        return 0

    if args.dataset:
        value = read_dataset(args.path, args.dataset)
        print(_format_value(value))
        return 0

    for line in summarize_h5(args.path):
        print(line)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ModuleNotFoundError, OSError, KeyError, IndexError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
