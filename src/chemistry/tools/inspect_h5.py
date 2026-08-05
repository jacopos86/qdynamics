"""Print the structure and metadata of an HDF5 chemistry result file.

Usage
-----
    python -m src.chemistry.tools.inspect_h5 results/pyscf/ele.h5
    python -m src.chemistry.tools.inspect_h5 results/pyscf/ele.h5 --data
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def _format_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return np.array2string(value, threshold=20, edgeitems=3)
    return str(value)


def inspect_h5(path, show_data=False, max_items=20):
    """Print HDF5 metadata, dataset shapes and optionally small data previews."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    print(f"File: {path}")
    with h5py.File(path, "r") as handle:
        print("\nStructure:")

        def visit(name, obj):
            indent = "  " * (name.count("/") + 1)
            if isinstance(obj, h5py.Group):
                print(f"{indent}{name}/")
                for key, value in obj.attrs.items():
                    print(f"{indent}  @{key} = {_format_value(value)}")
            elif isinstance(obj, h5py.Dataset):
                print(f"{indent}{name}: shape={obj.shape}, dtype={obj.dtype}")
                if show_data:
                    data = obj[()]
                    flat = np.asarray(data).reshape(-1)
                    preview = flat[:max_items]
                    suffix = " ..." if flat.size > max_items else ""
                    print(f"{indent}  data={_format_value(preview)}{suffix}")

        for key, value in handle.attrs.items():
            print(f"  @{key} = {_format_value(value)}")
        handle.visititems(visit)


def inspect_h5_dict(path, include_data=False, max_items=20):
    """Return a JSON-serializable summary of an HDF5 file."""
    path = Path(path)
    summary = {"file": str(path), "metadata": {}, "datasets": {}}
    with h5py.File(path, "r") as handle:
        summary["metadata"] = {
            str(k): _format_value(v) for k, v in handle["metadata"].attrs.items()
        } if "metadata" in handle else {}

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                item = {"shape": list(obj.shape), "dtype": str(obj.dtype)}
                if include_data:
                    flat = np.asarray(obj[()]).reshape(-1)
                    item["preview"] = [
                        _format_value(v) for v in flat[:max_items]
                    ]
                summary["datasets"][name] = item

        handle.visititems(visit)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="HDF5 file to inspect")
    parser.add_argument(
        "--data", action="store_true", help="include dataset values"
    )
    parser.add_argument(
        "--max-items", type=int, default=20,
        help="maximum values to show per dataset (default: 20; use 0 for all)",
    )
    parser.add_argument(
        "--json", type=Path,
        help="write a JSON summary to this path instead of printing text",
    )
    args = parser.parse_args()
    max_items = None if args.max_items == 0 else args.max_items
    if args.json:
        summary = inspect_h5_dict(args.file, include_data=args.data,
                                  max_items=max_items)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"JSON summary written to {args.json}")
    else:
        inspect_h5(args.file, show_data=args.data, max_items=max_items)


if __name__ == "__main__":
    main()
