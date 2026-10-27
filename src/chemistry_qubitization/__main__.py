"""Read chemistry HDF5 files or a directory and save a qubit Hamiltonian."""
import argparse
from pathlib import Path
import sys

# Also support: python src/chemistry_qubitization/__main__.py INPUT ...
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.chemistry_qubitization import load_integrals, build_hamiltonian, save_hamiltonian


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="HDF5 files or a directory containing one calculation")
    parser.add_argument("--output", "-o", required=True, type=Path)
    parser.add_argument("--cutoffs", nargs="+", type=int, help="Maximum boson occupation, shared or per mode")
    parser.add_argument("--encoding", choices=("binary", "unary"), default="binary")
    parser.add_argument("--particles", nargs=2, type=int, metavar=("ALPHA", "BETA"))
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        integrals = load_integrals(args.inputs, particles=args.particles)
        qubit_hamiltonian = build_hamiltonian(integrals, cutoffs=args.cutoffs, encoding=args.encoding, tolerance=args.tolerance)
        save_hamiltonian(qubit_hamiltonian, args.output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Saved {qubit_hamiltonian.n_qubits}-qubit Hamiltonian ({qubit_hamiltonian.operator.count_number_terms()} Pauli terms) to {args.output}")


if __name__ == "__main__":
    main()
