"""
input_parser.py
---------------
Input-file parsing module for Psi4 jobs.

Responsibility: read a job input file (.inp) and return a validated
settings dict. This mirrors the PySCF parser style while preserving the
Psi4-specific calculation parameters used by Psi4Driver.

Input file format
-----------------
  key = value               settings, one per line
  # ...                     comments (also allowed after a value)
  geometry ... end          multi-line block with atom coordinates,
                            or:  geometry = path/to/file.xyz

Recognized keys (with defaults):
  method  (scf)
  basis   (sto-3g)          basis name or per-element map
  basis_set_file (basis.gbs)
  optimized_coordinate_file (optimized.xyz)
  reference (rhf)           rhf | uhf | rohf | rks | uks
  scf_type  (pk)            pk | df | direct | out_of_core | cd
  charge  (0)
  multiplicity (1)
  unit    (Angstrom)        Angstrom | Bohr
  output  (None)            optional output path prefix
  write_h5 (true)
  write_matrix_elements (true)
  e_converg (1e-8)
  d_converg (1e-8)
  max_iter (1000)
  guess (sad)
  soscf (false)
  soscf_max_iter (0)
  write_vibration (false)
  write_eph       (false)
"""
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_DEFAULTS = {
    "method": "scf",
    "basis": "sto-3g",
    "basis_set_file": "basis.gbs",
    "optimized_coordinate_file": "optimized.xyz",
    "reference": "rhf",
    "scf_type": "pk",
    "charge": 0,
    "multiplicity": 1,
    "unit": "Angstrom",
    "output": None,
    "write_h5": True,
    "write_matrix_elements": True,
    "e_converg": 1.0e-8,
    "d_converg": 1.0e-8,
    "max_iter": 1000,
    "guess": "sad",
    "soscf": False,
    "soscf_max_iter": 0,
    "write_vibration": False,
    "write_eph": False,
}

_VALID_SCF_TYPES = {"pk", "df", "direct", "out_of_core", "cd"}
_VALID_REFERENCES = {"rhf", "uhf", "rohf", "rks", "uks"}
_VALID_UNITS = {"ANGSTROM", "BOHR"}
_BOOL_MAP = {
    "true": True,
    "yes": True,
    "1": True,
    "false": False,
    "no": False,
    "0": False,
}


def _clean(line):
    """Strip inline comments and surrounding whitespace."""
    idx = line.find("#")
    if idx >= 0:
        line = line[:idx]
    return line.strip()


def _read_xyz(path):
    """Read an .xyz file and return (unit, atom_lines)."""
    lines = Path(path).read_text().strip().splitlines()
    try:
        natoms = int(lines[0].split()[0])
        unit = lines[1].strip() if len(lines) > 1 else "Angstrom"
        atom_lines = lines[2:2 + natoms]
    except (ValueError, IndexError):
        unit = "Angstrom"
        atom_lines = lines
    return unit, atom_lines


def _write_xyz(path, atom_lines, unit):
    """Write a Psi4-compatible XYZ file used by the current Psi4 driver."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = [str(len(atom_lines)), unit, *atom_lines]
    path.write_text("\n".join(contents) + "\n")


def _parse_basis_map(value):
    """
    Parse a compact per-element basis map.

    Examples
    --------
    basis = sto-3g
    basis = H:sto-3g, O:6-31g
    """
    entries = [item.strip() for item in value.split(",")]
    if not any(":" in item for item in entries):
        return value
    basis_map = {}
    for entry in entries:
        symbol, sep, basis_name = entry.partition(":")
        if not sep or not symbol.strip() or not basis_name.strip():
            raise ValueError(f"cannot parse basis mapping entry {entry!r}")
        basis_map[symbol.strip()] = basis_name.strip()
    return basis_map


def parse_input(path):
    """
    Parse a Psi4 job input file.

    Returns
    -------
    dict with keys: coordinate_file, optimized_coordinate_file, basis_set_file,
                    basis_set, charge, multiplicity, unit, output,
                    psi4_calc_parameters, write_vibration, write_eph
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    settings = dict(_DEFAULTS)
    geometry_lines = []
    in_geometry = False

    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = _clean(raw)
        if not line:
            continue

        if in_geometry:
            if line.lower() == "end":
                in_geometry = False
            else:
                geometry_lines.append(line)
            continue
        if line.lower() == "geometry":
            in_geometry = True
            continue

        if "=" not in line:
            raise ValueError(f"{path}:{lineno}: cannot parse line: {raw!r}")
        key, _, value = line.partition("=")
        key, value = key.strip().lower(), value.strip()

        if key == "geometry":
            coordinate_file = Path(value)
            if not coordinate_file.is_absolute():
                coordinate_file = path.parent / coordinate_file
            settings["unit"], geometry_lines = _read_xyz(coordinate_file)
            settings["coordinate_file"] = coordinate_file
        elif key == "basis":
            settings[key] = _parse_basis_map(value)
        elif key in ("charge", "multiplicity", "max_iter", "soscf_max_iter"):
            settings[key] = int(value)
        elif key in ("e_converg", "d_converg"):
            settings[key] = float(value)
        elif key in ("soscf", "write_h5", "write_matrix_elements",
                     "write_vibration", "write_eph"):
            try:
                settings[key] = _BOOL_MAP[value.lower()]
            except KeyError:
                raise ValueError(f"{path}:{lineno}: {key} must be true/false, got {value!r}")
        elif key in _DEFAULTS:
            settings[key] = value
        else:
            raise ValueError(f"{path}:{lineno}: unknown keyword {key!r}")

    if in_geometry:
        raise ValueError(f"{path}: geometry block not closed with 'end'")
    if not geometry_lines:
        raise ValueError(
            f"{path}: no geometry given "
            "(use a 'geometry ... end' block or 'geometry = file.xyz')"
        )

    settings["reference"] = settings["reference"].lower()
    if settings["reference"] not in _VALID_REFERENCES:
        raise ValueError(
            f"{path}: unsupported reference {settings['reference']!r} "
            f"(choose from {sorted(_VALID_REFERENCES)})"
        )

    settings["scf_type"] = settings["scf_type"].lower()
    if settings["scf_type"] not in _VALID_SCF_TYPES:
        raise ValueError(
            f"{path}: unsupported scf_type {settings['scf_type']!r} "
            f"(choose from {sorted(_VALID_SCF_TYPES)})"
        )
    if settings["unit"].upper() not in _VALID_UNITS:
        raise ValueError(f"{path}: unit must be Angstrom or Bohr")

    if "coordinate_file" not in settings:
        settings["coordinate_file"] = path.with_suffix(".xyz")
        _write_xyz(settings["coordinate_file"], geometry_lines, settings["unit"])

    settings["coordinate_file"] = Path(settings["coordinate_file"])
    settings["optimized_coordinate_file"] = Path(settings["optimized_coordinate_file"])
    settings["basis_set_file"] = Path(settings["basis_set_file"])
    if settings["output"] is not None:
        settings["output"] = Path(settings["output"])

    parsed = {
        "coordinate_file": settings["coordinate_file"],
        "optimized_coordinate_file": settings["optimized_coordinate_file"],
        "basis_set_file": settings["basis_set_file"],
        "basis_set": settings["basis"],
        "charge": settings["charge"],
        "multiplicity": settings["multiplicity"],
        "unit": settings["unit"],
        "output": settings["output"],
        "write_h5": settings["write_h5"],
        "write_matrix_elements": settings["write_matrix_elements"],
        "write_vibration": settings["write_vibration"],
        "write_eph": settings["write_eph"],
        "psi4_calc_parameters": {
            "scf_type": settings["scf_type"],
            "reference": settings["reference"],
            "method": settings["method"],
            "e_converg": settings["e_converg"],
            "d_converg": settings["d_converg"],
            "max_iter": settings["max_iter"],
            "guess": settings["guess"],
            "soscf": settings["soscf"],
            "soscf_max_iter": settings["soscf_max_iter"],
        },
    }

    log.info(
        f"Input parsed: {path} "
        f"(method={settings['method']}, basis={settings['basis']}, "
        f"{len(geometry_lines)} atoms)"
    )
    return parsed
