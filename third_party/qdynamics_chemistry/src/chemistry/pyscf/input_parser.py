"""
input_parser.py
---------------
Input-file parsing module.

Responsibility: read a job input file (.inp) and return a validated
settings dict. Knows nothing about PySCF or file writing.

Input file format
-----------------
  key = value               settings, one per line
  # ...                     comments (also allowed after a value)
  geometry ... end          multi-line block with atom coordinates,
                            or:  geometry = path/to/file.xyz

Recognized keys (with defaults):
  method  (RHF)     RHF | ROHF | UHF | RKS | UKS
  basis   (sto-3g)
  xc      (None)    DFT functional, RKS/UKS only
  charge  (0)
  spin    (0)       2S, number of unpaired electrons
  unit    (Angstrom)  Angstrom | Bohr
  output  (required)  output path prefix, e.g. results/h2o
  write_h5      (true)
  write_fcidump (true)
  fcidump_tol   (1e-12)
  write_vibration (false)   harmonic vibrational analysis
  write_eph       (false)   electron-phonon coupling matrix elements
  eph_method      (analytic) analytic | finite_difference
  eph_cutoff_frequency (80.0) cm^-1
  eph_keep_imag_frequency (false)
  eph_fd_step     (0.001) Cartesian-normal-mode displacement in Bohr
  e_converg       (1e-12) SCF energy convergence
  d_converg       (1e-8)  SCF density/gradient convergence
  max_iter        (100)   maximum SCF iterations
"""
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_DEFAULTS = {
    'method': 'RHF',
    'basis': 'sto-3g',
    'xc': None,
    'charge': 0,
    'spin': 0,
    'unit': 'Angstrom',
    'output': None,          # required
    'write_h5': True,
    'write_fcidump': True,
    'fcidump_tol': 1e-12,
    'write_vibration': False,
    'write_eph': False,
    'eph_method': 'analytic',
    'eph_cutoff_frequency': 80.0,
    'eph_keep_imag_frequency': False,
    'eph_fd_step': 0.001,
    'e_converg': 1.0e-12,
    'd_converg': 1.0e-8,
    'max_iter': 100,
    'isotope_masses': None,
}

_VALID_METHODS = {'RHF', 'ROHF', 'UHF', 'RKS', 'UKS'}
_VALID_UNITS = {'ANGSTROM', 'BOHR'}
_BOOL_MAP = {'true': True, 'yes': True, '1': True,
             'false': False, 'no': False, '0': False}


def _clean(line):
    """Strip inline comments and surrounding whitespace."""
    idx = line.find('#')
    if idx >= 0:
        line = line[:idx]
    return line.strip()


def _read_xyz(path):
    """Read an .xyz file, return the atom block as a PySCF-friendly string."""
    lines = Path(path).read_text().strip().splitlines()
    try:
        natoms = int(lines[0].split()[0])
        atom_lines = lines[2:2 + natoms]
    except (ValueError, IndexError):
        atom_lines = lines
    return '\n'.join(atom_lines)


def parse_input(path):
    """
    Parse a job input file.

    Returns
    -------
    dict with keys: mol_str, method, basis, xc, charge, spin, unit,
                    output (Path), write_h5, write_fcidump, fcidump_tol
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

        # ---- geometry block ----
        if in_geometry:
            if line.lower() == 'end':
                in_geometry = False
            else:
                geometry_lines.append(line)
            continue
        if line.lower() == 'geometry':
            in_geometry = True
            continue

        # ---- key = value ----
        if '=' not in line:
            raise ValueError(f"{path}:{lineno}: cannot parse line: {raw!r}")
        key, _, value = line.partition('=')
        key, value = key.strip().lower(), value.strip()

        if key == 'geometry':          # geometry = some/file.xyz
            geometry_path = Path(value)
            if not geometry_path.is_absolute():
                geometry_path = path.parent / geometry_path
            geometry_lines = _read_xyz(geometry_path).splitlines()
        elif key in ('charge', 'spin', 'max_iter'):
            settings[key] = int(value)
        elif key == 'isotope_masses':
            settings[key] = [float(item.strip()) for item in value.split(',')]
        elif key in ('fcidump_tol', 'eph_cutoff_frequency', 'eph_fd_step',
                     'e_converg', 'd_converg'):
            settings[key] = float(value)
        elif key in ('write_h5', 'write_fcidump', 'write_vibration', 'write_eph',
                     'eph_keep_imag_frequency'):
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

    # ---- validation ----
    if not geometry_lines:
        raise ValueError(f"{path}: no geometry given "
                         "(use a 'geometry ... end' block or 'geometry = file.xyz')")
    if settings['output'] is None:
        raise ValueError(f"{path}: 'output' is required (e.g. output = results/h2o)")

    settings['method'] = settings['method'].upper()
    if settings['method'] not in _VALID_METHODS:
        raise ValueError(f"{path}: unsupported method {settings['method']!r} "
                         f"(choose from {sorted(_VALID_METHODS)})")
    if settings['unit'].upper() not in _VALID_UNITS:
        raise ValueError(f"{path}: unit must be Angstrom or Bohr")
    settings['eph_method'] = settings['eph_method'].lower()
    if settings['eph_method'] not in ('analytic', 'finite_difference', 'fd'):
        raise ValueError(f"{path}: eph_method must be analytic or finite_difference")
    if settings['eph_fd_step'] <= 0:
        raise ValueError(f"{path}: eph_fd_step must be positive")
    if settings['e_converg'] <= 0 or settings['d_converg'] <= 0:
        raise ValueError(f"{path}: SCF convergence thresholds must be positive")
    if settings['max_iter'] <= 0:
        raise ValueError(f"{path}: max_iter must be positive")

    settings['output'] = Path(settings['output'])
    settings['mol_str'] = '\n'.join(geometry_lines)
    if (settings['isotope_masses'] is not None
            and len(settings['isotope_masses']) != len(geometry_lines)):
        raise ValueError(f"{path}: isotope_masses must contain one value per atom")

    log.info(f"Input parsed: {path} "
             f"(method={settings['method']}, basis={settings['basis']}, "
             f"{len(geometry_lines)} atoms)")
    return settings
