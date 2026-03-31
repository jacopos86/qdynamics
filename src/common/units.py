import pint

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

# Define aliases you use often
THz = ureg.terahertz
ps  = ureg.picosecond
ns  = ureg.nanosecond
fs  = ureg.femtosecond
us  = ureg.microsecond
eV  = ureg.electron_volt
T   = ureg.tesla
K   = ureg.kelvin
Ang = ureg.angstrom

# Planck constant
h   = ureg.planck_constant

# internal units dict
Eh = 1. * ureg.hartree
a0 = 1. * ureg.bohr

internal_units = {
    "energy": Eh,
    "length": a0,
    "mass": ureg.electron_mass,
    "time": ureg.hbar / Eh
}