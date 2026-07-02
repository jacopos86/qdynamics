from scipy.constants import physical_constants
from scipy.constants import electron_volt
from scipy.constants import Rydberg, e, h, c
#
# physical constants module
#
mp = physical_constants["proton mass"][0]
mp = mp / electron_volt * 1.E30 * 1.E-20
# proton mass (eV fs^2/Ang^2)
mp = mp * 1.E-6                      # eV ps^2/Ang^2
# electron mass
me = physical_constants["electron mass"][0]
me = me / electron_volt * 1.E30 * 1.E-20
# electron mass (eV fs^2/Ang^2)
me = me * 1.E-6                      # eV ps^2/Ang^2
#
THz_to_ev = physical_constants["Planck constant"][0]
# J sec
THz_to_ev = THz_to_ev / electron_volt * 1.E12
#
kb = physical_constants["Boltzmann constant"][0]
kb = kb / electron_volt              # eV/K
#
hbar = physical_constants["Planck constant over 2 pi"][0]
hbar = hbar / electron_volt * 1.E12  # eV ps
#
hartree2joule = physical_constants["Hartree energy"][0]  # J
hartree2ev = hartree2joule / electron_volt
#
rytoev = Rydberg * h * c / e
AUTOA = physical_constants["Bohr radius"][0] / 1.E-10
#
# tolerance parameter
#
eps = 1.E-7
#
# Bohr magneton (eV/G)
#
mu_B = physical_constants["Bohr magneton in eV/T"][0]  # eV T^-1
g_e = 2.00231930436256
gamma_e = g_e * mu_B / 2.0                             # eV T^-1
#
# nuclear gyromagnetic ratio  : gamma_n / 2pi
#
gamma_n = physical_constants["nuclear magneton in MHz/T"][0]
gamma_n = gamma_n * 1.E-4            # MHz / G