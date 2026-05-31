import numpy as np
from scipy.constants import physical_constants
from scipy.constants import electron_volt
from src.common.units import h, Q_
#
# physical constants module
#
mp = Q_(physical_constants["proton mass"][0], "kg")
# electron mass
me = Q_(physical_constants["electron mass"][0], "kg")
#
kb = Q_(physical_constants["Boltzmann constant"][0] / electron_volt, "eV / K")
#
hbar = h / (2. * np.pi)
#
# Bohr magneton (eV/T)
#
mu_B = Q_(physical_constants["Bohr magneton in eV/T"][0], "eV / T")  # eV T^-1
g_e = 2.00231930436256
gamma_e = g_e * mu_B / 2.0                                           # eV T^-1
#
# nuclear gyromagnetic ratio  : gamma_n / 2pi
#
gamma_n = Q_(physical_constants["nuclear magneton in MHz/T"][0], "MHz / T")