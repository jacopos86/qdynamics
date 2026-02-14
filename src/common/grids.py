import numpy as np

#
#    This module contains the functions
#    to produce real time and frequency grids
#

# set w_grid

def set_w_grid(w_max, nwg):
    dw = w_max / (nwg - 1)
    w_grid = np.zeros(nwg)
    # compute w grid
    for iw in range(nwg):
        w_grid[iw] = iw * dw
    return w_grid

# set time arrays

def set_time_grid_A(T, nt):
    # time interv.
    dt = T / (nt - 1)
    time = np.zeros(nt)
    for it in range(nt):
        time[it] = it * dt
    return time

#
# time arrays

def set_time_grid_B(T, dt):
    nt = int(T / dt)
    time = np.linspace(0., T, nt)
    return time

# set temperature grid

def set_temperatures(Tlist):
    temperatures = np.array(Tlist)
    return temperatures