import numpy as np
from matplotlib import pyplot

# References
#
#
#
#

#---------------------------------------------------------------------------------------------------#
# Project 4

# Initial parameter declarations
nspace_tbd = 101010101        # number of spatial grid points (int)
ntime_tbd = 101010101         # number of time steps to be solved over (int)
tau_tbd = 101010101           # tau is the time step to be solved with (float)

# Function that solves the 1D time dependent Schroedinger Equation
def sch_eqn(nspace, ntime, tau, method='ftcs', length=200, potential=[], wparam=[10,0,0.5]):
    
    # Converting input arguments into shorter easier versions for calculations
    L = length
    h = L/nspace      # x_grid spacing parameter

    # Constants
    _plk_given = 1
    _m_given = 0.5

    # Initializing the x grid array to solve over
    x_grid = np.arange(-L/2,L/2,h)

    # Initializing the time grid array to solve over
    t_grid = np.arange(0,ntime*tau,tau)

    # Initializing the potential array
    Vx_initial = np.zeros(np.size(x_grid))  # Creating zeros array of potentials to match each x position
    Vx_initial[potential] = 1               # Setting V = 1 for all values specified by input arg potential


    # Logic pathway for FTCS integration (default)
    # if method == 'ftcs':
        
        # stability analysis for FTCS
        # stab_val = some equation from lab 10

        # if stab_val is stable:
        #     integrate (see lab 10)
        # else:
        #     print('FTCS integration is unstable for input parameters.')
        #     print('Please try again.')
        #     break

    # Logic pathway for Crank_Nicolson integration
    # elif method == 'crank':
        # no stability analysis required, stable for all tau


    # putting a meaningless return here so i avoid errors while setting up the function
    return t_grid