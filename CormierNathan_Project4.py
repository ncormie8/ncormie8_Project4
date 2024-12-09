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
    
    # converting input arguments into shorter easier versions for calculations
    N = nspace
    t = ntime
    
    # putting a meaningless return here so i avoid errors while setting up the function
    return t 