import numpy as np
from matplotlib import pyplot

# References
# Lab 11; Reused and adapted:
#       - x_grid, 
#       - t_grid, 
#       - Inital condition code for V(x)
#       - Method lower casing
#   
#
#

#---------------------------------------------------------------------------------------------------#
# Project 4

# Initial parameter declarations (using values from Lab 11 to test functionality for now)
nspace_tbd = 300         # number of spatial grid points (int)
ntime_tbd = 501         # number of time steps to be solved over (int)
tau_tbd = 0.016666666666666666 # tau is the time step to be solved with (float)

# Function that solves the 1D time dependent Schroedinger Equation
def sch_eqn(nspace, ntime, tau, method='ftcs', length=200, potential=[], wparam=[10,0,0.5]):
    
    # Converting input arguments into shorter better versions for integration
    L = length
    h = L/(nspace-1)         # x_grid spacing parameter (minus 1 so it goes perfectly from -L/2 to L/2)
    method = method.lower()  # modifying input string to be all lowercase for versatility

    # Constants
    _plk_given = 1
    _m_given = 0.5

    # Initializing the x grid array to solve over (tested - appears to be working properly)
    x_grid = np.arange(-L/2,L/2+h,h)

    # Initializing the time grid array to solve over (tested - appears to be working properly)
    t_grid = np.arange(0,ntime*tau,tau)

    # Initializing the potential array (tested - appears to be working properly)
    Vx_initial = np.empty(np.size(x_grid))  # Creating empty array of potentials to match each x position
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

t_test = sch_eqn(nspace_tbd,ntime_tbd,tau_tbd)
print(t_test)