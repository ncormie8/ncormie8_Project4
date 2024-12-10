import numpy as np
from matplotlib import pyplot
import time                # for code testing and user experience (time.sleep())

# References
#  Lab 10; Reused and adapted:
#       - spectral_radius() to perform stability analysis of FTCS integration 
#       - Probably going to need the make_initialcond() and make_tridiagonal() for matrix constuction

#
#  Lab 11; Reused and adapted:
#       - x_grid, 
#       - t_grid, 
#       - Inital condition code for V(x)
#       - Method lower casing
#       - Stability analysis logic from spectral_radius() output
# 
#  Western Brightspace. (n.d.). Physics 3926, Project 4: The Schrodinger Equation
#  UWO. https://westernu.brightspace.com/d2l/le/enhancedSequenceViewer/29104?url=https%3A%2F%2F832d4e9b-197e-4ab6-a2ca-c4f7aae74b20.sequences.api.brightspace.com%2F29104%2Factivity%2F2552026%3FfilterOnDatesAndDepth%3D1
#       - Used provided instuctions to complete Project 4
#       - Reused and adapted description of sch_eqn() and its parameters for sch_eqn() Docstring
#
#

# Reused functions from previous labs and projects:

# Lab 10 - Spectral radius; used in performing stability analysis of FTCS integration
def spectral_radius(A):
    '''Takes input parameter A as an array, and returns the absolute maximum 
    eigenvalue of the input array A.'''

    # copmputes eigenvalues and eigen vectors of the input array A
    # and assigns them to variables eigval and eigvect
    eigval, eigvect = np.linalg.eig(a=A)

    # generating array containing the absolute values of all elements
    # in eigval
    abs = np.abs(eigval)

    # setting maxAbs equal to abs at the index where the maximum arugment is located
    maxAbs = abs[np.argmax(abs)]
    
    return maxAbs

#---------------------------------------------------------------------------------------------------#
# Project 4 - Main

# Initial parameter declarations (using values from Lab 11 to test functionality for now)
nspace_tbd = 300         # number of spatial grid points (int)
ntime_tbd = 501         # number of time steps to be solved over (int)
tau_tbd = 0.016666666666666666 # tau is the time step to be solved with (float)

# Function that solves the one-dimensional, time-dependent Schroedinger Equation
def sch_eqn(nspace, ntime, tau, method='ftcs', length=200, potential=[], wparam=[10,0,0.5]):
    '''Function which solves the one-dimensional, time-dependent Schroedinger Equation.
    Requires input parameters nspace (int) for the number of spatial grid points, ntime (int)
    for the number of time steps to be solved over, and tau for the time step to be used (float).
    Accepts optional parameters method (Default = 'ftcs'), length for the size of the spatial grid
    (Default = 200), potential for the spatial index values where the potential V(x) is set to
    the initial condition value 1 (Default = []), and wparam for the inital conditions of the wave
    corresponding to sigma0, x0, and k0 (Default = [10,0,0.5]). Returns solution as a two-dimensional 
    array psi(x,t), and a one-dimensional array which gives the total probability computed for each
    timestep of integration.'''

    # Converting input arguments into shorter better versions for integration
    L = length
    h = L/(nspace-1)         # x_grid spacing parameter (minus 1 so it goes perfectly from -L/2 to L/2)
    method = method.lower()  # modifying input string to be all lowercase for versatility

    # Constants
    _plk_given = 1  # plancks constant
    _m_given = 0.5  # particle mass

    # Initializing the x grid array to solve over (tested - appears to be working properly)
    x_grid = np.arange(-L/2,L/2+h,h)

    # Initializing the time grid array to solve over (tested - appears to be working properly)
    t_grid = np.arange(0,ntime*tau,tau)

    # Initializing the potential array (tested - appears to be working properly)
    Vx_initial = np.empty(np.size(x_grid))  # Creating empty array of potentials to match each x position
    Vx_initial[potential] = 1               # Setting V = 1 for all values specified by input arg potential

    # Evolution matrices for FTCS and Crank-Nicolson integration schemes
    # NEXT STEP

    # Logic pathway for FTCS integration (default)
    if method == 'ftcs':
        
        # Temporary placeholder for actual evolution matrix of schEQN
        evolution_matrix_tbd = [] 

        # Stability analysis for FTCS integration
        stab_val = spectral_radius(evolution_matrix_tbd)
    
        # If stable, proceed with FTCS integration
        if stab_val < 1:
            print('FTCS integation is stable for input parameters.')
            print('PROCEEDING WITH INTEGRATION')
    
            # Then integrate with FTCS scheme
            return #solution to sch eqn
        
        # If unstable, notify user and terminate integration
        else:
            print('FTCS integration is unstable for input parameters.')
            print('Please try again.')
            print('INTEGRATION TERMINATED')
            return # nothing, integration will not proceed 


    # Logic pathway for Crank_Nicolson integration
    # elif method == 'crank':
        # no stability analysis required, stable for all tau