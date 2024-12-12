import numpy as np
from numpy import linalg
from matplotlib import pyplot as plt
import time                # for code testing and user experience (time.sleep())

# References:
#  Lab 4; Reused:
#       - Code to fix the random state for reproducibility
#
#  Lab 10; Reused and adapted:
#       - spectral_radius() to perform stability analysis of FTCS integration 
#       - adapted make_initialcond() to be able to apply the initial condition to psi/Schroedingers Equation
#       - make_tridiagonal() for evolution matrix constuction
#
#  Lab 11; Reused and adapted:
#       - x_grid definition
#       - t_grid definition
#       - Inital condition code for V(x)
#       - Input method lower casing
#       - Stability analysis logic from spectral_radius() output
#       - Copied solving loops for use in FTCS and Crank-Nicolson integration schemes
# 
#  Western Brightspace. (n.d.). Physics 3926, Project 4: The Schrodinger Equation
#  UWO. https://westernu.brightspace.com/d2l/le/enhancedSequenceViewer/29104?url=https%3A%2F%2F832d4e9b-197e-4ab6-a2ca-c4f7aae74b20.sequences.api.brightspace.com%2F29104%2Factivity%2F2552026%3FfilterOnDatesAndDepth%3D1
#       - Used provided instuctions to complete Project 4
#       - Reused and adapted description of sch_eqn() and its parameters for sch_eqn() Docstring
#  
#  Numerical Methods For Physics (Python) - Alejandro L. Garcia 2E revised, 2017. pgs 227-240

# Fixing random state for reproducibility


# Reused functions from previous labs and projects:

# Lab 10 - Spectral radius; used in performing stability analysis of FTCS integration
def spectral_radius(A):
    '''Takes input parameter A as an array, and returns the absolute maximum 
    eigenvalue of the input array A.'''
    
    # computes eigenvalues and eigen vectors of the input array A
    # and assigns them to variables eigval and eigvect
    eigval, eigvect = np.linalg.eig(A)
    
    # generating array containing the absolute values of all elements
    # in eigval
    abs = np.abs(eigval)
    
    # setting maxAbs equal to abs at the index where the maximum arugment is located
    maxAbs = abs[np.argmax(abs)]
    
    return maxAbs

# Lab 10 - Make tridiagonal; used in constructing evolution matrix
def make_tridiagonal(N, b, d, a):
    '''Takes input parameters N for size of square array, b for the value along lower/below diagonal,
    d for the value along the central diagonal, a for the value along the upper/above diagonal. Returns matrix A with
    lower diagonal values of b, central diagonal values of d, upper diagonal values of a, and all other values of 0,
    of shape (N,N).'''

    # initializing A array as N x N array of zeros
    A = np.zeros(shape=(N,N))
    
    # filling central diagonal of a with input argument d
    np.fill_diagonal(a=A,val=d)
    
    # initializing BEL array as N x N array with lower diagonal equal to b
    BEL = np.diag(np.full(shape=(N-1),fill_value=b),k=-1)
    
    # initializing ABV array as N x N arr ay with above diagonal equal to a
    ABV = np.diag(np.full(shape=(N-1),fill_value=a),k=1)
    
    # adding upper and lower diagonals to A to be returned
    A = A + BEL + ABV
    return A

# Lab 10 - Make initial condition; modified to apply initial condition to Shroedingers Equation
def make_initialcond_sch(wparam,space_grid):
    '''Function which computes the initial condition of a Gaussian wave packet. Intakes two input
    parameters wparam and space_grid. The first argument wparam (list), contains the initial packet width, 
    initial particle localization, and wave number as shown here; [sig_0, x0, k0]. The second argument 
    space_grid (list), contains position values xi of the spatial grid. Returns the values of the Gaussian 
    wave packet at time t = 0 at all spatial grid positions as a list.'''
    
    # Unpacking input wave parameters from wparam arugment
    sig_0 = wparam[0]     # initial wave packet width
    x0 = wparam[1]        # initial particle localization
    k0 = wparam[2]        # wavenumber

    # Setting input spatial grid equal to x for to condense code
    x = space_grid

    # Calculating constant for determining initial Gaussian wave packet form
    const_psix0 = 1/(sig_0*(np.pi)**(1/2.))**(1/2.)

    # Initializing psi(x,0) array as list full of zeros of the same size as the input spatial 
    # grid array x
    psi_x0 = np.zeros(np.size(x))

    # Calculating the initial condition values of psi for all values of x at time t = 0
    psi_x0 = const_psix0*(np.exp(_imag_i*k0*x)*np.exp((-1*(x-x0)**2)/(2*(sig_0**2))))

    return psi_x0

#------------------------------------------------------------------------------------------------------------#
# Project 4 - Main

# Probability function
def total_probability(wavefunction):
    '''This function calculates the total probability of finding the particle in time point t.'''
    # Probability function loop   
    psiSquare = wavefunction*np.conjugate(wavefunction)
    total_prob = np.sum(psiSquare)
    return total_prob


# Initial parameter declarations (using values from Lab 11 to test functionality for now)
nspace_tbd = 100  # number of spatial grid points (int)
ntime_tbd = 501   # number of time steps to be solved over (int)
tau_tbd = 1e-500    # tau is the time step to be solved with (float)

# Global constants (identified by _ as first character)
_h_bar_given = 1.0  # plancks constant
_m_given = 0.5     # particle mass (m)
_imag_i = 1j      # complex number i (root(-1))

# Function that solves the one-dimensional, time-dependent Schroedinger Equation
def sch_eqn(nspace, ntime, tau, method='ftcs', length=200, potential=[], wparam=[10,0,0.5]):
    '''Function which solves the one-dimensional, time-dependent Schroedinger Equation.
    
    Requires input parameters:
      - nspace (int): The number of spatial grid points  
      - ntime (int): The number of time steps to be solve over
      - tau (float): The timestep of integration.
    
    Accepts optional parameters:
      - method (string): The method of integration to be used; either 'ftcs' or 'crank' (Default = 'ftcs') (case insensitive)  
      - length (int): The size of the spatial grid (Default = 200)
      - potential (list): A spatial index with values that correspond to where the potential V(x) is set to the initial condition value 1 (Default = [])
      - wparam (list): The parameters of the initial Gaussian wave function corresponding to sigma0, x0, and k0 (Default = [10,0,0.5])
    
    Returns the solution to the one-dimensional, time-dependent Schroedinger Equation as a two-dimensional array psi(x,t), 
    and a one-dimensional array which gives the total probability computed for each timestep of integration.'''

    # Converting input arguments into shorter better versions for integration
    L = length
    h = L/(nspace-1)         # x_grid spacing parameter (minus 1 so it goes perfectly from -L/2 to L/2)
    method = method.lower()  # modifying input string to be all lowercase for versatility

    # Initializing the x grid array to solve over (tested - appears to be working properly)
    x_grid = np.arange(nspace)*h - L/2
    
    # Initializing the time grid array to solve over (tested - appears to be working properly)
    t_grid = np.linspace(0,ntime*tau,ntime)

    # Initializing the potential array (tested - appears to be working properly)
    Vx = np.zeros(np.size(x_grid))  # creating zeros array of potentials to match each x position (used empty originally and caused many issues)
    Vx[potential] = 1               # setting V = 1 for all values specified by input arg potential
    # ^^^ PROBLEM SOURCE

    # Creating the Psi(x,t) function
    psi = np.zeros((nspace,ntime),dtype=complex)  # initializing empty array of shape (nspace x ntime) with complex type so as not to lose imaginary components
    psi[:,0] = make_initialcond_sch(wparam,x_grid)     # setting values of psi at t = 0 to calculated values of the initial wave function
    probabilities = np.zeros(ntime) 

    # Constuction of Hamiltonian matrix for use in FTCS and Crank-Nicolson (CN) integration schemes
    coeff_Ham = (-1*_h_bar_given**2)/(2*_m_given*(h**2))  
    H = make_tridiagonal(nspace,coeff_Ham,-2*coeff_Ham,coeff_Ham) + np.diag(Vx)
    
    # Implementation of periodic boundary conditions for the Hamiltonian matrix
    H[0,-1] = coeff_Ham
    H[-1,0] = coeff_Ham
    
    # Evolution matrices for FTCS and CN integration schemes
    I = np.identity(nspace)
    A_ftcs = I - _imag_i*tau/_h_bar_given*H
    A_cn= np.dot(np.linalg.inv(I + (_imag_i*tau/(2*_h_bar_given))*H),(I - (_imag_i*tau/(2*_h_bar_given))*H))

    # Logic pathway for FTCS integration (default)
    if method == 'ftcs': 
        
        # Stability analysis for FTCS integration
        stab_val = spectral_radius(A_ftcs)

        # If stable, proceed with FTCS integration
        if stab_val <= 1:
            print('FTCS integation is stable for input timestep.')
            print('PROCEEDING WITH INTEGRATION')
            # Main FTCS Loop - ranges from one to ntime since psi at t = 0 is the initial condition
            for i in range(1, ntime):
                psi[:, i]  = A_ftcs.dot(psi[:,i-1])  # Solving Schroedingers Equation with FTCS method
                probabilities[i] = total_probability(psi[:,i])  # determines total prob fo finding particle at time i
            
            return psi, x_grid, t_grid, probabilities

        # If unstable, notify user and terminate integration
        else:
            print('FTCS integration is unstable for input timestep.\nPlease try again.\nINTEGRATION TERMINATED')
            return [0,0,0,0] # all zeros incdicate failed integration, stops expected 4 elements unpacking error from occuring

    # # Logic pathway for Crank_Nicolson integration
    elif method == 'crank':
        # no stability analysis required, stable for all tau
        # Main Crank-Nicolson Loop - ranges from one to ntime since psi at t = 0 is the initial condition
        for i in range(1, ntime):
        # Solving Schroedingers Equation with CN method
            psi[:, i]  = A_cn.dot(psi[:,i-1])
            probabilities[i] = total_probability(psi[:,i])
       
        return  psi, x_grid, t_grid, probabilities
        
psi, x, t, probabs = sch_eqn(nspace_tbd,ntime_tbd,tau_tbd)
