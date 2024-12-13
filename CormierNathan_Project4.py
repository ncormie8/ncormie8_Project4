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
#  Lab 12; reused and adapted:
#       - Vertical stacked plotting code for displaying psi(x) and P(x) at a given time
#
#  Western Brightspace. (n.d.). Physics 3926, Project 4: The Schrodinger Equation
#  UWO. https://westernu.brightspace.com/d2l/le/enhancedSequenceViewer/29104?url=https%3A%2F%2F832d4e9b-197e-4ab6-a2ca-c4f7aae74b20.sequences.api.brightspace.com%2F29104%2Factivity%2F2552026%3FfilterOnDatesAndDepth%3D1
#       - Used provided instuctions to complete Project 4
#       - Reused and adapted description of sch_eqn() and its parameters for sch_eqn() Docstring
#  
#  Numerical Methods For Physics (Python) - Alejandro L. Garcia 2E revised, 2017. pgs 227-240

# Reused/Modified functions from previous labs and projects:

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
    psiSquare = np.dot(wavefunction,np.conjugate(wavefunction))
    total_prob = np.sum(psiSquare)
    
    return np.real(total_prob) # return as real component to avoid "ComplexWarning: Casting complex values to real discards the imaginary part" as probability is real

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

    # Initializing the x grid array to solve over
    x_grid = np.arange(nspace)*h - L/2
    
    # Initializing the time grid array to solve over
    t_grid = np.arange(0,ntime*tau,tau)

    # Initializing the potential array
    Vx = np.zeros(nspace)  # creating zeros array of potentials to match each x position
    Vx[potential] = 1      # setting V = 1 for all values specified by input arg potential

    # Creating the Psi(x,t) function
    psi = np.zeros((nspace,ntime),dtype=complex)    # initializing empty array of shape (nspace x ntime) with complex type so as not to lose imaginary components
    psi[:,0] = make_initialcond_sch(wparam,x_grid)  # setting values of psi at t = 0 to calculated values of the initial wave function

    # Normalizing psi(x,0)
    psi[:,0] = psi[:,0]/(total_probability(psi[:,0])**(1/2.))

    # Creating and initializing the probability function
    probability = np.zeros(ntime)
    probability[0] = total_probability(psi[:,0])  # initial probability should be 100%
    
    # Construction of Hamiltonian matrix for use in FTCS and Crank-Nicolson (CN) integration schemes
    coeff_Ham = (-1*_h_bar_given**2)/(2*_m_given*(h**2))  
    H = make_tridiagonal(nspace,coeff_Ham,-2*coeff_Ham,coeff_Ham) + np.diag(Vx)
    
    # Implementation of periodic boundary conditions for the Hamiltonian matrix
    H[0,-1] = coeff_Ham
    H[-1,0] = coeff_Ham
    
    # Evolution matrices for FTCS and CN integration schemes
    I = np.identity(nspace)
    A_ftcs = I - (_imag_i*tau/_h_bar_given)*H
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
                psi[:, i]  = np.dot(A_ftcs,psi[:,i-1])  # Solving Schroedingers Equation with FTCS method
                probability[i] = total_probability(psi[:,i])  # determines total probability of finding particle at time i
            
            return psi, x_grid, t_grid, probability

        # If unstable, notify user and terminate integration
        else:
            print('FTCS integration is unstable for input timestep.\nPlease try again.\nINTEGRATION TERMINATED')
            return [0,[0],0,0] # all zeros incdicate failed integration, stops expected 4 elements unpacking error from occuring

    # # Logic pathway for Crank_Nicolson integration
    elif method == 'crank':
        # no stability analysis required, stable for all tau
        # Main Crank-Nicolson Loop - ranges from one to ntime since psi at t = 0 is the initial condition
        for i in range(1, ntime):
        # Solving Schroedingers Equation with CN method
                psi[:, i] = np.dot(A_cn,psi[:,i-1])  # Solving Schroedingers Equation with CN method
                probability[i] = total_probability(psi[:,i])  # determines total probability of finding particle at time i
      
        return  psi, x_grid, t_grid, probability
                
psi, x, t, total_probs = sch_eqn(100,10001,1e-2,method='crank')
results = [psi,x,t,total_probs]

### COMMENTING and formatting not done yet but fully functional
def sch_plot(results,save=False):
    '''Plot the outputs of sch_eqn() at a user selected time point and produce a plot of the Gaussian waveform, 
    probability density, or both. Users may optionally save plots upon generation.
    
    Required parameter: 
    - results (list): contains the results of sch_eqn() function call [phi, x_grid, t_grid, total_probabilities]
    
    Optional parameter:
    - save (Bool): enables automatic saving of generated plots when set to True (Default = False)'''
    
    # Unpacking results from sch_eqn()
    psi = results[0]
    x = results[1]
    t = results[2]
    total_probs = results[3]
    
    # Checking if the input results are from failed or successful integration
    if  x[0] == 0:  # (This will never be true unless integration fails)
        print('\nsch_eqn() did not complete integration.\nPLOTTING TERMINATED')
        return

    # Setting upper and lower bounds for selectable timepoints for the user to choose from
    timePt_lb = int(t[0])
    timePt_ub = int(np.size(t)-1)

    # Extracting the value of the timestep
    timestep = t[1]

    # Promting user for desired timepoint to plot phi, prob, or both at
    print('The timestep is '+str(timestep)+'s')
    timePt_selected = int(input('Please input a timepoint within the following range; '+str(timePt_lb)+'-'+str(timePt_ub)+'\n'+'Input: '))

    # Verifying that a valid time point within the given range was entered
    if timePt_selected <= timePt_ub and timePt_selected >= timePt_lb:
        
        # Defining the actual time value that the desired plot type is to be generated at
        actual_time = t[timePt_selected]
        print('Time selected = '+str(actual_time)+'s')

        # Promting user for desired plot type; phi, prob, or both
        print('Would you like a plot of psi or probability?')
        plot_type = (str(input('options:\n- psi\n- prob\n- both\nInput: ')))
        plot_type = plot_type.lower()

        # Logical pathway for plotting the Gaussian waveform at the specified time
        if plot_type == 'psi':
            plt.plot(x,np.real(psi[:,timePt_selected])) # taking real part of psi to avoid casting warnings
            plt.xlabel('X position')
            plt.ylabel('Psi(x)')
            plt.title('Positional at time '+str(actual_time)+'s')
            plt.grid()

            # Logical pathway for saving the plot of phi if the user called for it
            if save is True:
                plt.savefig('psi_at_t_'+str(actual_time)+'s.png')    # Saving the plot
                print('Figure saved sucessfully.')                   # Verifiying that the plot was saved

            plt.show()
            print('Psi plotting complete')
            return

        # Logical pathway for plotting probability density
        elif plot_type =='prob':
            # Calculating the probability denstiy for the selected time
            pDensity = np.real(psi[:,timePt_selected]*np.conjugate(psi[:,timePt_selected]))
            
            # Plotting probability density in terms of position at the specified time
            plt.plot(x,pDensity)
            plt.xlabel('X position')
            plt.ylabel('Probability Density')
            plt.title('Positional Probability Density at time '+str(actual_time)+'s')
            plt.grid()
            
            # Logical pathway for saving the plot of prob if the user called for it
            if save is True:
                plt.savefig('prob_at_t_'+str(actual_time)+'s.png')   # Saving the plot
                print('Figure saved sucessfully.')                   # Verifiying that the plot was saved

            plt.show()
            print('Probability plot complete.')
            return
        
        # Logical pathway for plotting both types
        elif plot_type == 'both':
            # Defining 1 figure with 2 data sets
            fig, (ax1,ax2) = plt.subplots(2)

            # Plotting the Gaussian waveform in terms of position at the specified time
            ax1.plot(x,np.real(psi[:,timePt_selected])) 

            # Formatting and labeling for the top panel graph (waveform)
            ax1.set_title('Positional Gaunssian Wavefunction Psi at time '+str(actual_time)+'s')
            ax1.set_xlabel('x position')
            ax1.set_ylabel('Psi(x)')
            ax1.grid()

            # Calculating and plotting the probability density in terms of position at the specified time
            pDensity = np.real(psi[:,timePt_selected]*np.conjugate(psi[:,timePt_selected]))
            ax2.plot(x,pDensity)

            # Formatting and labeling for the bottom panel graph (probability density)
            ax2.set_title('Positional Probability Density at time '+str(actual_time)+'s')
            ax2.set_xlabel('x position')
            ax2.set_ylabel('Probability Density')
            ax2.grid()

            # Fixes layout issues by stopping plots from having overlapping
            fig.tight_layout()
            
            # Logical pathway for saving the dual plot if the user called for it
            if save is True:
                plt.savefig('psi_and_prob_at_t_'+str(actual_time)+'s.png')    # Saving the plot
                print('Figure saved sucessfully.')                            # Verifiying that the plot was saved
            
            plt.show()
            print('Dual plot complete.')
            return

        # Logical pathway for an invalid user input plot type
        else:
            print('Invalid input plot type. Please try again.')
            return
        
    # Logical pathway for an invalid user input timepoint
    else:
        print('Invalid timepoint selected. Please try again.')
        return 

sch_plot(results)