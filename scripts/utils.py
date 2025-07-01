import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from classes.regular import simulation
from classes.voronoi import voronoi_fire

from typing import List, Dict, Any


def infinite_pc(l: List[int], save_route:str, fire_args: Dict[str,Any], pc_args: Dict[str,Any]) -> None:
    """
    This function computes and plots the scaling of the Pc for infinite systems.
    It takes a list with the L size of the sytem to compute the corresponding Pc values.
    The function plots the Pc values against the system sizes.
    """
    
    # Initialize an empty array to store Pc values
    pc_values = np.zeros(len(l))    
    #pc_values = np.array([0.52,0.511,0.510,0.509,0.508,0.507, 0.505])
    #pc_values = np.array([0.538,0.540,0.541, 0.545,0.536,0.519,0.514,0.527,0.513])
    
    # Loop through each size in the list
    for i,size in enumerate(l):
        # Create matrix for each size
        matrix = np.ones((size,size))
        matrix[size//2,size//2] = 2
        # Create a simulation object with the given size
        sim = simulation.squareForest(**fire_args, initialForest=matrix)
        
        # Run the simulation to get the percolation threshold
        #pc = sim.estimate_percolation_threshold(**pc_args, matrix=matrix, fixed_value=fire_args['burningThreshold'])
        pc = sim.fit_percolation_threshold(**pc_args,matrix=matrix)
        # Append the Pc value to the list
        pc_values[i] = pc
        
    # Execute the fit using curve fit to estimate pc_inf, a and nu
    try:
        popt, pcov = curve_fit(pc_scaling, np.array(l), pc_values, np.array([0.5,5,1]))
        # Extract parameters
        pc_inf, a_fit, nu_fit= popt
        print(f"Estimated pc(∞) = {pc_inf:.5f}")
        print(f"Estimated ν = {nu_fit:.5f}")
        print(f"Estimated a = {a_fit:.5f}")
        L_inv_nu = np.array(l)**(-1/nu_fit)
        #plt.scatter(L_inv_nu, pc_values, label="Simulation data")

        # Plotting
        plt.scatter(1/np.array(l)**nu_fit, pc_values, label="Simulation data")
        plt.plot(1/np.array(l)**nu_fit, pc_scaling(np.array(l), pc_inf, a_fit,nu_fit), label="Fit", color='red')
        #plt.plot(L_inv_nu, pc_inf + a_fit * np.array(l)**(-1/nu_fit), label="Fit", color='red')
        #plt.plot(1/line, pc_scaling(line, 0.5,1,1), label="Fit", color='red')
        plt.xlabel(r"$L^{-1/\nu}$")
        plt.ylabel(r"$p_c(L)$")
        plt.title("Finite-Size Scaling Fit")
        plt.legend()
        plt.grid(True)
        caption = f"pc(∞) = {pc_inf:.5f}" + f"\nν = {nu_fit:.5f}" + f"\na = {a_fit:.5f}"
        plt.legend(title=caption)
        plt.tight_layout()
        plt.savefig(save_route + 'pc_infinite_systems.png')
    except:
        print('fit not converged.')
    
    # Plot the Pc values against the system sizes
    #plt.plot(np.array(l), pc_values, marker='o', linestyle='-', color='b')
    #plt.xlabel('System Size (L)')
    #plt.ylabel('Percolation Threshold (Pc)')
    #plt.title('Scaling of Percolation Threshold (Pc) for Infinite Systems')
    #plt.grid(True)
    plt.savefig(save_route + 'pc_infinite_systems.png')  


# Define scaling function
def pc_scaling(L, pc_inf, a, nu,b):
    '''
    This function is used by infinite_pc function to exert a regression to find the parameters pc_inf, a and nu
    '''
    return pc_inf + a * L**(-nu)

