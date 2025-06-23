import numpy as np
import matplotlib.pyplot as plt

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
    
    # Loop through each size in the list
    for i,size in enumerate(l):
        # Create matrix for each size
        matrix = np.ones((size,size))
        matrix[size//2,size//2] = 2
        # Create a simulation object with the given size
        sim = simulation.squareForest(**fire_args, initialForest=matrix)
        
        # Run the simulation to get the percolation threshold
        pc = sim.estimate_percolation_threshold(**pc_args, matrix=matrix, fixed_value=fire_args['burningThreshold'])
        
        # Append the Pc value to the list
        pc_values[i] = pc
    
    # Plot the Pc values against the system sizes
    plt.plot(1/np.array(l), pc_values, marker='o', linestyle='-', color='b')
    plt.xlabel('System Size (L)')
    plt.ylabel('Percolation Threshold (Pc)')
    plt.title('Scaling of Percolation Threshold (Pc) for Infinite Systems')
    plt.grid(True)
    plt.savefig(save_route + 'pc_infinite_systems.png')  



