import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.spatial import Voronoi

from classes.regular import simulation
from classes.voronoi import voronoi_fire
from scripts.routes import data_route


from typing import List, Dict, Any


def infinite_pc(l: List[int], save_route:str, fire_args: Dict[str,Any], pc_args: Dict[str,Any],tessellation:str) -> None:
    """
    This function computes and plots the scaling of the Pc for infinite systems.
    It takes a list with the L size of the sytem to compute the corresponding Pc values.
    The function plots the Pc values against the system sizes.
    """
    pc_values = np.zeros(len(l))    
    pc_error = np.zeros(len(l))
    
    # Initialize an empty array to store Pc values
    #pc_values = np.array([0.52,0.511,0.510,0.509,0.508,0.507, 0.505])
    #pc_values = np.array([0.533,0.5309,0.527, 0.5286,0.5227,0.5207,0.525,0.523,0.52,0.524,0.524,0.521,0.521,0.519,0.5219]) - 0.02
    #pc_values = np.array([0.5011,0.5058,0.5099, 0.5011,0.5058,0.4994,0.5087,0.5034,0.5046,0.5034,0.5034,0.5046,0.499,0.5058,0.5046])    # Squared 
    #pc_values = np.array([0.6866907724050582,00.6859349145063431,0.6821760536046251, 0.6769483769483771,0.6762701476987192,0.6753692467978183,0.6726216011930298,0.6721435721435722,0.6721435721435722,0.6689076831933976,0.6689853118424547,0.6687074829931974,0.667181046352474,0.6621030785887929,0.666185050356479])    # Triangular 

    # Loop through each size in the list
    for i,size in enumerate(l):
        # Create a simulation object with the given size

        if tessellation == "squared":
            # Create matrix for each size
            matrix = np.ones((size,size))
            matrix[size//2,size//2] = 2

            sim = simulation.squareForest(**fire_args, initialForest=matrix)
            route = data_route['squared'] + pc_args['fixed'] + str(pc_args['fixed_value']) + '_pc_infinite_systems.csv'
            pc, error = sim.fit_percolation_threshold(**pc_args,matrix=matrix)

        elif tessellation == "triangular":
            # Create matrix for each size
            matrix = np.ones((size,size))
            matrix[size//2,size//2] = 2

            sim = simulation.triangularForest(**fire_args, initialForest=matrix)
            route = data_route['triangular'] + pc_args['fixed'] + str(pc_args['fixed_value']) + '_pc_infinite_systems.csv'
            pc, error = sim.fit_percolation_threshold(**pc_args,matrix=matrix)

        elif tessellation == "hexagonal":
        # Create matrix for each size
            matrix = np.ones((size,size))
            matrix[size//2,size//2] = 2
            sim = simulation.heaxgonalForest(**fire_args, initialForest=matrix)
            route = data_route['hexagon'] + pc_args['fixed'] + str(pc_args['fixed_value']) + '_pc_infinite_systems.csv'
            pc, error = sim.fit_percolation_threshold(**pc_args,matrix=matrix)

        elif tessellation == "voronoi":

            nPoints = size*size
            points = np.random.rand(nPoints, 2)
            vor = Voronoi(points)

            sim = voronoi_fire.voronoiFire(**fire_args, voronoi=vor, initialFire=1)
            route = data_route['voronoi'] + pc_args['fixed'] + str(pc_args['fixed_value']) + '_pc_infinite_systems.csv'

            pc, error = sim.fit_percolation_threshold(**pc_args)
        
        # Run the simulation to get the percolation threshold
        # Append the Pc value adn its error to the list
        pc_values[i] = pc
        pc_error[i] = error
        
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
        invL_nu = 1 / np.array(l)**nu_fit
        fit_line = pc_scaling(np.array(l), pc_inf, a_fit, nu_fit)

        plt.figure(figsize=(6, 5))

        # Datos con barras de error
        plt.errorbar(invL_nu, pc_values, yerr=pc_error, fmt='o', color='black',
                     ecolor='gray', elinewidth=1.5, capsize=4, label="Simulation")

        # Curva ajustada
        plt.plot(invL_nu, fit_line, '-', color='red', label="Fit")

        # Ejes y texto
        plt.xlabel(r"$L^{-1/\nu}$", fontsize=14)
        plt.ylabel(r"$p_c(L)$", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, alpha=0.3)

        # Texto con parámetros y sus errores
        caption = (r"$p_c(\infty) = {:.3f} \pm {:.3f}$" + "\n" +
                   r"$\nu = {:.3f} \pm {:.3f}$" + "\n" +
                   r"$a = {:.3f} \pm {:.3f}$").format(
                       pc_inf, np.sqrt(pcov[0, 0]),
                       nu_fit, np.sqrt(pcov[2, 2]),
                       a_fit, np.sqrt(pcov[1, 1])
                   )

        plt.legend(title=caption, fontsize=11, title_fontsize=12, loc="best")

        # Ajuste final y guardado
        plt.tight_layout()
        plt.savefig(save_route + pc_args['fixed'] + str(pc_args['fixed_value']) + '_pc_infinite_systems.png', dpi=300)

        # Crear DataFrame con los datos
        df = pd.DataFrame({
            'L': l,
            '1/L^nu': invL_nu,
            'pc(L)': pc_values,
            'pc_error': pc_error
        })
        
        # Guardar como CSV
        df.to_csv(route, index=False)
    except:
        print('fit not converged.')
    
    


# Define scaling function
def pc_scaling(L, pc_inf, a, nu):
    '''
    This function is used by infinite_pc function to exert a regression to find the parameters pc_inf, a and nu
    '''
    return pc_inf + a * L**(-nu)


# voronoi objects sizes for finding infinite system scaling pc
