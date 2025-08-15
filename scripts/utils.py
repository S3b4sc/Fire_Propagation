import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.spatial import Voronoi
import joblib
from fitter import Fitter

from classes.regular import simulation
from classes.voronoi import voronoi_fire
from scripts.routes import data_route
from classes.fit.fitting import lognormal
from scipy.signal import savgol_filter


from typing import List, Dict, Any, Tuple

from scipy.stats import johnsonsu  # Best fit for oblique cut

# Function to estimate the infinite system pc

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
    #pc_values = np.array([0.51645,0.512847,0.512968,0.511546,0.50986,0.50776,0.509264,0.506862])
    #pc_error = np.zeros([len(l)])

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
        popt, pcov = curve_fit(pc_scaling, np.array(l), pc_values, np.array([0.5,20,1.3]),maxfev=20000) #------
        # Extract parameters
        pc_inf, a_fit, nu_fit= popt
        print(f"Estimated pc(∞) = {pc_inf:.5f}")
        print(f"Estimated ν = {nu_fit:.5f}")
        print(f"Estimated a = {a_fit:.5f}")
        L_inv_nu = np.array(l)**(-1/nu_fit)
        plt.scatter(L_inv_nu, pc_values, label="Simulation data")

        # Plotting
        invL_nu = 1 / np.array(l)**nu_fit
        fit_line = pc_scaling(np.array(l), pc_inf, nu_fit) #------
        plt.figure(figsize=(6, 5))

        # Datos con barras de error
        plt.errorbar(invL_nu, pc_values, yerr=pc_error, fmt='o', color='black',
                     ecolor='gray', elinewidth=1.5, capsize=4, label="Simulation")

        # Curva ajustada
        plt.plot(invL_nu, fit_line, '-', color='red', label="Fit")
        #plt.plot(invL_nu,pc_scaling(np.array(l), 0.5,20,1.33))

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

        
    except:
        # Crear DataFrame con los datos
        df = pd.DataFrame({
            'L': l,
            'pc(L)': pc_values,
            'pc_error': pc_error
        })
        
        print('fit not converged.')
        # Guardar como CSV
        df.to_csv(save_route + 'pc_scaling_data.csv', index=False)
        
    
    
# Define scaling function
def pc_scaling(L, pc_inf, a, nu):
    '''
    This function is used by infinite_pc function to exert a regression to find the parameters pc_inf, a and nu
    '''
    return pc_inf + a * L**(-nu)

# 

def extract_oblique_cut(data: pd.DataFrame, ps_vals: np.ndarray, pb_vals: np.ndarray, tol=1e-6):
    """
    Given a list of (p_site, p_bond) values, find matching time values in the data.
    A small tolerance is used in case of floating point mismatch.
    """
    times = []
    for ps, pb in zip(ps_vals, pb_vals):
        match = data[(np.abs(data['P_site'] - ps) < tol) & (np.abs(data['P_bond'] - pb) < tol)]
        if not match.empty:
            times.append(match['time'].values[0])
        else:
            times.append(np.nan)  # or raise warning
    return np.array(times)



def sigma(
    folder_path: str,
    save_path:   str,
    tessellation: str,
    n_full:       int,
    m_full:       int,
    n_refine:     int,
    m_refine:     int,
    find_dist:   bool = False,
    vertical:    bool = False,
    grad_frac:   float = 0.2,
    initial_guess_ps:List[float] = [0,1],
    initial_guess_pb:List[float] = [0,1]
) -> None:
    
    """
    Fully simulation-based range selection: first sample coarsely,
    detect active region by gradient, then refine sampling and fit.

    Parameters
    ----------
    folder_path  : str  (unused)
    save_path    : str  prefix for output files
    tessellation : str  one of 'squared','triangular','hexagonal','voronoi'
    n_full, m_full  : ints for coarse sampling and replicas
    n_refine, m_refine: for refined sampling
    find_dist    : bool, if True summarize distribution only
    vertical     : bool, if True do P_site=1, sweep P_bond
    grad_frac    : float fraction of max gradient to threshold
    initial_guess:List[float] = [0,1] are the initial values for pb and ps 
    """
    # 1) define full cut in parameter space
    if vertical:
        ps_full = np.ones(n_full)
        pb_full = np.linspace(initial_guess_pb[0],initial_guess_pb[1],n_full)
    else:
        ps_full = np.linspace(initial_guess_ps[0], initial_guess_ps[1], n_full)
        pb_full = ps_full.copy()
        #ps_full = np.linspace(initial_guess_ps[0],initial_guess_ps[1],n_full)
        #pb_full = np.ones(n_full)

    raw_times = np.zeros(n_full)
    for i, (ps, pb) in enumerate(zip(ps_full, pb_full)):
        pivot_times = np.zeros(m_full)

        for j in range(m_full):
            if tessellation == 'squared':

                matrix = np.ones((700,700))
                matrix[350,350] = 2
                forest = simulation.squareForest(burningThreshold=1,occuProba=1 ,initialForest=matrix)
                pivot_times[j] = forest.propagateFire(ps, pb)

                if j == (m_full-1):
                    forest = simulation.squareForest(burningThreshold=1,occuProba=1 ,initialForest=matrix)
                    final_time = forest.propagateFire(1,1)

            elif tessellation == 'triangular':
            
                matrix = np.ones((100,100))
                matrix[50,50] = 2
                forest = simulation.triangularForest(burningThreshold=1,occuProba=1 ,initialForest=matrix)
                pivot_times[j] = forest.propagateFire(ps, pb)

                if j == (m_full-1):
                    forest = simulation.triangularForest(burningThreshold=1,occuProba=1 ,initialForest=matrix)
                    final_time = forest.propagateFire(1,1)

            elif tessellation == 'hexagonal':
                matrix = np.ones((500,500))
                matrix[250,250] = 2
                forest = simulation.heaxgonalForest(burningThreshold=1,occuProba=1 ,initialForest=matrix)
                pivot_times[j] = forest.propagateFire(ps, pb)

                if j == (m_full-1):
                    forest = simulation.heaxgonalForest(burningThreshold=1,occuProba=1 ,initialForest=matrix)
                    final_time = forest.propagateFire(1,1)

            elif tessellation == 'voronoi':
                nPoints = 100*100
                points = np.random.rand(nPoints, 2)
                vor = Voronoi(points)

                forest = voronoi_fire.voronoiFire(1,1,vor,1)
                pivot_times[j] = forest.propagateFire(ps, pb, centered=True)

                if j == (m_full-1):
                    forest = voronoi_fire.voronoiFire(1,1,vor,1)
                    final_time = forest.propagateFire(1,1)
            
        
        raw_times[i] = np.mean(pivot_times)

    ## Extract time values along the diagonal
    #times_on_diagonal = extract_oblique_cut(data, ps_range_complete, pb_range_complete)

    # Filter based of lattest times
    #plt.plot(raw_times)
    #plt.savefig('try.png')

    # Compute the smoothed gradient
    g = gradient_smoothed(raw_times, dx=1/n_full, window=11, polyorder=2)

    # Pick fractional tolerance
    tol_frac = 0.01   # 10%

    # Compute the absolute‐gradient threshold
    grad_thresh = tol_frac * np.max(np.abs(g))

    # Build mask
    mask = (
        (raw_times > final_time)    # time‐offset criterion
        & (np.abs(g) > grad_thresh) # relative‐threshold criterion
    )

    #mask =  raw_times > (final_time) & (np.abs(gradient_smoothed(raw_times,1/n_full)) > 12)
    #print(gradient_smoothed(raw_times,1/n_full))
    #plt.plot(raw_times[mask]/100)
    #plt.plot(raw_times/100)
    #plt.plot(np.abs(gradient_smoothed(raw_times,1/n_full))/1000)
    #plt.savefig('try.png')
    #print(final_time)
    #print(mask)
    pb_range = pb_full[mask]
    ps_range = ps_full[mask]
    times_range = raw_times[mask]

    # Execute simulations and store results
    #known endpoints
    ps0, pb0 = ps_range[0], pb_range[0]
    ps1, pb1 = ps_range[-1], pb_range[-1]

    #print(ps0,ps1)
    #print(pb0,ps1)

    if vertical:
        # Choose as many ps_fine values as you like
        pb_fine = np.linspace(pb0, pb1, n_refine)

        # Compute the corresponding pb_fine EXACTLY on that line
        ps_fine = np.ones(n_refine)

    else:
        # Compute slope and intercept
        slope = (pb1 - pb0) / (ps1 - ps0)
        b = pb0 - slope * ps0

        # Choose as many ps_fine values as you like
        ps_fine = np.linspace(ps0, ps1, n_refine)
        #print(ps_fine[0], ps_fine[-1])

        # Compute the corresponding pb_fine EXACTLY on that line
        pb_fine = slope * ps_fine + b

    # Run your simulations


    # Fill points with simulations
    
    new_times = np.zeros(n_refine)
    new_errors = np.zeros(n_refine)
    for i, (ps, pb) in enumerate(zip(ps_fine, pb_fine)):
        pivot_times = np.zeros(m_refine)
        for j in range(m_refine):
            if tessellation == 'squared':

                matrix = np.ones((700,700))
                matrix[350,350] = 2
                forest = simulation.squareForest(burningThreshold=1,occuProba=1 ,initialForest=matrix)
                pivot_times[j] = forest.propagateFire(ps, pb)

            elif tessellation == 'triangular':
            
                matrix = np.ones((100,100))
                matrix[50,50] = 2
                forest = simulation.triangularForest(burningThreshold=1,occuProba=1 ,initialForest=matrix)
                pivot_times[j] = forest.propagateFire(ps, pb)

            elif tessellation == 'hexagonal':
                matrix = np.ones((100,100))
                matrix[50,50] = 2
                forest = simulation.heaxgonalForest(burningThreshold=1,occuProba=1 ,initialForest=matrix)
                pivot_times[j] = forest.propagateFire(ps, pb)

            elif tessellation == 'voronoi':
                nPoints = 100*100
                points = np.random.rand(nPoints, 2)
                vor = Voronoi(points)

                forest = voronoi_fire.voronoiFire(1,1,vor,1)
                pivot_times[j] = forest.propagateFire(ps, pb, centered=True)
            
        
        new_times[i] = np.mean(pivot_times)
        new_errors[i] = np.std(pivot_times)
        

    #print(forest.propagateFire(1, 0.8))

    # Execute fit to extract distribution width

    # Compute the x_data for fit over the oblique line
    # SHift them so they start from 0
    delta_ps = ps_fine - ps_fine[0]
    delta_pb = pb_fine - pb_fine[0]
    #print(delta_pb[0], delta_pb[-1])

    if find_dist:

        oblique_data   = np.sqrt(delta_ps**2 + delta_pb**2)
        save = save_path + 'oblique_cut_' + tessellation + '_upper_fixed.png'
        #popt, pcov = curve_fit(johnsonsu_pdf, oblique_data, new_times, p0=p0, maxfev = 5000)
        mask2 = new_times > (final_time)
        new_times =  new_times[mask2]
        oblique_data = oblique_data[mask2]
        new_errors = new_errors[mask2]
        f = Fitter(new_times,
               timeout=30)
        f.fit()

        # Best distributions sorted by sum of squared errors (SSE)
        #f.summary()
        print(f.get_best())

        # Optional: plot
        f.plot_pdf()
        plt.savefig(save_path + 'oblique_cut_' + tessellation + '_1.png')

    else:

        best_params = {
        'a': -2.03,
        'b': 1.39,
        'loc': -0.00309,
        'scale': 0.0189,
        'A': 4,
        'C': 101
        }
        p0 = [best_params['a'], best_params['b'], best_params['loc'], best_params['scale'], best_params['A'], best_params['C']]


        oblique_data   = np.sqrt(delta_ps**2 + delta_pb**2)
        save = save_path + 'oblique_cut_' + tessellation + '_middle_fixed_700.png'
        #popt, pcov = curve_fit(johnsonsu_pdf, oblique_data, new_times, p0=p0, maxfev = 5000)
        mask2 = new_times > (final_time)
        new_times =  new_times[mask2]
        oblique_data = oblique_data[mask2]
        new_errors = new_errors[mask2]

        df = pd.DataFrame({
        "time": new_times,
        "data": oblique_data,
        "error": new_errors,
        "pb_fine": pb_fine[mask2],
        "ps_fine": ps_fine[mask2]
        })

        # Guardar como CSV
        df.to_csv("datos_filtrados.csv", index=False)        


        plt.plot(oblique_data,new_times)
        plt.savefig('try.png')
        popt, pcov, fwhm, fwhm_err = fit_and_plot_johnsonsu_with_fwhm(oblique_data, new_times,new_errors, p0, save)
#   
        #print(popt)

        # Plot
        #plt.plot(oblique_data, new_times, 'o',  label='simulations')
        ##plt.plot(oblique_data, johnsonsu_pdf(oblique_data, *popt), '-', label='johnsonsu fit')
        #plt.xlabel('P_site = P_bond')
        #plt.ylabel('Time')
        #plt.title('Time along 45° diagonal cut')
        #plt.grid(True)
        #plt.legend()
        #plt.savefig(save_path + 'oblique_cut_' + tessellation + '_down.png')
       
# Define the PDF model function
def johnsonsu_pdf(x, a, b, loc, scale, A, C):
    """JohnsonSU PDF scaled by A and shifted by C."""
    return C + A * johnsonsu.pdf(x, a, b, loc=loc, scale=scale)
    
    

def fit_and_plot_johnsonsu_with_fwhm(x_data, y_data, y_errors, p0, save, mc_runs=500):
    """
    x_data, y_data : arrays of shape (n,)
    y_err          : array of shape (n,) containing the pointwise stddev from your m replicas
    p0             : initial guess for [a, b, loc, scale, A, C]
    save           : path+filename for the output PNG
    mc_runs        : how many Monte‑Carlo samples to draw
    """
    plt.plot(x_data,y_data, 'ko')
    plt.savefig('try.png')
    # --- 1) Fit ---

    lower_bounds = [-6, 0.1, -0.3, 1e-4, 0, 0]
    upper_bounds = [10, 10, 0.2, 0.2, 5000, 5000]

    popt, pcov = curve_fit(johnsonsu_pdf,
                           x_data, y_data, #sigma = y_errors,          # weight by errors
                           absolute_sigma=True,    # so pcov is in true units
                           p0=p0, maxfev=20000, bounds=(lower_bounds, upper_bounds))
    
    a_fit, b_fit, loc_fit, scale_fit, A_fit, C_fit = popt
    print(a_fit, b_fit, loc_fit, scale_fit, A_fit, C_fit)

    # --- 2) Fine grid & fit curve ---
    x_fine = np.linspace(x_data.min(), x_data.max(), 1000)
    y_fine = C_fit + A_fit * johnsonsu.pdf(
        x_fine, a_fit, b_fit, loc=loc_fit, scale=scale_fit
    )

    #print(johnsonsu_pdf(x_fine,a_fit,b_fit,loc_fit,scale_fit,10,12))

    # --- 3) Nominal FWHM ---
    baseline = C_fit
    peak     = y_fine.max()
    half_max = 0.5*(peak + baseline)
    mask     = y_fine >= half_max
    x_left, x_right = x_fine[mask][0], x_fine[mask][-1]
    fwhm_nom = x_right - x_left

    # --- 4) Monte‑Carlo error propagation ---
    try:
        draws = np.random.multivariate_normal(popt, pcov, size=mc_runs)
        fwhm_samples = []
        for a, b, loc, scale, A, C in draws:
            y_mc = C + A * johnsonsu.pdf(
                x_fine, a, b, loc=loc, scale=scale
            )
            hm = 0.5*(y_mc.max() + C)
            m  = y_mc >= hm
            if m.any():
                fwhm_samples.append(x_fine[m][-1] - x_fine[m][0])
        fwhm_err = np.std(fwhm_samples)
    except:
        print('Montecarlo failed...')

    # --- 5) Plot ---
    fig, ax = plt.subplots(figsize=(6, 4))

    # Data with error bars
    ax.plot(x_data, y_data, 'o', color='black', ms=6, label='Data')

    ax.fill_between(
    x_data,
    y_data - y_errors,
    y_data + y_errors,
    color='gray',
    alpha=0.3,
    label='±1σ'
    )

    # Fit line
    ax.plot(
        x_fine, y_fine, '-',
        lw=2, color='C1', label='Johnson SU fit'
    )

    # Half‑max line
    ax.hlines(
        half_max, x_left, x_right,
        ls='--', lw=1.5, color='C2',
        label=f'½ max = {half_max:.2f}'
    )
    ax.vlines(
        [x_left, x_right],
        ymin=baseline, ymax=half_max,
        ls=':', lw=1, color='C2'
    )

    # Grid
    ax.grid(True, ls=':', lw=0.7, alpha=0.8)

    try:
        # Labels & title
        ax.set_xlabel('x', fontsize=12, family='serif')
        ax.set_ylabel('Time', fontsize=12, family='serif')
        ax.set_title('Johnson SU Fit with FWHM ± error', fontsize=14, family='serif')
        ax.tick_params(labelsize=10)
    except:
        print('Error')

    try:
        # Annotation moved lower‑left
        ax.annotate(
            rf"$\mathrm{{FWHM}} = {fwhm_nom:.4f}\,\pm\,{fwhm_err:.4f}$",
            xy=(0.6, 0.055), xycoords='axes fraction',
            fontsize=11, color='darkred',
            ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.3',
                      fc='white', ec='darkred', alpha=0.9)
        )
    except:
        print('Error')

    # Legend
    ax.legend(frameon=False, fontsize=10, loc='upper right')

    plt.tight_layout()
    plt.savefig(save, dpi=300)
    plt.close(fig)

    return popt, pcov, fwhm_nom, fwhm_err


def smooth_times(times, window=3, polyorder=2):
    # https://es.wikipedia.org/wiki/Filtro_de_Savitzky%E2%80%93Golay
    # window must be odd and ≥ polyorder+2
    return savgol_filter(times, window_length=window, polyorder=polyorder)

def gradient_smoothed(times, dx=1.0, window=3, polyorder=1):
    y = smooth_times(times, window, polyorder)
    return np.gradient(y, dx)

def draw(x_data, y_data, p0):

    popt, pcov = curve_fit(johnsonsu_pdf,
                           x_data, y_data,          # weight by errors
                           absolute_sigma=True)
    
    a_fit, b_fit, loc_fit, scale_fit, A_fit, C_fit = popt

    # --- 2) Fine grid & fit curve ---
    x_fine = np.linspace(x_data.min(), x_data.max(), 1000)
    y_fine = C_fit + A_fit * johnsonsu.pdf(
        x_fine, a_fit, b_fit, loc=loc_fit, scale=scale_fit
    )
         
    return x_fine, y_fine,a_fit, b_fit, loc_fit, scale_fit, A_fit, C_fit


