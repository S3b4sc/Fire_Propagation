import numpy as np
from classes.regular import teselado
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os
import pandas as pd
import seaborn as sns
import joblib
import scipy.optimize as opt
import scipy.stats as stats

from classes.regular.auxiliarfunc import percolation_check, Apply_occupation_proba, log_criteria_niter
from classes.fit.fitting import expFit

import classes.regular.fire_plus as fire

from numba import njit

zeroArray = np.zeros(1)

class forestFire():
    
    def __init__(self, 
                 burningThreshold:float, occuProba:float,initialForest:np.ndarray,
                 neighbours:list,
                 neighboursBoolTensor: np.ndarray,
                 wind:np.ndarray = zeroArray,
                 topography:np.ndarray = zeroArray,
                 saveHistoricalPropagation:bool = False,
                 ) -> None:
        
        
        self.burningThreshold = burningThreshold
        self.occuProba = occuProba
        self.forest = np.copy(initialForest)
        self.neighbours = neighbours
        self.neighboursBoolTensor = neighboursBoolTensor
        self.wind = wind
        self.topography = topography

        self.forestSize = initialForest.shape
        self.historicalFirePropagation = []
        self.saveHistoricalPropagation = saveHistoricalPropagation
        
    
    def propagateFire(self,ps:float,pb:float):
        # Apply ocuppation probability
        self.forest = Apply_occupation_proba(self.forest,ps)
        
        if np.sum(self.forest == 2) == 0:
            print('The forest does not have burning trees')
        else:
            self.historicalFirePropagation, steps, self.forest = fire.propagate_fire_cpp(self.forest.astype(np.int32), pb, self.neighbours, self.neighboursBoolTensor.astype(np.int32), True)
            
            return steps
        

    #-------------------------------------------------------------------------------------------------
    # Methods to calculate statistic params of many simulations
    #-------------------------------------------------------------------------------------------------
 
    def propagationTime(self,saveRoute:str, n:int,m:int, matrix:np.ndarray):
        '''
         args: 
         - n: amount of values for p to consider in the interval 0 to 1
         - m: amount ot tials for each p
         - matrix: Initial fire matrix
        ''' 
        
        # Define the array to store the final propagation time
        # Each row is a fixed p, the columns are trials
        finalTimes = np.zeros((n,m))
        meanFinaltimes = np.zeros(n)
        meanFinaltimesStd = np.zeros(n)
        P = np.linspace(0,1,n)
        
    
        for i,p in enumerate(P):
            #self.burningThreshold = p
            for j in range(m):
                self.forest = np.copy(matrix)
                finalTimes[i,j] = self.propagateFire(1,p)
            
            meanFinaltimes[i] = np.mean(finalTimes[i,:])
            meanFinaltimesStd[i] = np.std(finalTimes[i,:])
            
        # Reduce negative error bars for physical meaning
        Y_err_lower = np.minimum(meanFinaltimes,meanFinaltimesStd)
            
        plt.errorbar(P, meanFinaltimes, yerr=[Y_err_lower,meanFinaltimesStd], capsize=5, ecolor='red', marker='o', linestyle='None')
        plt.xlabel('$P$')
        plt.ylabel('$t(p)$')
        plt.title(r'Burning time as a function of p\nErrorbar = 1$\sigma$')
        plt.savefig(saveRoute + '.png')
        #plt.show()
        
    def percolationThreshold(self,n:int,m:int, 
                             matrix:np.ndarray, 
                             plot:bool=False, 
                             fixed:str = 'bond', 
                             fixed_value:float=1,
                             saveRoute:str=''):
        '''
         args: 
         - n: amount of values for p to consider in the interval 0 to 1
         - m: amount ot tials for each p
         - matrix: Initial fire matrix
        '''
        percolationResults = np.zeros((n,m))
        P = np.linspace(0,1,n)
        
        for i,p in enumerate(P):
            for j in range(m):
                self.forest = np.copy(matrix)
                # If fixed is bond we compute the p site critical value
                if fixed == 'bond':
                    _ = self.propagateFire(p, fixed_value)
                # If not, we calculate the p bond critical value
                else:
                    _ = self.propagateFire(fixed_value,p)
                percolationResults[i,j] = percolation_check(self.forest)
                
        # Delta of p
        delta = np.round(1/n,2)
        
        # Calculate the frequency of percolation for each p
        percolation_frequencies = percolationResults.mean(axis=1)
        #print(percolation_frequencies > 0.5)
        
        # Get the percolation threshold
        p_c = 0.5 #np.round(P[percolation_frequencies > 0.5][0],2)

        if plot:
            # Plot
            plt.plot(P, percolation_frequencies, marker='o')
            plt.xlabel("$P$")
            plt.ylabel("Percolation Frequency")
            plt.title("Percolation Probability vs. p")
            plt.grid()
            plt.text(0.63, 1.15, f'Percolation threshold: {p_c} +- {delta}', fontsize=10, color="blue")
            plt.savefig(saveRoute + '.png')
        
        return p_c
    
    def estimate_percolation_threshold(self,m, matrix,n_iter, fixed, fixed_value):
        """
        Estima el valor crítico de p utilizando un ajuste con una distribución chi-cuadrado ajustable.

        Args:
        - p_values: lista de valores de p a probar.
        - m: número de simulaciones por cada p.
        - matrix: matriz inicial del incendio.
        - n_iter: Cantidad de iteraciones para convergencia.

        Returns:
        - pc: valor crítico estimado.
        - popt: parámetros ajustados (A, pc, k, C).
        - p_values: valores de p usados en el ajuste.
        - times: tiempos promedio de propagación.
        """
    
        size = 0.1
        probalities = np.arange(0.1,1,size)

        n = len(probalities)
        pcs = np.zeros(n_iter)
        for i in range(n_iter):
            
            print('---------------------------------------')
            print('For fixed ' + fixed + f' {fixed_value}')
            times = np.zeros(n)

            for p_index,p in enumerate(probalities):
                lista = np.zeros(m)
                for s in range(m):
                    self.forest = matrix.copy()
                    if fixed == 'bond':
                        lista[s] = self.propagateFire(p,fixed_value)
                    else:
                        lista[s] = self.propagateFire(fixed_value,p)
                times[p_index] = np.mean(lista)
            
            pivot = probalities[np.argmax(times)]
            
            
            pcs[i] = pivot
            probalities = np.arange(pivot-size*5,pivot+size*5,size)
            
            size = size/10
            n = len(probalities)
        
        Pc = pcs[-1]
        print(Pc)

        return Pc

        
    #-------------------------------------------------------------------------------------------------
    
    def P_inf_criticalExponent(self,
                     save_route: str,
                     intervalTol: float,
                     n: int, m: int,
                     n2: int, m2: int,
                     fixed: str, fixed_values: list,
                     initial: np.ndarray,
                     method: str) -> None:
        '''
        Computes and plots the critical exponent beta just above pc (p > pc).
        Saves CSV files and a plot showing the log-log fit.

        Args:
        - save_route (str): Folder path to save outputs.
        - intervalTol (float): Proportional interval size above pc.
        - n, m: Parameters for estimating pc.
        - n2: Number of p-values above pc.
        - m2: Number of fire propagations per p-value.
        - fixed: "bond" or "site", indicating the fixed parameter.
        - fixed_values: Values of the fixed parameter.
        - initial: Initial forest state.
        - method: "freq" or other method for pc estimation.
        '''

        # Step 1: Estimate pc for each fixed value
        p_c = np.zeros(len(fixed_values))
        for i, p in enumerate(fixed_values):
            self.forest = np.copy(initial)
            if method == 'freq':
                p_c[i] = self.percolationThreshold(n=n, m=m, matrix=self.forest, plot=False, fixed=fixed, fixed_value=p)
            else:
                p_c[i] = self.estimate_percolation_threshold(m=m, matrix=self.forest, n_iter=3, fixed=fixed, fixed_value=p)

        # Step 2: Run simulations for p > pc
        tabular_info = np.zeros((n2, len(fixed_values)))
        p_values = np.zeros((n2, len(fixed_values)))

        for j, fixed_value in enumerate(fixed_values):
            #P_plus = np.linspace(p_c[j], p_c[j] * (1 + intervalTol), n2)
            P_plus = np.linspace(p_c[j] + 0.01, p_c[j] + 0.12 ,  n2)
            p_values[:, j] = P_plus - p_c[j]
            average_t_plus = np.zeros(n2)

            for i, p_plus in enumerate(P_plus):
                t_plus = np.zeros(m2)
                for k in range(m2):
                    self.forest = np.copy(initial)
                    if fixed == 'bond':
                        _ = self.propagateFire(ps=p_plus, pb=fixed_value)
                        # Compute the fraction of site for the giant cluster considering the occupation probability
                        t_plus[k] = np.sum(self.forest == 3) / (self.forest.size*p_plus)
                    else:
                        _ = self.propagateFire(ps=fixed_value, pb=p_plus)
                        # Compute the fraction of site for the giant cluster considering the occupation probability
                        t_plus[k] = np.sum(self.forest == 3) / (self.forest.size*fixed_value)
                        
                average_t_plus[i] = t_plus.mean()

            tabular_info[:, j] = average_t_plus
        
        # Step 3: Log-log transformation
        log_t_data = np.log(tabular_info)
        log_p_values = np.log(p_values)
        critical_exponents = np.zeros(len(fixed_values))

        # Step 4: Plot
        plt.figure(figsize=(8, 6))
        for i in range(len(fixed_values)):
            slope, intercept, *_ = linregress(log_p_values[:, i], log_t_data[:, i])
            critical_exponents[i] = slope

            # Plotting
            plt.plot(log_p_values[:, i], log_t_data[:, i], 'o', label=f'{fixed} = {fixed_values[i]:.2f}')
            plt.plot(log_p_values[:, i],
                     slope * log_p_values[:, i] + intercept,
                     '--', label=f'fit: β = {slope:.3f}')

        plt.xlabel(r'$\log(p - p_c)$')
        plt.ylabel(r'$\log(p_\infty)$')
        plt.title(r'Log-Log Fit to Estimate $\beta$ (Above $p_c$)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_route + 'beta_exponent_fit.png', dpi=300)
        plt.close()

        # Step 5: Save data
        #t_data_df = pd.DataFrame(log_t_data, columns=[f'P: {val}' for val in fixed_values])
        #p_data_df = pd.DataFrame(log_p_values, columns=[f'P: {val}' for val in fixed_values])
        #exponents_df = pd.DataFrame([critical_exponents], columns=[f'β (P: {val})' for val in fixed_values])
#
        #t_data_df.to_csv(save_route + 'log_t_exponents_above_pc.csv')
        #p_data_df.to_csv(save_route + 'log_p_values_above_pc.csv')
        #exponents_df.to_csv(save_route + 'critical_beta_exponents.csv')



    def criticalExponent_cluster(self,
                         save_route:str, 
                         intervalTol:float, 
                         n:int,m:int,n2:int,m2:int,
                         fixed:str,fixed_values:list,
                         initial:np.ndarray,
                         method:str)->None:
        '''
        The critical exponents are saved on a matrix with a shape (1,len(fixed_values),2)
        On the first layer we firn the left critical exponents, and on the sencod, the rigth ones, each column
        is a fixed value.
        args:
        n2: int -> Number of points on the interval to find the critical exponent
        m2: int -> Number of simulations with fixed p_bond and p_site to compute average propagation time
        '''
        # fiexed = 1 is for fixing p_bond and varying p_site
        
        # Step 1: Estimate pc for each fixed value
        p_c = np.zeros(len(fixed_values))
        
        for i, p in enumerate(fixed_values):
            self.forest = np.copy(initial)
            if method == 'freq':
                p_c[i] = self.percolationThreshold(n=n, m=m, matrix=self.forest, plot=False, fixed=fixed, fixed_value=p)
            else:
                p_c[i] = self.estimate_percolation_threshold(m=m, matrix=self.forest, n_iter=3, fixed=fixed, fixed_value=p)

        # Step 2: Run simulations for p > pc
        chi = np.zeros((n2, len(fixed_values)))
        p_values = np.zeros((n2, len(fixed_values)))
        chi_err = np.zeros((n2,len(fixed_values)))
        
        
        
        for j, fixed_value in enumerate(fixed_values):
            P_plus = np.linspace(p_c[j] - 0.06, p_c[j] - 0.01 ,  n2)
            p_values[:, j] = P_plus - p_c[j]
            average_t_plus = np.zeros(n2)
        
            for i, p_plus in enumerate(P_plus):
                hist = np.zeros(m2)
                for k in range(m2):
                    self.forest = np.copy(initial)
                    if fixed == 'bond':
                        _ = self.propagateFire(ps=p_plus, pb=fixed_value)
                        # Compute the burned sites
                        burned = np.sum(self.forest == 3)
                        hist[j] = burned
                    else:
                        _ = self.propagateFire(ps=fixed_value, pb=p_plus)
                        # Compute the burned sites
                        burned = np.sum(self.forest == 3)
                        hist[j] = burned
                        
                # Susceptibilidad ≈ varianza / N (asumiendo sistema finito)
                chi[i] = np.var(hist) / (initial.size)
                #chi[i] = np.mean(hist)
                chi_err[i] = np.std(hist) / (np.sqrt(m2) * initial.size)  # Error estándar de la varianza

        for i in range(n2):
            hist = np.zeros(m2)
            for j in range(m2):
                self.forest = initial.copy()
                _ = self.propagateFire(ps=1, pb=pp[i])
                burned = np.sum(self.forest == 3)
                hist[j] = burned

            # Susceptibilidad ≈ varianza / N (asumiendo sistema finito)
            chi[i] = np.var(hist) / (initial.size)
            #chi[i] = np.mean(hist)
            chi_err[i] = np.std(hist) / (np.sqrt(m_reps) * initial.size)  # Error estándar de la varianza

        # --- Ajuste lineal log-log ---
        log_p = np.log(np.abs(pp - Pc))
        log_chi = np.log(chi)

        # Filtramos puntos válidos (sin log(0) ni chi <= 0)
        valid = (chi > 0)
        log_p = log_p[valid]
        log_chi = log_chi[valid]
        chi_err_plot = chi_err[valid]

        slope, intercept, r_value, _, _ = linregress(log_p, log_chi)
        gamma = -slope

        print(f"Exponente crítico gamma estimado: {gamma:.4f}")
        print(f"Coeficiente de correlación R = {r_value:.4f}")

        # --- Gráfica ---
        plt.figure(figsize=(7, 5))
        plt.errorbar(log_p, log_chi, yerr=chi_err_plot/chi[valid], fmt='o', label='Datos simulados', capsize=4)
        plt.plot(log_p, intercept + slope * log_p, '-', label=f'Ajuste: γ ≈ {gamma:.2f}')
        plt.xlabel(r'$\log(|p - p_c|)$')
        plt.ylabel(r'$\log(\chi)$')
        plt.title('Estimación del exponente crítico γ')
        
        #plt.plot(pp,chi)
        
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('./susceptibility_fit.png')
        plt.show()
    
        
        
        
        
        
        
        

        for j, fixed_value in enumerate(fixed_values):
            #P_plus = np.linspace(p_c[j], p_c[j] * (1 + intervalTol), n2)
            P_plus = np.linspace(p_c[j] + 0.01, p_c[j] + 0.12 ,  n2)
            p_values[:, j] = P_plus - p_c[j]
            average_t_plus = np.zeros(n2)

            for i, p_plus in enumerate(P_plus):
                t_plus = np.zeros(m2)
                for k in range(m2):
                    self.forest = np.copy(initial)
                    if fixed == 'bond':
                        _ = self.propagateFire(ps=p_plus, pb=fixed_value)
                        # Compute the fraction of site for the giant cluster considering the occupation probability
                        t_plus[k] = np.sum(self.forest == 3) / (self.forest.size*p_plus)
                    else:
                        _ = self.propagateFire(ps=fixed_value, pb=p_plus)
                        # Compute the fraction of site for the giant cluster considering the occupation probability
                        t_plus[k] = np.sum(self.forest == 3) / (self.forest.size*fixed_value)
                        
                average_t_plus[i] = t_plus.mean()

            tabular_info[:, j] = average_t_plus
        
        # Step 3: Log-log transformation
        log_t_data = np.log(tabular_info)
        log_p_values = np.log(p_values)
        critical_exponents = np.zeros(len(fixed_values))

        # Step 4: Plot
        plt.figure(figsize=(8, 6))
        for i in range(len(fixed_values)):
            slope, intercept, *_ = linregress(log_p_values[:, i], log_t_data[:, i])
            critical_exponents[i] = slope

            # Plotting
            plt.plot(log_p_values[:, i], log_t_data[:, i], 'o', label=f'{fixed} = {fixed_values[i]:.2f}')
            plt.plot(log_p_values[:, i],
                     slope * log_p_values[:, i] + intercept,
                     '--', label=f'fit: β = {slope:.3f}')

        plt.xlabel(r'$\log(p - p_c)$')
        plt.ylabel(r'$\log(p_\infty)$')
        plt.title(r'Log-Log Fit to Estimate $\beta$ (Above $p_c$)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_route + 'beta_exponent_fit.png', dpi=300)
        plt.close()
        
        
        
            
        
    def compareBondSite(self,resolution:int, imagePath,
                        folder_path, file_name, matrix,
                        tesellation_type:str,
                        propTimeThreshold:int=120):
        # Verificar si la carpeta existe, si no, crearla
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_path = os.path.join(folder_path, file_name)

        # Verificar si el archivo .csv existe
        if not os.path.isfile(file_path):
            # Generar datos de ejemplo y guardarlos en un archivo .csv
            print("Archivo no encontrado. Creando archivo .csv...")
            p_site = np.linspace(0, 1., resolution)  # Valores de 0 a 1 con paso 0.1
            p_bond = np.linspace(0, 1., resolution)  # Valores de 0 a 1 con paso 0.1
            P_bond, P_site = np.meshgrid(p_bond, p_site)

            # Load to the model for personalized niters
            rf_model = joblib.load(folder_path  + '3d_regression_model.pkl')
            
            time = np.zeros(len(p_site)*len(p_bond))  # Ejemplo de datos para z
            
            count = 0
            for ps in p_site:
                for pb in p_bond:
                    
                    # Apply criteria for n_iter
                    expected_gradient = rf_model.predict(np.array([[ps,pb]]))
                    n_iter = log_criteria_niter(expected_gradient)
                    
                    times_for_average = np.ones(n_iter, dtype=int)
                    for i in range(n_iter):
                        self.forest = np.copy(matrix)
                        times_for_average[i] = self.propagateFire(ps,pb)
                    time[count] = np.mean(times_for_average)
                    count += 1
                print(ps)

            data = pd.DataFrame({
                'P_site': P_site.flatten(),
                'P_bond': P_bond.flatten(),
                'time': time
            })
            data.to_csv(file_path, index=False)
        else:
            print("Archivo .csv encontrado.")

        # Leer el archivo .csv
        data = pd.read_csv(file_path)

        # Crear un mapa de calor
        print("Generando mapa de calor...")
        heatmap_data = data.pivot_table(index='P_bond', columns='P_site', values='time')
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(heatmap_data, cmap='viridis', cbar_kws={'label': '\nTime (a.u)'})

        # Customize the size of the colorbar label
        cbar = ax.collections[0].colorbar
        cbar.set_label('\nTime (a.u)', fontsize=15)

        # Configurar ticks manualmente
        ticks = np.arange(0, 1.1, 0.1)  # De 0 a 1 en pasos de 0.1
        ax.set_xticks(np.linspace(0, heatmap_data.shape[1] - 1, len(ticks)))  # Ticks ajustados al tamaño de la matriz
        ax.set_yticks(np.linspace(0, heatmap_data.shape[0] - 1, len(ticks)))
        ax.set_xticklabels([f"{tick:.1f}" for tick in ticks])
        ax.set_yticklabels([f"{tick:.1f}" for tick in ticks])
        ax.invert_yaxis()
        ax.set_aspect(1)
        
        # Execute the fit
        function,ps,pb,popt = expFit(data,propTimeThreshold)
        # Plot results
        x = np.linspace(0,1,100)
        x_indices = x * (heatmap_data.shape[1] - 1)  # Scale x values to heatmap indices
        y_indices = function(x,*popt) * (heatmap_data.shape[0] - 1)  # Scale y values to heatmap indices

        ax.plot(x_indices, y_indices,'r-',label='fit: $%5.1f \\, exp( - %5.1f  P_{occupancy}) + %5.1f$' % tuple(popt), zorder=10)

        plt.title(f"Comparative heat map for {tesellation_type} tesellation", size=20)
        plt.xlabel(r"$P_{occupancy}$", size=15)
        plt.ylabel(r"$P_{spread}$", size=15)
        plt.legend(loc='upper left', fontsize=13)
        plt.tight_layout()
        plt.savefig(imagePath+'.png', format='png')
        #plt.show()

        # Crear las mallas para las coordenadas X, Y, y los valores Z
        X, Y = np.meshgrid(heatmap_data.columns.astype(float), heatmap_data.index.astype(float))
        Z = heatmap_data.values

        #------------------------------------------------------------------------------------------
        # Crear la figura y el gráfico 3D
        fig2 = plt.figure(figsize=(12, 8))
        ax2 = fig2.add_subplot(111, projection='3d')

        # Crear el gráfico de superficie
        surface = ax2.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k')

        # Añadir barra de color
        cbar = fig2.colorbar(surface, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label('Valor de tiempo')

        # Etiquetas y título
        ax2.set_title("Gráfico 3D de los datos (P_site, P_bond, time)")
        ax2.set_xlabel("P_bond")
        ax2.set_ylabel("P_site")
        ax2.set_zlabel("time")

        # Rotar para una mejor vista inicial
        ax2.view_init(elev=30, azim=-30)
        
        
        # Guardar la imagen
        plt.savefig(imagePath + '_3D' +'.png', format='png')
        #plt.show()
#=============================================================================================================================================
class squareForest(forestFire):
    """
    This is a subclass of the general class forestFire, but specifily designed for simulate
    the propagation in a central square distribution of trees
    """
    def __init__(self,
                 burningThreshold:float,
                 occuProba:float,
                 initialForest:np.ndarray,
                 wind:np.ndarray = zeroArray,
                 topography:np.ndarray = zeroArray,
                 saveHistoricalPropagation:bool = False):
        
        neighboursBoolTensor = np.ones((4,*initialForest.shape), dtype=bool)
        neighbours = [(-1,0),(1,0),(0,1),(0,-1)]
        super().__init__(burningThreshold, occuProba, initialForest, neighbours, neighboursBoolTensor, wind, topography, saveHistoricalPropagation)
    
    def animate(self, fileName, interval=100):
        
        if (self.saveHistoricalPropagation):
            
            print('Starting simulation, wait a sec...')
            # Simulate fire
            _ = self.propagateFire(self.occuProba,self.burningThreshold)

            print('Simulation has finished. Initializing animation...')
            teselado.squareAnimationPlot(fileName,
                                         self.historicalFirePropagation,
                                         interval,
                                         p_bond=self.burningThreshold,
                                         p_site=self.occuProba)
            print('Done.')
        else:
            print('Historical data not found.')
    
    def plot(self, fileName):
        return
    
    
#=============================================================================================================================================
class heaxgonalForest(forestFire):
    """
    This is a subclass of the general class forestFire, but specifily designed for simulate
    the propagation in a central hexagonal distribution of trees
    """
    def __init__(self,
                 burningThreshold:float,
                 occuProba:float,
                 initialForest:np.ndarray,
                 wind:np.ndarray = zeroArray,
                 topography:np.ndarray = zeroArray,
                 saveHistoricalPropagation:bool = False):
        
        
        rows,columns = initialForest.shape
        neighboursBoolTensor = hexagonalNeighboursBooleanTensor(columns,rows)
        neighbours = [(0,1),(0,-1),(-1,0),(1,0),(-1,1),(-1,1),(-1,-1),(-1,-1)]
        super().__init__(burningThreshold, occuProba,initialForest, neighbours, neighboursBoolTensor, wind, topography, saveHistoricalPropagation)
    
    def animate(self, fileName, interval=100):
        if (self.saveHistoricalPropagation):
            
            print('Starting simulation, wait a sec...')
            # Simulate fire
            _ = self.propagateFire(self.occuProba,self.burningThreshold)
            
            print('Simulation has finished. Initializing animation...')
            teselado.hexagonalAnimationPlot(filename=fileName,
                                            historical= self.historicalFirePropagation,
                                            interval=interval,
                                            size=self.forestSize,
                                            p_bond=self.burningThreshold,
                                            p_site=self.occuProba)
            print('Done.')
        else:
            print('Historical data not found.')
    
#=============================================================================================================================================
class triangularForest(forestFire):
    """
    This is a subclass of the general class forestFire, but specifily designed for simulate
    the propagation in a central triangular distribution of trees
    """
    def __init__(self, burningThreshold:float,occuProba:float, initialForest:np.ndarray, wind:np.ndarray = zeroArray, topography:np.ndarray = zeroArray, saveHistoricalPropagation:bool = False):
        rows,columns = initialForest.shape
        neighboursBoolTensor = triangularNeighboursBooleanTensor(columns,rows)
        neighbours = [(0,1),(0,-1),(1,0),(-1,0)]
        super().__init__(burningThreshold, occuProba,initialForest, neighbours, neighboursBoolTensor, wind, topography, saveHistoricalPropagation)
    
    def animate(self, fileName, interval=100):
        if (self.saveHistoricalPropagation):
            
            print('Starting simulation, wait a sec...')
            # Simulate fire
            _ = self.propagateFire(self.occuProba,self.burningThreshold)
            
            print('Simulation has finished. Initializing animation...')
            teselado.triangularAnimationPlot(filename=fileName,
                                            historical= self.historicalFirePropagation,
                                            interval=interval,
                                            size=self.forestSize,
                                            p_bond=self.burningThreshold,
                                            p_site=self.occuProba)
            print('Done.')
        else:
            print('Historical data not found.')

#=======================================================================================
# Maybe more forest types, boronoy, etc
#=======================================================================================

#---------------------------------------------------------------------------------------
# Auxiliar functions to change probability depending rounding conditions
#---------------------------------------------------------------------------------------
def windContribution(x):
    return x

def topographyContribution(x):
    return x

def hexagonalNeighboursBooleanTensor(columns,rows):
    """
    This function compute the boolean neighbours tensor for an hexagonal forest
    of size (y,x)
    """
    booleanTensor = np.ones((8,rows,columns), dtype=bool)

    evenColumns = np.zeros((rows,columns), dtype=bool)
    evenColumns[:, ::2] = True

    oddColumns = np.zeros((rows,columns), dtype=bool)
    oddColumns[:, 1::2] = True

    booleanTensor[4] = booleanTensor[5] = evenColumns
    booleanTensor[6] = booleanTensor[7] = oddColumns
    return booleanTensor

def triangularNeighboursBooleanTensor(columns,rows):
    """
    This function compute the boolean neighbours tensor for an hexagonal forest
    of size (y,x)
    """
    booleanTensor = np.ones((4,rows,columns), dtype=bool)

    evenColumns = np.zeros((rows,columns), dtype=bool)
    evenColumns[:, ::2] = True
    for i in range(1, rows, 2):  # Fila impar, comenzando desde 1
        evenColumns[i] = np.roll(evenColumns[i], shift=1)

    oddColumns = ~evenColumns

    booleanTensor[2] = evenColumns
    booleanTensor[3] = oddColumns
    return booleanTensor




@njit
def propagate_step(forest, neighboursBoolTensor, pb):
    height, width = forest.shape
    new_forest = forest.copy()
    probabilityMatrixForest = np.random.rand(height, width)

    burningNeighbours = np.zeros((height, width), dtype=np.int32)

    for i in range(height):
        for j in range(width):
            for k in range(8):
                if neighboursBoolTensor[k, i, j] and forest[i, j] != 0:
                    ni = i + (k // 3) - 1
                    nj = j + (k % 3) - 1
                    if 0 <= ni < height and 0 <= nj < width:
                        if forest[ni, nj] == 2:
                            burningNeighbours[i, j] += 1

    hasNewBurning = False
    for i in range(height):
        for j in range(width):
            if forest[i, j] == 1 and burningNeighbours[i, j] > 0:
                prob = 1.0 - (1.0 - probabilityMatrixForest[i, j]) ** (1.0 / burningNeighbours[i, j])
                if prob <= pb:
                    new_forest[i, j] = 2
                    hasNewBurning = True
            elif forest[i, j] == 2:
                new_forest[i, j] = 3

    return new_forest, hasNewBurning