import numpy as np
import dask.array as da
from scipy.sparse import csr_matrix


from classes.voronoi.voronoi_teselation import generateAnimation
import numpy as np
from scipy.sparse import csr_matrix, dok_matrix
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from classes.voronoi.auxiliarfunc import applyOcupation, log_criteria_niter
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
import os
import joblib
from scipy.stats import linregress

from classes.fit.fitting import expFit, fit_best_model, model_dict
from scipy.ndimage import label



class voronoiFire():
    def __init__(self,
                 burningThreshold:float, occuProba:float, voronoi:object, initialFire:int,saveHistoricalPropagation:bool = False) -> None:

        # Extract the object attributes fron the arguments
        self.burningThreshold = burningThreshold
        self.occuProba = occuProba
        self.voronoi = voronoi
        self.initialFire = initialFire

        # Extract useful information
        self.neighbours = voronoi.ridge_points
        self.numPoints = self.voronoi.points.shape[0]

        # Create the neighbours table
        self.neighboursTable = dok_matrix((self.numPoints,self.numPoints))
        #dok = self.neighboursTable.todok()
        for i,j in self.neighbours:
            self.neighboursTable[i,j] = 1
            self.neighboursTable[j,i] = 1
        self.neighboursTable = self.neighboursTable.tocsr()
        
        # Set the initial fire status
        self.status = np.ones(self.numPoints)
        self.createBorder()
        self.initialConfiguration = np.copy(self.status)
        #self.status = applyOcupation(self.status, self.occuProba)
        
        # Space to save historical fire status
        self.historicalFirePropagation = [np.copy(self.status)]
        self.saveHistoricalPropagation = saveHistoricalPropagation
        
        
        # Guardar índices de puntos de borde por posición
        self.left_border = []
        self.right_border = []
        self.top_border = []
        self.bottom_border = []
        epsilon = 0.05  # Margen para considerar un punto como "en el borde"

        for i, (x, y) in enumerate(self.voronoi.points):
            if self.status[i] == 0:  # Solo considerar celdas de borde
                if x < epsilon:
                    self.left_border.append(i)
                elif x > 1 - epsilon:
                    self.right_border.append(i)
                if y < epsilon:
                    self.bottom_border.append(i)
                elif y > 1 - epsilon:
                    self.top_border.append(i)

        # Convertir listas a arrays de numpy para vectorización
        self.left_border = np.array(self.left_border)
        self.right_border = np.array(self.right_border)
        self.top_border = np.array(self.top_border)
        self.bottom_border = np.array(self.bottom_border)

    def propagateFire(self, ps:float, pb:float, percolates_check:bool=False, centered:bool = False):
        
        self.status = applyOcupation(self.status,ps)

        if centered:
            # We find the closet point to the center of the grid
            center = np.array([0.5, 0.5])

            # Calculamos la distancia euclidiana al centro para cada punto
            distances = np.linalg.norm(self.voronoi.points - center, axis=1)

            # Obtenemos el índice del punto más cercano
            closest_point = np.argmin(distances)
            #print(closest_point)
            self.status[closest_point] = 2    
        else: 
            self.status[self.initialFire] = 2


        # Guardar índices de borde
        border_indices = np.where(self.status == 0)[0]
        
        if np.sum(self.status == 2) == 0:
            print('The forest does not have burning trees')
            return 0, False

        else:
            thereIsFire = True
            propagationTime = 0

            while thereIsFire:
                propagationTime += 1
                mask = (self.status == 2).astype(int)

                # Matrix that contains the amount of burning neighbours each tree has

                N = self.neighboursTable.dot(mask)

                # Get the modified Threshold for each tree
                newThreshold = 1-(1-pb)**N

                # Generate aleatory number for each point
                probability = np.random.rand(self.numPoints)

                # find which trees could burn
                couldBurn = (probability < newThreshold)

                # Find those trees that will brun in the next step
                newBurningTrees = (self.status == 1) & couldBurn & (N>0)

                # State burned trees
                self.status[self.status == 2] = 3

                # Set new burning trees
                self.status[newBurningTrees] = 2

                if (self.saveHistoricalPropagation):
                    self.historicalFirePropagation.append(np.copy(self.status))


                thereIsFire = False if np.sum(newBurningTrees) == 0 else True
                
                if percolates_check:
                    # ---------------------
                    # Detectar percolación entre lados opuestos
                    # ---------------------
                    burned_indices = np.where(self.status == 3)[0]

                    left_burned = np.intersect1d(self.left_border, burned_indices)
                    right_burned = np.intersect1d(self.right_border, burned_indices)
                    top_burned = np.intersect1d(self.top_border, burned_indices)
                    bottom_burned = np.intersect1d(self.bottom_border, burned_indices)

                    percolates_horizontal = (len(left_burned) > 0) and (len(right_burned) > 0)
                    percolates_vertical = (len(top_burned) > 0) and (len(bottom_burned) > 0)

                    # El incendio percola si atraviesa completamente en alguna dirección
                    percolates = percolates_horizontal or percolates_vertical

                    return propagationTime, percolates
                
            return propagationTime
    
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++          Add commentMore actions
    
    def propagationtime(self,saveName:str,n:int, m:int):
        """ 
        Method to calculate and plot the propagation time as a funcion of the percolation threshold for voronoi tessellation
        
        args:
            - saveName: str     name to save the plot
            - n: int    How     many different percolationo threshold are to be considered
            - m: int    How     many simulations for each fixed percolation threshold 
            
        returns:
            None        saves the figure on the route graphs/voronoi/saveName
        """

        finalTimes = np.zeros((n,m))
        meanFinaltimes = np.zeros(n)
        meanFinaltimesStd = np.zeros(n)
        P = np.linspace(0,1,n)

        fixed_status = np.copy(self.status)

        for i,p in enumerate(P):

            self.burningThreshold = p
            for j in range(m):
                self.status = np.copy(fixed_status)
                finalTimes[i,j] = self.propagateFire()

            meanFinaltimes[i] = np.mean(finalTimes[i,:])
            meanFinaltimesStd[i] = np.std(finalTimes[i,:])

        # Reduce negative error bars for physical meaning
        Y_err_lower = np.minimum(meanFinaltimes,meanFinaltimesStd)

        plt.errorbar(P, meanFinaltimes, yerr=[Y_err_lower,meanFinaltimesStd], capsize=5, ecolor='red', marker='o', linestyle='None')
        plt.xlabel('$P$')
        plt.ylabel('$t(p)$')
        plt.title(r'Burning time as a function of p\nErrorbar = 1$\sigma$')
        plt.savefig(saveName + '.png')
    
    
 #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++       
    def animate(self,filename, interval = 100):
        self.saveHistoricalPropagation = True
        print('Starting simulation, wait a sec...')
        # Simulate fire
        self.propagateFire(self.occuProba, self.burningThreshold)   #,centered = True

        print('Simulation has finished. Initializing animation...')
        #print(percolates)
        generateAnimation(self.voronoi,
                          filename,
                          self.historicalFirePropagation,
                          interval,
                          p_bond=self.burningThreshold,
                          p_site=self.occuProba)

    def createBorder(self):
        max_length = 10./np.sqrt(self.numPoints)
        for i in range(self.numPoints):
            region_index = self.voronoi.point_region[i]  # Get index for the ith point's region
            region = self.voronoi.regions[region_index]  # Get region by index


            # if region is infinite, set status 0
            if -1 in region:
                self.status[i] = 0
                continue

            # if region is finite, calculate length\perimeter
            polygon = Polygon(self.voronoi.vertices[region])  # create region's polygon
            perimeter = polygon.length  # perimeter of polygon

            # if perimeter is higher than max_length, asign status 0
            if perimeter > max_length:
                self.status[i] = 0


    def fit_percolation_threshold(self,n:int,m:int, 
                             plot:bool=False, 
                             fixed:str = 'bond', 
                             fixed_value:float=1,
                             saveRoute:str='./',
                             exploring_range:list=[0,1],
                             width:float = 0.05):
        """
        """
        results = np.zeros((n,m))
        P = np.linspace(exploring_range[0],exploring_range[1],n)
        fixed_status = np.ones(self.numPoints)

        for i,p in enumerate(P):
            for j in range(m):
                
                self.status = np.copy(fixed_status)
                # If fixed is bond we compute the p site critical value
                
                if fixed == 'bond':
                    t = self.propagateFire(ps=p, pb=fixed_value, centered=True)
                # If not, we calculate the p bond critical value
                else:
                    t = self.propagateFire(ps=fixed_value,pb=p, centered=True)

                results[i,j] = t
                
        # Delta of p
        delta = np.round((exploring_range[1] - exploring_range[0])/n,2)
        
        # Calculate the variance of time propagation for each p
        p_variance = results.var(axis=1)
        
        # Get the p with the higher variance
        fit_pivot = P[ np.argmax(p_variance) ]

        # Run simulations arount the pivot (10 points in the interval)
        fit_data_raw = np.zeros((20,2*m))
        fit_P = np.linspace(fit_pivot - width, fit_pivot + width, 20)

        for i,p in enumerate(fit_P):
            for j in range(2*m):

                self.status = np.copy(fixed_status)
                # If fixed is bond we compute the p site critical value
                if fixed == 'bond':
                    t = self.propagateFire(ps=p, pb=fixed_value)
                # If not, we calculate the p bond critical value
                else:
                    t = self.propagateFire(ps=fixed_value,pb=p)

                fit_data_raw[i,j] = t

        fit_data = fit_data_raw.mean(axis=1)
        fit_data_var = fit_data_raw.var(axis=1)
        
        # Perform fit to variance data
        pc_estimate, model_name, model_params, pc_error = fit_best_model(fit_P, fit_data)

        if plot:
            plt.figure(figsize=(8, 5))
            plt.plot(fit_P, fit_data, 'o', label="Simulated Data", color="blue")
            
            # Curva suave del modelo elegido
            p_smooth = np.linspace(fit_P[0], fit_P[-1], 200)
            model_func = model_dict[model_name][0]
            y_smooth = model_func(p_smooth, *model_params)

            plt.plot(p_smooth, y_smooth, '-', label=f"{model_name.capitalize()} fit", color="red")
            plt.axvline(pc_estimate, color='green', linestyle='--', label=f"Estimated $p_c$ = {pc_estimate:.5f}")
            plt.xlabel("Probability")
            plt.ylabel("Mean propagation time")
            plt.title(f"{model_name.capitalize()} Fit to Estimate $p_c$")
            plt.legend()
            plt.grid(True)
            plt.savefig(saveRoute + 'pc_fit_' + f'{self.numPoints}' + '.png')

        
        print('---------------------------------------------------')
        print('estimated pc:', pc_estimate)
        print('max time prop:', fit_P[ np.argmax(fit_data)] )
        print('max var:', fit_P[ np.argmax(fit_data_var)] )

        #pc_error = np.abs(pc_estimate - fit_P[ np.argmax(fit_data)])
        return pc_estimate, pc_error
                
    def compareBondSite(self,resolution:int,
                        imagePath:str,
                        folder_path:str, file_name:str,
                        propTimeThreshold:int=120, fit:bool = False):
        
        # Verificar si la carpeta existe, si no, crearla
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print(type(file_name))
        file_path = os.path.join(folder_path, file_name)

        # Verificar si el archivo .csv existe
        if not os.path.isfile(file_path):
            # Generar datos de ejemplo y guardarlos en un archivo .csv
            print("Archivo no encontrado. Creando archivo .csv...")
            p_site = np.linspace(0, 1., resolution)  # Valores de 0 a 1 con paso 0.1
            p_bond = np.linspace(0, 1., resolution)  # Valores de 0 a 1 con paso 0.1
            P_site, P_bond = np.meshgrid(p_site, p_bond)

            # Load to the model for personalized niters
            rf_model = joblib.load(folder_path + '3d_regression_model.pkl')

            time = np.zeros(len(p_site)*len(p_bond))  # Ejemplo de datos para z

            count = 0
            for ps in p_site:
                print(ps)
                for pb in p_bond:
                    # Apply criteria for n_iter
                    expected_gradient = rf_model.predict(np.array([[pb,ps]]))
                    n_iter = log_criteria_niter(expected_gradient)


                    times_for_average = np.ones(n_iter, dtype=int)
                    for i in range(n_iter):
                        self.status = np.copy(self.initialConfiguration)
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

        # Configurar ticks manualmente
        ticks = np.arange(0, 1.1, 0.1)  # De 0 a 1 en pasos de 0.1
        ax.set_xticks(np.linspace(0, heatmap_data.shape[1] - 1, len(ticks)))  # Ticks ajustados al tamaño de la matriz
        ax.set_yticks(np.linspace(0, heatmap_data.shape[0] - 1, len(ticks)))
        ax.set_xticklabels([f"{tick:.1f}" for tick in ticks])
        ax.set_yticklabels([f"{tick:.1f}" for tick in ticks])
        ax.invert_yaxis()
        ax.set_aspect(1)

        plt.title(f"Comparative heat map for voronoi tesellation", size=20)
        plt.xlabel(r"$P_{occupancy}$", size=15)
        plt.ylabel(r"$P_{spread}$", size=15)

        if fit:

            # Execute the fit
            function,ps,pb,popt = expFit(data,propTimeThreshold)
            # Plot results
            x = np.linspace(0,1,100)
            x_indices = x * (heatmap_data.shape[1] - 1)  # Scale x values to heatmap indices
            y_indices = function(x,*popt) * (heatmap_data.shape[0] - 1)  # Scale y values to heatmap indices

            ax.plot(x_indices, y_indices,'r-',label='fit: %5.3f exp( - %5.3f p_site) + %5.3f' % tuple(popt), zorder=10)
        #plt.legend()
        plt.savefig(imagePath+'.png', format='png')


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
        
    def percolationThreshold(self,n:int,m:int,  
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
        
        fixed_status = np.copy(self.status)
        
        for i,p in enumerate(P):
            for j in range(m):
                self.status = np.copy(fixed_status)
                # If fixed is bond we compute the p site critical value
                if fixed == 'bond':
                    _ = self.propagateFire(p, fixed_value)
                # If not, we calculate the p bond critical value
                else:
                    _ = self.propagateFire(fixed_value,p)
                percolationResults[i,j] = percolation_check(self.status)
                
        # Delta of p
        delta = np.round(1/n,2)
        
        # Calculate the frequency of percolation for each p
        percolation_frequencies = percolationResults.mean(axis=1)
        
        # Get the percolation threshold
        p_c = np.round(P[percolation_frequencies > 0.5][0],2)

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
        
        print('---------------------------------------')
        print('For fixed ' + fixed + f' {fixed_value}')
        for i in range(n_iter):
            
            
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
        #p_c = np.array([0.52])
        # Step 1: Estimate pc for each fixed value
        p_c = np.zeros(len(fixed_values))
        for i, p in enumerate(fixed_values):
            self.forest = np.copy(initial)
            if method == 'freq':
                p_c[i] = self.percolationThreshold(n=n, m=m, matrix=self.forest, plot=False, fixed=fixed, fixed_value=p)
            else:
                p_c[i] = self.estimate_percolation_threshold(m=m, matrix=self.forest, n_iter=4, fixed=fixed, fixed_value=p)
        # Step 2: Run simulations for p > pc
        tabular_info = np.zeros((n2, len(fixed_values)))
        p_values = np.zeros((n2, len(fixed_values)))

        for j, fixed_value in enumerate(fixed_values):
            #P_plus = np.linspace(p_c[j], p_c[j] * (1 + intervalTol), n2)
            P_plus = np.linspace(p_c[j] + 0.01, p_c[j] + 0.05 ,  n2)
            p_values[:, j] = P_plus - p_c[j]
            average_t_plus = np.zeros(n2)

            for i, p_plus in enumerate(P_plus):
                t_plus = np.zeros(m2)
                k = 0
                while k < m2:
                    self.forest = np.copy(initial)
                    if fixed == 'bond':
                        _ = self.propagateFire(ps=p_plus, pb=fixed_value)
                        # Compute the fraction of site for the giant cluster considering the occupation probability
                        # Only account for percolating clusters
                        if percolation_check(self.forest):
                            t_plus[k] = np.sum(self.forest == 3) / (self.forest.size*p_plus)
                            k += 1
                            
                    else:
                        _ = self.propagateFire(ps=fixed_value, pb=p_plus)
                        # Compute the fraction of site for the giant cluster considering the occupation probability
                        # Only account for percolating clusters
                        if percolation_check(self.forest):
                            t_plus[k] = np.sum(self.forest == 3) / (self.forest.size*fixed_value)
                            k += 1
                        
                        
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

    def gamma_exponent(self,
                         save_route:str, 
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

        # Step 2: Run simulations for p < pc
        
        
        chi = np.zeros((n2, len(fixed_values)))
        p_values = np.zeros((n2, len(fixed_values)))
        
        
        
        
        for j, fixed_value in enumerate(fixed_values):
            
            P_minus = np.linspace(p_c[j] - 0.1, p_c[j] - 0.001 ,  n2)
            p_values[:, j] = np.abs(P_minus - p_c[j])
        
            for i, p_minus in enumerate(P_minus):
                S = np.zeros(m2)   # Store cluster sizes
                self.forest = np.copy(initial)
                
                k = 0
                while k < m2:     
                    if fixed == 'bond':
                        _ = self.propagateFire(ps=p_minus, pb=fixed_value)
                    else:
                        _ = self.propagateFire(ps=fixed_value, pb=p_minus)

                    # Compute the burned sites
                    if not (percolation_check(self.forest)):    # Only count not infinite cluster sizes
                        S[k] = np.sum(self.forest == 3)
                        k += 1
                        
                chi[i,j] = np.mean(S**2) #/ np.mean(S)       # Suceptibility
               
        
        # --- Ajuste lineal log-log ---
        log_p = np.log(p_values)
        log_chi = np.log(chi)
        gamma = np.zeros(len(fixed_values))

        
        # Step 4: Plot
        plt.figure(figsize=(8, 6))
        for i in range(len(fixed_values)):
            
            slope, intercept, *_ = linregress(log_p[:,i], log_chi[:,i])
            #result = linregress(log_p, log_chi)
            gamma[i] = -slope
        
            # Plotting
            plt.plot(log_p, log_chi, 'o', label=f'{fixed} = {fixed_values[i]:.2f}')
            plt.plot(log_p, slope * log_p + intercept,'--', label=f'Ajuste: γ ≈ {gamma[i]:.2f}')
            #plt.errorbar(log_p, log_chi, yerr=chi_err, fmt='o', label='Datos simulados', capsize=4)
        
        
        #print(f"Coeficiente de correlación R = {r_value:.4f}")        

        plt.xlabel(r'$\log(|p - p_c|)$')
        plt.ylabel(r'$\log(\chi)$')
        plt.title('Estimación del exponente crítico γ')
        
        #plt.plot(pp,chi)
        
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_route + 'gamma_exponent_fit.png', dpi=300)
        #plt.show()
    
    def criticalExponent(self,save_route:str, intervalTol:float, n:int,m:int,n2:int,m2:int, fixed:str,fixed_values:list):
        '''
        the critical exponents are saved on a matrix with a shape (1,len(fixed_values),2)
        On the first layer we firn the left critical exponents, and on the sencod, the rigth ones, each column
        is a fixed value.
        args:
        n2: int -> Number of points on the interval to find the critical exponent
        m2: int -> Number of simulations with fixed p_bond and p_site to compute average propagation time
        '''
        # fiexed = 1 is for fixing p_bond and varying p_site
        fixed_status = np.copy(self.status)
        # Critical thresholds
        p_c = np.zeros(len(fixed))
        for i in range(p_c):
            self.status = np.copy(fixed_status)
            p_c[i] = self.percolationThreshold(n=n,m=m, matrix=self.forest,fixed=fixed,fixed_value=fixed_values[i])
        
        # Store time values values for regression
        tabular_info = np.zeros((n2,len(fixed_values)),2)
        # Store p values for regression
        p_values = np.zeros((n2,len(fixed_values)),2)
        
        
        # Compute for each fixed value
        for j,fixed_value in enumerate(fixed_values):
            # Critical exponent from left
            P_minus = np.linspace(p_c[j] * (1-intervalTol), p_c[j], n2)
            p_values[:,j,0] = P_minus
            # Critical exponent from right
            P_plus = np.linspace(p_c[j], p_c[j] * (intervalTol + 1), n2)
            p_values[:,j,1] = P_plus
            # Store average times
            average_t_minus = np.zeros(n2)
            average_t_plus = np.zeros(n2)
            
            for i,(p,p_plus) in enumerate(zip(P_minus,P_plus)):
                t_minus = np.zeros(m2)
                t_plus = np.zeros(m2)
                # Calculate the average propagation time
                
                # For fixed p_bond
                if fixed == 'bond':
                    # For left critical exponent
                    for k in range(m2):
                        self.status = np.copy(fixed_status)
                        t_minus[k] = self.propagateFire(ps=p,pb=fixed_value)

                    average_t_minus[i] = t_minus.mean()
                    
                    # for right critical exponent
                    for k in range(m2):
                        self.status = np.copy(fixed_status)
                        t_plus[k] = self.propagateFire(ps=p_plus,pb=fixed_value)

                    average_t_plus[i] = t_plus.mean()
                    
                    
                # For fixed P_site
                else:
                    # For left critical exponent
                    for k in range(m2):
                        self.status = np.copy(fixed_status)
                        t_minus[k] = self.propagateFire(ps=fixed_value,pb=p)

                    average_t_minus[i] = t_minus.mean()
                    
                    # for right critical exponent
                    for k in range(m2):
                        self.status = np.copy(fixed_status)
                        t_plus[k] = self.propagateFire(ps=fixed_value,pb=p_plus)

                    average_t_plus[i] = t_plus.mean()
                    
                    
            # Store the propagation time for each fixed value for left critical exponent   
            tabular_info[:,j,0] = average_t_minus[:]
            # Store the propagation time for each fixed value for right critical exponent
            tabular_info[:,j,1] = average_t_plus[:]
            
        # Compute logarithm
        log_t_data = np.log(tabular_info)
        log_p_values = np.log(p_values)
        
        # Space to store critical exponents
        critical_exponents = np.zeros((1,len(fixed_values),2))
        
        
        for i in range(len(fixed_values)):
            # Calculate and store left critical exponents
            slope_minus, intercept_minus, r_value_minus, p_value_minus, std_err_minus = linregress(log_t_data[:,i,0], log_p_values[:,i,0])
            critical_exponents[0,i,0] = slope_minus
            
            # Calculate and store rigth critical exponents
            slope_plus, intercept_plus, r_value_plus, p_value_plus, std_err_plus = linregress(log_t_data[:,i,1], log_p_values[:,i,1])
            critical_exponents[0,i,1] = slope_plus
            
        # Reshape data to save as a csv
        left_t_exponents_data = pd.DataFrame(log_t_data[:,:,0])
        rigt_t_exponents_data = pd.DataFrame(log_t_data[:,:,1])
        
        left_p_exponents_data = pd.DataFrame(log_p_values[:,:,0])
        rigt_p_exponents_data = pd.DataFrame(log_p_values[:,:,1])
        
        left_critical_exponents = pd.DataFrame(critical_exponents[:,:,0])
        rigth_critical_exponents = pd.DataFrame(critical_exponents[:,:,1])
        
        left_t_exponents_data.to_csv(save_route + 'left_t_exponents_data_' + {fixed} + '.csv')
        rigt_t_exponents_data.to_csv(save_route + 'rigt_t_exponents_data_' + {fixed} + '.csv')
        left_p_exponents_data.to_csv(save_route + 'left_p_exponents_data_' + {fixed} + '.csv')
        rigt_p_exponents_data.to_csv(save_route + 'rigt_p_exponents_data_' + {fixed} + '.csv')
        
        left_critical_exponents.to_csv(save_route + 'left_critical_exponents_' + {fixed} + '.csv')
        rigth_critical_exponents.to_csv(save_route + 'rigth_critical_exponents_' + {fixed} + '.csv')    
        
def percolation_check(array):
    # Identificar regiones conectadas de árboles quemados, representados por el número 3
    labeled_array, num_features = label(array == 3)
    
    # Obtener las etiquetas presentes en los bordes superior e inferior (percolación vertical)
    first_row_labels = set(labeled_array[0, :])
    last_row_labels = set(labeled_array[-1, :])
    
    # Verificar si alguna etiqueta de la primera fila está en la última fila (percolación vertical)
    vertical_common_labels = first_row_labels.intersection(last_row_labels)
    
    # Obtener las etiquetas presentes en los bordes izquierdo y derecho (percolación horizontal)
    first_col_labels = set(labeled_array[:, 0])
    last_col_labels = set(labeled_array[:, -1])

    # Verificar si alguna etiqueta de la primera columna está en la última columna (percolación horizontal)
    horizontal_common_labels = first_col_labels.intersection(last_col_labels)

    # Si hay etiquetas en común en cualquiera de las direcciones, hay percolación
    return bool(vertical_common_labels - {0}) or bool(horizontal_common_labels - {0})  # Excluir 0 porque no es una región etiquetada