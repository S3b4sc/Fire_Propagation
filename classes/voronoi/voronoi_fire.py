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

from classes.fit.fitting import expFit
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
        
        # Set the initial fire status
        self.status = np.ones(self.numPoints)
        self.createBorder()
        self.initialConfiguration = np.copy(self.status)
        #self.status = applyOcupation(self.status, self.occuProba)
        
        
    
        # Create the neighbours table
        
        self.neighboursTable = dok_matrix((self.numPoints,self.numPoints))
        #dok = self.neighboursTable.todok()
        
        for i,j in self.neighbours:
            self.neighboursTable[i,j] = 1
            self.neighboursTable[j,i] = 1
            
        self.neighboursTable = self.neighboursTable.tocsr()
        
        # Space to save historical fire status
        self.historicalFirePropagation = [np.copy(self.status)]
        self.saveHistoricalPropagation = saveHistoricalPropagation
    
    def propagateFire(self, ps:float, pb:float):
        self.status = applyOcupation(self.status,ps)
        self.status[self.initialFire] = 2
        if np.sum(self.status == 2) == 0:
            print('The forest does not have burning trees')
            
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
            
            return propagationTime
        
    
    
 #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++          
    
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
        _ = self.propagateFire()

        print('Simulation has finished. Initializing animation...')
        generateAnimation(self.voronoi,
                          filename,
                          self.historicalFirePropagation,
                          interval,
                          p_bond=self.burningThreshold,
                          p_site=self.occuProba)
    
    def createBorder(self):
        max_length = 10./np.sqrt(self.numPoints)
        for i in range(self.numPoints):
            region_index = self.voronoi.point_region[i]  # Get index for the i point's region
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
    
    def compareBondSite(self,resolution:int,n_iter:int, imagePath, folder_path, file_name,propTimeThreshold:int=120):
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
        heatmap_data = data.pivot_table(index='P_site', columns='P_bond', values='time')
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(heatmap_data, cmap='viridis', cbar_kws={'label': 'Valor de tiempo'})

        # Configurar ticks manualmente
        ticks = np.arange(0, 1.1, 0.1)  # De 0 a 1 en pasos de 0.1
        ax.set_xticks(np.linspace(0, heatmap_data.shape[1] - 1, len(ticks)))  # Ticks ajustados al tamaño de la matriz
        ax.set_yticks(np.linspace(0, heatmap_data.shape[0] - 1, len(ticks)))
        ax.set_xticklabels([f"{tick:.1f}" for tick in ticks])
        ax.set_yticklabels([f"{tick:.1f}" for tick in ticks])
        ax.invert_yaxis()
        ax.set_aspect(1)

        plt.title("Mapa de calor de los datos (p_site, p_bond, time)")
        plt.xlabel("p_site")
        plt.ylabel("p_bond")
        
        
        # Execute the fit
        function,ps,pb,popt = expFit(data,propTimeThreshold)
        # Plot results
        x = np.linspace(0,1,100)
        x_indices = x * (heatmap_data.shape[1] - 1)  # Scale x values to heatmap indices
        y_indices = function(x,*popt) * (heatmap_data.shape[0] - 1)  # Scale y values to heatmap indices

        ax.plot(x_indices, y_indices,'r-',label='fit: %5.3f exp( - %5.3f p_site) + %5.3f' % tuple(popt), zorder=10)
        plt.legend()
        plt.savefig(imagePath+'.png', format='png')
        
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