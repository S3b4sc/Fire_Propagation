import numpy as np
from classes.regular import teselado
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os
import pandas as pd
import seaborn as sns
import joblib

from classes.regular.auxiliarfunc import percolation_check, Apply_occupation_proba, log_criteria_niter
from classes.fit.fitting import expFit


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
        self.historicalFirePropagation = [np.copy(initialForest)]
        self.saveHistoricalPropagation = saveHistoricalPropagation
        
    
    def propagateFire(self,ps:float,pb:float):
        # Apply ocuppation probability
        self.forest = Apply_occupation_proba(self.forest,ps)
        
        if np.sum(self.forest == 2) == 0:
            print('The forest does not have burning trees')
        else:
            
            thereIsFire = True
            propagationTime = 0
            while thereIsFire:
                # Record time propagation in terms of iterations
                propagationTime += 1
                
                neighboursTensor = self.createNeighbourTensor()
                couldPropagate = neighboursTensor == 2
                amoungOfBurningNeighbours = np.sum(couldPropagate, axis=0)
                cellsToEvaluate = amoungOfBurningNeighbours > 0
                burningTrees = (self.forest == 2)

                probabilityMatrixForest = np.random.rand(*self.forestSize)
                probabilityMatrixForest[cellsToEvaluate] = 1. - (1. - probabilityMatrixForest[cellsToEvaluate]) ** (1/amoungOfBurningNeighbours[cellsToEvaluate])

                #-----------------------------------------------------------------------------------------------------
                # Here could appear a function to modificate probabilityMatrixForest depending of wind and topography
                #-----------------------------------------------------------------------------------------------------

                couldBurn = (probabilityMatrixForest <= pb)

                newBurningTrees = (self.forest == 1) & couldBurn & np.logical_or.reduce(couldPropagate,axis=0)

                # Update forest matrix for the next step
                self.forest[burningTrees] = 3
                self.forest[newBurningTrees] = 2

                # Only save historical data to plot if required
                if (self.saveHistoricalPropagation):
                    self.historicalFirePropagation.append(np.copy(self.forest))
                
        
                thereIsFire = False if np.sum(newBurningTrees) == 0 else True
            
            #print('Fire extinguished')
            return propagationTime
        
        
    def createNeighbourTensor(self):
        neighborhoodSize = len(self.neighbours)
        tensor = np.zeros((neighborhoodSize, *self.forestSize))

        for i, neigh in enumerate(self.neighbours):
            x,y = neigh
            tensor[i] = np.roll(self.forest, (-x,y), axis=(1,0))
            if x:
                if x == 1.:
                    tensor[i, : , -1 ] = 0
                elif x == -1.:
                    tensor[i, : , 0 ] = 0
                else:
                    continue # Maybe Another condition and method for second neighbours and more
            if y:
                if y == 1.:
                    tensor[i, 0 , : ] = 0
                elif y == -1.:
                    tensor[i, -1 , : ] = 0
                else:
                    continue # Maybe Another condition and method for second neighbours and more
            else:
                continue
        tensor = tensor * self.neighboursBoolTensor
        return tensor

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
        #plt.savefig(saveRoute + '.png')
        plt.show()
        
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
    #-------------------------------------------------------------------------------------------------
    
    def criticalExponent(self,save_route:str, intervalTol:float, n:int,m:int,n2:int,m2:int, fixed:str,fixed_values:list,initial:np.ndarray):
        '''
        the critical exponents are saved on a matrix with a shape (1,len(fixed_values),2)
        On the first layer we firn the left critical exponents, and on the sencod, the rigth ones, each column
        is a fixed value.
        args:
        n2: int -> Number of points on the interval to find the critical exponent
        m2: int -> Number of simulations with fixed p_bond and p_site to compute average propagation time
        '''
        # fiexed = 1 is for fixing p_bond and varying p_site
        
        # Critical thresholds
        p_c = np.zeros(len(fixed))
        for i in range(p_c):
            self.forest = np.copy(initial)
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
                        self.forest = np.copy(initial)
                        t_minus[k] = self.propagateFire(ps=p,pb=fixed_value)

                    average_t_minus[i] = t_minus.mean()
                    
                    # for right critical exponent
                    for k in range(m2):
                        self.forest = np.copy(initial)
                        t_plus[k] = self.propagateFire(ps=p_plus,pb=fixed_value)

                    average_t_plus[i] = t_plus.mean()
                    
                    
                # For fixed P_site
                else:
                    # For left critical exponent
                    for k in range(m2):
                        self.forest = np.copy(initial)
                        t_minus[k] = self.propagateFire(ps=fixed_value,pb=p)

                    average_t_minus[i] = t_minus.mean()
                    
                    # for right critical exponent
                    for k in range(m2):
                        self.forest = np.copy(initial)
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
    
        
    def compareBondSite(self,resolution:int,n_iter:int, imagePath, folder_path, file_name, matrix,propTimeThreshold:int=120):
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
            rf_model = joblib.load(folder_path  + '3d_regression_model.pkl')
            
            time = np.zeros(len(p_site)*len(p_bond))  # Ejemplo de datos para z
            
            count = 0
            for ps in p_site:
                for pb in p_bond:
                    
                    # Apply criteria for n_iter
                    expected_gradient = rf_model.predict(np.array([[pb,ps]]))
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
        
        # Execute the fit
        function,ps,pb,popt = expFit(data,propTimeThreshold)
        # Plot results
        x = np.linspace(0,1,100)
        x_indices = x * (heatmap_data.shape[1] - 1)  # Scale x values to heatmap indices
        y_indices = function(x,*popt) * (heatmap_data.shape[0] - 1)  # Scale y values to heatmap indices

        ax.plot(x_indices, y_indices,'r-',label='fit: %5.3f exp( - %5.3f p_site) + %5.3f' % tuple(popt), zorder=10)

        plt.title("Mapa de calor de los datos (p_site, p_bond, time)")
        plt.xlabel("p_site")
        plt.ylabel("p_bond")
        plt.legend()
        plt.savefig(imagePath+'.png', format='png')
        #plt.show()

        # Crear las mallas para las coordenadas X, Y, y los valores Z
        X, Y = np.meshgrid(heatmap_data.columns.astype(float), heatmap_data.index.astype(float))
        Z = heatmap_data.values

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