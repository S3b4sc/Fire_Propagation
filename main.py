from classes.regular import simulation
from classes.voronoi import voronoi_fire
from scripts.utils import infinite_pc

from classes.fit import fitting

from scripts.menu import menu
import numpy as np
from scipy.spatial import Voronoi

from scripts.routes import routes_dict, data_route
import matplotlib.pyplot as plt
import os

import time

if __name__ == '__main__':
    usrChoice = menu()
    matrix = np.ones((70,70))
    matrix[35,35] = 2
    #os.environ["QT_QPA_PLATFORM"] = "wayland"
  
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
    if usrChoice == 1:
        try:
            tessellation = int(input("Choose one: \n1   Squared\n2   Triangular\n3   Hexagonal\n4   Voronoi\n"))
        except:
            print('Not a valid option.')
            
            
        if tessellation == 1:
            
            name = 'squaredAnimation_test'
            route = routes_dict['squared'] +  name
            forest = simulation.squareForest(burningThreshold=0.7,occuProba=1 ,initialForest=matrix, saveHistoricalPropagation=True)
            
            forest.animate(route)    
            #start = time.time()
            #steps = forest.propagateFire(ps=1,pb=0.55)
            #print("Execution time:", time.time() - start)
            #print(steps)
            
        elif tessellation == 2:
            
            name = 'triangularAnimation'
            route = routes_dict['triangular'] + name
            forest = simulation.triangularForest(burningThreshold=0.7,occuProba=0.95 ,initialForest=matrix, saveHistoricalPropagation=True)
            forest.animate(route)
         
        
        elif tessellation == 3:
            
            name = 'hexagonalAnimation'
            route = routes_dict['hexagon'] + name    
            forest = simulation.heaxgonalForest(burningThreshold=0.95,occuProba=0.95 ,initialForest=matrix, saveHistoricalPropagation=True)
            forest.animate(route)
            
        elif tessellation == 4:
            # Create Voronoi diagram
            #np.random.seed(23) 
            nPoints = 25*25
            points = np.random.rand(nPoints, 2)
            vor = Voronoi(points)
            
            name = 'voronoiAnimation'
            route = routes_dict['voronoi'] + name

            voronoi = voronoi_fire.voronoiFire(0.5,0.95,vor,1)
            voronoi.animate(route)
            
        else:
            print('That is not an option, try again.')
            
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    elif usrChoice == 2:
        
        n = 30    # Amount of values to consider for p
        m = 20      # Amount of trials per p 
        
        try:
            tessellation = int(input("Choose one: \n1   Squared\n2   Triangular\n3   Hexagonal\n4   Voronoi\n"))
        except:
            print('Not a valid option.')
            
        
        if tessellation == 1:
            
            name = 'SquaredFinalTimes'
            route = routes_dict['squared'] +  name
            forest = simulation.squareForest(burningThreshold=0.95,occuProba=0.95 ,initialForest=matrix, saveHistoricalPropagation=True)
            forest.propagationTime(route,n,m, matrix)
            
            
        elif tessellation == 2:
            
            name = 'TriangularFinalTimes'
            route = routes_dict['triangular'] + name 
            forest = simulation.triangularForest(burningThreshold=0.55, occuProba=1 ,initialForest=matrix)
            forest.propagationTime(route,n,m, matrix)
            
        elif tessellation == 3:
            
            name = 'hexagonFinalTimes'
            route = routes_dict['hexagon'] + name
            forest = simulation.heaxgonalForest(burningThreshold=0.55, occuProba=1 ,initialForest=matrix)
            forest.propagationTime(route,n,m, matrix)
        
        elif tessellation == 4:
            # Create Voronoi diagram
            nPoints = 10000
            points = np.random.rand(nPoints, 2)
            vor = Voronoi(points)
            
            name = 'voronoiFinalTimes'
            route = routes_dict['voronoi'] + name

            voronoi = voronoi_fire.voronoiFire(0.4,0.5,vor,1,)
            voronoi.propagationtime(saveName=route,n=80,m=100)
            
        else:
            print('That is not an option, try again.')
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    elif usrChoice == 3:
        n = 50    # Amount of values to consider for p
        m = 80     # Amount of trials per p        
        n_iter = 4
        
        try:
            tessellation = int(input("Choose one: \n1   Squared\n2  Triangular\n3    Hexagonal\n4   Voronoi\n"))
        except:
            print('Not a valid option.')    
        
        if tessellation == 1:
            
            name = 'SquaredPercolationThreshold'
            route = routes_dict['squared'] +  name
            forest = simulation.squareForest(burningThreshold=0.95,occuProba=0.95 ,initialForest=matrix, saveHistoricalPropagation=True)
            #p_c = forest.percolationThreshold(n,m,matrix,True,"site", fixed_value=1,saveRoute=route)
            
            #p_c = forest.estimate_percolation_threshold(m=m, matrix=matrix, n_iter=n_iter,fixed='site', fixed_value=1)
            p_c = forest.fit_percolation_threshold(n,m,matrix,True,'site',exploring_range=[0.4,0.6])
           
            #print("The percolation threshold is: ",p_c)
            
            
        elif tessellation == 2:
            
            name = 'TriangularPercolationThreshold'
            route = routes_dict['triangular'] + name 
            forest = simulation.triangularForest(burningThreshold=0.55, occuProba=1 ,initialForest=matrix)
            #p_c = forest.percolationThreshold(n,m,matrix,True,"site",fixed_value=1,saveRoute=route)
            p_c = forest.estimate_percolation_threshold(m=m, matrix=matrix, n_iter=n_iter,fixed='site', fixed_value=1)
            
            print("The percolation threshold is: ",p_c)
            
        elif tessellation == 3:
            
            name = 'hexagonalPercolationThreshold'
            route = routes_dict['hexagon'] + name
            forest = simulation.heaxgonalForest(burningThreshold=0.55, occuProba=1 ,initialForest=matrix)
            #p_c = forest.percolationThreshold(n,m,matrix,True,"site", saveRoute=route)
            #p_c = forest.estimate_percolation_threshold(m=m, matrix=matrix, n_iter=n_iter,fixed='site', fixed_value=1)
            #print("The percolation threshold is: ",p_c)
            p_c = forest.fit_percolation_threshold(n,m,matrix,True,'site',exploring_range=[0.4,0.6], saveRoute='./graphs/hexagonal/')
        
        elif tessellation == 4:
            name = 'voronoiPercolationThreshold'
            route = routes_dict['voronoi'] + name

            #np.random.seed(23) 
            nPoints = 10000
            points = np.random.rand(nPoints, 2)
            vor = Voronoi(points)


            forest = voronoi_fire.voronoiFire(burningThreshold=0.55, occuProba=1, voronoi=vor , initialFire=1)
            #p_c = forest.percolationThreshold(n,m,matrix,True,"site", saveRoute=route)
            #p_c = forest.estimate_percolation_threshold(m=m, matrix=matrix, n_iter=n_iter,fixed='site', fixed_value=1)
            #print("The percolation threshold is: ",p_c)
            p_c = forest.fit_percolation_threshold(n,m,True,'site',exploring_range=[0.3,0.5])
        
        else:
            print('That is not an option, try again.')
        
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    elif usrChoice == 4:
        n = 20      # Amount of values to consider for p in the range (0,1) to find p_c
        m = 25      # Amount of trialas per each p to find p_c
        
        n2 = 40      # Amount of values to consider for p 
        m2 = 30     # 
        
        epsilon = 0.002*6
        delta = 0.002
        
        try:
            tessellation = int(input("Choose one: \n1   Squared\n2  Triangular\n3    Hexagonal\n4   Voronoi\n"))
        except:
            print('Not a valid option.')
            
        if tessellation == 1:
            saveRoute = './data/squared/'
            forest = simulation.squareForest(burningThreshold=0.55,occuProba=1. , initialForest=matrix)
            criticalExponent = forest.P_inf_criticalExponent(save_route=saveRoute, intervalTol=1e-4,
                                                       n=n,m=m, n2=n2,m2=m2,fixed='site',
                                                       fixed_values=[1],initial=matrix,
                                                       method='pivot')
            
        elif tessellation == 2:
            saveRoute = './data/triangular/'
            forest = simulation.triangularForest(burningThreshold=0.55,occuProba=1. , initialForest=matrix)
            #criticalExponent = forest.criticalExponent(saveRoute,epsilon,delta,n,m,n2,m2,matrix)
            criticalExponent = forest.P_inf_criticalExponent(save_route=saveRoute, intervalTol=1e-4,
                                                       n=n,m=m, n2=n2,m2=m2,fixed='site',
                                                       fixed_values=[1],initial=matrix,
                                                       method='pivot')
        elif tessellation == 3:
            saveRoute = './data/hexagonal/'
            forest = simulation.heaxgonalForest(burningThreshold=0.55,occuProba=1. , initialForest=matrix)
            #criticalExponent = forest.criticalExponent(saveRoute,epsilon,delta,n,m,n2,m2,matrix)
            criticalExponent = forest.P_inf_criticalExponent(save_route=saveRoute, intervalTol=1e-4,
                                                       n=n,m=m, n2=n2,m2=m2,fixed='site',
                                                       fixed_values=[1],initial=matrix,
                                                       method='pivot')
        
        print(rf'$P_{{inf}}$ critical exponent is: $\\beta = {criticalExponent}$')

        #criticalExponent = forest.criticalExponent_cluster(saveRoute,1e-3,n,m,n2,m2,'site',[0.6,0.7,0.8],matrix,'pivot')
        

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    elif usrChoice == 5:
        n = 35    # Amount of values to consider for p
        m = 5      # Amount of trials per p       
        
        try:
            tessellation = int(input("Choose one: \n1   Squared\n2  Triangular\n3    Hexagonal\n4   Voronoi\n"))
        except:
            print('Not a valid option.')    
        
        if tessellation == 1:
            
            folder_path = data_route['squared']
            file_name = "datos_alta_resolucion.csv"
            name = 'squaredCompareProbabilities_alta_resolucion'
            imagePath = routes_dict['squared'] + name
            propTimeThreshold = 150
            
            forest = simulation.squareForest(burningThreshold=0.95,occuProba=0.95 ,initialForest=matrix)
            forest.compareBondSite(1000,imagePath,folder_path, file_name,matrix, 'squared', propTimeThreshold) 
            
            
        elif tessellation == 2:
            
            folder_path = data_route['triangular']
            file_name = "datos_alta_resolucion.csv"
            name = 'triangularCompareProbabilities_alta_resolucion'
            imagePath = routes_dict['triangular'] + name
            propTimeThreshold = 215
            
            forest = simulation.triangularForest(burningThreshold=0.95,occuProba=0.95 ,initialForest=matrix)
            forest.compareBondSite(1000,imagePath,folder_path, file_name,matrix, 'triangular',propTimeThreshold) 
            
        elif tessellation == 3:
            
            folder_path = data_route['hexagon']
            file_name = "datos_alta_resolucion.csv"
            name = 'hexagonalCompareProbabilities_alta_resolucion'
            imagePath = routes_dict['hexagon'] + name
            propTimeThreshold = 150
            
            forest = simulation.heaxgonalForest(burningThreshold=0.95,occuProba=0.95 ,initialForest=matrix)
            forest.compareBondSite(1000,imagePath,folder_path, file_name,matrix, 'hexagonal',propTimeThreshold) 
        
        elif tessellation == 4:
            nPoints = 10000
            points = np.random.rand(nPoints, 2)
            vor = Voronoi(points)
            folder_path = data_route['voronoi']
            file_name = "datos_alta_resolucion.csv"
            propTimeThreshold = 150
            

            name = 'voronoiCompareProbabilities_alta_resolucion'
            imagePath = routes_dict['voronoi'] + name
            forest = voronoi_fire.voronoiFire(burningThreshold=0.95, occuProba=0.95, voronoi=vor, initialFire=1)
            forest.compareBondSite(1000,imagePath,folder_path, file_name, propTimeThreshold) 
        
        else:
            print('That is not an option, try again.')
        
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    elif usrChoice == 6:
        for tessellation in range(1,5):
            for i,probability in enumerate(np.arange(0.1,1.1,0.1)):
                print(f'Tessellation: {tessellation}, probability: {probability}')

                if tessellation == 1:
                    matrix = np.ones((100,100))
                    matrix[50,50] = 2

                    name = 'squaredAnimation' + '_' + str(i)
                    route = routes_dict['squared'] +  name
                    forest = simulation.squareForest(burningThreshold=probability, occuProba=0.95 ,initialForest=matrix, saveHistoricalPropagation=True)
                    forest.animate(route)    


                elif tessellation == 2:
                    matrix = np.ones((100,100))
                    matrix[50,50] = 2

                    name = 'triangularAnimation' + '_' + str(i)
                    route = routes_dict['triangular'] + name
                    forest = simulation.triangularForest(burningThreshold=probability, occuProba=0.95 ,initialForest=matrix, saveHistoricalPropagation=True)
                    forest.animate(route)


                elif tessellation == 3:
                    matrix = np.ones((100,100))
                    matrix[50,50] = 2

                    name = 'hexagonalAnimation' + '_' + str(i)
                    route = routes_dict['hexagon'] + name    
                    forest = simulation.heaxgonalForest(burningThreshold=probability, occuProba=0.95 ,initialForest=matrix, saveHistoricalPropagation=True)
                    forest.animate(route)

                elif tessellation == 4:
                    # Create Voronoi diagram
                    nPoints = 10000
                    points = np.random.rand(nPoints, 2)
                    vor = Voronoi(points)

                    name = 'voronoiAnimation' + '_' + str(i)
                    route = routes_dict['voronoi'] + name

                    voronoi = voronoi_fire.voronoiFire(burningThreshold=probability, occuProba=0.95, voronoi=vor, initialFire=1)
                    voronoi.animate(route)
    
    elif usrChoice == 7:
        saveRoute = routes_dict['voronoi']
        fitting.expFit(dataRoute='data/voronoi/datos.csv',propTimeThreshold=130,saveRoute=saveRoute)
        
    elif usrChoice == 8:
        n = 35    # Amount of values to consider for p
        m = 5      # Amount of trials per p       
        
        try:
            tessellation = int(input("Choose one: \n1   Squared\n2  Triangular\n3    Hexagonal\n4   Voronoi\n"))
        except:
            print('Not a valid option.')    
        
        if tessellation == 1:
            matrix = np.ones((500,500))
            matrix[250,250] = 2
            
            folder_path = data_route['squared']
            
            
            forest = simulation.squareForest(burningThreshold=0.95,occuProba=0.95 ,initialForest=matrix)
            forest.gamma_exponent(save_route=folder_path,
                                    n=20,
                                    m=30,
                                    n2=30,
                                    m2=70,
                                    fixed='bond',
                                    fixed_values=[0.7],
                                    initial=matrix,
                                    method='pivot')
            
            
        elif tessellation == 2:
            matrix = np.ones((500,500))
            matrix[250,250] = 2
            
            folder_path = data_route['triangular']
            
            
            forest = simulation.triangularForest(burningThreshold=0.95,occuProba=0.95 ,initialForest=matrix)
            forest.gamma_exponent(save_route=folder_path,
                                    n=20,
                                    m=30,
                                    n2=20,
                                    m2=15,
                                    fixed='bond',
                                    fixed_values=[0.7],
                                    initial=matrix,
                                    method='pivot')
            
        elif tessellation == 3:
            matrix = np.ones((100,100))
            matrix[50,50] = 2
            
            folder_path = data_route['hexagon']
            
            
            forest = simulation.heaxgonalForest(burningThreshold=0.95,occuProba=0.95 ,initialForest=matrix)
            forest.criticalExponent(save_route=folder_path,
                                    intervalTol=1e-4,
                                    n=100,
                                    m=15,
                                    n2=100,
                                    m2=15,
                                    fixed='bond',
                                    fixed_values=[0.5,0.6,0.7],
                                    initial=matrix)
        
        #elif tessellation == 4:
        #    
        #    # Still on tests
        #    nPoints = 10000
        #    points = np.random.rand(nPoints, 2)
        #    vor = Voronoi(points)
        #    folder_path = data_route['voronoi']
        #    
        #    
#
        #    name = 'voronoiCompareProbabilities'
        #    imagePath = routes_dict['voronoi'] + name
        #    forest = voronoi_fire.voronoiFire(burningThreshold=0.95, occuProba=0.95, voronoi=vor, initialFire=1)
            
        
        else:
            print('That is not an option, try again.')
            
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    elif usrChoice == 9:

        info = {
        'burningThreshold': 1,
        'occuProba': 1,
        'saveHistoricalPropagation': False
        }

        try:
            tessellation = int(input("Choose one: \n1   Squared\n2  Triangular\n3    Hexagonal\n4   Voronoi\n"))
        except:
            print('Not a valid option.')

        if tessellation == 1:
        
            pc_fit_args = {
                'n':14,
                'm':100,
                'fixed':'bond',
                'fixed_value': 1,
                'exploring_range':[0.52,0.66]
            }

            infinite_pc(l=[25,50,60,70,100,150,200,250,300,350,400,450,500], 
                        save_route=routes_dict['squared'],
                        fire_args=info,
                        pc_args=pc_fit_args, tessellation='squared')
            

        if tessellation == 2:
        
            pc_fit_args = {
                'n':20,
                'm':100,
                'fixed':'bond',
                'fixed_value': 1,
                'exploring_range':[0.6,0.8]
            }

            infinite_pc(l=[50,60,70,100,125,150,170,200,250,300,350,400,450,500], 
                        save_route=routes_dict['triangular'],
                        fire_args=info,
                        pc_args=pc_fit_args, tessellation='triangular')
            
        if tessellation == 3:
        
            pc_fit_args = {
                'n':20,
                'm':100,
                'fixed':'bond',
                'fixed_value': 1,
                'exploring_range':[0.5,0.7]
            }

            infinite_pc(l=[50,60,70,100,125,150,170,200,250,300,350,400,450,500], 
                        save_route=routes_dict['hexagon'],
                        fire_args=info,
                        pc_args=pc_fit_args, tessellation='hexagonal')
            
        if tessellation == 4:

            pc_fit_args = {
                'n':20,
                'm':110,
                'fixed':'site',
                'fixed_value': 1,
                'exploring_range':[0.25,0.45]
            }            

            infinite_pc(l=[50,60,70,100,125,150,170,200,250,300,350,400,450,500], 
                        save_route=routes_dict['voronoi'],
                        fire_args=info,
                        pc_args=pc_fit_args, tessellation='voronoi')