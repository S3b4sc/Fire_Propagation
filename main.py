from classes.regular import simulation
from classes.voronoi import voronoi_fire

from classes.fit import fitting

from scripts.menu import menu
import numpy as np
from scipy.spatial import Voronoi

from scripts.routes import routes_dict, data_route
import matplotlib.pyplot as plt


import time

if __name__ == '__main__':
    usrChoice = menu()
    matrix = np.ones((10,10))
    matrix[5,5] = 2
  
  
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
    if usrChoice == 1:
        try:
            tessellation = int(input("Choose one: \n1   Squared\n2   Triangular\n3   Hexagonal\n4   Voronoi\n"))
        except:
            print('Not a valid option.')
            
            
        if tessellation == 1:
            
            name = 'squaredAnimation_test'
            route = routes_dict['squared'] +  name
            forest = simulation.squareForest(burningThreshold=0.95,occuProba=0.95 ,initialForest=matrix, saveHistoricalPropagation=True)
            #forest.animate(route)    
            start = time.time()
            history, steps = forest.propagateFire(ps=1,pb=1)
            print("Execution time:", time.time() - start)
            print(steps)
            
        elif tessellation == 2:
            
            name = 'triangularAnimation'
            route = routes_dict['triangular'] + name
            forest = simulation.triangularForest(burningThreshold=0.95,occuProba=0.95 ,initialForest=matrix, saveHistoricalPropagation=True)
            forest.animate(route)
        
        
        elif tessellation == 3:
            
            name = 'hexagonalAnimation'
            route = routes_dict['hexagon'] + name    
            forest = simulation.heaxgonalForest(burningThreshold=0.95,occuProba=0.95 ,initialForest=matrix, saveHistoricalPropagation=True)
            forest.animate(route)
            
        elif tessellation == 4:
            # Create Voronoi diagram
            nPoints = 10000
            points = np.random.rand(nPoints, 2)
            vor = Voronoi(points)
            
            name = 'voronoiAnimation'
            route = routes_dict['voronoi'] + name

            voronoi = voronoi_fire.voronoiFire(1.,0.5,vor,1,)
            voronoi.animate(route)
            
        else:
            print('That is not an option, try again.')
            
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    elif usrChoice == 2:
        
        n = 100    # Amount of values to consider for p
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
        n = 35    # Amount of values to consider for p
        m = 40      # Amount of trials per p        
        n_iter = 3
        
        try:
            tessellation = int(input("Choose one: \n1   Squared\n2  Triangular\n3    Hexagonal\n4   Voronoi\n"))
        except:
            print('Not a valid option.')    
        
        if tessellation == 1:
            
            name = 'SquaredPercolationThreshold'
            route = routes_dict['squared'] +  name
            forest = simulation.squareForest(burningThreshold=0.95,occuProba=0.95 ,initialForest=matrix, saveHistoricalPropagation=True)
            #p_c = forest.percolationThreshold(n,m,matrix,True,"site", fixed_value=1,saveRoute=route)
            p_c = forest.estimate_percolation_threshold(m=m, matrix=matrix, n_iter=n_iter)
            
           
            print("The percolation threshold is: ",p_c)
            
            
        elif tessellation == 2:
            
            name = 'TriangularPercolationThreshold'
            route = routes_dict['triangular'] + name 
            forest = simulation.triangularForest(burningThreshold=0.55, occuProba=1 ,initialForest=matrix)
            #p_c = forest.percolationThreshold(n,m,matrix,False,"site",fixed_value=1,saveRoute=route)
            p_c = forest.estimate_percolation_threshold(m=m, matrix=matrix, n_iter=n_iter)
            
            print("The percolation threshold is: ",p_c)
            
        elif tessellation == 3:
            
            name = 'hexagonalPercolationThreshold'
            route = routes_dict['hexagon'] + name
            forest = simulation.heaxgonalForest(burningThreshold=0.55, occuProba=1 ,initialForest=matrix)
            p_c = forest.percolationThreshold(n,m,matrix,True,"site", saveRoute=route)
            print("The percolation threshold is: ",p_c)
        
        elif tessellation == 4:
            print('Not implemented just yet')
        
        else:
            print('That is not an option, try again.')
        
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    elif usrChoice == 4:
        n = 20      # Amount of values to consider for p in the range (0,1) to find p_c
        m = 10      # Amount of trialas per each p to find p_c
        
        n2 = 20      # Amount of values to consider for p 
        m2 = 10     # 
        saveRoute = './data/squared/'
        epsilon = 0.002*6
        delta = 0.002
        
        forest = simulation.squareForest(burningThreshold=0.55,occuProba=1. , initialForest=matrix)
        criticalExponent = forest.criticalExponent(saveRoute,epsilon,delta,n,m,n2,m2,matrix)
        print(criticalExponent)

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
            matrix = np.ones((100,100))
            matrix[50,50] = 2
            
            folder_path = data_route['squared']
            
            
            forest = simulation.squareForest(burningThreshold=0.95,occuProba=0.95 ,initialForest=matrix)
            forest.criticalExponent(save_route=folder_path,
                                    intervalTol=1e-4,
                                    n=100,
                                    m=15,
                                    n2=100,
                                    m2=15,
                                    fixed='bond',
                                    fixed_values=[0.65,0.75,0.85],
                                    initial=matrix)
            
            
        elif tessellation == 2:
            matrix = np.ones((100,100))
            matrix[50,50] = 2
            
            folder_path = data_route['triangular']
            
            
            forest = simulation.triangularForest(burningThreshold=0.95,occuProba=0.95 ,initialForest=matrix)
            forest.criticalExponent(save_route=folder_path,
                                    intervalTol=1e-4,
                                    n=100,
                                    m=15,
                                    n2=100,
                                    m2=15,
                                    fixed='bond',
                                    fixed_values=[0.5,0.6,0.7],
                                    initial=matrix)
            
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

    
