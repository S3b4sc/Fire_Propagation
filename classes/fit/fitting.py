import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def expFit(data, propTimeThreshold:int):
    
    # Filter those of interest based on propagation time
    phaseTransitionData = data[ data['time'] > propTimeThreshold ]
    
    # Extrac the desired data for fitting
    ps =  phaseTransitionData['P_site']
    pb =  phaseTransitionData['P_bond']
    
    
    function = lambda pb,A,B,C : A * np.exp( -B *pb ) + C
    
    # Execute the fit 
    popt, pcov = curve_fit(function, pb, ps,maxfev=5000)
    
    return function,ps,pb,popt
    
    




