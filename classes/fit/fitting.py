import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import mean_squared_error


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
    
    
def gaussian(p, A, mu, sigma):
    return A * np.exp(-((p - mu)**2) / (2 * sigma**2))

def poly4(x, a, b, c, d, e):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e


def lognormal(x, A, mu, sigma):
    # Evitar valores <= 0 (por log)
    x = np.where(x <= 0, 1e-6, x)
    return A * (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(- (np.log(x) - mu)**2 / (2 * sigma**2))


def lorentzian(x, A, x0, gamma):
    return A * gamma**2 / ((x - x0)**2 + gamma**2)


def skew_normal(x, A, xi, omega, alpha):
    return 2 * A * norm.pdf((x - xi) / omega) * norm.cdf(alpha * (x - xi) / omega)


# Diccionario de modelos
model_dict = {
    "gaussian": (gaussian, lambda x, y: [max(y), x[np.argmax(y)], 0.01]),
    "lorentzian": (lorentzian, lambda x, y: [max(y), x[np.argmax(y)], 0.01]),
    "poly4": (poly4, lambda x, y: [1e-3]*5),
    "lognormal": (lognormal, lambda x, y: [max(y), np.log(x[np.argmax(y)] + 1e-3), 0.2]),
    "skew_normal": (skew_normal, lambda x, y: [max(y), x[np.argmax(y)], 0.05, 5])
}
from typing import Tuple

def fit_best_model(x: np.ndarray, y: np.ndarray, bootstrap_n: int = 500) -> Tuple[float, str, np.ndarray, np.ndarray, float]:
    """
    Ajusta varios modelos a (x, y) y retorna:
    - el valor x que da el máximo del mejor ajuste (estimación de pc)
    - el nombre del mejor modelo
    - los parámetros del modelo
    - la matriz de covarianza (o None si no aplica)
    - el error en pc (estimado con la matriz o por bootstrap si polinomio)
    """
    best_model_name = None
    best_params = None
    best_cov = None
    best_fit_func = None
    lowest_mse = np.inf

    for name, (model_func, init_func) in model_dict.items():
        try:
            p0 = init_func(x, y)
            popt, pcov = curve_fit(model_func, x, y, p0=p0, maxfev=10000)
            y_pred = model_func(x, *popt)
            mse = mean_squared_error(y, y_pred)

            if mse < lowest_mse:
                lowest_mse = mse
                best_model_name = name
                best_params = popt
                best_cov = pcov
                best_fit_func = model_func
        except Exception:
            continue

    if best_model_name is None:
        raise RuntimeError("Ningún modelo pudo ajustarse correctamente.")

    # Estimar pc como el máximo de la curva ajustada
    x_dense = np.linspace(min(x), max(x), 1000)
    y_dense = best_fit_func(x_dense, *best_params)
    pc_estimate = x_dense[np.argmax(y_dense)]

    # === Estimar el error de pc ===
    if best_model_name == "poly4":
        pc_samples = []
        for _ in range(bootstrap_n):
            # Resamplea índices con reemplazo
            idx = np.random.choice(len(x), len(x), replace=True)
            x_boot = x[idx]
            y_boot = y[idx]

            try:
                p0 = model_dict["poly4"][1](x_boot, y_boot)
                popt_boot, _ = curve_fit(model_dict["poly4"][0], x_boot, y_boot, p0=p0, maxfev=5000)
                y_fit_boot = model_dict["poly4"][0](x_dense, *popt_boot)
                pc_boot = x_dense[np.argmax(y_fit_boot)]
                pc_samples.append(pc_boot)
            except Exception:
                continue  # Saltar errores

        if len(pc_samples) > 10:
            pc_error = np.std(pc_samples)
        else:
            pc_error = np.nan  # no hay suficiente datos bootstrap
    else:
        # Si hay matriz de covarianza, toma el error del parámetro relevante
        try:
            pc_error = np.sqrt(best_cov[1, 1])
        except:
            pc_error = np.nan

    return pc_estimate, best_model_name, best_params, pc_error
