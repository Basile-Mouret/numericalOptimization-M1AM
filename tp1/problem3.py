import numpy as np

# Oracles of the Himmelblau function

def f(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


def grad_f(x):
    return np.array([4*x[0]*(x[0]*x[0] + x[1] - 11) + 2*(x[0]+x[1]*x[1]-7),
                     2*(x[0]*x[0] + x[1] - 11 + 4*x[1]*(x[0]+x[1]*x[1]-7))])

def hessian_f(x):
    return np.array([[12*x[0]*x[0]+ 4*x[1]-42, 4*(x[0]-x[1])],[4*(x[0]+x[1]), 4*x[0]+12*x[1]*x[1]-26]])


# useful constants for plotting

lb = -5.0
ub = 5.0
nb_points = 100
levels = [0.0, 3.0, 15.0, 65.0, 180.0, 300.0]
