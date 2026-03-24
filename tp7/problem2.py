import numpy as np


d = 10

A = np.random.rand(d,d)
A = 0.5*(A + A.T)

def f(x):
    return np.inner(x,A@x)
    
def grad_f(x):
    return 2*A@x

#projection onto a sphere of radius 1

def proj(x):
    return x/np.linalg.norm(x)
