import numpy as np

# Different 2d functions for the surface
# (For representation, not actually used)
# f(x,y) = x^2 + y^2
def bowl(x):
    return x[0]**2 + x[1]**2

# f(x,y) = x^2 + 10y^2
def valley(x):
    return x[0]**2 + 10*x[1]**2

# f(x,y) = (1 - x)^2 + 100(y-x^2)^2
# smoothened for better visuals
def rosenbrock(x):
    return 1/10*(1-x[0])**2 + 5*(x[1]-x[0]**2)**2

#f(x,y) = x^2 - y^2
def saddle(x):
    return x[0]**2 - x[1]**2

# Gradients of the surface functions
def grad_bowl(x):
    return np.array([2*x[0], 2*x[1]])

def grad_valley(x):
    return np.array([2 * x[0], 20 * x[1]])

def grad_rb(x):
    return np.array([
        -2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2),
        200*(x[1] - x[0]**2)
    ])

def grad_saddle(x):
    return np.array([2*x[0], -2*x[1]])