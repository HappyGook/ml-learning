import numpy as np

# Different gradient descent algorithms

# Simple gradient descent always computes gradients with the same weight
def classic_descent(x0, step_size, iters, gradient, clip = 10):
    xs = [x0]
    x=x0.copy()
    for _ in range(iters):
        g = gradient(x)
        norm = np.linalg.norm(g)
        if norm > clip:
            g = g * (clip/ norm)
        x = x - step_size * g
        # So the arrays aren't filled with infs/nans
        if not np.all(np.isfinite(x)):
            break
        xs.append(x.copy())
    return np.array(xs)

# Gradient with momentum computes updates as a linear combination
def momentum_descent(x0, step_size, iters, gradient, alpha, clip = 10):
    xs = [x0]
    x = x0.copy()
    delta = np.zeros_like(x)
    for _ in range(iters):
        g = gradient(x)
        norm = np.linalg.norm(g)
        if norm > clip:
            g = g * (clip / norm)
        delta = alpha*delta - step_size * g
        x = x + delta
        if not np.all(np.isfinite(x)):
            break
        xs.append(x.copy())
    return np.array(xs)
