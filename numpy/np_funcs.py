import numpy as np

# Universal funcs in Numpy
np1 = np.array([4,16,36,64,128,333,4,1])
neg = np.array([-4,-16,-36,64,-128,333,4,0])

# transforms each element into its square root
print(np.sqrt(np1))

# transforms each element into it's abs value
print(np.absolute(neg))

# transforms each element into exponentials
print(np.exp(np1))

# finds min/max from array
print(f"max: {np.max(np1)}, min: {np.min(np1)}")

# transforms negative elements into -1, 0 as 0, positive as 1
print(np.sign(neg))

# Trigonometrical funcs sin,cos, tan or also
# sinh, tanh, cosh - hyperbolic
# arcsin, arccos, arctan
print(np.tanh(np1))

# logarithmic log / log2 / log10 / logn via np.emath
print(np.log(np1))