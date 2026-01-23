import numpy as np

# Numpy has the classic python slicing
np1 = np.arange(0,10,1)
print(np1)
print(np1[1:6]) # [1 2 3 4 5]
print(np1[:6]) # [0 1 2 3 4 5]
print(np1[6:]) # [6 7 8 9]
print(np1[:-1]) # [0 1 2 3 4 5 6 7 8]
print(np1[-6:]) # [4 5 6 7 8 9]
print(np1[-5:-2]) # [5 6 7]
# Also possible with steps
print(np1[1:6:2],"\n") # [1 3 5]

# Slicing with multi-dim array
np2d = np.array([
    [1,2,3,4,5],
    [4,5,6,7,8]])
print(np2d[0:2, 3:5]) # 0:2 defines which rows (0 and 1), 3:5 which elements of a row