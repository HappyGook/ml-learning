import numpy as np

# get a shape of an array
np1 = np.array([2,7,35,768,32,68])
print(np1.shape) # (6,)

np2 = np.array([[2,7,35,768,32,68,8,9],[2,7,35,768,32,68,0,7],[2,7,35,768,32,68,5,7]])
print(np2)
print(np2.shape) # (3,8)

# reshaping
np3 = np1.reshape(2,3) # 2 rows á 3 elements. Array must have exact passing amount of elements
print(np3)
print(np3.shape) # (2,3)

np4 = np2.reshape(4,6) # The same works here. 4*6 = 3*8
print(np4)
print(np4.shape)

np5 = np2.reshape(2,6,2) # Reshape into a 3D array
print(np5)
print(np5.shape)

# Flatten an array
np_flat = np5.reshape(-1) # Both works
np_flat2 = np5.flatten()
print(np_flat)
print(np_flat2)