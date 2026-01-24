import numpy as np

# Copy and View in numpy
np1 = np.array([6,76,84,2,57,8,4])

# Create a View. Views share memory with original arrays
# Therefore, modification of an array/view also affects the other paired element
# Slicing creates views
np_view = np1.view()
print(f"Original np1: {np1}\nView np_view: {np_view}\n")

np_view[0]=80085 # also affects the original

print(f"Changed np1: {np1}\nView np_view: {np_view}\n")

# Create a copy. Copy exists in its own allocated data buffer.
# Therefore they are disconnected
np_copy = np1.copy()
print(f"Original np1: {np1}\nCopy np_copy: {np_copy}\n")

np1[0]=112023

print(f"Changed np1: {np1}\nCopy np_copy: {np_copy}\n")