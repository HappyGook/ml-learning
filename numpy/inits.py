import numpy as np

# basic lists in python
lit = [1,"2",3, True]
"""
Takes a lot of memory, because each element is stored as an object
with it's overhead. Each pointer is 8 Bytes, overheads vary
list object
 ├── ptr ──> PyLongObject(1) 28 Bytes
 ├── ptr ──> PyUnicodeObject("2") 50 Bytes
 ├── ptr ──> PyLongObject(3) 28 Bytes
 └── ptr ──> PyBoolObject(True) 28 Bytes (bool is int :/ )
 All together: 134 Bytes + internal overhead = 206 Bytes
"""
size = lit.__sizeof__()
for item in lit:
    size += item.__sizeof__()
print(f"Size of lit: {size}, type: {type(lit)}\n")
# Numpy gives
# ndarray = n-dimensional array
np1 = np.array([1,"2",3, True], dtype=object)
"""
forced same behavior
ndarray
 ├── ptr ──> PyLongObject(1)
 ├── ptr ──> PyUnicodeObject("2")
 ├── ptr ──> PyLongObject(3)
 └── ptr ──> PyBoolObject(True)
 Same 166 Bytes (32 Pointers + 134 elements)
"""
size = 0
for item in np1:
    size += item.__sizeof__()
print(f"Size of np1 is: {size + np1.nbytes}, dtype: {np1.dtype}\n")


np2 = np.array([1,"2",3, True], dtype='<U4')
print(f"Size of np2 is: {np2.nbytes}, dtype: {np2.dtype}\n")
"""
numpy finds a common dtype for all elements. Here - string of size 4
Array becomes => np.array([1,"2",3, True], dtype='<U4')
Contiguous memory: | '1\0\0\0' | '2\0\0\0' | '3\0\0\0' | 'True' |
No overheads, one pointer (like C), so 4*16 bytes = 64 Bytes
"""

"""
Init methods
np.arange creates an array of elements in range, with given step
np.zeros creates an array of zeros
np.full((size), fill_value) creates an array and fills it with given value
"""
np3 = np.arange(10)
# OR
np4 = np.arange(10,24,2)
# Zeros
np5 = np.zeros(10)
#Multidimensional works the same
np6 = np.full((3,4),6)

# Converting lists to np's:
np8 = np.array(lit)
print(np8)