import numpy as np

# These 2 arrays are broadcastable
np1 = np.array([4,16,36,64,128,333,4,1])
np2 = np.array([[1],[2],[3],[4]])
np3 = np.array([[1,2,3,4],[5,6,7,8]])
np4 = np.array([[1,2,3,4],
                [5,6,7,8],
                [9,10,11,12],
                [13,14,15,16]])

print(f"Shape of np1: {np1.shape}, of np2: {np2.shape}\n")

print(f"{np1*np2}\n shape: {(np1*np2).shape}\n")
"""
[[   4   16   36   64  128  333    4    1] <-- np1 * np2[0]
 [   8   32   72  128  256  666    8    2] <-- np1 * np2[1]
 [  12   48  108  192  384  999   12    3] <-- np1 * np2[2]
 [  16   64  144  256  512 1332   16    4]] <-- np1 * np2[3]
 shape: (4, 8)
"""


try:
    print(np1*np3) # Shapes don't match (8,) and (2,4)
except Exception as e: (
    print(f"Trying to broadcast np1 and np3 results in: [{e}]\n"))

print(f"np2 and np4 match (4,1) and (4,4): \n {np2*np4} shape: {(np2*np4).shape}")