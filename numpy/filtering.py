import numpy as np

# Filtering
np1=np.array([5,214,756,132,8,43,7,9832,23])

# Basic boolean filter to choose exact elements
filter1 = [True, True, False, False, False, False, False, False, False]
# Pass filter into array so it gets applied
print(np1[filter1])

# Another possibility - iteratively create an array via logic in loop
even_filter = []
for i in np1:
    if i%2==0: even_filter.append(True)
    else: even_filter.append(False)

print(np1[even_filter])

# Create filter directly from the array (same but shortcut)
odd_filter = np1%2==1
print(np1[odd_filter])

def prime(n):
    for i in range(2, int(np.sqrt(n) + 1)):
        if n%i==0: return False
    return True

# Also doesn't work with custom funcs, same as with searching
# prime_filter = prime(np1) <-- Broken!
prime_filter = []
for i in np1:
    if prime(i): prime_filter.append(True)
    else: prime_filter.append(False)
print(np1[prime_filter])