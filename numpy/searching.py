import numpy as np

# Searching
np1 = np.array([4,6,83,4,62,8,83,98,4])

found = np.where(np1==4) # Saves as a tuple (array([0, 3, 8]),)
print(found[0]) # Properly print positions

# Print the elements itself
print(np1[found]) # np1[found[0]] would do the same

# Other where possibilities
evens = np.where(np1%2==0)
bigs = np.where(np1>50)
print(f"[EVENS] Indices:{evens[0]}, elements:{np1[evens]}")
print(f"[BIGS] Indices:{bigs[0]}, elements:{np1[bigs]}")


def prime(n):
    for i in range(2, int(np.sqrt(n) + 1)):
        if n%i==0: return False
    return True

 #primes = np.where(prime(np1)) <-- DOESN'T WORK; can't pass own func into it
 # The idea is to mask
np2 = np.array([True if prime(i) else False for i in np1])
primes = np.where(np2==True)
print(f"[PRIMES] Indices:{primes[0]}, elements:{np1[primes]}")