import numpy as np


def h(n,N):
    h = []
    for i in range(N):
        h.append(n)
    h = tuple(i for i in h)
    print(type(h))
    return h
    
h(2,3)
print(np.arange(3))