import numpy as np
import sympy as sp
import galois

def generate_central_map(field, m, v, rng):
    n = v + m
    F = np.zeros((m, n, n), dtype=np.int64) # m many nxn matricies
    F[:, :v, :] = rng.integers(0, field.order, size=(m, v, n)) # random ints in  first v rows
    return field(np.triu(F)) # zero out below the diagonal

