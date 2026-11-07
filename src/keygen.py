import numpy as np
import sympy as sp
import galois

def generate_central_map(field, m, v, rng):
    n = v + m
    F = np.zeros((m, n, n), dtype=np.int64) # m many nxn matricies
    F[:, :v, :] = rng.integers(0, field.order, size=(m, v, n)) # random ints in  first v rows
    return field(np.triu(F)) # zero out below the diagonal

def _generate_candidate_transformation(field, v, m, rng):
    n = v + m
    return field(rng.integers(0, field.order, size=(n, n)))


def _check_transformation(T):
    return bool(np.linalg.matrix_rank(T) == T.shape[0])


def generate_transformation(field, v, m, rng):
    candidate_transformation = _generate_candidate_transformation(field, v, m, rng)
    while not _check_transformation(candidate_transformation):
        candidate_transformation = _generate_candidate_transformation(field, v, m, rng)
    return candidate_transformation

def compute_public_key(F, T):
    field = type(F)
    N = T.T @ F @ T                                                 # (m, n, n)
    upper = field(np.triu(np.array(N)))                             # diagonal and above
    above = field(np.triu(np.array(N.swapaxes(-1, -2)), k=1))       # reflected entries
    return upper + above