import numpy as np


def eval_quadratic_map(Ms, x, q):
    """Evaluate m quadratic forms at x, with all arithmetic done mod q.

    Ms: int array of shape (m, n, n). Each Ms[k] is an upper-triangular
        matrix holding the coefficients of the k-th polynomial.
    x:  int array of shape (n,) for one point, or (B, n) for a batch.
    q:  a prime, so that the integers mod q form a field.

    Returns an array of shape (m,) for one point, or (B, m) for a batch.
    """
    Ms = np.asarray(Ms, dtype=np.int64)
    x = np.asarray(x, dtype=np.int64)

    single = (x.ndim == 1)
    if single:
        x = x[None, :]  # treat one point as a batch of size 1

    # out[b, k] = sum over i, j of  x[b, i] * Ms[k, i, j] * x[b, j]
    out = np.einsum("bi,kij,bj->bk", x, Ms, x) % q

    return out[0] if single else out