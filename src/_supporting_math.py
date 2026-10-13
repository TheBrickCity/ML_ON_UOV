import numpy as np


def eval_quadratic_map(Ms, x):
    """Evaluate m quadratic forms at x. All arithmetic happens in the field of Ms and x.

    Ms: galois field array of shape (m, n, n). Ms[k] is the upper-triangular
        coefficient matrix of polynomial k.
    x:  galois field array (same field) of shape (n,) for one point, or (B, n) for a batch.

    Returns a field array of shape (m,) for one point, or (B, m) for a batch.
    """
    single = (x.ndim == 1)
    if single:
        x = x[None, :]                       # treat one point as a batch of size 1

    row = x[:, None, None, :]                # (B, 1, 1, n): each point as a row vector
    col = x[:, None, :, None]                # (B, 1, n, 1): each point as a column vector
    out = (row @ Ms) @ col                   # (B, m, 1, 1): x^T M_k x for every point and polynomial
    out = out[:, :, 0, 0]                    # (B, m)

    return out[0] if single else out