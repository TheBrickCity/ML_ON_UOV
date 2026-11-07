import numpy as np
from _supporting_math import eval_quadratic_map


def valid_rate(P, t, x_hat):
    """Score predicted preimages x_hat for the targets t.

    P:     public key, GF array of shape (m, n, n)
    t:     targets, GF array of shape (B, m)
    x_hat: predicted inputs, GF array of shape (B, n)

    Returns (valid, per_equation):
      valid:        fraction of samples where P(x_hat) == t in all m equations
      per_equation: fraction of individual equations that are satisfied
    """
    matches = (eval_quadratic_map(P, x_hat) == t)      # (B, m) array of True/False
    return float(matches.all(axis=1).mean()), float(matches.mean())