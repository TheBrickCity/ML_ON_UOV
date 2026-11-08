import numpy as np

from field import make_field
from keygen import generate_central_map, generate_transformation, compute_public_key
from _supporting_math import eval_quadratic_map


class UOV:
    def __init__(self, q, v, o, m, rng):
        # classic UOV has exactly as many oil variables as equations
        assert o == m, f"classic UOV needs o == m, got o={o}, m={m}"
        self.q = q
        self.v = v
        self.o = o
        self.m = m
        self.n = v + o              # total number of variables
        self.rng = rng              # np.random.Generator, shared with the rest of the code
        self.field = make_field(q)  # galois field class GF(q)

        # private key
        self.F = None               # central map, (m, n, n), no oil x oil terms
        self.T = None               # invertible scramble, (n, n)
        # public key
        self.P = None               # (m, n, n), upper triangular
        # samples: plain int64 arrays (field elements stored as their integer labels 0..q-1)
        self.x = None               # inputs,  (num_samples, n)
        self.t = None               # targets, (num_samples, m), t = P(x)

    def generate_keys(self):
        self._generate_central_map()
        self._generate_transformation()
        self._generate_public_key()

    def generate_samples(self, num_samples):
        # needs a key first; calling this again draws a fresh set of samples under the SAME key
        if self.P is None:
            raise RuntimeError("call generate_keys() before generate_samples()")
        self._generate_inputs(num_samples)
        self._generate_targets()

    def _generate_central_map(self):
        F = generate_central_map(self.field, m=self.m, v=self.v, rng=self.rng)
        self.F = F
        return F

    def _generate_transformation(self):
        T = generate_transformation(self.field, v=self.v, m=self.m, rng=self.rng)
        self.T = T
        return T

    def _generate_public_key(self):
        P = compute_public_key(self.F, self.T)
        self.P = P
        return P

    def _generate_inputs(self, num_samples):
        # every coordinate uniform over {0, ..., q-1}
        x = self.rng.integers(0, self.q, size=(num_samples, self.n), dtype=np.int64)
        self.x = x
        return x

    def _generate_targets(self):
        # t = P(x): evaluate all m public quadratics at every sample, in the field GF(q)
        t = eval_quadratic_map(self.P, self.field(self.x))
        self.t = np.array(t, dtype=np.int64)   # back to plain ints so torch/numpy can use them
        return self.t