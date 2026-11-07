from field import *
from keygen import *
from _supporting_math import *

import json
import numpy as np

config_data = json.load(open('config.json'))
globals().update(config_data)
print(q,o,v,m,seed)
GF = make_field(q)
rng = np.random.default_rng(seed)


# # temporary testing

# F = generate_central_map(GF, v=v, m=m, rng=rng)
# print(F.shape)                                    # (m, v + m, v + m)
#
# w = GF(rng.integers(0, q, size=(100, m)))
# zeros = GF(np.zeros((100, v), dtype=np.int64))
# pts = np.hstack([zeros, w])                       # vinegar part is 0
# print(np.any(eval_quadratic_map(F, pts)))         # expect False: F vanishes there


F = generate_central_map(GF, v=v, m=m, rng=rng)
T = generate_transformation(GF, v, m, rng)
P = compute_public_key(F, T)
print(P.shape)       # (m, v + m, v + m)
print(np.any(P[:, v:, :] != 0))     # True: P no longer has zero oil rows