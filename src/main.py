from field import *
from keygen import *
from _supporting_math import *

import json
import numpy as np

config_data = json.load(open('config.json'))
globals().update(config_data)
print(q,o,v,m,seed)
GF = make_field(q)



# temporary testing
rng = np.random.default_rng(seed)
F = generate_central_map(GF, v=v, m=m, rng=rng)
print(F.shape)                                    # (m, v + m, v + m)

w = GF(rng.integers(0, q, size=(100, m)))
zeros = GF(np.zeros((100, v), dtype=np.int64))
pts = np.hstack([zeros, w])                       # vinegar part is 0
print(np.any(eval_quadratic_map(F, pts)))         # expect False: F vanishes there
