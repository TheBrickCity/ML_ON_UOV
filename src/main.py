from field import *

import json
import numpy as np

config_data = json.load(open('config.json'))
globals().update(config_data)

print(q,o,v,m)
GF = make_field(q)
print(GF.order, GF.characteristic, GF.irreducible_poly)