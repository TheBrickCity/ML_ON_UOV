from _supporting_math import *
import numpy as np

# q = 7
# A = np.array([[[1,2],[0,3]], [[4,0],[0,5]]])
#
# print(eval_quadratic_map(A, np.array([[3,4],[5,2]]),q))


import json

config_data = json.load(open('config.json'))
globals().update(config_data)

print(q,o,v,m)