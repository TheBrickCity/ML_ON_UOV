# from field import *
# from keygen import *
# from _supporting_math import *
#
# import json
# import numpy as np
#
# config_data = json.load(open('config.json'))
# globals().update(config_data)
# print(q,o,v,m,seed)
# GF = make_field(q)
# rng = np.random.default_rng(seed)
#
# F = generate_central_map(GF, v=v, m=m, rng=rng)
# T = generate_transformation(GF, v, m, rng)
# P = compute_public_key(F, T)
# print(P.shape)       # (m, v + m, v + m)
# print(np.any(P[:, v:, :] != 0))     # True: P no longer has zero oil rows

import numpy as np
import torch

from UOV_sample_manager import UOV
from dataset import UOVDataset
from model import UOVTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, "|", torch.cuda.get_device_name(0) if device.type == "cuda" else "no GPU found")

uov = UOV(q=16, v=4, o=3, m=3, rng=np.random.default_rng(0))
uov.generate_keys()
uov.generate_samples(256)
ds = UOVDataset(uov, base=16)

model = UOVTransformer(ds.vocab_size, ds.src_len, ds.tgt_len).to(device)
src, tgt = ds.src[:8].to(device), ds.tgt[:8].to(device)

logits = model(src, tgt[:, :-1])                      # forward pass
loss = torch.nn.functional.cross_entropy(logits.reshape(-1, ds.vocab_size), tgt[:, 1:].reshape(-1))
loss.backward()                                       # backward pass
generated = model.generate(src, ds.SOS, ds.tgt_len)   # autoregressive generation
x_hat = ds.decode_target(generated)                   # back to CPU ints

print("logits on:", logits.device, "| loss:", round(loss.item(), 3), "| generated:", tuple(generated.shape), "| decoded:", x_hat.shape)