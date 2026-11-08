import numpy as np
import torch
from torch.utils.data import Dataset


class UOVDataset(Dataset):
    def __init__(self, UOV, base, include_key=False, separator=True):
        self.q = UOV.q
        self.n = UOV.n
        self.m = UOV.m
        self.base = base
        self.include_key = include_key
        self.separator = separator

        self.digits_per_int = self._digit_len(self.q, self.base)  # how many base B digits are needed per field element

        # unique tokens for separation (between elements) and start/end of sequence (decoder)
        self.SEP = base
        self.SOS = base + 1
        self.EOS = base + 2
        self.vocab_size = base + 3

        # encoder input: the target t = P(x), optionally preceded by the public key coefficients
        src_values = UOV.t                                          # (N, m)
        if include_key:
            key = self._key_coefficients(UOV.P)                     # (m * n(n+1)/2,), same for every sample
            key = np.tile(key, (len(UOV.t), 1))                     # (N, m * n(n+1)/2)
            src_values = np.concatenate([key, UOV.t], axis=1)

        src = self._encode_values(src_values)                       # (N, src_len)
        tgt_body = self._encode_values(UOV.x)                       # (N, tgt_len - 2)

        # decoder target: [SOS] x [EOS]
        N = len(tgt_body)
        sos = np.full((N, 1), self.SOS, dtype=np.int64)
        eos = np.full((N, 1), self.EOS, dtype=np.int64)
        tgt = np.concatenate([sos, tgt_body, eos], axis=1)

        # everything is encoded once, up front, so __getitem__ is just an index
        self.src = torch.tensor(src, dtype=torch.long)
        self.tgt = torch.tensor(tgt, dtype=torch.long)
        self.src_len = self.src.shape[1]
        self.tgt_len = self.tgt.shape[1]

    def _digit_len(self, q, base):  # returns k the smallest amount of digits to uniquely represent each value 0,...,q-1
        k = 0
        val = 1
        while val < q:
            val *= base
            k += 1
        return k

    def _key_coefficients(self, P):  # the n(n+1)/2 upper-triangular entries of each of the m matrices, flattened
        rows, cols = np.triu_indices(self.n)
        return np.array(P, dtype=np.int64)[:, rows, cols].reshape(-1)

    def _encode_values(self, values):
        # values: int array (N, count). Returns tokens (N, L): each value as digits_per_int base B
        # digits (most significant first), with SEP between values (not after the last) if separator=True.
        powers = self.base ** np.arange(self.digits_per_int - 1, -1, -1, dtype=np.int64)  # (d,) e.g. [B^2, B, 1]
        digits = (values[:, :, None] // powers) % self.base                               # (N, count, d)
        if self.separator:
            sep = np.full(values.shape + (1,), self.SEP, dtype=np.int64)
            digits = np.concatenate([digits, sep], axis=2)                                # (N, count, d + 1)
        tokens = digits.reshape(len(values), -1)
        if self.separator:
            tokens = tokens[:, :-1]                                                       # drop the trailing SEP
        return tokens

    def decode_target(self, tokens):
        # inverse of the target encoding: (B, tgt_len) tokens [SOS] ... [EOS] -> (B, n) ints
        if torch.is_tensor(tokens):
            tokens = tokens.cpu().numpy()
        body = tokens[:, 1:-1]                                      # strip SOS and EOS
        if self.separator:
            pad = np.full((len(body), 1), self.SEP, dtype=np.int64)
            body = np.concatenate([body, pad], axis=1)              # put the trailing SEP back so every group has the same width
        width = self.digits_per_int + (1 if self.separator else 0)
        groups = body.reshape(len(body), -1, width)[:, :, :self.digits_per_int]
        powers = self.base ** np.arange(self.digits_per_int - 1, -1, -1, dtype=np.int64)
        return (groups * powers).sum(axis=2)

    def __len__(self):  # number of training samples
        return len(self.src)

    def __getitem__(self, idx):  # how the model grabs a single sample
        return self.src[idx], self.tgt[idx]