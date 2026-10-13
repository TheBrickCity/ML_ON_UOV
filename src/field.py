import galois

# Irreducible polynomials for the fields GF(2^k) we use.
# 16 and 256 are the UOV spec's. The others are not in the spec, so we take the
# numerically smallest irreducible polynomial of each degree (this also reproduces the spec's two choices).
IRREDUCIBLE = {
    4:   "x^2 + x + 1",
    8:   "x^3 + x + 1",
    16:  "x^4 + x + 1",
    32:  "x^5 + x^2 + 1",
    64:  "x^6 + x + 1",
    128: "x^7 + x + 1",
    256: "x^8 + x^4 + x^3 + x + 1",
}


def make_field(q):
    if q in IRREDUCIBLE:
        return galois.GF(q, irreducible_poly=IRREDUCIBLE[q])
    return galois.GF(q)