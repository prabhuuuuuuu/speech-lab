"""
Lab 5 (d) - ROUGE-1, ROUGE-2 and ROUGE-L (implemented from scratch).

    ROUGE-N  : clipped n-gram overlap.   P = overlap / #ngrams(candidate)
                                          R = overlap / #ngrams(reference)
    ROUGE-L  : longest common subsequence (LCS) of the token sequences.
               P = LCS / |candidate|, R = LCS / |reference|
    F1 = 2PR / (P + R)

Tokens are lower-cased alphanumeric words (no stemming, punctuation ignored).
"""

import re
from collections import Counter


def _tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _prf(overlap, n_cand, n_ref):
    p = overlap / n_cand if n_cand else 0.0
    r = overlap / n_ref if n_ref else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def rouge_n(candidate, reference, n):
    c, r = _ngrams(_tokens(candidate), n), _ngrams(_tokens(reference), n)
    overlap = sum((c & r).values())
    return _prf(overlap, sum(c.values()), sum(r.values()))


def lcs_length(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(candidate, reference):
    c, r = _tokens(candidate), _tokens(reference)
    return _prf(lcs_length(c, r), len(c), len(r))


def rouge_all(candidate, reference):
    return {"ROUGE-1": rouge_n(candidate, reference, 1),
            "ROUGE-2": rouge_n(candidate, reference, 2),
            "ROUGE-L": rouge_l(candidate, reference)}
