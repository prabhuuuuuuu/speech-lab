"""
Lab 5 (b) - Graph / statistical extractive summarisers (implemented from scratch).

Every summariser takes the pre-processed Document and the number of sentences k
and returns (selected_indices_in_document_order, score_per_sentence).

    1. Frequency-based     score(S) = sum of normalised content-word frequencies
    2. TF-IDF-based        score(S) = sum of TF-IDF weights of its words (sentence = "document")
    3. TextRank            PageRank on a sentence graph, edge = word-overlap similarity
                           sim(Si,Sj) = |Si n Sj| / (log|Si| + log|Sj|)          (Mihalcea & Tarau 2004)
    4. LexRank             eigenvector centrality on a thresholded idf-modified-cosine graph (Erkan & Radev 2004)
    5. Feature-based       weighted sum of position, length, frequency, TF-IDF, heading/keyword and
                           numeric-data features
"""

import math
from collections import Counter

import numpy as np

from preprocess import content_words

DOMAIN_KEYWORDS = {"teaching", "quality", "infrastructure", "laboratory", "lab", "labs", "faculty",
                   "satisfaction", "satisfied", "overall", "facilities"}


def _top_k(scores, k):
    k = min(k, len(scores))
    return sorted(np.argsort(-np.asarray(scores), kind="stable")[:k].tolist())


def _normalise(x):
    x = np.asarray(x, dtype=float)
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 0 else np.ones_like(x)


# --------------------------------------------------------------------------- 1
def word_frequencies(sentences):
    freq = Counter(w for s in sentences for w in content_words(s))
    top = max(freq.values())
    return {w: c / top for w, c in freq.items()}          # normalised to [0, 1]


def frequency_summary(doc, k):
    sents = doc.sentences
    freq = word_frequencies(sents)
    scores = [sum(freq[w] for w in content_words(s)) for s in sents]
    return _top_k(scores, k), np.array(scores)


# --------------------------------------------------------------------------- 2
def tfidf_matrix(sentences):
    """TF(t,s) = count/len(s); IDF(t) = log(N / df(t)); each sentence is one 'document'."""
    toks = [content_words(s) for s in sentences]
    vocab = sorted({w for t in toks for w in t})
    index = {w: i for i, w in enumerate(vocab)}
    N = len(sentences)
    df = Counter(w for t in toks for w in set(t))
    idf = np.array([math.log(N / df[w]) for w in vocab])
    tf = np.zeros((N, len(vocab)))
    for i, t in enumerate(toks):
        for w, c in Counter(t).items():
            tf[i, index[w]] = c / len(t)
    return tf * idf, vocab, idf


def tfidf_summary(doc, k):
    m, _, _ = tfidf_matrix(doc.sentences)
    scores = m.sum(axis=1)
    return _top_k(scores, k), scores


# --------------------------------------------------------------------------- 3
def pagerank(W, d=0.85, tol=1e-6, max_iter=200):
    """Weighted PageRank: S(Vi) = (1-d) + d * sum_j  w_ji / sum_k w_jk * S(Vj)."""
    n = len(W)
    row = W.sum(axis=1, keepdims=True)
    P = np.divide(W, row, out=np.zeros_like(W), where=row > 0)       # row-stochastic
    s = np.ones(n)
    for it in range(1, max_iter + 1):
        new = (1 - d) + d * P.T @ s
        if np.abs(new - s).sum() < tol:
            return new, it
        s = new
    return s, max_iter


def textrank_similarity(sentences):
    toks = [set(content_words(s)) for s in sentences]
    n = len(sentences)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and len(toks[i]) > 1 and len(toks[j]) > 1:
                W[i, j] = len(toks[i] & toks[j]) / (math.log(len(toks[i])) + math.log(len(toks[j])))
    return W


def textrank_summary(doc, k, d=0.85):
    W = textrank_similarity(doc.sentences)
    scores, iters = pagerank(W, d)
    textrank_summary.iterations = iters
    return _top_k(scores, k), scores


# --------------------------------------------------------------------------- 4
def lexrank_summary(doc, k, threshold=0.05, d=0.85, tol=1e-8):
    """
    Continuous LexRank: edge weight = idf-modified cosine similarity (kept only if > threshold),
    row-normalised into a Markov matrix; the stationary distribution (power method with
    damping d) is the centrality score of every sentence.
    """
    m, _, _ = tfidf_matrix(doc.sentences)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    unit = np.divide(m, norms, out=np.zeros_like(m), where=norms > 0)
    cos = unit @ unit.T                                      # idf-modified cosine
    np.fill_diagonal(cos, 0.0)
    A = np.where(cos > threshold, cos, 0.0)
    n = len(A)
    row = A.sum(axis=1, keepdims=True)
    B = np.where(row > 0, A / np.where(row > 0, row, 1), 1.0 / n)   # isolated node -> uniform jump
    p = np.ones(n) / n
    for _ in range(1000):                                    # power method with damping
        new = (1 - d) / n + d * B.T @ p
        if np.abs(new - p).sum() < tol:
            break
        p = new
    lexrank_summary.density = float((A > 0).sum() / (n * (n - 1)))
    return _top_k(p, k), p


# --------------------------------------------------------------------------- 5
FEATURE_WEIGHTS = {"position": 0.25, "frequency": 0.20, "tfidf": 0.25, "length": 0.10,
                   "keyword": 0.15, "numeric": 0.05}


def feature_table(doc):
    sents = doc.sentences
    meta = doc.sentence_meta
    _, freq = frequency_summary(doc, 1)
    _, tfidf = tfidf_summary(doc, 1)
    lengths = np.array([len(content_words(s)) for s in sents], dtype=float)
    pos = np.array([1.0 - si / plen for _, si, plen in meta])         # first sentence of a paragraph = 1
    heading_words = [set(content_words(doc.headings[pi])) for pi, _, _ in meta]
    keyword = np.array([len(set(content_words(s)) & (DOMAIN_KEYWORDS | hw))
                        for s, hw in zip(sents, heading_words)], dtype=float)
    numeric = np.array([any(c.isdigit() for c in s) for s in sents], dtype=float)
    length = np.minimum(lengths / np.median(lengths), 1.0)          # penalise only short sentences
    return {"position": pos, "frequency": _normalise(freq), "tfidf": _normalise(tfidf),
            "length": length, "keyword": _normalise(keyword), "numeric": numeric}


def feature_summary(doc, k, weights=FEATURE_WEIGHTS):
    F = feature_table(doc)
    scores = sum(w * F[name] for name, w in weights.items())
    feature_summary.features = F
    return _top_k(scores, k), scores


STATISTICAL_METHODS = {
    "Frequency-based": frequency_summary,
    "TF-IDF": tfidf_summary,
    "TextRank": textrank_summary,
    "LexRank": lexrank_summary,
    "Feature-based scoring": feature_summary,
}
