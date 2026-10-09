"""
Lab 3 - Named Entity Recognition for voice-assistant speech-to-text transcripts
===============================================================================

Entities: PERSON (PER), LOCATION (LOC), ORGANIZATION (ORG), DATE (DATE)

  1. Convert the sentences into BIO tags
  2. Create word-level features (CRF) / embeddings (LSTM)
  3. CRF-based NER model        (linear-chain CRF implemented from scratch in PyTorch)
  4. Viterbi decoding           (implemented from scratch in NumPy, verified by brute force)
  5. LSTM-based NER model       (BiLSTM + character-CNN + word embeddings, PyTorch)
  6. Test both models on unseen speech transcripts
  7. Compare predictions and evaluation metrics
  8. Analyse one case where the models disagree
  9. Evaluate with appropriate metrics (entity-level P/R/F1, token accuracy,
     per-type report, lower-cased "raw ASR" robustness test)

Run:
    python ner_crf_lstm.py
    python ner_crf_lstm.py --text "call Ravi from Infosys in Pune on Monday"
    python ner_crf_lstm.py --audio my_command.wav     (needs SpeechRecognition + internet)
"""

import argparse
import itertools
import random
import re
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction import DictVectorizer

from ner_data import (TAGS, ENTITY_TYPES, TRAIN_TRANSCRIPTS, TEST_TRANSCRIPTS,
                      load_dataset, strip_markup, tokenize)

try:  # make Windows consoles print UTF-8 safely
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SEED = 42
TAG2IDX = {t: i for i, t in enumerate(TAGS)}
IDX2TAG = {i: t for t, i in TAG2IDX.items()}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def banner(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# =============================================================================
# Evaluation helpers (entity-level, CoNLL-style exact span match)
# =============================================================================
def get_entities(tags):
    """Return a list of (type, start, end) spans from a BIO tag sequence."""
    entities, ent_type, start = [], None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        if tag.startswith("I-") and ent_type == tag[2:]:
            continue
        if ent_type is not None:
            entities.append((ent_type, start, i - 1))
            ent_type = None
        if tag != "O":           # B-X, or an I-X that starts a new entity
            ent_type, start = tag[2:], i
    return entities


def evaluate(gold_seqs, pred_seqs):
    stats = {t: Counter() for t in ENTITY_TYPES}
    tok_correct = tok_total = 0
    for gold, pred in zip(gold_seqs, pred_seqs):
        tok_correct += sum(g == p for g, p in zip(gold, pred))
        tok_total += len(gold)
        g_ents, p_ents = set(get_entities(gold)), set(get_entities(pred))
        for t in ENTITY_TYPES:
            g = {e for e in g_ents if e[0] == t}
            p = {e for e in p_ents if e[0] == t}
            stats[t]["tp"] += len(g & p)
            stats[t]["fp"] += len(p - g)
            stats[t]["fn"] += len(g - p)

    def prf(c):
        p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    report = {t: (*prf(stats[t]), stats[t]["tp"] + stats[t]["fn"]) for t in ENTITY_TYPES}
    total = sum(stats.values(), Counter())
    report["micro"] = (*prf(total), total["tp"] + total["fn"])
    macro = [np.mean([report[t][k] for t in ENTITY_TYPES]) for k in range(3)]
    report["macro"] = (*macro, total["tp"] + total["fn"])
    report["token_acc"] = tok_correct / tok_total
    return report


def print_report(name, rep):
    print(f"\n{name}")
    print(f"  {'Entity':<10}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}")
    for k in ENTITY_TYPES + ["micro", "macro"]:
        p, r, f, s = rep[k]
        label = k if k in ENTITY_TYPES else f"{k} avg"
        print(f"  {label:<10}{p:>10.3f}{r:>10.3f}{f:>10.3f}{s:>10d}")
    print(f"  Token accuracy: {rep['token_acc']:.3f}")


# =============================================================================
# Task 2a - Hand-crafted word-level features for the CRF
# =============================================================================
MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"}
WEEKDAYS = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
RELATIVE_DAYS = {"today", "tomorrow", "tonight", "yesterday", "weekend", "day"}
TITLES = {"dr", "mr", "mrs", "ms", "prof", "professor", "doctor", "sir", "madam"}
ORG_SUFFIXES = {"university", "hospital", "bank", "airlines", "airways", "services",
                "railways", "cinemas", "pharmacy", "power", "college", "institute"}
PREPOSITIONS = {"from", "to", "in", "at", "near", "on", "with", "for", "by", "through"}


def word_shape(word):
    shape = re.sub("[A-Z]", "X", word)
    shape = re.sub("[a-z]", "x", shape)
    shape = re.sub("[0-9]", "d", shape)
    return re.sub(r"(.)\1+", r"\1\1", shape)          # compress: Xxxxxx -> Xxx


def word2features(sent, i):
    w = sent[i]
    lw = w.lower()
    f = {
        "bias": 1.0,
        "w.lower": lw,
        "w.prefix3": lw[:3],
        "w.suffix2": lw[-2:],
        "w.suffix3": lw[-3:],
        "w.shape": word_shape(w),
        "w.istitle": w.istitle(),
        "w.isupper": w.isupper() and len(w) > 1,
        "w.isdigit": w.isdigit(),
        "w.is_month": lw in MONTHS,
        "w.is_weekday": lw in WEEKDAYS,
        "w.is_relative_day": lw in RELATIVE_DAYS,
        "w.is_org_suffix": lw in ORG_SUFFIXES,
        "w.is_ordinal": bool(re.fullmatch(r"\d+(st|nd|rd|th)", lw)),
    }
    for off in (-2, -1, 1, 2):
        j = i + off
        if 0 <= j < len(sent):
            o = sent[j]
            f[f"{off}:w.lower"] = o.lower()
            f[f"{off}:w.istitle"] = o.istitle()
            f[f"{off}:w.is_title_word"] = o.lower() in TITLES
            f[f"{off}:w.is_month"] = o.lower() in MONTHS
            f[f"{off}:w.isdigit"] = o.isdigit()
            f[f"{off}:w.is_prep"] = o.lower() in PREPOSITIONS
            f[f"{off}:w.is_org_suffix"] = o.lower() in ORG_SUFFIXES
        else:
            f[f"{off}:PAD"] = True
    if i == 0:
        f["BOS"] = True
    if i == len(sent) - 1:
        f["EOS"] = True
    return f


def sent2features(tokens):
    return [word2features(tokens, i) for i in range(len(tokens))]


# =============================================================================
# Task 3 - Linear-chain CRF (from scratch, PyTorch autograd for the gradients)
# =============================================================================
class LinearChainCRF(nn.Module):
    """
    score(x, y) = start[y1] + sum_t emit(x_t, y_t) + sum_t trans[y_{t-1}, y_t] + end[yT]
    emit(x_t, y) = W_y . f(x_t)  (log-linear over the sparse hand-crafted features)
    P(y | x)     = exp(score(x, y)) / Z(x)   ;   Z(x) from the forward algorithm
    """

    def __init__(self, n_features, n_tags):
        super().__init__()
        self.emit = nn.Linear(n_features, n_tags)
        self.transitions = nn.Parameter(torch.zeros(n_tags, n_tags))   # [from, to]
        self.start = nn.Parameter(torch.zeros(n_tags))
        self.end = nn.Parameter(torch.zeros(n_tags))

    def emissions(self, X):
        return self.emit(X)

    def path_score(self, em, tags, mask):
        """Score of the gold tag path (batched). em: B,T,K  tags/mask: B,T"""
        score = self.start[tags[:, 0]] + em[:, 0].gather(1, tags[:, :1]).squeeze(1)
        for t in range(1, tags.shape[1]):
            step = self.transitions[tags[:, t - 1], tags[:, t]] + em[:, t].gather(1, tags[:, t:t + 1]).squeeze(1)
            score = score + step * mask[:, t]
        last = tags.gather(1, (mask.sum(1).long() - 1).unsqueeze(1)).squeeze(1)
        return score + self.end[last]

    def log_partition(self, em, mask):
        """Forward algorithm in log space: log Z(x) = log sum_y exp(score(x, y))."""
        alpha = self.start + em[:, 0]                                    # B,K
        for t in range(1, em.shape[1]):
            nxt = torch.logsumexp(alpha.unsqueeze(2) + self.transitions + em[:, t].unsqueeze(1), dim=1)
            alpha = torch.where(mask[:, t].unsqueeze(1).bool(), nxt, alpha)
        return torch.logsumexp(alpha + self.end, dim=1)

    def nll(self, X, tags, mask):
        """Negative log-likelihood  -log P(y|x) = log Z(x) - score(x, y), summed over the batch."""
        em = self.emissions(X)
        return (self.log_partition(em, mask) - self.path_score(em, tags, mask)).sum()

    def numpy_params(self):
        return (self.transitions.detach().numpy(), self.start.detach().numpy(),
                self.end.detach().numpy())


def pad_crf_batch(X_list, y_list):
    B, T, F = len(X_list), max(len(x) for x in X_list), X_list[0].shape[1]
    X = torch.zeros(B, T, F)
    tags = torch.zeros(B, T, dtype=torch.long)
    mask = torch.zeros(B, T)
    for b, (x, y) in enumerate(zip(X_list, y_list)):
        X[b, :len(x)] = x
        tags[b, :len(y)] = y
        mask[b, :len(y)] = 1
    return X, tags, mask


def train_crf(X_train, y_train, n_features, c2=0.05, max_iter=150):
    set_seed()
    crf = LinearChainCRF(n_features, len(TAGS))
    X, tags, mask = pad_crf_batch(X_train, y_train)
    opt = torch.optim.LBFGS(crf.parameters(), lr=1.0, max_iter=max_iter,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = crf.nll(X, tags, mask)
        l2 = c2 * (crf.emit.weight.pow(2).sum() + crf.transitions.pow(2).sum())
        total = loss + l2
        total.backward()
        return total

    t0 = time.time()
    loss = opt.step(closure)
    print(f"CRF trained with L-BFGS (L2 c2={c2}) in {time.time() - t0:.1f}s, "
          f"final objective = {loss.item():.3f}")
    return crf


# =============================================================================
# Task 4 - Viterbi decoding (from scratch, NumPy)
# =============================================================================
def viterbi_decode(emissions, transitions, start, end):
    """
    delta[0, j]  = start[j] + emit[0, j]
    delta[t, j]  = max_i ( delta[t-1, i] + trans[i, j] ) + emit[t, j]
    psi[t, j]    = argmax_i ( ... )      (back-pointer)
    best_last    = argmax_j ( delta[T-1, j] + end[j] ),  then follow back-pointers.
    """
    T, K = emissions.shape
    delta = np.zeros((T, K))
    psi = np.zeros((T, K), dtype=int)
    delta[0] = start + emissions[0]
    for t in range(1, T):
        cand = delta[t - 1][:, None] + transitions          # K(from) x K(to)
        psi[t] = cand.argmax(axis=0)
        delta[t] = cand.max(axis=0) + emissions[t]
    final = delta[-1] + end
    best = [int(final.argmax())]
    for t in range(T - 1, 0, -1):
        best.append(int(psi[t, best[-1]]))
    best.reverse()
    return best, float(final.max()), delta, psi


def brute_force_best(emissions, transitions, start, end):
    """Exhaustively score all K^T tag paths (only feasible for short sentences)."""
    T, K = emissions.shape
    paths = np.array(list(itertools.product(range(K), repeat=T)))
    scores = start[paths[:, 0]] + end[paths[:, -1]] + emissions[np.arange(T), paths].sum(axis=1)
    if T > 1:
        scores += transitions[paths[:, :-1], paths[:, 1:]].sum(axis=1)
    k = int(scores.argmax())
    return list(paths[k]), float(scores[k])


def crf_predict(crf, X, decoder="viterbi"):
    trans, start, end = crf.numpy_params()
    with torch.no_grad():
        em = crf.emissions(X).numpy()
    if decoder == "greedy":                    # ignores transition scores
        return [IDX2TAG[i] for i in em.argmax(axis=1)]
    path, _, _, _ = viterbi_decode(em, trans, start, end)
    return [IDX2TAG[i] for i in path]


# =============================================================================
# Task 2b + 5 - Embeddings and the BiLSTM(+char-CNN) tagger
# =============================================================================
PAD, UNK = "<PAD>", "<UNK>"


class Vocab:
    def __init__(self, sentences):
        words = Counter(w.lower() for s in sentences for w in s)
        chars = Counter(c for s in sentences for w in s for c in w)
        self.w2i = {PAD: 0, UNK: 1}
        for w in sorted(words):
            self.w2i[w] = len(self.w2i)
        self.c2i = {PAD: 0, UNK: 1}
        for c in sorted(chars):
            self.c2i[c] = len(self.c2i)

    def word_ids(self, tokens):
        return [self.w2i.get(w.lower(), 1) for w in tokens]

    def char_ids(self, tokens, max_len=20):
        return [[self.c2i.get(c, 1) for c in w[:max_len]] for w in tokens]


class BiLSTMTagger(nn.Module):
    """
    token representation = [ word embedding (lower-cased word) ; char-CNN(word) ]
    -> BiLSTM -> Linear -> softmax over BIO tags (independent per token)
    """

    def __init__(self, n_words, n_chars, n_tags, word_dim=50, char_dim=25,
                 char_filters=30, hidden=64, dropout=0.3):
        super().__init__()
        self.word_emb = nn.Embedding(n_words, word_dim, padding_idx=0)
        self.char_emb = nn.Embedding(n_chars, char_dim, padding_idx=0)
        self.char_cnn = nn.Conv1d(char_dim, char_filters, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(word_dim + char_filters, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, n_tags)

    def token_representation(self, words, chars):
        B, T, L = chars.shape
        w = self.word_emb(words)                                   # B,T,Dw
        c = self.char_emb(chars.view(B * T, L)).transpose(1, 2)    # B*T,Dc,L
        c = torch.relu(self.char_cnn(c)).max(dim=2).values         # B*T,F
        return torch.cat([w, c.view(B, T, -1)], dim=-1)

    def forward(self, words, chars):
        x = self.dropout(self.token_representation(words, chars))
        h, _ = self.lstm(x)
        return self.out(self.dropout(h))


def make_batch(vocab, batch, word_dropout=0.0):
    T = max(len(toks) for toks, _ in batch)
    L = max(min(len(w), 20) for toks, _ in batch for w in toks)
    words = torch.zeros(len(batch), T, dtype=torch.long)
    chars = torch.zeros(len(batch), T, L, dtype=torch.long)
    tags = torch.full((len(batch), T), -100, dtype=torch.long)
    for b, (toks, tg) in enumerate(batch):
        ids = vocab.word_ids(toks)
        if word_dropout:   # randomly replace words by <UNK> so the model learns to use chars
            ids = [1 if random.random() < word_dropout else i for i in ids]
        words[b, :len(toks)] = torch.tensor(ids)
        for t, cid in enumerate(vocab.char_ids(toks)):
            chars[b, t, :len(cid)] = torch.tensor(cid)
        if tg is not None:
            tags[b, :len(tg)] = torch.tensor([TAG2IDX[x] for x in tg])
    return words, chars, tags


def train_lstm(train_data, vocab, epochs=60, batch_size=8, lr=0.005):
    set_seed()
    model = BiLSTMTagger(len(vocab.w2i), len(vocab.c2i), len(TAGS))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_data)
        total = 0.0
        for i in range(0, len(train_data), batch_size):
            words, chars, tags = make_batch(vocab, train_data[i:i + batch_size], word_dropout=0.15)
            logits = model(words, chars)
            loss = loss_fn(logits.view(-1, len(TAGS)), tags.view(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item()
        if epoch == 1 or epoch % 15 == 0:
            print(f"  epoch {epoch:3d}  loss = {total:.4f}")
    print(f"BiLSTM trained in {time.time() - t0:.1f}s")
    return model


def lstm_predict(model, vocab, tokens, return_probs=False):
    model.eval()
    with torch.no_grad():
        words, chars, _ = make_batch(vocab, [(tokens, None)])
        probs = torch.softmax(model(words, chars)[0], dim=-1).numpy()
    tags = [IDX2TAG[i] for i in probs.argmax(axis=1)]
    return (tags, probs) if return_probs else tags


# =============================================================================
# Pretty printing helpers
# =============================================================================
def entities_str(tokens, tags):
    ents = get_entities(tags)
    return ", ".join(f"{' '.join(tokens[s:e + 1])}<{t}>" for t, s, e in ents) or "(none)"


def print_aligned(tokens, rows):
    """rows: list of (label, tags)."""
    width = [max(len(tokens[i]), *(len(r[1][i]) for r in rows)) + 2 for i in range(len(tokens))]
    print("  " + f"{'Token':<7}" + "".join(f"{t:<{w}}" for t, w in zip(tokens, width)))
    for label, tags in rows:
        print("  " + f"{label:<7}" + "".join(f"{t:<{w}}" for t, w in zip(tags, width)))


def transcribe_audio(path):
    try:
        import speech_recognition as sr
    except ImportError:
        print("SpeechRecognition is not installed: pip install SpeechRecognition")
        return None
    r = sr.Recognizer()
    with sr.AudioFile(path) as src:
        audio = r.record(src)
    try:
        return r.recognize_google(audio)
    except Exception as e:  # network / recognition failure
        print(f"Speech recognition failed: {e}")
        return None


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Lab 3 - CRF vs LSTM NER for speech transcripts")
    ap.add_argument("--text", help="extra transcript to tag with both models")
    ap.add_argument("--audio", help="WAV/FLAC file to transcribe (Google STT) and tag")
    ap.add_argument("--epochs", type=int, default=60, help="BiLSTM epochs")
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    # ------------------------------------------------------------------ Task 1
    banner("TASK 1 - Convert the transcripts into BIO tags")
    train = load_dataset(TRAIN_TRANSCRIPTS)
    test = load_dataset(TEST_TRANSCRIPTS)
    print(f"Training transcripts: {len(train)} | unseen test transcripts: {len(test)}")
    print(f"Tag set ({len(TAGS)}): {TAGS}\n")
    for toks, tags in train[:3]:
        print("  " + " | ".join(f"{w}/{t}" for w, t in zip(toks, tags)))
    tag_counts = Counter(t for _, tags in train for t in tags)
    print("\nTag distribution in training data:", dict(sorted(tag_counts.items())))

    # ------------------------------------------------------------------ Task 2
    banner("TASK 2 - Word-level features (CRF) and embeddings (LSTM)")
    vec = DictVectorizer(sparse=False)
    train_feats = [sent2features(t) for t, _ in train]
    vec.fit([f for s in train_feats for f in s])
    to_tensor = lambda feats: torch.tensor(vec.transform(feats), dtype=torch.float32)
    X_train = [to_tensor(f) for f in train_feats]
    y_train = [torch.tensor([TAG2IDX[t] for t in tags]) for _, tags in train]
    print(f"CRF feature space: {len(vec.feature_names_)} sparse binary/real features")
    example = word2features(train[0][0], 5)
    print(f"Feature dictionary for the token '{train[0][0][5]}' in '{' '.join(train[0][0])}':")
    for k in list(example)[:16]:
        print(f"    {k:<22} = {example[k]}")
    print("    ...")

    vocab = Vocab([t for t, _ in train])
    print(f"\nLSTM vocabulary: {len(vocab.w2i)} words (+<PAD>,<UNK>), {len(vocab.c2i)} characters")
    print("LSTM token representation = word embedding (50-d, learned) + char-CNN embedding (30-d)")
    print("  -> the char-CNN lets the model build a vector for words never seen in training")

    # ------------------------------------------------------------------ Task 3
    banner("TASK 3 - CRF-based NER model (linear-chain CRF, from scratch)")
    crf = train_crf(X_train, y_train, len(vec.feature_names_))
    trans, start, end = crf.numpy_params()
    print("\nLearned transition scores (row = previous tag, column = next tag):")
    print("         " + "".join(f"{t:>8}" for t in TAGS))
    for i, t in enumerate(TAGS):
        print(f"{t:>8} " + "".join(f"{trans[i, j]:8.2f}" for j in range(len(TAGS))))
    W = crf.emit.weight.detach().numpy()
    names = vec.feature_names_
    print("\nTop-5 features per tag (highest emission weights):")
    for k, t in enumerate(TAGS):
        top = np.argsort(-W[k])[:5]
        print(f"  {t:<7}: " + ", ".join(f"{names[i]} ({W[k, i]:.2f})" for i in top))

    # ------------------------------------------------------------------ Task 4
    banner("TASK 4 - Viterbi decoding (best tag sequence)")
    demo_toks, demo_gold = test[0]
    with torch.no_grad():
        em = crf.emissions(to_tensor(sent2features(demo_toks))).numpy()
    path, best_score, delta, psi = viterbi_decode(em, trans, start, end)
    print("Sentence:", " ".join(demo_toks))
    print("\nViterbi trellis delta[t, tag] (best score of any path ending in tag at position t):")
    print(f"  {'token':<12}" + "".join(f"{t:>8}" for t in TAGS) + "   best")
    for t, tok in enumerate(demo_toks):
        print(f"  {tok:<12}" + "".join(f"{delta[t, j]:8.1f}" for j in range(len(TAGS)))
              + f"   {IDX2TAG[int(delta[t].argmax())]}")
    print("\nBack-tracked best path:", [IDX2TAG[i] for i in path])
    print(f"Best path score       : {best_score:.3f}")

    # verify Viterbi against exhaustive search on the shortest test sentence
    short_toks = min((t for t, _ in test), key=len)
    with torch.no_grad():
        em_s = crf.emissions(to_tensor(sent2features(short_toks))).numpy()
    if len(short_toks) <= 6:
        vp, vs, _, _ = viterbi_decode(em_s, trans, start, end)
        bp, bs = brute_force_best(em_s, trans, start, end)
        print(f"\nCorrectness check on '{' '.join(short_toks)}' "
              f"({len(TAGS)}^{len(short_toks)} = {len(TAGS) ** len(short_toks):,} paths):")
        print(f"  Viterbi    : {[IDX2TAG[i] for i in vp]}  score={vs:.4f}")
        print(f"  Brute force: {[IDX2TAG[i] for i in bp]}  score={bs:.4f}")
        print("  -> identical:", vp == bp, "| Viterbi cost O(T*K^2) instead of O(K^T)")

    # optional: compare against the sklearn-crfsuite library CRF (if installed)
    crfsuite = None
    try:
        import sklearn_crfsuite
        crfsuite = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.05, c2=0.05,
                                        max_iterations=200, all_possible_transitions=True)
        crfsuite.fit(train_feats, [tags for _, tags in train])
        print("\n(sklearn-crfsuite reference CRF also trained for comparison)")
    except ImportError:
        print("\n(sklearn-crfsuite not installed - skipping library CRF reference)")

    # ------------------------------------------------------------------ Task 5
    banner("TASK 5 - LSTM-based NER model (BiLSTM + char-CNN)")
    lstm = train_lstm(list(train), vocab, epochs=args.epochs)
    n_params = sum(p.numel() for p in lstm.parameters())
    print(f"Trainable parameters: {n_params:,}")

    # ------------------------------------------------------------------ Task 6
    banner("TASK 6 - Test both models on UNSEEN speech transcripts")
    gold = [tags for _, tags in test]
    crf_pred = [crf_predict(crf, to_tensor(sent2features(t))) for t, _ in test]
    greedy_pred = [crf_predict(crf, to_tensor(sent2features(t)), "greedy") for t, _ in test]
    lstm_pred = [lstm_predict(lstm, vocab, t) for t, _ in test]
    train_words = {w.lower() for t, _ in train for w in t}
    for k, (toks, g) in enumerate(test):
        oov = [w for w in toks if w.lower() not in train_words]
        print(f"\n[{k + 1:02d}] {' '.join(toks)}")
        print(f"     OOV words: {oov if oov else 'none'}")
        print(f"     GOLD : {entities_str(toks, g)}")
        print(f"     CRF  : {entities_str(toks, crf_pred[k])}")
        print(f"     LSTM : {entities_str(toks, lstm_pred[k])}")

    # ------------------------------------------------------------------ Task 7
    banner("TASK 7 - Compare predictions and evaluation metrics")
    rep_crf = evaluate(gold, crf_pred)
    rep_lstm = evaluate(gold, lstm_pred)
    rep_greedy = evaluate(gold, greedy_pred)
    agree = sum(a == b for a, b in zip(crf_pred, lstm_pred))
    print(f"Transcripts where CRF and LSTM agree exactly : {agree}/{len(test)}")
    print(f"Tokens where CRF and LSTM agree             : "
          f"{sum(x == y for a, b in zip(crf_pred, lstm_pred) for x, y in zip(a, b))}/"
          f"{sum(len(a) for a in gold)}")
    print(f"\n  {'Model':<34}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Tok-Acc':>10}")
    rows = [("CRF (Viterbi decoding)", rep_crf), ("CRF (greedy, no transitions)", rep_greedy),
            ("BiLSTM + char-CNN (softmax)", rep_lstm)]
    if crfsuite is not None:
        rows.insert(1, ("sklearn-crfsuite (reference)",
                        evaluate(gold, crfsuite.predict([sent2features(t) for t, _ in test]))))
    for name, r in rows:
        p, rc, f, _ = r["micro"]
        print(f"  {name:<34}{p:>10.3f}{rc:>10.3f}{f:>10.3f}{r['token_acc']:>10.3f}")

    def invalid_bio(seqs):
        return sum(1 for s in seqs for a, b in zip(["O"] + s[:-1], s)
                   if b.startswith("I-") and a[2:] != b[2:])
    print(f"\nInvalid BIO transitions (e.g. O -> I-PER): CRF-Viterbi={invalid_bio(crf_pred)}, "
          f"CRF-greedy={invalid_bio(greedy_pred)}, LSTM={invalid_bio(lstm_pred)}")

    # ------------------------------------------------------------------ Task 8
    banner("TASK 8 - Analysis of a case where the models disagree")
    diff_idx = [k for k in range(len(test)) if crf_pred[k] != lstm_pred[k]]
    if not diff_idx:
        print("Both models produced identical tag sequences on every test transcript.")
    else:
        # prefer a case where exactly one of the models is right
        def n_err(p, g):
            return sum(a != b for a, b in zip(p, g))
        k = max(diff_idx, key=lambda i: abs(n_err(crf_pred[i], gold[i]) - n_err(lstm_pred[i], gold[i])))
        toks = test[k][0]
        print(f"Transcript: {' '.join(toks)}\n")
        print_aligned(toks, [("GOLD", gold[k]), ("CRF", crf_pred[k]), ("LSTM", lstm_pred[k])])
        _, probs = lstm_predict(lstm, vocab, toks, return_probs=True)
        with torch.no_grad():
            x = to_tensor(sent2features(toks)).numpy()
        print()
        for i, tok in enumerate(toks):
            if crf_pred[k][i] == lstm_pred[k][i]:
                continue
            winner = ("CRF" if crf_pred[k][i] == gold[k][i] else
                      "LSTM" if lstm_pred[k][i] == gold[k][i] else "neither")
            print(f"Token '{tok}': gold={gold[k][i]}  CRF={crf_pred[k][i]}  "
                  f"LSTM={lstm_pred[k][i]}  -> correct: {winner}")
            print(f"  In training vocabulary: {tok.lower() in train_words}"
                  f"{'' if tok.lower() in train_words else '  (LSTM sees <UNK> + char-CNN only)'}")
            ci = TAG2IDX[crf_pred[k][i]]
            contrib = W[ci] * x[i]
            top = np.argsort(-contrib)[:5]
            print(f"  CRF evidence for {crf_pred[k][i]}: "
                  + ", ".join(f"{names[j]} ({contrib[j]:+.2f})" for j in top if contrib[j] != 0))
            top_lstm = np.argsort(-probs[i])[:3]
            print("  LSTM softmax: " + ", ".join(f"{IDX2TAG[j]}={probs[i, j]:.2f}" for j in top_lstm))
        print("""
Interpretation:
  * The CRF scores each tag with explicit, human-designed evidence (capitalisation,
    word shape, suffixes, neighbouring words such as 'Dr', 'from', 'in', 'on', month
    names) and Viterbi picks the globally best *sequence*; the learned transition
    scores strongly penalise invalid moves such as O -> I-PER.
  * The BiLSTM must learn everything from 60 sentences. Unseen names are mapped to
    <UNK>, so it relies on the char-CNN and context; its softmax decides each token
    independently, so it can emit inconsistent BIO sequences or confuse ORG/LOC when
    the context ('at', 'in', 'from') is shared by both types.
  * With so little training data, feature engineering + structured decoding (CRF)
    usually generalises better; with thousands of sentences (see Lab 4) the neural
    model catches up and typically overtakes the CRF.""")

    # ------------------------------------------------------------------ Task 9
    banner("TASK 9 - Evaluation with appropriate metrics")
    print("Metric: entity-level exact-match Precision / Recall / F1 (CoNLL-2003 style).")
    print("An entity counts as correct only if BOTH its boundaries and its type match.")
    print_report("CRF (Viterbi) on unseen transcripts", rep_crf)
    print_report("BiLSTM + char-CNN on unseen transcripts", rep_lstm)

    try:
        from seqeval.metrics import classification_report as seqeval_report
        print("\nCross-check with the seqeval library (CRF):")
        print(seqeval_report(gold, crf_pred, digits=3, zero_division=0))
        print("Cross-check with the seqeval library (LSTM):")
        print(seqeval_report(gold, lstm_pred, digits=3, zero_division=0))
    except ImportError:
        print("\n(seqeval not installed - skipping library cross-check)")

    # robustness: raw ASR output without true-casing
    lower_test = [([w.lower() for w in t], g) for t, g in test]
    crf_low = [crf_predict(crf, to_tensor(sent2features(t))) for t, _ in lower_test]
    lstm_low = [lstm_predict(lstm, vocab, t) for t, _ in lower_test]
    r1, r2 = evaluate(gold, crf_low), evaluate(gold, lstm_low)
    print("Robustness test - the same unseen transcripts, lower-cased (ASR without true-casing):")
    print(f"  CRF  micro-F1: {rep_crf['micro'][2]:.3f} (cased) -> {r1['micro'][2]:.3f} (lower-cased)")
    print(f"  LSTM micro-F1: {rep_lstm['micro'][2]:.3f} (cased) -> {r2['micro'][2]:.3f} (lower-cased)")
    drop_crf = rep_crf["micro"][2] - r1["micro"][2]
    drop_lstm = rep_lstm["micro"][2] - r2["micro"][2]
    more, less = ("CRF", "LSTM") if drop_crf > drop_lstm else ("LSTM", "CRF")
    print(f"  The {more} loses more F1: its strongest features/cues are capitalisation-based")
    print(f"  (w.istitle, w.shape=Xxx). The {less} is less affected"
          + (" because its word embeddings are built from lower-cased words and it relies more on context."
             if less == "LSTM" else "."))
    print("  Real ASR pipelines should true-case the transcript first, or train on lower-cased data.")

    # ------------------------------------------------------------- user input
    extra = []
    if args.audio:
        txt = transcribe_audio(args.audio)
        if txt:
            print(f"\nTranscribed audio: {txt}")
            extra.append(txt)
    if args.text:
        extra.append(args.text)
    for txt in extra:
        toks = tokenize(strip_markup(txt))
        banner(f"Custom transcript: {' '.join(toks)}")
        c = crf_predict(crf, to_tensor(sent2features(toks)))
        l = lstm_predict(lstm, vocab, toks)
        print_aligned(toks, [("CRF", c), ("LSTM", l)])
        print(f"\n  CRF entities : {entities_str(toks, c)}")
        print(f"  LSTM entities: {entities_str(toks, l)}")


if __name__ == "__main__":
    main()
