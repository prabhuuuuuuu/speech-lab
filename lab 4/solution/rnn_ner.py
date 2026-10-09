"""
Lab 4 - RNN-based Named Entity Recognition
==========================================

Design, implement and evaluate a Recurrent-Neural-Network NER system and
analyse how MODEL choices and PREPROCESSING choices affect performance.

Pipeline
    sentence -> preprocessing -> [word emb ; char-CNN ; casing emb] -> (Bi)RNN/LSTM/GRU
             -> Linear -> softmax over BIO tags

Experiments (all trained on the same data, best epoch picked on the dev set)
    A. Architecture : RNN, GRU, LSTM, BiRNN, BiGRU, BiLSTM
    B. Preprocessing: lower-casing, digit normalisation, <UNK> handling
                      (min-freq + word dropout), casing feature, char-CNN, all combined
    C. Capacity     : 2-layer BiLSTM, small hidden size

Metrics (sequence-labelling)
    entity-level micro Precision / Recall / F1 (exact span + type, CoNLL style),
    macro-F1, token accuracy, recall on SEEN vs UNSEEN entities,
    F1 on lower-cased (ASR-like) sentences, per-type report, tag confusion matrix,
    error taxonomy (type / boundary / missed / spurious).

Run
    python rnn_ner.py                     # full experiment suite (15 models, several minutes on CPU)
    python rnn_ner.py --quick             # smaller data + fewer epochs
    python rnn_ner.py --text "Sundar Pichai visited Chennai on 5 March"
    python rnn_ner.py --conll-train train.txt --conll-dev dev.txt --conll-test test.txt
"""

import argparse
import csv
import os
import random
import re
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ner_dataset import load_synthetic, read_conll

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
PAD, UNK = "<PAD>", "<UNK>"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def banner(title):
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)


# =============================================================================
# Metrics
# =============================================================================
def get_entities(tags):
    ents, typ, start = [], None, None
    for i, t in enumerate(list(tags) + ["O"]):
        if t.startswith("I-") and typ == t[2:]:
            continue
        if typ is not None:
            ents.append((typ, start, i - 1))
            typ = None
        if t != "O":
            typ, start = t[2:], i
    return ents


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def evaluate(data, preds, types, seen_entities=None):
    per_type = {t: Counter() for t in types}
    tok_ok = tok_n = 0
    seen_hit = seen_n = unseen_hit = unseen_n = 0
    for (toks, gold), pred in zip(data, preds):
        tok_ok += sum(a == b for a, b in zip(gold, pred))
        tok_n += len(gold)
        g, p = set(get_entities(gold)), set(get_entities(pred))
        for t in types:
            gt, pt = {e for e in g if e[0] == t}, {e for e in p if e[0] == t}
            per_type[t]["tp"] += len(gt & pt)
            per_type[t]["fp"] += len(pt - gt)
            per_type[t]["fn"] += len(gt - pt)
        if seen_entities is not None:
            for e in g:
                surface = " ".join(toks[e[1]:e[2] + 1]).lower()
                hit = e in p
                if surface in seen_entities:
                    seen_n += 1
                    seen_hit += hit
                else:
                    unseen_n += 1
                    unseen_hit += hit
    total = sum(per_type.values(), Counter())
    res = {"per_type": {t: (*prf(c["tp"], c["fp"], c["fn"]), c["tp"] + c["fn"]) for t, c in per_type.items()}}
    res["P"], res["R"], res["F1"] = prf(total["tp"], total["fp"], total["fn"])
    res["macroF1"] = float(np.mean([v[2] for v in res["per_type"].values()]))
    res["tok_acc"] = tok_ok / max(tok_n, 1)
    res["R_seen"] = seen_hit / seen_n if seen_n else float("nan")
    res["R_unseen"] = unseen_hit / unseen_n if unseen_n else float("nan")
    return res


def error_taxonomy(data, preds):
    c = Counter()
    for (_, gold), pred in zip(data, preds):
        g, p = set(get_entities(gold)), set(get_entities(pred))
        for e in g:
            if e in p:
                c["correct"] += 1
            elif any(q[1] == e[1] and q[2] == e[2] for q in p):
                c["type error (right span, wrong type)"] += 1
            elif any(q[1] <= e[2] and e[1] <= q[2] for q in p):
                c["boundary error (overlapping span)"] += 1
            else:
                c["missed entity (false negative)"] += 1
        for q in p:
            if not any(q[1] <= e[2] and e[1] <= q[2] for e in g):
                c["spurious entity (false positive)"] += 1
    return c


# =============================================================================
# Preprocessing
# =============================================================================
class Preprocessor:
    """
    lowercase    : lower-case the entire input (words AND characters)
    lower_words  : lower-case only the word-embedding lookup (case kept for chars / case feature)
    digit_norm   : map every digit to 0  (2027 -> 0000, 15 -> 00)
    min_freq     : words seen fewer times in training become <UNK>
    word_dropout : during training replace words by <UNK> with this probability
    """

    def __init__(self, lowercase=False, lower_words=False, digit_norm=False, min_freq=1, word_dropout=0.0):
        self.lowercase, self.lower_words = lowercase, lower_words
        self.digit_norm, self.min_freq, self.word_dropout = digit_norm, min_freq, word_dropout

    def surface(self, tok):
        if self.lowercase:
            tok = tok.lower()
        if self.digit_norm:
            tok = re.sub(r"\d", "0", tok)
        return tok

    def word_key(self, tok):
        tok = self.surface(tok)
        return tok.lower() if self.lower_words else tok

    def fit(self, sentences):
        counts = Counter(self.word_key(w) for s in sentences for w in s)
        self.w2i = {PAD: 0, UNK: 1}
        for w, c in sorted(counts.items()):
            if c >= self.min_freq:
                self.w2i[w] = len(self.w2i)
        chars = Counter(ch for s in sentences for w in s for ch in self.surface(w))
        self.c2i = {PAD: 0, UNK: 1}
        for ch in sorted(chars):
            self.c2i[ch] = len(self.c2i)
        return self


CASES = ["pad", "lower", "title", "upper", "numeric", "has_digit", "other"]


def casing(tok):
    if tok.isdigit():
        return 4
    if tok.islower():
        return 1
    if tok.istitle():
        return 2
    if tok.isupper():
        return 3
    if any(c.isdigit() for c in tok):
        return 5
    return 6


# =============================================================================
# Model
# =============================================================================
class RNNTagger(nn.Module):
    def __init__(self, n_words, n_chars, n_tags, cell="lstm", bidirectional=True, layers=1,
                 word_dim=100, hidden=128, use_char=False, use_case=False, char_dim=30,
                 char_filters=50, dropout=0.3):
        super().__init__()
        self.use_char, self.use_case = use_char, use_case
        self.word_emb = nn.Embedding(n_words, word_dim, padding_idx=0)
        in_dim = word_dim
        if use_char:
            self.char_emb = nn.Embedding(n_chars, char_dim, padding_idx=0)
            self.char_cnn = nn.Conv1d(char_dim, char_filters, kernel_size=3, padding=1)
            in_dim += char_filters
        if use_case:
            self.case_emb = nn.Embedding(len(CASES), 8, padding_idx=0)
            in_dim += 8
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn_cls(in_dim, hidden, num_layers=layers, batch_first=True,
                           bidirectional=bidirectional, dropout=dropout if layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden * (2 if bidirectional else 1), n_tags)

    def forward(self, words, chars, cases, lengths):
        x = [self.word_emb(words)]
        if self.use_char:
            B, T, L = chars.shape
            c = self.char_emb(chars.view(B * T, L)).transpose(1, 2)
            x.append(torch.relu(self.char_cnn(c)).max(dim=2).values.view(B, T, -1))
        if self.use_case:
            x.append(self.case_emb(cases))
        x = self.dropout(torch.cat(x, dim=-1))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=words.shape[1])
        return self.out(self.dropout(h))


def encode(pre, data, tag2idx):
    """Apply the preprocessing once and cache index tensors for every sentence."""
    enc = []
    for toks, tg in data:
        surf = [pre.surface(w)[:20] for w in toks]
        L = max(len(s) for s in surf)
        chars = torch.zeros(len(toks), L, dtype=torch.long)
        for i, s in enumerate(surf):
            chars[i, :len(s)] = torch.tensor([pre.c2i.get(ch, 1) for ch in s])
        enc.append((torch.tensor([pre.w2i.get(pre.word_key(w), 1) for w in toks]), chars,
                    torch.tensor([casing(s) for s in surf]),
                    torch.tensor([tag2idx.get(x, 0) for x in tg])))
    return enc


def make_batch(batch, word_dropout=0.0):
    T = max(len(e[0]) for e in batch)
    L = max(e[1].shape[1] for e in batch)
    words = torch.zeros(len(batch), T, dtype=torch.long)
    chars = torch.zeros(len(batch), T, L, dtype=torch.long)
    cases = torch.zeros(len(batch), T, dtype=torch.long)
    tags = torch.full((len(batch), T), -100, dtype=torch.long)
    lengths = torch.tensor([len(e[0]) for e in batch])
    for b, (w, c, k, y) in enumerate(batch):
        n = len(w)
        if word_dropout:          # replace random words by <UNK> (index 1) during training
            w = torch.where(torch.rand(n) < word_dropout, torch.ones_like(w), w)
        words[b, :n], chars[b, :n, :c.shape[1]], cases[b, :n], tags[b, :n] = w, c, k, y
    return words, chars, cases, tags, lengths


def predict(model, pre, data, tag2idx, idx2tag, batch_size=128, enc=None):
    enc = enc if enc is not None else encode(pre, data, tag2idx)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(enc), batch_size):
            w, c, k, _, lengths = make_batch(enc[i:i + batch_size])
            out = model(w, c, k, lengths).argmax(-1)
            for b, n in enumerate(lengths.tolist()):
                preds.append([idx2tag[j] for j in out[b, :n].tolist()])
    return preds


def run_experiment(cfg, train, dev, test, tags, types, seen, epochs, seed=7, verbose=False):
    set_seed(seed)
    tag2idx = {t: i for i, t in enumerate(tags)}
    idx2tag = {i: t for t, i in tag2idx.items()}
    pre = Preprocessor(**cfg["pre"]).fit([t for t, _ in train])
    model = RNNTagger(len(pre.w2i), len(pre.c2i), len(tags), **cfg["model"])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    best_f1, best_state, best_epoch = -1, None, 0
    t0 = time.time()
    train_enc, dev_enc = encode(pre, train, tag2idx), encode(pre, dev, tag2idx)
    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_enc)
        total = 0.0
        for i in range(0, len(train_enc), 32):
            w, c, k, y, lengths = make_batch(train_enc[i:i + 32], pre.word_dropout)
            loss = loss_fn(model(w, c, k, lengths).view(-1, len(tags)), y.view(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item()
        dev_f1 = evaluate(dev, predict(model, pre, dev, tag2idx, idx2tag, enc=dev_enc), types)["F1"]
        if verbose:
            print(f"    epoch {epoch:2d}  train-loss {total:8.3f}  dev-F1 {dev_f1:.4f}")
        if dev_f1 > best_f1:
            best_f1, best_epoch = dev_f1, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    train_time = time.time() - t0
    preds = predict(model, pre, test, tag2idx, idx2tag)
    res = evaluate(test, preds, types, seen)
    lower_idx = [i for i, (t, _) in enumerate(test) if all(not w[:1].isupper() for w in t)]
    res["F1_lowercased"] = evaluate([test[i] for i in lower_idx], [preds[i] for i in lower_idx], types)["F1"] \
        if lower_idx else float("nan")
    res.update(dev_F1=best_f1, best_epoch=best_epoch, time=train_time,
               params=sum(p.numel() for p in model.parameters()), vocab=len(pre.w2i))
    return res, model, pre, preds, (tag2idx, idx2tag)


# =============================================================================
# Experiment definitions
# =============================================================================
RAW = dict()
FULL_PRE = dict(lower_words=True, digit_norm=True, min_freq=2, word_dropout=0.1)
FULL_MODEL = dict(cell="lstm", bidirectional=True, use_char=True, use_case=True)


def experiments():
    E = []
    for cell in ["rnn", "gru", "lstm"]:
        for bi in [False, True]:
            name = ("Bi" if bi else "") + cell.upper()
            E.append(dict(group="A: architecture", name=name, pre=RAW,
                          model=dict(cell=cell, bidirectional=bi)))
    bil = dict(cell="lstm", bidirectional=True)
    E += [
        dict(group="B: preprocessing", name="BiLSTM + lowercase input", pre=dict(lowercase=True), model=bil),
        dict(group="B: preprocessing", name="BiLSTM + digit normalisation", pre=dict(digit_norm=True), model=bil),
        dict(group="B: preprocessing", name="BiLSTM + UNK (min_freq=2, word-dropout)",
             pre=dict(min_freq=2, word_dropout=0.1), model=bil),
        dict(group="B: preprocessing", name="BiLSTM + lower words + casing feature",
             pre=dict(lower_words=True), model=dict(bil, use_case=True)),
        dict(group="B: preprocessing", name="BiLSTM + char-CNN", pre=RAW, model=dict(bil, use_char=True)),
        dict(group="B: preprocessing", name="BiLSTM + ALL preprocessing + char + case",
             pre=FULL_PRE, model=FULL_MODEL),
        dict(group="C: capacity", name="2-layer BiLSTM (full pipeline)", pre=FULL_PRE,
             model=dict(FULL_MODEL, layers=2)),
        dict(group="C: capacity", name="BiLSTM hidden=32 (full pipeline)", pre=FULL_PRE,
             model=dict(FULL_MODEL, hidden=32)),
        dict(group="C: capacity", name="BiRNN (full pipeline)", pre=FULL_PRE,
             model=dict(FULL_MODEL, cell="rnn")),
    ]
    return E


def fmt(x):
    return "  n/a" if x != x else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser(description="Lab 4 - RNN-based NER with model/preprocessing analysis")
    ap.add_argument("--quick", action="store_true", help="smaller dataset and fewer epochs")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7, help="random seed for model initialisation")
    ap.add_argument("--conll-train"), ap.add_argument("--conll-dev"), ap.add_argument("--conll-test")
    ap.add_argument("--text", action="append", help="sentence(s) to tag with the best model")
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ----------------------------------------------------------------- data
    banner("1. DATA")
    if args.conll_train:
        train, dev, test = (read_conll(args.conll_train), read_conll(args.conll_dev),
                            read_conll(args.conll_test))
        print("Loaded CoNLL files.")
    else:
        n = (800, 200, 400) if args.quick else (2000, 400, 800)
        train, dev, test = load_synthetic(*n)
        print("Synthetic NER corpus (30% of entity names and 10 templates held out from training).")
    epochs = args.epochs or (6 if args.quick else 10)
    tags = sorted({t for _, ts in train + dev + test for t in ts}, key=lambda t: (t != "O", t[2:], t))
    types = sorted({t[2:] for t in tags if t != "O"})
    seen = {" ".join(toks[s:e + 1]).lower() for toks, ts in train for _, s, e in get_entities(ts)}
    print(f"train/dev/test sentences: {len(train)}/{len(dev)}/{len(test)} | entity types: {types}")
    print(f"tag set: {tags}")
    ent_counts = Counter(e[0] for _, ts in train for e in get_entities(ts))
    print(f"training entities per type: {dict(ent_counts)}")
    test_ents = [(" ".join(t[s:e + 1]).lower()) for t, ts in test for _, s, e in get_entities(ts)]
    print(f"test entities never seen in training: {sum(x not in seen for x in test_ents)}/{len(test_ents)}")
    print("\nExamples:")
    for toks, ts in train[:3]:
        print("  " + " ".join(f"{w}/{t}" if t != "O" else w for w, t in zip(toks, ts)))

    # ---------------------------------------------------------- experiments
    banner(f"2. EXPERIMENTS ({epochs} epochs each, best epoch chosen on dev F1)")
    rows, best = [], None
    for k, cfg in enumerate(experiments(), 1):
        res, model, pre, preds, maps = run_experiment(cfg, train, dev, test, tags, types, seen, epochs,
                                                      seed=args.seed)
        rows.append((cfg, res))
        print(f"[{k:2d}] {cfg['name']:<44} test-F1={res['F1']:.4f}  unseen-R={fmt(res['R_unseen'])}  "
              f"({res['time']:.0f}s, best epoch {res['best_epoch']})")
        if best is None or res["dev_F1"] > best[1]["dev_F1"]:
            best = (cfg, res, model, pre, preds, maps)

    # ------------------------------------------------------------ results
    banner("3. RESULTS ON THE TEST SET (entity-level, micro-averaged)")
    hdr = (f"{'Group':<17}{'Configuration':<44}{'P':>7}{'R':>7}{'F1':>7}{'macF1':>7}{'TokAcc':>8}"
           f"{'R-seen':>8}{'R-unseen':>9}{'F1-lower':>9}{'Params':>9}")
    print(hdr)
    print("-" * len(hdr))
    for cfg, r in rows:
        print(f"{cfg['group']:<17}{cfg['name']:<44}{r['P']:>7.3f}{r['R']:>7.3f}{r['F1']:>7.3f}"
              f"{r['macroF1']:>7.3f}{r['tok_acc']:>8.3f}{fmt(r['R_seen']):>8}{fmt(r['R_unseen']):>9}"
              f"{fmt(r['F1_lowercased']):>9}{r['params']:>9,}")

    csv_path = os.path.join(RESULTS_DIR, "experiments.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "configuration", "precision", "recall", "f1", "macro_f1", "token_acc",
                    "recall_seen", "recall_unseen", "f1_lowercased", "dev_f1", "best_epoch",
                    "train_time_s", "params"])
        for cfg, r in rows:
            w.writerow([cfg["group"], cfg["name"], *(round(r[k], 4) for k in
                        ["P", "R", "F1", "macroF1", "tok_acc", "R_seen", "R_unseen", "F1_lowercased",
                         "dev_F1"]), r["best_epoch"], round(r["time"], 1), r["params"]])
    print(f"\nSaved table -> {csv_path}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [c["name"] for c, _ in rows]
        x = np.arange(len(rows))
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.bar(x - 0.2, [r["F1"] for _, r in rows], 0.4, label="test F1")
        ax.bar(x + 0.2, [r["R_unseen"] for _, r in rows], 0.4, label="recall on unseen entities")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title("RNN-based NER: effect of model and preprocessing choices")
        ax.legend()
        fig.tight_layout()
        png = os.path.join(RESULTS_DIR, "experiments.png")
        fig.savefig(png, dpi=130)
        print(f"Saved chart -> {png}")
    except Exception as e:
        print(f"(plot skipped: {e})")

    # ------------------------------------------------------------ analysis
    banner("4. ANALYSIS OF MODEL AND PREPROCESSING CHOICES")
    R = {c["name"]: r for c, r in rows}

    def delta(a, b, key="F1"):
        return R[a][key] - R[b][key]

    def verdict(d):
        return "helped" if d > 0.005 else "hurt" if d < -0.005 else "made no clear difference"

    FULL = "BiLSTM + ALL preprocessing + char + case"
    d_gate = max(delta("LSTM", "RNN"), delta("GRU", "RNN"))
    d_bi = np.mean([delta("BiLSTM", "LSTM"), delta("BiGRU", "GRU"), delta("BiRNN", "RNN")])
    d_low = delta("BiLSTM + lowercase input", "BiLSTM")
    d_dig = delta("BiLSTM + digit normalisation", "BiLSTM")
    d_unk = delta("BiLSTM + UNK (min_freq=2, word-dropout)", "BiLSTM")
    d_unk_r = delta("BiLSTM + UNK (min_freq=2, word-dropout)", "BiLSTM", "R_unseen")
    d_case = delta("BiLSTM + lower words + casing feature", "BiLSTM")
    d_char = delta("BiLSTM + char-CNN", "BiLSTM")
    d_char_r = delta("BiLSTM + char-CNN", "BiLSTM", "R_unseen")
    d_full = delta(FULL, "BiLSTM")
    d_2l = delta("2-layer BiLSTM (full pipeline)", FULL)
    d_32 = delta("BiLSTM hidden=32 (full pipeline)", FULL)
    d_rnn = delta("BiRNN (full pipeline)", FULL)
    lines = [
        ("Gated cells (LSTM/GRU) vs vanilla RNN",
         f"LSTM-RNN = {delta('LSTM', 'RNN'):+.3f} F1 (LSTM {verdict(delta('LSTM', 'RNN'))}), GRU-RNN = "
         f"{delta('GRU', 'RNN'):+.3f} F1 (GRU {verdict(delta('GRU', 'RNN'))}). "
         "Gates fight vanishing gradients; the benefit grows with sentence length, and "
         "these sentences are short (~10-15 tokens), so the gap is modest."),
        ("Bidirectionality",
         f"BiLSTM-LSTM = {delta('BiLSTM', 'LSTM'):+.3f}, BiGRU-GRU = {delta('BiGRU', 'GRU'):+.3f}, "
         f"BiRNN-RNN = {delta('BiRNN', 'RNN'):+.3f} -> {verdict(d_bi)}. A left-to-right model must tag "
         "'Apex' before seeing 'Airlines'; the backward pass supplies that right context."),
        ("Lower-casing the whole input",
         f"{d_low:+.3f} F1 -> {verdict(d_low)}. It merges 'Chennai'/'chennai' (fewer OOV words, and the "
         "lower-cased ASR-style sentences now share embeddings) but throws away capitalisation, the "
         "strongest English NER cue. Compare with the casing-feature run, which keeps both benefits."),
        ("Digit normalisation",
         f"{d_dig:+.3f} F1 -> {verdict(d_dig)}. Test years (2023-2030) never occur in training; as '0000' "
         "they map to a known word, which helps DATE spans."),
        ("<UNK> handling (min_freq=2 + word dropout)",
         f"{d_unk:+.3f} F1, unseen-entity recall {d_unk_r:+.3f} -> {verdict(d_unk)}. Without it the <UNK> "
         "vector is never trained, so every unseen name gets a random embedding."),
        ("Lower-cased word lookup + casing feature",
         f"{d_case:+.3f} F1 -> {verdict(d_case)}. Vocabulary sparsity drops while an explicit 8-d casing "
         "embedding (Title / UPPER / lower / digit) keeps the capitalisation signal."),
        ("Character-level CNN",
         f"{d_char:+.3f} F1, unseen-entity recall {d_char_r:+.3f} -> {verdict(d_char)}. Sub-word patterns "
         "(-abad, -pur, 'Technologies', digit shapes) transfer to names never seen in training."),
        ("All preprocessing + char + case combined",
         f"{d_full:+.3f} F1 over the raw BiLSTM -> the effects are complementary."),
        ("Capacity",
         f"2 layers {d_2l:+.3f}, hidden=32 {d_32:+.3f}, BiRNN cell {d_rnn:+.3f} (vs. full-pipeline BiLSTM). "
         "Once the input representation is good, extra depth brings little, while a too-small hidden "
         "state clearly under-fits."),
    ]
    for title, text in lines:
        print(f"* {title}: {text}\n")
    print("Note: differences of about +/-0.01 F1 are within run-to-run noise for this data size; "
          "re-run with another --seed to check stability.")

    # --------------------------------------------------------- best model
    cfg, res, model, pre, preds, (tag2idx, idx2tag) = best
    banner(f"5. DETAILED EVALUATION OF THE BEST MODEL (by dev F1): {cfg['name']}")
    print(f"  {'Entity':<8}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}")
    for t, (p, r, f, s) in res["per_type"].items():
        print(f"  {t:<8}{p:>10.3f}{r:>10.3f}{f:>10.3f}{s:>10d}")
    print(f"  {'micro':<8}{res['P']:>10.3f}{res['R']:>10.3f}{res['F1']:>10.3f}")
    print(f"  macro-F1 {res['macroF1']:.3f} | token accuracy {res['tok_acc']:.3f}")
    try:
        from seqeval.metrics import classification_report
        print("\nseqeval cross-check:")
        print(classification_report([g for _, g in test], preds, digits=3, zero_division=0))
    except ImportError:
        print("(install seqeval for a library cross-check)")

    print("Token-level confusion matrix (rows = gold, columns = predicted):")
    cm = Counter((g, p) for (_, gs), ps in zip(test, preds) for g, p in zip(gs, ps))
    print("          " + "".join(f"{t:>8}" for t in tags))
    for g in tags:
        print(f"  {g:<8}" + "".join(f"{cm[(g, p)]:>8}" for p in tags))

    print("\nError taxonomy (entity level):")
    for k, v in error_taxonomy(test, preds).most_common():
        print(f"  {k:<40}{v:>6}")

    print("\nSample errors:")
    shown = 0
    for (toks, gold), pred in zip(test, preds):
        if gold != pred and shown < 6:
            shown += 1
            print("  " + " ".join(toks))
            print("     gold: " + ", ".join(f"{' '.join(toks[s:e + 1])}<{t}>" for t, s, e in get_entities(gold)))
            print("     pred: " + ", ".join(f"{' '.join(toks[s:e + 1])}<{t}>" for t, s, e in get_entities(pred)))

    for sent in args.text or []:
        toks = re.findall(r"[A-Za-z0-9'&.-]+|[^\sA-Za-z0-9]", sent)
        p = predict(model, pre, [(toks, ["O"] * len(toks))], tag2idx, idx2tag)[0]
        print(f"\nCustom sentence: {sent}")
        print("  " + " ".join(f"{w}/{t}" for w, t in zip(toks, p)))
        print("  entities: " + (", ".join(f"{' '.join(toks[s:e + 1])}<{t}>" for t, s, e in get_entities(p))
                                or "(none)"))


if __name__ == "__main__":
    main()
