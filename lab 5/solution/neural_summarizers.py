"""
Lab 5 (c) 1-5 - Deep-learning summarisers trained FROM SCRATCH (PyTorch).

  Extractive (score every sentence, pick the top-k):
    RNN extractor            hierarchical vanilla RNN : words -> sentence vector -> document RNN
    LSTM extractor           same architecture with LSTM cells
    Self-attention extractor scaled dot-product multi-head self-attention, implemented by hand,
                             over words (sentence encoder) and over sentences (document encoder)

  Abstractive (generate a new sentence for every paragraph):
    RNN encoder-decoder      vanilla RNN, context = final encoder state only
    LSTM encoder-decoder     LSTM, context = final encoder state only (information bottleneck)
    Seq2Seq + attention      BiLSTM encoder + LSTM decoder with Bahdanau (additive) attention
    Transformer              encoder-decoder Transformer (sinusoidal positions, masked decoder
                             self-attention, cross-attention), trained from scratch

All models are trained on the synthetic in-domain corpus from feedback_corpus.py.
"""

import math
import random
import re
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from feedback_corpus import generate_documents

PAD, UNK, BOS, EOS = 0, 1, 2, 3


def ntokenize(text):
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*|[.,!?%]", text.lower())


class Vocab:
    def __init__(self, texts, min_freq=1):
        c = Counter(t for x in texts for t in ntokenize(x))
        self.itos = ["<pad>", "<unk>", "<bos>", "<eos>"] + sorted(w for w, n in c.items() if n >= min_freq)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, text, max_len=None):
        ids = [self.stoi.get(t, UNK) for t in ntokenize(text)]
        return ids[:max_len] if max_len else ids

    def decode(self, ids):
        words = []
        for i in ids:
            if i == EOS:
                break
            if i not in (PAD, BOS):
                words.append(self.itos[i])
        return detokenize(words)

    def __len__(self):
        return len(self.itos)


def detokenize(words):
    text = " ".join(words)
    text = re.sub(r"\s+([.,!?%])", r"\1", text).strip()
    return text[:1].upper() + text[1:] if text else text


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pad_2d(seqs, min_len=1):
    L = max(max(len(s) for s in seqs), min_len)
    out = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out, torch.tensor([max(len(s), 1) for s in seqs])


def sinusoidal_encoding(max_len, d):
    pe = torch.zeros(max_len, d)
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# =============================================================================
# Extractive models
# =============================================================================
class HierarchicalRNNExtractor(nn.Module):
    """word-level BiRNN/BiLSTM -> mean-pooled sentence vectors -> sentence-level BiRNN/BiLSTM -> score"""

    def __init__(self, vocab_size, cell="lstm", emb=64, hid=64):
        super().__init__()
        R = nn.LSTM if cell == "lstm" else nn.RNN
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.word_rnn = R(emb, hid, batch_first=True, bidirectional=True)
        self.sent_rnn = R(2 * hid, hid, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.2)
        self.score = nn.Linear(4 * hid, 1)

    def forward(self, sents, lengths):                          # sents: S x L (one document)
        x = self.drop(self.emb(sents))
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.word_rnn(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=sents.shape[1])
        mask = (sents != PAD).unsqueeze(-1).float()
        sent_vec = (h * mask).sum(1) / mask.sum(1).clamp(min=1)  # S x 2H
        d, _ = self.sent_rnn(sent_vec.unsqueeze(0))              # 1 x S x 2H (document context)
        return self.score(self.drop(torch.cat([d[0], sent_vec], -1))).squeeze(-1)


class MultiHeadSelfAttention(nn.Module):
    """Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V, computed for several heads (from scratch)."""

    def __init__(self, d_model, heads):
        super().__init__()
        assert d_model % heads == 0
        self.h, self.dk = heads, d_model // heads
        self.Wq, self.Wk = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.Wv, self.Wo = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)

    def forward(self, x, pad_mask=None):                        # x: B x T x D, pad_mask: B x T (True = pad)
        B, T, _ = x.shape
        split = lambda t: t.view(B, T, self.h, self.dk).transpose(1, 2)        # B x h x T x dk
        Q, K, V = split(self.Wq(x)), split(self.Wk(x)), split(self.Wv(x))
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.dk)                  # B x h x T x T
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask[:, None, None, :], -1e9)
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, T, -1)
        return self.Wo(out), attn


class SelfAttentionBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d, heads)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(0.1)

    def forward(self, x, pad_mask=None):
        a, w = self.attn(x, pad_mask)
        x = self.n1(x + self.drop(a))                     # Add & Norm
        return self.n2(x + self.drop(self.ff(x))), w      # Feed-forward, Add & Norm


class SelfAttentionExtractor(nn.Module):
    def __init__(self, vocab_size, d=64, heads=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d, padding_idx=PAD)
        self.register_buffer("pe", sinusoidal_encoding(256, d))
        self.word_block = SelfAttentionBlock(d, heads)
        self.pool_query = nn.Linear(d, 1)                 # attention pooling -> sentence vector
        self.sent_block = SelfAttentionBlock(d, heads)
        self.score = nn.Linear(2 * d, 1)

    def forward(self, sents, lengths=None, return_attention=False):
        S, L = sents.shape
        pad = sents == PAD
        x = self.emb(sents) * math.sqrt(self.emb.embedding_dim) + self.pe[:L]
        x, word_attn = self.word_block(x, pad)
        pool = self.pool_query(x).squeeze(-1).masked_fill(pad, -1e9).softmax(-1)    # S x L
        sent_vec = (pool.unsqueeze(-1) * x).sum(1)                                   # S x d
        y, sent_attn = self.sent_block((sent_vec + self.pe[:S]).unsqueeze(0))
        logits = self.score(torch.cat([y[0], sent_vec], -1)).squeeze(-1)
        if return_attention:
            return logits, pool, sent_attn[0].mean(0)
        return logits


def train_extractor(model, docs, vocab, epochs=8, lr=2e-3, seed=0, log=print):
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    data = []
    for doc in docs:
        sents = [s for p in doc for s in p["sentences"]]
        labels = [l for p in doc for l in p["labels"]]
        data.append((pad_2d([vocab.encode(s, 40) for s in sents]), torch.tensor(labels, dtype=torch.float)))
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        random.shuffle(data)
        total = 0.0
        for (x, lengths), y in data:
            loss = loss_fn(model(x, lengths), y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item()
        if ep in (1, epochs) or ep % 4 == 0:
            log(f"      epoch {ep:2d}  loss {total / len(data):.4f}")
    log(f"      trained in {time.time() - t0:.1f}s")
    return model


def extract(model, sentences, vocab, k):
    model.eval()
    with torch.no_grad():
        x, lengths = pad_2d([vocab.encode(s, 40) for s in sentences])
        scores = torch.sigmoid(model(x, lengths)).numpy()
    return sorted(np.argsort(-scores)[:k].tolist()), scores


# =============================================================================
# Abstractive models
# =============================================================================
class Seq2Seq(nn.Module):
    """
    Bidirectional encoder (RNN or LSTM) -> decoder (same cell).
      attention=False : decoder only sees the final encoder state (fixed-size context vector)
      attention=True  : Bahdanau attention  e_ti = v^T tanh(W_s s_{t-1} + W_h h_i),
                        a_t = softmax(e_t),  c_t = sum_i a_ti h_i
    """

    def __init__(self, vocab_size, cell="lstm", attention=False, emb=96, hid=128):
        super().__init__()
        self.cell, self.attention = cell, attention
        R = nn.LSTM if cell == "lstm" else nn.RNN
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.encoder = R(emb, hid, batch_first=True, bidirectional=True)
        self.bridge_h = nn.Linear(2 * hid, hid)
        self.bridge_c = nn.Linear(2 * hid, hid)
        self.decoder = R(emb + (2 * hid if attention else 0), hid, batch_first=True)
        if attention:
            self.W_h, self.W_s = nn.Linear(2 * hid, hid, bias=False), nn.Linear(hid, hid)
            self.v = nn.Linear(hid, 1, bias=False)
        self.out = nn.Linear(hid + (2 * hid if attention else 0), vocab_size)
        self.drop = nn.Dropout(0.2)

    def encode(self, src, lengths):
        x = self.drop(self.emb(src))
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        H, state = self.encoder(packed)
        H, _ = pad_packed_sequence(H, batch_first=True, total_length=src.shape[1])
        h = state[0] if self.cell == "lstm" else state
        h0 = torch.tanh(self.bridge_h(torch.cat([h[0], h[1]], -1))).unsqueeze(0)
        if self.cell == "lstm":
            c = state[1]
            return H, (h0, torch.tanh(self.bridge_c(torch.cat([c[0], c[1]], -1))).unsqueeze(0))
        return H, h0

    def step(self, y_prev, state, H, src_mask):
        e = self.drop(self.emb(y_prev)).unsqueeze(1)                      # B x 1 x E
        attn = None
        if self.attention:
            s = (state[0] if self.cell == "lstm" else state)[-1]          # B x hid
            scores = self.v(torch.tanh(self.W_h(H) + self.W_s(s).unsqueeze(1))).squeeze(-1)
            attn = scores.masked_fill(~src_mask, -1e9).softmax(-1)        # B x T
            ctx = (attn.unsqueeze(-1) * H).sum(1, keepdim=True)           # B x 1 x 2H
            e = torch.cat([e, ctx], -1)
        o, state = self.decoder(e, state)
        o = o.squeeze(1)
        if self.attention:
            o = torch.cat([o, ctx.squeeze(1)], -1)
        return self.out(self.drop(o)), state, attn

    def forward(self, src, lengths, tgt_in):
        H, state = self.encode(src, lengths)
        mask = src != PAD
        logits = []
        for t in range(tgt_in.shape[1]):                                  # teacher forcing
            lo, state, _ = self.step(tgt_in[:, t], state, H, mask)
            logits.append(lo)
        return torch.stack(logits, 1)

    @torch.no_grad()
    def generate(self, src, lengths, max_len=25):
        self.eval()
        H, state = self.encode(src, lengths)
        mask = src != PAD
        y = torch.full((src.shape[0],), BOS, dtype=torch.long)
        out, attns = [], []
        for _ in range(max_len):
            lo, state, a = self.step(y, state, H, mask)
            y = lo.argmax(-1)
            out.append(y)
            attns.append(a)
            if (y == EOS).all():
                break
        return torch.stack(out, 1), attns


def causal_mask(n):
    """True above the diagonal = position t may not attend to future positions > t."""
    return torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)


class TransformerSummarizer(nn.Module):
    def __init__(self, vocab_size, d=128, heads=4, layers=2, ff=256):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(vocab_size, d, padding_idx=PAD)
        self.register_buffer("pe", sinusoidal_encoding(512, d))
        self.tf = nn.Transformer(d_model=d, nhead=heads, num_encoder_layers=layers,
                                 num_decoder_layers=layers, dim_feedforward=ff, dropout=0.1,
                                 batch_first=True)
        self.out = nn.Linear(d, vocab_size)

    def embed(self, x):
        return self.emb(x) * math.sqrt(self.d) + self.pe[:x.shape[1]]

    def forward(self, src, lengths, tgt_in):
        causal = causal_mask(tgt_in.shape[1])                    # masked (look-ahead) self-attention
        h = self.tf(self.embed(src), self.embed(tgt_in), tgt_mask=causal,
                    src_key_padding_mask=src == PAD, tgt_key_padding_mask=tgt_in == PAD,
                    memory_key_padding_mask=src == PAD)
        return self.out(h)

    @torch.no_grad()
    def generate(self, src, lengths, max_len=25):
        self.eval()
        memory = self.tf.encoder(self.embed(src), src_key_padding_mask=src == PAD)
        y = torch.full((src.shape[0], 1), BOS, dtype=torch.long)
        for _ in range(max_len):
            causal = causal_mask(y.shape[1])
            h = self.tf.decoder(self.embed(y), memory, tgt_mask=causal,
                                memory_key_padding_mask=src == PAD)
            nxt = self.out(h[:, -1]).argmax(-1, keepdim=True)
            y = torch.cat([y, nxt], 1)
            if (nxt == EOS).all():
                break
        return y[:, 1:], None


def paragraph_pairs(docs):
    return [(" ".join(p["sentences"]), p["summary"]) for doc in docs for p in doc]


def train_seq2seq(model, pairs, vocab, epochs=12, lr=2e-3, batch=32, seed=0, log=print):
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    data = [(vocab.encode(s, 120), vocab.encode(t) + [EOS]) for s, t in pairs]
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        random.shuffle(data)
        total = 0.0
        for i in range(0, len(data), batch):
            chunk = data[i:i + batch]
            src, lengths = pad_2d([s for s, _ in chunk])
            tgt, _ = pad_2d([t for _, t in chunk])
            tgt_in = torch.cat([torch.full((len(chunk), 1), BOS), tgt[:, :-1]], 1)
            logits = model(src, lengths, tgt_in)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        if ep in (1, epochs) or ep % 4 == 0:
            log(f"      epoch {ep:2d}  loss {total / math.ceil(len(data) / batch):.4f}")
    log(f"      trained in {time.time() - t0:.1f}s")
    return model


def abstractive_summary(model, paragraphs, vocab):
    out_lines, attn_info = [], []
    for p in paragraphs:
        src, lengths = pad_2d([vocab.encode(" ".join(p), 120)])
        ids, attns = model.generate(src, lengths)
        text = vocab.decode(ids[0].tolist())
        if text and text not in out_lines:
            out_lines.append(text)
        attn_info.append((src[0], ids[0], attns))
    return " ".join(out_lines), attn_info


# =============================================================================
# Driver used by main.py
# =============================================================================
def run_all(doc, k, epochs=12, n_docs=400, seed=0, log=print):
    docs = generate_documents(n_docs, seed=seed)
    pairs = paragraph_pairs(docs)
    vocab = Vocab([s for d in docs for p in d for s in p["sentences"]] + [t for _, t in pairs])
    log(f"   synthetic training corpus: {len(docs)} documents, {len(pairs)} paragraph/summary pairs, "
        f"vocabulary {len(vocab)}")
    sents = doc.sentences
    results, extras = {}, {}

    for name, cell in [("RNN extractive (hierarchical RNN)", "rnn"),
                       ("LSTM extractive (hierarchical BiLSTM)", "lstm")]:
        log(f"   training {name}")
        m = train_extractor(HierarchicalRNNExtractor(len(vocab), cell), docs, vocab, max(4, epochs // 2),
                            seed=seed, log=log)
        idx, _ = extract(m, sents, vocab, k)
        results[name] = ("extractive", " ".join(sents[i] for i in idx), idx)

    log("   training Self-attention extractive")
    m = train_extractor(SelfAttentionExtractor(len(vocab)), docs, vocab, max(4, epochs // 2), seed=seed, log=log)
    idx, scores = extract(m, sents, vocab, k)
    results["Self-attention extractive"] = ("extractive", " ".join(sents[i] for i in idx), idx)
    with torch.no_grad():
        x, lengths = pad_2d([vocab.encode(s, 40) for s in sents])
        _, pool, sent_attn = m(x, lengths, return_attention=True)
    best = int(np.argmax(scores))
    words = [vocab.itos[i] for i in x[best].tolist() if i != PAD]
    top_words = sorted(zip(words, pool[best][:len(words)].tolist()), key=lambda t: -t[1])[:6]
    extras["self_attention"] = (sents[best], top_words, sent_attn.numpy())

    for name, cell, attn in [("RNN encoder-decoder (abstractive)", "rnn", False),
                             ("LSTM encoder-decoder (abstractive)", "lstm", False),
                             ("Seq2Seq + Bahdanau attention (abstractive)", "lstm", True)]:
        log(f"   training {name}")
        m = train_seq2seq(Seq2Seq(len(vocab), cell, attn), pairs, vocab, epochs, seed=seed, log=log)
        text, info = abstractive_summary(m, doc.paragraphs, vocab)
        results[name] = ("abstractive", text, None)
        if attn:
            src, ids, attns = info[0]
            rows = []
            for t, a in enumerate(attns):
                if a is None or ids[t].item() == EOS:
                    break
                j = int(a[0].argmax())
                rows.append((vocab.itos[ids[t].item()], vocab.itos[src[j].item()], float(a[0, j])))
            extras["bahdanau"] = rows

    log("   training Transformer (from scratch)")
    m = train_seq2seq(TransformerSummarizer(len(vocab)), pairs, vocab, epochs, lr=5e-4, seed=seed, log=log)
    text, _ = abstractive_summary(m, doc.paragraphs, vocab)
    results["Transformer encoder-decoder (abstractive)"] = ("abstractive", text, None)
    return results, extras
