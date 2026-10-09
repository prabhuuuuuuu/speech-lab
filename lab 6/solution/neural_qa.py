"""
Lab 6 (ii) - Deep-learning extractive QA with pre-trained Transformers (retriever + reader).

  Retriever : TF-IDF (uni+bi-grams) cosine similarity -> top-k passages
  Reader    : BERT-style model fine-tuned on SQuAD.  Input = [CLS] question [SEP] passage [SEP]
              Two linear heads give a START and an END logit for every passage token; the answer is
              the span (i <= j, length <= max_answer_len) maximising start_logit[i] + end_logit[j].
              Implemented directly on AutoModelForQuestionAnswering (no pipeline) so every step is
              visible.
  No-answer : SQuAD-2.0 models (deepset/roberta-base-squad2) score the "null" answer at the [CLS]
              position; if null_score > best_span_score the system abstains. SQuAD-1.1 models
              (distilbert-base-cased-distilled-squad) were never trained to abstain and always answer.
"""

import time

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    def __init__(self, passages):
        self.passages = passages
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.matrix = self.vec.fit_transform(passages)

    def retrieve(self, question, k=2):
        sims = cosine_similarity(self.vec.transform([question]), self.matrix)[0]
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order]


class ExtractiveReader:
    def __init__(self, model_name, can_abstain=None, max_length=384, stride=128):
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        try:
            from transformers.utils import logging
            logging.set_verbosity_error()
        except Exception:
            pass
        self.name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).eval()
        self.can_abstain = ("squad2" in model_name.lower()) if can_abstain is None else can_abstain
        self.max_length, self.stride = max_length, stride

    @torch.no_grad()
    def read(self, question, context, max_answer_len=30, top_n=20):
        enc = self.tok(question, context, truncation="only_second", max_length=self.max_length,
                       stride=self.stride, return_overflowing_tokens=True, return_offsets_mapping=True,
                       padding=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")
        enc.pop("overflow_to_sample_mapping", None)
        out = self.model(**{k: v for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")})
        best = {"text": "", "score": -1e9, "start_prob": 0.0, "end_prob": 0.0}
        null_score = -1e9
        for w in range(out.start_logits.shape[0]):
            s, e = out.start_logits[w].numpy(), out.end_logits[w].numpy()
            seq_ids = enc.sequence_ids(w)
            ctx = [i for i, sid in enumerate(seq_ids) if sid == 1]
            null_score = max(null_score, float(s[0] + e[0]))          # [CLS] = "no answer"
            ps, pe = np.exp(s - s.max()), np.exp(e - e.max())
            ps, pe = ps / ps.sum(), pe / pe.sum()
            starts = [i for i in np.argsort(-s)[:top_n] if i in ctx]
            ends = [j for j in np.argsort(-e)[:top_n] if j in ctx]
            for i in starts:
                for j in ends:
                    if j < i or j - i + 1 > max_answer_len:
                        continue
                    score = float(s[i] + e[j])
                    if score > best["score"]:
                        a, b = int(offsets[w][i][0]), int(offsets[w][j][1])
                        best = {"text": context[a:b], "score": score,
                                "start_prob": float(ps[i]), "end_prob": float(pe[j])}
        best["null_score"] = null_score
        best["confidence"] = best["start_prob"] * best["end_prob"]
        return best


class NeuralQA:
    def __init__(self, passages, model_name, top_k=2):
        self.retriever = TfidfRetriever(passages)
        self.reader = ExtractiveReader(model_name)
        self.passages, self.top_k = passages, top_k

    def answer(self, question):
        t0 = time.time()
        hits = self.retriever.retrieve(question, self.top_k)
        cands = []
        for pid, sim in hits:
            r = self.reader.read(question, self.passages[pid])
            r.update(passage=pid, retrieval_score=sim)
            cands.append(r)
        best = max(cands, key=lambda r: r["score"])
        best["abstained"] = False
        if self.reader.can_abstain and all(c["null_score"] > c["score"] for c in cands):
            best["abstained"] = True                     # every passage prefers the null answer
        best["answer"] = None if best["abstained"] else best["text"].strip(" .,;")
        best["latency"] = time.time() - t0
        best["retrieved"] = hits
        return best
