"""
Lab 5 (c) 6-9 - Pre-trained Transformer summarisers (Hugging Face transformers).

  6. BERT extractive : BERT sentence embeddings (mean pooling) -> K-Means with k clusters ->
                       the sentence closest to every centroid is selected (Miller, 2019).
                       An SBERT (sentence-BERT) variant is also provided.
  7. BART            : facebook/bart-large-cnn        (denoising seq2seq, fine-tuned on CNN/DailyMail)
  8. PEGASUS         : google/pegasus-cnn_dailymail   (gap-sentence-generation pre-training)
  9. T5              : google-t5/t5-small             ("summarize: " text-to-text prefix)

Models are downloaded automatically on first use into the Hugging Face cache
(~/.cache/huggingface). Approximate sizes: BERT 440 MB, MiniLM 90 MB, BART 1.6 GB,
PEGASUS 2.3 GB, T5-small 240 MB.
"""

import time

import numpy as np
import torch

DEFAULT_MODELS = {
    "bert": "google-bert/bert-base-uncased",
    "sbert": "sentence-transformers/all-MiniLM-L6-v2",
    "bart": "facebook/bart-large-cnn",
    "pegasus": "google/pegasus-cnn_dailymail",
    "t5": "google-t5/t5-small",
}


def _quiet():
    try:
        from transformers.utils import logging
        logging.set_verbosity_error()
    except Exception:
        pass


# ----------------------------------------------------------------------------- BERT
@torch.no_grad()
def sentence_embeddings(sentences, model_name):
    from transformers import AutoModel, AutoTokenizer
    _quiet()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    enc = tok(sentences, padding=True, truncation=True, max_length=128, return_tensors="pt")
    hidden = model(**enc).last_hidden_state                       # B x T x 768
    mask = enc["attention_mask"].unsqueeze(-1).float()
    emb = (hidden * mask).sum(1) / mask.sum(1)                    # mean pooling
    return torch.nn.functional.normalize(emb, dim=-1).numpy()


def bert_extractive(doc, k, model_name=DEFAULT_MODELS["bert"]):
    from sklearn.cluster import KMeans
    sents = doc.sentences
    E = sentence_embeddings(sents, model_name)
    k = min(k, len(sents))
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(E)
    chosen = []
    for c in range(k):
        members = np.where(km.labels_ == c)[0]
        d = np.linalg.norm(E[members] - km.cluster_centers_[c], axis=1)
        chosen.append(int(members[d.argmin()]))
    centrality = E @ E.mean(0)                                    # cosine to document centroid
    return sorted(chosen), centrality, km.labels_


# ----------------------------------------------------------------------- seq2seq
def _chunks(paragraphs, tok, max_tokens, prefix=""):
    """Greedy packing of whole paragraphs into chunks that fit the model's input limit."""
    chunks, cur = [], ""
    for p in paragraphs:
        text = " ".join(p)
        cand = (cur + " " + text).strip()
        if cur and len(tok(prefix + cand).input_ids) > max_tokens:
            chunks.append(cur)
            cur = text
        else:
            cur = cand
    if cur:
        chunks.append(cur)
    return chunks


@torch.no_grad()
def seq2seq_summary(doc, model_name, prefix="", max_input=1024, max_new_tokens=140, min_length=60,
                    num_beams=4, length_penalty=2.0):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    _quiet()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).eval()
    chunks = _chunks(doc.paragraphs, tok, max_input, prefix)
    outs = []
    for ch in chunks:
        enc = tok(prefix + ch, return_tensors="pt", truncation=True, max_length=max_input)
        ids = model.generate(**enc, num_beams=num_beams, max_new_tokens=max_new_tokens,
                             min_length=max(10, min_length // len(chunks)), no_repeat_ngram_size=3,
                             length_penalty=length_penalty, early_stopping=True)
        text = tok.decode(ids[0], skip_special_tokens=True)
        outs.append(text.replace("<n>", " ").strip())
    return " ".join(outs), len(chunks)


def run_all(doc, k, models=None, log=print):
    models = {**DEFAULT_MODELS, **(models or {})}
    results, extras = {}, {}

    def timed(name, fn):
        t0 = time.time()
        try:
            out = fn()
            log(f"   {name:<38} done in {time.time() - t0:5.1f}s")
            return out
        except Exception as e:  # e.g. no internet on first run
            log(f"   {name:<38} FAILED: {type(e).__name__}: {str(e)[:150]}")
            return None

    out = timed(f"BERT extractive ({models['bert']})", lambda: bert_extractive(doc, k, models["bert"]))
    if out:
        idx, centrality, labels = out
        results["BERT extractive (K-Means on BERT embeddings)"] = (
            "extractive", " ".join(doc.sentences[i] for i in idx), idx)
        extras["bert"] = (centrality, labels)
    out = timed(f"SBERT extractive ({models['sbert']})", lambda: bert_extractive(doc, k, models["sbert"]))
    if out:
        results["SBERT extractive (K-Means on SBERT embeddings)"] = (
            "extractive", " ".join(doc.sentences[i] for i in out[0]), out[0])

    for label, key, kw in [
        ("BART", "bart", dict(max_input=1024)),
        ("PEGASUS", "pegasus", dict(max_input=1024)),
        ("T5", "t5", dict(prefix="summarize: ", max_input=512, max_new_tokens=90, min_length=50)),
    ]:
        out = timed(f"{label} ({models[key]})", lambda: seq2seq_summary(doc, models[key], **kw))
        if out:
            text, n_chunks = out
            results[f"{label} ({models[key].split('/')[-1]}, abstractive)"] = ("abstractive", text, None)
            if n_chunks > 1:
                log(f"      (input split into {n_chunks} chunks to fit the {kw['max_input']}-token limit)")
    return results, extras
