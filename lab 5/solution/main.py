"""
Lab 5 - Text Summarisation of student-feedback reports
=======================================================

 a) Pre-processing         : unnecessary-text removal, sentence segmentation, tokenisation
 b) Graph / statistical    : Frequency, TF-IDF, TextRank, LexRank, Feature-based scoring
 c) Deep learning          : RNN, LSTM, Encoder-Decoder Seq2Seq (+attention), Self-attention,
                             Transformer (trained from scratch) and pre-trained BERT (extractive),
                             BART, PEGASUS, T5 (abstractive)
 d) Evaluation             : ROUGE-1, ROUGE-2, ROUGE-L against a human reference summary

Run
    python main.py                          # everything
    python main.py --skip-pretrained        # offline: statistical + from-scratch neural models
    python main.py --skip-neural --skip-pretrained   # statistical methods only (seconds)
    python main.py --input my_report.txt --reference my_reference.txt --k 4
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

from preprocess import content_words, preprocess, tokenize
from rouge import rouge_all
from statistical_summarizers import (FEATURE_WEIGHTS, STATISTICAL_METHODS, textrank_summary)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))


def banner(t):
    print("\n" + "=" * 100)
    print(t)
    print("=" * 100)


def short(s, n=95):
    return s if len(s) <= n else s[:n - 3] + "..."


def main():
    ap = argparse.ArgumentParser(description="Lab 5 - extractive & abstractive summarisation + ROUGE")
    ap.add_argument("--input", default=os.path.join(HERE, "data", "student_feedback.txt"))
    ap.add_argument("--reference", default=os.path.join(HERE, "data", "reference_summary.txt"))
    ap.add_argument("--k", type=int, default=None, help="sentences in extractive summaries "
                                                         "(default = number of paragraphs)")
    ap.add_argument("--epochs", type=int, default=8, help="epochs for the from-scratch neural models")
    ap.add_argument("--train-docs", type=int, default=200,
                    help="synthetic training documents for the from-scratch neural models")
    ap.add_argument("--skip-neural", action="store_true", help="skip the from-scratch neural models")
    ap.add_argument("--skip-pretrained", action="store_true", help="skip BERT/BART/PEGASUS/T5")
    ap.add_argument("--bart", default=None, help="e.g. sshleifer/distilbart-cnn-12-6")
    ap.add_argument("--pegasus", default=None, help="e.g. google/pegasus-xsum")
    ap.add_argument("--t5", default=None, help="e.g. google-t5/t5-base")
    ap.add_argument("--out", default=os.path.join(HERE, "outputs"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(args.input, encoding="utf-8") as f:
        raw = f.read()
    with open(args.reference, encoding="utf-8") as f:
        reference = f.read().strip()

    # ------------------------------------------------------------------ (a)
    banner("(a) PRE-PROCESSING")
    doc = preprocess(raw)
    sents = doc.sentences
    k = args.k or len(doc.paragraphs)
    print(f"Raw document: {len(raw.split())} words, {len(raw.splitlines())} lines")
    print(f"Document title detected: {doc.title!r}")
    print("\nRemoved unnecessary text:")
    for reason, text in doc.removed:
        print(f"  - [{reason}] {short(text, 80)}")
    print(f"\nSentence segmentation -> {len(sents)} sentences in {len(doc.paragraphs)} paragraphs:")
    i = 0
    for heading, para in zip(doc.headings, doc.paragraphs):
        print(f"  Paragraph: {heading or '(no heading)'}")
        for s in para:
            print(f"    S{i:02d}: {s}")
            i += 1
    print("\nTokenisation example (S00):")
    print("  tokens       :", tokenize(sents[0]))
    print("  content words:", content_words(sents[0]), "(stop-words removed)")
    print(f"\nSummary length: k = {k} sentences | reference summary: {len(reference.split())} words")

    summaries = {}       # name -> (family, kind, text, indices)

    # ------------------------------------------------------------------ (b)
    banner("(b) GRAPH / STATISTICAL EXTRACTIVE SUMMARISATION")
    lead = [si for si, (pi, pos, _) in enumerate(doc.sentence_meta) if pos == 0][:k]
    summaries["Lead baseline (first sentence of each paragraph)"] = (
        "baseline", "extractive", " ".join(sents[i] for i in lead), lead)
    for name, fn in STATISTICAL_METHODS.items():
        idx, scores = fn(doc, k)
        summaries[name] = ("statistical", "extractive", " ".join(sents[i] for i in idx), idx)
        order = np.argsort(-scores)
        print(f"\n--- {name} ---")
        if name == "TextRank":
            print(f"  PageRank converged after {textrank_summary.iterations} iterations (d = 0.85)")
        if name == "LexRank":
            from statistical_summarizers import lexrank_summary
            print(f"  similarity graph density (cosine > 0.05): {lexrank_summary.density:.2f}")
        print("  Top sentences:  " + ", ".join(f"S{j:02d}={scores[j]:.3f}" for j in order[:k + 2]))
        print(f"  Selected      : {['S%02d' % j for j in idx]}")
        if name == "Feature-based scoring":
            from statistical_summarizers import feature_summary
            F = feature_summary.features
            print("  Feature weights:", FEATURE_WEIGHTS)
            print("  " + f"{'sent':<6}" + "".join(f"{f:>10}" for f in F) + f"{'score':>10}")
            for j in range(len(sents)):
                print("  " + f"S{j:02d}   " + "".join(f"{F[f][j]:>10.2f}" for f in F) + f"{scores[j]:>10.3f}")

    # ------------------------------------------------------------------ (c)
    if not args.skip_neural:
        banner("(c) DEEP-LEARNING SUMMARISERS TRAINED FROM SCRATCH (RNN / LSTM / Seq2Seq / Self-attention / Transformer)")
        import torch
        torch.set_num_threads(max(1, os.cpu_count() // 2))
        from neural_summarizers import run_all as run_neural
        t0 = time.time()
        res, extras = run_neural(doc, k, epochs=args.epochs, n_docs=args.train_docs)
        print(f"   all from-scratch models trained in {time.time() - t0:.0f}s")
        for name, (kind, text, idx) in res.items():
            summaries[name] = ("neural (scratch)", kind, text, idx)
        if "bahdanau" in extras:
            print("\n   Bahdanau attention while summarising paragraph 1 "
                  "(generated word -> most-attended source word, weight):")
            print("   " + ", ".join(f"{g}->{s} ({w:.2f})" for g, s, w in extras["bahdanau"]))
        if "self_attention" in extras:
            sent, words, _ = extras["self_attention"]
            print(f"\n   Self-attention extractor - highest-scoring sentence: {short(sent)}")
            print("   words with the largest attention-pooling weights: "
                  + ", ".join(f"{w} ({a:.2f})" for w, a in words))

    if not args.skip_pretrained:
        banner("(c) PRE-TRAINED TRANSFORMERS (BERT extractive, BART, PEGASUS, T5)")
        from pretrained_summarizers import run_all as run_pretrained
        models = {k_: v for k_, v in {"bart": args.bart, "pegasus": args.pegasus, "t5": args.t5}.items() if v}
        res, _ = run_pretrained(doc, k, models)
        for name, (kind, text, idx) in res.items():
            summaries[name] = ("pre-trained", kind, text, idx)

    # ------------------------------------------------------------- summaries
    banner("GENERATED SUMMARIES")
    print(f"REFERENCE SUMMARY:\n  {reference}\n")
    for name, (fam, kind, text, idx) in summaries.items():
        sel = f"  sentences {['S%02d' % j for j in idx]}" if idx is not None else ""
        print(f"[{fam} | {kind}] {name}{sel}\n  {text}\n")

    # ------------------------------------------------------------------ (d)
    banner("(d) ROUGE EVALUATION AGAINST THE REFERENCE SUMMARY (F1, with P/R for ROUGE-L)")
    rows = []
    for name, (fam, kind, text, _) in summaries.items():
        r = rouge_all(text, reference)
        rows.append((name, fam, kind, len(text.split()), r))
    hdr = (f"{'Method':<52}{'Type':<12}{'Words':>6}{'R-1':>8}{'R-2':>8}{'R-L':>8}"
           f"{'R-L P':>8}{'R-L R':>8}")
    print(hdr)
    print("-" * len(hdr))
    for name, fam, kind, n, r in rows:
        print(f"{short(name, 51):<52}{kind:<12}{n:>6}{r['ROUGE-1'][2]:>8.3f}{r['ROUGE-2'][2]:>8.3f}"
              f"{r['ROUGE-L'][2]:>8.3f}{r['ROUGE-L'][0]:>8.3f}{r['ROUGE-L'][1]:>8.3f}")

    best = max(rows, key=lambda x: x[4]["ROUGE-L"][2])
    best_ext = max((x for x in rows if x[2] == "extractive"), key=lambda x: x[4]["ROUGE-L"][2])
    abst = [x for x in rows if x[2] == "abstractive"]
    print(f"\nBest overall (ROUGE-L F1): {best[0]} = {best[4]['ROUGE-L'][2]:.3f}")
    print(f"Best extractive          : {best_ext[0]} = {best_ext[4]['ROUGE-L'][2]:.3f}")
    if abst:
        best_abs = max(abst, key=lambda x: x[4]["ROUGE-L"][2])
        print(f"Best abstractive         : {best_abs[0]} = {best_abs[4]['ROUGE-L'][2]:.3f}")
    print("""
Observations
  * ROUGE-1 measures content (word) overlap, ROUGE-2 fluency / phrase overlap and ROUGE-L the
    longest in-order overlap. Extractive methods copy whole sentences, so their precision is
    limited by long sentences; abstractive models can paraphrase but may hallucinate.
  * Graph methods (TextRank / LexRank) favour 'central' sentences that share words with many others;
    frequency/TF-IDF sums favour long sentences; the feature-based scorer adds positional knowledge
    (first sentence of each paragraph is usually the topic sentence) and heading keywords.
  * The from-scratch neural models learn the domain from the synthetic corpus - they are only as
    good as their training data. The pre-trained BART / PEGASUS were fine-tuned on news
    (CNN/DailyMail), so they produce fluent but news-style, partly copied summaries; T5-small is
    the weakest because of its size and 512-token input limit.
  * CAUTION when reading the table: the synthetic training targets of the from-scratch abstractive
    models use the same short phrasing style as the reference summary ("X is good but Y is poor").
    Their high ROUGE is therefore partly a vocabulary/style match, not proof that they are better
    summarisers than BART/PEGASUS. Check faithfulness manually: an abstractive model can be fluent
    and still state the wrong polarity (e.g. "laboratory facilities are good") - ROUGE does not
    detect such hallucinations.
  * With a single document and one reference, differences of a few ROUGE points are not
    statistically meaningful; a real evaluation would use many reports and several references.""")

    # -------------------------------------------------------------- save
    with open(os.path.join(args.out, "rouge_scores.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "family", "type", "words", "rouge1_p", "rouge1_r", "rouge1_f",
                    "rouge2_p", "rouge2_r", "rouge2_f", "rougeL_p", "rougeL_r", "rougeL_f"])
        for name, fam, kind, n, r in rows:
            w.writerow([name, fam, kind, n] + [round(v, 4) for m in ("ROUGE-1", "ROUGE-2", "ROUGE-L")
                                               for v in r[m]])
    with open(os.path.join(args.out, "summaries.md"), "w", encoding="utf-8") as f:
        f.write("# Generated summaries\n\n## Reference\n\n" + reference + "\n\n")
        for name, (fam, kind, text, _) in summaries.items():
            r = rouge_all(text, reference)
            f.write(f"## {name}\n\n*{fam}, {kind}* - ROUGE-1 {r['ROUGE-1'][2]:.3f}, "
                    f"ROUGE-2 {r['ROUGE-2'][2]:.3f}, ROUGE-L {r['ROUGE-L'][2]:.3f}\n\n{text}\n\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.arange(len(rows))
        fig, ax = plt.subplots(figsize=(14, 6))
        for off, m in zip((-0.27, 0, 0.27), ("ROUGE-1", "ROUGE-2", "ROUGE-L")):
            ax.bar(x + off, [r[4][m][2] for r in rows], 0.27, label=m)
        ax.set_xticks(x)
        ax.set_xticklabels([short(r[0], 40) for r in rows], rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("F1")
        ax.set_title("ROUGE F1 of each summarisation technique vs. the reference summary")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "rouge_chart.png"), dpi=130)
    except Exception as e:
        print(f"(chart skipped: {e})")
    print(f"\nSaved: {os.path.join(args.out, 'summaries.md')}, rouge_scores.csv, rouge_chart.png")


if __name__ == "__main__":
    main()
