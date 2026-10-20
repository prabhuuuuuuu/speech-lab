"""
Lab 2 - BoW, N-grams, Word2Vec and Topic Modeling
=================================================

Part A  clean a review corpus, remove stop words, lemmatize (or stem), build the
        Bag-of-Words matrix and report the top 20 words, vocabulary size and sparsity
Part B  generate unigram, bigram and trigram matrices; report the top 15 bigrams
        and top 10 trigrams
Part C  train a CBOW Word2Vec model and inspect nearest neighbours
Part D  fit a five-topic LDA model and report coherent topics

The default corpus is a deterministic 2,000-review synthetic product-review set
(see review_data.py).  A real Amazon / Flipkart / other review file can be used:

    python main.py
    python main.py --csv Reviews.csv --text-column Text --reviews 2500
    python main.py --json reviews.json --reviews 1500 --topics 5
    python main.py --save-model

Outputs (in outputs/): report.md, bow_summary.json, top_words.csv,
top_bigrams.csv, top_trigrams.csv, lda_topics.csv, review_corpus.csv and
lab2_overview.png.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gensim import corpora
from gensim.models import LdaModel, Word2Vec
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from review_data import clean_headline, load_reviews

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "outputs"

STOP_WORDS = set(ENGLISH_STOP_WORDS)
STOP_WORDS |= {
    "also", "really", "quite", "would", "could", "one", "get", "got",
    "buy", "bought", "ordered", "order", "product", "item", "amazon",
    "flipkart", "pack", "packet", "seller", "arrive", "look", "like",
}
DOMAIN_STOP_WORDS = sorted(
    {"order", "product", "item", "amazon", "flipkart", "pack", "packet", "seller"}
    & STOP_WORDS
)

CONTRACTIONS = (
    (r"won't", "will not"),
    (r"can't", "can not"),
    (r"n't", " not"),
    (r"'re", " are"),
    (r"'ve", " have"),
    (r"'ll", " will"),
    (r"'d", " would"),
    (r"'m", " am"),
)
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


try:
    from nltk import pos_tag
    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer

    _LEMMATIZER = WordNetLemmatizer()
    pos_tag(["test"])
    NORMALISATION = "WordNet lemmatization (POS-aware)"
    _WORDNET_POS = {"a": wordnet.ADJ, "n": wordnet.NOUN, "v": wordnet.VERB,
                    "r": wordnet.ADV}

    def normalise(tokens: list[str]) -> list[str]:
        out = []
        for word, tag in pos_tag(tokens):
            pos = _WORDNET_POS.get(tag[0].lower(), wordnet.NOUN)
            out.append(_LEMMATIZER.lemmatize(word, pos))
        return out

except Exception:
    from nltk.stem import PorterStemmer

    _STEMMER = PorterStemmer()
    NORMALISATION = "Porter stemming (WordNet unavailable)"

    def normalise(tokens: list[str]) -> list[str]:
        return [_STEMMER.stem(token) for token in tokens]


def clean_text(text: str) -> str:
    text = html.unescape(str(text)).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    for pattern, replacement in CONTRACTIONS:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"'s\b", "", text)
    text = re.sub(r"s'\b", "s", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess(text: str) -> list[str]:
    tokens = [token for token in TOKEN_RE.findall(clean_text(text)) if len(token) > 1]
    tokens = [token for token in tokens if token not in STOP_WORDS]
    tokens = normalise(tokens)
    return [token for token in tokens if token not in STOP_WORDS and len(token) > 1]


def tokenize_corpus(reviews: list[dict]) -> list[list[str]]:
    return [preprocess(review["text"]) for review in reviews]


def build_bow(tokenized: list[list[str]]):
    vectorizer = CountVectorizer(analyzer=lambda doc: doc, min_df=1, dtype=np.int64)
    matrix = vectorizer.fit_transform(tokenized)
    terms = vectorizer.get_feature_names_out()
    counts = np.asarray(matrix.sum(axis=0)).ravel().astype(np.int64)
    top = [(terms[i], int(counts[i])) for i in np.argsort(-counts)]
    return matrix, terms, counts, top


def build_ngrams(tokenized: list[list[str]], n: int):
    documents = [" ".join(tokens) for tokens in tokenized]
    vectorizer = CountVectorizer(
        ngram_range=(n, n), token_pattern=r"(?u)\b\w+\b", min_df=1, dtype=np.int64
    )
    matrix = vectorizer.fit_transform(documents)
    terms = vectorizer.get_feature_names_out()
    counts = np.asarray(matrix.sum(axis=0)).ravel().astype(np.int64)
    top = [(terms[i], int(counts[i])) for i in np.argsort(-counts)]
    return matrix, terms, counts, top


def train_cbow(tokenized: list[list[str]], args):
    model = Word2Vec(
        sentences=tokenized,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        sg=0,
        workers=1,
        epochs=args.epochs,
        seed=args.seed,
        sample=1e-3,
        negative=5,
    )
    return model


def nearest_words(model: Word2Vec, candidates: list[str], topn: int = 6):
    selected = [word for word in candidates if word in model.wv.key_to_index]
    if not selected:
        selected = model.wv.index_to_key[:5]
    return [(word, model.wv.most_similar(word, topn=topn)) for word in selected]


def fit_lda(tokenized: list[list[str]], num_topics: int, seed: int):
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=3, no_above=0.40, keep_n=20000)
    corpus = [dictionary.doc2bow(tokens) for tokens in tokenized]
    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=10,
        iterations=80,
        random_state=seed,
        alpha="auto",
        eta="auto",
        eval_every=None,
    )
    topics = model.show_topics(num_topics=num_topics, num_words=10, formatted=False)
    coherent_topics = [(topic_id, list(terms)) for topic_id, terms in topics]
    try:
        coherence = CoherenceModel(
            model=model, texts=tokenized, dictionary=dictionary,
            coherence="c_v", processes=1,
        ).get_coherence()
    except Exception:
        coherence = None
    dominant = []
    for bow in corpus:
        distribution = model.get_document_topics(bow, minimum_probability=0.0)
        dominant.append(max(distribution, key=lambda pair: pair[1])[0])
    sizes = Counter(dominant)
    return model, coherent_topics, coherence, sizes, dominant


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def md_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def save_plot(path: Path, top_words, top_bigrams, topic_sizes, topics) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    fig.suptitle("Lab 2 - Review corpus: BoW, n-grams and LDA topics", fontsize=15)

    words = [word for word, _ in top_words[:20]][::-1]
    counts = [count for _, count in top_words[:20]][::-1]
    axes[0, 0].barh(words, counts, color="#4C72B0")
    axes[0, 0].set_title("Top 20 words (Bag of Words)")
    axes[0, 0].set_xlabel("frequency")

    bigrams = [gram for gram, _ in top_bigrams[:15]][::-1]
    bigram_counts = [count for _, count in top_bigrams[:15]][::-1]
    axes[0, 1].barh(bigrams, bigram_counts, color="#DD8452")
    axes[0, 1].set_title("Top 15 bigrams")
    axes[0, 1].set_xlabel("frequency")

    labels = [f"Topic {k}" for k in sorted(topic_sizes)]
    values = [topic_sizes[k] for k in sorted(topic_sizes)]
    axes[1, 0].bar(labels, values, color="#55A868")
    axes[1, 0].set_title("Dominant LDA topic per review")
    axes[1, 0].set_ylabel("reviews")
    axes[1, 0].tick_params(axis="x", rotation=20)

    axes[1, 1].axis("off")
    axes[1, 1].set_title("Top words per topic", loc="left")
    text = []
    for topic_id, terms in topics:
        top_terms = ", ".join(word for word, _ in terms[:8])
        text.append(f"Topic {topic_id}: {top_terms}")
    axes[1, 1].text(0.01, 0.95, "\n\n".join(text), va="top", ha="left",
                    fontsize=10, family="monospace", wrap=True)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lab 2 - BoW, n-grams, CBOW Word2Vec and LDA topic modeling"
    )
    parser.add_argument("--csv", help="CSV review file (Amazon / Flipkart / other)")
    parser.add_argument("--json", help="JSON review file (list of strings or objects)")
    parser.add_argument("--text-column", help="CSV text column (auto-detected by default)")
    parser.add_argument("--reviews", type=int, default=2000,
                        help="synthetic reviews to generate or maximum rows to load")
    parser.add_argument("--topics", type=int, default=5)
    parser.add_argument("--vector-size", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--save-model", action="store_true",
                        help="also save outputs/cbow_word2vec.model")
    args = parser.parse_args()

    if args.csv and args.json:
        parser.error("use either --csv or --json, not both")
    if args.reviews < 100:
        parser.error("--reviews must be at least 100")

    source = args.csv or args.json
    reviews, source_name, synthetic = load_reviews(
        source, text_column=args.text_column, n=args.reviews, seed=args.seed
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 92)
    print("Lab 2 - BoW, N-grams, Word2Vec (CBOW) and Topic Modeling")
    print("=" * 92)
    print(f"Dataset        : {source_name}")
    print(f"Reviews        : {len(reviews)}")
    print(f"Normalisation  : {NORMALISATION}")

    tokenized = tokenize_corpus(reviews)
    lengths = [len(tokens) for tokens in tokenized]
    print(f"Token counts   : {sum(lengths)} total, {np.mean(lengths):.1f} per review "
          f"(min {min(lengths)}, max {max(lengths)})")
    print(f"Removed words  : {len(STOP_WORDS)} stop words")
    print(f"Domain stops   : {', '.join(DOMAIN_STOP_WORDS)}")

    print("\n" + "-" * 92)
    print("PART A - Bag of Words")
    print("-" * 92)
    sample = reviews[0]["text"]
    print(f"Original review : {clean_headline(sample)}")
    print(f"Cleaned tokens  : {' '.join(tokenized[0][:30])}")
    print("Normal forms    : " +
          ", ".join(f"{word} -> {preprocess(word)[0]}"
                    for word in ("roasting", "roasted", "roasts")
                    if preprocess(word)))

    bow_matrix, vocabulary, word_counts, top_words = build_bow(tokenized)
    n_reviews, vocab_size = bow_matrix.shape
    nonzero = int(bow_matrix.nnz)
    sparsity = 1.0 - nonzero / (n_reviews * vocab_size)
    print(f"BoW matrix      : {n_reviews} reviews x {vocab_size} terms")
    print(f"Non-zero cells  : {nonzero:,}")
    print(f"Sparsity        : {sparsity:.2%}")
    print("\nTop 20 frequent words:")
    print(md_table(["rank", "word", "count"],
                   [[i + 1, word, count] for i, (word, count) in enumerate(top_words[:20])]))

    print("\n" + "-" * 92)
    print("PART B - N-grams")
    print("-" * 92)
    ngram_info = {}
    for n in (1, 2, 3):
        matrix, terms, counts, top = build_ngrams(tokenized, n)
        ngram_info[n] = {"matrix": matrix, "terms": terms, "counts": counts, "top": top}
        print(f"{['Unigrams', 'Bigrams', 'Trigrams'][n - 1]:<9}: "
              f"{matrix.shape[0]} docs x {len(terms):,} unique, "
              f"{int(counts.sum()):,} tokens in total")
    top_bigrams = ngram_info[2]["top"]
    top_trigrams = ngram_info[3]["top"]
    print("\nTop 15 bigrams:")
    print(md_table(["rank", "bigram", "count"],
                   [[i + 1, gram, count] for i, (gram, count) in enumerate(top_bigrams[:15])]))
    print("\nTop 10 trigrams:")
    print(md_table(["rank", "trigram", "count"],
                   [[i + 1, gram, count] for i, (gram, count) in enumerate(top_trigrams[:10])]))

    print("\n" + "-" * 92)
    print("PART C - CBOW Word2Vec")
    print("-" * 92)
    model = train_cbow(tokenized, args)
    print(f"Architecture    : CBOW (sg=0), vector_size={args.vector_size}, "
          f"window={args.window}, min_count={args.min_count}, epochs={args.epochs}")
    print(f"Word2Vec vocab  : {len(model.wv):,} words")
    candidates = ["coffee", "tea", "chocolate", "snack", "spice", "fresh",
                  "delivery", "packaging", "quality", "price", "recommend"]
    neighbours = nearest_words(model, candidates)
    if neighbours:
        sample_word = neighbours[0][0]
        print(f"Vector sample   : {sample_word} -> "
              f"{np.array2string(model.wv[sample_word][:8], precision=3)} ...")
    print("\nNearest neighbours (cosine similarity):")
    for word, similar in neighbours:
        print(f"  {word:<12} -> " + ", ".join(f"{other} ({score:.3f})" for other, score in similar))
    if args.save_model:
        model.save(str(out / "cbow_word2vec.model"))
        print(f"Model saved     : {out / 'cbow_word2vec.model'}")

    print("\n" + "-" * 92)
    print(f"PART D - LDA ({args.topics} topics)")
    print("-" * 92)
    lda, topics, coherence, topic_sizes, dominant = fit_lda(
        tokenized, args.topics, args.seed
    )
    for topic_id, terms in topics:
        print(f"Topic {topic_id}: " + " | ".join(f"{word} {weight:.3f}" for word, weight in terms))
    print(f"\nDominant topic per review: " +
          ", ".join(f"T{topic}: {topic_sizes.get(topic, 0)}" for topic in range(args.topics)))
    if coherence is not None:
        print(f"LDA coherence (c_v): {coherence:.4f}")

    print("\nExample review per topic:")
    examples = {}
    for topic in range(args.topics):
        index = next((i for i, assigned in enumerate(dominant) if assigned == topic), None)
        if index is not None:
            examples[topic] = reviews[index]["text"]
            print(f"  Topic {topic}: {clean_headline(examples[topic])}")

    print("\nSaving outputs ...")
    write_csv(out / "top_words.csv", ["rank", "word", "count"],
              [[i + 1, word, count] for i, (word, count) in enumerate(top_words[:20])])
    write_csv(out / "top_bigrams.csv", ["rank", "bigram", "count"],
              [[i + 1, gram, count] for i, (gram, count) in enumerate(top_bigrams[:15])])
    write_csv(out / "top_trigrams.csv", ["rank", "trigram", "count"],
              [[i + 1, gram, count] for i, (gram, count) in enumerate(top_trigrams[:10])])
    write_csv(out / "lda_topics.csv", ["topic", "rank", "word", "weight"],
              [[topic, rank + 1, word, round(float(weight), 6)]
               for topic, terms in topics for rank, (word, weight) in enumerate(terms)])
    write_csv(out / "review_corpus.csv", ["id", "category", "rating", "text"],
              [[r["id"], r["category"], r["rating"], r["text"]] for r in reviews])

    summary = {
        "dataset": source_name,
        "synthetic": synthetic,
        "reviews": len(reviews),
        "normalisation": NORMALISATION,
        "stop_words": len(STOP_WORDS),
        "domain_stop_words": DOMAIN_STOP_WORDS,
        "token_count": int(sum(lengths)),
        "tokens_per_review": float(np.mean(lengths)),
        "part_a": {
            "vocabulary_size": int(vocab_size),
            "matrix_shape": [int(n_reviews), int(vocab_size)],
            "non_zero_cells": nonzero,
            "sparsity": float(sparsity),
            "top_words": [[word, int(count)] for word, count in top_words[:20]],
        },
        "part_b": {
            "unique_unigrams": int(len(ngram_info[1]["terms"])),
            "unique_bigrams": int(len(ngram_info[2]["terms"])),
            "unique_trigrams": int(len(ngram_info[3]["terms"])),
            "top_bigrams": [[gram, int(count)] for gram, count in top_bigrams[:15]],
            "top_trigrams": [[gram, int(count)] for gram, count in top_trigrams[:10]],
        },
        "part_c": {
            "architecture": "CBOW",
            "vector_size": args.vector_size,
            "window": args.window,
            "min_count": args.min_count,
            "epochs": args.epochs,
            "vocabulary_size": int(len(model.wv)),
            "nearest_neighbours": {
                word: [[other, round(float(score), 6)] for other, score in similar]
                for word, similar in neighbours
            },
        },
        "part_d": {
            "num_topics": args.topics,
            "coherence_c_v": None if coherence is None else float(coherence),
            "topics": [
                {"topic": int(topic_id),
                 "terms": [[word, round(float(weight), 6)] for word, weight in terms]}
                for topic_id, terms in topics
            ],
            "dominant_topic_counts": {
                str(topic): int(topic_sizes.get(topic, 0)) for topic in range(args.topics)
            },
            "example_reviews": {str(k): v for k, v in examples.items()},
        },
    }
    (out / "bow_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = [
        "# Lab 2 - BoW, N-grams, Word2Vec and Topic Modeling",
        "",
        f"* Dataset: **{source_name}**",
        f"* Reviews used: **{len(reviews)}**",
        f"* Cleaning: lower-case, HTML/URL/e-mail removal, punctuation and number removal, "
        f"contraction expansion, stop-word removal, {NORMALISATION}",
        f"* Tokens: **{sum(lengths):,}** ({np.mean(lengths):.1f} per review)",
        "",
        "## Part A - Bag of Words",
        "",
        "### Top 20 frequent words",
        "",
        md_table(["rank", "word", "count"],
                 [[i + 1, word, count] for i, (word, count) in enumerate(top_words[:20])]),
        "",
        f"* Vocabulary size: **{vocab_size:,}**",
        f"* BoW matrix: **{n_reviews} x {vocab_size:,}**",
        f"* Non-zero cells: **{nonzero:,}**",
        f"* Sparsity: **{sparsity:.2%}**",
        "",
        "## Part B - N-grams",
        "",
        md_table(["n-gram order", "matrix", "unique n-grams", "total tokens"],
                 [["unigram", f"{ngram_info[1]['matrix'].shape[0]} x {len(ngram_info[1]['terms']):,}",
                   f"{len(ngram_info[1]['terms']):,}", f"{int(ngram_info[1]['counts'].sum()):,}"],
                  ["bigram", f"{ngram_info[2]['matrix'].shape[0]} x {len(ngram_info[2]['terms']):,}",
                   f"{len(ngram_info[2]['terms']):,}", f"{int(ngram_info[2]['counts'].sum()):,}"],
                  ["trigram", f"{ngram_info[3]['matrix'].shape[0]} x {len(ngram_info[3]['terms']):,}",
                   f"{len(ngram_info[3]['terms']):,}", f"{int(ngram_info[3]['counts'].sum()):,}"]]),
        "",
        "### Top 15 bigrams",
        "",
        md_table(["rank", "bigram", "count"],
                 [[i + 1, gram, count] for i, (gram, count) in enumerate(top_bigrams[:15])]),
        "",
        "### Top 10 trigrams",
        "",
        md_table(["rank", "trigram", "count"],
                 [[i + 1, gram, count] for i, (gram, count) in enumerate(top_trigrams[:10])]),
        "",
        "## Part C - CBOW Word2Vec",
        "",
        f"* Architecture: **CBOW** (`sg=0`), vector size {args.vector_size}, window {args.window}, "
        f"min_count {args.min_count}, {args.epochs} epochs",
        f"* Word2Vec vocabulary: **{len(model.wv):,}** words",
        "",
        md_table(["query word", "nearest neighbours"],
                 [[word, ", ".join(f"{other} ({score:.3f})" for other, score in similar)]
                  for word, similar in neighbours]),
        "",
        "## Part D - LDA topics",
        "",
    ]
    if coherence is not None:
        report.append(f"Coherence (c_v): **{coherence:.4f}**")
        report.append("")
    topic_rows = []
    for topic_id, terms in topics:
        topic_rows.append([topic_id, topic_sizes.get(topic_id, 0),
                           ", ".join(word for word, _ in terms[:10])])
    report += [
        md_table(["topic", "reviews dominated", "top 10 words"], topic_rows),
        "",
        "### Example review per topic",
        "",
    ]
    for topic in range(args.topics):
        if topic in examples:
            report.append(f"* **Topic {topic}**: {clean_headline(examples[topic])}")
    report += [
        "",
        "## Outputs",
        "",
        "`top_words.csv`, `top_bigrams.csv`, `top_trigrams.csv`, `lda_topics.csv`, "
        "`review_corpus.csv`, `bow_summary.json`, `lab2_overview.png`",
        "",
    ]
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")

    save_plot(out / "lab2_overview.png", top_words, top_bigrams, topic_sizes, topics)
    print(f"\nReport          : {out / 'report.md'}")
    print(f"Summary JSON    : {out / 'bow_summary.json'}")
    print(f"Figure          : {out / 'lab2_overview.png'}")
    print("Done.")


if __name__ == "__main__":
    main()
