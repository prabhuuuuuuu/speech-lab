import re
import numpy as np
import matplotlib.pyplot as plt

from gensim.models import Word2Vec, FastText
from sklearn.decomposition import PCA


with open("corpus.txt", "r", encoding="utf-8") as file:
    sentences = file.readlines()


def preprocess(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-zA-Z0-9\s-]", "", sentence)
    return sentence.split()


tokenized_corpus = [preprocess(sentence) for sentence in sentences]

print("===== PART A: CORPUS PREPARATION =====")
print("Number of sentences:", len(tokenized_corpus))

print("\nTokenized sentences:")
for i, sentence in enumerate(tokenized_corpus[:5], 1):
    print(i, sentence)

vocabulary = sorted(set(
    word
    for sentence in tokenized_corpus
    for word in sentence
))

print("\nVocabulary size:", len(vocabulary))

required_words = [
    "learn",
    "learning",
    "learned",
    "learner",
    "machine",
    "machinelearning",
    "cybersecurity",
    "healthcare"
]

print("\nRequired words:")
for word in required_words:
    print(word, "->", "Present" if word in vocabulary else "Not Present")

print("\nUnseen word:")
print(
    "machine-learning ->",
    "Present" if "machine-learning" in vocabulary else "Not Present"
)


def train_word2vec(model_type, window):
    sg = 0 if model_type == "CBOW" else 1

    model = Word2Vec(
        sentences=tokenized_corpus,
        vector_size=100,
        window=window,
        min_count=1,
        workers=4,
        sg=sg,
        seed=42,
        epochs=100
    )

    return model


selected_words = [
    "learning",
    "machine",
    "intelligence",
    "student"
]


word2vec_models = {}

print("\n===== PART B: WORD2VEC =====")

for model_type in ["CBOW", "Skip-gram"]:

    for window in [2, 4]:

        model = train_word2vec(model_type, window)

        key = f"{model_type}_window_{window}"
        word2vec_models[key] = model

        print("\n------------------------------")
        print(model_type, "Window =", window)
        print("------------------------------")

        for word in selected_words:

            print("\nVector for", word, ":")
            print(model.wv[word][:10])

            print("5 most similar:")
            print(model.wv.most_similar(word, topn=5))


print("\n===== PART B COMPARISON =====")

for key, model in word2vec_models.items():

    print("\n", key)

    for word in selected_words:
        similar = model.wv.most_similar(word, topn=5)
        print(word, ":", [x[0] for x in similar])


def build_cooccurrence_matrix(corpus, vocabulary, window=4):
    word_to_index = {
        word: i for i, word in enumerate(vocabulary)
    }

    matrix = np.zeros(
        (len(vocabulary), len(vocabulary)),
        dtype=float
    )

    for sentence in corpus:

        for i, word in enumerate(sentence):

            start = max(0, i - window)
            end = min(len(sentence), i + window + 1)

            for j in range(start, end):

                if i == j:
                    continue

                context_word = sentence[j]

                if context_word in word_to_index:
                    matrix[
                        word_to_index[word],
                        word_to_index[context_word]
                    ] += 1

    return matrix


def train_glove(corpus, vocabulary, dimensions=50, window=4, epochs=200):
    matrix = build_cooccurrence_matrix(
        corpus,
        vocabulary,
        window
    )

    matrix = np.log1p(matrix)

    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

    dimensions = min(dimensions, len(S))

    vectors = U[:, :dimensions] @ np.diag(
        np.sqrt(S[:dimensions])
    )

    return {
        word: vectors[i]
        for i, word in enumerate(vocabulary)
    }


glove = train_glove(
    tokenized_corpus,
    vocabulary
)


def cosine_similarity(vector1, vector2):
    denominator = (
        np.linalg.norm(vector1) *
        np.linalg.norm(vector2)
    )

    if denominator == 0:
        return 0

    return np.dot(vector1, vector2) / denominator


def glove_similar_words(word, topn=5):

    if word not in glove:
        return []

    scores = []

    for other_word in glove:

        if other_word == word:
            continue

        score = cosine_similarity(
            glove[word],
            glove[other_word]
        )

        scores.append(
            (other_word, score)
        )

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return scores[:topn]


print("\n===== PART C: GLOVE =====")

for word in selected_words:

    print("\nVector for", word, ":")
    print(glove[word][:10])

    print("5 most similar:")
    print(glove_similar_words(word))

print("\nDifference between GloVe and Word2Vec similarity results:")

w2v_model_for_comparison = word2vec_models.get("Skip-gram_window_4")
if w2v_model_for_comparison:
    for word in selected_words:
        glove_sim = set([x[0] for x in glove_similar_words(word)])
        w2v_sim = set([x[0] for x in w2v_model_for_comparison.wv.most_similar(word, topn=5)])
        print(f"\nWord: {word}")
        print(f"Words only in GloVe top 5: {glove_sim - w2v_sim}")
        print(f"Words only in Word2Vec top 5: {w2v_sim - glove_sim}")


print("\n===== GLOVE GLOBAL CONTEXT ANALYSIS =====")

for word in [
    "learning",
    "machine",
    "intelligence",
    "student"
]:

    print(
        word,
        "->",
        glove_similar_words(word)
    )


print("\n===== PART D: FASTTEXT =====")

fasttext_model = FastText(
    sentences=tokenized_corpus,
    vector_size=100,
    window=4,
    min_count=1,
    workers=4,
    sg=1,
    seed=42,
    epochs=100
)


fasttext_words = [
    "learn",
    "learning",
    "learned",
    "learner"
]


for word in fasttext_words:

    vector = fasttext_model.wv[word]

    print("\n", word)
    print("Vector:")
    print(vector[:10])


unseen_word = "machine-learning"

print("\nUnseen word:", unseen_word)

try:

    vector = fasttext_model.wv[unseen_word]

    print("Embedding generated: YES")
    print("Vector:")
    print(vector[:10])

except KeyError:

    print("Embedding generated: NO")


print("\n===== PART E: COSINE SIMILARITY =====")

word_pairs = [
    ("learning", "learner"),
    ("machine", "intelligence"),
    ("student", "learner"),
    ("artificial", "intelligence"),
    ("learn", "learning")
]


w2v_model = word2vec_models["Skip-gram_window_4"]


print(
    f"{'Word Pair':30}"
    f"{'Word2Vec':15}"
    f"{'GloVe':15}"
    f"{'FastText':15}"
)


similarity_results = []


for word1, word2 in word_pairs:

    w2v_score = cosine_similarity(
        w2v_model.wv[word1],
        w2v_model.wv[word2]
    )

    glove_score = cosine_similarity(
        glove[word1],
        glove[word2]
    )

    fasttext_score = cosine_similarity(
        fasttext_model.wv[word1],
        fasttext_model.wv[word2]
    )

    similarity_results.append(
        (
            word1,
            word2,
            w2v_score,
            glove_score,
            fasttext_score
        )
    )

    pair = f"{word1} - {word2}"

    print(
        f"{pair:30}"
        f"{w2v_score:<15.4f}"
        f"{glove_score:<15.4f}"
        f"{fasttext_score:<15.4f}"
    )


print("\n===== PART E: ANALYSIS =====")

morphological_pairs = [
    ("learning", "learner"),
    ("learn", "learning")
]


for word1, word2 in morphological_pairs:

    scores = {
        "Word2Vec": cosine_similarity(
            w2v_model.wv[word1],
            w2v_model.wv[word2]
        ),
        "GloVe": cosine_similarity(
            glove[word1],
            glove[word2]
        ),
        "FastText": cosine_similarity(
            fasttext_model.wv[word1],
            fasttext_model.wv[word2]
        )
    }

    best_model = max(
        scores,
        key=scores.get
    )

    print(
        f"{word1} - {word2}:",
        best_model,
        "=", 
        round(scores[best_model], 4)
    )


print("\nRare/unseen word handling:")
print("FastText can generate a vector for machine-learning using character subwords.")


print("\nSemantic relationship:")
semantic_scores = {}

for word1, word2 in [
    ("machine", "intelligence"),
    ("student", "learner"),
    ("artificial", "intelligence")
]:

    semantic_scores[
        f"{word1}-{word2}"
    ] = {
        "Word2Vec": cosine_similarity(
            w2v_model.wv[word1],
            w2v_model.wv[word2]
        ),
        "GloVe": cosine_similarity(
            glove[word1],
            glove[word2]
        ),
        "FastText": cosine_similarity(
            fasttext_model.wv[word1],
            fasttext_model.wv[word2]
        )
    }

for pair, scores in semantic_scores.items():

    best = max(
        scores,
        key=scores.get
    )

    print(
        pair,
        "->",
        best,
        "=",
        round(scores[best], 4)
    )


print("\n===== PART F: PCA VISUALIZATION =====")

visualization_words = [
    "artificial",
    "intelligence",
    "machine",
    "learning",
    "learn",
    "learned",
    "learner",
    "student",
    "students",
    "cybersecurity",
    "healthcare",
    "applications"
]


vectors = []

valid_words = []

for word in visualization_words:

    if word in w2v_model.wv:

        vectors.append(
            w2v_model.wv[word]
        )

        valid_words.append(word)


pca = PCA(n_components=2)

reduced_vectors = pca.fit_transform(vectors)


plt.figure(figsize=(10, 7))

plt.scatter(
    reduced_vectors[:, 0],
    reduced_vectors[:, 1]
)


for i, word in enumerate(valid_words):

    plt.annotate(
        word,
        (
            reduced_vectors[i, 0],
            reduced_vectors[i, 1]
        )
    )


plt.title("Word2Vec Embeddings - PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()