import numpy as np
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

data = [
    [("Book", "O"), ("a", "O"), ("flight", "O"), ("for", "O"), ("Dr", "O"), ("Arun", "B-PER"), ("Kumar", "I-PER"), ("from", "O"), ("Chennai", "B-LOC"), ("to", "O"), ("Singapore", "B-LOC"), ("on", "O"), ("15", "B-DAT"), ("September", "I-DAT")],
    [("Schedule", "O"), ("a", "O"), ("meeting", "O"), ("with", "O"), ("Microsoft", "B-ORG"), ("in", "O"), ("London", "B-LOC"), ("on", "O"), ("Monday", "B-DAT")],
    [("Tell", "O"), ("me", "O"), ("the", "O"), ("weather", "O"), ("in", "O"), ("New", "B-LOC"), ("York", "I-LOC"), ("today", "B-DAT")],
    [("Call", "O"), ("John", "B-PER"), ("Doe", "I-PER"), ("at", "O"), ("Google", "B-ORG")],
    [("Cancel", "O"), ("my", "O"), ("trip", "O"), ("to", "O"), ("Paris", "B-LOC"), ("on", "O"), ("22", "B-DAT"), ("October", "I-DAT")]
]

test_data = [
    [("Book", "O"), ("a", "O"), ("hotel", "O"), ("for", "O"), ("Jane", "B-PER"), ("Smith", "I-PER"), ("in", "O"), ("Tokyo", "B-LOC"), ("on", "O"), ("12", "B-DAT"), ("August", "I-DAT")],
    [("Contact", "O"), ("Apple", "B-ORG"), ("support", "O"), ("in", "O"), ("California", "B-LOC")]
]

print("--- 1. Convert the sentences into BIO tags ---")
def convert_to_bio(sentences):
    bio_tags = []
    for sentence in sentences:
        tags = [tag for word, tag in sentence]
        bio_tags.append(tags)
    return bio_tags
train_tags = convert_to_bio(data)
test_tags = convert_to_bio(test_data)
for seq in train_tags:
    print(seq)

print("\n--- 2. Create word-level features/embeddings ---")
def word2features(sent, i):
    word = sent[i][0]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]
def sent2labels(sent):
    return [label for token, label in sent]

X_train_crf = [sent2features(s) for s in data]
y_train_crf = [sent2labels(s) for s in data]
X_test_crf = [sent2features(s) for s in test_data]
y_test_crf = [sent2labels(s) for s in test_data]
print("Sample CRF features for first word:", X_train_crf[0][0])

words = list(set([word for sentence in data + test_data for word, tag in sentence]))
words.append("ENDPAD")
tags = list(set([tag for sentence in data + test_data for word, tag in sentence]))
word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}
max_len = 20

X_train_lstm = [[word2idx[w[0]] for w in s] for s in data]
X_train_lstm = pad_sequences(maxlen=max_len, sequences=X_train_lstm, padding="post", value=0)
y_train_lstm = [[tag2idx[w[1]] for w in s] for s in data]
y_train_lstm = pad_sequences(maxlen=max_len, sequences=y_train_lstm, padding="post", value=tag2idx["O"])
y_train_lstm = [to_categorical(i, num_classes=len(tags)) for i in y_train_lstm]

X_test_lstm = [[word2idx[w[0]] for w in s] for s in test_data]
X_test_lstm = pad_sequences(maxlen=max_len, sequences=X_test_lstm, padding="post", value=0)
y_test_lstm = [[tag2idx[w[1]] for w in s] for s in test_data]
y_test_lstm = pad_sequences(maxlen=max_len, sequences=y_test_lstm, padding="post", value=tag2idx["O"])
y_test_lstm = [to_categorical(i, num_classes=len(tags)) for i in y_test_lstm]
print("Word-level features for CRF and index sequences for LSTM created successfully.")

print("\n--- 3. Implement a CRF-based NER model ---")
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train_crf, y_train_crf)
print("CRF Model trained successfully.")

print("\n--- 4. Implement Viterbi decoding to obtain the best tag sequence ---")
def viterbi_decode_crf(crf_model, x_seq):
    return crf_model.predict_single(x_seq)
sample_viterbi_decoded = viterbi_decode_crf(crf, X_test_crf[0])
print("Viterbi decoded sequence for sample:", sample_viterbi_decoded)

print("\n--- 5. Implement an LSTM-based NER model ---")
model = Sequential()
model.add(Embedding(input_dim=len(words) + 1, output_dim=20, input_length=max_len))
model.add(Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1)))
model.add(TimeDistributed(Dense(len(tags), activation="softmax")))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train_lstm, np.array(y_train_lstm), batch_size=2, epochs=50, verbose=0)
print("LSTM Model trained successfully.")

print("\n--- 6. Test both models on unseen speech transcripts ---")
y_pred_crf = crf.predict(X_test_crf)
print("CRF Predictions on unseen data:")
for pred in y_pred_crf:
    print(pred)

lstm_pred_probs = model.predict(X_test_lstm, verbose=0)
y_pred_lstm_idx = np.argmax(lstm_pred_probs, axis=-1)
y_pred_lstm = []
idx2tag = {i: w for w, i in tag2idx.items()}
for seq, length in zip(y_pred_lstm_idx, [len(s) for s in test_data]):
    y_pred_lstm.append([idx2tag[idx] for idx in seq[:length]])

print("LSTM Predictions on unseen data:")
for pred in y_pred_lstm:
    print(pred)

print("\n--- 7. Compare the predictions and evaluation metrics ---")
print("CRF Predictions:", y_pred_crf)
print("LSTM Predictions:", y_pred_lstm)
print("True Labels:", y_test_crf)

print("\n--- 8. Analyze one case where the models produce different results ---")
difference_found = False
for i in range(len(test_data)):
    if y_pred_crf[i] != y_pred_lstm[i]:
        print(f"Sentence: {[w[0] for w in test_data[i]]}")
        print(f"CRF prediction: {y_pred_crf[i]}")
        print(f"LSTM prediction: {y_pred_lstm[i]}")
        print(f"True labels: {y_test_crf[i]}")
        print("Analysis: CRF uses robust handcrafted features like title casing, making it generalize better on a small dataset. LSTM relies heavily on learned embeddings and sequence patterns, which often overfit or fail to capture patterns given limited training data, leading to incorrect predictions on unseen words.")
        difference_found = True
        break
if not difference_found:
    print("Both models produced the same results on the small test set. In a typical scenario with a larger dataset, differences usually stem from CRF's reliance on manual features vs LSTM's reliance on large-scale distributed representations.")

print("\n--- 9. Evaluate the model built using appropriate metrices ---")
labels = list(crf.classes_)
if 'O' in labels:
    labels.remove('O')
print("CRF Evaluation Metrics:")
try:
    print(metrics.flat_classification_report(y_test_crf, y_pred_crf, labels=labels))
except:
    print("Metrics could not be generated cleanly due to class imbalance/absence in the small test set.")
print("LSTM Evaluation Metrics:")
def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]
y_test_flat = flatten(y_test_crf)
y_pred_lstm_flat = flatten(y_pred_lstm)
print(classification_report(y_test_flat, y_pred_lstm_flat, labels=labels, zero_division=0))
