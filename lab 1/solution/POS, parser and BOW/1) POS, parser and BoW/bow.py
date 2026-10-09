import sys
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from utils import get_text_from_input, modify_word_order
from nltk.util import skipgrams

def skipgram_analyzer(text):
    tokens = nltk.word_tokenize(text.lower())
    return [" ".join(sg) for sg in skipgrams(tokens, 2, 2)]

if __name__ == "__main__":
    hardcoded_docs = [
        "the inventor built the car",
        "the car was fast",
        "he saw the fast car",
        "the inventor built a clock",
        "the fast clock was built by him"
    ]
    


    print("\nBinary Bag of Words (Hardcoded Docs)")
    vectorizer_bin = CountVectorizer(binary=True)
    try:
        X_bin = vectorizer_bin.fit_transform(hardcoded_docs)
        print("Vocabulary:", list(vectorizer_bin.get_feature_names_out()))
        for i, vec in enumerate(X_bin.toarray(), 1):
            print(f"Doc {i}: {vec}")
    except ValueError:
        print("Failed to process hardcoded documents (Binary).")

    print("\nSkip-Gram Bag of Words (Hardcoded Docs)")
    vectorizer_sg = CountVectorizer(analyzer=skipgram_analyzer)
    try:
        X_sg = vectorizer_sg.fit_transform(hardcoded_docs)
        print("Vocabulary:", list(vectorizer_sg.get_feature_names_out()))
        for i, vec in enumerate(X_sg.toarray(), 1):
            print(f"Doc {i}: {vec}")
    except Exception as e:
        print(f"Failed to process hardcoded skip-grams: {e}")
