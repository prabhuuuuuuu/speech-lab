docs = [
    "the inventor built the car",
    "the car was fast",
    "he saw the fast car",
    "the inventor built a clock"
]

vocabulary = []
for doc in docs:
    words = doc.split()
    for word in words:
        if word not in vocabulary:
            vocabulary.append(word)


vocabulary.sort()

print("The Bag:")
print(vocabulary)


for i, doc in enumerate(docs, 1):
    words = doc.split()
    vector = []
    

    for vocab_word in vocabulary:
        word_count = words.count(vocab_word)
        vector.append(word_count)
        
    print(f"Doc {i}: '{doc}'")
    print(f"Bag of words: {vector}\n")
