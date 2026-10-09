
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from gtts import gTTS
import os

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('maxent_ne_chunker_tab', quiet=True) 
nltk.download('words', quiet=True)


def process_text_pipeline(text):
    if not text.strip():
        print("Empty text")
        return
    
    tokens = word_tokenize(text)
    print("Tokenization")
    print(tokens)
    print(f"Total number of tokens: {len(tokens)}\n")

    pos_tags = nltk.pos_tag(tokens)
    print("POS Tag")
    print(pos_tags)

    parsed_tree = nltk.ne_chunk(pos_tags)
    print("Parsing")
    print(parsed_tree)
    print("")
    
    bag_of_words = dict(FreqDist(tokens))
    print("Bag Of Words")

    for word, count in sorted(bag_of_words.items(), key=lambda item: item[1], reverse=True):
        print(f"  '{word}': {count}")
    print("")
    
    try:
        tts = gTTS(text=text, lang='en')
        output_filename = "output.mp3"
        tts.save(output_filename)
    except Exception as e:
        print(f"Speech generation failed: {e}")

if __name__ == "__main__":
    
    import sys
    user_paragraph = sys.stdin.read()
    
    process_text_pipeline(user_paragraph)
