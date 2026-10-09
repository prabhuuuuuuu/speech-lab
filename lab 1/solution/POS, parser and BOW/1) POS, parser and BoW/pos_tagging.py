import sys
import nltk
from nltk.tokenize import word_tokenize
from utils import get_text_from_input, modify_word_order

# Skip NLTK downloads to prevent network hangs
# nltk.download('punkt', quiet=True)
# nltk.download('punkt_tab', quiet=True)
# nltk.download('averaged_perceptron_tagger', quiet=True)
# nltk.download('averaged_perceptron_tagger_eng', quiet=True)

def perform_pos_tagging(text, label="Original"):
    pass
    print(f"Text: {text}")
    
    if not text.strip():
        print("Empty text provided.")
        return

    # Tokenize
    tokens = word_tokenize(text)
    print("\nTokens:")
    print(tokens)
    
    # POS Tagging
    pos_tags = nltk.pos_tag(tokens)
    print("\nPart-of-Speech Tags:")
    for token, tag in pos_tags:
        print(f"{token:15} -> {tag}")
    
    return pos_tags

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_val = sys.argv[1]
    else:
        # Default sample if no input is provided
        input_val = "The inventor built a incredibly fast car."
        
    print(f"Input: {input_val}")
    
    # Extract text from input (audio/text file, or raw string)
    text = get_text_from_input(input_val)
    
    if text:
        # Task A: POS Tagging on Original text
        perform_pos_tagging(text, label="Original")
        
        # Task E: Modify word order and see effect on POS Tagging
        modified_text = modify_word_order(text)
        perform_pos_tagging(modified_text, label="Modified (Reversed Order)")
