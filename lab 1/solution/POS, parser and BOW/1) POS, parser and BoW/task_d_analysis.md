# Task D: Analysis and Interpretation

This report addresses the questions formulated in Task D based on the outputs and observations from Tasks A, B, C, and E.

## 1. Which information is preserved by BoW?

Bag of Words (BoW) preserves the **vocabulary** and the **frequency** of each word within the document or audio segment. It tells us exactly *which* words are present and *how often* they occur. This makes BoW highly effective for document classification tasks where the presence of certain keywords is a strong indicator of the topic (e.g., classifying an email as spam if words like "free" and "money" appear frequently).

## 2. Which linguistic information is lost by BoW?

BoW completely loses all **syntactic and structural information**, including:
- **Word Order:** As demonstrated by our Task E implementation, reversing the word order of a sentence produces the exact same BoW vector. 
- **Grammar and Syntax:** Relationships between words (subject-verb-object) are lost. "The dog bit the man" and "The man bit the dog" have identical BoW representations, despite having completely opposite meanings.
- **Context and Semantic Nuance:** Sarcasm, negations (e.g., "not good" vs "good"), and contextual meanings of words are discarded because each word is treated as an isolated, independent entity.

## 3. Why can parsing provide information that BoW cannot?

Parsing (both Dependency and Constituency) analyzes the grammatical structure of a sentence, allowing it to provide information that BoW cannot:
- **Dependency Parsing** reveals the exact relationships between words (e.g., identifying which noun is the subject of the verb, and which adjectives modify which nouns). It tells us *who* did *what* to *whom*. 
- **Constituency Parsing** groups words into logical phrases (noun phrases, verb phrases), which helps in understanding the hierarchical structure of the sentence. 

By taking structural hierarchy and grammar rules into account, parsing can distinguish between sentences with identical vocabularies but different meanings (e.g., "The man bit the dog"), which BoW treats identically. Furthermore, as shown when we reorder words in Task E, the parse tree changes dramatically or breaks entirely, proving that parsing is sensitive to the word order and grammatical coherence that BoW ignores.
