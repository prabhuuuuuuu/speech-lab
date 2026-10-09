# Speech & NLP Lab - Labs 1 to 6

| Lab | Topic | Solution folder | Main script |
|---|---|---|---|
| 1 | POS tagging, parsing, Bag of Words (text + speech) | `lab 1/solution/POS, parser and BOW/1) POS, parser and BoW/` | `pos_tagging.py`, `parsing.py`, `bow.py`, `HOC.py` |
| 2 | Word embeddings: Word2Vec, GloVe, FastText | `lab 2/solution/3) Word Embeddings (W2Vec, Glove, Fasttext)/3) Word Embeddings (W2Vec, Glove, Fasttext)/` | `main.py` |
| 3 | NER for speech transcripts: BIO, CRF + Viterbi, LSTM | `lab 3/solution/` | `ner_crf_lstm.py` |
| 4 | RNN-based NER + model/preprocessing analysis | `lab 4/solution/` | `rnn_ner.py` |
| 5 | Text summarisation (statistical, neural, pre-trained) + ROUGE | `lab 5/solution/` | `main.py` |
| 6 | Knowledge-based vs deep-learning question answering | `lab 6/solution/` | `qa_system.py` |

---

## 1. One-time setup (Windows PowerShell)

Python 3.10 - 3.13 works. From the `speech lab` folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; [nltk.download(p) for p in ['punkt','punkt_tab','averaged_perceptron_tagger','averaged_perceptron_tagger_eng','maxent_ne_chunker','maxent_ne_chunker_tab','words']]"
```

(macOS / Linux: `python3 -m venv .venv` and `source .venv/bin/activate`.)

Notes
* No GPU is needed; everything runs on CPU.
* Labs 5 and 6 download pre-trained models from Hugging Face **the first time** they run
  (internet required, about 5 GB in total, cached in `%USERPROFILE%\.cache\huggingface`). Later runs work offline.
* Folder names contain spaces - always wrap paths in quotes when you `cd`.

---

## Lab 1 - POS tagging, parsing and Bag of Words

```powershell
cd "lab 1\solution\POS, parser and BOW\1) POS, parser and BoW"
python pos_tagging.py "The inventor built an incredibly fast car."   # A) tokens + POS tags (+ E: reversed order)
python pos_tagging.py harvard.wav                                     # same, from speech (Google STT, needs internet)
python parsing.py "The inventor built an incredibly fast car."       # B) dependency table + constituency tree,
                                                                      #    opens dependency_tree.html in the browser
python parsing.py harvard.wav
python bow.py                                                         # C) vocabulary + BoW vectors (binary, skip-gram)
python code2.py                                                       # C) BoW built by hand
python HOC.py "The aeroplane flies in the blue sky"                   # E) word order changed -> POS, parse, BoW again
echo "Your paragraph here." | python code.py                         # full pipeline from stdin (+ text-to-speech output.mp3)
```

D) The written analysis is in `task_d_analysis.md`.
If speech recognition has no internet, `utils.py` falls back to the Harvard-sentences text.

## Lab 2 - Word2Vec, GloVe, FastText

```powershell
cd "lab 2\solution\3) Word Embeddings (W2Vec, Glove, Fasttext)\3) Word Embeddings (W2Vec, Glove, Fasttext)"
python main.py
```

Uses `corpus.txt` (must be in the current folder). Prints Parts A-E (CBOW / Skip-gram with
windows 2 and 4, GloVe from a co-occurrence matrix, FastText on unseen `machine-learning`, the
cosine-similarity table) and opens the PCA plot (Part F) - close the plot window to finish.

## Lab 3 - NER for voice-assistant transcripts (CRF + Viterbi vs LSTM)

```powershell
cd "lab 3\solution"
python ner_crf_lstm.py                                       # all 9 tasks, ~1-2 min
python ner_crf_lstm.py --text "call Ravi Kumar from Infosys in Pune on 3 March"
python ner_crf_lstm.py --audio my_command.wav                # transcribe a WAV with Google STT, then tag it
```

| Task | Where in the output |
|---|---|
| 1 BIO tags | `TASK 1` - 60 annotated training transcripts (`ner_data.py`), converted from inline markup to BIO |
| 2 Features / embeddings | `TASK 2` - hand-crafted word features (shape, case, suffixes, months, titles, context +-2) for the CRF; word + char-CNN embeddings for the LSTM |
| 3 CRF | `TASK 3` - linear-chain CRF written from scratch (forward algorithm, L-BFGS); learned transition matrix and top features |
| 4 Viterbi | `TASK 4` - Viterbi written from scratch in NumPy, full trellis printed, checked against brute force over all 531,441 paths |
| 5 LSTM | `TASK 5` - BiLSTM + character-CNN tagger (PyTorch) |
| 6 Unseen transcripts | `TASK 6` - 20 test transcripts with mostly unseen names |
| 7 Compare | `TASK 7` - P/R/F1 of CRF-Viterbi, CRF-greedy, sklearn-crfsuite reference, LSTM; invalid-BIO counts |
| 8 Disagreement case | `TASK 8` - token-by-token explanation (CRF feature contributions vs LSTM softmax) |
| 9 Metrics | `TASK 9` - entity-level P/R/F1 per type (seqeval cross-check) + lower-cased ASR robustness test |

Typical result: CRF F1 approx. 0.94, BiLSTM approx. 0.88 (the CRF wins on 60 sentences; it drops much more on lower-cased text).

## Lab 4 - RNN-based NER

```powershell
cd "lab 4\solution"
python rnn_ner.py --quick                     # small run, a few minutes
python rnn_ner.py                             # full run: 15 configurations (~20-30 min on CPU)
python rnn_ner.py --text "Sundar Pichai visited Chennai on 5 March 2027"
python rnn_ner.py --conll-train train.txt --conll-dev valid.txt --conll-test test.txt   # e.g. CoNLL-2003
```

* Data: synthetic NER corpus (`ner_dataset.py`) - 30 % of names and 10 sentence templates are held out of
  training, so the test set really measures generalisation to unseen entities. Any CoNLL file can be used instead.
* Experiments: **A** RNN / GRU / LSTM, uni- vs bi-directional; **B** preprocessing - lower-casing, digit
  normalisation, `<UNK>` handling (min-freq + word dropout), casing feature, char-CNN, all combined;
  **C** capacity - 2 layers, small hidden size.
* Metrics: entity-level micro P/R/F1, macro-F1, token accuracy, recall on seen vs unseen entities, F1 on
  lower-cased sentences, per-type report, confusion matrix, error taxonomy.
* Outputs: `results/experiments.csv`, `results/experiments.png`.
  Typical full-run result: plain BiLSTM approx. 0.88 F1 -> full pipeline approx. 0.94-0.95 F1.

## Lab 5 - Text summarisation of student feedback

```powershell
cd "lab 5\solution"
python main.py --skip-neural --skip-pretrained    # (a) + (b) + (d) only - seconds, offline
python main.py --skip-pretrained                  # + from-scratch RNN/LSTM/Seq2Seq/attention/Transformer (~8 min)
python main.py                                    # everything incl. BERT, BART, PEGASUS, T5 (~15 min on CPU,
                                                  #   longer on the first run while models download)
python main.py --input my_report.txt --reference my_summary.txt --k 4
python main.py --bart sshleifer/distilbart-cnn-12-6 --pegasus google/pegasus-xsum    # smaller / other models
```

| Part | Implementation |
|---|---|
| a) Pre-processing | `preprocess.py` - removes title, metadata lines, separators, page numbers, bracketed notes, URLs/e-mails, boiler-plate, repeated punctuation; sentence segmentation; tokenisation + stop-words |
| b) Statistical | `statistical_summarizers.py` (from scratch) - Frequency, TF-IDF, TextRank (PageRank), LexRank (idf-cosine graph), Feature-based (position, length, frequency, TF-IDF, heading keywords, numbers) |
| c) Deep learning, trained from scratch | `neural_summarizers.py` - hierarchical RNN and BiLSTM extractors, self-attention extractor (multi-head attention written by hand), RNN / LSTM encoder-decoder, Seq2Seq + Bahdanau attention, Transformer encoder-decoder; trained on the synthetic feedback corpus in `feedback_corpus.py` |
| c) Pre-trained | `pretrained_summarizers.py` - BERT and SBERT extractive (embeddings + K-Means), BART (`facebook/bart-large-cnn`), PEGASUS (`google/pegasus-cnn_dailymail`), T5 (`t5-small`) |
| d) Evaluation | `rouge.py` (from scratch) - ROUGE-1, ROUGE-2, ROUGE-L precision / recall / F1 |

Input: `data/student_feedback.txt`; reference: `data/reference_summary.txt`.
Outputs: `outputs/summaries.md`, `outputs/rouge_scores.csv`, `outputs/rouge_chart.png`.
Read the "CAUTION" note printed at the end: the from-scratch abstractive models were trained on targets
written in the same style as the reference, so their high ROUGE partly reflects that shared wording.

## Lab 6 - Knowledge-based vs deep-learning QA

```powershell
cd "lab 6\solution"
python qa_system.py --kb-only                         # knowledge-based part only (no downloads)
python qa_system.py                                   # KB + DistilBERT + RoBERTa + hybrid, full evaluation
python qa_system.py --ask "Where was Sundar Pichai born?" --ask "Who runs Google?"
python qa_system.py --interactive
```

* Knowledge base: 70 triples in `kb_data.py` (exported to `outputs/knowledge_base.json`, drawn in
  `outputs/knowledge_graph.png`).
* Knowledge-based QA (`kbqa.py`): entity linking (aliases + fuzzy matching for typos), relation detection,
  expected-answer-type detection, SPARQL-style forward/inverse queries, multi-hop reasoning over
  `located_in`, COUNT aggregation, abstains when unsure. Shows the query and the evidence path.
* Deep-learning QA (`neural_qa.py`): TF-IDF retriever + pre-trained extractive readers
  `distilbert-base-cased-distilled-squad` (SQuAD 1.1) and `deepset/roberta-base-squad2` (SQuAD 2.0, can say "no answer").
  Start/end span selection is written out explicitly.
* Evaluation: 34 questions in 9 categories (simple, paraphrase, inverse, multi-hop, typo, aggregation,
  text-only, KB-only, unanswerable) - Exact Match, F1, coverage, precision on answered questions,
  abstention accuracy, latency, per-category EM. Results in `outputs/qa_results.json`.
  Typical: KB EM 0.91, DistilBERT 0.56, RoBERTa 0.76, hybrid 1.00.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `OSError: Can't find model 'en_core_web_sm'` | `python -m spacy download en_core_web_sm` |
| NLTK `LookupError: Resource punkt ... not found` | run the `nltk.download` line from the setup section |
| Hugging Face download fails / is slow | check internet; the first run of Labs 5 and 6 needs it. Lab 5 skips a failed model and continues |
| Lab 5 too slow | `--skip-pretrained`, or `--epochs 4 --train-docs 100` |
| `UnicodeEncodeError` in the console | `$env:PYTHONIOENCODING="utf-8"` before running |
| Lab 1 `ner_system.py` needs TensorFlow | that older NER script is superseded by Lab 3 (PyTorch); install `tensorflow` only if you want to run it |
