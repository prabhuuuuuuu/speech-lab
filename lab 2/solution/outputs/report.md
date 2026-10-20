# Lab 2 - BoW, N-grams, Word2Vec and Topic Modeling

* Dataset: **synthetic review corpus (2000 reviews)**
* Reviews used: **2000**
* Cleaning: lower-case, HTML/URL/e-mail removal, punctuation and number removal, contraction expansion, stop-word removal, WordNet lemmatization (POS-aware)
* Tokens: **36,859** (18.4 per review)

## Part A - Bag of Words

### Top 20 frequent words

| rank | word | count |
|---|---|---|
| 1 | flavour | 1436 |
| 2 | fresh | 743 |
| 3 | kitchen | 656 |
| 4 | tea | 650 |
| 5 | chocolate | 612 |
| 6 | aroma | 534 |
| 7 | value | 517 |
| 8 | taste | 512 |
| 9 | balance | 503 |
| 10 | exactly | 490 |
| 11 | coffee | 452 |
| 12 | poor | 441 |
| 13 | money | 430 |
| 14 | price | 430 |
| 15 | quality | 422 |
| 16 | open | 363 |
| 17 | spice | 357 |
| 18 | good | 340 |
| 19 | long | 340 |
| 20 | strength | 322 |

* Vocabulary size: **312**
* BoW matrix: **2000 x 312**
* Non-zero cells: **31,198**
* Sparsity: **95.00%**

## Part B - N-grams

| n-gram order | matrix | unique n-grams | total tokens |
|---|---|---|---|
| unigram | 2000 x 312 | 312 | 36,859 |
| bigram | 2000 x 4,560 | 4,560 | 34,859 |
| trigram | 2000 x 12,636 | 12,636 | 32,859 |

### Top 15 bigrams

| rank | bigram | count |
|---|---|---|
| 1 | value money | 430 |
| 2 | flavour balance | 414 |
| 3 | quality consistent | 302 |
| 4 | repeat purchase | 302 |
| 5 | purchase kitchen | 302 |
| 6 | guest ask | 285 |
| 7 | flavour poor | 281 |
| 8 | good value | 270 |
| 9 | content fresh | 245 |
| 10 | home long | 243 |
| 11 | shopping list | 217 |
| 12 | weekly shopping | 217 |
| 13 | soon open | 197 |
| 14 | regret buying | 174 |
| 15 | description label | 171 |

### Top 10 trigrams

| rank | trigram | count |
|---|---|---|
| 1 | repeat purchase kitchen | 302 |
| 2 | good value money | 270 |
| 3 | weekly shopping list | 217 |
| 4 | taste description label | 170 |
| 5 | poor value money | 160 |
| 6 | receive tasted previous | 138 |
| 7 | batch receive tasted | 138 |
| 8 | guest ask flavour | 93 |
| 9 | shopping list flavour | 81 |
| 10 | list flavour balance | 72 |

## Part C - CBOW Word2Vec

* Architecture: **CBOW** (`sg=0`), vector size 100, window 5, min_count 3, 30 epochs
* Word2Vec vocabulary: **305** words

| query word | nearest neighbours |
|---|---|
| coffee | bold (0.737), body (0.727), mellow (0.713), robust (0.707), crema (0.688), watery (0.667) |
| tea | fragrance (0.810), leaf (0.798), delicate (0.788), infusion (0.770), floral (0.765), artificial (0.693) |
| chocolate | melt (0.786), dark (0.767), velvety (0.761), snap (0.754), creamy (0.743), sweetness (0.731) |
| snack | handy (0.773), oiliness (0.771), flavourful (0.755), spicy (0.754), office (0.737), crunch (0.719) |
| spice | freshness (0.793), authentic (0.647), purity (0.628), free (0.625), vibrant (0.604), strong (0.604) |
| fresh | balanced (0.631), balance (0.547), clean (0.527), content (0.510), authentic (0.484), secure (0.480) |
| packaging | package (0.778), content (0.642), secure (0.483), properly (0.470), neat (0.468), outer (0.467) |
| quality | hesitation (0.321), package (0.301), recommend (0.264), properly (0.262), price (0.253), content (0.247) |
| price | money (0.884), value (0.809), good (0.721), family (0.559), large (0.550), gram (0.546) |
| recommend | bin (0.718), grit (0.592), hard (0.586), lump (0.572), contain (0.563), stone (0.556) |

## Part D - LDA topics

Coherence (c_v): **0.4369**

| topic | reviews dominated | top 10 words |
|---|---|---|
| 0 | 713 | kitchen, fresh, tea, balance, exactly, long, coffee, aroma, good, spice |
| 1 | 380 | chocolate, melt, cocoa, dark, sweetness, fresh, bar, snap, truffle, sweet |
| 2 | 364 | fresh, taste, crunch, chip, snack, value, texture, crunchy, roast, spice |
| 3 | 124 | coffee, weekly, list, add, shopping, roast, stale, bean, broken, mix |
| 4 | 419 | tea, poor, taste, aroma, stale, bitter, new, smell, colour, label |

### Example review per topic

* **Topic 0**: I keep the turmeric powder in the kitchen because the purity is fragrant. The flavour is well balanced and the purity is fragrant. The spices tasted fresh and the flavour was authentic. The quality has been consistent...
* **Topic 1**: The cocoa nibs had a sugary texture. The flavour was poor and the texture was sugary. It tasted waxy and left a strange coating in my mouth. The packaging was damaged and the product looked old. Most of the pack went...
* **Topic 2**: I noticed a light spice in the roasted nuts. The flavour is well balanced and the flavour is light. The pack stayed fresh for a week after opening because of the zip lock. I would buy this again without hesitation.
* **Topic 3**: I could not finish the instant coffee because the crema was bitter. The flavour was poor and the crema was bitter. The vacuum seal was broken and the coffee had gone stale. I would not recommend this to anyone.
* **Topic 4**: The lemon tea smelled bitter as soon as the pack was opened. The leaf was bitter and the aftertaste was harsh. The leaves were mostly dust and the infusion turned cloudy. The packaging was torn and the product looked...

## Outputs

`top_words.csv`, `top_bigrams.csv`, `top_trigrams.csv`, `lda_topics.csv`, `review_corpus.csv`, `bow_summary.json`, `lab2_overview.png`
