"""
Lab 2 - Review corpus for BoW, n-gram, Word2Vec and topic-modelling experiments.

The default corpus is a deterministic set of 2,000 product reviews written in the
style of the Amazon Fine Food Reviews / Flipkart Product Reviews datasets.  A real
CSV or JSON dataset can be supplied instead with ``--csv`` or ``--json``; the text
column is detected automatically or can be named with ``--text-column``.
"""

from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path

CATEGORIES = {
    "coffee": {
        "products": [
            "coffee", "coffee beans", "ground coffee", "instant coffee",
            "cold brew", "espresso", "filter coffee", "latte mix",
        ],
        "aspects": [
            "aroma", "roast", "flavour", "crema", "grind", "bitterness",
            "aftertaste", "strength", "body",
        ],
        "good": [
            "rich", "smooth", "bold", "fresh", "aromatic", "balanced",
            "mellow", "robust",
        ],
        "bad": [
            "stale", "bitter", "burnt", "weak", "watery", "harsh",
            "flat", "oily",
        ],
        "details_good": [
            "The beans were evenly roasted and the bag smelled wonderful when I opened it.",
            "It brews a clean cup without any muddy sediment.",
            "A little goes a long way, so the pack lasts for weeks.",
            "The crema lasted until the last sip of the espresso.",
            "It works equally well in a French press and in the drip machine.",
        ],
        "details_bad": [
            "The packet was past its best before date and the beans smelled damp.",
            "The grind was far too coarse for my espresso machine.",
            "There were broken beans and a lot of dust at the bottom of the bag.",
            "The roast tasted burnt and left a harsh aftertaste.",
            "The vacuum seal was broken and the coffee had gone stale.",
        ],
    },
    "tea": {
        "products": [
            "green tea", "black tea", "herbal tea", "masala chai",
            "jasmine tea", "tea bags", "oolong tea", "lemon tea",
        ],
        "aspects": [
            "aroma", "colour", "flavour", "strength", "leaf", "infusion",
            "aftertaste", "fragrance",
        ],
        "good": [
            "fresh", "fragrant", "smooth", "light", "refreshing", "floral",
            "crisp", "delicate",
        ],
        "bad": [
            "stale", "dusty", "bitter", "weak", "tasteless", "harsh",
            "artificial", "flat",
        ],
        "details_good": [
            "The leaves opened nicely in hot water and gave a clear amber cup.",
            "It tastes light and refreshing without any bitterness.",
            "The tea bags did not tear and the strings stayed attached.",
            "A cup after dinner has become part of my daily routine.",
            "The second infusion was almost as good as the first.",
        ],
        "details_bad": [
            "The leaves were mostly dust and the infusion turned cloudy.",
            "It tasted bitter even when I brewed it for less than a minute.",
            "The tea bags tore open as soon as they touched hot water.",
            "There was no aroma at all and the colour was pale.",
            "The packet smelled of cardboard rather than tea.",
        ],
    },
    "chocolate": {
        "products": [
            "dark chocolate", "milk chocolate", "chocolate bar",
            "cocoa powder", "chocolate cookies", "hot chocolate mix",
            "chocolate truffles", "cocoa nibs",
        ],
        "aspects": [
            "flavour", "texture", "sweetness", "cocoa", "finish", "melt",
            "snap", "aroma",
        ],
        "good": [
            "rich", "smooth", "creamy", "dark", "sweet", "velvety",
            "fresh", "intense",
        ],
        "bad": [
            "stale", "grainy", "waxy", "bitter", "sugary", "melted",
            "bland", "dry",
        ],
        "details_good": [
            "The chocolate melted smoothly on the tongue and was not overly sweet.",
            "There was a good snap when I broke the bar.",
            "The cocoa flavour is deep and the sweetness is balanced.",
            "It arrived in a cool box, so nothing had melted on the way.",
            "The truffles looked beautiful and tasted even better.",
        ],
        "details_bad": [
            "The bar had melted and reformed into a grainy block.",
            "It tasted waxy and left a strange coating in my mouth.",
            "The chocolate was well past its date and had a stale smell.",
            "The sugar overwhelmed any cocoa flavour.",
            "The bar arrived broken into small pieces.",
        ],
    },
    "snacks": {
        "products": [
            "potato chips", "namkeen", "cookies", "biscuits",
            "roasted nuts", "instant noodles", "popcorn", "samosa",
        ],
        "aspects": [
            "taste", "crunch", "salt", "freshness", "oiliness", "spice",
            "texture", "flavour",
        ],
        "good": [
            "crispy", "fresh", "crunchy", "tasty", "savoury", "light",
            "flavourful", "spicy",
        ],
        "bad": [
            "stale", "soggy", "oily", "bland", "salty", "greasy",
            "tough", "rancid",
        ],
        "details_good": [
            "The chips were crisp and the masala coating was evenly spread.",
            "The pack stayed fresh for a week after opening because of the zip lock.",
            "It is a handy snack for office breaks and the portion is generous.",
            "The namkeen was crunchy and not too oily.",
            "The nuts were roasted evenly and had a pleasant toasted flavour.",
        ],
        "details_bad": [
            "The chips were soggy and tasted of old frying oil.",
            "The pack was mostly broken pieces and crumbs.",
            "It was far too salty and we could not finish one bowl.",
            "The biscuits had absorbed moisture and lost their crunch.",
            "The seal was open and the snacks had gone rancid.",
        ],
    },
    "spices": {
        "products": [
            "turmeric powder", "chilli powder", "garam masala",
            "black pepper", "cumin seeds", "cardamom", "coriander powder",
            "cinnamon sticks",
        ],
        "aspects": [
            "aroma", "colour", "flavour", "purity", "heat", "texture",
            "freshness", "strength",
        ],
        "good": [
            "fragrant", "fresh", "pungent", "aromatic", "pure", "strong",
            "vibrant", "authentic",
        ],
        "bad": [
            "stale", "bland", "dusty", "weak", "adulterated", "bitter",
            "mild", "damp",
        ],
        "details_good": [
            "The powder was finely ground and the colour was bright.",
            "A small pinch was enough to perfume the whole curry.",
            "The aroma filled the kitchen as soon as the packet was opened.",
            "The spices tasted fresh and the flavour was authentic.",
            "The whole spices were clean and free of stems.",
        ],
        "details_bad": [
            "The powder had clumped into hard lumps.",
            "There was hardly any aroma and the curry tasted flat.",
            "The packet contained grit and small stones.",
            "The colour was dull and the spice had lost its punch.",
            "It tasted more of chilli than of the spice on the label.",
        ],
    },
}

DELIVERY_GOOD = [
    "on time", "a day early", "in perfect condition", "quickly",
    "well before the promised date",
]
DELIVERY_BAD = [
    "three days late", "in a damaged box", "with a torn outer package",
    "much later than promised", "in a crushed carton",
]
PACK_GOOD = ["secure", "clean", "properly sealed", "neat", "sturdy"]
PACK_BAD = ["damaged", "torn", "leaking", "crushed", "poorly sealed"]
SIZES = ["family pack", "value pack", "small pack", "large pack", "500 gram pack"]

TEXT_COLUMNS = (
    "text", "reviewtext", "review_text", "review", "reviewbody", "review_body",
    "content", "body", "comments", "comment", "summary", "description",
)


def _make_review(rng: random.Random, category: str, sentiment: str) -> str:
    data = CATEGORIES[category]
    product = rng.choice(data["products"])
    aspect1, aspect2 = rng.sample(data["aspects"], 2)
    good1, good2 = rng.sample(data["good"], 2)
    bad1, bad2 = rng.sample(data["bad"], 2)
    size = rng.choice(SIZES)

    positive_product = [
        f"The {product} had a {good1} {aspect1}.",
        f"I keep the {product} in the kitchen because the {aspect1} is {good1}.",
        f"I added the {product} to my weekly shopping list.",
        f"I served the {product} to guests and they asked where I bought it.",
        f"I noticed a {good1} {aspect2} in the {product}.",
    ]
    positive_aspect = [
        f"The {aspect1} is {good1} and the {aspect2} is {good2}.",
        f"The {aspect1} is {good1}, exactly what I look for in {category}.",
        f"The flavour is well balanced and the {aspect1} is {good1}.",
    ]
    positive_generic = [
        f"I ordered the {product} from this seller and it arrived {rng.choice(DELIVERY_GOOD)}.",
        f"The packaging was {rng.choice(PACK_GOOD)} and the contents were fresh when they arrived.",
        f"At this price the {size} is good value for money.",
        "I would buy this again without hesitation.",
        "This has become a repeat purchase for our kitchen.",
        "The quality has been consistent across orders.",
        "Everyone at home liked it and the pack did not last long.",
    ]
    negative_product = [
        f"The {product} had a {bad1} {aspect1}.",
        f"I could not finish the {product} because the {aspect1} was {bad1}.",
        f"The {product} smelled {bad1} as soon as the pack was opened.",
        f"The {product} tasted nothing like the description on the label.",
        f"I regret buying the {product} because the {aspect2} was {bad2}.",
    ]
    negative_aspect = [
        f"The {aspect1} was {bad1} and the {aspect2} was {bad2}.",
        f"The {aspect1} was {bad1} even though the pack looked new.",
        f"The flavour was poor and the {aspect1} was {bad1}.",
    ]
    negative_generic = [
        f"I ordered the {product} from this seller and it arrived {rng.choice(DELIVERY_BAD)}.",
        f"The packaging was {rng.choice(PACK_BAD)} and the product looked old.",
        f"For this price the {size} is poor value for money.",
        "I would not recommend this to anyone.",
        "I will not order this again.",
        "The batch I received tasted nothing like the previous order.",
        "Most of the pack went into the bin.",
        "The quality has gone down since my last order.",
    ]

    if sentiment == "positive":
        chosen = [rng.choice(positive_product), rng.choice(positive_aspect),
                  rng.choice(data["details_good"])] + rng.sample(positive_generic, rng.randint(1, 2))
    elif sentiment == "negative":
        chosen = [rng.choice(negative_product), rng.choice(negative_aspect),
                  rng.choice(data["details_bad"])] + rng.sample(negative_generic, rng.randint(1, 2))
    else:
        chosen = [rng.choice(positive_product), rng.choice(positive_aspect),
                  rng.choice(negative_product), rng.choice(negative_aspect),
                  rng.choice(positive_generic), rng.choice(negative_generic)]
        rng.shuffle(chosen)
    return " ".join(chosen)


def generate_reviews(n: int = 2000, seed: int = 42) -> list[dict]:
    """Build a deterministic, balanced synthetic review dataset."""
    rng = random.Random(seed)
    categories = list(CATEGORIES)
    reviews = []
    for i in range(n):
        category = categories[i % len(categories)]
        sentiment = rng.choices(
            ["positive", "negative", "mixed"], weights=[0.55, 0.30, 0.15]
        )[0]
        rating = {"positive": rng.choice([4, 5]), "negative": rng.choice([1, 2]),
                  "mixed": 3}[sentiment]
        reviews.append({
            "id": i + 1,
            "category": category,
            "rating": rating,
            "text": _make_review(rng, category, sentiment),
        })
    return reviews


def _read_csv(path: Path, text_column: str | None) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row")
        fields = {name.strip().lower(): name for name in reader.fieldnames}
        column = None
        if text_column:
            column = fields.get(text_column.lower())
            if column is None:
                raise ValueError(
                    f"column '{text_column}' not found; available: {', '.join(reader.fieldnames)}"
                )
        else:
            for candidate in TEXT_COLUMNS:
                if candidate in fields:
                    column = fields[candidate]
                    break
        if column is None:
            raise ValueError(
                "could not find a review-text column; pass --text-column. "
                f"Available: {', '.join(reader.fieldnames)}"
            )
        rows = []
        for i, row in enumerate(reader):
            text = (row.get(column) or "").strip()
            if text:
                rows.append({"id": i + 1, "category": "unknown",
                             "rating": row.get(fields.get("score", ""), ""),
                             "text": text})
    return rows


def _read_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("reviews", "data", "records", "items"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError("JSON must contain a list of reviews")
    rows = []
    for i, item in enumerate(data):
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            lowered = {str(k).lower(): v for k, v in item.items()}
            text = next((str(lowered[k]) for k in TEXT_COLUMNS if lowered.get(k)), "")
        else:
            text = ""
        if text.strip():
            rows.append({"id": i + 1, "category": "unknown", "rating": "",
                         "text": text.strip()})
    return rows


def load_reviews(
    path: str | Path | None = None,
    text_column: str | None = None,
    n: int = 2000,
    seed: int = 42,
) -> tuple[list[dict], str, bool]:
    """Return (reviews, source description, synthetic flag)."""
    if path is None:
        return generate_reviews(n=n, seed=seed), f"synthetic review corpus ({n} reviews)", True
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        rows = _read_csv(path, text_column)
    elif suffix == ".json":
        rows = _read_json(path)
    else:
        rows = [
            {"id": i + 1, "category": "unknown", "rating": "", "text": line.strip()}
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines())
            if line.strip()
        ]
    if not rows:
        raise ValueError(f"no non-empty reviews found in {path}")
    if len(rows) > n:
        rng = random.Random(seed)
        rows = rng.sample(rows, n)
    return rows, f"{path} ({len(rows)} reviews used)", False


def clean_headline(text: str) -> str:
    """One-line version of a review, used in the markdown report."""
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= 220 else text[:217].rstrip() + "..."
