"""
Lab 5 (a) - Pre-processing of the student-feedback document.

  * removal of unnecessary text : report headers / metadata lines ("Report ID: ..."),
    separator lines, page numbers, bracketed editorial notes, URLs, e-mail
    addresses, boiler-plate sentences ("Thank you for participating ..."),
    repeated punctuation ("!!!"), extra white-space
  * sentence segmentation      : rule-based splitter that protects abbreviations
  * tokenisation               : lower-cased word tokens (+ stop-word filtering)

The section headings ("Teaching Quality", ...) are not sentences; they are kept
as paragraph headings and used as a feature by the feature-based summariser.
"""

import re
from dataclasses import dataclass, field

STOPWORDS = set("""
a about above after again against all also am an and any are aren't as at be because been before being
below between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down
during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here
here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself
let's me more most mustn't my myself no nor not of off on once only or other ought our ours ourselves out
over own same shan't she she'd she'll she's should shouldn't so some such than that that's the their
theirs them themselves then there there's these they they'd they'll they're they've this those through
to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's
where where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're
you've your yours yourself yourselves however also hence thus till like
""".split())

ABBREVIATIONS = ["Dr", "Mr", "Mrs", "Ms", "Prof", "e.g", "i.e", "etc", "No", "vs", "St", "Jr", "Sr", "Dept"]

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
BOILERPLATE_RE = re.compile(r"(thank you for|for (any )?queries|page \d+ of \d+|all rights reserved)", re.I)
METADATA_RE = re.compile(r"^\s*[A-Z][A-Za-z ]{1,30}:\s+\S")
SEPARATOR_RE = re.compile(r"^\s*[=\-_*~#]{3,}\s*$")


@dataclass
class Document:
    title: str = ""
    headings: list = field(default_factory=list)      # heading of every paragraph
    paragraphs: list = field(default_factory=list)    # list[list[str]] sentences per paragraph
    removed: list = field(default_factory=list)       # (reason, text) log

    @property
    def sentences(self):
        return [s for p in self.paragraphs for s in p]

    @property
    def sentence_meta(self):
        """(paragraph index, position inside paragraph, paragraph length) for each sentence."""
        return [(pi, si, len(p)) for pi, p in enumerate(self.paragraphs) for si in range(len(p))]

    def text(self, paragraph_sep="\n\n"):
        return paragraph_sep.join(" ".join(p) for p in self.paragraphs)


def tokenize(text):
    """Lower-cased word tokens; keeps internal hyphens/apostrophes (wi-fi, students')."""
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.lower())


def content_words(text):
    return [w for w in tokenize(text) if w not in STOPWORDS and len(w) > 1]


def split_sentences(text):
    protected = text
    for ab in ABBREVIATIONS:
        protected = re.sub(rf"\b{re.escape(ab)}\.", ab.replace(".", "<prd>") + "<prd>", protected)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", protected)
    return [p.replace("<prd>", ".").strip() for p in parts if p.strip()]


def _is_heading(line):
    words = line.split()
    return 0 < len(words) <= 5 and not re.search(r"[.!?:;,]$", line) and line[0].isupper()


def preprocess(raw, min_words=4):
    doc = Document()
    raw = re.sub(r"<[^>]+>", " ", raw.replace("\r\n", "\n"))          # HTML tags
    current, heading = [], ""

    def flush():
        nonlocal current, heading
        if current:
            _add_paragraph(doc, " ".join(current), heading, min_words)
        current, heading = [], ""

    for line in raw.split("\n"):
        s = line.strip()
        if not s:
            flush()
            continue
        if SEPARATOR_RE.match(s):
            doc.removed.append(("separator line", s))
            flush()
            continue
        if METADATA_RE.match(s) and not re.search(r"[.!?]$", s):
            doc.removed.append(("metadata line", s))
            continue
        if re.fullmatch(r"page \d+ of \d+", s, re.I):
            doc.removed.append(("page number", s))
            continue
        letters = [c for c in s if c.isalpha()]
        if letters and sum(c.isupper() for c in letters) / len(letters) > 0.8 and len(s.split()) > 2:
            doc.title = doc.title or s
            doc.removed.append(("document title", s))
            continue
        if _is_heading(s) and not current:
            heading = s
            continue
        current.append(s)
    flush()
    return doc


def _add_paragraph(doc, text, heading, min_words):
    for note in re.findall(r"\[[^\]]*\]|\([^)]*note[^)]*\)", text, flags=re.I):
        doc.removed.append(("bracketed note", note))
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"([!?.])\1+", r"\1", text)                          # "!!!" -> "!"
    text = re.sub(r"\s+", " ", text).strip()
    sentences = []
    for sent in split_sentences(text):
        if URL_RE.search(sent) or EMAIL_RE.search(sent):
            doc.removed.append(("sentence with URL / e-mail", sent))
            continue
        if BOILERPLATE_RE.search(sent):
            doc.removed.append(("boiler-plate sentence", sent))
            continue
        if len(tokenize(sent)) < min_words:
            doc.removed.append(("too short", sent))
            continue
        sent = sent.replace("!", ".") if sent.endswith("!") else sent
        sentences.append(sent)
    if sentences:
        doc.paragraphs.append(sentences)
        doc.headings.append(heading)
