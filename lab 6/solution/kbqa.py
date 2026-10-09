"""
Lab 6 (i) - Knowledge-Based Question Answering over a triple store.

Pipeline (as in the lecture: question analysis -> query -> answer):
  1. Normalise the question.
  2. Entity linking   : longest exact match against entity names + aliases, then fuzzy
                        matching (difflib) to tolerate typos ("Gogle" -> Google).
  3. Relation detection: ordered regular-expression patterns ("who founded" -> founded_by,
                        "when ... founded" -> founded_in, "birthplace" -> born_in ...).
  4. Expected answer type: "which state", "which country", "which city", "which company" ...
  5. Structured query : forward  (entity, relation, ?x)
                        inverse  (?x, relation, entity)   if the forward query is empty
  6. Answer-type check : if the answer has the wrong type (city when a state was asked),
                        follow located_in edges (multi-hop reasoning, up to 3 hops).
  7. Aggregation      : "how many ..." -> COUNT(?x)
  If any step fails the system ABSTAINS ("I don't know") instead of guessing.
"""

import difflib
import re
from collections import defaultdict
from dataclasses import dataclass, field

from kb_data import ALIASES, ENTITY_TYPES, TRIPLES


def normalise(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


RELATION_PATTERNS = [            # order matters: more specific patterns first
    ("launch_date", [r"\bwhen\b.*\blaunch", r"\blaunch date\b"]),
    ("launched_by", [r"\blaunch"]),
    ("founded_in", [r"\b(when|which year|what year)\b.*\b(found|establish|start)", r"\byear of (founding|establishment)\b"]),
    ("founded_by", [r"\bfound(ed|er|ers)?\b", r"\bestablish", r"\bstarted\b"]),
    ("born_on", [r"\bwhen\b.*\bborn\b", r"\bbirth ?(date|day)\b", r"\bdate of birth\b"]),
    ("born_in", [r"\bborn\b", r"\bbirth ?place\b", r"\bplace of birth\b", r"\bhometown\b"]),
    ("ceo", [r"\bceo\b", r"\bchief executive\b", r"\bheads?\b", r"\bled by\b", r"\bleads?\b", r"\brun by\b"]),
    ("headquartered_in", [r"\bheadquarter"]),
    ("capital", [r"\bcapital\b"]),
    ("currency", [r"\bcurrency\b", r"\bmoney\b"]),
    ("educated_at", [r"\bstud(y|ied)\b", r"\battend", r"\beducat", r"\balma mater\b", r"\bgraduat"]),
    ("known_for", [r"\b(known|famous|remembered) for\b", r"\bdiscover"]),
    ("award", [r"\bawards?\b", r"\bprizes?\b", r"\bwon\b", r"\bwin\b"]),
    ("created_by", [r"\bcreat", r"\binvent", r"\bdevelop", r"\bdesign"]),
    ("released_in", [r"\breleas"]),
    ("field", [r"\bfield\b", r"\barea of (research|work)\b"]),
    ("position", [r"\bpresident\b", r"\bposition\b", r"\boffice\b"]),
    ("occupation", [r"\bprofession\b", r"\boccupation\b", r"\bjob\b", r"^who (was|is)\b"]),
    ("located_in", [r"\bwhere\b", r"\blocated\b", r"\bsituated\b", r"\bin which (city|state|country)\b"]),
]

TYPE_WORDS = {"city": "city", "town": "city", "state": "state", "country": "country", "nation": "country",
              "company": "organization", "organisation": "organization", "organization": "organization",
              "institute": "organization", "university": "organization", "agency": "organization",
              "person": "person"}


@dataclass
class KBAnswer:
    answer: str = None
    entity: str = None
    relation: str = None
    query: str = ""
    path: list = field(default_factory=list)
    reason: str = ""


class KnowledgeBase:
    def __init__(self, triples=TRIPLES, types=ENTITY_TYPES, aliases=ALIASES):
        self.triples = triples
        self.fwd = defaultdict(list)                     # (s, r) -> [o]
        self.inv = defaultdict(list)                     # (o, r) -> [s]
        for s, r, o in triples:
            self.fwd[(s, r)].append(o)
            self.inv[(o, r)].append(s)
        self.type_of = {e: t for t, es in types.items() for e in es}
        self.entities = sorted({s for s, _, _ in triples} | {o for _, _, o in triples})
        self.relations = sorted({r for _, r, _ in triples})
        self.alias_to_entity = {}
        for e in self.entities:
            self.alias_to_entity[normalise(e)] = e
        for e, al in aliases.items():
            for a in al:
                self.alias_to_entity[normalise(a)] = e

    # --------------------------------------------------------------- step 2
    def link_entities(self, question):
        toks = normalise(question).split()
        found, used = [], set()
        for n in range(min(6, len(toks)), 0, -1):           # longest exact match first
            for i in range(len(toks) - n + 1):
                span = set(range(i, i + n))
                if span & used:
                    continue
                gram = " ".join(toks[i:i + n])
                if gram in self.alias_to_entity:
                    found.append((self.alias_to_entity[gram], gram, "exact"))
                    used |= span
        keys = [k for k in self.alias_to_entity if len(k) >= 4]
        for n in (3, 2, 1):                                  # fuzzy match for typos
            for i in range(len(toks) - n + 1):
                span = set(range(i, i + n))
                if span & used:
                    continue
                gram = " ".join(toks[i:i + n])
                if len(gram) < 4:
                    continue
                cand = [k for k in keys if len(k.split()) == n]
                m = difflib.get_close_matches(gram, cand, n=1, cutoff=0.85)
                if m:
                    found.append((self.alias_to_entity[m[0]], gram, "fuzzy"))
                    used |= span
        return found

    # --------------------------------------------------------------- step 3
    @staticmethod
    def detect_relations(question):
        q = question.lower()
        return [rel for rel, pats in RELATION_PATTERNS if any(re.search(p, q) for p in pats)]

    @staticmethod
    def expected_type(question):
        m = re.search(r"\b(?:which|what)\s+(\w+)", question.lower())
        return TYPE_WORDS.get(m.group(1)) if m else None

    # ------------------------------------------------------------- steps 5-7
    def climb(self, entity, wanted):
        """Follow located_in edges until an entity of the wanted type is reached."""
        path, cur = [], entity
        for _ in range(3):
            if wanted is None or self.type_of.get(cur) == wanted:
                return cur, path
            nxt = self.fwd.get((cur, "located_in"))
            if not nxt:
                break
            path.append(f"({cur}, located_in, {nxt[0]})")
            cur = nxt[0]
        return (cur, path) if wanted is None or self.type_of.get(cur) == wanted else (None, path)

    def answer(self, question):
        res = KBAnswer()
        entities = self.link_entities(question)
        relations = self.detect_relations(question)
        wanted = self.expected_type(question)
        if not entities:
            res.reason = "no known entity found in the question"
            return res
        if not relations:
            res.reason = f"entity {entities[0][0]!r} found, but no known relation matched the question"
            return res
        for ent, _, how in entities:
            for rel in relations:
                for direction in ("forward", "inverse"):
                    vals = (self.fwd if direction == "forward" else self.inv).get((ent, rel))
                    if not vals:
                        continue
                    res.entity, res.relation = ent, rel
                    res.query = (f"SELECT ?x WHERE {{ <{ent}> <{rel}> ?x }}" if direction == "forward"
                                 else f"SELECT ?x WHERE {{ ?x <{rel}> <{ent}> }}")
                    res.path = [f"({ent}, {rel}, {v})" if direction == "forward" else f"({v}, {rel}, {ent})"
                                for v in vals]
                    if how == "fuzzy":
                        res.path.insert(0, f"[fuzzy entity link -> {ent}]")
                    if re.match(r"\s*how many\b", question.lower()):
                        res.query = res.query.replace("SELECT ?x", "SELECT (COUNT(?x) AS ?n)")
                        res.answer = str(len(vals))
                        return res
                    answers = []
                    for v in vals:
                        a, hops = self.climb(v, wanted if rel not in ("founded_in", "born_on") else None)
                        if a:
                            answers.append(a)
                            res.path += hops
                    if answers:
                        res.answer = " and ".join(dict.fromkeys(answers))
                        return res
        res.reason = (f"no triple matches entity {[e for e, _, _ in entities]} with relation(s) {relations}"
                      + (f" and answer type '{wanted}'" if wanted else ""))
        return res
