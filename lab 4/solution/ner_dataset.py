"""
Lab 4 - Dataset utilities for RNN-based NER.

Two data sources are supported:

1. A synthetic, reproducible English NER corpus (default, works offline).
   Sentences are produced from ~50 news / assistant style templates whose slots
   are filled from entity pools.  To make the task realistic:
     * 30 % of every entity pool is HELD OUT - those names never occur in the
       training split, so the test set measures generalisation to unseen names;
     * 10 templates are used only for dev/test (unseen contexts);
     * some names are ambiguous (Jordan, Washington, Georgia, Amazon, ...);
     * ~12 % of sentences are lower-cased to imitate ASR / informal text;
     * years in dev/test (2023-2030) never appear in training (2012-2022).

2. Any CoNLL-formatted corpus (e.g. CoNLL-2003): one "token ... tag" per line,
   blank line between sentences.  IOB1 tags are converted to BIO (IOB2).
"""

import random

FIRST_NAMES = """Arun Priya Rahul Meena Vikram Anjali Karthik Divya Suresh Lakshmi Ramesh Kavya
Sanjay Neha Deepa Arjun Pooja Harini Venkatesh Shalini Rohan Ananya Gopal Sneha Aditya Nisha
Kavitha Harish Meera Thomas John Sarah David Emily Michael Fatima Maria James Robert Linda
Daniel Laura Kevin Olivia Ahmed Yuki Chen Elena Carlos Sofia Ibrahim Grace Peter Hannah Ravi
Anand Kiran Varun Swathi Naveen Rekha Imran Ayesha Joseph Mary""".split()

LAST_NAMES = """Kumar Sharma Iyer Singh Menon Reddy Gupta Raman Narayanan Babu Rao Krishnan Das
Patel Verma Subramanian Smith Miller Brown Watson Khan Garcia Lee Wilson Taylor Anderson
Martin Thompson Nair Pillai Joshi Mehta Chatterjee Banerjee Mukherjee Fernandes D'Souza Ali
Tanaka Wang Ivanova Lopez Rossi Schmidt Muller Johnson Williams Jones Davis Clark""".split()

AMBIGUOUS_PER = ["Jordan", "Washington", "Georgia", "Victoria", "Florence", "Paris"]

LOCATIONS = """Chennai Mumbai Delhi Bangalore Hyderabad Kolkata Pune Ahmedabad Jaipur Lucknow
Kochi Coimbatore Madurai Vellore Trichy Mysore Mangalore Goa Chandigarh Bhopal Indore Patna
Nagpur Surat Visakhapatnam Guwahati Shimla Ooty London Paris Berlin Tokyo Singapore Dubai
Sydney Toronto Bangkok Amsterdam Rome Milan Madrid Moscow Beijing Seoul Cairo Nairobi Lagos
Boston Chicago Seattle Houston Denver Doha Muscat Riyadh Istanbul Vienna Prague Dublin
Lisbon Oslo Stockholm Helsinki Zurich Geneva India China Japan Germany France Brazil Canada
Australia Kenya Egypt Nepal Bhutan Sri_Lanka New_York San_Francisco Los_Angeles Hong_Kong
Kuala_Lumpur New_Delhi Tamil_Nadu Kerala Karnataka Maharashtra Gujarat Rajasthan Punjab
Jordan Washington Georgia Victoria Florence Amazon_Basin""".split()

ORGANIZATIONS = """Infosys Wipro Google Microsoft Amazon Apple IBM Intel Samsung Sony Toyota Tesla
Netflix Flipkart Zoho Swiggy Zomato Paytm Ola Uber Reliance Accenture Cognizant Capgemini
Deloitte Oracle Adobe Nvidia Siemens Bosch Airtel ISRO NASA UNESCO UNICEF WHO Interpol
Tata_Motors Tata_Consultancy_Services State_Bank_of_India HDFC_Bank ICICI_Bank Axis_Bank
Air_India Qatar_Airways Emirates Indian_Railways Apollo_Hospital Fortis_Hospital
Anna_University VIT_University IIT_Madras IIT_Bombay Delhi_University Harvard_University
Stanford_University Oxford_University World_Bank Reserve_Bank_of_India United_Nations
Infosys_Foundation Larsen_and_Toubro Hindustan_Unilever Mahindra_Group Chelsea Arsenal
Real_Madrid Mumbai_Indians Chennai_Super_Kings""".split()

# compositional organisation names: "<Word> <Suffix>"
ORG_WORDS = """Sunrise Bluewave Greenfield Silverline Northstar Pinnacle Horizon Evergreen
Redstone Brightpath Crystal Summit Lotus Phoenix Orbit Vertex Nova Apex Zenith Coral""".split()
ORG_SUFFIXES = """Technologies Industries Solutions Systems Bank Hospital University Labs Motors
Pharmaceuticals Foundation Airlines Media Energy Logistics Capital""".split()

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def ordinal(n):
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def make_date(rng, years):
    d, m, y = rng.randint(1, 28), rng.choice(MONTHS), rng.choice(years)
    w = rng.choice(WEEKDAYS)
    return rng.choice([
        f"{d} {m}", f"{m} {d}", f"{d} {m} {y}", f"{m} {y}", f"{ordinal(d)} of {m}",
        w, f"next {w}", f"last {w}", "tomorrow", "yesterday", "today", f"{y}",
        f"{m} {d} {y}", "next week", "last month", f"early {m}",
    ]).split()


# Templates.  Slots: {PER} {LOC} {ORG} {DATE}.  The last 10 are dev/test only.
TEMPLATES = [
    "{PER} joined {ORG} as a senior engineer in {DATE}",
    "{ORG} announced a new office in {LOC} on {DATE}",
    "Book a flight for {PER} from {LOC} to {LOC} on {DATE}",
    "Schedule a meeting with {PER} at {ORG} on {DATE}",
    "{PER} was born in {LOC}",
    "The CEO of {ORG} , {PER} , visited {LOC} last week",
    "Call {PER} and tell him the report is due {DATE}",
    "Remind me to email {PER} from {ORG} on {DATE}",
    "What is the weather like in {LOC} {DATE}",
    "{ORG} shares rose sharply after the results on {DATE}",
    "Dr {PER} will speak at {ORG} in {LOC}",
    "{PER} met {PER} in {LOC} to discuss the deal with {ORG}",
    "The conference will be held in {LOC} from {DATE} onwards",
    "{ORG} signed an agreement with {ORG} in {LOC}",
    "Send a message to {PER} saying the demo is on {DATE}",
    "Mr {PER} moved from {LOC} to {LOC} in {DATE}",
    "Find hotels near {LOC} for {DATE}",
    "{PER} , a professor at {ORG} , won the award on {DATE}",
    "Police in {LOC} arrested two men on {DATE}",
    "{ORG} opened its largest store in {LOC}",
    "Transfer money to {PER} through {ORG} {DATE}",
    "The match between {ORG} and {ORG} was played in {LOC} on {DATE}",
    "Prime Minister {PER} arrived in {LOC} on {DATE}",
    "Navigate to {ORG} in {LOC}",
    "{PER} said that {ORG} will hire 500 people in {LOC}",
    "Heavy rain was reported in {LOC} on {DATE}",
    "Set a reminder to call {PER} on {DATE}",
    "{ORG} was founded by {PER} in {DATE}",
    "Ms {PER} is the new director of {ORG}",
    "Flights from {LOC} to {LOC} were cancelled {DATE}",
    "{PER} and {PER} will travel to {LOC} next month",
    "Cancel my appointment with {PER} on {DATE}",
    "Researchers at {ORG} published the study on {DATE}",
    "The office of {ORG} in {LOC} will remain closed {DATE}",
    "Tell {PER} that I will reach {LOC} {DATE}",
    "{PER} scored twice for {ORG} on {DATE}",
    "Students from {LOC} visited {ORG} on {DATE}",
    "{ORG} plans to invest in {LOC} by {DATE}",
    "How far is {LOC} from {LOC}",
    "{PER} works for {ORG} in {LOC}",
    # ---- held-out contexts (dev / test only) ----
    "According to {PER} , the deal between {ORG} and {ORG} closed on {DATE}",
    "Order a cab for {PER} to {ORG} {DATE}",
    "Officials in {LOC} confirmed that {PER} resigned on {DATE}",
    "{PER} flew to {LOC} to sign a contract with {ORG}",
    "Is it going to snow in {LOC} on {DATE}",
    "A delegation from {ORG} met {PER} in {LOC}",
    "The festival in {LOC} begins on {DATE}",
    "Professor {PER} left {ORG} after ten years",
    "Ask {PER} whether {ORG} replied {DATE}",
    "Protesters gathered outside the {ORG} headquarters in {LOC}",
]
N_TEST_ONLY_TEMPLATES = 10


def _split_pool(pool, rng, held_out=0.3):
    pool = sorted(set(pool))
    rng.shuffle(pool)
    k = int(len(pool) * held_out)
    return pool[k:], pool[:k]          # (train_pool, held_out_pool)


class SyntheticNER:
    def __init__(self, seed=13):
        rng = random.Random(seed)
        self.first_tr, self.first_ho = _split_pool(FIRST_NAMES, rng)
        self.last_tr, self.last_ho = _split_pool(LAST_NAMES, rng)
        self.loc_tr, self.loc_ho = _split_pool(LOCATIONS, rng)
        self.org_tr, self.org_ho = _split_pool(ORGANIZATIONS, rng)
        self.orgw_tr, self.orgw_ho = _split_pool(ORG_WORDS, rng)

    def _person(self, rng, unseen):
        first = rng.choice(self.first_ho if unseen else self.first_tr)
        last = rng.choice(self.last_ho if unseen else self.last_tr)
        r = rng.random()
        if r < 0.05:
            return [rng.choice(AMBIGUOUS_PER)]
        if r < 0.25:
            return [first]
        return [first, last]

    def _entity(self, kind, rng, unseen, years):
        if kind == "PER":
            return self._person(rng, unseen)
        if kind == "LOC":
            return rng.choice(self.loc_ho if unseen else self.loc_tr).split("_")
        if kind == "ORG":
            if rng.random() < 0.35:
                word = rng.choice(self.orgw_ho if unseen else self.orgw_tr)
                return [word, rng.choice(ORG_SUFFIXES)]
            return rng.choice(self.org_ho if unseen else self.org_tr).split("_")
        if kind == "DATE":
            return make_date(rng, years)
        raise ValueError(kind)

    def generate(self, n, split, seed):
        rng = random.Random(seed)
        if split == "train":
            templates = TEMPLATES[:-N_TEST_ONLY_TEMPLATES]
            years, p_unseen = list(range(2012, 2023)), 0.0
        else:
            templates = TEMPLATES
            years, p_unseen = list(range(2023, 2031)), 0.5
        data = []
        for _ in range(n):
            tpl = rng.choice(templates)
            tokens, tags = [], []
            for piece in tpl.split():
                if piece.startswith("{") and piece.endswith("}"):
                    kind = piece[1:-1]
                    ent = self._entity(kind, rng, rng.random() < p_unseen, years)
                    tokens += ent
                    tags += [f"B-{kind}"] + [f"I-{kind}"] * (len(ent) - 1)
                else:
                    tokens.append(piece)
                    tags.append("O")
            if rng.random() < 0.12:                  # ASR / informal style
                tokens = [t.lower() for t in tokens]
            data.append((tokens, tags))
        return data


def load_synthetic(n_train=3000, n_dev=500, n_test=1000, seed=13):
    gen = SyntheticNER(seed)
    return (gen.generate(n_train, "train", seed + 1),
            gen.generate(n_dev, "dev", seed + 2),
            gen.generate(n_test, "test", seed + 3))


# ---------------------------------------------------------------------------
# CoNLL reader
# ---------------------------------------------------------------------------
def iob1_to_bio(tags):
    out = []
    for i, t in enumerate(tags):
        if t.startswith("I-"):
            prev = out[-1] if out else "O"
            if prev == "O" or prev[2:] != t[2:]:
                t = "B-" + t[2:]
        out.append(t)
    return out


def read_conll(path):
    sents, toks, tags = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("-DOCSTART-"):
                if toks:
                    sents.append((toks, iob1_to_bio(tags)))
                    toks, tags = [], []
                continue
            parts = line.split()
            toks.append(parts[0])
            tags.append(parts[-1])
    if toks:
        sents.append((toks, iob1_to_bio(tags)))
    return sents
