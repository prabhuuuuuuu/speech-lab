"""
Lab 3 - Annotated dataset of voice-assistant speech-to-text transcripts.

Each transcript is written exactly as an ASR engine (e.g. Google STT) would
return it: no punctuation, numbers as digits, true-cased words.

Entities are annotated inline using a light markup:
        [entity text|TYPE]
TYPE is one of PER, LOC, ORG, DATE.  Everything outside brackets is "O".

`parse_markup()` converts a marked-up transcript into (tokens, BIO tags),
which is Task 1 of the lab ("Convert the sentences into BIO tags").
"""

import re

ENTITY_TYPES = ["PER", "LOC", "ORG", "DATE"]
TAGS = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")]

# ---------------------------------------------------------------------------
# Small annotated TRAINING set (60 transcripts)
# ---------------------------------------------------------------------------
TRAIN_TRANSCRIPTS = [
    "Book a flight for Dr [Arun Kumar|PER] from [Chennai|LOC] to [Singapore|LOC] on [15 September|DATE]",
    "Schedule a meeting with [Priya Sharma|PER] at [Infosys|ORG] on [Monday|DATE]",
    "Call [Rahul|PER] and tell him I will reach [Mumbai|LOC] [tomorrow|DATE]",
    "What is the weather like in [New York|LOC] [today|DATE]",
    "Remind me to email [Meena Iyer|PER] from [Google|ORG] on [next Friday|DATE]",
    "Book a train ticket from [Delhi|LOC] to [Agra|LOC] for [Sunday|DATE]",
    "Send a message to [John Smith|PER] saying the [Microsoft|ORG] demo is on [3 March|DATE]",
    "Find hotels in [London|LOC] for [25 December|DATE]",
    "Set a reminder to call Mr [Vikram Singh|PER] on [Wednesday|DATE]",
    "Book a cab to [Apollo Hospital|ORG] in [Chennai|LOC]",
    "Cancel my [Air India|ORG] flight to [Dubai|LOC] on [21 June|DATE]",
    "Schedule an interview with [Anjali Menon|PER] from [Tata Consultancy Services|ORG] on [10 January|DATE]",
    "Navigate to [VIT University|ORG] in [Vellore|LOC]",
    "Order food from [Swiggy|ORG] to my office in [Bangalore|LOC]",
    "Play the latest podcast by [David Miller|PER]",
    "Transfer five thousand rupees to [Kavya|PER] through [State Bank of India|ORG] [today|DATE]",
    "How far is [Hyderabad|LOC] from [Pune|LOC]",
    "Book a flight with [IndiGo|ORG] from [Kolkata|LOC] to [Delhi|LOC] on [5 August|DATE]",
    "Remind me about the meeting with Professor [Ravi Shankar|PER] on [Thursday|DATE]",
    "Call [Amazon|ORG] customer care [tomorrow|DATE] morning",
    "Send the report to [Lakshmi Narayanan|PER] at [Wipro|ORG] by [Friday|DATE]",
    "Show me restaurants near [Marina Beach|LOC] in [Chennai|LOC]",
    "Book a doctor appointment with Dr [Emily Watson|PER] at [Manipal Hospital|ORG] on [12 May|DATE]",
    "Is it going to rain in [Kochi|LOC] [this weekend|DATE]",
    "Text [Suresh Babu|PER] that I am stuck in traffic near [Guindy|LOC]",
    "Book train tickets on [Indian Railways|ORG] from [Madurai|LOC] to [Coimbatore|LOC] for [next Monday|DATE]",
    "Schedule a call with [Neha Gupta|PER] from [HDFC Bank|ORG] at four pm [today|DATE]",
    "Find flights from [Berlin|LOC] to [Paris|LOC] on [1 April|DATE]",
    "Remind me to wish [Karthik|PER] on his birthday [9 October|DATE]",
    "Get directions to [Anna University|ORG] from [Tambaram|LOC]",
    "Book an [Uber|ORG] to the airport in [Bangalore|LOC]",
    "Set up a video call with [Michael Brown|PER] and [Fatima Khan|PER] on [Tuesday|DATE]",
    "Check the status of my [Emirates|ORG] flight from [Dubai|LOC] to [Sydney|LOC]",
    "Email the invoice to [Zoho|ORG] before [30 June|DATE]",
    "Book a hotel in [Goa|LOC] from [24 December|DATE] to [28 December|DATE]",
    "Call [Deepa Raman|PER] at [Reliance|ORG]",
    "What time does [Apollo Pharmacy|ORG] close in [Adyar|LOC]",
    "Remind me to pay the [Airtel|ORG] bill on [Saturday|DATE]",
    "Book a flight for [Arjun Reddy|PER] from [Hyderabad|LOC] to [San Francisco|LOC] on [18 November|DATE]",
    "Tell [Sarah|PER] that the meeting at [Google|ORG] is moved to [Thursday|DATE]",
    "Find a pharmacy near [Velachery|LOC]",
    "Schedule a team lunch with [Ramesh|PER] and [Divya|PER] [next Wednesday|DATE]",
    "How is the traffic from [Whitefield|LOC] to [Electronic City|LOC] right now",
    "Book movie tickets at [PVR Cinemas|ORG] in [Coimbatore|LOC] for [Sunday|DATE]",
    "Message [Kiran Rao|PER] that the [Infosys|ORG] interview is on [14 February|DATE]",
    "Set an alarm for six am [tomorrow|DATE]",
    "Book a flight to [Kuala Lumpur|LOC] with [Malaysia Airlines|ORG] on [2 October|DATE]",
    "Call Dr [Sanjay Gupta|PER] at [AIIMS|ORG] in [Delhi|LOC]",
    "Remind me to submit the assignment to Professor [Anand|PER] [day after tomorrow|DATE]",
    "Show me the weather forecast for [Ooty|LOC] on [Saturday|DATE]",
    "Schedule a meeting with the [Capgemini|ORG] team on [8 July|DATE]",
    "Find bus tickets from [Bangalore|LOC] to [Mangalore|LOC] for [this Friday|DATE]",
    "Send flowers to [Pooja|PER] in [Chandigarh|LOC] on [16 March|DATE]",
    "Book a consultation with Dr [Rajesh Khanna|PER] at [Apollo Hospital|ORG] [next Tuesday|DATE]",
    "Pay my electricity bill to [Tata Power|ORG] [today|DATE]",
    "Call [Venkatesh|PER] from [Larsen and Toubro|ORG] after lunch",
    "What is the distance between [Rome|LOC] and [Milan|LOC]",
    "Reserve a room at [Marriott|ORG] in [Kolkata|LOC] for [31 December|DATE]",
    "Join the [Zoom|ORG] call with [Harini|PER] at five",
    "Book a flight for Mrs [Shalini Iyer|PER] from [Trivandrum|LOC] to [Muscat|LOC] on [27 January|DATE]",
]

# ---------------------------------------------------------------------------
# UNSEEN test transcripts (most entities never appear in training)
# ---------------------------------------------------------------------------
TEST_TRANSCRIPTS = [
    "Book a flight for Dr [Meera Krishnan|PER] from [Ahmedabad|LOC] to [Toronto|LOC] on [12 August|DATE]",
    "Schedule a meeting with [Rohan Das|PER] from [Accenture|ORG] on [next Tuesday|DATE]",
    "Call [Ananya|PER] and tell her I will reach [Jaipur|LOC] [tomorrow|DATE]",
    "What is the weather like in [Bangkok|LOC] [today|DATE]",
    "Send a message to [Thomas Lee|PER] saying the [Flipkart|ORG] review is on [30 November|DATE]",
    "Book a cab to [Fortis Hospital|ORG] in [Mohali|LOC]",
    "Cancel my [Vistara|ORG] flight to [Lucknow|LOC] on [Saturday|DATE]",
    "Remind me to email Professor [Gopal Rao|PER] at [IIT Madras|ORG] on [Friday|DATE]",
    "Find hotels in [Amsterdam|LOC] for [19 April|DATE]",
    "Transfer money to [Sneha Patel|PER] through [Axis Bank|ORG] [today|DATE]",
    "Book train tickets from [Trichy|LOC] to [Chennai|LOC] for [this Sunday|DATE]",
    "Set a reminder to call Mr [Aditya Verma|PER] at [Cognizant|ORG] on [Monday|DATE]",
    "How far is [Mysore|LOC] from [Bangalore|LOC]",
    "Schedule an interview with [Maria Garcia|PER] from [Apple|ORG] on [6 September|DATE]",
    "Navigate to [SRM University|ORG] in [Kattankulathur|LOC]",
    "Text [Harish|PER] that I am waiting near [Central Station|LOC]",
    "Book a flight with [Qatar Airways|ORG] from [Doha|LOC] to [Chennai|LOC] on [4 May|DATE]",
    "Remind me to meet [Nisha|PER] at [Starbucks|ORG] in [Koramangala|LOC] [next Monday|DATE]",
    "Order groceries from [BigBasket|ORG] for [Sunday|DATE]",
    "Book a doctor appointment with Dr [Kavitha Subramanian|PER] on [22 October|DATE]",
]


_MARKUP = re.compile(r"\[([^\]|]+)\|([A-Z]+)\]")


def tokenize(text):
    """Whitespace tokenizer that also splits off punctuation (ASR output rarely has any)."""
    return re.findall(r"[A-Za-z0-9']+|[^\sA-Za-z0-9]", text)


def parse_markup(marked):
    """Task 1: convert an inline-annotated transcript into tokens + BIO tags."""
    tokens, tags = [], []
    pos = 0
    for m in _MARKUP.finditer(marked):
        for tok in tokenize(marked[pos:m.start()]):
            tokens.append(tok)
            tags.append("O")
        ent_tokens = tokenize(m.group(1))
        ent_type = m.group(2)
        assert ent_type in ENTITY_TYPES, f"Unknown entity type {ent_type} in: {marked}"
        for k, tok in enumerate(ent_tokens):
            tokens.append(tok)
            tags.append(("B-" if k == 0 else "I-") + ent_type)
        pos = m.end()
    for tok in tokenize(marked[pos:]):
        tokens.append(tok)
        tags.append("O")
    return tokens, tags


def load_dataset(transcripts):
    return [parse_markup(t) for t in transcripts]


def strip_markup(marked):
    """Return the raw transcript (what the ASR would actually output)."""
    return _MARKUP.sub(lambda m: m.group(1), marked)
