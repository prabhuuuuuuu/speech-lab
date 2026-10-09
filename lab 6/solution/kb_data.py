"""
Lab 6 - Knowledge base, text passages and evaluation questions.

KNOWLEDGE BASE  : structured facts stored as (subject, relation, object) triples,
                  plus an entity-type table and an alias table (for entity linking).
PASSAGES        : unstructured Wikipedia-style text covering (mostly) the same facts.
                  Used by the deep-learning extractive QA system.
                  Deliberate differences, to make the comparison interesting:
                    * text-only facts : Chandrayaan-3 landing date / region, Kalam's nickname
                    * KB-only fact    : the year Python was released
TEST_QUESTIONS  : (question, gold answers, category). Empty gold list = unanswerable.
"""

TRIPLES = [
    # ---- countries -------------------------------------------------------------
    ("India", "capital", "New Delhi"), ("India", "currency", "Indian Rupee"),
    ("France", "capital", "Paris"), ("France", "currency", "Euro"),
    ("Japan", "capital", "Tokyo"), ("Japan", "currency", "Japanese Yen"),
    ("Poland", "capital", "Warsaw"), ("Germany", "capital", "Berlin"),
    # ---- states ----------------------------------------------------------------
    ("Tamil Nadu", "capital", "Chennai"), ("Tamil Nadu", "located_in", "India"),
    ("Karnataka", "capital", "Bengaluru"), ("Karnataka", "located_in", "India"),
    ("West Bengal", "capital", "Kolkata"), ("West Bengal", "located_in", "India"),
    ("California", "located_in", "United States"),
    # ---- cities ----------------------------------------------------------------
    ("New Delhi", "located_in", "India"), ("Chennai", "located_in", "Tamil Nadu"),
    ("Vellore", "located_in", "Tamil Nadu"), ("Madurai", "located_in", "Tamil Nadu"),
    ("Rameswaram", "located_in", "Tamil Nadu"), ("Tiruchirappalli", "located_in", "Tamil Nadu"),
    ("Bengaluru", "located_in", "Karnataka"), ("Kolkata", "located_in", "West Bengal"),
    ("Kharagpur", "located_in", "West Bengal"), ("Paris", "located_in", "France"),
    ("Tokyo", "located_in", "Japan"), ("Warsaw", "located_in", "Poland"),
    ("Berlin", "located_in", "Germany"), ("Ulm", "located_in", "Germany"),
    ("Mountain View", "located_in", "California"),
    # ---- organisations -----------------------------------------------------------
    ("VIT University", "located_in", "Vellore"), ("VIT University", "founded_in", "1984"),
    ("VIT University", "founded_by", "G. Viswanathan"),
    ("IIT Madras", "located_in", "Chennai"), ("IIT Madras", "founded_in", "1959"),
    ("IIT Kharagpur", "located_in", "Kharagpur"), ("IIT Kharagpur", "founded_in", "1951"),
    ("Infosys", "founded_by", "N. R. Narayana Murthy"), ("Infosys", "founded_in", "1981"),
    ("Infosys", "headquartered_in", "Bengaluru"), ("Infosys", "ceo", "Salil Parekh"),
    ("Google", "founded_by", "Larry Page"), ("Google", "founded_by", "Sergey Brin"),
    ("Google", "founded_in", "1998"), ("Google", "headquartered_in", "Mountain View"),
    ("Google", "ceo", "Sundar Pichai"),
    ("ISRO", "headquartered_in", "Bengaluru"), ("ISRO", "founded_in", "1969"),
    ("ISRO", "founded_by", "Vikram Sarabhai"),
    # ---- people ------------------------------------------------------------------
    ("Sundar Pichai", "born_in", "Madurai"), ("Sundar Pichai", "educated_at", "IIT Kharagpur"),
    ("A. P. J. Abdul Kalam", "born_in", "Rameswaram"),
    ("A. P. J. Abdul Kalam", "born_on", "15 October 1931"),
    ("A. P. J. Abdul Kalam", "occupation", "aerospace scientist"),
    ("A. P. J. Abdul Kalam", "position", "President of India"),
    ("Marie Curie", "born_in", "Warsaw"), ("Marie Curie", "field", "physics and chemistry"),
    ("Marie Curie", "award", "Nobel Prize in Physics"), ("Marie Curie", "award", "Nobel Prize in Chemistry"),
    ("C. V. Raman", "born_in", "Tiruchirappalli"), ("C. V. Raman", "award", "Nobel Prize in Physics"),
    ("C. V. Raman", "known_for", "Raman effect"),
    ("Albert Einstein", "born_in", "Ulm"), ("Albert Einstein", "known_for", "theory of relativity"),
    ("Albert Einstein", "award", "Nobel Prize in Physics"),
    ("Vikram Sarabhai", "occupation", "physicist"),
    # ---- other -------------------------------------------------------------------
    ("Chandrayaan-3", "launched_by", "ISRO"), ("Chandrayaan-3", "launch_date", "14 July 2023"),
    ("Python", "created_by", "Guido van Rossum"), ("Python", "released_in", "1991"),
]

ENTITY_TYPES = {
    "country": ["India", "France", "Japan", "Poland", "Germany", "United States"],
    "state": ["Tamil Nadu", "Karnataka", "West Bengal", "California"],
    "city": ["New Delhi", "Paris", "Tokyo", "Warsaw", "Berlin", "Chennai", "Bengaluru", "Kolkata",
             "Vellore", "Madurai", "Rameswaram", "Tiruchirappalli", "Kharagpur", "Ulm", "Mountain View"],
    "organization": ["VIT University", "IIT Madras", "IIT Kharagpur", "Infosys", "Google", "ISRO"],
    "person": ["Sundar Pichai", "A. P. J. Abdul Kalam", "Marie Curie", "C. V. Raman", "Albert Einstein",
               "Vikram Sarabhai", "G. Viswanathan", "N. R. Narayana Murthy", "Salil Parekh",
               "Larry Page", "Sergey Brin", "Guido van Rossum"],
    "mission": ["Chandrayaan-3"],
    "language": ["Python"],
}

ALIASES = {
    "Bengaluru": ["bangalore"],
    "VIT University": ["vit", "vellore institute of technology", "vit vellore"],
    "IIT Madras": ["indian institute of technology madras", "iitm"],
    "IIT Kharagpur": ["indian institute of technology kharagpur"],
    "ISRO": ["indian space research organisation", "indian space research organization"],
    "A. P. J. Abdul Kalam": ["abdul kalam", "apj abdul kalam", "dr kalam"],
    "C. V. Raman": ["cv raman", "chandrasekhara venkata raman"],
    "N. R. Narayana Murthy": ["narayana murthy"],
    "Chandrayaan-3": ["chandrayaan 3", "chandrayaan3", "chandrayaan iii"],
    "United States": ["usa", "united states of america", "america"],
    "Tiruchirappalli": ["trichy"],
    "Albert Einstein": ["einstein"],
    "Marie Curie": ["madame curie"],
    "Sundar Pichai": ["pichai"],
}

PASSAGES = [
    "India is a country in South Asia. Its capital is New Delhi, and its currency is the Indian Rupee. "
    "Tamil Nadu is a state in southern India whose capital is Chennai. Karnataka is another southern "
    "state; its capital is Bengaluru, also known as Bangalore. West Bengal, in eastern India, has "
    "Kolkata as its capital.",

    "France is a country in Western Europe with Paris as its capital, and it uses the Euro. Japan's "
    "capital is Tokyo and its currency is the Japanese Yen. Warsaw is the capital of Poland, and "
    "Berlin is the capital of Germany.",

    "Vellore Institute of Technology (VIT University) is a private university located in Vellore, "
    "Tamil Nadu. It was founded in 1984 by G. Viswanathan. IIT Madras, one of India's premier "
    "engineering institutes, is located in Chennai and was established in 1959. IIT Kharagpur, the "
    "oldest IIT, was established in 1951 in Kharagpur, West Bengal.",

    "Infosys is an Indian multinational information technology company headquartered in Bengaluru. "
    "It was founded in 1981 by N. R. Narayana Murthy and six other engineers. Salil Parekh is the "
    "chief executive officer (CEO) of Infosys.",

    "Google was founded in 1998 by Larry Page and Sergey Brin while they were PhD students at "
    "Stanford University. The company is headquartered in Mountain View, California. Sundar Pichai "
    "has been the CEO of Google since 2015.",

    "Sundar Pichai was born in Madurai, Tamil Nadu, in 1972. He studied metallurgical engineering at "
    "IIT Kharagpur before moving to the United States for higher studies.",

    "A. P. J. Abdul Kalam was born on 15 October 1931 in Rameswaram, Tamil Nadu. He was an aerospace "
    "scientist who later served as the 11th President of India, and he is popularly known as the "
    "Missile Man of India.",

    "The Indian Space Research Organisation (ISRO) is India's national space agency, headquartered in "
    "Bengaluru. It was founded in 1969, and Vikram Sarabhai is regarded as its founder. ISRO launched "
    "the Chandrayaan-3 mission on 14 July 2023. The Chandrayaan-3 lander touched down near the south "
    "pole of the Moon on 23 August 2023, making India the first country to land in that region.",

    "Marie Curie was a physicist and chemist born in Warsaw, Poland. She won two Nobel Prizes: the "
    "Nobel Prize in Physics in 1903 and the Nobel Prize in Chemistry in 1911. C. V. Raman was born in "
    "Tiruchirappalli and received the Nobel Prize in Physics in 1930 for the discovery of the Raman "
    "effect. Albert Einstein, born in Ulm, Germany, is best known for the theory of relativity.",

    "Python is a high-level programming language created by Guido van Rossum. It emphasises code "
    "readability and is widely used in data science and machine learning.",
]

# (question, gold answers (any is correct), category)
TEST_QUESTIONS = [
    ("What is the capital of India?", ["New Delhi"], "simple"),
    ("Who founded Infosys?", ["N. R. Narayana Murthy", "Narayana Murthy"], "simple"),
    ("When was IIT Madras founded?", ["1959"], "simple"),
    ("Where was Sundar Pichai born?", ["Madurai"], "simple"),
    ("Who is the CEO of Google?", ["Sundar Pichai"], "simple"),
    ("What is the currency of Japan?", ["Japanese Yen", "Yen"], "simple"),
    ("What is C. V. Raman known for?", ["Raman effect", "the discovery of the Raman effect"], "simple"),
    ("Who launched Chandrayaan-3?", ["ISRO", "Indian Space Research Organisation"], "simple"),
    ("Where is ISRO headquartered?", ["Bengaluru", "Bangalore"], "simple"),
    ("When was A. P. J. Abdul Kalam born?", ["15 October 1931"], "simple"),

    ("Which city serves as the capital of Tamil Nadu?", ["Chennai"], "paraphrase"),
    ("Who is the founder of VIT University?", ["G. Viswanathan"], "paraphrase"),
    ("In which year was Google established?", ["1998"], "paraphrase"),
    ("Tell me the birthplace of Marie Curie.", ["Warsaw"], "paraphrase"),
    ("Who heads Infosys as chief executive?", ["Salil Parekh"], "paraphrase"),
    ("Which institute did Sundar Pichai attend?", ["IIT Kharagpur"], "paraphrase"),

    ("Chennai is the capital of which state?", ["Tamil Nadu"], "inverse"),
    ("Which company is led by Sundar Pichai?", ["Google"], "inverse"),
    ("Which organisation was founded by Vikram Sarabhai?", ["ISRO", "Indian Space Research Organisation"],
     "inverse"),

    ("In which state was Sundar Pichai born?", ["Tamil Nadu"], "multi-hop"),
    ("In which country is VIT University located?", ["India"], "multi-hop"),
    ("In which country was Albert Einstein born?", ["Germany"], "multi-hop"),
    ("In which state is the headquarters of Infosys?", ["Karnataka"], "multi-hop"),

    ("Who is the CEO of Gogle?", ["Sundar Pichai"], "typo/alias"),
    ("Where was Abdul Kalam born?", ["Rameswaram"], "typo/alias"),
    ("Where is Bangalore located?", ["Karnataka"], "typo/alias"),

    ("How many Nobel Prizes did Marie Curie win?", ["two", "2"], "aggregation"),

    ("When did Chandrayaan-3 land on the Moon?", ["23 August 2023"], "text-only"),
    ("Near which region of the Moon did Chandrayaan-3 land?", ["south pole", "the south pole of the Moon"],
     "text-only"),
    ("By what nickname is Abdul Kalam popularly known?", ["Missile Man of India", "the Missile Man of India"],
     "text-only"),

    ("When was Python released?", ["1991"], "kb-only"),

    ("Who is the CEO of ISRO?", [], "unanswerable"),
    ("What is the capital of Brazil?", [], "unanswerable"),
    ("Who founded Microsoft?", [], "unanswerable"),
]
