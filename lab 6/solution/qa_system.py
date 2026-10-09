"""
Lab 6 - Knowledge-Based vs Deep-Learning Question Answering
===========================================================

 1. Construct a small knowledge base of factual triples (saved as JSON + graph picture)
 2. Knowledge-based QA : entity linking -> relation detection -> structured query -> answer
 3. Deep-learning QA   : TF-IDF retriever + pre-trained extractive reader (BERT-family, SQuAD)
                         - distilbert-base-cased-distilled-squad (SQuAD 1.1, always answers)
                         - deepset/roberta-base-squad2           (SQuAD 2.0, can say "no answer")
 4. Hybrid             : KB first, fall back to the neural reader when the KB abstains
 5. Evaluation         : Exact Match, token F1, coverage, precision on answered questions,
                         abstention accuracy on unanswerable questions, latency, per-category EM

Run
    python qa_system.py                                  # full demo + evaluation
    python qa_system.py --ask "Where was Sundar Pichai born?"
    python qa_system.py --interactive
    python qa_system.py --kb-only                        # no model download needed
"""

import argparse
import json
import os
import re
import string
import sys
import time
from collections import Counter, defaultdict

from kb_data import PASSAGES, TEST_QUESTIONS, TRIPLES
from kbqa import KnowledgeBase

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
READERS = {"DistilBERT (SQuAD 1.1)": "distilbert-base-cased-distilled-squad",
           "RoBERTa (SQuAD 2.0)": "deepset/roberta-base-squad2"}
WORDS_TO_NUM = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}


def banner(t):
    print("\n" + "=" * 100)
    print(t)
    print("=" * 100)


# ----------------------------------------------------------------- metrics (SQuAD style)
def norm_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(WORDS_TO_NUM.get(w, w) for w in s.split())


def exact_match(pred, golds):
    if not golds:
        return float(pred is None)
    return float(pred is not None and any(norm_answer(pred) == norm_answer(g) for g in golds))


def f1_score(pred, golds):
    if not golds:
        return float(pred is None)
    if pred is None:
        return 0.0
    best = 0.0
    for g in golds:
        p, t = norm_answer(pred).split(), norm_answer(g).split()
        common = sum((Counter(p) & Counter(t)).values())
        if common:
            pr, rc = common / len(p), common / len(t)
            best = max(best, 2 * pr * rc / (pr + rc))
    return best


def summarise(rows):
    ans = [r for r in rows if r["gold"]]
    unans = [r for r in rows if not r["gold"]]
    answered = [r for r in ans if r["pred"] is not None]
    return {
        "EM": sum(r["em"] for r in rows) / len(rows),
        "F1": sum(r["f1"] for r in rows) / len(rows),
        "EM (answerable)": sum(r["em"] for r in ans) / len(ans),
        "Coverage": len(answered) / len(ans),
        "Precision@answered": (sum(r["em"] for r in answered) / len(answered)) if answered else 0.0,
        "Abstention acc.": (sum(r["pred"] is None for r in unans) / len(unans)) if unans else float("nan"),
        "Latency (ms)": 1000 * sum(r["latency"] for r in rows) / len(rows),
    }


# ----------------------------------------------------------------------- helpers
def save_kb(kb, out_dir):
    path = os.path.join(out_dir, "knowledge_base.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"triples": [list(t) for t in TRIPLES], "entity_types": kb.type_of}, f, indent=1)
    png = os.path.join(out_dir, "knowledge_graph.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx
        g = nx.DiGraph()
        for s, r, o in TRIPLES:
            g.add_edge(s, o, label=r)
        colors = {"country": "#e41a1c", "state": "#ff7f00", "city": "#fdbf6f", "organization": "#377eb8",
                  "person": "#4daf4a", "mission": "#984ea3", "language": "#984ea3"}
        fig = plt.figure(figsize=(22, 16))
        pos = nx.spring_layout(g, k=0.9, seed=3)
        nx.draw_networkx(g, pos, node_size=650, font_size=7, arrowsize=8, edge_color="#999999",
                         node_color=[colors.get(kb.type_of.get(n), "#dddddd") for n in g.nodes])
        nx.draw_networkx_edge_labels(g, pos, edge_labels=nx.get_edge_attributes(g, "label"), font_size=6)
        plt.title("Knowledge graph (red=country, orange=state, yellow=city, blue=organisation, "
                  "green=person, grey=literal)")
        plt.axis("off")
        fig.savefig(png, dpi=110, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        png = f"(graph picture skipped: {e})"
    return path, png


def show_kb_answer(kb, q):
    r = kb.answer(q)
    print(f"Q: {q}")
    links = kb.link_entities(q)
    print(f"   entities : {[(e, how) for e, _, how in links] or '-'}")
    print(f"   relations: {kb.detect_relations(q) or '-'} | expected type: {kb.expected_type(q) or '-'}")
    if r.answer is not None:
        print(f"   query    : {r.query}")
        print(f"   evidence : {' -> '.join(r.path)}")
        print(f"   ANSWER   : {r.answer}")
    else:
        print(f"   ANSWER   : I don't know ({r.reason})")
    return r


def show_neural_answer(name, qa, q):
    r = qa.answer(q)
    pid, sim = r["retrieved"][0]
    print(f"   [{name}] top passage #{pid} (tf-idf {sim:.2f}) | span score {r['score']:.2f}"
          f" | null score {r['null_score']:.2f} | confidence {r['confidence']:.2f}")
    print(f"   [{name}] ANSWER: {r['answer'] if r['answer'] is not None else 'no answer (abstained)'}")
    return r


def main():
    ap = argparse.ArgumentParser(description="Lab 6 - knowledge-based vs deep-learning QA")
    ap.add_argument("--ask", action="append", help="ask your own question (repeatable)")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--kb-only", action="store_true", help="skip the neural readers")
    ap.add_argument("--top-k", type=int, default=2, help="passages passed to the reader")
    args = ap.parse_args()
    out_dir = os.path.join(HERE, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------- 1. KB
    banner("1. KNOWLEDGE BASE CONSTRUCTION")
    kb = KnowledgeBase()
    print(f"{len(TRIPLES)} triples | {len(kb.entities)} entities/literals | {len(kb.relations)} relations")
    print("Relations:", ", ".join(kb.relations))
    print("Sample triples:")
    for t in TRIPLES[::9]:
        print("   ", t)
    js, png = save_kb(kb, out_dir)
    print(f"Saved: {js}\n       {png}")

    # ---------------------------------------------------------- 2. KBQA
    banner("2. KNOWLEDGE-BASED QA - retrieval and answer extraction from structured knowledge")
    for q in ["Who is the CEO of Google?", "Chennai is the capital of which state?",
              "In which country is VIT University located?", "How many Nobel Prizes did Marie Curie win?",
              "Who is the CEO of Gogle?", "When did Chandrayaan-3 land on the Moon?"]:
        show_kb_answer(kb, q)
        print()

    systems = {"Knowledge-based QA": lambda q: (kb.answer(q).answer, {})}
    neural = {}
    if not args.kb_only:
        banner("3. DEEP-LEARNING EXTRACTIVE QA (TF-IDF retriever + pre-trained Transformer reader)")
        from neural_qa import NeuralQA
        for name, model in READERS.items():
            t0 = time.time()
            try:
                neural[name] = NeuralQA(PASSAGES, model, args.top_k)
                print(f"Loaded {model} in {time.time() - t0:.1f}s "
                      f"({'can' if neural[name].reader.can_abstain else 'cannot'} abstain)")
            except Exception as e:
                print(f"Could not load {model}: {type(e).__name__}: {str(e)[:150]}")
        demo = ["Where was Marie Curie born?", "When did Chandrayaan-3 land on the Moon?",
                "In which country is VIT University located?", "Who founded Microsoft?"]
        for q in demo:
            print(f"\nQ: {q}")
            for name, qa in neural.items():
                show_neural_answer(name, qa, q)
        for name, qa in neural.items():
            systems[f"DL: {name}"] = (lambda qa_: (lambda q: (qa_.answer(q)["answer"], {})))(qa)
        if "RoBERTa (SQuAD 2.0)" in neural:
            rob = neural["RoBERTa (SQuAD 2.0)"]

            def hybrid(q):
                a = kb.answer(q).answer
                return (a, {"source": "KB"}) if a is not None else (rob.answer(q)["answer"], {"source": "DL"})
            systems["Hybrid (KB -> RoBERTa fallback)"] = hybrid

    # ------------------------------------------------------- 4. evaluation
    banner(f"4. EVALUATION on {len(TEST_QUESTIONS)} questions "
           f"({sum(1 for _, g, _ in TEST_QUESTIONS if not g)} unanswerable)")
    results = {}
    for sname, fn in systems.items():
        rows = []
        for q, gold, cat in TEST_QUESTIONS:
            t0 = time.time()
            pred, _ = fn(q)
            rows.append({"q": q, "gold": gold, "cat": cat, "pred": pred, "latency": time.time() - t0,
                         "em": exact_match(pred, gold), "f1": f1_score(pred, gold)})
        results[sname] = rows

    names = list(results)
    short = {n: n.replace("DL: ", "").replace(" (KB -> RoBERTa fallback)", "")[:20] for n in names}
    print(f"{'Question':<52}{'Gold':<20}" + "".join(f"{short[n]:<22}" for n in names))
    print("-" * (72 + 22 * len(names)))
    for i, (q, gold, cat) in enumerate(TEST_QUESTIONS):
        cells = []
        for n in names:
            r = results[n][i]
            mark = "+" if r["em"] else ("~" if r["f1"] > 0 else "x")
            cells.append(f"{mark} {str(r['pred'] if r['pred'] is not None else '<no answer>')[:18]:<20}")
        g = gold[0] if gold else "<none>"
        print(f"{q[:50]:<52}{g[:18]:<20}" + "".join(cells))
    print("(+ exact match, ~ partial overlap, x wrong)")

    print(f"\n{'Metric':<22}" + "".join(f"{short[n]:>22}" for n in names))
    summ = {n: summarise(results[n]) for n in names}
    for m in next(iter(summ.values())):
        fmt = "{:>22.1f}" if "Latency" in m else "{:>22.3f}"
        print(f"{m:<22}" + "".join(fmt.format(summ[n][m]) for n in names))

    print(f"\nExact Match per question category:")
    cats = list(dict.fromkeys(c for _, _, c in TEST_QUESTIONS))
    print(f"{'Category':<16}{'#':>4}" + "".join(f"{short[n]:>22}" for n in names))
    for c in cats:
        idx = [i for i, (_, _, cc) in enumerate(TEST_QUESTIONS) if cc == c]
        print(f"{c:<16}{len(idx):>4}" + "".join(
            f"{sum(results[n][i]['em'] for i in idx) / len(idx):>22.2f}" for n in names))

    with open(os.path.join(out_dir, "qa_results.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summ, "per_question": results}, f, indent=1)
    print(f"\nSaved per-question results -> {os.path.join(out_dir, 'qa_results.json')}")

    # ---------------------------------------------------------- 5. analysis
    banner("5. COMPARISON AND ANALYSIS")
    kbs = summ["Knowledge-based QA"]
    print(f"* Knowledge-based QA: precision on answered questions = {kbs['Precision@answered']:.2f}, "
          f"coverage = {kbs['Coverage']:.2f}, abstention accuracy = {kbs['Abstention acc.']:.2f}.")
    print("  + exact, explainable answers (the query and triple path are shown), multi-hop reasoning by "
          "following\n    located_in edges, aggregation (COUNT), knows when it does not know.")
    print("  - only answers what is in the KB and what the hand-written relation patterns recognise "
          "(fails on\n    'land on the Moon', 'nickname'); building and maintaining the KB is manual work.")
    print("  ! caveat: the relation patterns were written together with these test questions, so the KB score "
          "is optimistic;\n    new phrasings (e.g. 'Who runs Google?', 'What's the HQ city of Infosys?') can "
          "break it. Try them with --ask.")
    for n in names:
        if n.startswith("DL"):
            s = summ[n]
            print(f"* {n}: EM = {s['EM']:.2f}, F1 = {s['F1']:.2f}, coverage = {s['Coverage']:.2f}, "
                  f"abstention accuracy = {s['Abstention acc.']:.2f}, latency = {s['Latency (ms)']:.0f} ms.")
    if neural:
        print("  + handles paraphrases and facts that exist only in free text, no schema or rules needed.")
        print("  - can only copy a span from ONE retrieved passage: no multi-hop joins across passages "
              "(VIT -> Vellore -> Tamil Nadu -> India)\n    and no counting; errors propagate from the retriever; "
              "SQuAD-1.1 readers always return a span, even for unanswerable questions.")
    if "Hybrid (KB -> RoBERTa fallback)" in summ:
        h = summ["Hybrid (KB -> RoBERTa fallback)"]
        print(f"* Hybrid: EM = {h['EM']:.2f}, F1 = {h['F1']:.2f} - the KB answers precisely what it covers and "
              "the neural reader fills the gaps,\n  which is how practical QA assistants are built.")
    print("* Metrics: Exact Match and token-F1 (SQuAD) measure answer correctness; coverage vs. "
          "precision@answered shows the\n  answer-or-abstain trade-off; abstention accuracy measures "
          "behaviour on unanswerable questions.")

    # ---------------------------------------------------------- user input
    questions = list(args.ask or [])
    if args.interactive:
        print("\nInteractive mode - type a question (empty line to quit).")
    while True:
        if questions:
            q = questions.pop(0)
        elif args.interactive:
            try:
                q = input("\nQuestion> ").strip()
            except EOFError:
                break
            if not q:
                break
        else:
            break
        banner(f"YOUR QUESTION: {q}")
        kr = show_kb_answer(kb, q)
        for name, qa in neural.items():
            show_neural_answer(name, qa, q)
        if neural:
            fallback = "RoBERTa (SQuAD 2.0)" if "RoBERTa (SQuAD 2.0)" in neural else next(iter(neural))
            final = kr.answer if kr.answer is not None else neural[fallback].answer(q)["answer"]
            print(f"   HYBRID ANSWER: {final if final is not None else 'I do not know'}")


if __name__ == "__main__":
    main()
