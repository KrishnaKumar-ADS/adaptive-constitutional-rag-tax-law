"""
Generate 1000 indirect, convoluted, and complex questions and append them to the existing dataset.
These questions are designed to be "confusing" and "not direct", forcing the model to extract the core legal issue.
"""

import json, random, pathlib

random.seed(42)

ROOT = pathlib.Path(__file__).resolve().parent.parent
SECTION_INDEX = ROOT / "data" / "processed" / "section_index.json"
ARTICLE_INDEX = ROOT / "data" / "processed" / "article_index.json"
OUT_TRAIN = ROOT / "data" / "processed" / "finetune_train.jsonl"
OUT_EVAL  = ROOT / "data" / "processed" / "finetune_eval.jsonl"

SYSTEM_PROMPT = (
    "You are a precise Indian tax-law assistant. "
    "Cite only verified sections. Abstain if evidence is insufficient."
)

with open(SECTION_INDEX, "r", encoding="utf-8") as f:
    sections = json.load(f)
with open(ARTICLE_INDEX, "r", encoding="utf-8") as f:
    articles = json.load(f)

all_keys = list(sections.keys()) + list(articles.keys())
corpus = {**sections, **articles}

def _txt(entry: dict) -> str:
    t = entry.get("text", "")
    return t[:1000].rstrip() + ("..." if len(t) > 1000 else "")

def _cite(key: str) -> str:
    if " " in key:
        return key.split(" ", 1)[1]
    return key

CONFUSING_Q_TEMPLATES = [
    "A highly unusual petition has been filed where the applicant conflates the overarching framework of {key} with an entirely unrelated commercial dispute involving an offshore entity. Stripping away the commercial fluff, what is the core legal principle dictated by {key} that would actually govern the procedural aspects of this case?",
    "If a taxpayer deliberately attempts to construct a Byzantine corporate structure to obfuscate their liabilities, claiming that the archaic interpretations of {key} grant them absolute immunity, how does the strict, literal reading of {key} dismantle this convoluted defense?",
    "Consider a paradoxical scenario: A state legislature passes a resolution that seems to directly contravene the federal tax mandate, citing the protections of {key}. However, a deeper reading reveals a contradiction. What are the exact provisions of {key}, and why do they not support such a broad, confusing interpretation?",
    "During a tribunal hearing, an advocate presents a dizzying array of precedents, ultimately hinging their entire multi-million rupee claim on a singular, obscure reading of {key}. Could you clarify the actual, straightforward statutory mandate of {key} to dispel this confusion?",
    "An intricate web of financial transactions spanning three decades is suddenly halted by an assessing officer who vaguely references {key}. The assessee is bewildered, claiming {key} has no relevance to historical trusts. What exactly does {key} state, and how does it cut through this complex temporal dispute?",
    "Imagine a situation where a non-resident alien, operating through a convoluted chain of intermediaries, asserts that {key} completely absolves them of the requirement to maintain statutory accounts. Why is this indirect reliance on {key} legally flawed based on its text?",
    "A legal scholar writes a highly theoretical, 500-page treatise arguing that {key} implicitly rewrites the relationship between the taxpayer and the sovereign by introducing a 'moral obligation' clause. Ignoring the theoretical noise, what are the concrete, written directives of {key}?",
    "Amidst a chaotic corporate merger involving five differing jurisdictions and conflicting state laws, the apex court zeroes in exclusively on {key} to resolve the deadlock. What is the fundamental statutory language in {key} that provides the definitive answer to such a tangled dispute?",
    "A taxpayer files a 200-page grievance claiming that their constitutional right to trade is being violently suppressed by the mere existence of {key}. Given the labyrinthine nature of their complaint, what does {key} actually say, and what are its standard legal implications?",
    "In a seemingly contradictory move, a statutory body cites {key} both to grant a massive tax rebate and simultaneously levy a punitive penalty on the same entity. To untangle this paradoxical application, what is the exact wording and intended scope of {key}?"
]

new_examples = []
keys_list = list(corpus.keys())

# Generate 1000 confusing questions
for _ in range(1000):
    key = random.choice(keys_list)
    entry = corpus[key]
    
    q_tmpl = random.choice(CONFUSING_Q_TEMPLATES)
    q = q_tmpl.format(key=key)
    
    ans = (
        f"Answer: The confusion presented in the scenario stems from a misapplication or overcomplication of the statutory text. "
        f"To resolve the convoluted arguments, we must look at the strict, literal wording of the provision.\n\n"
        f"The text states:\n\"{_txt(entry)}\"\n\n"
        f"**Analysis of the core issue:**\n"
        f"1. **Dispelling the Confusion:** The hypothetical scenario introduces elements (such as offshore disputes, archaic interpretations, or moral obligations) that are largely irrelevant to the explicit mandate of the law.\n"
        f"2. **Strict Application:** The provision strictly dictates the rules within its specific domain. Any indirect or labyrinthine defense that attempts to stretch the meaning of this text beyond its literal bounds is legally invalid.\n"
        f"3. **Conclusion:** By isolating the actual text of {key}, it is clear that the law provides a concrete, straightforward directive that cuts through the surrounding theoretical or commercial noise.\n"
        f"Evidence: {_cite(key)}\n"
        f"Confidence: 0.95\n"
        f"Decision: Answered"
    )
    
    new_examples.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{q}\n\nEvidence: [{key}] ..."},
            {"role": "assistant", "content": ans},
        ],
        "category": "reasoning"
    })

random.shuffle(new_examples)
new_train = new_examples[:800]
new_eval = new_examples[800:]

def read_existing(filepath):
    lines = []
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    lines.append(json.loads(line))
    return lines

old_train = read_existing(OUT_TRAIN)
old_eval = read_existing(OUT_EVAL)

combined_train = old_train + new_train
combined_eval = old_eval + new_eval

with open(OUT_TRAIN, "w", encoding="utf-8") as f:
    for ex in combined_train:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

with open(OUT_EVAL, "w", encoding="utf-8") as f:
    for ex in combined_eval:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Original Train: {len(old_train)}, New Indirect Train: {len(new_train)}, Total Train: {len(combined_train)}")
print(f"Original Eval: {len(old_eval)}, New Indirect Eval: {len(new_eval)}, Total Eval: {len(combined_eval)}")
