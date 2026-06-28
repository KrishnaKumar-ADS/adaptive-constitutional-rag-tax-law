"""
Generate expanded fine-tuning datasets by appending at least 2 detailed
questions per section and article to the original dataset.
"""

import json, random, textwrap, pathlib, hashlib, re

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

# ---------------------------------------------------------------------------
# Load corpus
# ---------------------------------------------------------------------------
with open(SECTION_INDEX, "r", encoding="utf-8") as f:
    sections = json.load(f)
with open(ARTICLE_INDEX, "r", encoding="utf-8") as f:
    articles = json.load(f)

all_keys = list(sections.keys()) + list(articles.keys())
corpus = {**sections, **articles}

def _txt(entry: dict) -> str:
    """Return an extensive portion of the text for detailed answers."""
    t = entry.get("text", "")
    return t[:1000].rstrip() + ("..." if len(t) > 1000 else "")

def _cite(key: str) -> str:
    """'Section 10' -> '10',  'Article 14' -> '14'"""
    if " " in key:
        return key.split(" ", 1)[1]
    return key

# ---------------------------------------------------------------------------
# Generators for Detailed Questions
# ---------------------------------------------------------------------------
def generate_detailed_factoid(key: str, entry: dict) -> dict:
    q_templates = [
        "Provide a comprehensive explanation of {key}.",
        "What are the detailed provisions and implications of {key}?",
        "Please summarize {key} and outline its main directives.",
        "Break down the statutory language of {key}. What does it dictate?",
        "Explain the scope and applicability of {key} in detail."
    ]
    q = random.choice(q_templates).format(key=key)
    
    text_content = _txt(entry)
    ans = (
        f"Answer: {key} lays down specific statutory provisions. "
        f"According to the text:\n\n{text_content}\n\n"
        f"This provision ensures proper legal compliance within its domain by explicitly stating the rules and exceptions where applicable.\n"
        f"Evidence: {_cite(key)}\n"
        f"Confidence: 0.95\n"
        f"Decision: Answered"
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{q}\n\nEvidence: [{key}] ..."},
            {"role": "assistant", "content": ans},
        ],
        "category": "factoid"
    }

def generate_reasoning_or_multihop(key: str, entry: dict) -> dict:
    # 50% chance reasoning, 50% chance multihop
    if random.random() > 0.5:
        # Detailed Reasoning
        q_templates = [
            "A dispute arises regarding the interpretation of {key}. If a taxpayer claims exemption or specific treatment under this provision, how would an assessing officer evaluate the claim based on the strict wording of {key}?",
            "Consider a hypothetical scenario where an entity is audited for compliance with {key}. What are the critical conditions they must satisfy according to the text?",
            "Analyze the legal requirements of {key}. In a case where a party fails to meet these requirements, what is the default legal position?"
        ]
        q = random.choice(q_templates).format(key=key)
        
        ans = (
            f"Answer: Evaluating a claim under {key} requires a strict reading of its statutory text. The provision states:\n"
            f"\"{_txt(entry)}\"\n\n"
            f"**Analysis:**\n"
            f"1. **Conditions Precedent:** The entity must satisfy all mandatory conditions outlined (e.g., filing requirements, time limits, or specific statuses).\n"
            f"2. **Legal Consequence:** If the conditions are met, the statutory benefit or procedure applies. If they fail, the assessing officer or relevant authority is bound to deny the claim or apply the default punitive/taxable provisions as described in the Act.\n"
            f"3. **Conclusion:** Compliance with {key} is mandatory, and any deviation renders the specific claim invalid under the strict interpretation of Indian tax and constitutional law.\n"
            f"Evidence: {_cite(key)}\n"
            f"Confidence: 0.9\n"
            f"Decision: Answered"
        )
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{q}\n\nEvidence: [{key}] ..."},
                {"role": "assistant", "content": ans},
            ],
            "category": "reasoning"
        }
    else:
        # Multi-hop
        other_key = random.choice(all_keys)
        while other_key == key:
            other_key = random.choice(all_keys)
            
        q = f"How do the provisions of {key} interact with the legal framework established in {other_key}? Provide a detailed analysis of both provisions."
        
        ans = (
            f"Answer: To understand the interaction between {key} and {other_key}, we must examine both provisions:\n\n"
            f"**Regarding {key}:**\n{_txt(entry)}\n\n"
            f"**Regarding {other_key}:**\n{_txt(corpus[other_key])}\n\n"
            f"**Interaction Analysis:**\n"
            f"{key} provides a specific substantive or procedural rule, whereas {other_key} may offer an overarching constitutional principle or an additional statutory condition. In Indian law, specific provisions must be read harmoniously with overarching frameworks. If {key} acts as an exception or specific directive, it operates within the boundaries set by {other_key}. A taxpayer or state entity must ensure compliance with both, ensuring that the specific actions under {key} do not violate the broader mandate of {other_key}.\n"
            f"Evidence: {_cite(key)}, {_cite(other_key)}\n"
            f"Confidence: 0.9\n"
            f"Decision: Answered"
        )
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{q}\n\nEvidence: [{key}, {other_key}] ..."},
                {"role": "assistant", "content": ans},
            ],
            "category": "multi_hop"
        }

# ---------------------------------------------------------------------------
# Generate New Dataset
# ---------------------------------------------------------------------------
new_examples = []

for key, entry in corpus.items():
    # At least 2 questions per key
    new_examples.append(generate_detailed_factoid(key, entry))
    new_examples.append(generate_reasoning_or_multihop(key, entry))

# Generate some additional adversarial and unanswerable to maintain robustness
for _ in range(200):
    fake = f"Section {random.randint(1000,9999)}Z"
    q = f"Provide a detailed breakdown of the legal implications of {fake}."
    ans = (
        "Answer: I must abstain. The citation referenced in the question appears to be fabricated or invalid. "
        "No such provision exists in the verifiable text of the Income-Tax Act or the Constitution of India.\n"
        "Evidence: \nConfidence: 0.1\nDecision: Abstained"
    )
    new_examples.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{q}\n\nEvidence: None"},
            {"role": "assistant", "content": ans},
        ],
        "category": "adversarial"
    })

for _ in range(100):
    q = "Explain the detailed tax implications and depreciation rules for a time machine under the Income Tax Act."
    ans = (
        "Answer: I must abstain. The retrieved evidence does not contain sufficient "
        "information to answer this question as the subject matter is fictitious.\n"
        "Evidence: \nConfidence: 0.1\nDecision: Abstained"
    )
    new_examples.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{q}\n\nEvidence: None"},
            {"role": "assistant", "content": ans},
        ],
        "category": "unanswerable"
    })

# Shuffle the new examples
random.shuffle(new_examples)

# Split new examples 80/20
split_idx = int(len(new_examples) * 0.8)
new_train = new_examples[:split_idx]
new_eval  = new_examples[split_idx:]

# ---------------------------------------------------------------------------
# Append to existing files
# ---------------------------------------------------------------------------
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

print(f"Original Train: {len(old_train)}, New Train: {len(new_train)}, Total Train: {len(combined_train)}")
print(f"Original Eval: {len(old_eval)}, New Eval: {len(new_eval)}, Total Eval: {len(combined_eval)}")
print(f"Total new questions generated: {len(new_examples)}")
