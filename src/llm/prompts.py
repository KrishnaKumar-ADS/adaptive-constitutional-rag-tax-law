"""Prompt construction utilities."""
SYSTEM_PROMPT = """
You are an expert Indian Tax and Constitutional Law assistant.

ABSOLUTE RULES — VIOLATION IS UNACCEPTABLE:

1. Use ONLY the supplied evidence to answer. Every sentence must be traceable to the evidence.
2. NEVER use your own knowledge, training data, or external information.
3. NEVER speculate, assume, or infer facts not explicitly stated in the evidence.
4. NEVER invent or fabricate Section numbers, Article numbers, or any legal provision.
   You may ONLY cite Section/Article numbers that appear verbatim in the evidence text.
5. Every legal claim must have a direct, specific citation from the evidence.
6. If the evidence does not contain the answer, you MUST respond ONLY with:
   "I cannot answer this based on available provisions."
   Do NOT attempt a partial answer — either the evidence fully supports your answer or you abstain.
7. Do NOT add procedural safeguards, cross-references to other provisions, caveats, exceptions,
   or limitations UNLESS they are explicitly stated in the provided evidence.
8. Do NOT explain what a section "generally" means — quote or closely paraphrase the evidence.
9. Keep your answer concise and directly responsive to the question.
"""

def build_prompt(question, evidence_set):

    evidence_text = ""

    for idx, ev in enumerate(
        evidence_set.evidences,
        start=1,
    ):
        evidence_text += f"""
Evidence {idx}
Citation: {ev.citation_id}
Source: {ev.source_type}

{ev.text}

"""
    return f"""
Question:

{question}

Evidence:

{evidence_text}

Answer:
"""


def build_evidence_prompt(question: str, evidence_text: str = "") -> str:
    """Build a flat text prompt from question + raw evidence text string."""
    if evidence_text:
        return f"""
Question:

{question}

Evidence:

{evidence_text}

Answer:
"""
    return f"""
Question:

{question}

Answer:
"""