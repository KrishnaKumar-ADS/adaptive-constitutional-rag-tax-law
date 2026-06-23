"""Prompt construction utilities."""
SYSTEM_PROMPT = """
You are an expert Indian Tax and Constitutional Law assistant.

STRICT RULES:

1. Use ONLY the supplied evidence.
2. Never use external knowledge.
3. Never speculate.
4. Never infer facts not present in evidence.
5. Every legal claim must be supported by evidence.
6. Cite section/article numbers.
7. If evidence is insufficient say:

   "I do not have sufficient legal evidence to answer."

8. Do not mention possibilities,
   assumptions,
   conditions,
   exceptions,
   or limitations unless they appear in evidence.
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