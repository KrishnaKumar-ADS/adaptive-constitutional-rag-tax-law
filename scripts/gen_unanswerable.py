import json
from pathlib import Path

def generate_unanswerable():
    output_file = Path("data/qa_dataset/unanswerable_questions.jsonl")
    
    # Hand-crafted unanswerable questions
    questions = [
        "What is the capital gains tax rate on cryptocurrency in Section 115BBH?",
        "How is the Goods and Services Tax (GST) calculated for digital services?",
        "What are the specific penalties for violating the GDPR under Indian tax law?",
        "Does Article 370 of the Constitution provide tax exemptions?",
        "What is the corporate tax rate for alien entities operating in space?",
        "How does the law handle taxation of time travel expenses?",
        "What is the deduction limit for buying a private jet under Section 80C?",
        "Are bribes paid to foreign officials tax deductible?",
        "What is the tax treatment of magical artifacts?",
        "How do I file taxes for my pet dog's modeling income?"
    ]
    
    data = []
    for q in questions:
        data.append({
            "question": q,
            "answer": "I must abstain. The retrieved evidence does not contain sufficient information to answer this question.",
            "evidence_citations": [],
            "category": "unanswerable",
            "difficulty": "medium"
        })
        
    with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Saved {len(data)} unanswerable questions to {output_file}")

if __name__ == "__main__":
    generate_unanswerable()
