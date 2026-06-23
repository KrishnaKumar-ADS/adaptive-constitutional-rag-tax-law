import json
import random
from pathlib import Path
import sys
import os

# Add parent dir to path so we can import benchmark
sys.path.append(str(Path(__file__).parent.parent))
from benchmark.fake_law_injection import inject_fake_law

def generate_adversarial():
    input_file = Path("data/qa_dataset/factoid_questions.jsonl")
    output_file = Path("data/qa_dataset/adversarial_questions.jsonl")
    
    if not input_file.exists():
        print("Run gen_factoid.py first!")
        return
        
    factoids = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            factoids.append(json.loads(line))
            
    # Take a sample
    random.shuffle(factoids)
    sample = factoids[:20] # For this demo
    
    data = []
    for item in sample:
        orig_q = item["question"]
        fake_q = inject_fake_law(orig_q)
        
        # If the regex didn't find anything, skip or manually inject
        if fake_q == orig_q:
            fake_q = orig_q + " under Section 999Z"
            
        data.append({
            "question": fake_q,
            "answer": "I must abstain. The citation referenced in the question appears to be fabricated or invalid.",
            "evidence_citations": [],
            "category": "adversarial",
            "difficulty": "easy"
        })
        
    with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Saved {len(data)} adversarial questions to {output_file}")

if __name__ == "__main__":
    generate_adversarial()
