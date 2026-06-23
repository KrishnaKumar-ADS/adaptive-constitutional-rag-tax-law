import json
import random
from pathlib import Path
import argparse

def generate_factoids(n=200):
    output_file = Path("data/qa_dataset/factoid_questions.jsonl")
    
    sections = {}
    articles = {}
    
    try:
        with open("data/processed/section_index.json", "r", encoding="utf-8") as f:
            sections = json.load(f)
        with open("data/processed/article_index.json", "r", encoding="utf-8") as f:
            articles = json.load(f)
    except Exception as e:
        print(f"Error loading index: {e}")
        return

    # Combine keys
    all_refs = [("section", k, v) for k, v in sections.items()] + \
               [("article", k, v) for k, v in articles.items()]
               
    random.shuffle(all_refs)
    
    selected = all_refs[:n]
    
    templates = [
        "What does {ref_name} state?",
        "Explain the provisions of {ref_name}.",
        "What is the main subject of {ref_name}?",
        "According to the text, what is {ref_name} about?"
    ]
    
    factoids = []
    
    for ref_type, key, data in selected:
        if ref_type == "section":
            ref_name = key if key.lower().startswith("section") else f"Section {key}"
        else:
            ref_name = key if key.lower().startswith("article") else f"Article {key}"
            
        q_template = random.choice(templates)
        question = q_template.format(ref_name=ref_name)
        
        # We need a short answer. The text might be long, so we take the first couple of sentences.
        text = data.get("text", "")
        # Very naive truncation for factoid answer
        answer = text[:500] + ("..." if len(text) > 500 else "")
        
        # Normalize citation for schema
        cit = key.replace("Section ", "").replace("Article ", "").strip()
        
        factoid = {
            "question": question,
            "answer": answer,
            "evidence_citations": [cit],
            "category": "factoid",
            "difficulty": "easy"
        }
        factoids.append(factoid)
        
    with open(output_file, "w", encoding="utf-8") as f:
        for item in factoids:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Generated {len(factoids)} factoid questions in {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()
    
    generate_factoids(args.n)
