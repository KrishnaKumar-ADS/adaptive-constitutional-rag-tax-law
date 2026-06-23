import json
import random
import time
from pathlib import Path
import argparse
from src.llm.openrouter import inference_openrouter

def generate_reasoning(n=20):
    output_file = Path("data/qa_dataset/reasoning_questions.jsonl")
    
    sections = {}
    try:
        with open("data/processed/section_index.json", "r", encoding="utf-8") as f:
            sections = json.load(f)
    except Exception as e:
        print(f"Error loading index: {e}")
        return

    all_refs = [("section", k, v) for k, v in sections.items()]
    random.shuffle(all_refs)
    
    selected = all_refs[:n]
    reasoning_data = []
    
    print(f"Generating {n} reasoning questions. This will take ~{n * 4} seconds due to rate limits...")
    
    for ref_type, key, data in selected:
        text = data.get("text", "")
        cit = key.replace("Section ", "").replace("Article ", "").strip()
        
        prompt = f"""
Given the following tax law provision:
---
{text[:2000]}
---

Generate ONE reasoning question whose answer requires interpreting (not just quoting) this provision, along with a model answer.
Return the result strictly as a valid JSON object with the following keys:
"question": <the reasoning question string>
"answer": <the detailed answer string>

Do NOT wrap the response in markdown blocks like ```json. Just output the raw JSON object.
"""
        
        try:
            response_text = inference_openrouter(prompt)
            # Basic cleanup if the model wraps in markdown
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
                
            parsed = json.loads(response_text)
            
            reasoning_data.append({
                "question": parsed["question"],
                "answer": parsed["answer"],
                "evidence_citations": [cit],
                "category": "reasoning",
                "difficulty": "medium"
            })
            print(f"✓ Generated reasoning question for Section {cit}")
        except Exception as e:
            print(f"Failed to generate for {cit}: {e}")
            
        # Respect OpenRouter 20 req/min rate limit
        time.sleep(3.5)
        
    with open(output_file, "w", encoding="utf-8") as f:
        for item in reasoning_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Saved {len(reasoning_data)} reasoning questions to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Number of questions to generate")
    args = parser.parse_args()
    
    generate_reasoning(args.n)
