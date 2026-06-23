import json
import random
import time
from pathlib import Path
import argparse
from src.llm.openrouter import inference_openrouter

def generate_multihop(n=10):
    output_file = Path("data/qa_dataset/multi_hop_questions.jsonl")
    
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

    sec_keys = list(sections.keys())
    art_keys = list(articles.keys())
    
    multihop_data = []
    
    print(f"Generating {n} multihop questions...")
    
    for _ in range(n):
        s_key = random.choice(sec_keys)
        a_key = random.choice(art_keys)
        
        s_text = sections[s_key].get("text", "")
        a_text = articles[a_key].get("text", "")
        
        cit_s = s_key.replace("Section ", "").replace("Article ", "").strip()
        cit_a = a_key.replace("Section ", "").replace("Article ", "").strip()
        
        prompt = f"""
Given the following two legal provisions:

PROVISION 1 (Tax Section {cit_s}):
{s_text[:1500]}

PROVISION 2 (Constitutional Article {cit_a}):
{a_text[:1500]}

Generate ONE multi-hop question that requires synthesizing information from BOTH provisions to answer.
Return the result strictly as a valid JSON object with the following keys:
"question": <the reasoning question string>
"answer": <the detailed answer string>

Do NOT wrap the response in markdown blocks like ```json.
"""
        try:
            # We'll just use the fast model to save credits / rate limits
            response_text = inference_openrouter(prompt, model="openai/gpt-oss-20b:free")
            response_text = response_text.strip()
            if response_text.startswith("```json"): response_text = response_text[7:]
            if response_text.startswith("```"): response_text = response_text[3:]
            if response_text.endswith("```"): response_text = response_text[:-3]
                
            parsed = json.loads(response_text)
            
            multihop_data.append({
                "question": parsed["question"],
                "answer": parsed["answer"],
                "evidence_citations": [cit_s, cit_a],
                "category": "multi_hop",
                "difficulty": "hard"
            })
            print(f"✓ Generated multi-hop question for Section {cit_s} + Article {cit_a}")
        except Exception as e:
            print(f"Failed to generate multihop: {e}")
            
        time.sleep(3.5)
        
    with open(output_file, "w", encoding="utf-8") as f:
        for item in multihop_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Saved {len(multihop_data)} multihop questions to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()
    generate_multihop(args.n)
