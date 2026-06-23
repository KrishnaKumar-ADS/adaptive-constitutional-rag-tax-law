import json
import random
from pathlib import Path
import os

def load_jsonl(filepath):
    data = []
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data

def convert_to_chatml(item):
    """
    Format into ChatML style instruction pairs.
    """
    sys_prompt = "You are a precise Indian tax-law assistant. Cite only verified sections. Abstain if evidence is insufficient."
    
    # We will just pass the question as the user prompt for now
    # In a full run, we'd also inject retrieved evidence, but since this dataset
    # is for training the model to behave correctly, we'll append a dummy evidence
    # string or just leave it as question -> answer
    
    citations = ", ".join(item.get("evidence_citations", []))
    ev_text = f"\n\nEvidence: [Section {citations}] ..." if citations else "\n\nEvidence: None"
    
    user_prompt = item["question"] + ev_text
    
    # Format the target assistant response
    confidence = 0.9 if citations else 0.1
    decision = "Answered" if citations else "Abstained"
    
    assistant_resp = f"Answer: {item['answer']}\nEvidence: {citations}\nConfidence: {confidence}\nDecision: {decision}"
    
    return {
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_resp}
        ],
        "category": item.get("category", "unknown")
    }

def prepare_dataset():
    input_dir = Path("data/qa_dataset")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = [
        "factoid_questions.jsonl",
        "reasoning_questions.jsonl",
        "multi_hop_questions.jsonl",
        "unanswerable_questions.jsonl",
        "adversarial_questions.jsonl"
    ]
    
    train_data = []
    eval_data = []
    
    for filename in files:
        filepath = input_dir / filename
        data = load_jsonl(filepath)
        
        if not data:
            print(f"Warning: {filename} is empty or missing.")
            continue
            
        random.shuffle(data)
        
        # 80/20 split
        split_idx = int(len(data) * 0.8)
        
        for item in data[:split_idx]:
            train_data.append(convert_to_chatml(item))
            
        for item in data[split_idx:]:
            eval_data.append(convert_to_chatml(item))
            
    random.shuffle(train_data)
    random.shuffle(eval_data)
    
    with open(output_dir / "finetune_train.jsonl", "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    with open(output_dir / "finetune_eval.jsonl", "w", encoding="utf-8") as f:
        for item in eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Prepared {len(train_data)} train and {len(eval_data)} eval examples.")

if __name__ == "__main__":
    prepare_dataset()
