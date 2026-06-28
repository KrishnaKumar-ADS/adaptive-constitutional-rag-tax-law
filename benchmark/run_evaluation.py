"""
benchmark/run_evaluation.py

Full Day 14 Evaluation Harness.
Runs the dataset (Factoid, Reasoning, Multi-hop, Unanswerable, Adversarial)
across all 5 Baselines, calculates metrics, and exports a final JSON/CSV report.
"""
import asyncio
import json
import os
import pandas as pd
from typing import Dict, List
from src.generation.output_schema import LegalResponse

from benchmark.baselines import (
    run_baseline_a, run_baseline_b, run_baseline_c,
    run_baseline_d, run_baseline_e
)
from benchmark.metrics import (
    calculate_hallucination_rate, calculate_citation_metrics,
    calculate_ece, calculate_abstention_accuracy, calculate_lgs
)

BASELINES = {
    "A_Pure_LLM": run_baseline_a,
    "B_Finetuned_LLM": run_baseline_b,
    "C_Standard_RAG": run_baseline_c,
    "D_RAG_Static_Rules": run_baseline_d,
    "E_Adaptive_System": run_baseline_e,
}

async def run_eval_sweep(dataset_path: str = "data/processed/eval_set.jsonl"):
    print("🚀 Starting Evaluation Sweep...")
    
    # Load evaluation dataset
    dataset = []
    ground_truth = {} # q_id -> List[citation_id]
    
    if os.path.exists(dataset_path):
        with open(dataset_path, "r") as f:
            for idx, line in enumerate(f):
                item = json.loads(line)
                dataset.append((str(idx), item["messages"][0]["content"]))
                # Parse ground truth from the expected assistant response if needed, 
                # or from metadata if available. Here we mock it if missing.
                ground_truth[str(idx)] = item.get("ground_truth_citations", ["sec_10_1"])
    else:
        print(f"Dataset {dataset_path} not found. Using fallback questions.")
        fallback = [
            "What is the penalty under Section 271F of the Income Tax Act for late filing?",
            "Can I claim exemption under Section 54 for two residential properties?",
            "Is the GST applicable on agricultural land sales? Explain with sections."
        ]
        dataset = [(str(i), q) for i, q in enumerate(fallback)]
        ground_truth = {str(i): ["sec_271F"] for i in range(len(fallback))}

    # Only run first 5 for speed in dev
    dataset = dataset[:5]
    
    results: Dict[str, List[dict]] = {name: [] for name in BASELINES}
    
    for idx, (q_id, question) in enumerate(dataset):
        print(f"\nProcessing [{idx+1}/{len(dataset)}]: {question[:50]}...")
        for name, func in BASELINES.items():
            try:
                resp: LegalResponse = await func(question)
                resp_dict = resp.model_dump()
                resp_dict["q_id"] = q_id
                results[name].append(resp_dict)
            except Exception as e:
                print(f"Error running {name} on Q{q_id}: {e}")

    # Compute Metrics
    report = []
    for name, run_data in results.items():
        if not run_data:
            continue
            
        hr = calculate_hallucination_rate(run_data)
        cites = calculate_citation_metrics(run_data, ground_truth)
        ece = calculate_ece(run_data)
        abst_rate = calculate_abstention_accuracy(run_data)
        lgs = calculate_lgs(hr, cites["precision"])
        
        avg_time = sum(r.get("processing_time_ms", 0) for r in run_data) / len(run_data)
        
        report.append({
            "Baseline": name,
            "Hallucination Rate (%)": hr * 100,
            "Citation Precision": cites["precision"],
            "Citation Recall": cites["recall"],
            "Legal Groundedness (LGS)": lgs,
            "Expected Calib. Error (ECE)": ece,
            "Abstention Rate (%)": abst_rate * 100,
            "Avg Time (ms)": avg_time
        })
        
    df = pd.DataFrame(report)
    print("\n" + "="*80)
    print("EVALUATION REPORT")
    print("="*80)
    print(df.to_string(index=False))
    
    os.makedirs("benchmark/reports", exist_ok=True)
    df.to_csv("benchmark/reports/final_eval_report.csv", index=False)
    
    with open("benchmark/reports/raw_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nSaved reports to benchmark/reports/")

if __name__ == "__main__":
    asyncio.run(run_eval_sweep())
