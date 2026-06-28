"""
benchmark/metrics/citation_precision_recall.py
"""
def calculate_citation_metrics(results: list[dict], ground_truth: dict) -> dict:
    """
    Calculate Citation Precision and Recall.
    ground_truth maps question_id to list of relevant section/article IDs.
    """
    total_precision = 0.0
    total_recall = 0.0
    count = 0
    
    for idx, r in enumerate(results):
        q_id = str(idx) # Simplified for baseline test
        true_cites = set(ground_truth.get(q_id, []))
        if not true_cites:
            continue
            
        pred_cites = {ev["citation_id"] for ev in r.get("evidence", [])}
        
        # True Positives
        tp = len(pred_cites.intersection(true_cites))
        
        precision = tp / len(pred_cites) if pred_cites else 0.0
        recall = tp / len(true_cites)
        
        total_precision += precision
        total_recall += recall
        count += 1
        
    if count == 0:
        return {"precision": 0.0, "recall": 0.0}
        
    return {
        "precision": total_precision / count,
        "recall": total_recall / count
    }
