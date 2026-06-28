"""
benchmark/metrics/hallucination_rate.py
"""
def calculate_hallucination_rate(results: list[dict]) -> float:
    """
    Hallucination Rate (HR)
    Percentage of 'Answered' responses where verdict is 'Invalid' or 'Partially Supported'.
    Abstentions do not count towards hallucinations.
    """
    answered_count = 0
    hallucination_count = 0
    
    for r in results:
        if r.get("decision", "Answered") == "Abstained":
            continue
            
        answered_count += 1
        verdict = r.get("verdict", "Valid")
        if verdict in ["Invalid", "Partially Supported"]:
            hallucination_count += 1
            
    if answered_count == 0:
        return 0.0
        
    return hallucination_count / answered_count
