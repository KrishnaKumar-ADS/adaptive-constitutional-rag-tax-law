"""
benchmark/metrics/calibration_error.py
"""
import numpy as np

def calculate_ece(results: list[dict], bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE)
    Difference between confidence (1 - uncertainty) and actual accuracy.
    """
    confidences = []
    accuracies = []
    
    for r in results:
        unc = r.get("uncertainty_score", 0.0)
        confidences.append(1.0 - unc)
        
        # Accuracy: 1 if Valid, 0 if Invalid/Partial
        verdict = r.get("verdict", "Valid")
        accuracies.append(1 if verdict == "Valid" else 0)
        
    if not confidences:
        return 0.0
        
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)
    
    bin_boundaries = np.linspace(0, 1, bins + 1)
    ece = 0.0
    
    for i in range(bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
            
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_count / len(results)) * np.abs(bin_acc - bin_conf)
            
    return float(ece)
