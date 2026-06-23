import pytest
import numpy as np
from src.uncertainty.features import retrieval_confidence, evidence_agreement, coverage, entropy
from src.uncertainty.uncertainty_model import UncertaintyModel

def test_retrieval_confidence():
    assert retrieval_confidence([0.9, 0.8, 0.7, 0.6]) == pytest.approx(0.8) # Mean of top 3 (0.9, 0.8, 0.7)
    assert retrieval_confidence([0.5]) == pytest.approx(0.5)
    assert retrieval_confidence([]) == 0.0

def test_evidence_agreement():
    assert evidence_agreement([1.0, 0.8, 0.9]) == pytest.approx(0.9)
    assert evidence_agreement([]) == 0.0

def test_coverage():
    query = {"tax", "income", "agricultural"}
    evidence = {"agricultural", "income", "section", "10"}
    assert coverage(query, evidence) == pytest.approx(2/3) # "income", "agricultural" match
    assert coverage(set(), evidence) == 1.0

def test_entropy():
    # High entropy (flat distribution)
    scores_flat = [0.5, 0.5, 0.5, 0.5]
    ent_flat = entropy(scores_flat)
    
    # Low entropy (spiky distribution)
    scores_spike = [0.9, 0.1, 0.1, 0.1]
    ent_spike = entropy(scores_spike)
    
    assert ent_flat > ent_spike
    assert ent_flat > 0.9 # Should be close to 1 for perfectly flat

def test_uncertainty_model_train_predict():
    model = UncertaintyModel()
    
    # Dummy data
    X = np.array([
        [0.9, 0.9, 0.8, 0.2], # Good features -> 0
        [0.8, 0.8, 0.7, 0.3], # Good features -> 0
        [0.2, 0.3, 0.1, 0.9], # Bad features -> 1
        [0.1, 0.2, 0.2, 0.8], # Bad features -> 1
    ])
    y = np.array([0, 0, 1, 1])
    
    model.train(X, y)
    
    preds = model.predict_proba(X)
    assert preds[0] < 0.5
    assert preds[2] > 0.5
