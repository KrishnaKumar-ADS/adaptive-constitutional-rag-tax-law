from sklearn.calibration import CalibratedClassifierCV
import joblib
from pathlib import Path
import numpy as np

def calibrate_and_save(base_model, X_val: np.ndarray, y_val: np.ndarray, save_path: str = "data/processed/calibrated_uncertainty_model.joblib"):
    """
    Applies Platt scaling (sigmoid calibration) to the base model using a validation set,
    and saves the resulting calibrated model to disk.
    """
    calibrated = CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv=2)
    calibrated.fit(X_val, y_val)
    
    # Ensure directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(calibrated, save_path)
    print(f"Saved calibrated model to {save_path}")
    return calibrated

def load_calibrated_model(load_path: str = "data/processed/calibrated_uncertainty_model.joblib"):
    """
    Loads the calibrated model from disk.
    """
    return joblib.load(load_path)
