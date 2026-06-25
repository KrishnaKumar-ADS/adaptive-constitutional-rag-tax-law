from src.config import settings

def select_strictness(uncertainty_score: float) -> str:
    """
    Routes the uncertainty score into one of three strictness tiers based on configured thresholds.
    """
    if uncertainty_score < settings.UNCERTAINTY_LOW_THRESHOLD:
        return "low"
    elif uncertainty_score < settings.UNCERTAINTY_HIGH_THRESHOLD:
        return "medium"
    return "high"
