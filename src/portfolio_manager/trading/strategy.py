from typing import List


def basic_signal(scores: List[float], threshold: float = 0.1) -> str:
    """Generate a simple trading signal from sentiment scores."""
    if not scores:
        return "hold"
    avg_score = sum(scores) / len(scores)
    if avg_score > threshold:
        return "buy"
    if avg_score < -threshold:
        return "sell"
    return "hold"
