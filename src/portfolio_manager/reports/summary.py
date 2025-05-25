from typing import List


def generate_morning_brief(headlines: List[str], scores: List[float]) -> str:
    """Create a simple morning summary from headlines and sentiment scores."""
    lines = ["Morning Brief:"]
    for text, score in zip(headlines, scores):
        sentiment = "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"
        lines.append(f"- {text} ({sentiment})")
    return "\n".join(lines)
