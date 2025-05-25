from typing import List

POSITIVE_WORDS = {"rise", "gain", "up", "surge", "positive", "bull"}
NEGATIVE_WORDS = {"fall", "drop", "down", "recession", "negative", "bear"}


class SentimentAnalyzer:
    """Very small rule-based sentiment analyzer."""

    def score(self, texts: List[str]) -> List[float]:
        scores = []
        for text in texts:
            lower = text.lower()
            score = 0
            for word in POSITIVE_WORDS:
                if word in lower:
                    score += 1
            for word in NEGATIVE_WORDS:
                if word in lower:
                    score -= 1
            scores.append(score / max(len(text.split()), 1))
        return scores
