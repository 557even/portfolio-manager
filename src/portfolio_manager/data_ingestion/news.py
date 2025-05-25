import json
from pathlib import Path
from typing import List, Dict


def fetch_news(path: str) -> List[Dict]:
    """Load news articles from a JSON file."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        with p.open() as f:
            data = json.load(f)
    except Exception:
        return []
    if isinstance(data, list):
        return data
    return data.get("articles", [])
