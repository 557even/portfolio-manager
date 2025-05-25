# Portfolio Manager

Automated Daily Brief for Stock Trading with Sentiment Analysis

This project aims to build a system that fetches financial data and news, analyzes sentiment using GPT-based models, and generates a concise morning briefing alongside automated trading signals. The repository will host data ingestion utilities, sentiment analysis tools, summary generation code, and robo-investing strategies.

## Project Structure

```
src/portfolio_manager/        - Core application modules
├── data_ingestion/           - Fetch market data, news, and transcripts
├── sentiment/                - Sentiment analysis utilities
├── reports/                  - Morning summary generation
├── trading/                  - Robo-investing logic
tests/                        - Unit and integration tests
```

### Example
```python
from portfolio_manager import SentimentAnalyzer, generate_morning_brief, basic_signal
texts = ["Stocks jump after earnings", "Economy signals slowdown"]
sa = SentimentAnalyzer()
sc = sa.score(texts)
print(generate_morning_brief(texts, sc))
print(basic_signal(sc))
```

## Getting Started

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run tests to verify the environment:
   ```bash
   pytest
   ```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is provided for educational purposes and carries no warranty.
