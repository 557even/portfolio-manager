# Portfolio Manager

Automated Daily Brief for Stock Trading with Sentiment Analysis

This project aims to build a system that fetches financial data and news, analyzes sentiment using GPT-based models, and generates a concise morning briefing alongside automated trading signals. The repository will host data ingestion utilities, sentiment analysis tools, summary generation code, and robo-investing strategies.

Detailed steps for implementing these features can be found in the [project roadmap](docs/ROADMAP.md).

## Project Structure

```
src/portfolio_manager/        - Core application modules
├── data_ingestion/           - Fetch market data, news, and transcripts
├── sentiment/                - Sentiment analysis utilities
├── reports/                  - Morning summary generation
├── trading/                  - Robo-investing logic
tests/                        - Unit and integration tests
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
