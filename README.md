# Portfolio Manager

A research project blending the market coverage of **Bloomberg** with the community insights of **Seeking Alpha**. The goal is to create a personal analytics and trading platform that aggregates real-time market data, performs sentiment analysis on news and transcripts, and produces concise portfolio reports with automated trade suggestions.

## Overview

The application collects data from multiple financial APIs, analyzes sentiment using GPT-based models, and surfaces trading signals. The project is organized into small modules so that each feature can evolve independently.

## Project Structure

```
src/portfolio_manager/        - Core application packages
├── data_ingestion/           - Download market data, news, and transcripts
├── market_data/              - Real-time price and fundamentals interface
├── sentiment/                - Sentiment analysis utilities
├── analysis/                 - Quantitative portfolio analytics
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
