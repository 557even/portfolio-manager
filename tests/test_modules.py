from portfolio_manager.sentiment import SentimentAnalyzer
from portfolio_manager.reports import generate_morning_brief
from portfolio_manager.trading import basic_signal


def test_pipeline():
    # Use mocked data to avoid network calls
    headlines = [
        "Stocks rise on positive earnings",
        "Market falls amid recession fears",
    ]

    analyzer = SentimentAnalyzer()
    scores = analyzer.score(headlines)
    summary = generate_morning_brief(headlines, scores)
    signal = basic_signal(scores)

    assert isinstance(summary, str)
    assert signal in {"buy", "sell", "hold"}
    assert len(scores) == len(headlines)
