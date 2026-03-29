# main.py
import yfinance as yf
import pandas as pd
import numpy as np
from grader_engine import PersonalizedStockGrader
from sentiment_engine import NewsSentimentEngine

def fetch_stock_data(tickers):
    # Pulls live financial metrics from Yahoo Fincance as ML features
    stock_data = []
    recent_news = {}

    for ticker in tickers:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Feature Extraction
        pe = info.get('trailingPE', info.get('forwardPE', 0))
        debt_to_equity = info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else 0
        roe = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
        eps = info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') else 0
        sector = info.get('sector', 'Unknown')

        stock_data.append({
            'ticker': ticker,
            'sector': sector,
            'pe_ratio': pe,
            'debt_to_equity': debt_to_equity,
            'roe': roe,
            'eps_growth': eps
        })

        # Headline Extraction
        news = stock.news
        headlines = [item.get('title') for item in news if item.get('title')] if news else []
        recent_news[ticker] = headlines

    return pd.DataFrame(stock_data), recent_news

def generate_training_data(samples = 5000):

    np.random.seed(42)


    pe = np.random.normal(loc = 25, scale = 15, size = samples)
    debt_to_equity = np.random.uniform(low = 0, high = 3, size = samples)
    roe = np.random.normal(loc = 15, scale = 10, size = samples)
    eps = np.random.normal(loc = 10, scale = 20, size = samples)

    signal = (roe * 0.8) + (eps * 0.5) - (pe * 0.2) - (debt_to_equity * 5)
    
    noise = np.random.normal(loc = 0, scale = 15, size = samples)

    target_return = signal + noise
    
    return pd.DataFrame({
        'pe_ratio': pe,
        'debt_to_equity': debt_to_equity,
        'roe': roe,
        'eps_growth': eps,
        'target_return': target_return
    })



def main():
    # Target portfolio
    target_tickers = ['SOFI', 'AMD', 'QS']
    preferred_sectors = ['Technology', 'Basic Materials', 'Industrials']

    # Initialize our custom engines
    print("Initializing Grader Engines...")
    sentiment_engine = NewsSentimentEngine()
    grader = PersonalizedStockGrader(preferred_sectors)

    his = generate_training_data()
    grader.train_ml_model(his)
    # Fetch live data
    current_market_data, recent_news_dict = fetch_stock_data(target_tickers)
    
    print("\n--- Raw Data Extracted ---")
    print(current_market_data)
    print("\n--- Sample Headlines Extracted ---")
    for ticker, headlines in recent_news_dict.items():
        print(f"{ticker}: {len(headlines)} recent articles found.")

        

if __name__ == "__main__":
    main()