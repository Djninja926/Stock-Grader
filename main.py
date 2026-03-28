# main.py
import yfinance as yf
import pandas as pd
from grader_engine import PersonalizedStockGrader
from sentiment_engine import NewsSentimentEngine

def main():
    # Defining our target portfolio and interests
    target_tickers = ['VXUS', 'SLV', 'SLDP']
    preferred_sectors = ['Technology', 'Basic Materials', 'Industrials']

    # Initialize our custom engines
    print("Initializing Grader Engines...")
    sentiment_engine = NewsSentimentEngine()
    grader = PersonalizedStockGrader(preferred_sectors)

    # We will build the yfinance data extraction and execution logic here next

if __name__ == "__main__":
    main()