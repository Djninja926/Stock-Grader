# sentiment_engine.py
from transformers import pipeline
import warnings

# Suppress some of the Hugging Face warning noise for cleaner terminal output
warnings.filterwarnings("ignore", category=UserWarning)

class NewsSentimentEngine:
    def __init__(self):
        print("Loading FinBERT NLP Model (this may take a moment)...")
        self.analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")

    def analyze_headlines(self, headlines):
        # We will build the logic here next
        pass