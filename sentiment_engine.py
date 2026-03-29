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
        if not headlines:
            return 50.0
        
        try:
            results = self.analyzer(headlines)
        except Exception as e:
            print(f"Error analyzing headlines: {e}")
            return 50.0
        
        score = 0

        for result in results:
            label = result['label']
            confidence = result['score']


            if label == 'positive':
                score += (50 + (50 * confidence))
            elif label == 'negative':
                score += (50 - (50 * confidence))
            else:
                score += 50;