# grader_engine.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class PersonalizedStockGrader:
    def __init__(self, preferred_sectors, ml_weight=0.4, rule_weight=0.4, news_weight=0.2):
        self.preferred_sectors = preferred_sectors
        self.ml_weight = ml_weight
        self.rule_weight = rule_weight
        self.news_weight = news_weight
        
        self.ml_model = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

    def train_ml_model(self, historical_data):
        # We will build the training logic here next
        pass