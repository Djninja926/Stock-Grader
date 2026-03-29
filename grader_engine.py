# grader_engine.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class PersonalizedStockGrader:
    def __init__(self, preferred_sectors, ml_weight = 0.4, rule_weight = 0.4, news_weight = 0.2):
        self.preferred_sectors = preferred_sectors
        self.ml_weight = ml_weight
        self.rule_weight = rule_weight
        self.news_weight = news_weight
        
        # The Pipeline: Scales data first, then feeds to the Random Forest
        self.ml_model = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestRegressor(n_estimators = 100, random_state = 42))
        ])

    def train_ml_model(self, historical_data): # Teaches the Random Forest to associate finacial data with future returns
        # Define Features
        x = historical_data[['pe_ratio', 'debt_to_equity', 'roe', 'eps_growth']]
        y = historical_data['target_return']
        
        # Split the data into Training and Testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

        # Train the model
        # print("Fitting Random Forest Regressor to historical data...")
        self.ml_model.fit(x_train, y_train)
        
        # 4. Evaluate the model
        predictions = self.ml_model.predict(x_test)
        
        # Calculate how wrong the model's guesses were
        mse = mean_squared_error(y_test, predictions)
        # Calculate how well the model explains the variance in the data
        r2 = r2_score(y_test, predictions)

        print(f"Training Complete! Model Accuracy Metrics:")
        print(f"  - R-squared (0 to 1): {r2:.2f}")
        print(f"  - Mean Squared Error: {mse:.2f}")
