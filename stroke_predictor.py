import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

class StrokePredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.xgb_model = XGBClassifier()
        self.rf_model = RandomForestClassifier()
    
    def preprocess_data(self, df):
        """Preprocess the input dataframe."""
        # Drop ID column if exists
        if 'id' in df.columns:
            df = df.drop("id", axis=1)
        
        # Handle missing values
        df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
        
        # Encode categorical variables
        encoding_maps = {
            'gender': {'Male': 0, 'Female': 1, 'Other': 2},
            'ever_married': {'Yes': 0, 'No': 1},
            'work_type': {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 
                         'children': 3, 'Never_worked': 4},
            'Residence_type': {'Urban': 0, 'Rural': 1},
            'smoking_status': {'formerly smoked': 0, 'never smoked': 1, 
                             'smokes': 2, 'Unknown': 3}
        }
        
        for column, mapping in encoding_maps.items():
            df[column] = df[column].map(mapping)
        
        return df
    
    def train_models(self, X, y):
        """Train both XGBoost and Random Forest models."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        self.xgb_model.fit(X_train_scaled, y_train)
        self.rf_model.fit(X_train_scaled, y_train)
        
        return X_test_scaled, y_test
    
    def save_models(self, xgb_path="model.json", rf_path="model.pkl"):
        """Save trained models to files."""
        self.xgb_model.save_model(xgb_path)
        joblib.dump(self.rf_model, rf_path)
    
    def load_models(self, xgb_path="model.json", rf_path="model.pkl"):
        """Load models from files."""
        self.xgb_model.load_model(xgb_path)
        self.rf_model = joblib.load(rf_path)
