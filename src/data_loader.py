import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class DarwinDataLoader:
    def __init__(self, data_path, artifacts_dir="models"):
        self.data_path = data_path
        self.artifacts_dir = artifacts_dir
        os.makedirs(artifacts_dir, exist_ok=True)
        
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()

    def load_and_preprocess(self):
        """Loads data, imputes missing values, scales features, and encodes labels."""
        print(f"Loading dataset from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        
        # 1. Handle Missing Values (Numeric only)
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        # 2. Encode Target
        # Assuming 'class' is the target column based on your original script
        if 'class' not in df.columns:
            raise ValueError("Column 'class' not found in dataset.")
            
        y = self.label_encoder.fit_transform(df['class'])
        X = df[numeric_cols]

        # 3. Scale Features
        X_scaled = self.scaler.fit_transform(X)
        
        # 4. Save Artifacts (Scaler & Encoder) for later inference
        joblib.dump(self.scaler, os.path.join(self.artifacts_dir, "scaler.pkl"))
        joblib.dump(self.label_encoder, os.path.join(self.artifacts_dir, "encoder.pkl"))
        
        return X_scaled, y

    def get_split_data(self, test_size=0.2, random_state=42):
        X, y = self.load_and_preprocess()
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
