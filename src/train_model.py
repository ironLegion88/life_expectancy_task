import os
import pandas as pd
import numpy as np
import pickle

# Import the preprocessor
from data_preprocessing import Preprocessor

class LinearRegression:
    """
    Implementation of Linear Regression using Gradient Descent.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        # TODO: Implement initialization
        pass

    def fit(self, X, y):
        # TODO: Implement the training logic (Gradient Descent)
        pass

    def predict(self, X):
        # TODO: Implement the prediction logic
        pass

def main():
    """
    Main function to train the model and save it.
    """
    # --- 1. Load and Preprocess Data ---
    DATA_PATH = os.path.join("..", "data", "train_data.csv")
    MODELS_DIR = os.path.join("..", "models")
    
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    print("Loading data...")
    raw_df = pd.read_csv(DATA_PATH)
    
    print("Preprocessing data...")
    preprocessor = Preprocessor()
    processed_df = preprocessor.fit_transform(raw_df)
    
    # Separate features (X) and target (y)
    X = processed_df.drop(columns=['life_expectancy'])
    y = processed_df['life_expectancy']

    # --- 2. Train the Model ---
    print("Training Linear Regression model...")
    # TODO: Instantiate and train the model

    # --- 3. Save the Model ---
    print("Saving model...")
    # TODO: Save the trained model using pickle
    
    print("Training script finished successfully.")

if __name__ == '__main__':
    main()