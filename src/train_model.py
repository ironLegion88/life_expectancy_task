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
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None # Will handle the bias term explicitly

    def fit(self, X, y):
        """
        Trains the linear regression model.

        Args:
            X (pd.DataFrame or np.ndarray): Training features
            y (pd.Series or np.ndarray): Training target
        """
        n_samples, n_features = X.shape
        
        # 1. Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Convert to numpy arrays for performance
        X = np.array(X)
        y = np.array(y)

        # 2. Implement Gradient Descent
        for _ in range(self.n_iterations):
            # Calculate predictions: y_pred = X.w + b
            y_pred = np.dot(X, self.weights) + self.bias

            # Calculate gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Makes predictions using the trained model.

        Args:
            X (pd.DataFrame or np.ndarray): Features for which to make predictions.

        Returns:
            np.ndarray: The predicted values.
        """
        if self.weights is None:
            raise RuntimeError("The model has not been trained yet. Call 'fit' first.")
        
        # Convert to numpy array
        X = np.array(X)
        
        return np.dot(X, self.weights) + self.bias

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