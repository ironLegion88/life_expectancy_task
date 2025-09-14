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

        # List to store the cost
        self.cost_history = []

        # 2. Implement Gradient Descent
        for i in range(self.n_iterations):
            # Calculate predictions: y_pred = X.w + b
            y_pred = np.dot(X, self.weights) + self.bias

            # Calculate cost (MSE) for this iteration
            cost = np.mean((y_pred - y) ** 2)
            self.cost_history.append(cost)

            # Print cost every 100 iterations to monitor progress
            if i % 100 == 0:
                print(f"Iteration {i}: MSE = {cost:.4f}")

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
    
class RidgeRegression(LinearRegression):
    """
    Implementation of Ridge Regression (L2 Regularization).
    Inherits from LinearRegression and overrides the fitting method to include
    a regularization term.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000, alpha=0.1):
        super().__init__(learning_rate=learning_rate, n_iterations=n_iterations)
        self.alpha = alpha

    def fit(self, X, y):
        """
        Trains the ridge regression model by overriding the parent's fit method.
        """
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0

        X = np.array(X)
        y = np.array(y)
        
        self.cost_history = []

        # Gradient Descent with L2 Regularization
        for i in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Cost (MSE + L2 penalty)
            mse = np.mean((y_pred - y) ** 2)
            l2_penalty = (self.alpha / (2 * n_samples)) * np.sum(np.square(self.weights))
            cost = mse + l2_penalty
            self.cost_history.append(cost)

            if i % 1000 == 0:
                print(f"Iteration {i}: Cost = {cost:.4f}")

            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Weight gradient with the regularization term
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + (self.alpha / n_samples) * self.weights
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

def main():
    """
    Main function to train the model and save it.
    """
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

    print("Training Ridge Regression model...")
    ridge_model = RidgeRegression(learning_rate=0.01, n_iterations=10000, alpha=0.5)
    ridge_model.fit(X, y)
    
    print("Model training complete.")
    print(f"Learned Bias (Intercept): {ridge_model.bias:.4f}")
    print(f"Learned Weights: {ridge_model.weights}")

    MODEL_ARTIFACTS = {
        'model': ridge_model,
        'preprocessor': preprocessor
    }
    
    MODEL_PATH = os.path.join(MODELS_DIR, "regression_model3.pkl")
    print(f"Saving model artifacts to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(MODEL_ARTIFACTS, f)
    
    print("Training script finished successfully.")

if __name__ == '__main__':
    main()