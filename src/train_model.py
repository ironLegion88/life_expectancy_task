import os
import pandas as pd
import numpy as np
import pickle

# Import the preprocessor
from data_preprocessing import Preprocessor, k_fold_split
from predict import calculate_mse, calculate_rmse, calculate_r2

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
        
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Convert to numpy arrays for performance
        X = np.array(X)
        y = np.array(y)

        # List to store the cost
        self.cost_history = []

        # Implement Gradient Descent
        for i in range(self.n_iterations):
            # Calculate predictions: y_pred = X.w + b
            y_pred = np.dot(X, self.weights) + self.bias

            # Calculate cost (MSE) for this iteration
            cost = 0.5 * np.mean((y_pred - y) ** 2)
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
            mse = 0.5 * np.mean((y_pred - y) ** 2)
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

class LassoRegression(LinearRegression):
    """
    Implementation of Lasso Regression (L1 Regularization).
    Inherits from LinearRegression and overrides the fitting method to include
    an L1 regularization term using the subgradient method.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000, alpha=0.1):
        super().__init__(learning_rate=learning_rate, n_iterations=n_iterations)
        self.alpha = alpha

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        X = np.array(X)
        y = np.array(y)

        self.cost_history = []

        for i in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias

            mse = 0.5 * np.mean((y_pred - y) ** 2)
            l1_penalty = (self.alpha / n_samples) * np.sum(np.abs(self.weights))
            cost = mse + l1_penalty

            self.cost_history.append(cost)

            if i % 1000 == 0:
                print(f"Iteration {i}: Cost = {cost:.4f}")

            db = (1 / n_samples) * np.sum(y_pred - y)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + (self.alpha / n_samples) * np.sign(self.weights)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db


# Polynomial feature generator (degree 2)
def polynomial_features(X, degree=2):
    """
    Generates polynomial features up to the given degree for input X (numpy array or DataFrame).
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    n_samples, n_features = X.shape

    features = [X]

    # Add polynomial terms
    for d in range(2, degree + 1):
        features.append(X ** d)

    # Add interaction terms (only for degree 2 for simplicity)
    if degree >= 2:
        for i in range(n_features):
            for j in range(i + 1, n_features):
                interaction = (X[:, i] * X[:, j]).reshape(-1, 1)
                features.append(interaction)
                
    return np.concatenate(features, axis=1)

# Polynomial regression classes (degree 2) appended at the end to avoid NameError
class PolynomialRegression(LinearRegression):
    """
    Degree-n polynomial regression using feature expansion.
    """
    def __init__(self, degree=2, learning_rate=0.01, n_iterations=1000):
        super().__init__(learning_rate=learning_rate, n_iterations=n_iterations)
        self.degree = degree

    def fit(self, X, y):
        X_poly = polynomial_features(X, degree=self.degree)
        super().fit(X_poly, y)

    def predict(self, X):
        X_poly = polynomial_features(X, degree=self.degree)
        return super().predict(X_poly)

class PolynomialRidgeRegression(RidgeRegression):
    """
    Degree-n polynomial regression with L2 regularization.
    """
    def __init__(self, degree=2, learning_rate=0.01, n_iterations=1000, alpha=0.1):
        super().__init__(learning_rate=learning_rate, n_iterations=n_iterations, alpha=alpha)
        self.degree = degree

    def fit(self, X, y):
        X_poly = polynomial_features(X, degree=self.degree)
        super().fit(X_poly, y)

    def predict(self, X):
        X_poly = polynomial_features(X, degree=self.degree)
        return super().predict(X_poly)

class PolynomialLassoRegression(LassoRegression):
    """
    Degree-n polynomial regression with L1 regularization.
    """
    def __init__(self, degree=2, learning_rate=0.01, n_iterations=1000, alpha=0.1):
        super().__init__(learning_rate=learning_rate, n_iterations=n_iterations, alpha=alpha)
        self.degree = degree

    def fit(self, X, y):
        X_poly = polynomial_features(X, degree=self.degree)
        super().fit(X_poly, y)

    def predict(self, X):
        X_poly = polynomial_features(X, degree=self.degree)
        return super().predict(X_poly)



def main():
    """
    Main function to train the model and save it.
    """
    DATA_PATH = os.path.join("..", "data", "train_data.csv")
    print(DATA_PATH)
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

    # --- 2. Train the Polynomial Model ---
    print("\n--- Training Polynomial Regression Model ---")
    poly_model = PolynomialRegression(learning_rate=0.001, n_iterations=100000)
    poly_model.fit(X, y)
    print("Polynomial model training complete.")

    print("\n--- Quick Performance Sanity Check ---")
    X_poly = polynomial_features(X, degree=2)
    predictions = poly_model.predict(X)
    
    y_np = np.array(y)
    mse = calculate_mse(y_np, predictions)
    rmse = calculate_rmse(mse)
    r2 = calculate_r2(y_np, predictions)
    
    print(f"  MSE on training data: {mse:.2f}")
    print(f"  RMSE on training data: {rmse:.2f}")
    print(f"  R-squared on training data: {r2:.2f}")


    folds = k_fold_split(processed_df, k=5, random_seed=42)

if __name__ == '__main__':
    main()