import os
import pandas as pd
import numpy as np
import pickle

# Import the preprocessor
from data_preprocessing import Preprocessor, k_fold_split

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
    Only supports degree=2 for now.
    Returns a new numpy array with original, squared, and interaction terms.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    n_samples, n_features = X.shape
    features = [X]

    features.append(X ** degree)
    for i in range(n_features):
        for j in range(i+1, n_features):
            interaction = (X[:, i] * X[:, j]).reshape(-1, 1)
            features.append(interaction)
    return np.concatenate(features, axis=1)

class PolynomialRegression(LinearRegression):
    """
    Simple degree-2 polynomial regression using LinearRegression on polynomial features.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        super().__init__(learning_rate=learning_rate, n_iterations=n_iterations)

    def fit(self, X, y):
        X_poly = polynomial_features(X, degree=2)
        super().fit(X_poly, y)

    def predict(self, X):
        X_poly = polynomial_features(X, degree=2)
        return super().predict(X_poly)

class PolynomialRidgeRegression(RidgeRegression):
    """
    Degree-2 polynomial regression with L2 regularization.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000, alpha=0.1):
        super().__init__(learning_rate=learning_rate, n_iterations=n_iterations, alpha=alpha)

    def fit(self, X, y):
        X_poly = polynomial_features(X, degree=2)
        super().fit(X_poly, y)

    def predict(self, X):
        X_poly = polynomial_features(X, degree=2)
        return super().predict(X_poly)

class PolynomialLassoRegression(LassoRegression):
    """
    Degree-2 polynomial regression with L1 regularization.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000, alpha=0.1):
        super().__init__(learning_rate=learning_rate, n_iterations=n_iterations, alpha=alpha)

    def fit(self, X, y):
        X_poly = polynomial_features(X, degree=2)
        super().fit(X_poly, y)

    def predict(self, X):
        X_poly = polynomial_features(X, degree=2)
        return super().predict(X_poly)



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
    
    X = processed_df.drop(columns=['life_expectancy'])
    y = processed_df['life_expectancy']

    folds = k_fold_split(processed_df, k=5, random_seed=42)

    # experiemenent runner to make it easy to train multiple models
    experiments = [
        ("LinearRegression_0.01", LinearRegression, {"learning_rate": 0.01, "n_iterations": 2000}),
        ("LinearRegression_0.05", LinearRegression, {"learning_rate": 0.05, "n_iterations": 2000}),
        ("RidgeRegression_0.1", RidgeRegression, {"learning_rate": 0.01, "n_iterations": 2000, "alpha": 0.1}),
        ("RidgeRegression_1.0", RidgeRegression, {"learning_rate": 0.01, "n_iterations": 2000, "alpha": 1.0}),
        ("LassoRegression_0.1", LassoRegression, {"learning_rate": 0.01, "n_iterations": 2000, "alpha": 0.1}),
        ("LassoRegression_1.0", LassoRegression, {"learning_rate": 0.01, "n_iterations": 2000, "alpha": 1.0}),
        ("Poly2_Linear", PolynomialRegression, {"learning_rate": 0.01, "n_iterations": 2000}),
        ("Poly2_Ridge_0.1", PolynomialRidgeRegression, {"learning_rate": 0.01, "n_iterations": 2000, "alpha": 0.1}),
        ("Poly2_Ridge_1.0", PolynomialRidgeRegression, {"learning_rate": 0.01, "n_iterations": 2000, "alpha": 1.0}),
        ("Poly2_Lasso_0.1", PolynomialLassoRegression, {"learning_rate": 0.01, "n_iterations": 2000, "alpha": 0.1}),
        ("Poly2_Lasso_1.0", PolynomialLassoRegression, {"learning_rate": 0.01, "n_iterations": 2000, "alpha": 1.0}),
    ]

    def mse(y_true, y_pred): return np.mean((y_true - y_pred) ** 2)
    def rmse(y_true, y_pred): return np.sqrt(mse(y_true, y_pred))
    def r2_score(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0: return 1.0 # Handle case of perfect prediction or constant y
        return 1 - (ss_res / ss_tot)

    results = []
    best_avg_rmse = float('inf')
    best_model_info = {}

    for exp_idx, (exp_name, model_class, params) in enumerate(experiments):
        fold_metrics = []
        print(f"\nRunning Experiment: {exp_name}")

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            # Create a NEW model instance for each fold
            model = model_class(**params)

            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
        
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            mse_val = mse(y_val, y_pred)
            rmse_val = rmse(y_val, y_pred)
            r2_val = r2_score(y_val, y_pred)
            fold_metrics.append({"mse": mse_val, "rmse": rmse_val, "r2": r2_val})
            print(f"  Fold {fold_idx}: RMSE={rmse_val:.4f}, R2={r2_val:.4f}")

        avg_rmse = np.mean([m["rmse"] for m in fold_metrics])
        avg_r2 = np.mean([m["r2"] for m in fold_metrics])
        
        results.append({"experiment": exp_name, "avg_rmse": avg_rmse, "avg_r2": avg_r2})

        if avg_rmse < best_avg_rmse:
            best_avg_rmse = avg_rmse
            best_model_info = {
                "name": exp_name,
                "class": model_class,
                "params": params,
                "avg_rmse": avg_rmse
            }
            print(f"  New best model found: {exp_name} with avg RMSE: {avg_rmse:.4f}")

    # After finding the best hyperparameters, train the final model on ALL data
    print("\n--------------------------------------------------")
    print(f"Best model hyperparameters found: {best_model_info['name']}")
    print(f"Training final model on all data...")

    final_model = best_model_info['class'](**best_model_info['params'])
    
    
    final_model.fit(X, y)

    final_model_path = os.path.join(MODELS_DIR, "regression_model_final.pkl")
    with open(final_model_path, "wb") as f:
        pickle.dump(final_model, f)
    print(f"Best model saved to {final_model_path}")

    # Print summary table
    print("Experiment Results Summary:\n")
    summary_df = pd.DataFrame(results).sort_values(by='avg_rmse').reset_index(drop=True)
    print(summary_df)

if __name__ == '__main__':
    main()