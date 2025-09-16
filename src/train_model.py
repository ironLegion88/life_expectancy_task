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
            if i % 1000 == 0:
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
    Main function to run a hyperparameter tuning experiment, save all models,
    and identify the best one to save as the final model.
    """
    # --- 1. Setup ---
    DATA_PATH = os.path.join("..", "data", "train_data.csv")
    MODELS_DIR = os.path.join("..", "models")
    
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    def calculate_mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def calculate_rmse(y_true, y_pred):
        return np.sqrt(calculate_mse(y_true, y_pred))

    def calculate_r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0: return 1.0
        return 1 - (ss_res / ss_tot)

    # --- 2. Load Data ---
    print("Loading data...")
    raw_df = pd.read_csv(DATA_PATH)
    
    # Define features (X) and target (y) from the raw data
    # We do all preprocessing inside the CV loop
    y = raw_df['Life expectancy ']
    X = raw_df.drop(columns=['Life expectancy '])

    # --- 3. Define Experiments ---
    experiments = [
    ("LinearRegression_LR0.1", LinearRegression, {"learning_rate": 0.1, "n_iterations": 50000}),
    ("LinearRegression_LR0.01", LinearRegression, {"learning_rate": 0.01, "n_iterations": 50000}),
    ("Ridge_A0.01", RidgeRegression, {"learning_rate": 0.1, "n_iterations": 50000, "alpha": 0.01}),
    ("Ridge_A0.5", RidgeRegression, {"learning_rate": 0.1, "n_iterations": 50000, "alpha": 0.5}),
    ("Ridge_A10", RidgeRegression, {"learning_rate": 0.1, "n_iterations": 50000, "alpha": 10.0}),
    ("Lasso_A0.01", LassoRegression, {"learning_rate": 0.1, "n_iterations": 50000, "alpha": 0.01}),
    ("Lasso_A0.5", LassoRegression, {"learning_rate": 0.1, "n_iterations": 50000, "alpha": 0.5}),
    ("Lasso_A1.0", LassoRegression, {"learning_rate": 0.1, "n_iterations": 50000, "alpha": 1.0}),
    ("Poly2_Linear_LR0.01_Iter50k", PolynomialRegression, {"degree": 2, "learning_rate": 0.01, "n_iterations": 50000}),
    ("Poly2_Linear_LR0.001_Iter100k", PolynomialRegression, {"degree": 2, "learning_rate": 0.001, "n_iterations": 100000}),
    ("Poly2_Ridge_A0.1", PolynomialRidgeRegression, {"degree": 2, "learning_rate": 0.001, "n_iterations": 50000, "alpha": 0.1}),
    ("Poly2_Ridge_A1.0", PolynomialRidgeRegression, {"degree": 2, "learning_rate": 0.001, "n_iterations": 50000, "alpha": 1.0}),
    ("Poly2_Ridge_A10.0", PolynomialRidgeRegression, {"degree": 2, "learning_rate": 0.001, "n_iterations": 50000, "alpha": 10.0}),
    ("Poly2_Lasso_A0.1", PolynomialLassoRegression, {"degree": 2, "learning_rate": 0.001, "n_iterations": 50000, "alpha": 0.1}),
    ("Poly2_Lasso_A1.0", PolynomialLassoRegression, {"degree": 2, "learning_rate": 0.001, "n_iterations": 50000, "alpha": 1.0}),
]

    # --- 4. Run Cross-Validation Experiment ---
    folds = k_fold_split(raw_df, k=5, random_seed=42)
    results = []
    best_avg_rmse = float('inf')
    best_model_info = {}

    for exp_idx, (exp_name, model_class, params) in enumerate(experiments, start=1):
        fold_metrics = []
        print(f"\n===== Running Experiment #{exp_idx}: {exp_name} =====")

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            preprocessor = Preprocessor()
            
            # Use .iloc to select rows from the original dataframes
            X_train_raw, y_train_raw = X.iloc[train_idx], y.iloc[train_idx]
            X_val_raw, y_val_raw = X.iloc[val_idx], y.iloc[val_idx]
            
            # FIT the preprocessor ONLY on the training data for this fold
            preprocessor.fit(pd.concat([X_train_raw, y_train_raw], axis=1))
            
            # TRANSFORM both train and validation sets
            processed_train = preprocessor.transform(pd.concat([X_train_raw, y_train_raw], axis=1))
            processed_val = preprocessor.transform(pd.concat([X_val_raw, y_val_raw], axis=1))

            X_train, y_train = processed_train.drop(columns=['life_expectancy']), processed_train['life_expectancy']
            X_val, y_val = processed_val.drop(columns=['life_expectancy']), processed_val['life_expectancy']
            
            # Instantiate a new model for each fold
            model = model_class(**params)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)

            rmse_val = calculate_rmse(y_val, y_pred)
            r2_val = calculate_r2(y_val, y_pred)
            fold_metrics.append({"rmse": rmse_val, "r2": r2_val})
            print(f"  Fold {fold_idx+1}: RMSE={rmse_val:.4f}, R2={r2_val:.4f}")

        avg_rmse = np.mean([m["rmse"] for m in fold_metrics])
        avg_r2 = np.mean([m["r2"] for m in fold_metrics])
        
        results.append({"experiment": f"model{exp_idx}_{exp_name}", "avg_rmse": avg_rmse, "avg_r2": avg_r2})

        if avg_rmse < best_avg_rmse:
            best_avg_rmse = avg_rmse
            # Store the index so we know which model file was the best
            best_model_info = {
                "name": exp_name,
                "class": model_class,
                "params": params,
                "model_index": exp_idx 
            }
            print(f"  >>> New best model found: #{exp_idx} ({exp_name}) with avg RMSE: {avg_rmse:.4f}")

        # --- 5. Train and Save This Experimental Model on Full Data ---
        print(f"  Training '{exp_name}' on full data for saving...")
        full_data_preprocessor = Preprocessor().fit(raw_df)
        processed_full = full_data_preprocessor.transform(raw_df)
        X_full, y_full = processed_full.drop(columns=['life_expectancy']), processed_full['life_expectancy']
        
        exp_model = model_class(**params)
        exp_model.fit(X_full, y_full)
        
        model_path = os.path.join(MODELS_DIR, f"regression_model{exp_idx}.pkl")
        
        # Save both the model and the preprocessor used for it
        artifacts = {'model': exp_model, 'preprocessor': full_data_preprocessor}
        with open(model_path, "wb") as f:
            pickle.dump(artifacts, f)
        print(f"  Saved model to {model_path}")

    # --- 6. Train and Save the FINAL Best Model ---
    print("\n--------------------------------------------------")
    print(f"Best model after all experiments was #{best_model_info['model_index']}: {best_model_info['name']}")
    print(f"Training final version on all data...")

    final_preprocessor = Preprocessor().fit(raw_df)
    processed_final = final_preprocessor.transform(raw_df)
    X_final, y_final = processed_final.drop(columns=['life_expectancy']), processed_final['life_expectancy']
    
    final_model = best_model_info['class'](**best_model_info['params'])
    final_model.fit(X_final, y_final)

    final_artifacts = {
        'model': final_model,
        'preprocessor': final_preprocessor
    }
    final_model_path = os.path.join(MODELS_DIR, "regression_model_final.pkl")
    with open(final_model_path, "wb") as f:
        pickle.dump(final_artifacts, f)
    print(f"Final best model saved to {final_model_path}")

    # --- 7. Print Summary ---
    print("\n===== Experiment Results Summary =====\n")
    summary_df = pd.DataFrame(results).sort_values(by='avg_rmse').reset_index(drop=True)
    print(summary_df)


    # experiemenent runner to make it easy to train multiple models
    experiments = [
        ("LinearRegression_0.3", LinearRegression, {"learning_rate": 0.3, "n_iterations": 10000}),
        ("LinearRegression_0.1", LinearRegression, {"learning_rate": 0.1, "n_iterations": 10000}),
        ("RidgeRegression_0.2", RidgeRegression, {"learning_rate": 0.3, "n_iterations": 10000, "alpha": 0.2}),
        ("RidgeRegression_1", RidgeRegression, {"learning_rate": 0.3, "n_iterations": 10000, "alpha": 1}),
        ("LassoRegression_1", LassoRegression, {"learning_rate": 0.3, "n_iterations": 10000, "alpha": 1}),
        ("LassoRegression_0.2", LassoRegression, {"learning_rate": 0.3, "n_iterations": 10000, "alpha": 0.2}),
        # ("Poly2_Linear", PolynomialRegression, {"learning_rate": 0.01, "n_iterations": 2000}),
        # ("Poly2_Ridge_0.1", PolynomialRidgeRegression, {"learning_rate": 0.01, "n_iterations": 2000, "alpha": 0.1}),
        # ("Poly2_Ridge_1.0", PolynomialRidgeRegression, {"learning_rate": 0.01, "n_iterations": 2000, "alpha": 1.0}),
        # ("Poly2_Lasso_0.1", PolynomialLassoRegression, {"learning_rate": 0.01, "n_iterations": 2000, "alpha": 0.1}),
        # ("Poly2_Lasso_1.0", PolynomialLassoRegression, {"learning_rate": 0.01, "n_iterations": 2000, "alpha": 1.0}),
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