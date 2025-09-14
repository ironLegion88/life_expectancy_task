import os
import argparse
import pickle
import pandas as pd
import numpy as np

# Import the Preprocessor class
from data_preprocessing import Preprocessor
# Import the LinearRegression class
from train_model import LinearRegression, RidgeRegression

def calculate_mse(y_true, y_pred):
    """Calculates Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

def calculate_rmse(mse):
    """Calculates Root Mean Squared Error."""
    return np.sqrt(mse)

def calculate_r2(y_true, y_pred):
    """Calculates R-squared (R2) score."""
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

def main(args):
    """
    Main function to load a model, evaluate it, and save results.
    """
    print("Starting evaluation script...")
    
    # --- 1. Load Model and Preprocessor ---
    print(f"Loading model artifacts from {args.model_path}...")
    with open(args.model_path, 'rb') as f:
        artifacts = pickle.load(f)
    
    model = artifacts['model']
    preprocessor = artifacts['preprocessor']

    # --- 2. Load and Preprocess Data ---
    print(f"Loading data from {args.data_path}...")
    eval_df = pd.read_csv(args.data_path)
    
    print("Preprocessing data...")
    # Call 'transform', not 'fit_transform', to use the learned parameters
    processed_df = preprocessor.transform(eval_df)
    
    X_eval = processed_df.drop(columns=['life_expectancy'])
    y_eval = processed_df['life_expectancy']

    # --- 3. Make Predictions ---
    print("Generating predictions...")
    predictions = model.predict(X_eval)
    
    # --- 4. Calculate Metrics ---
    print("Calculating metrics...")
    y_eval_np = np.array(y_eval) # Ensure y is a numpy array
    
    mse = calculate_mse(y_eval_np, predictions)
    rmse = calculate_rmse(mse)
    r2 = calculate_r2(y_eval_np, predictions)
    
    print(f"  MSE: {mse:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R-squared: {r2:.2f}")
    
    # --- 5. Save Predictions and Metrics ---    
    # Save predictions
    print(f"Saving predictions to {args.predictions_output_path}...")
    pd.DataFrame(predictions).to_csv(args.predictions_output_path, header=False, index=False)
    
    # Save metrics in the specified format
    print(f"Saving metrics to {args.metrics_output_path}...")
    with open(args.metrics_output_path, 'w') as f:
        f.write("Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {mse:.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}\n")
        f.write(f"R-squared (R2) Score: {r2:.2f}\n")
        
    print("Evaluation script finished successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained regression model.")
    
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model file.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data CSV file for evaluation.')
    parser.add_argument('--metrics_output_path', type=str, required=True, help='Path to save the evaluation metrics text file.')
    parser.add_argument('--predictions_output_path', type=str, required=True, help='Path to save the predictions CSV file.')
    
    args = parser.parse_args()
    main(args)