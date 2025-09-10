import os
import argparse
import pickle
import pandas as pd
import numpy as np

# Import the Preprocessor class
from data_preprocessing import Preprocessor
# Import the LinearRegression class
from train_model import LinearRegression


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
    
    # TODO: Calculate metrics
    
    # TODO: Save predictions and metrics
    
    print("Evaluation script finished successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained regression model.")
    
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model file.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data CSV file for evaluation.')
    parser.add_argument('--metrics_output_path', type=str, required=True, help='Path to save the evaluation metrics text file.')
    parser.add_argument('--predictions_output_path', type=str, required=True, help='Path to save the predictions CSV file.')
    
    args = parser.parse_args()
    main(args)