import pandas as pd
import numpy as np
import os

def k_fold_split(data, k=5, random_seed=None):
    """
    Splits the dataset indices into k folds for cross-validation.
    Returns a list of (train_indices, val_indices) tuples for each fold.
    Args:
        data (pd.DataFrame or np.ndarray): The dataset to split.
        k (int): Number of folds.
        random_seed (int, optional): Seed for reproducibility.
    Returns:
        List of (train_indices, val_indices) tuples.
    """
    if isinstance(data, pd.DataFrame):
        n_samples = len(data)
    else:
        n_samples = data.shape[0]
    indices = np.arange(n_samples)
    if random_seed is not None:
        np.random.seed(random_seed)
    np.random.shuffle(indices)
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_indices, val_indices))
        current = stop
    return folds

class Preprocessor:
    """
    A class to handle all preprocessing steps for the Life Expectancy dataset.
    This includes cleaning column names, imputing missing values, encoding
    categorical features, and scaling numerical features.
    """
    def __init__(self):
        self.imputation_medians = None
        self.scaling_params = {}
        self.numeric_cols = None
        self.categorical_cols = None
        self.target_col = 'life_expectancy'
        self.features_to_drop = ['country']
        self.one_hot_columns = None

    def _clean_column_names(self, df):
        """Standardizes column names."""
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
        return df

    def fit(self, df):
        """
        Learns the necessary parameters (medians, means, stds) from the
        training data.

        Args:
            df (pd.DataFrame): The training dataframe.
        """
        # 1. Clean column names first
        df = self._clean_column_names(df.copy())

        # 2. Drop rows where the target is missing
        df.dropna(subset=[self.target_col], inplace=True)

        # 3. Identify feature types (before encoding)
        self.numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        # Remove target from numeric columns list
        if self.target_col in self.numeric_cols:
            self.numeric_cols.remove(self.target_col)
        
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        # Remove features planned to drop from the list
        for col in self.features_to_drop:
            if col in self.categorical_cols:
                self.categorical_cols.remove(col)

        # 4. Learn imputation values (medians) for numeric columns
        self.imputation_medians = df[self.numeric_cols].median()
        print(f"{self.imputation_medians}")
        

        # 5. Learn scaling parameters (mean and std) for numeric columns
        # These will be learned after imputation to be technically correct,
        # so we'll compute them on a temporarily imputed dataframe.
        temp_df_imputed = df[self.numeric_cols].fillna(self.imputation_medians)
        self.scaling_params['means'] = temp_df_imputed.mean()
        self.scaling_params['stds'] = temp_df_imputed.std()

        # Store one-hot encoded column names
        temp_X = df.drop(columns=[self.target_col] + self.features_to_drop)
        temp_X = pd.get_dummies(temp_X, columns=self.categorical_cols, drop_first=True, dtype=float)
        self.one_hot_columns = temp_X.columns.to_list()

        print("Preprocessor fitted successfully.")
        return self

    def transform(self, df):
        """
        Applies the learned transformations to the data.

        Args:
            df (pd.DataFrame): The dataframe to transform.

        Returns:
            pd.DataFrame: The preprocessed dataframe.
        """
        if self.imputation_medians is None or not self.scaling_params:
            raise RuntimeError("You must call 'fit' on train dataset before calling 'transform'.")

        # 1. Clean column names
        df = self._clean_column_names(df.copy())

        # 2. Drop rows where target is missing
        df.dropna(subset=[self.target_col], inplace=True)
        
        # 3. Separate target variable
        y = df[self.target_col] if self.target_col in df.columns else None
        X = df.drop(columns=[self.target_col] + self.features_to_drop)

        # 4. Impute missing values in numeric columns
        X[self.numeric_cols] = X[self.numeric_cols].fillna(self.imputation_medians)
        
        # 5. Handle categorical variables with one-hot encoding
        X = pd.get_dummies(X, columns=self.categorical_cols, drop_first=True, dtype=float)

        # 6. Scale numerical features
        # Only scale columns that were present during fitting
        cols_to_scale = [col for col in self.numeric_cols if col in X.columns]
        X[cols_to_scale] = (X[cols_to_scale] - self.scaling_params['means']) / self.scaling_params['stds']
        
        # Combine features and target back into one dataframe for simplicity
        processed_df = pd.concat([X, y], axis=1) if y is not None else X
        
        print("Data transformed successfully.")
        return processed_df

    def fit_transform(self, df):
        """A convenience method to fit and then transform."""
        self.fit(df)
        return self.transform(df)

# Main function to execute preprocessing
if __name__ == '__main__':
    # Load the raw data
    try:
        # Use os.path.join for platform-independent path handling
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'train_data.csv')
        raw_df = pd.read_csv(data_path)
    except FileNotFoundError:
        print("Error: train_data.csv not found. Make sure it's in the 'data' directory.")
    else:
        print("Original DataFrame shape:", raw_df.shape)
        
        # Instantiate and run the preprocessor
        preprocessor = Preprocessor()
        processed_data = preprocessor.fit_transform(raw_df)
        
        print("\nProcessed DataFrame shape:", processed_data.shape)
        print("\nProcessed DataFrame Head:")
        print(processed_data.head())
        
        print("\nChecking for missing values after processing:")
        print(processed_data.isnull().sum().sum())