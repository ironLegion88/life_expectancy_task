import pandas as pd
import numpy as np

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

        # 5. Learn scaling parameters (mean and std) for numeric columns
        # These will be learned after imputation to be technically correct,
        # so we'll compute them on a temporarily imputed dataframe.
        temp_df_imputed = df[self.numeric_cols].fillna(self.imputation_medians)
        self.scaling_params['means'] = temp_df_imputed.mean()
        self.scaling_params['stds'] = temp_df_imputed.std()

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
            raise RuntimeError("You must call 'fit' before calling 'transform'.")

        # 1. Clean column names
        df = self._clean_column_names(df.copy())

        # 2. Drop rows where target is missing
        df.dropna(subset=[self.target_col], inplace=True)
        
        # 3. Separate target variable
        y = df[self.target_col]
        X = df.drop(columns=[self.target_col] + self.features_to_drop, errors='ignore')

        # 4. Impute missing values in numeric columns
        X[self.numeric_cols] = X[self.numeric_cols].fillna(self.imputation_medians)
        
        # 5. Handle categorical variables with one-hot encoding
        X = pd.get_dummies(X, columns=self.categorical_cols, drop_first=True, dtype=float)

        # 6. Scale numerical features
        # Only scale columns that were present during fitting
        cols_to_scale = [col for col in self.numeric_cols if col in X.columns]
        X[cols_to_scale] = (X[cols_to_scale] - self.scaling_params['means']) / self.scaling_params['stds']
        
        # Combine features and target back into one dataframe for simplicity
        processed_df = pd.concat([X, y], axis=1)
        
        print("Data transformed successfully.")
        return processed_df

    def fit_transform(self, df):
        """A convenience method to fit and then transform."""
        pass