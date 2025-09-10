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
        pass

    def transform(self, df):
        """
        Applies the learned transformations to the data.

        Args:
            df (pd.DataFrame): The dataframe to transform.

        Returns:
            pd.DataFrame: The preprocessed dataframe.
        """
        pass

    def fit_transform(self, df):
        """A convenience method to fit and then transform."""
        pass