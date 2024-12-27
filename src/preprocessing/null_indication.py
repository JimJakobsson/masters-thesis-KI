import pandas as pd

class NullIndicator:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def add_null_indicators(self) -> pd.DataFrame:
        """
        Adds binary features indicating whether numerical features are null.

        Returns:
        pd.DataFrame: The dataframe with added null indicator features.
        """
        numerical_features = self.df.select_dtypes(include=[np.number]).columns

        for feature in numerical_features:
            self.df[f'{feature}_nan'] = self.df[feature].isnull().astype(int)
        
        return self.df  
