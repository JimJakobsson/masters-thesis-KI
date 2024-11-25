from typing import Optional
import pandas as pd

class LabelProcessor:
    """Handles creation and processing of labels"""
    def __init__(self, threshold: int = 2005):
        self._threshold = threshold
    
    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary labels based on the 'death_yrmon' column"""

        df = df.copy()
        # Remove rows with null death_yrmon.
        #inplace=True modifies the original DataFrame
        df.dropna(subset=['death_yrmon'], inplace=True) 
        
        # Standardize death_yrmon format
        df['death_yrmon'] = df['death_yrmon'].apply(
            lambda x: str(int(x)) if pd.notnull(x) else None
        )

        # Create labels
        df['labels'] = df['death_yrmon'].apply(
            lambda x: self._create_label(x)
        )
        
        # Clean up and validate
        df = df.dropna(subset=['labels'])
        df['labels'] = df['labels'].astype(int)
        
        self._print_label_distribution(df['labels'])
        return df
    @staticmethod
    def _create_label(self, death_yrmon: str) -> Optional[int]:
        """Create label based on year from 'death_yrmon'"""
        try:
            if death_yrmon and len(death_yrmon) >= 4:
                year = int(death_yrmon[:4])
                return 0 if year > self._threshold else 1
            return None
        except (ValueError, TypeError):
            return None
        
    def _print_label_distribution(self, labels: pd.Series) -> None:
        """Print label distribution"""
        distribution = labels.value_counts().to_dict()
        print(f"Labels created: {distribution}")
        print("Labels successfully set")