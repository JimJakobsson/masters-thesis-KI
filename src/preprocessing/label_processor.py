from typing import Optional
import pandas as pd

class LabelProcessor:
    """Handles creation and processing of labels"""
    
    

    def create_labels(self, df: pd.DataFrame, base_year: int, death_threshold: int) -> pd.DataFrame:
        """Create binary labels based on the 'death_yrmon' column"""
        threshold = base_year + death_threshold
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
            lambda x: self._create_label(x, threshold)
        )
        
        # Clean up and validate
        # Drop rows with null labels
        df = df.dropna(subset=['labels'])
        df['labels'] = df['labels'].astype(int)
        
        self._print_label_distribution(df['labels'])
        return df
    
    @staticmethod
    def _create_label(death_yrmon: str, threshold: int) -> Optional[int]:
        """Create label based on year from 'death_yrmon'"""
        try:
            if death_yrmon and len(death_yrmon) >= 4:
                year = int(death_yrmon[:4])
                return 0 if year > threshold else 1
            return None
        except (ValueError, TypeError):
            return None
        
    @staticmethod    
    def _print_label_distribution(labels: pd.Series) -> None:
        """Print label distribution"""
        distribution = labels.value_counts().to_dict()
        print(f"Labels created: {distribution}")
        print("Labels successfully set")