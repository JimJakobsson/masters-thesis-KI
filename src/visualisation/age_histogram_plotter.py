import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path


class AgeHistogramPlotter():
    def __init__(self, output_dir: Path):
        """
        Initialize the AgeHistogramPlotter.
        
        Args:
            output_dir (Path): Directory where the plot will be saved
        """
        self.output_dir = output_dir
    

    def plot_age_distribution(self, data: pd.DataFrame) -> None:
        """
        Create a histogram of age distribution with specific age groups.
        
        Args:
            data (pd.DataFrame): DataFrame containing a 'birthdate1' column
        """

        ages = data['age']
        bins = [50, 55, 60, 65, 70, 75, 80]
        labels = ['50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80-85']

        plt.figure(figsize=(10, 9))
        plt.hist(ages, bins=bins, color='b', edgecolor='black', alpha=0.7)
        plt.title('Age Distribution of Participats', fontsize=14, pad=20)
        plt.xlabel('Age Groups', fontsize=12)
        plt.ylabel('Number of Participants', fontsize=12)
        plt.xticks(ticks=bins, labels=labels, rotation=45)
        plt.tight_layout()
        
        # Save plot with DPI=300
        output_path = self.output_dir / 'age_distribution_hist.pdf'
        plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Age distribution histogram plot created successfully and saved to {output_path}")