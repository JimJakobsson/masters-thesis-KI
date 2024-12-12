import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random

class AgeExploration:
    # def __init__(self):
        
    def box_plot_age_combined(self, data):
        """
        Boxplot of age distribution for all classes combined.
        """
        print("Creating boxplot of age distribution for all classes combined...")
        birthdate1 = data['birthdate1']
     
        # Calculate age. The first four digits of the birthdate1 column represent the year of birth.
        age = 1985 - birthdate1.astype(str).str[:4].astype(int)
        age_df = pd.DataFrame({'age': age})
        # Plot
        plt.figure(figsize=(6, 6))
        sns.boxplot(y='age', data=age_df)
        plt.title(f'Age distribution for all classes combined')
        # Save plot as pdf
        plt.savefig('age_distribution_combined.pdf', format='pdf')
        plt.close()
        print("Boxplot created successfully.")

    def box_plot_age_classes(self, X, y):
        """
        Boxplot of age distribution in IPT1.
        """
        print("Creating boxplot of age distribution in IPT1...")
        birthdate1 = X['birthdate1']
       
        #Calculate age. The first four digits of the birthdate1 column represent the year of birth.
        age = 1985 - birthdate1.astype(str).str[:4].astype(int)
        age_df = pd.DataFrame({'age': age, 'labels': y})
        #Plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='labels', y='age', data=age_df)
        plt.title('Age distribution for the classes 0 and 1 in IPT1')
        #save plot as pdf
        plt.savefig('age_distribution.pdf', format='pdf')
        plt.close()
        print("Boxplot created successfully.")

    def age_distribution_histogram(self, data):
        """
        Age distribution in IPT1.
        """
        print("Creating age histogram plot...")
        birthdate1 = data['birthdate1']
        # Calculate age. The first four digits of the birthdate1 column represent the year of birth.
        age = 1985 - birthdate1.astype(str).str[:4].astype(int)
        age_df = pd.DataFrame({'age': age})
        # Plot
        plt.figure(figsize=(10, 6))
        bins = list(range(40, 90, 5))
        sns.histplot(data=age_df, x='age', bins=bins, kde=True)
        plt.xticks(bins)
        plt.title('Age distribution in IPT1')
        plt.xlabel('Age', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.savefig('age_distribution_hist.pdf', format='pdf')
        plt.close()
        print("Age distribution histogram plot created successfully.")

    