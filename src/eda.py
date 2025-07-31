import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def quick_f1_eda(file_path='clean_f1_data.csv'):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return
    
    # Print basic information
    print(f"DataFrame shape: {df.shape}")
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nStatistical summary:")
    print(df.describe())
    
    # Define numeric columns to analyze
    numeric_cols = ['QualifyingPosition', 'GridPosition', 'FinishingPosition', 
                   'AirTemp', 'TrackTemp', 'PitStopCount', 'StintCount', 'NumLaps']
    
    # 1. Weather countplot
    plt.figure(figsize=(10, 6))
    weather_counts = df['Weather'].value_counts()
    sns.countplot(data=df, x='Weather', order=weather_counts.index)
    plt.title("Weather")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # 2. Histograms of numeric columns
    df[numeric_cols].hist(figsize=(12, 8), bins=20)
    plt.tight_layout()
    plt.show()
    
    # 3. Correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlations")
    plt.tight_layout()
    plt.show()
    
    # 4. Qualifying vs Finishing position scatterplot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='QualifyingPosition', y='FinishingPosition', hue='Weather')
    plt.title("Quali vs Finish")
    plt.tight_layout()
    plt.show()
    
    # 5. Grid vs Finishing position scatterplot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='GridPosition', y='FinishingPosition', hue='Team', legend=False)
    plt.title("Grid vs Finish")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    quick_f1_eda() 