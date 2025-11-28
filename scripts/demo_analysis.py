import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_report():
    print("Loading data...")
    try:
        df = pd.read_csv('data/raw/github_functions.csv')
    except FileNotFoundError:
        print("Error: Data file not found.")
        return

    print(f"Total Functions: {len(df)}")
    print(f"Total Repositories: {df['repo_name'].nunique()}")
    print(f"Average Function Length: {df['num_lines'].mean():.2f} lines")
    
    print("\nTop Repositories by Function Count:")
    print(df['repo_name'].value_counts().head())

    # Generate Plot
    plt.figure(figsize=(10, 6))
    sns.histplot(df['num_lines'], bins=50, kde=True)
    plt.title('Distribution of Function Lengths (Collected Data)')
    plt.xlabel('Number of Lines')
    plt.ylabel('Count')
    plt.savefig('docs/function_length_dist.png')
    print("\nPlot saved to docs/function_length_dist.png")

if __name__ == "__main__":
    generate_report()
