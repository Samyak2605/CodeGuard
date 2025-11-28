import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def print_header(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def check_day1():
    print_header("DAY 1: PROJECT SETUP")
    required_files = [
        "src/__init__.py", "configs/config.yaml", "requirements.txt", 
        "README.md", ".gitignore", "setup.py"
    ]
    missing = []
    for f in required_files:
        if os.path.exists(f):
            print(f"✅ Found {f}")
        else:
            print(f"❌ Missing {f}")
            missing.append(f)
            
    if not missing:
        print("\nResult: Day 1 Setup COMPLETE")
    else:
        print("\nResult: Day 1 Setup INCOMPLETE")

def check_day2():
    print_header("DAY 2: DATA COLLECTION")
    raw_file = "data/raw/github_functions.csv"
    if os.path.exists(raw_file):
        df = pd.read_csv(raw_file)
        print(f"✅ Raw Data File Found: {raw_file}")
        print(f"   - Total Functions Collected: {len(df)}")
        print(f"   - Repositories: {df['repo_name'].nunique()}")
        print("\nResult: Day 2 Collection COMPLETE")
        return df
    else:
        print(f"❌ Raw Data File Missing: {raw_file}")
        print("\nResult: Day 2 Collection INCOMPLETE")
        return None

def check_day3():
    print_header("DAY 3: AUTOMATED LABELING")
    clean_file = "data/processed/labeled_functions_clean.csv"
    if os.path.exists(clean_file):
        df = pd.read_csv(clean_file)
        print(f"✅ Clean Labeled Dataset Found: {clean_file}")
        print(f"   - Total Clean Functions: {len(df)}")
        
        smells = ['has_long_method', 'has_high_complexity', 'has_too_many_params', 'has_deep_nesting', 'has_no_docstring']
        print("\nCode Smell Distribution:")
        for smell in smells:
            count = df[smell].sum()
            pct = (count / len(df)) * 100
            print(f"   - {smell.replace('has_', '').title()}: {count} ({pct:.1f}%)")
            
        print(f"\n   - Average Quality Score: {df['quality_score'].mean():.1f}/100")
        
        # Generate Plot
        plt.figure(figsize=(12, 6))
        counts = df[smells].sum().sort_values(ascending=False)
        sns.barplot(x=counts.values, y=[s.replace('has_', '').replace('_', ' ').title() for s in counts.index])
        plt.title('Code Smell Prevalence in Clean Dataset')
        plt.xlabel('Count')
        plt.tight_layout()
        plt.savefig('docs/demo_day3_results.png')
        print("\n✅ Visualization saved to docs/demo_day3_results.png")
        
        print("\nResult: Day 3 Labeling COMPLETE")
    else:
        print(f"❌ Clean Labeled Dataset Missing: {clean_file}")
        print("\nResult: Day 3 Labeling INCOMPLETE")

if __name__ == "__main__":
    print_header("CODEGUARD PROJECT DEMO (DAYS 1-3)")
    check_day1()
    check_day2()
    check_day3()
    print("\n" + "="*60)
    print(" DEMO COMPLETE")
    print("="*60)
