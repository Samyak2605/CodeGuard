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

def check_day4():
    print_header("DAY 4: EDA & FEATURE ENGINEERING")
    
    # Check Notebooks
    notebooks = ["notebooks/03_EDA_Deep_Dive.ipynb", "notebooks/04_Feature_Analysis.ipynb"]
    for nb in notebooks:
        if os.path.exists(nb):
            print(f"✅ Found Notebook: {nb}")
        else:
            print(f"❌ Missing Notebook: {nb}")

    # Check Source Code
    src_files = ["src/features/feature_extractor.py", "src/features/preprocessor.py"]
    for f in src_files:
        if os.path.exists(f):
            print(f"✅ Found Source: {f}")
        else:
            print(f"❌ Missing Source: {f}")

    # Check Splits
    splits = {
        "Train": "data/processed/train.csv",
        "Val": "data/processed/val.csv",
        "Test": "data/processed/test.csv"
    }
    
    all_splits_exist = True
    for name, path in splits.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"✅ Found {name} Set: {path} ({len(df)} samples)")
            if name == "Train":
                # Check for features
                feature_cols = [c for c in df.columns if c not in ['code', 'repo_name', 'quality_score', 'has_long_method']]
                print(f"   - Features Detected: {len(feature_cols)} (e.g., {', '.join(feature_cols[:3])}...)")
        else:
            print(f"❌ Missing {name} Set: {path}")
            all_splits_exist = False
            
    if all_splits_exist:
        print("\nResult: Day 4 Feature Engineering COMPLETE")
    else:
        print("\nResult: Day 4 Feature Engineering INCOMPLETE")

def check_day5():
    print_header("DAY 5: BASELINE ML MODELS")
    
    # Check Scripts & Configs
    files = [
        "configs/mlflow_config.py",
        "src/models/data_loader.py",
        "src/models/xgboost_baseline.py",
        "scripts/train_baselines.py"
    ]
    for f in files:
        if os.path.exists(f):
            print(f"✅ Found File: {f}")
        else:
            print(f"❌ Missing File: {f}")

    # Check Models
    models_dir = "models/baselines"
    if os.path.exists(models_dir):
        models = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        print(f"✅ Found {len(models)} Trained Models in {models_dir}")
    else:
        print(f"❌ Missing Models Directory: {models_dir}")

    # Check Results
    results_file = "results/baseline_results.csv"
    if os.path.exists(results_file):
        print(f"✅ Found Results: {results_file}")
        try:
            df = pd.read_csv(results_file)
            print("\nBaseline Performance Summary:")
            print(df[['smell', 'val_f1', 'test_f1']].to_string(index=False))
        except Exception as e:
            print(f"⚠️ Could not read results: {e}")
    else:
        print(f"❌ Missing Results: {results_file}")

    # Check Analysis
    nb = "notebooks/05_Baseline_Analysis.ipynb"
    if os.path.exists(nb):
        print(f"✅ Found Analysis Notebook: {nb}")
    else:
        print(f"❌ Missing Analysis Notebook: {nb}")

    print("\nResult: Day 5 Baseline Models COMPLETE")

def check_day6():
    print_header("DAY 6: BASELINE OPTIMIZATION")
    
    # Check Scripts
    scripts = [
        "src/models/smote_handler.py",
        "src/models/xgboost_with_smote.py",
        "src/models/hyperparameter_tuning.py",
        "src/models/random_forest_baseline.py",
        "src/models/ensemble.py"
    ]
    for s in scripts:
        if os.path.exists(s):
            print(f"✅ Found Script: {s}")
        else:
            print(f"❌ Missing Script: {s}")

    # Check Results CSVs
    csvs = [
        "results/smote_results.csv",
        "results/hyperparameter_tuning_results.csv",
        "results/random_forest_results.csv",
        "results/ensemble_results.csv"
    ]
    for c in csvs:
        if os.path.exists(c):
            print(f"✅ Found Results: {c}")
        else:
            print(f"❌ Missing Results: {c}")

    # Check Final Report
    report = "results/BASELINE_FINAL_REPORT.md"
    if os.path.exists(report):
        print(f"✅ Found Final Report: {report}")
        
        # Show Ensemble Results
        ensemble_res = "results/ensemble_results.csv"
        if os.path.exists(ensemble_res):
            try:
                df = pd.read_csv(ensemble_res)
                print("\nEnsemble Performance (Best Models):")
                print(df[['smell', 'ensemble_type', 'val_f1']].to_string(index=False))
            except:
                pass
    else:
        print(f"❌ Missing Final Report: {report}")

    print("\nResult: Day 6 Optimization COMPLETE")

if __name__ == "__main__":
    print_header("CODEGUARD PROJECT DEMO (DAYS 1-6)")
    check_day1()
    check_day2()
    check_day3()
    check_day4()
    check_day5()
    check_day6()
    print("\n" + "="*60)
    print(" DEMO COMPLETE")
    print("="*60)
