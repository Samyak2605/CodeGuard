import pandas as pd
import random
import os

def manual_verification(input_csv="data/processed/labeled_functions.csv", sample_size=20):
    """
    Interactive manual verification tool.
    """
    print("Loading labeled dataset...")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: File {input_csv} not found.")
        return

    smells = ['has_long_method', 'has_high_complexity', 'has_too_many_params', 'has_deep_nesting', 'has_no_docstring']
    
    results = {smell: {'correct': 0, 'incorrect': 0} for smell in smells}
    
    print(f"\nStarting Manual Verification (Sample size: {sample_size} per smell)")
    print("="*60)
    
    for smell in smells:
        print(f"\nVerifying: {smell}")
        print("-" * 30)
        
        # Filter for positive examples of this smell
        candidates = df[df[smell] == True]
        
        if len(candidates) < sample_size:
            print(f"Warning: Only {len(candidates)} samples found for {smell}")
            samples = candidates
        else:
            samples = candidates.sample(sample_size)
            
        for idx, row in samples.iterrows():
            print(f"\nFunction: {row['function_name']} (from {row['repo_name']})")
            print(f"File: {row['file_path']}")
            print("-" * 20)
            print(row['code'])
            print("-" * 20)
            print(f"Label: {smell} = TRUE")
            print(f"Metrics: Lines={row['num_lines']}, Complexity={row['complexity_score']}, Params={row['num_params']}, Depth={row['max_nesting_depth']}")
            
            while True:
                choice = input("Is this label CORRECT? (y/n): ").lower()
                if choice in ['y', 'n']:
                    break
            
            if choice == 'y':
                results[smell]['correct'] += 1
            else:
                results[smell]['incorrect'] += 1
                
    print("\n" + "="*60)
    print("VERIFICATION RESULTS")
    print("="*60)
    
    total_correct = 0
    total_samples = 0
    
    for smell, stats in results.items():
        n = stats['correct'] + stats['incorrect']
        if n > 0:
            acc = (stats['correct'] / n) * 100
            print(f"{smell}: {stats['correct']}/{n} ({acc:.1f}%)")
            total_correct += stats['correct']
            total_samples += n
            
    if total_samples > 0:
        overall_acc = (total_correct / total_samples) * 100
        print("-" * 60)
        print(f"OVERALL ACCURACY: {overall_acc:.1f}%")
        
if __name__ == "__main__":
    manual_verification()
