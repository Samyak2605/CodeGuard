import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple

class DataLoader:
    def __init__(self):
        self.feature_cols = None
        self.label_cols = [
            'has_long_method',
            'has_high_complexity', 
            'has_too_many_params',
            'has_deep_nesting',
            'has_no_docstring'
        ]
        self.exclude_cols = [
            'code', 'repo_name', 'file_path', 'function_name',
            'quality_score', 'num_issues',
            # Exclude direct label proxies to avoid data leakage/trivial learning
            'complexity_score', 'num_params', 'max_nesting_depth',
            'has_docstring', 'docstring_length', 'docstring_to_code_ratio'
        ]
    
    def load_data(self) -> Tuple:
        """Load train, val, test sets"""
        print("Loading datasets...")
        
        train_df = pd.read_csv('data/processed/train.csv')
        val_df = pd.read_csv('data/processed/val.csv')
        test_df = pd.read_csv('data/processed/test.csv')
        
        # Identify feature columns (everything except labels and metadata)
        all_cols = set(train_df.columns)
        exclude = set(self.label_cols + self.exclude_cols)
        # Also exclude any other non-feature columns if present (e.g. 'has_issues' from stratification)
        exclude.add('has_issues')
        
        self.feature_cols = sorted(list(all_cols - exclude))
        
        print(f"✅ Loaded datasets:")
        print(f"   Train: {len(train_df)} samples")
        print(f"   Val: {len(val_df)} samples")
        print(f"   Test: {len(test_df)} samples")
        print(f"   Features: {len(self.feature_cols)}")
        print(f"   Labels: {len(self.label_cols)}")
        
        return train_df, val_df, test_df
    
    def prepare_data_for_smell(self, train_df, val_df, test_df, smell: str):
        """Prepare X, y for specific code smell"""
        
        # Features
        X_train = train_df[self.feature_cols].values
        X_val = val_df[self.feature_cols].values
        X_test = test_df[self.feature_cols].values
        
        # Labels
        y_train = train_df[smell].values
        y_val = val_df[smell].values
        y_test = test_df[smell].values
        
        # Check class distribution
        train_pos = y_train.sum()
        train_neg = len(y_train) - train_pos
        imbalance_ratio = max(train_pos, train_neg) / min(train_pos, train_neg) if min(train_pos, train_neg) > 0 else float('inf')
        
        print(f"\n{smell}:")
        print(f"  Train - Positive: {train_pos} ({train_pos/len(y_train)*100:.1f}%)")
        print(f"  Train - Negative: {train_neg} ({train_neg/len(y_train)*100:.1f}%)")
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_feature_names(self):
        """Return list of feature names"""
        return self.feature_cols

if __name__ == "__main__":
    loader = DataLoader()
    loader.load_data()
