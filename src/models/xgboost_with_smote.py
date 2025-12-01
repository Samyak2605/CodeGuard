import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.xgboost_baseline import XGBoostBaseline
from src.models.smote_handler import SMOTEHandler
from src.models.data_loader import DataLoader
import mlflow
import pandas as pd

class XGBoostWithSMOTE(XGBoostBaseline):
    def __init__(self, smell_name: str, use_smote=True):
        super().__init__(smell_name)
        self.use_smote = use_smote
        self.smote_handler = SMOTEHandler()
    
    def train(self, X_train, y_train, X_val, y_val, params=None):
        """Train with optional SMOTE preprocessing"""
        
        print(f"\n{'='*60}")
        print(f"Training XGBoost with SMOTE for: {self.smell_name}")
        print(f"{'='*60}")
        
        # Apply SMOTE if enabled
        if self.use_smote:
            # Check imbalance level
            needs_smote, ratio = self.smote_handler.analyze_imbalance(y_train)
            
            if ratio > 3:
                # Severe imbalance: use SMOTE + undersampling
                X_train, y_train = self.smote_handler.apply_smote_with_undersampling(
                    X_train, y_train
                )
            elif ratio > 1.5:
                # Moderate imbalance: use standard SMOTE
                X_train, y_train = self.smote_handler.apply_smote(
                    X_train, y_train, sampling_strategy='auto'
                )
            else:
                print("✅ Data is balanced. Skipping SMOTE.")
        
        # Call parent train method
        return super().train(X_train, y_train, X_val, y_val, params)

# Train all models with SMOTE
def train_all_with_smote():
    """Train all 5 models with SMOTE"""
    
    from configs.mlflow_config import MLflowConfig
    
    # Setup
    mlflow_config = MLflowConfig()
    mlflow_config.setup_mlflow()
    
    loader = DataLoader()
    train_df, val_df, test_df = loader.load_data()
    feature_names = loader.get_feature_names()
    
    code_smells = [
        'has_long_method',
        'has_high_complexity',
        'has_too_many_params',
        'has_deep_nesting',
        'has_no_docstring'
    ]
    
    results = []
    
    for smell in code_smells:
        print(f"\n{'='*70}")
        print(f"Training with SMOTE: {smell}")
        print(f"{'='*70}")
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare_data_for_smell(
            train_df, val_df, test_df, smell
        )
        
        # Train with SMOTE
        model = XGBoostWithSMOTE(smell_name=smell, use_smote=True)
        model.set_feature_names(feature_names)
        
        metrics = model.train(X_train, y_train, X_val, y_val)
        
        # Test set evaluation
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)
        test_metrics = model._calculate_metrics(y_test, y_test_pred, y_test_proba, "test")
        
        results.append({
            'smell': smell,
            'method': 'XGBoost + SMOTE',
            'val_f1': metrics['val_f1'],
            'test_f1': test_metrics['test_f1']
        })
        
        # Save model
        model.save_model(f'models/baselines/xgboost_smote_{smell}.pkl')
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    results = train_all_with_smote()
    print("\n\nRESULTS WITH SMOTE:")
    print(results)
    os.makedirs('results', exist_ok=True)
    results.to_csv('results/smote_results.csv', index=False)
