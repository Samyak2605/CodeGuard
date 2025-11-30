import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.data_loader import DataLoader
from src.models.xgboost_baseline import XGBoostBaseline
from configs.mlflow_config import MLflowConfig
import pandas as pd
import time
import mlflow

def train_all_baselines():
    """Train XGBoost baseline for all 5 code smells"""
    
    # Setup MLflow
    mlflow_config = MLflowConfig()
    mlflow_config.setup_mlflow()
    
    # Load data
    loader = DataLoader()
    train_df, val_df, test_df = loader.load_data()
    feature_names = loader.get_feature_names()
    
    # Code smells to train
    code_smells = [
        'has_long_method',
        'has_high_complexity',
        'has_too_many_params',
        'has_deep_nesting',
        'has_no_docstring'
    ]
    
    # Store all results
    results = []
    
    print(f"\n{'='*70}")
    print(f"TRAINING BASELINE MODELS FOR ALL CODE SMELLS")
    print(f"{'='*70}\n")
    
    # Train model for each smell
    for smell in code_smells:
        print(f"\n{'='*70}")
        print(f"Training model {len(results)+1}/5: {smell}")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        # Prepare data for this smell
        X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare_data_for_smell(
            train_df, val_df, test_df, smell
        )
        
        # Create and train model
        model = XGBoostBaseline(smell_name=smell)
        model.set_feature_names(feature_names)
        
        metrics = model.train(X_train, y_train, X_val, y_val)
        
        # Test set evaluation
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)
        test_metrics = model._calculate_metrics(y_test, y_test_pred, y_test_proba, "test")
        
        # Save model
        model.save_model(f'models/baselines/xgboost_{smell}.pkl')
        
        training_time = time.time() - start_time
        
        # Store results
        results.append({
            'smell': smell,
            'val_accuracy': metrics['val_accuracy'],
            'val_precision': metrics['val_precision'],
            'val_recall': metrics['val_recall'],
            'val_f1': metrics['val_f1'],
            'val_roc_auc': metrics['val_roc_auc'],
            'test_accuracy': test_metrics['test_accuracy'],
            'test_f1': test_metrics['test_f1'],
            'training_time_seconds': training_time
        })
        
        print(f"\n✅ Completed in {training_time:.2f} seconds")
    
    # Create summary report
    print(f"\n\n{'='*70}")
    print(f"BASELINE TRAINING COMPLETE - SUMMARY")
    print(f"{'='*70}\n")
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/baseline_results.csv', index=False)
    print(f"\n✅ Results saved to results/baseline_results.csv")
    
    # Calculate averages
    print(f"\n{'='*70}")
    print(f"AVERAGE METRICS ACROSS ALL CODE SMELLS")
    print(f"{'='*70}")
    print(f"Validation Accuracy: {results_df['val_accuracy'].mean():.4f}")
    print(f"Validation F1 Score: {results_df['val_f1'].mean():.4f}")
    print(f"Validation ROC-AUC: {results_df['val_roc_auc'].mean():.4f}")
    print(f"Test Accuracy: {results_df['test_accuracy'].mean():.4f}")
    print(f"Test F1 Score: {results_df['test_f1'].mean():.4f}")
    print(f"Total Training Time: {results_df['training_time_seconds'].sum():.2f} seconds")
    
    print(f"\n🎉 All baseline models trained successfully!")
    print(f"📊 View experiments in MLflow UI: http://localhost:5000")
    
    return results_df

if __name__ == "__main__":
    results = train_all_baselines()
