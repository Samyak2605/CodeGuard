from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
import xgboost as xgb
import mlflow
import numpy as np
import pandas as pd
import time
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

class HyperparameterTuner:
    def __init__(self, smell_name: str):
        self.smell_name = smell_name
        self.best_params = None
        self.best_score = None
        self.cv_results = None
    
    def tune_xgboost(self, X_train, y_train, quick_search=False):
        """
        Perform grid search for XGBoost hyperparameters
        """
        
        print(f"\n{'='*60}")
        print(f"HYPERPARAMETER TUNING: {self.smell_name}")
        print(f"{'='*60}")
        
        # Define parameter grid
        if quick_search:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [6, 10],
                'learning_rate': [0.1, 0.3],
                'subsample': [0.8],
                'colsample_bytree': [0.8],
                'min_child_weight': [1, 3]
            }
            print("Using QUICK search (smaller grid)")
        else:
            param_grid = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 6, 10, 15],
                'learning_rate': [0.01, 0.05, 0.1, 0.3],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'min_child_weight': [1, 3, 5],
                'gamma': [0, 0.1, 0.2]
            }
            print("Using FULL search (larger grid)")
        
        # Calculate total combinations
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        print(f"\nTotal parameter combinations: {total_combinations}")
        print(f"With 3-fold CV: {total_combinations * 3} model fits")
        
        # Create base model
        base_model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=42,
            n_jobs=-1
        )
        
        # Create F1 scorer
        f1_scorer = make_scorer(f1_score)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring=f1_scorer,
            cv=3,  # 3-fold cross-validation
            n_jobs=-1,  # Use all CPU cores
            verbose=1,
            return_train_score=True
        )
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"tuning_{self.smell_name}"):
            
            # Log that we're tuning
            mlflow.log_param("tuning_method", "GridSearchCV")
            mlflow.log_param("cv_folds", 3)
            mlflow.log_param("total_combinations", total_combinations)
            
            # Perform grid search
            start_time = time.time()
            grid_search.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Get best parameters and score
            self.best_params = grid_search.best_params_
            self.best_score = grid_search.best_score_
            self.cv_results = pd.DataFrame(grid_search.cv_results_)
            
            # Log best parameters
            mlflow.log_params(self.best_params)
            mlflow.log_metric("best_cv_f1", self.best_score)
            mlflow.log_metric("tuning_time_seconds", training_time)
            
            # Print results
            print(f"\n{'='*60}")
            print(f"TUNING COMPLETE: {self.smell_name}")
            print(f"{'='*60}")
            print(f"\nBest Parameters:")
            for param, value in self.best_params.items():
                print(f"  {param}: {value}")
            print(f"\nBest CV F1 Score: {self.best_score:.4f}")
            print(f"Training Time: {training_time:.2f} seconds")
            
            # Save results
            os.makedirs('results', exist_ok=True)
            self.cv_results.to_csv(f'results/tuning_{self.smell_name}.csv', index=False)
        
        return self.best_params, self.best_score

# Run hyperparameter tuning for all smells
def tune_all_models(quick_search=True):
    """Tune hyperparameters for all 5 code smells"""
    
    from src.models.data_loader import DataLoader
    from src.models.smote_handler import SMOTEHandler
    from configs.mlflow_config import MLflowConfig
    import joblib
    
    # Setup
    mlflow_config = MLflowConfig()
    mlflow_config.setup_mlflow()
    
    loader = DataLoader()
    train_df, val_df, test_df = loader.load_data()
    
    smote_handler = SMOTEHandler()
    
    code_smells = [
        'has_long_method',
        'has_high_complexity',
        'has_too_many_params',
        'has_deep_nesting',
        'has_no_docstring'
    ]
    
    tuning_results = []
    
    for smell in code_smells:
        print(f"\n{'='*70}")
        print(f"TUNING MODEL {len(tuning_results)+1}/5: {smell}")
        print(f"{'='*70}")
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare_data_for_smell(
            train_df, val_df, test_df, smell
        )
        
        # Apply SMOTE if needed
        needs_smote, ratio = smote_handler.analyze_imbalance(y_train)
        if ratio > 1.5:
            X_train, y_train = smote_handler.apply_smote(X_train, y_train)
        
        # Tune hyperparameters
        tuner = HyperparameterTuner(smell_name=smell)
        best_params, best_score = tuner.tune_xgboost(X_train, y_train, quick_search=quick_search)
        
        # Train final model with best params
        final_model = xgb.XGBClassifier(**best_params, random_state=42)
        final_model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_val_pred = final_model.predict(X_val)
        val_f1 = f1_score(y_val, y_val_pred)
        
        # Evaluate on test set
        y_test_pred = final_model.predict(X_test)
        test_f1 = f1_score(y_test, y_test_pred)
        
        tuning_results.append({
            'smell': smell,
            'best_cv_f1': best_score,
            'val_f1': val_f1,
            'test_f1': test_f1,
            'best_params': str(best_params)
        })
        
        # Save best model
        os.makedirs('models/tuned', exist_ok=True)
        joblib.dump(final_model, f'models/tuned/xgboost_{smell}_tuned.pkl')
        
        print(f"\n✅ Model saved: models/tuned/xgboost_{smell}_tuned.pkl")
    
    # Save all results
    results_df = pd.DataFrame(tuning_results)
    results_df.to_csv('results/hyperparameter_tuning_results.csv', index=False)
    
    print(f"\n\n{'='*70}")
    print(f"HYPERPARAMETER TUNING COMPLETE - SUMMARY")
    print(f"{'='*70}\n")
    print(results_df[['smell', 'best_cv_f1', 'val_f1', 'test_f1']])
    
    return results_df

if __name__ == "__main__":
    # Use quick_search=True for faster testing (15-30 mins)
    results = tune_all_models(quick_search=True)
