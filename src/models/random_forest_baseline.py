from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

class RandomForestBaseline:
    def __init__(self, smell_name: str):
        self.smell_name = smell_name
        self.model = None
    
    def train(self, X_train, y_train, X_val, y_val, params=None):
        """Train Random Forest classifier"""
        
        if params is None:
            params = {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'random_state': 42,
                'n_jobs': -1
            }
        
        print(f"\n{'='*60}")
        print(f"Training Random Forest for: {self.smell_name}")
        print(f"{'='*60}")
        
        with mlflow.start_run(run_name=f"rf_{self.smell_name}"):
            
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("model_type", "random_forest")
            mlflow.log_param("smell_type", self.smell_name)
            
            # Train model
            self.model = RandomForestClassifier(**params)
            self.model.fit(X_train, y_train)
            
            # Predictions
            y_val_pred = self.model.predict(X_val)
            
            # Metrics
            metrics = {
                'val_accuracy': accuracy_score(y_val, y_val_pred),
                'val_precision': precision_score(y_val, y_val_pred, zero_division=0),
                'val_recall': recall_score(y_val, y_val_pred, zero_division=0),
                'val_f1': f1_score(y_val, y_val_pred, zero_division=0)
            }
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(self.model, "model")
            
            print(f"\nResults:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
            
            return metrics
    
    def save_model(self, path):
        """Save model"""
        joblib.dump(self.model, path)

# Train all Random Forests
def train_all_random_forests():
    """Train RF for all 5 smells and compare with XGBoost"""
    
    from src.models.data_loader import DataLoader
    from src.models.smote_handler import SMOTEHandler
    from configs.mlflow_config import MLflowConfig
    
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
    
    results = []
    
    for smell in code_smells:
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare_data_for_smell(
            train_df, val_df, test_df, smell
        )
        
        # Apply SMOTE
        needs_smote, ratio = smote_handler.analyze_imbalance(y_train)
        if ratio > 1.5:
            X_train, y_train = smote_handler.apply_smote(X_train, y_train)
        
        # Train Random Forest
        model = RandomForestBaseline(smell_name=smell)
        metrics = model.train(X_train, y_train, X_val, y_val)
        
        # Test set
        y_test_pred = model.model.predict(X_test)
        test_f1 = f1_score(y_test, y_test_pred)
        
        results.append({
            'smell': smell,
            'model': 'RandomForest',
            'val_f1': metrics['val_f1'],
            'test_f1': test_f1
        })
    
    results_df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/random_forest_results.csv', index=False)
    
    print("\n\nRANDOM FOREST RESULTS:")
    print(results_df)
    print(f"\nAverage Val F1: {results_df['val_f1'].mean():.4f}")
    
    return results_df

if __name__ == "__main__":
    results = train_all_random_forests()
