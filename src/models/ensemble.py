from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
import mlflow
import joblib
import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

class EnsembleModel:
    def __init__(self, smell_name: str):
        self.smell_name = smell_name
        self.ensemble_model = None
        self.ensemble_type = None
    
    def create_voting_ensemble(self, xgb_params, rf_params):
        """
        Create voting ensemble of XGBoost + Random Forest
        """
        
        xgb_model = xgb.XGBClassifier(**xgb_params, random_state=42)
        rf_model = RandomForestClassifier(**rf_params, random_state=42)
        
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('rf', rf_model)
            ],
            voting='soft',  # Use probability averaging
            n_jobs=-1
        )
        self.ensemble_type = 'voting'
        return self.ensemble_model
    
    def create_stacking_ensemble(self, xgb_params, rf_params):
        """
        Create stacking ensemble with LogisticRegression as meta-learner
        """
        
        xgb_model = xgb.XGBClassifier(**xgb_params, random_state=42)
        rf_model = RandomForestClassifier(**rf_params, random_state=42)
        
        self.ensemble_model = StackingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('rf', rf_model)
            ],
            final_estimator=LogisticRegression(random_state=42),
            cv=3,  # 3-fold CV for meta-features
            n_jobs=-1
        )
        self.ensemble_type = 'stacking'
        return self.ensemble_model
    
    def train(self, X_train, y_train, X_val, y_val, ensemble_type='voting'):
        """Train ensemble model"""
        
        print(f"\n{'='*60}")
        print(f"Training {ensemble_type.upper()} Ensemble for: {self.smell_name}")
        print(f"{'='*60}")
        
        # Default parameters (could be tuned)
        xgb_params = {
            'n_estimators': 200,
            'max_depth': 10,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        
        rf_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5
        }
        
        # Create ensemble
        if ensemble_type == 'voting':
            self.create_voting_ensemble(xgb_params, rf_params)
        elif ensemble_type == 'stacking':
            self.create_stacking_ensemble(xgb_params, rf_params)
        
        # MLflow tracking
        with mlflow.start_run(run_name=f"{ensemble_type}_{self.smell_name}"):
            mlflow.log_param("ensemble_type", ensemble_type)
            mlflow.log_param("smell_type", self.smell_name)
            
            # Train
            self.ensemble_model.fit(X_train, y_train)
            
            # Predictions
            y_val_pred = self.ensemble_model.predict(X_val)
            
            # Metrics
            metrics = {
                'val_accuracy': accuracy_score(y_val, y_val_pred),
                'val_f1': f1_score(y_val, y_val_pred, zero_division=0)
            }
            
            mlflow.log_metrics(metrics)
            
            print(f"\n{ensemble_type.upper()} Ensemble Results:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
            
            return metrics

    def predict(self, X):
        """Make predictions"""
        return self.ensemble_model.predict(X)
    
    def save_model(self, path):
        """Save ensemble model"""
        joblib.dump(self.ensemble_model, path)

# Train all ensembles
def train_all_ensembles():
    """Train ensemble models for all 5 smells"""
    
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
        
        # Try both ensemble types
        for ensemble_type in ['voting', 'stacking']:
            model = EnsembleModel(smell_name=smell)
            metrics = model.train(X_train, y_train, X_val, y_val, ensemble_type=ensemble_type)
            
            # Test set
            y_test_pred = model.predict(X_test)
            test_f1 = f1_score(y_test, y_test_pred)
            
            results.append({
                'smell': smell,
                'ensemble_type': ensemble_type,
                'val_f1': metrics['val_f1'],
                'test_f1': test_f1
            })
            
            # Save best ensemble (Voting usually better/simpler)
            if ensemble_type == 'voting':
                os.makedirs('models/ensemble', exist_ok=True)
                model.save_model(f'models/ensemble/{smell}_voting.pkl')
    
    results_df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/ensemble_results.csv', index=False)
    
    print("\n\nENSEMBLE RESULTS:")
    print(results_df)
    
    # Compare averages
    voting_avg = results_df[results_df['ensemble_type'] == 'voting']['val_f1'].mean()
    stacking_avg = results_df[results_df['ensemble_type'] == 'stacking']['val_f1'].mean()
    
    print(f"\nAVERAGE PERFORMANCE:")
    print(f"  Voting Ensemble: {voting_avg:.4f}")
    print(f"  Stacking Ensemble: {stacking_avg:.4f}")
    
    return results_df

if __name__ == "__main__":
    results = train_all_ensembles()
