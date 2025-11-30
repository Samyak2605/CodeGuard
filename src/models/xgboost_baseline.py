import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import mlflow
import mlflow.xgboost
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

class XGBoostBaseline:
    def __init__(self, smell_name: str):
        self.smell_name = smell_name
        self.model = None
        self.best_params = None
        self.feature_names = None
    
    def train(self, X_train, y_train, X_val, y_val, params=None):
        """
        Train XGBoost classifier with default or custom parameters
        """
        
        # Default parameters
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'random_state': 42,
                'n_jobs': -1
            }
        
        print(f"\n{'='*60}")
        print(f"Training XGBoost for: {self.smell_name}")
        print(f"{'='*60}")
        print(f"Parameters: {params}")
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"xgboost_{self.smell_name}_baseline"):
            
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("smell_type", self.smell_name)
            mlflow.log_param("model_type", "xgboost")
            
            # Create and train model
            self.model = xgb.XGBClassifier(
                **params,
                early_stopping_rounds=10
            )
            
            # Use early stopping
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Get predictions
            y_train_pred = self.model.predict(X_train)
            y_val_pred = self.model.predict(X_val)
            
            y_train_proba = self.model.predict_proba(X_train)[:, 1]
            y_val_proba = self.model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            train_metrics = self._calculate_metrics(y_train, y_train_pred, y_train_proba, "train")
            val_metrics = self._calculate_metrics(y_val, y_val_pred, y_val_proba, "val")
            
            # Log metrics to MLflow
            mlflow.log_metrics(train_metrics)
            mlflow.log_metrics(val_metrics)
            
            # Log model
            mlflow.xgboost.log_model(self.model, "model")
            
            # Create and log visualizations
            self._log_visualizations(y_val, y_val_pred, y_val_proba)
            
            # Print results
            print(f"\n{'='*60}")
            print(f"RESULTS FOR {self.smell_name}")
            print(f"{'='*60}")
            print(f"\nTraining Set:")
            for metric, value in train_metrics.items():
                print(f"  {metric}: {value:.4f}")
            print(f"\nValidation Set:")
            for metric, value in val_metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            # Store best params
            self.best_params = params
            
            return val_metrics
    
    def _calculate_metrics(self, y_true, y_pred, y_proba, prefix):
        """Calculate all evaluation metrics"""
        
        metrics = {
            f'{prefix}_accuracy': accuracy_score(y_true, y_pred),
            f'{prefix}_precision': precision_score(y_true, y_pred, zero_division=0),
            f'{prefix}_recall': recall_score(y_true, y_pred, zero_division=0),
            f'{prefix}_f1': f1_score(y_true, y_pred, zero_division=0),
            f'{prefix}_roc_auc': roc_auc_score(y_true, y_proba)
        }
        
        return metrics
    
    def _log_visualizations(self, y_true, y_pred, y_proba):
        """Create and log visualization plots"""
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {self.smell_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        filename = f'temp_confusion_matrix_{self.smell_name}.png'
        plt.savefig(filename)
        mlflow.log_artifact(filename)
        plt.close()
        os.remove(filename)
        
        # 2. Feature Importance (top 15)
        if self.feature_names:
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance_df, x='importance', y='feature')
            plt.title(f'Top 15 Feature Importances - {self.smell_name}')
            plt.tight_layout()
            filename = f'temp_feature_importance_{self.smell_name}.png'
            plt.savefig(filename)
            mlflow.log_artifact(filename)
            plt.close()
            os.remove(filename)
        
        # 3. ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_true, y_proba):.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.smell_name}')
        plt.legend()
        plt.tight_layout()
        filename = f'temp_roc_curve_{self.smell_name}.png'
        plt.savefig(filename)
        mlflow.log_artifact(filename)
        plt.close()
        os.remove(filename)
    
    def set_feature_names(self, feature_names):
        """Store feature names for visualization"""
        self.feature_names = feature_names
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)[:, 1]
    
    def save_model(self, path):
        """Save model to disk"""
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"✅ Model saved to {path}")
