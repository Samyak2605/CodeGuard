import mlflow
import os

class MLflowConfig:
    def __init__(self):
        self.tracking_uri = "./mlruns"
        self.experiment_name = "codeguard-baseline"
        self.artifact_location = "./mlruns/artifacts"
    
    def setup_mlflow(self):
        """Initialize MLflow tracking"""
        # Set tracking URI (local directory)
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            mlflow.create_experiment(
                name=self.experiment_name
                # Let MLflow decide artifact location (usually inside mlruns/exp_id)
            )
        
        mlflow.set_experiment(self.experiment_name)
        print(f"✅ MLflow experiment '{self.experiment_name}' ready")
        print(f"📊 View dashboard: mlflow ui --port 5000")
    
    def start_run(self, run_name: str):
        """Start MLflow run with context manager"""
        return mlflow.start_run(run_name=run_name)

if __name__ == "__main__":
    config = MLflowConfig()
    config.setup_mlflow()
