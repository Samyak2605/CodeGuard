import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import joblib
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def handle_missing_values(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Impute missing values with median."""
    logger.info("Handling missing values...")
    imputer = SimpleImputer(strategy='median')
    df[feature_cols] = imputer.fit_transform(df[feature_cols])
    return df

def scale_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Scale features using RobustScaler."""
    logger.info("Scaling features...")
    scaler = RobustScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Save scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/feature_scaler.pkl')
    logger.info("Scaler saved to models/feature_scaler.pkl")
    
    return df

def create_splits(df: pd.DataFrame) -> tuple:
    """Create stratified train/val/test splits (70/15/15)."""
    logger.info("Creating data splits...")
    
    # Stratify by num_issues to ensure balanced splits
    # If num_issues has rare classes (e.g. 5 issues), we might need to bin or use random split
    stratify_col = 'num_issues'
    if df[stratify_col].value_counts().min() < 2:
        logger.warning("Warning: Rare classes in num_issues. Stratifying by 'has_issues' instead.")
        # Ensure has_issues exists (it might not if we dropped it earlier, but we shouldn't have)
        if 'has_issues' not in df.columns:
             df['has_issues'] = df['num_issues'] > 0
        stratify_col = 'has_issues'
        
    # Double check if even has_issues is valid (should be, but safety first)
    if df[stratify_col].value_counts().min() < 2:
         logger.warning("Warning: Stratification not possible. Using random split.")
         stratify_col = None

    train_df, temp_df = train_test_split(
        df, 
        test_size=0.3, 
        stratify=df[stratify_col] if stratify_col else None,
        random_state=42
    )
    
    # Check stratification for the second split
    stratify_col_temp = stratify_col
    if stratify_col_temp and temp_df[stratify_col_temp].value_counts().min() < 2:
        logger.warning("Warning: Rare classes in temp split. Disabling stratification for val/test split.")
        stratify_col_temp = None

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df[stratify_col_temp] if stratify_col_temp else None,
        random_state=42
    )
    
    logger.info(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"Val: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    logger.info(f"Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    # Save splits
    os.makedirs('data/processed', exist_ok=True)
    train_df.to_csv('data/processed/train.csv', index=False)
    val_df.to_csv('data/processed/val.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.features.feature_extractor import FeatureExtractor
    
    # Load data
    df = pd.read_csv('data/processed/labeled_functions_clean.csv')
    
    # Extract features
    extractor = FeatureExtractor()
    df = extractor.extract_features_from_dataset(df)
    
    # Identify feature columns (exclude original metadata and labels)
    exclude_cols = ['code', 'repo_name', 'file_path', 'function_name', 
                    'has_long_method', 'has_high_complexity', 'has_too_many_params', 
                    'has_deep_nesting', 'has_no_docstring', 'quality_score', 
                    'num_issues', 'complexity_score', 'num_params', 'max_nesting_depth', 
                    'num_lines', 'parse_error']
    # Note: 'num_lines', 'complexity_score' etc are also features, but we might have re-extracted them.
    # FeatureExtractor adds columns. Let's use the list from extractor.
    feature_cols = extractor.feature_names
    
    # Preprocess
    df = handle_missing_values(df, feature_cols)
    df = scale_features(df, feature_cols)
    
    # Split
    create_splits(df)
