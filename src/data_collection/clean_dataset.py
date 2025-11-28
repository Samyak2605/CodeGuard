import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def clean_dataset(input_csv="data/processed/labeled_functions.csv", output_csv="data/processed/labeled_functions_clean.csv"):
    logger.info("Cleaning dataset...")
    df = pd.read_csv(input_csv)
    initial_count = len(df)
    
    # 1. Remove parse errors
    df_clean = df[df['parse_error'] == False].copy()
    parse_errors = initial_count - len(df_clean)
    
    # 2. Remove duplicates (based on code content)
    df_clean = df_clean.drop_duplicates(subset=['code'])
    duplicates = initial_count - parse_errors - len(df_clean)
    
    # 3. Remove outliers (too short or too long)
    # Note: 'num_lines' might be missing if parse_error was True, but we filtered those.
    df_final = df_clean[(df_clean['num_lines'] >= 3) & (df_clean['num_lines'] <= 300)]
    outliers = len(df_clean) - len(df_final)
    
    logger.info(f"Original functions:     {initial_count}")
    logger.info(f"Parse errors removed:   -{parse_errors} ({parse_errors/initial_count*100:.1f}%)")
    logger.info(f"Duplicates removed:     -{duplicates}")
    logger.info(f"Outliers removed:       -{outliers}")
    logger.info("-" * 30)
    logger.info(f"Final clean dataset:    {len(df_final)} functions")
    
    df_final.to_csv(output_csv, index=False)
    logger.info(f"Saved to {output_csv}")

if __name__ == "__main__":
    clean_dataset()
