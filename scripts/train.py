import argparse

def main():
    parser = argparse.ArgumentParser(description="Train CodeGuard models")
    parser.add_argument("--model", type=str, default="xgboost", choices=["xgboost", "codebert"])
    args = parser.parse_args()
    
    print(f"Starting training for model: {args.model}")
    
    # TODO: Load data
    # TODO: Initialize model
    # TODO: Run training loop
    
if __name__ == "__main__":
    main()
