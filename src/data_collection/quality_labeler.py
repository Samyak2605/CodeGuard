import ast
import pandas as pd
import radon.complexity as radon_cc
from tqdm import tqdm
import logging
import os
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("labeling.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CodeQualityLabeler:
    def __init__(self):
        # Define thresholds
        self.THRESHOLDS = {
            'long_method': 50,
            'high_complexity': 10,
            'too_many_params': 5,
            'deep_nesting': 4
        }
        
    def _calculate_nesting_depth(self, node, depth=0):
        """Recursively calculate maximum nesting level."""
        max_depth = depth
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.AsyncFor, ast.AsyncWith, ast.Try)):
                max_depth = max(max_depth, self._calculate_nesting_depth(child, depth + 1))
            else:
                max_depth = max(max_depth, self._calculate_nesting_depth(child, depth))
        return max_depth

    def _calculate_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity using radon."""
        try:
            return radon_cc.cc_visit(code)[0].complexity
        except Exception:
            return 0

    def label_function(self, code: str) -> dict:
        """
        Analyze a single function and return labels.
        """
        result = {
            'has_long_method': False,
            'has_high_complexity': False,
            'has_too_many_params': False,
            'has_deep_nesting': False,
            'has_no_docstring': False,
            'quality_score': 100,
            'num_issues': 0,
            'complexity_score': 0,
            'num_params': 0,
            'max_nesting_depth': 0,
            'num_lines': 0,
            'parse_error': False
        }
        
        try:
            # Parse code
            tree = ast.parse(code)
            func_def = tree.body[0]
            
            if not isinstance(func_def, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result['parse_error'] = True
                return result

            # 1. Long Method
            num_lines = len(code.splitlines())
            result['num_lines'] = num_lines
            if num_lines > self.THRESHOLDS['long_method']:
                result['has_long_method'] = True
                result['quality_score'] -= 20
                result['num_issues'] += 1

            # 2. High Complexity
            complexity = self._calculate_complexity(code)
            result['complexity_score'] = complexity
            if complexity > self.THRESHOLDS['high_complexity']:
                result['has_high_complexity'] = True
                result['quality_score'] -= 25
                result['num_issues'] += 1

            # 3. Too Many Parameters
            num_params = len(func_def.args.args)
            result['num_params'] = num_params
            if num_params > self.THRESHOLDS['too_many_params']:
                result['has_too_many_params'] = True
                result['quality_score'] -= 20
                result['num_issues'] += 1

            # 4. Deep Nesting
            max_depth = self._calculate_nesting_depth(func_def)
            result['max_nesting_depth'] = max_depth
            if max_depth > self.THRESHOLDS['deep_nesting']:
                result['has_deep_nesting'] = True
                result['quality_score'] -= 20
                result['num_issues'] += 1

            # 5. No Docstring
            if not ast.get_docstring(func_def):
                result['has_no_docstring'] = True
                result['quality_score'] -= 15
                result['num_issues'] += 1
                
            # Ensure score is within 0-100
            result['quality_score'] = max(0, result['quality_score'])

        except Exception as e:
            logger.debug(f"Error analyzing function: {e}")
            result['parse_error'] = True
            
        return result

    def label_dataset(self, input_csv: str, output_csv: str):
        """
        Label entire dataset with progress tracking.
        """
        logger.info(f"Loading dataset from {input_csv}...")
        try:
            df = pd.read_csv(input_csv)
        except FileNotFoundError:
            logger.error(f"Input file {input_csv} not found.")
            return

        logger.info(f"Labeling {len(df)} functions...")
        
        results = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Labeling"):
            labels = self.label_function(row['code'])
            # Merge original data with labels
            results.append({**row.to_dict(), **labels})
            
            # Save incrementally every 1000 items
            if (idx + 1) % 1000 == 0:
                pd.DataFrame(results).to_csv(output_csv, index=False)
                
        # Final save
        pd.DataFrame(results).to_csv(output_csv, index=False)
        logger.info(f"Labeling complete. Saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code Quality Labeler")
    parser.add_argument("--input", default="data/raw/github_functions.csv", help="Input CSV file")
    parser.add_argument("--output", default="data/processed/labeled_functions.csv", help="Output CSV file")
    args = parser.parse_args()
    
    labeler = CodeQualityLabeler()
    labeler.label_dataset(args.input, args.output)
