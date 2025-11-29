import ast
import numpy as np
import pandas as pd
import radon.complexity as radon_cc
from radon.metrics import mi_visit, h_visit
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self):
        self.feature_names = [
            # Basic
            'num_lines', 'num_chars', 'num_tokens', 'avg_line_length', 'blank_lines',
            # Complexity
            'cyclomatic_complexity', 'num_if_statements', 'num_for_loops', 
            'num_while_loops', 'num_try_except', 'max_nesting_depth',
            # Function
            'num_parameters', 'has_return', 'has_docstring', 'docstring_length',
            'num_decorators', 'has_type_hints', 'num_default_args', 'has_varargs',
            # Structure
            'num_function_calls', 'num_assignments', 'num_comparisons',
            'num_boolean_ops', 'num_list_comps', 'num_lambda',
            # Naming
            'num_variables', 'avg_var_name_length', 'snake_case_ratio',
            'single_char_vars', 'uppercase_vars',
            # Comments
            'num_comments', 'comment_ratio', 'docstring_to_code_ratio',
            # Dependency
            'num_imports', 'num_from_imports', 'has_wildcard_import',
            # Advanced
            'halstead_volume', 'maintainability_index'
        ]

    def extract_basic_metrics(self, code: str) -> dict:
        lines = code.split('\n')
        return {
            'num_lines': len(lines),
            'num_chars': len(code),
            'num_tokens': len(code.split()),
            'avg_line_length': np.mean([len(line) for line in lines]) if lines else 0,
            'blank_lines': code.count('\n\n')
        }

    def extract_complexity_metrics(self, code: str, tree: ast.AST) -> dict:
        try:
            cc = radon_cc.cc_visit(code)[0].complexity
        except:
            cc = 0
            
        return {
            'cyclomatic_complexity': cc,
            'num_if_statements': len([n for n in ast.walk(tree) if isinstance(n, ast.If)]),
            'num_for_loops': len([n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.AsyncFor))]),
            'num_while_loops': len([n for n in ast.walk(tree) if isinstance(n, ast.While)]),
            'num_try_except': len([n for n in ast.walk(tree) if isinstance(n, ast.Try)]),
            'max_nesting_depth': self._calculate_nesting_depth(tree)
        }

    def _calculate_nesting_depth(self, node, depth=0):
        max_depth = depth
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.AsyncFor, ast.AsyncWith, ast.Try)):
                max_depth = max(max_depth, self._calculate_nesting_depth(child, depth + 1))
            else:
                max_depth = max(max_depth, self._calculate_nesting_depth(child, depth))
        return max_depth

    def extract_function_features(self, code: str, tree: ast.AST) -> dict:
        try:
            func = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))][0]
            docstring = ast.get_docstring(func)
            return {
                'num_parameters': len(func.args.args),
                'has_return': any(isinstance(n, ast.Return) for n in ast.walk(func)),
                'has_docstring': docstring is not None,
                'docstring_length': len(docstring) if docstring else 0,
                'num_decorators': len(func.decorator_list),
                'has_type_hints': any(a.annotation is not None for a in func.args.args) or func.returns is not None,
                'num_default_args': len(func.args.defaults),
                'has_varargs': func.args.vararg is not None or func.args.kwarg is not None
            }
        except IndexError:
            return {k: 0 for k in ['num_parameters', 'docstring_length', 'num_decorators', 'num_default_args']} | \
                   {k: False for k in ['has_return', 'has_docstring', 'has_type_hints', 'has_varargs']}

    def extract_structure_features(self, tree: ast.AST) -> dict:
        return {
            'num_function_calls': len([n for n in ast.walk(tree) if isinstance(n, ast.Call)]),
            'num_assignments': len([n for n in ast.walk(tree) if isinstance(n, ast.Assign)]),
            'num_comparisons': len([n for n in ast.walk(tree) if isinstance(n, ast.Compare)]),
            'num_boolean_ops': len([n for n in ast.walk(tree) if isinstance(n, ast.BoolOp)]),
            'num_list_comps': len([n for n in ast.walk(tree) if isinstance(n, ast.ListComp)]),
            'num_lambda': len([n for n in ast.walk(tree) if isinstance(n, ast.Lambda)])
        }

    def extract_naming_features(self, tree: ast.AST) -> dict:
        var_names = [n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)]
        unique_vars = set(var_names)
        total_vars = len(unique_vars)
        
        return {
            'num_variables': total_vars,
            'avg_var_name_length': np.mean([len(v) for v in unique_vars]) if total_vars else 0,
            'snake_case_ratio': sum(1 for v in unique_vars if '_' in v) / total_vars if total_vars else 0,
            'single_char_vars': sum(1 for v in unique_vars if len(v) == 1) / total_vars if total_vars else 0,
            'uppercase_vars': sum(1 for v in unique_vars if v.isupper()) / total_vars if total_vars else 0
        }

    def extract_comment_features(self, code: str) -> dict:
        lines = code.split('\n')
        num_comments = sum(1 for line in lines if line.strip().startswith('#'))
        return {
            'num_comments': num_comments,
            'comment_ratio': num_comments / len(lines) if lines else 0,
            'docstring_to_code_ratio': 0 # Placeholder, hard to calc accurately without more parsing
        }

    def extract_dependency_features(self, tree: ast.AST) -> dict:
        return {
            'num_imports': len([n for n in ast.walk(tree) if isinstance(n, ast.Import)]),
            'num_from_imports': len([n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]),
            'has_wildcard_import': any(n.names[0].name == '*' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom))
        }

    def extract_advanced_features(self, code: str) -> dict:
        try:
            mi = mi_visit(code, multi=False)
            h = h_visit(code)
            return {
                'halstead_volume': h.volume,
                'maintainability_index': mi
            }
        except:
            return {'halstead_volume': 0, 'maintainability_index': 0}

    def extract_all_features(self, code: str) -> dict:
        features = {}
        try:
            tree = ast.parse(code)
            features.update(self.extract_basic_metrics(code))
            features.update(self.extract_complexity_metrics(code, tree))
            features.update(self.extract_function_features(code, tree))
            features.update(self.extract_structure_features(tree))
            features.update(self.extract_naming_features(tree))
            features.update(self.extract_comment_features(code))
            features.update(self.extract_dependency_features(tree))
            features.update(self.extract_advanced_features(code))
        except Exception as e:
            # logger.error(f"Extraction error: {e}")
            features = {k: np.nan for k in self.feature_names}
            features['extraction_error'] = True
            
        return features

    def extract_features_from_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Extracting features...")
        feature_list = []
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            features = self.extract_all_features(row['code'])
            feature_list.append(features)
        
        features_df = pd.DataFrame(feature_list)
        result = pd.concat([df, features_df], axis=1)
        
        # Remove rows with extraction errors
        if 'extraction_error' in result.columns:
            error_count = result['extraction_error'].sum()
            logger.info(f"Removing {error_count} samples with extraction errors.")
            result = result[result['extraction_error'] != True].drop(columns=['extraction_error'])
            
        return result

if __name__ == "__main__":
    df = pd.read_csv('data/processed/labeled_functions_clean.csv')
    extractor = FeatureExtractor()
    df_features = extractor.extract_features_from_dataset(df)
    print(f"Features extracted. New shape: {df_features.shape}")
    # Temporary save to check
    # df_features.to_csv('data/processed/temp_features.csv', index=False)
