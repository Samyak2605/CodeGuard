def extract_features(code_snippet):
    """
    Extracts static features from a code snippet.
    
    Args:
        code_snippet (str): The source code to analyze.
        
    Returns:
        dict: A dictionary of extracted features.
    """
    features = {
        "loc": 0,
        "cyclomatic_complexity": 0,
        "num_params": 0,
        "nesting_depth": 0
    }
    
    # TODO: Implement feature extraction logic using 'ast' and 'radon'
    
    return features
