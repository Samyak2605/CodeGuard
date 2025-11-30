# CodeGuard Baseline Model Report

## Executive Summary
Trained XGBoost baseline models for 5 code smell detection tasks.
Average validation F1 score: **95.33%**

## Performance by Code Smell

| Code Smell | Accuracy | Precision | Recall | F1 | ROC-AUC |
|------------|----------|-----------|--------|-----|---------|
| Long Method | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| High Complexity | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| Too Many Params | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| Deep Nesting | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| No Docstring | 79.51% | 79.61% | 73.87% | 76.64% | 88.13% |
| **Average** | **95.90%** | **95.92%** | **94.77%** | **95.33%** | **97.63%** |

## Key Findings

### Strengths
- **Perfect Detection (100%)**: Long Method, High Complexity, Too Many Params, Deep Nesting.
    - **Reason**: The engineered features (e.g., `cyclomatic_complexity`, `num_parameters`) are the exact same metrics used to generate the labels. This confirms the model can perfectly learn the static analysis rules.
- **Robustness**: Even with direct "score" columns excluded, the model found equivalent features (e.g., `num_tokens` for `num_lines`).

### Challenges
- **No Docstring (76.64% F1)**:
    - This was the only non-trivial task because we explicitly excluded `has_docstring` and `docstring_length` to prevent data leakage.
    - The model had to infer "missing docstring" from other structural features, which is much harder.
    - **Implication**: This is the most realistic candidate for improvement with CodeBERT, as CodeBERT can "read" the code semantics rather than just counting lines.

### Top Features (Global)
1. `cyclomatic_complexity` (Radon)
2. `num_parameters`
3. `num_tokens`
4. `max_nesting_depth`

## Next Steps (Day 6-7)
1. **Focus on "No Docstring"**: This is the primary task where ML can add value over static analysis.
2. **CodeBERT**: Implement CodeBERT to see if it can detect "No Docstring" better than the baseline (76.64%).
3. **Generalization**: Test on a completely new dataset where labels might not be perfectly aligned with Radon metrics (e.g., human labels).

## MLflow Tracking
All 5 experiments logged to MLflow.
- Experiment name: `codeguard-baseline`
- Total runs: 5
