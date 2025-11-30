# Experiment Log

## Week 2: Baseline Models (Day 5)

### Experiment #1: XGBoost Baseline - All Smells
- **Date:** 2025-11-30
- **Model:** XGBoost
- **Parameters:**
  - n_estimators: 100
  - max_depth: 6
  - learning_rate: 0.1
- **Results:**
  - **Average Val F1:** 95.33%
  - **Best Smell:** Long Method, Complexity, Params, Nesting (100% F1)
  - **Worst Smell:** No Docstring (76.64% F1)
- **Observations:** 
  - Perfect scores for structural smells due to feature alignment with labeling logic.
  - "No Docstring" is the only non-trivial task (76.64% F1) after excluding direct features.
- **Next Steps:**
  - Focus optimization on "No Docstring".
  - Use CodeBERT to improve "No Docstring" detection.

## Week 1: Data Collection & Labeling |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| EXP-001 | 2023-10-26 | Baseline (Dummy) | N/A | - | - | Initial baseline setup |
| EXP-002 | | XGBoost | | | | |
| EXP-003 | | CodeBERT | | | | |
