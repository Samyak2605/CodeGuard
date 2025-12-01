# Experiment Log

## Week 2: Baseline Optimization (Day 6)

### Experiment #2: Ensemble Methods (Voting/Stacking)
- **Date:** 2025-12-01
- **Model:** Voting Classifier (XGBoost + Random Forest) & Stacking
- **Results:**
  - **Best "No Docstring" F1:** 80.73% (Stacking)
  - **Improvement:** +4.09% over baseline
- **Observations:**
  - Stacking Ensemble provided the best performance for the hardest task.
  - Structural smells remain at 100% due to feature alignment.
- **Next Steps:**
  - Proceed to CodeBERT implementation (Week 3).

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
