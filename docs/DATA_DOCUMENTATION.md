# CodeGuard Dataset Documentation

## Collection (Day 2)
- **Source**: GitHub public repositories
- **Repositories Processed**: 10 (Initial Batch)
- **Functions Collected**: 3,647
- **Collection Date**: 2025-11-28

## Labeling (Day 3)
- **Method**: Automated labeling using `ast` and `radon`.
- **Code Smells Detected**: 5 types.
- **Manual Verification**: Script provided (`src/data_collection/verify_labels.py`).

## Code Smell Definitions

### 1. Long Method
- **Threshold**: > 50 lines
- **Rationale**: Functions should do one thing and be concise.
- **Detection**: Counting lines in the function body.

### 2. High Complexity
- **Threshold**: Cyclomatic Complexity > 10
- **Rationale**: Complex code is hard to test and maintain.
- **Detection**: Using `radon` library.

### 3. Too Many Parameters
- **Threshold**: > 5 parameters
- **Rationale**: Indicates poor design or need for a data object.
- **Detection**: Counting arguments in function definition.

### 4. Deep Nesting
- **Threshold**: Nesting level > 4
- **Rationale**: Reduces readability and indicates complex logic.
- **Detection**: Recursive AST traversal.

### 5. No Docstring
- **Threshold**: Missing docstring
- **Rationale**: Lack of documentation hinders understanding.
- **Detection**: `ast.get_docstring()`.

## Dataset Statistics (Cleaned)
- **Total Clean Functions**: 1,624
- **Original Count**: 3,647
- **Parse Errors Removed**: 2,018 (55.3%)
- **Duplicates Removed**: 5

## Known Limitations
- **Parse Errors**: A significant portion of code could not be parsed, likely due to syntax errors in the source or Python version mismatches (e.g., Python 2 code).
- **Context**: Automated labeling does not understand the semantic context of the code.

## Next Steps
- **Manual Verification**: User to verify a sample of labels.
- **ML Training**: Use this dataset to train CodeBERT in Week 2.
