# CodeGuard: ML-Powered Code Quality Analyzer

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)

## 🎯 Overview
CodeGuard is an advanced machine learning system designed to automatically detect code quality issues and "code smells" in Python repositories. By leveraging state-of-the-art NLP models like CodeBERT alongside traditional static analysis metrics, CodeGuard aims to provide intelligent, context-aware code quality assessments that go beyond simple linting rules.

## 🔍 Problem Statement
Traditional static analysis tools rely on rigid, rule-based heuristics that often miss context or produce high false-positive rates. As software complexity grows, maintaining code quality becomes increasingly difficult. CodeGuard addresses this by treating code quality as a machine learning problem, learning from thousands of high-quality open-source repositories to identify patterns indicative of poor maintainability, high complexity, and potential bugs.

## 📊 Approach
CodeGuard employs a hybrid approach combining:
1.  **Static Analysis**: Extracting quantitative metrics (cyclomatic complexity, nesting depth, etc.).
2.  **Machine Learning**: Using XGBoost on extracted features for efficient classification.
3.  **Deep Learning**: Fine-tuning CodeBERT to understand semantic context and detect subtle code smells.

## 🏗️ Architecture
The system consists of a data collection pipeline that mines GitHub repositories, a feature extraction engine that computes code metrics, and a dual-model inference engine. (Architecture diagram to be added).

## 📈 Results (Target)
| Metric | Baseline (Rule-based) | Target (ML/DL) |
| :--- | :--- | :--- |
| Precision | 65% | >85% |
| Recall | 70% | >80% |
| F1-Score | 67% | >82% |

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Git

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/codeguard-ml.git
    cd codeguard-ml
    ```

2.  **Create a virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install the package in editable mode**
    ```bash
    pip install -e .
    ```

### Usage
(Coming soon: Instructions for running the data collector and training scripts)

## 📁 Project Structure
```
codeguard-ml/
├── data/              # Dataset storage
│   ├── raw/           # Raw collected repositories
│   └── processed/     # Processed datasets for training
├── src/               # Source code
│   ├── models/        # ML/DL model implementations
│   ├── data_collector.py
│   ├── feature_extractor.py
│   └── preprocessing.py
├── notebooks/         # Jupyter notebooks for EDA and experiments
├── tests/             # Unit tests
├── configs/           # Configuration files
├── scripts/           # Training and utility scripts
└── docs/              # Documentation and reports
```

## 🗂️ Dataset
We are collecting a dataset of 1000 public GitHub repositories. The data collection pipeline filters for Python projects with high star counts to ensure a baseline of quality, while also identifying specific commits or files known to contain refactoring opportunities (code smells).

## 🔬 Methodology
We focus on detecting five key code smells:
1.  **Long Method**: Functions exceeding 50 lines.
2.  **High Complexity**: Cyclomatic complexity > 10.
3.  **Duplicated Code**: Semantically similar code blocks.
4.  **Too Many Parameters**: Functions with > 5 parameters.
5.  **Deep Nesting**: Nesting levels > 4.

## 📝 Development Log
Follow our experimental progress in [EXPERIMENTS.md](docs/EXPERIMENTS.md).

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author
[Your Name]
AI/ML Engineer

## 🙏 Acknowledgments
- Microsoft for the CodeBERT model.
- The open-source community for the tools and libraries used.
