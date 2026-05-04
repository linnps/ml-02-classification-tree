# 02 — Tree-Based Classification

> Status: 🟡 Planned (skeleton only)

## Goal

Predict a categorical label from mixed-type tabular features. Walk up the
tree-based ladder — single decision tree → random forest → gradient boosting —
and show what each step buys you in accuracy and what it costs in
interpretability.

## Topics covered

- Logistic regression as a baseline
- Decision tree, with depth tuning and overfitting visualization
- Random forest, with feature-importance interpretation
- Gradient boosting (XGBoost / LightGBM)
- Class imbalance handling (resampling, class weights)
- ROC / PR curves, confusion matrix, F1

## Dataset

**Self-generated synthetic data — no third-party / copyrighted datasets.**
A custom generator (`generate_data.py`) produces a tabular classification
problem with a configurable, *known* decision boundary — non-linear
interactions (XOR-style), multimodal feature distributions, and adjustable
class imbalance. Because the ground-truth boundary is in our hands,
each model's success is measured against the optimal possible boundary,
not just holdout accuracy.

## Tech stack

- Python 3.10+
- pandas, scikit-learn, xgboost, lightgbm
- shap for feature attribution

## Run

```bash
pip install -r requirements.txt
python train.py
```

## Results

_To be filled in after implementation._

## What I learned

_To be filled in after implementation._

---
*Part of the [machine-learning portfolio](../README.md).*
