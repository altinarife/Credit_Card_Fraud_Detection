# Credit Card Fraud Detection (Imbalanced Classification)

Detect fraudulent card transactions when fraud is extremely rare (about 0.17% of rows). The project compares imbalance-handling methods and models, then picks the decision threshold that minimises **business cost** rather than using the default 0.5.

## Files

| File | Description |
|---|---|
| `Credit_Card_Fraud_Detection.ipynb` | Full notebook; every code line has a comment. Saved without outputs. |
| `creditcard.csv` | **You must download this** (see below) and place it next to the notebook. |

## Dataset

Kaggle, "Credit Card Fraud Detection" (ULB / Worldline): 284,807 transactions over two days, 492 frauds.
Columns: `Time`, `V1`-`V28` (anonymised PCA components), `Amount`, `Class` (1 = fraud).
Download: search "Credit Card Fraud Detection" on kaggle.com (free account needed).
Alternative for a harder project: IEEE-CIS Fraud Detection (needs extra feature engineering; the notebook would need adapting).

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
# optional, unlock extra cells (the notebook runs without them):
pip install imbalanced-learn xgboost shap
```

Python 3.9 or newer. Without `imbalanced-learn` the notebook falls back to random oversampling instead of SMOTE; without `xgboost` it uses scikit-learn's `HistGradientBoostingClassifier`; without `shap` it skips the SHAP plot (permutation importance is always shown).

## How to run

1. Put the CSV and the notebook in one folder, open the notebook, run all cells top to bottom (a few minutes on a laptop).
2. Figures are saved to `figs/` automatically.

## Workflow

1. Load, drop duplicates, check class balance and the "accuracy trap".
2. EDA: class counts, Amount distribution, correlations with fraud, fraud by hour.
3. Stratified 60/20/20 train/validation/test split; scale only `Time` and `Amount` (`RobustScaler`).
4. Baselines: dummy classifier, plain Logistic Regression.
5. Imbalance handling: class weights, oversampling/SMOTE (train only), undersampling.
6. Gradient boosting (HistGradientBoosting, XGBoost if installed).
7. Metrics: Average Precision (area under the PR curve), ROC-AUC, precision, recall, F1, PR curves.
8. Cost-based threshold: missed fraud costs its transaction amount, a false alarm costs a fixed review fee (`FP_COST`, default 5, adjustable). The threshold is tuned on validation data.
9. One-time evaluation on the test set, confusion matrix, explanation with permutation importance (SHAP optional).

## Notes

- Results are not included here: they depend on running the notebook on the real Kaggle file. The code was tested end to end on a synthetic dataset with the same columns, only to confirm it runs.
- `FP_COST = 5` is an assumption; change it to match your scenario and state it in your report.
- Use Average Precision and cost, not accuracy: a model that flags nothing is about 99.8% accurate.
- Features are anonymised PCA components, so findings cannot be explained in business terms.
- Real transactions are time-ordered; a time-based split is closer to production use than the random stratified split used here.
