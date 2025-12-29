# Fraud Detection System (Fintech Transactions)

## Project Overview
This project implements a **fraud detection model** using historical transaction data enriched with auxiliary risk signals. It demonstrates the **full data science lifecycle**: data cleaning, exploratory data analysis (EDA), feature engineering, modeling, evaluation, and fine-tuning.  

The workflow is designed to reflect **real-world fintech challenges**: messy multi-source data, imbalanced fraud labels, and the need for interpretable, business-driven models.

---

## Dataset Structure
- **train.csv** → 227,845 rows, 28 columns. Historical transactions with fraud labels (`Target`: 0 = Clean, 1 = Fraud).
- **test_share.csv** → 56,962 rows, 27 columns. Same structure as train, no `Target`.
- **Geo_scores.csv** → `(id, geo_score)` location-based risk scores.
- **Lambda_wts.csv** → `(Group, lambda_wt)` group-level weights.
- **Qset_tats.csv** → `(id, qsets_normalized_tat)` turnaround times.
- **instance_scores.csv** → `(id, instance_scores)` vulnerability scores.

files are merged into the training set by `id` or `Group`.

## Workflow Stages

### 1. Data Cleaning
- Convert string columns (`geo_score`, `qsets_normalized_tat`) to numeric.
- Handle missing values with median imputation.
- Deduplicate multiple entries per `id` by averaging.
- Merge auxiliary datasets into `train.csv`.

### 2. Exploratory Data Analysis (EDA)
- Class balance visualization (fraud vs clean).
- Feature distributions (`Normalised_FNT`, `geo_score`, `instance_scores`).
- Correlation heatmap for numeric features.

### 3. Feature Engineering
- **RiskIndex** = `geo_score + instance_scores + lambda_wt`
- **TatRatio** = `qsets_normalized_tat / (abs(Normalised_FNT)+1)`
- Encodes additional risk signals into composite features.

### 4. Modeling
- **Baseline:** Logistic Regression (interpretable).
- **Advanced:** Random Forest and XGBoost (ensemble/boosting).
- Handles class imbalance with `class_weight="balanced"`.

### 5. Evaluation
- Metrics: ROC-AUC, Precision-Recall curve, Confusion Matrix.
- Threshold tuning for business policies (approve/review/decline).

### 6. Fine-Tuning
- Hyperparameter optimization with `GridSearchCV`.
- Parameters tuned: `n_estimators`, `max_depth`, `min_samples_split`.
- Optimized for ROC-AUC (robust to imbalance).

--

## Business Impact
- Detects fraudulent transactions with high recall.
- Reduces false positives via threshold tuning.
- Provides interpretable risk features (`RiskIndex`, `TatRatio`) for compliance teams.
- Demonstrates **industry-grade fraud detection workflow** for fintech portfolios.

---

## How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Train baseline and advanced models
python run.py

# Explore results
jupyter notebook notebooks/EDA_and_Modeling.ipynb
