# Probability of Default Model Comparison — Lending Club 2007–2014

**Can gradient boosting models replace the industry-standard WoE + logistic regression approach in credit risk — and at what cost?**

This project benchmarks five tree-based machine learning models (XGBoost, LightGBM, CatBoost, Random Forest, Decision Tree) against the regulatory-standard WoE + logistic regression scorecard on Lending Club peer-to-peer loan data. All models are evaluated on the same hold-out test set using credit risk-specific metrics: AUC, GINI, Kolmogorov-Smirnov, and Brier Score.

📄 [Read the full paper](./Uzun_B_2022_PD_Model_Comparison.pdf)

---

## Key Results

| Model | AUC | KS | Brier Score | True Defaults Caught |
|-------|-----|----|-------------|----------------------|
| **XGBoost** | **0.702** | **0.298** | **0.068** | 1,483 |
| CatBoost | 0.696 | 0.288 | 0.065 | 1,147 |
| Random Forest | 0.673 | 0.254 | 0.068 | 2,873 |
| LightGBM | 0.621 | 0.130 | 0.082 | 1,792 |
| Decision Tree | 0.582 | 0.120 | 0.070 | 647 |
| Logistic Regression (WoE) | 0.691 | 0.278 | 0.071 | 4,648 |
| LASSO Logistic Reg. (WoE) | 0.689 | 0.276 | 0.072 | 4,811 |
| XGBoost (WoE) | 0.687 | 0.272 | 0.074 | 5,347 |
| CatBoost (WoE) | 0.685 | 0.272 | 0.069 | 3,213 |

**XGBoost achieves the best AUC (70.2%) and KS (29.8%)** — outperforming the WoE baseline in discriminatory power without any manual binning. However, WoE-based models identify far more actual defaults in absolute terms, because WoE binning concentrates the signal from bad borrowers. The gap is regulatory, not predictive.

---

## The Central Trade-off

| | WoE + Logistic Regression | Gradient Boosting (no WoE) |
|--|--|--|
| AUC | ~0.69 | **~0.70** |
| Defaults caught | **~4,800** | ~1,500 |
| Preprocessing cost | High (manual WoE binning) | Low |
| Regulatory compliance | ✅ Basel III / IFRS 9 | ❌ Explainability gap |
| Interpretability | Scorecard-readable | Feature importance only |

WoE models catch more defaults in absolute numbers because the binning directly encodes the relationship between each variable and default risk. Gradient boosting learns this automatically — but in a way that is harder to audit under current banking regulations.

**Bottom line:** XGBoost is not yet a drop-in replacement. It is a strong candidate once regulatory explainability requirements evolve, or when combined with WoE preprocessing (XGBoost(WoE) achieves both high AUC and the highest number of true defaults caught: 5,347).

---

## Methodology

**Data:** Lending Club 2007–2014 peer-to-peer loan data. Time-based split: 2007–2013 train (326,399 obs), 2014 test (139,886 obs). Default rate ~10.9% — imbalanced, making accuracy a misleading metric.

**Feature engineering:** 22 variables selected by Information Value (IV) from 150+ raw features. For logistic models: continuous variables discretized into WoE-scored bins. For ML models: one-hot encoding + partial WoE grouping for variables with missing values.

**Hyperparameter optimization:** Optuna (Bayesian, TPE sampler, 75 trials) for XGBoost, LightGBM, CatBoost, Random Forest.

| Model | Method | Key Hyperparameters |
|-------|--------|---------------------|
| LASSO | GridSearchCV (5-fold) | C = 0.01 |
| LightGBM | Optuna | max_depth=6, n_est=565, lr=1e-8 |
| CART | Optuna | max_depth=3, max_features=4 |
| Random Forest | Optuna | n_est=100, max_features=6 |
| XGBoost | Optuna | max_depth=5, n_est=348, lr=0.007 |
| CatBoost | Optuna | max_depth=2, n_est=589, lr=0.019 |

**Classification threshold:** p ≥ 0.80 → Non-Default. Conservative threshold reflecting credit risk practice where the cost of missing a default exceeds the cost of rejecting a good borrower.

**Metrics:** AUC · GINI (= 2·AUC − 1) · Kolmogorov-Smirnov · Brier Score · Precision · Recall · F1

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [Lending Club (Kaggle)](https://www.kaggle.com/datasets/wordsforthewise/lending-club) |
| Period | 2007–2014 |
| Train | 326,399 observations |
| Test | 139,886 observations |
| Default rate | ~10.9% |
| Features used | 22 (selected by IV) |
| Split | Time-based (not random) |

> **Data note:** The preprocessed training feature matrix (~300MB) is not included. The test feature matrix (`data/ml/X_test.csv`, 35MB, 93 features) and target vectors are available. Full preprocessing code is in the notebook.

---

## Repository Structure

```
├── notebooks/
│   └── ml_models_comparison.ipynb     # XGBoost, LightGBM, CatBoost + Optuna tuning
├── data/
│   └── ml/
│       ├── X_test.csv                 # Test features, 93 columns (35MB)
│       └── y_test.csv                 # Test labels (139,886 obs)
├── results/
│   ├── xgb_roc.png
│   ├── xgb_gini.png
│   ├── xgb_ks.png
│   ├── xgb_confusion_matrix.png
│   ├── xgb_classification_report.png
│   ├── xgb_feature_importance.png
│   └── xgb_brier_score.png
├── Uzun_B_2022_PD_Model_Comparison.pdf
└── README.md
```

---

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost optuna

jupyter notebook notebooks/ml_models_comparison.ipynb
```

> Download the raw Lending Club data from Kaggle and run the preprocessing cells to regenerate the training features. The notebook contains the full feature engineering pipeline.

---

## Context

IFRS 9 (2018) introduced lifetime expected credit loss requirements, increasing pressure on banks to improve PD model accuracy. Yet Basel III's IRB approach still demands interpretable, auditable models — which has historically locked institutions into logistic regression scorecards. This study quantifies the predictive trade-off and frames it within the regulatory constraint, contributing to the active debate on ML adoption in credit risk management.

---

Uzun, B. (2022). *Comparison of Probability of Default Models Used in Credit Risk Management.* Bachelor's Thesis, Department of Economics, Yıldız Technical University. Advisor: Prof. Dr. Hüseyin Taştan.
