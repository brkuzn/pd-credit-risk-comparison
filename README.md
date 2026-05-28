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

## Background: Credit Risk & Regulatory Context

Under the Basel III Internal Ratings-Based (IRB) approach, banks that model their own credit risk must estimate three components for each loan:

- **PD (Probability of Default)** — likelihood a borrower fails to repay within 12 months
- **LGD (Loss Given Default)** — share of exposure lost if default occurs
- **EAD (Exposure at Default)** — outstanding balance at the time of default

Expected Credit Loss is then: **ECL = PD × LGD × EAD**

IFRS 9 (effective 2018) extended this to *lifetime* ECL for stage 2 and stage 3 assets, significantly increasing the pressure on banks to produce accurate, forward-looking PD estimates. Yet Basel III's IRB approach simultaneously demands that models be interpretable and auditable — which has historically locked institutions into logistic regression scorecards. This study quantifies the predictive trade-off and frames it within that regulatory constraint.

---

## Methodology

### Data

Lending Club 2007–2014 peer-to-peer loan data. The train/test split is **time-based** (not random): 2007–2013 for training (326,399 obs) and 2014 for testing (139,886 obs). This mimics real-world model validation, where a model trained on historical data is applied to future unseen loans. Default rate is ~10.9%, making the dataset imbalanced — standard accuracy is therefore a misleading metric.

### Feature Selection via Information Value (IV)

From 150+ raw features, 22 variables were selected using **Information Value (IV)**, which measures how well a variable distinguishes defaulters from non-defaulters:

```math
IV = \sum_{i=1}^{n} \left( P_{\text{good},i} - P_{\text{bad},i} \right) \times \ln\left(\frac{P_{\text{good},i}}{P_{\text{bad},i}}\right)
```

where each bin $i$ contributes proportionally to how different its share of good (non-default) and bad (default) borrowers is. The standard interpretation:

| IV | Predictive Power |
|----|-----------------|
| < 0.02 | Useless |
| 0.02 – 0.1 | Weak |
| 0.1 – 0.3 | Medium |
| 0.3 – 0.5 | Strong |
| > 0.5 | Suspicious (may indicate data leakage) |

Top features by IV: `int_rate` (interest rate), `last_pymnt_amnt` (last payment amount), `total_rec_prncp` (principal received), `out_prncp` (outstanding principal), `installment`.

### Weight of Evidence (WoE) Preprocessing

For logistic regression models, continuous variables are **discretized into bins** and each bin is replaced by its Weight of Evidence score:

```math
WoE_i = \ln\left(\frac{P_{\text{good},i}}{P_{\text{bad},i}}\right)
```

This transformation has three practical advantages for credit risk:
1. It linearizes the relationship between each predictor and the log-odds of default, which satisfies the assumption of logistic regression.
2. It handles missing values naturally — missing data becomes its own bin with its own WoE score.
3. The resulting model coefficients are directly interpretable as a **scorecard**: each variable's contribution to the default probability is explicit and auditable, which satisfies Basel III documentation requirements.

For ML models (XGBoost, CatBoost, etc.), WoE binning was applied only to variables with significant missing values; all other variables were one-hot encoded.

### Hyperparameter Optimization

All ensemble models were tuned with **Optuna** (Bayesian optimization, Tree-structured Parzen Estimator sampler, 75 trials):

| Model | Method | Key Hyperparameters |
|-------|--------|---------------------|
| LASSO | GridSearchCV (5-fold) | C = 0.01 |
| LightGBM | Optuna | max_depth=6, n_est=565, lr=1e-8 |
| Decision Tree | Optuna | max_depth=3, max_features=4 |
| Random Forest | Optuna | n_est=100, max_features=6 |
| XGBoost | Optuna | max_depth=5, n_est=348, lr=0.007 |
| CatBoost | Optuna | max_depth=2, n_est=589, lr=0.019 |

**Classification threshold:** p ≥ 0.80 → Non-Default. This conservative threshold reflects credit risk practice: missing a true default (false negative) is far more costly than rejecting a creditworthy borrower (false positive).

---

## Evaluation Metrics

### ROC AUC

The **Receiver Operating Characteristic** curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at every classification threshold. The **Area Under the Curve (AUC)** summarizes this in a single number:

- AUC = 0.5 → random classifier (no discrimination)
- AUC = 1.0 → perfect classifier
- In credit risk, AUC > 0.65 is generally considered acceptable; > 0.70 is strong.

```math
\text{TPR (Recall)} = \frac{TP}{TP + FN}, \qquad \text{FPR} = \frac{FP}{FP + TN}
```

### GINI Coefficient

A rescaling of AUC that maps [0.5, 1.0] → [0, 1], making it easier to compare models in a credit risk context:

```math
\text{GINI} = 2 \times \text{AUC} - 1
```

XGBoost achieves GINI = 0.404, meaning it correctly ranks ~40% more borrower pairs by default risk than a random model.

### Kolmogorov-Smirnov (KS) Statistic

The **KS statistic** measures the maximum separation between the cumulative distribution of predicted scores for defaulters vs. non-defaulters:

```math
KS = \max_t \left| F_{\text{bad}}(t) - F_{\text{good}}(t) \right|
```

where $F_{\text{bad}}(t)$ and $F_{\text{good}}(t)$ are the CDFs of predicted default probabilities for bad and good borrowers respectively. KS is widely used in banking model validation:

| KS | Discriminatory Power |
|----|---------------------|
| < 0.20 | Poor |
| 0.20 – 0.40 | Acceptable |
| 0.40 – 0.60 | Good |
| > 0.60 | Very good (may indicate overfitting) |

XGBoost achieves KS = 0.298, placing it in the acceptable-to-good range.

### Brier Score

The **Brier Score** measures the mean squared error of predicted probabilities against actual binary outcomes:

```math
BS = \frac{1}{N} \sum_{i=1}^{N} \left(\hat{p}_i - y_i\right)^2
```

where $\hat{p}_i$ is the predicted default probability and $y_i \in \{0, 1\}$ is the true label. Lower is better. A score of 0 is perfect; a naïve model predicting the base rate always achieves BS ≈ 0.097 on this dataset (10.9% default rate). All models beat the naïve baseline.

### Precision, Recall, F1

At the chosen threshold (p < 0.80 → predicted default):

```math
\text{Precision} = \frac{TP}{TP + FP}, \qquad \text{Recall} = \frac{TP}{TP + FN}, \qquad F_1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
```

Because the dataset is imbalanced (~10.9% defaults), F1 and Recall are more informative than accuracy. The WoE-based models achieve significantly higher Recall at this threshold — they catch more actual defaults — because WoE binning explicitly codes the borrower risk tiers that the conservative threshold targets.

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

> **Data note:** The preprocessed training feature matrix (~300MB) is not included in this repo. The test feature matrix (`data/ml/X_test.csv`, 35MB, 93 features) and target vector are available. Full preprocessing code is in the notebook.

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

Uzun, B. (2022). *Comparison of Probability of Default Models Used in Credit Risk Management.* Bachelor's Thesis, Department of Economics, Yıldız Technical University. Advisor: Prof. Dr. Hüseyin Taştan.
