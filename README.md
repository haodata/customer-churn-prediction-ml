# Customer Churn Prediction

A machine learning project that predicts customer churn for a telecom provider, using the Telco Customer Churn dataset. The pipeline covers EDA, preprocessing, feature engineering, class-imbalance handling with SMOTE, and a comparison of three classification models, with an emphasis on **Recall** as the priority business metric.

## Overview

| | |
|---|---|
| **Dataset** | Telco Customer Churn (7,043 customers, 21 raw fields → 27 model features) |
| **Overall churn rate** | 26.5% (1,869 churned / 5,174 retained) |
| **Best AUC** | 0.828 (Logistic Regression) |
| **Top recall** | 0.83 (Logistic Regression) |

## Why Recall Matters Here

In churn prediction, missing a real churner (a false negative) is far more costly than flagging a loyal customer by mistake (a false positive) — the former is a lost retention opportunity, the latter is just a wasted outreach email. For that reason, this project optimizes for **Recall** and **F1** over raw Accuracy, and all models are evaluated at a lowered decision threshold of **0.3** (instead of the default 0.5) to catch more true churners.

## Pipeline

```
① Data Cleaning  →  ② Encoding  →  ③ Feature Engineering  →  ④ SMOTE Balancing  →  ⑤ Train / Evaluate
```

1. **Data Cleaning** — `TotalCharges` converted from string to numeric (11 missing values filled with the median); the non-predictive `customerID` column dropped.
2. **Encoding** — Binary categorical features mapped to 0/1; multi-valued categorical features (`InternetService`, `Contract`, `PaymentMethod`) one-hot encoded.
3. **Feature Engineering** — 3 new features added:
   - `tenure_group`: new (0–12mo) / mid (13–36mo) / long-term (>36mo)
   - `charges_per_month`: average monthly spend intensity
   - `num_services`: count of subscribed value-added services
4. **SMOTE Balancing** — Class imbalance fixed with SMOTE oversampling, applied **to the training set only** (after an 80/20 stratified split) to avoid data leakage and keep the test set representative of real-world class proportions. Training set balanced from ~4,139 retained / ~1,495 churned to **4,139 : 4,139**.
5. **Train / Evaluate** — Three models trained and compared on an 80/20 stratified split (1,409 held-out test samples), using a 0.3 classification threshold.

## Models & Results

| Model | AUC | Recall ★ | Precision | F1 | Accuracy |
|---|---|---|---|---|---|
| Logistic Regression | **0.828** | **0.83** | 0.47 | 0.60 | 0.71 |
| Random Forest | 0.822 | 0.81 | 0.49 | 0.61 | 0.72 |
| XGBoost | 0.811 | 0.74 | **0.52** | 0.61 | **0.75** |

**Model selection guidance:**
- To **maximize recall** and catch the most churners → **Logistic Regression** (AUC 0.828, Recall 0.83). It also offers the best interpretability for explaining decisions to business stakeholders.
- To **minimize false alarms** with higher precision → **XGBoost** (Accuracy 0.75, Precision 0.52).

## Top Churn Drivers (Feature Importance, Random Forest)

| Rank | Feature | Importance |
|---|---|---|
| 1 | TotalCharges | 0.113 |
| 2 | MonthlyCharges | 0.112 |
| 3 | charges_per_month | 0.107 |
| 4 | tenure | 0.104 |
| 5 | PaymentMethod = Electronic check | 0.087 |
| 6 | InternetService = Fiber optic | 0.060 |
| 7 | tenure_group | 0.043 |
| 8 | OnlineSecurity | 0.041 |
| 9 | TechSupport | 0.036 |
| 10 | Contract = Two year | 0.034 |

Spend and tenure-related features dominate — the top 4 alone account for roughly **44%** of total feature importance.

## Business Insights & Recommendations

| Driver | Insight | Recommended Action |
|---|---|---|
| **Contract type** | Month-to-month contracts churn the most (~42.7%) vs. ~2.8% for two-year contracts | Incentivize migration to annual/two-year contracts with discounts |
| **Tenure** | New customers (<12 months) churn far more than long-tenured ones | Strengthen onboarding and proactive care in the first 3 months |
| **MonthlyCharges** | Higher monthly bills correlate with higher churn | Proactively offer plan-optimization options to high-bill customers |
| **Internet service / add-ons** | Fiber users churn more than DSL users; customers without security or tech support churn more | Improve fiber service quality and bundle add-on services |

## Tech Stack

- **Data processing:** pandas, NumPy
- **Visualization:** matplotlib, seaborn
- **Class balancing:** imbalanced-learn (SMOTE)
- **Modeling:** scikit-learn (Logistic Regression, Random Forest), XGBoost

## Repository Structure

```
.
├── customer_churn_prediction.ipynb   # Full analysis notebook (EDA → modeling → insights)
└── README.md
└── index.html
└── archive.zip
```

## Data Source

[Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customer records with demographic, account, and service-usage attributes.

## Author
WENHAO LI
