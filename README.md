# Customer Churn & Retention Analytics

An end-to-end data analytics project analyzing customer churn drivers and
building a predictive model to flag at-risk customers, using the IBM Telco
Customer Churn dataset (7,043 customers).

Built as a portfolio project demonstrating the full analytics workflow:
data cleaning → exploratory analysis → statistical hypothesis testing →
predictive modeling → business recommendations.

## Key Results

- **Overall churn rate: 26.5%**
- Identified and statistically validated the top churn drivers: **contract type**
  (Cramer's V = 0.41), **tenure**, **internet service type**, and **payment method**
  (all p < 0.001)
- Built a **Random Forest classifier** (ROC-AUC = 0.846, 79% recall on churners) to
  score every customer's churn probability
- Quantified revenue at risk (**~$1.92M/year** from high-risk customers) and modeled
  the revenue impact of a targeted retention intervention (**~$309K/year** from a
  single contract-upgrade campaign)
- Delivered a prioritized, actionable retention strategy

## Project Structure

```
churn_retention_project/
├── data/
│   ├── raw/                          # Original Telco-Customer-Churn.csv
│   └── processed/                    # Cleaned + feature-engineered dataset
├── notebooks/
│   ├── 01_data_cleaning.py           # Missing values, type fixes, feature engineering
│   ├── 02_eda.py                     # Segment-level churn analysis + charts
│   ├── 03_statistical_tests.py       # Chi-square & t-tests for significance
│   └── 04_predictive_modeling.py     # Logistic Regression + Random Forest
├── outputs/
│   ├── figures/                      # 11 charts (churn by segment, model performance)
│   └── reports/                      # Summary stats, test results, model metrics,
│                                      # per-customer risk scores, business report
└── README.md
```

## How to Run

```bash
cd notebooks
python3 01_data_cleaning.py
python3 02_eda.py
python3 03_statistical_tests.py
python3 04_predictive_modeling.py
```

Requires: `pandas`, `numpy`, `matplotlib`, `scipy`, `scikit-learn`.

## Methodology

1. **Data Cleaning** — fixed `TotalCharges` type coercion issue (11 blank values
   for zero-tenure customers), standardized categorical labels, engineered tenure
   cohorts and an add-on service count.
2. **EDA** — churn rate broken out by contract, internet service, payment method,
   tenure cohort, and service engagement; 7 charts.
3. **Statistical Testing** — chi-square tests of independence (categorical drivers)
   and Welch's t-tests (continuous drivers) against churn, with effect sizes
   (Cramer's V), to confirm which EDA patterns are statistically significant
   rather than sampling noise. All 14 tested drivers were significant at α=0.05.
4. **Predictive Modeling** — Logistic Regression as an interpretable baseline,
   Random Forest as the primary model; evaluated on accuracy, precision, recall,
   F1, and ROC-AUC on a held-out 20% test set; class-weighted to prioritize
   recall on churners.
5. **Business Translation** — converted model output into a per-customer risk
   tier (Low/Medium/High), estimated revenue at risk, and modeled the ROI of a
   retention intervention.

## Files of Interest

- `outputs/reports/business_insights_and_recommendations.md` — full write-up of
  findings and recommendations
- `outputs/reports/customer_churn_risk_scores.csv` — every customer scored with
  churn probability and risk tier, ready for a retention team to action
- `outputs/reports/model_comparison.csv` — model evaluation metrics
- `outputs/reports/statistical_tests_summary.csv` — full hypothesis test results

## Author

IIT Roorkee
