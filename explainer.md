# Resume Points — Explained

This document walks through each resume bullet for the **Customer Churn & Retention
Analytics** project, explaining exactly what was built, how, and why — so you can
speak to it confidently in an interview.

---

## Project Overview

**What it is:** An end-to-end data analytics project that predicts which telecom
customers are likely to cancel their subscription (churn), identifies *why* they
churn using statistical testing, and translates the findings into a retention
strategy with a projected revenue impact.

**Dataset:** IBM's public Telco Customer Churn dataset — 7,043 customers, 21
original fields (demographics, account details, subscribed services, billing info,
and whether the customer churned).

**Workflow:** Data cleaning → Exploratory Data Analysis (EDA) → Statistical
hypothesis testing → Predictive modeling → Business recommendations. This mirrors
the real workflow of a business/data analyst, which is why it's a strong
interview talking point — it's not just "trained a model," it's the full
decision-support pipeline.

---

## Bullet 1

> *Predicted customer churn on 7,043-record telecom dataset using Random Forest &
> Logistic Regression, achieving 0.846 ROC-AUC.*

**What this means:**
- The **target variable** is `Churn` (Yes/No) — did the customer leave.
- Two models were trained on an 80/20 train-test split:
  - **Logistic Regression** — a linear, highly interpretable baseline. Each
    feature gets a coefficient showing its direction and strength of association
    with churn. Good for explaining "why" to stakeholders.
  - **Random Forest** — an ensemble of decision trees that captures non-linear
    relationships and feature interactions (e.g., "fiber + month-to-month" being
    worse than either alone). Used as the primary model.
- Both models were trained with `class_weight="balanced"` because only 26.5% of
  customers churned — without this correction, a model could get 73% accuracy by
  just predicting "no churn" for everyone, which is useless in practice.
- **ROC-AUC (Area Under the ROC Curve) = 0.846** measures how well the model
  ranks churners above non-churners across all probability thresholds. 0.5 = random
  guessing, 1.0 = perfect separation. 0.846 is a strong result for this type of
  tabular business data.
- Random Forest slightly outperformed Logistic Regression (0.846 vs. 0.841 AUC),
  and importantly achieved **79% recall** on churners — meaning it correctly
  flags 4 out of 5 customers who actually churn, which matters more than raw
  accuracy in a retention context (missing a churner is costlier than a false
  alarm).

**Why it matters for the resume:** Shows you can frame a business problem as a
supervised classification task, handle class imbalance correctly, and choose
the right evaluation metric (ROC-AUC/recall, not just accuracy) — a common gap
in junior candidates' understanding.

---

## Bullet 2

> *Validated churn drivers via chi-square and t-tests across 14 features,
> isolating contract type as the top predictor (p<0.001).*

**What this means:**
- Before trusting any pattern seen in charts (e.g., "month-to-month customers
  seem to churn more"), it needs to be statistically confirmed — otherwise you
  risk building a strategy around random noise.
- **Chi-square test of independence** was used for 10 categorical variables
  (Contract, Internet Service, Payment Method, Senior Citizen status, etc.) to
  test whether churn rate genuinely differs across categories, or whether the
  difference could plausibly be due to chance.
- **Welch's t-test** was used for 4 continuous variables (tenure, monthly
  charges, total charges, number of add-on services) to test whether the mean
  value genuinely differs between churned and retained customers.
- **p-value < 0.05** is the standard significance threshold — a p-value this low
  means there's less than a 5% chance the observed difference is due to random
  sampling variation.
- **Result:** all 14 tested drivers were statistically significant (p < 0.05),
  and several were significant at p < 0.001 (far below the threshold).
- **Cramer's V** (an effect size measure for categorical variables, 0 to 1) was
  used to rank *how strong* each significant relationship is — statistical
  significance alone doesn't tell you which factor matters most, especially in
  a large dataset where even tiny differences become "significant." Contract
  type had the highest Cramer's V (0.41), meaning it has by far the strongest
  relationship with churn of any variable tested.

**Why it matters for the resume:** Shows rigor — the ability to distinguish
"this looks different in a chart" from "this is a real, defensible effect,"
which is exactly what's expected of a data analyst presenting findings to
stakeholders.

---

## Bullet 3

> *Identified $1.92M in at-risk annual revenue by building a customer-level
> churn risk-scoring system.*

**What this means:**
- The trained Random Forest model doesn't just output "will churn / won't
  churn" — it outputs a **churn probability (0 to 1)** for every individual
  customer.
- Every customer in the dataset was scored and bucketed into three risk tiers:
  - **Low risk:** <30% churn probability
  - **Medium risk:** 30–60%
  - **High risk:** >60%
- **2,063 customers (29% of the base)** fell into the "High risk" tier.
- Summing their `MonthlyCharges` gives ~$160,000/month in revenue tied to
  customers likely to leave soon — annualized, that's **~$1.92M/year** of
  revenue at risk if nothing is done.
- This converts a model output (a probability, which isn't inherently
  meaningful to a business stakeholder) into a **dollar figure**, which is what
  actually drives business decisions and budget allocation for a retention team.

**Why it matters for the resume:** This is the step most technical candidates
skip — translating a model's output into business impact. It shows you think
like an analyst who has to justify decisions to non-technical stakeholders, not
just a modeler optimizing a metric.

---

## Bullet 4

> *Proposed a retention strategy projected to save ~$309K/year by cutting
> month-to-month churn 10 points.*

**What this means:**
- Month-to-month contract customers make up 55% of the customer base (3,875 of
  7,043) and churn at 42.7% — by far the highest-risk segment.
- A **scenario/what-if analysis** was run: if a retention intervention (e.g.,
  discount incentives to upgrade to annual contracts, proactive outreach in the
  first 6 months) reduced month-to-month churn by 10 percentage points (from
  42.7% to 32.7%), roughly **388 additional customers/year** would be retained.
- Multiplying retained customers × their average monthly charge ($66.40) × 12
  months gives the annualized revenue impact: **~$309,000/year** — from this
  single lever alone, without needing to model the entire customer base.
- This is presented as a **scenario**, not a guarantee — the report explicitly
  notes that any real intervention should be validated with an A/B test before
  full rollout, since the dataset is a single snapshot and can't establish
  causation on its own.

**Why it matters for the resume:** Demonstrates the ability to go from "here's
a pattern" to "here's a concrete, costed recommendation with a clear ROI
estimate" — the actual deliverable a business expects from a churn analysis,
not just a trained model sitting in a notebook.

---

## Quick Reference: Full Project Pipeline

| Stage | Script | Output |
|---|---|---|
| Data Cleaning | `01_data_cleaning.py` | Cleaned dataset, engineered features (tenure cohorts, add-on count) |
| EDA | `02_eda.py` | 7 charts: churn by contract, tenure, internet service, payment method |
| Statistical Testing | `03_statistical_tests.py` | Chi-square/t-test results with p-values and effect sizes |
| Predictive Modeling | `04_predictive_modeling.py` | Logistic Regression + Random Forest, ROC curve, confusion matrix, feature importance, per-customer risk scores |
| Business Translation | `business_insights_and_recommendations.md` | Revenue-at-risk estimate, retention ROI scenario, 6 prioritized recommendations |

## Likely Interview Follow-Up Questions

- **"Why Random Forest over Logistic Regression if the gains were small?"** →
  Random Forest had higher recall on churners (79% vs. lower for LR) and can
  capture feature interactions LR can't; LR was kept as an interpretable
  baseline for comparison and to sanity-check the RF's top features.
- **"Why class-weighted instead of oversampling (SMOTE)?"** → Simpler, avoids
  synthetic data risk, and achieved strong recall directly; a reasonable
  next step to test would be SMOTE or threshold-tuning for a precision/recall
  trade-off matched to actual retention-offer cost.
- **"How would you validate the $309K estimate?"** → Run a controlled A/B test
  on a sample of month-to-month customers with the retention offer vs. a
  holdout group, and measure actual churn reduction before rolling out
  broadly.
