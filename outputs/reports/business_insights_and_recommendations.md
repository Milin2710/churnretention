# Customer Churn & Retention Analytics — Business Insights & Recommendations

**Dataset:** IBM Telco Customer Churn (7,043 customers, 21 original fields)
**Overall churn rate:** 26.54%

---

## 1. Key Findings

### 1.1 Contract type is the single biggest churn driver
- Month-to-month customers churn at **42.7%**, vs. 11.3% for one-year and **2.8%** for
  two-year contracts.
- Statistically confirmed: chi-square test, Cramer's V = 0.41 (largest effect size of
  any categorical variable tested), p < 0.001.
- Month-to-month customers make up 55% of the base (3,875 of 7,043) but drive a
  disproportionate share of churn.

### 1.2 Tenure (customer lifecycle stage) is the second-strongest driver
- Customers in their first 6 months churn at **52.9%**; customers with 49-72 months
  of tenure churn at just **9.5%**.
- Confirmed with Welch's t-test (p < 0.001): churned customers average 18 months
  tenure vs. 37.6 months for retained customers.
- **Interpretation:** most churn risk is concentrated in the early relationship —
  the first 6-12 months are the critical retention window.

### 1.3 Fiber optic internet customers churn at nearly 2x the base rate
- Fiber optic churn rate: **41.9%** vs. ~19% for DSL and ~7.4% for no internet service.
- Likely drivers (not directly testable from this dataset, but suggested by the
  correlation with `MonthlyCharges`): higher price point and/or service reliability
  perception. Fiber customers also have the highest average monthly charges.

### 1.4 Payment method signals risk
- Electronic check users churn at **45.3%**, more than double the rate of customers
  on automatic payment methods (bank transfer / credit card, ~15-17%).
- This is one of the most actionable signals available at signup: it requires no
  behavioral history, only the payment method chosen.

### 1.5 Service "stickiness" reduces churn
- Customers with more add-on services (Online Security, Tech Support, Device
  Protection, etc.) churn less. Each additional add-on service is associated with
  a statistically significant reduction in churn (t-test p < 0.001).
- Customers with **zero** add-ons churn at a much higher rate than those with 3+.

### 1.6 Model confirms these are the top predictive features
A Random Forest classifier (ROC-AUC = 0.846, recall on churners = 79%) ranks
feature importance as:
1. Tenure
2. Two-year contract (protective)
3. Total charges
4. Fiber optic internet
5. Monthly charges
6. Electronic check payment method

This matches the statistical testing and EDA findings — a good sign the model
isn't picking up spurious patterns.

---

## 2. Revenue Impact

- **2,063 customers (29% of the base)** are scored "High risk" (>60% predicted churn
  probability) by the model, representing **~$160,000/month** (~$1.92M/year) of
  revenue currently at risk.
- **Scenario:** if a targeted retention program (contract upgrade incentives +
  proactive outreach) reduces churn among month-to-month customers by just
  **10 percentage points** (42.7% → 32.7%), the business retains an estimated
  **388 additional customers/year**, worth approximately **$309,000/year** in
  recurring revenue — from this single lever alone.

---

## 3. Recommendations

| Priority | Recommendation | Rationale |
|---|---|---|
| 1 | **Incentivize contract upgrades** — offer a discount or perk to move month-to-month customers to 1- or 2-year contracts, targeted at customers in their first 6 months. | Contract type has the largest effect size; the first 6 months is the highest-risk window. |
| 2 | **Early-tenure onboarding program** — proactive check-ins, usage tips, or a loyalty milestone at 3/6 months for new customers. | 0-6 month churn (53%) is ~2x the base rate; this is where most churn originates. |
| 3 | **Investigate fiber optic service experience** — price-to-value perception, reliability complaints, or competitive pressure specific to fiber customers. | Fiber churns at 42% despite (or because of) being the premium tier. |
| 4 | **Nudge electronic-check users to autopay** — small incentive (e.g., one-time bill credit) to switch to bank transfer or credit card autopay. | Cheapest, most actionable lever — a payment-method change, not a behavior change. |
| 5 | **Bundle add-on services (Tech Support, Online Security) into entry-tier plans** | More add-ons correlate with lower churn — likely both a stickiness and a value-perception effect. |
| 6 | **Deploy the churn risk score operationally** — route the 2,063 "High risk" customers (`customer_churn_risk_scores.csv`) to a retention team for proactive outreach, prioritized by revenue (MonthlyCharges) within tier. | Turns the model from an analysis artifact into an action list. |

---

## 4. Caveats & Next Steps

- This is a single-snapshot dataset (no time-series), so causal claims (e.g., "fiber
  causes churn") are not established — correlation and statistical association only.
  A/B testing any retention intervention is recommended before wide rollout.
- The Random Forest model favors **recall over precision** (class-weighted) by
  design, since in a retention context the cost of missing a would-be churner
  usually exceeds the cost of a wasted retention offer to a loyal customer — this
  trade-off should be validated against actual retention-offer cost data.
- Next steps: track cohort churn over time post-intervention, incorporate customer
  support ticket/complaint data if available, and test uplift modeling to target
  customers most *persuadable* by an offer (not just most likely to churn).
