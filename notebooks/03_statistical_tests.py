"""
03_statistical_tests.py
------------------------
Customer Churn & Retention Analytics — Statistical Significance Testing

Confirms which segment differences seen in EDA are statistically
significant (not just visually different), using:
  - Chi-square test of independence for categorical drivers vs Churn
  - Welch's t-test for continuous drivers (tenure, MonthlyCharges) vs Churn
"""

import pandas as pd
from scipy import stats

IN_PATH = "../data/processed/telco_churn_clean.csv"
OUT_PATH = "../outputs/reports/statistical_tests_summary.csv"

CATEGORICAL_DRIVERS = [
    "Contract", "InternetService", "PaymentMethod", "PaperlessBilling",
    "SeniorCitizen", "Partner", "Dependents", "TenureGroup",
    "OnlineSecurity_simplified", "TechSupport_simplified",
]

CONTINUOUS_DRIVERS = ["tenure", "MonthlyCharges", "TotalCharges", "NumAddonServices"]

ALPHA = 0.05


def chi_square_test(df, col, target="Churn"):
    contingency = pd.crosstab(df[col], df[target])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    # Cramer's V for effect size
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = (chi2 / (n * min_dim)) ** 0.5 if min_dim > 0 else float("nan")
    return chi2, p, cramers_v


def t_test(df, col, target="Churn_Flag"):
    churned = df.loc[df[target] == 1, col].dropna()
    retained = df.loc[df[target] == 0, col].dropna()
    t_stat, p = stats.ttest_ind(churned, retained, equal_var=False)  # Welch's t-test
    mean_diff = churned.mean() - retained.mean()
    return t_stat, p, mean_diff, churned.mean(), retained.mean()


def main():
    df = pd.read_csv(IN_PATH)
    results = []

    print("=== Chi-square tests: categorical drivers vs Churn ===\n")
    for col in CATEGORICAL_DRIVERS:
        chi2, p, cramers_v = chi_square_test(df, col)
        sig = "YES" if p < ALPHA else "no"
        print(f"{col:35s} chi2={chi2:9.2f}  p={p:.2e}  Cramer's V={cramers_v:.3f}  significant={sig}")
        results.append({
            "driver": col, "test": "chi-square", "statistic": chi2,
            "p_value": p, "effect_size": cramers_v, "effect_metric": "Cramer's V",
            "significant_at_0.05": p < ALPHA
        })

    print("\n=== Welch's t-tests: continuous drivers vs Churn ===\n")
    for col in CONTINUOUS_DRIVERS:
        t_stat, p, mean_diff, churn_mean, retain_mean = t_test(df, col)
        sig = "YES" if p < ALPHA else "no"
        print(f"{col:20s} t={t_stat:8.2f}  p={p:.2e}  churned_mean={churn_mean:8.2f}  "
              f"retained_mean={retain_mean:8.2f}  diff={mean_diff:8.2f}  significant={sig}")
        results.append({
            "driver": col, "test": "welch_t_test", "statistic": t_stat,
            "p_value": p, "effect_size": mean_diff, "effect_metric": "mean_diff (churned-retained)",
            "significant_at_0.05": p < ALPHA
        })

    results_df = pd.DataFrame(results).sort_values("p_value")
    results_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}")

    n_sig = results_df["significant_at_0.05"].sum()
    print(f"\n{n_sig} of {len(results_df)} tested drivers are statistically "
          f"significant predictors of churn at alpha=0.05.")

    # Top drivers by effect size (Cramer's V, categorical only)
    top_cat = results_df[results_df["test"] == "chi-square"].nlargest(5, "effect_size")
    print("\nTop 5 categorical drivers by effect size (Cramer's V):")
    print(top_cat[["driver", "effect_size", "p_value"]].to_string(index=False))


if __name__ == "__main__":
    main()
