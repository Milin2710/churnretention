"""
01_data_cleaning.py
--------------------
Customer Churn & Retention Analytics — Data Cleaning

Loads the raw IBM Telco Customer Churn dataset, fixes data types,
handles missing values, engineers a few useful fields, and writes
a cleaned dataset to data/processed/ for downstream EDA and modeling.
"""

import pandas as pd
import numpy as np

RAW_PATH = "../data/raw/Telco-Customer-Churn.csv"
OUT_PATH = "../data/processed/telco_churn_clean.csv"


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- TotalCharges: stored as string; blanks occur for tenure == 0 (new customers) ---
    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    n_missing = df["TotalCharges"].isna().sum()
    print(f"TotalCharges missing before fix: {n_missing}")

    # For brand-new customers (tenure == 0), TotalCharges is legitimately ~0
    df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0.0

    # Any remaining missing values: impute with median (robust to outliers)
    if df["TotalCharges"].isna().sum() > 0:
        median_val = df["TotalCharges"].median()
        df["TotalCharges"] = df["TotalCharges"].fillna(median_val)

    # --- SeniorCitizen: 0/1 -> Yes/No for consistency with other categoricals ---
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # --- Target variable: binary encode ---
    df["Churn_Flag"] = df["Churn"].map({"Yes": 1, "No": 0})

    # --- Standardize "No internet service" / "No phone service" to "No" ---
    # These are functionally "No" for the purposes of segment analysis;
    # kept as a separate engineered column set to preserve original detail too.
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "MultipleLines"
    ]
    for col in service_cols:
        df[col + "_simplified"] = df[col].replace(
            {"No internet service": "No", "No phone service": "No"}
        )

    # --- Feature engineering ---
    # Tenure buckets for cohort-style analysis
    bins = [-1, 6, 12, 24, 48, 72]
    labels = ["0-6 mo", "7-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"]
    df["TenureGroup"] = pd.cut(df["tenure"], bins=bins, labels=labels)

    # Count of subscribed add-on services (proxy for engagement/stickiness)
    addon_cols = [
        "OnlineSecurity_simplified", "OnlineBackup_simplified",
        "DeviceProtection_simplified", "TechSupport_simplified",
        "StreamingTV_simplified", "StreamingMovies_simplified"
    ]
    df["NumAddonServices"] = (df[addon_cols] == "Yes").sum(axis=1)

    # Average revenue per month of tenure (sanity-check / alt view of charges)
    df["AvgMonthlySpend"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
    )

    # Drop customerID duplicates if any
    before = len(df)
    df = df.drop_duplicates(subset="customerID")
    after = len(df)
    if before != after:
        print(f"Dropped {before - after} duplicate customerID rows")

    return df


def validate(df: pd.DataFrame) -> None:
    assert df["TotalCharges"].isna().sum() == 0, "TotalCharges still has NaNs"
    assert df["Churn_Flag"].isna().sum() == 0, "Churn_Flag mapping failed"
    assert df["customerID"].is_unique, "Duplicate customer IDs remain"
    print("Validation passed: no missing TotalCharges/Churn, IDs unique.")


def main():
    df_raw = load_raw(RAW_PATH)
    df_clean = clean(df_raw)
    validate(df_clean)

    df_clean.to_csv(OUT_PATH, index=False)
    print(f"\nSaved cleaned dataset -> {OUT_PATH}")
    print(f"Final shape: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
    print(f"Overall churn rate: {df_clean['Churn_Flag'].mean():.2%}")


if __name__ == "__main__":
    main()
