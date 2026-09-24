"""
04_predictive_modeling.py
--------------------------
Customer Churn & Retention Analytics — Predictive Modeling

Trains two models to predict churn:
  1. Logistic Regression (interpretable baseline)
  2. Random Forest (captures non-linear interactions, gives feature importance)

Evaluates both on a held-out test set (accuracy, precision, recall, F1, ROC-AUC),
saves a comparison chart, a confusion matrix, an ROC curve, and a feature
importance chart, and writes each customer's predicted churn probability
to data/processed/ for use in retention targeting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

IN_PATH = "../data/processed/telco_churn_clean.csv"
FIG_DIR = "../outputs/figures"
REPORT_DIR = "../outputs/reports"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY_TEXT_PRI = "#0b0b0b"
GRAY_TEXT_SEC = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": GRAY_TEXT_SEC,
    "text.color": GRAY_TEXT_PRI,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

RANDOM_STATE = 42


def load_and_prepare(path):
    df = pd.read_csv(path)

    # Features: drop identifiers, leakage-prone / redundant engineered cols, and target
    drop_cols = [
        "customerID", "Churn", "Churn_Flag", "TenureGroup", "AvgMonthlySpend",
        # drop raw service cols that have simplified counterparts to avoid duplication
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "MultipleLines",
    ]
    y = df["Churn_Flag"]
    X = df.drop(columns=drop_cols)

    # One-hot encode categoricals
    X_encoded = pd.get_dummies(X, drop_first=True)

    return X_encoded, y, df


def main():
    X, y, df_full = load_and_prepare(IN_PATH)
    print(f"Feature matrix: {X.shape[0]} rows, {X.shape[1]} features (after one-hot encoding)")

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df_full.index, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    print(f"Train churn rate: {y_train.mean():.2%} | Test churn rate: {y_test.mean():.2%}\n")

    # --- Scale features for Logistic Regression ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Model 1: Logistic Regression ---
    log_reg = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    log_reg.fit(X_train_scaled, y_train)
    y_pred_lr = log_reg.predict(X_test_scaled)
    y_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

    # --- Model 2: Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]

    # --- Evaluate ---
    def evaluate(name, y_true, y_pred, y_proba):
        return {
            "model": name,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_proba),
        }

    metrics = [
        evaluate("Logistic Regression", y_test, y_pred_lr, y_proba_lr),
        evaluate("Random Forest", y_test, y_pred_rf, y_proba_rf),
    ]
    metrics_df = pd.DataFrame(metrics)
    print("=== Model comparison ===")
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv(f"{REPORT_DIR}/model_comparison.csv", index=False)

    print("\n=== Classification report: Random Forest ===")
    print(classification_report(y_test, y_pred_rf, target_names=["Retained", "Churned"]))

    # --- Chart: Model comparison bar chart ---
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(metric_names))
    width = 0.35
    lr_vals = metrics_df.loc[metrics_df.model == "Logistic Regression", metric_names].values.flatten()
    rf_vals = metrics_df.loc[metrics_df.model == "Random Forest", metric_names].values.flatten()
    ax.bar(x - width/2, lr_vals, width, label="Logistic Regression", color=BLUE, zorder=3)
    ax.bar(x + width/2, rf_vals, width, label="Random Forest", color=ORANGE, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
    ax.set_ylim(0, 1.0)
    ax.set_title("Model performance comparison", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=False, labelcolor=GRAY_TEXT_SEC)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/08_model_comparison.png", dpi=150)
    plt.close(fig)
    print("\nSaved 08_model_comparison.png")

    # --- Chart: ROC curves ---
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, y_proba, color in [
        ("Logistic Regression", y_proba_lr, BLUE),
        ("Random Forest", y_proba_rf, ORANGE),
    ]:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC={auc:.3f})", zorder=3)
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, zorder=2)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=False, labelcolor=GRAY_TEXT_SEC, loc="lower right")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/09_roc_curve.png", dpi=150)
    plt.close(fig)
    print("Saved 09_roc_curve.png")

    # --- Chart: Confusion matrix (Random Forest) ---
    cm = confusion_matrix(y_test, y_pred_rf)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["Retained", "Churned"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix — Random Forest", fontsize=12, fontweight="bold", pad=12)
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else GRAY_TEXT_PRI
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14, fontweight="bold")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/10_confusion_matrix.png", dpi=150)
    plt.close(fig)
    print("Saved 10_confusion_matrix.png")

    # --- Chart: Feature importance (Random Forest, top 12) ---
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True).tail(12)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(importances.index, importances.values, color=BLUE, zorder=3)
    ax.set_xlabel("Feature importance")
    ax.set_title("Top drivers of churn (Random Forest)",
                 fontsize=12, fontweight="bold", pad=12)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/11_feature_importance.png", dpi=150)
    plt.close(fig)
    print("Saved 11_feature_importance.png")

    # --- Save per-customer churn probability (for retention targeting) ---
    all_proba = rf.predict_proba(X)[:, 1]
    scored = df_full[["customerID", "Contract", "tenure", "MonthlyCharges", "Churn"]].copy()
    scored["Predicted_Churn_Probability"] = all_proba
    scored["Risk_Tier"] = pd.cut(
        all_proba, bins=[-0.01, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"]
    )
    scored = scored.sort_values("Predicted_Churn_Probability", ascending=False)
    scored.to_csv(f"{REPORT_DIR}/customer_churn_risk_scores.csv", index=False)
    print(f"\nSaved customer_churn_risk_scores.csv ({len(scored)} customers scored)")
    print(scored["Risk_Tier"].value_counts())

    return metrics_df, importances


if __name__ == "__main__":
    main()
