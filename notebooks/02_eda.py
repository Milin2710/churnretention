"""
02_eda.py
---------
Customer Churn & Retention Analytics — Exploratory Data Analysis

Produces churn-rate breakdowns by key segments and saves charts to
outputs/figures/. Uses the validated dataviz reference palette.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

IN_PATH = "../data/processed/telco_churn_clean.csv"
FIG_DIR = "../outputs/figures"

# --- Reference palette (dataviz skill, light mode) ---
BLUE = "#2a78d6"      # series 1 - "Churned" / primary
ORANGE = "#eb6834"    # series 2 - secondary category
GRAY_TEXT_PRI = "#0b0b0b"
GRAY_TEXT_SEC = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
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


def pct_fmt(ax, axis="y"):
    fmt = mticker.PercentFormatter(xmax=1.0, decimals=0)
    if axis == "y":
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)


def churn_rate_by(df, col, order=None, title="", fname="", horizontal=False, figsize=(7, 4.5)):
    rate = df.groupby(col, observed=True)["Churn_Flag"].mean()
    if order is not None:
        rate = rate.reindex(order)
    else:
        rate = rate.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    overall = df["Churn_Flag"].mean()

    if horizontal:
        bars = ax.barh(rate.index.astype(str), rate.values, color=BLUE,
                        height=0.6, zorder=3)
        ax.axvline(overall, color=MUTED, linestyle="--", linewidth=1, zorder=2)
        ax.text(overall, len(rate) - 0.4, f" Overall {overall:.0%}",
                color=GRAY_TEXT_SEC, fontsize=9, va="center")
        pct_fmt(ax, axis="x")
        for bar, val in zip(bars, rate.values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{val:.0%}", va="center", fontsize=9, color=GRAY_TEXT_PRI)
        ax.set_xlabel("Churn rate")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    else:
        bars = ax.bar(rate.index.astype(str), rate.values, color=BLUE,
                       width=0.6, zorder=3)
        ax.axhline(overall, color=MUTED, linestyle="--", linewidth=1, zorder=2)
        ax.text(len(rate) - 0.5, overall, f"Overall {overall:.0%}",
                color=GRAY_TEXT_SEC, fontsize=9, va="bottom", ha="right")
        pct_fmt(ax, axis="y")
        for bar, val in zip(bars, rate.values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                     f"{val:.0%}", ha="center", fontsize=9, color=GRAY_TEXT_PRI)
        ax.set_ylabel("Churn rate")
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)

    ax.set_title(title, fontsize=12, fontweight="bold", color=GRAY_TEXT_PRI, pad=12)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/{fname}", dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")


def main():
    df = pd.read_csv(IN_PATH)

    overall = df["Churn_Flag"].mean()
    print(f"Overall churn rate: {overall:.2%}\n")

    # 1. Churn by Contract type
    churn_rate_by(
        df, "Contract",
        order=["Month-to-month", "One year", "Two year"],
        title="Churn rate by contract type",
        fname="01_churn_by_contract.png",
    )

    # 2. Churn by Internet Service
    churn_rate_by(
        df, "InternetService",
        title="Churn rate by internet service",
        fname="02_churn_by_internet_service.png",
    )

    # 3. Churn by Payment Method
    churn_rate_by(
        df, "PaymentMethod",
        title="Churn rate by payment method",
        fname="03_churn_by_payment_method.png",
        horizontal=True,
        figsize=(7.5, 4.5),
    )

    # 4. Churn by Tenure Group (retention curve style)
    churn_rate_by(
        df, "TenureGroup",
        order=["0-6 mo", "7-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"],
        title="Churn rate by tenure cohort",
        fname="04_churn_by_tenure_group.png",
    )

    # 5. Churn by number of add-on services (engagement proxy)
    churn_rate_by(
        df, "NumAddonServices",
        title="Churn rate by number of add-on services",
        fname="05_churn_by_addon_count.png",
    )

    # 6. Monthly Charges distribution: churned vs retained (overlaid density)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, color, name in [(0, "#898781", "Retained"), (1, BLUE, "Churned")]:
        subset = df.loc[df["Churn_Flag"] == label, "MonthlyCharges"]
        ax.hist(subset, bins=30, alpha=0.55, color=color, label=name,
                density=True, zorder=3)
    ax.set_xlabel("Monthly charges ($)")
    ax.set_ylabel("Density")
    ax.set_title("Monthly charges: churned vs. retained customers",
                 fontsize=12, fontweight="bold", color=GRAY_TEXT_PRI, pad=12)
    ax.legend(frameon=False, labelcolor=GRAY_TEXT_SEC)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/06_monthly_charges_distribution.png", dpi=150)
    plt.close(fig)
    print("Saved 06_monthly_charges_distribution.png")

    # 7. Churn by Senior Citizen status x Contract (grouped bar, 2 series)
    pivot = df.groupby(["Contract", "SeniorCitizen"], observed=True)["Churn_Flag"].mean().unstack()
    pivot = pivot.reindex(["Month-to-month", "One year", "Two year"])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = range(len(pivot))
    width = 0.35
    ax.bar([i - width/2 for i in x], pivot["No"], width, label="Not senior", color=BLUE, zorder=3)
    ax.bar([i + width/2 for i in x], pivot["Yes"], width, label="Senior citizen", color=ORANGE, zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(pivot.index)
    pct_fmt(ax, axis="y")
    ax.set_ylabel("Churn rate")
    ax.set_title("Churn rate by contract type and senior citizen status",
                 fontsize=12, fontweight="bold", color=GRAY_TEXT_PRI, pad=12)
    ax.legend(frameon=False, labelcolor=GRAY_TEXT_SEC)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/07_churn_contract_x_senior.png", dpi=150)
    plt.close(fig)
    print("Saved 07_churn_contract_x_senior.png")

    # --- Summary table for report ---
    summary = {
        "Overall churn rate": f"{overall:.2%}",
        "Month-to-month churn rate": f"{df[df.Contract=='Month-to-month']['Churn_Flag'].mean():.2%}",
        "Two-year contract churn rate": f"{df[df.Contract=='Two year']['Churn_Flag'].mean():.2%}",
        "Fiber optic churn rate": f"{df[df.InternetService=='Fiber optic']['Churn_Flag'].mean():.2%}",
        "0-6 month tenure churn rate": f"{df[df.TenureGroup=='0-6 mo']['Churn_Flag'].mean():.2%}",
        "49-72 month tenure churn rate": f"{df[df.TenureGroup=='49-72 mo']['Churn_Flag'].mean():.2%}",
        "Electronic check churn rate": f"{df[df.PaymentMethod=='Electronic check']['Churn_Flag'].mean():.2%}",
    }
    pd.Series(summary).to_csv("../outputs/reports/eda_summary_stats.csv", header=["value"])
    print("\nSaved eda_summary_stats.csv")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
