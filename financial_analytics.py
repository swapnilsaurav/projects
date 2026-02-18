"""
Financial Management Analytics (Answers 15 Business Questions)
Dataset: /mnt/data/Financial_Management_Dataset.csv

Outputs:
- Creates ./fin_reports/ with multiple CSV reports and optional plots.

How to run:
1) Ensure pandas, numpy, matplotlib, statsmodels installed.
2) python financial_analytics.py
"""

from __future__ import annotations
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

# Optional forecasting
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

# Optional plotting
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# -----------------------------
# CONFIG
# -----------------------------
CSV_PATH = "/mnt/data/Financial_Management_Dataset.csv"
OUTDIR = Path("./fin_reports")

ANOMALY_Z = 2.0                 # Q11: abnormal amount threshold
LARGE_TXN_QUANTILE = 0.99       # Q6: liquidity risk "large withdrawals/deposits"
BUDGET_MULTIPLIER = 1.10        # Q2: budget baseline * multiplier (derived budget)
FORECAST_STEPS = 1              # Q12: next month forecast horizon

# You can tune this list based on your org policy (Q9)
NON_OPERATIONAL_CATEGORIES = {"Maintenance", "Travel", "Ad-hoc Expenses"}


# -----------------------------
# HELPERS
# -----------------------------
def ensure_outdir() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "Transaction ID", "Date", "Account Name", "Department", "Transaction Type",
        "Category", "Amount", "Currency", "Approved By"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        bad = df[df["Date"].isna()].head(5)
        raise ValueError(f"Found invalid Date rows. Sample:\n{bad}")

    df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)
    df["Quarter"] = df["Date"].dt.to_period("Q").astype(str)

    ttype = df["Transaction Type"].astype(str).str.strip().str.lower()
    df["IsCredit"] = ttype.eq("credit")
    df["IsDebit"] = ttype.eq("debit")

    df["SignedAmount"] = np.where(df["IsCredit"], df["Amount"], -df["Amount"])
    return df

def save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUTDIR / name, index=False)

def safe_slope(y: np.ndarray) -> float:
    """Trend slope via simple linear fit; returns 0 if insufficient points."""
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


# -----------------------------
# Q1, Q3: Monthly trends, avg monthly expenditure per account
# -----------------------------
def q1_monthly_trends_spend_income(df: pd.DataFrame) -> pd.DataFrame:
    # monthly spend/income per department
    g = df.groupby(["YearMonth", "Department"])
    out = g.apply(
        lambda x: pd.Series({
            "Income": x.loc[x["IsCredit"], "Amount"].sum(),
            "Spending": x.loc[x["IsDebit"], "Amount"].sum(),
            "NetCashFlow": x["SignedAmount"].sum()
        })
    ).reset_index()
    out = out.sort_values(["YearMonth", "Department"])
    save(out, "Q1_monthly_trends_by_department.csv")
    return out

def q3_avg_monthly_expenditure_per_account(df: pd.DataFrame) -> pd.DataFrame:
    # average monthly debit per account
    deb = df[df["IsDebit"]].copy()
    monthly = deb.groupby(["YearMonth", "Account Name"])["Amount"].sum().reset_index()
    avg = monthly.groupby("Account Name")["Amount"].mean().reset_index()
    avg = avg.rename(columns={"Amount": "AvgMonthlyExpenditure"})
    avg = avg.sort_values("AvgMonthlyExpenditure", ascending=False)
    save(avg, "Q3_avg_monthly_expenditure_per_account.csv")
    return avg


# -----------------------------
# Q2: Budget overshoot (derived budgets, since dataset has no Budget column)
# -----------------------------
def q2_budget_overshoot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dataset doesn't contain explicit budgets, so we *derive* a budget baseline:
    - For each Department + Category, compute monthly debit.
    - BudgetBaseline = (historical mean monthly debit) * BUDGET_MULTIPLIER
    - Flag any month where monthly debit > BudgetBaseline
    """
    deb = df[df["IsDebit"]].copy()

    monthly = (deb.groupby(["YearMonth", "Department", "Category"])["Amount"]
               .sum().reset_index(name="MonthlyDebit"))

    baseline = (monthly.groupby(["Department", "Category"])["MonthlyDebit"]
                .mean().reset_index(name="MeanMonthlyDebit"))

    baseline["BudgetBaseline"] = baseline["MeanMonthlyDebit"] * BUDGET_MULTIPLIER

    merged = monthly.merge(baseline[["Department", "Category", "BudgetBaseline"]],
                           on=["Department", "Category"], how="left")

    merged["Overshoot"] = merged["MonthlyDebit"] > merged["BudgetBaseline"]
    overs = merged[merged["Overshoot"]].sort_values(["Department", "Category", "YearMonth"])

    save(merged.sort_values(["Department", "Category", "YearMonth"]),
         "Q2_budget_baseline_and_monthly_debit.csv")
    save(overs, "Q2_budget_overshoots.csv")
    return overs


# -----------------------------
# Q4: Net cash flow per month/quarter + by department/account
# -----------------------------
def q4_net_cashflow(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    by_month = df.groupby("YearMonth")["SignedAmount"].sum().reset_index(name="NetCashFlow")
    by_quarter = df.groupby("Quarter")["SignedAmount"].sum().reset_index(name="NetCashFlow")

    by_month_dept = df.groupby(["YearMonth", "Department"])["SignedAmount"].sum().reset_index(name="NetCashFlow")
    by_month_acct = df.groupby(["YearMonth", "Account Name"])["SignedAmount"].sum().reset_index(name="NetCashFlow")

    save(by_month, "Q4_net_cashflow_by_month.csv")
    save(by_quarter, "Q4_net_cashflow_by_quarter.csv")
    save(by_month_dept, "Q4_net_cashflow_by_month_department.csv")
    save(by_month_acct, "Q4_net_cashflow_by_month_account.csv")

    return {
        "month": by_month,
        "quarter": by_quarter,
        "month_dept": by_month_dept,
        "month_account": by_month_acct
    }


# -----------------------------
# Q5: Categories contributing most to inflow/outflow
# -----------------------------
def q5_top_inflow_outflow_categories(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    inflow = (df[df["IsCredit"]]
              .groupby("Category")["Amount"].sum()
              .reset_index(name="TotalInflow")
              .sort_values("TotalInflow", ascending=False))

    outflow = (df[df["IsDebit"]]
               .groupby("Category")["Amount"].sum()
               .reset_index(name="TotalOutflow")
               .sort_values("TotalOutflow", ascending=False))

    save(inflow, "Q5_top_inflow_categories.csv")
    save(outflow, "Q5_top_outflow_categories.csv")
    return {"inflow": inflow, "outflow": outflow}


# -----------------------------
# Q6: Large withdrawals/deposits pattern (liquidity risk proxy)
# -----------------------------
def q6_liquidity_risk_patterns(df: pd.DataFrame) -> pd.DataFrame:
    # Identify large credits and debits via high quantile threshold
    credit_thresh = df.loc[df["IsCredit"], "Amount"].quantile(LARGE_TXN_QUANTILE)
    debit_thresh = df.loc[df["IsDebit"], "Amount"].quantile(LARGE_TXN_QUANTILE)

    large = df[
        (df["IsCredit"] & (df["Amount"] >= credit_thresh)) |
        (df["IsDebit"] & (df["Amount"] >= debit_thresh))
    ].copy()

    # Add some helpful grouping fields
    large["LargeType"] = np.where(large["IsCredit"], "LargeDeposit", "LargeWithdrawal")
    large = large.sort_values(["Date", "Amount"], ascending=[True, False])

    save(large, "Q6_large_transactions_liquidity_risk.csv")
    return large


# -----------------------------
# Q7: Category trends up/down
# -----------------------------
def q7_category_expense_trends(df: pd.DataFrame) -> pd.DataFrame:
    deb = df[df["IsDebit"]].copy()
    monthly_cat = deb.groupby(["YearMonth", "Category"])["Amount"].sum().reset_index()

    # Compute slope per category over months
    trend_rows = []
    for cat, sub in monthly_cat.groupby("Category"):
        sub_sorted = sub.sort_values("YearMonth")
        slope = safe_slope(sub_sorted["Amount"].values)
        trend_rows.append({"Category": cat, "MonthlyTrendSlope": slope})

    trends = pd.DataFrame(trend_rows).sort_values("MonthlyTrendSlope", ascending=False)
    trends["Trend"] = np.where(trends["MonthlyTrendSlope"] > 0, "Upward",
                        np.where(trends["MonthlyTrendSlope"] < 0, "Downward", "Flat"))

    save(monthly_cat.sort_values(["Category", "YearMonth"]), "Q7_monthly_expense_by_category.csv")
    save(trends, "Q7_category_expense_trend_slopes.csv")
    return trends


# -----------------------------
# Q8: Top approvers by volume and amount
# -----------------------------
def q8_top_approvers(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Approved By").agg(
        ApprovalCount=("Transaction ID", "count"),
        TotalAmount=("Amount", "sum"),
        DebitAmount=("Amount", lambda x: x[df.loc[x.index, "IsDebit"]].sum()),
        CreditAmount=("Amount", lambda x: x[df.loc[x.index, "IsCredit"]].sum()),
    ).reset_index()

    g = g.sort_values(["ApprovalCount", "TotalAmount"], ascending=False)
    save(g, "Q8_top_approvers_volume_and_amount.csv")
    return g


# -----------------------------
# Q9: % spend in non-operational categories
# -----------------------------
def q9_non_operational_spend_share(df: pd.DataFrame) -> pd.DataFrame:
    deb = df[df["IsDebit"]].copy()
    total_spend = deb["Amount"].sum()

    deb["IsNonOperational"] = deb["Category"].isin(NON_OPERATIONAL_CATEGORIES)
    nonop_spend = deb.loc[deb["IsNonOperational"], "Amount"].sum()

    by_cat = (deb.groupby("Category")["Amount"].sum()
              .reset_index(name="TotalDebit")
              .sort_values("TotalDebit", ascending=False))

    summary = pd.DataFrame([{
        "TotalDebitSpend": total_spend,
        "NonOperationalDebitSpend": nonop_spend,
        "NonOperationalSpendPct": (nonop_spend / total_spend * 100.0) if total_spend else 0.0
    }])

    save(summary, "Q9_non_operational_spend_summary.csv")
    save(by_cat, "Q9_spend_by_category_all_debits.csv")
    return summary


# -----------------------------
# Q10: Departments with unusual transaction frequency or high amounts
# -----------------------------
def q10_department_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    # Frequency per dept per month + z-score within dept-month counts
    freq = (df.groupby(["YearMonth", "Department"])["Transaction ID"]
            .count().reset_index(name="TxnCount"))

    # Dept amount totals per month (absolute outflow+inflow)
    amt = (df.groupby(["YearMonth", "Department"])["Amount"]
           .sum().reset_index(name="TotalAmount"))

    merged = freq.merge(amt, on=["YearMonth", "Department"], how="left")

    # Z-scores across all dept-month points (simple global)
    merged["TxnCountZ"] = (merged["TxnCount"] - merged["TxnCount"].mean()) / (merged["TxnCount"].std(ddof=0) + 1e-9)
    merged["TotalAmountZ"] = (merged["TotalAmount"] - merged["TotalAmount"].mean()) / (merged["TotalAmount"].std(ddof=0) + 1e-9)

    flagged = merged[(merged["TxnCountZ"].abs() > ANOMALY_Z) | (merged["TotalAmountZ"].abs() > ANOMALY_Z)]
    flagged = flagged.sort_values(["YearMonth", "TxnCountZ", "TotalAmountZ"], ascending=[True, False, False])

    save(merged.sort_values(["YearMonth", "Department"]), "Q10_dept_month_frequency_and_amount.csv")
    save(flagged, "Q10_flagged_departments_unusual_activity.csv")
    return flagged


# -----------------------------
# Q11: Abnormal transactions > 2 std dev from mean (within category & type)
# -----------------------------
def q11_abnormal_transactions(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    # Compute z-score within (Category, Transaction Type)
    tmp["GroupMean"] = tmp.groupby(["Category", "Transaction Type"])["Amount"].transform("mean")
    tmp["GroupStd"] = tmp.groupby(["Category", "Transaction Type"])["Amount"].transform("std").fillna(0.0)
    tmp["Z"] = (tmp["Amount"] - tmp["GroupMean"]) / (tmp["GroupStd"] + 1e-9)

    outliers = tmp[tmp["Z"].abs() > ANOMALY_Z].copy()
    outliers = outliers.sort_values(["Z"], ascending=False)

    cols = ["Transaction ID", "Date", "Department", "Account Name", "Transaction Type",
            "Category", "Amount", "Currency", "Approved By", "Z"]
    save(outliers[cols], "Q11_abnormal_transactions_zscore.csv")
    return outliers[cols]


# -----------------------------
# Q12: Forecast next month spending (overall debit) using SARIMAX or fallback
# -----------------------------
def q12_forecast_next_month_spend(df: pd.DataFrame) -> pd.DataFrame:
    deb = df[df["IsDebit"]].copy()
    monthly = deb.groupby("YearMonth")["Amount"].sum().reset_index()
    monthly["YearMonth"] = pd.PeriodIndex(monthly["YearMonth"], freq="M").to_timestamp()
    monthly = monthly.sort_values("YearMonth")

    if len(monthly) < 6:
        # Not enough history; naive forecast
        forecast_val = float(monthly["Amount"].iloc[-1]) if len(monthly) else 0.0
        method = "Naive (last month carry-forward)"
    else:
        if HAS_STATSMODELS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y = monthly.set_index("YearMonth")["Amount"]

                # Simple SARIMAX
                try:
                    model = sm.tsa.SARIMAX(y, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0),
                                           enforce_stationarity=False, enforce_invertibility=False)
                    res = model.fit(disp=False)
                    pred = res.forecast(steps=FORECAST_STEPS)
                    forecast_val = float(pred.iloc[-1])
                    method = "SARIMAX(1,1,1)"
                except Exception:
                    forecast_val = float(y.iloc[-1])
                    method = "Naive (fallback due to SARIMAX error)"
        else:
            forecast_val = float(monthly["Amount"].iloc[-1])
            method = "Naive (statsmodels not installed)"

    last_month = monthly["YearMonth"].iloc[-1] if len(monthly) else pd.Timestamp.today()
    next_month = (pd.Period(last_month, freq="M") + 1).to_timestamp()

    out = pd.DataFrame([{
        "LastObservedMonth": last_month.strftime("%Y-%m"),
        "ForecastMonth": next_month.strftime("%Y-%m"),
        "ForecastDebitSpending": forecast_val,
        "Method": method
    }])

    save(monthly.rename(columns={"YearMonth": "Month", "Amount": "TotalDebitSpending"}),
         "Q12_monthly_debit_spending_series.csv")
    save(out, "Q12_next_month_spend_forecast.csv")
    return out


# -----------------------------
# Q13: Credit/Debit vary over time across accounts or currencies
# -----------------------------
def q13_credit_debit_over_time(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # Over time by account
    by_acct = df.groupby(["YearMonth", "Account Name"]).apply(
        lambda x: pd.Series({
            "TotalCredit": x.loc[x["IsCredit"], "Amount"].sum(),
            "TotalDebit": x.loc[x["IsDebit"], "Amount"].sum(),
            "Net": x["SignedAmount"].sum()
        })
    ).reset_index()

    # Over time by currency
    by_cur = df.groupby(["YearMonth", "Currency"]).apply(
        lambda x: pd.Series({
            "TotalCredit": x.loc[x["IsCredit"], "Amount"].sum(),
            "TotalDebit": x.loc[x["IsDebit"], "Amount"].sum(),
            "Net": x["SignedAmount"].sum()
        })
    ).reset_index()

    save(by_acct.sort_values(["YearMonth", "Account Name"]), "Q13_credit_debit_over_time_by_account.csv")
    save(by_cur.sort_values(["YearMonth", "Currency"]), "Q13_credit_debit_over_time_by_currency.csv")
    return {"by_account": by_acct, "by_currency": by_cur}


# -----------------------------
# Q14: Approvers consistently approving high-amount txns in specific depts
# -----------------------------
def q14_approver_high_amount_patterns(df: pd.DataFrame) -> pd.DataFrame:
    # Define "high amount" relative to overall distribution (e.g., top 10%)
    thresh = df["Amount"].quantile(0.90)
    high = df[df["Amount"] >= thresh].copy()

    out = high.groupby(["Approved By", "Department"]).agg(
        HighTxnCount=("Transaction ID", "count"),
        HighTxnTotal=("Amount", "sum"),
        AvgHighTxn=("Amount", "mean"),
        MaxHighTxn=("Amount", "max")
    ).reset_index().sort_values(["HighTxnTotal", "HighTxnCount"], ascending=False)

    save(out, "Q14_approver_high_amount_by_department.csv")
    return out


# -----------------------------
# Q15: Distribution of Credit/Debit by category + policy alignment flags
# -----------------------------
def q15_distribution_and_policy(df: pd.DataFrame) -> pd.DataFrame:
    pivot = (df.pivot_table(index="Category",
                            columns="Transaction Type",
                            values="Amount",
                            aggfunc="sum",
                            fill_value=0.0)
             .reset_index())

    # Normalize to shares
    cols = [c for c in pivot.columns if c != "Category"]
    pivot["Total"] = pivot[cols].sum(axis=1)
    for c in cols:
        pivot[f"{c}_Share"] = np.where(pivot["Total"] > 0, pivot[c] / pivot["Total"], 0.0)

    # Simple policy check example:
    # If a category is expected to be mostly Debit (expenses),
    # but Credit share is high, flag it. (Adjust logic as needed.)
    credit_col = None
    for c in cols:
        if str(c).strip().lower() == "credit":
            credit_col = c
            break

    if credit_col:
        pivot["PolicyFlag"] = np.where(pivot[f"{credit_col}_Share"] > 0.40,
                                       "Review (high credit share)",
                                       "OK")
    else:
        pivot["PolicyFlag"] = "No Credit column found"

    pivot = pivot.sort_values("Total", ascending=False)
    save(pivot, "Q15_txn_type_distribution_by_category.csv")
    return pivot


# -----------------------------
# OPTIONAL: plots
# -----------------------------
def make_basic_plots(df: pd.DataFrame) -> None:
    if not HAS_MPL:
        return

    # Plot 1: Net cashflow by month
    net = df.groupby("YearMonth")["SignedAmount"].sum().reset_index()
    plt.figure()
    plt.plot(net["YearMonth"], net["SignedAmount"])
    plt.xticks(rotation=45, ha="right")
    plt.title("Net Cash Flow by Month (Credit - Debit)")
    plt.tight_layout()
    plt.savefig(OUTDIR / "plot_net_cashflow_by_month.png", dpi=150)
    plt.close()

    # Plot 2: Monthly debit spending by category (top 6 categories)
    deb = df[df["IsDebit"]].copy()
    monthly_cat = deb.groupby(["YearMonth", "Category"])["Amount"].sum().reset_index()
    top_cats = (deb.groupby("Category")["Amount"].sum()
               .sort_values(ascending=False).head(6).index.tolist())
    mc = monthly_cat[monthly_cat["Category"].isin(top_cats)].copy()

    plt.figure()
    for cat in top_cats:
        sub = mc[mc["Category"] == cat]
        plt.plot(sub["YearMonth"], sub["Amount"], label=cat)
    plt.xticks(rotation=45, ha="right")
    plt.title("Monthly Debit Spending (Top Categories)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "plot_monthly_debit_top_categories.png", dpi=150)
    plt.close()


# -----------------------------
# MAIN
# -----------------------------
def main() -> None:
    ensure_outdir()
    df = load_data(CSV_PATH)

    # Answer all questions
    q1_monthly_trends_spend_income(df)          # Q1
    q2_budget_overshoot(df)                     # Q2 (derived)
    q3_avg_monthly_expenditure_per_account(df)  # Q3

    q4_net_cashflow(df)                         # Q4
    q5_top_inflow_outflow_categories(df)        # Q5
    q6_liquidity_risk_patterns(df)              # Q6

    q7_category_expense_trends(df)              # Q7
    q8_top_approvers(df)                        # Q8
    q9_non_operational_spend_share(df)          # Q9

    q10_department_anomalies(df)                # Q10
    q11_abnormal_transactions(df)               # Q11

    q12_forecast_next_month_spend(df)           # Q12
    q13_credit_debit_over_time(df)              # Q13

    q14_approver_high_amount_patterns(df)       # Q14
    q15_distribution_and_policy(df)             # Q15

    # Optional plots
    make_basic_plots(df)

    print(f"Done. Reports saved to: {OUTDIR.resolve()}")

if __name__ == "__main__":
    main()
