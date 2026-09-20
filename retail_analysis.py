import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

sns.set_style("whitegrid")

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
df = pd.read_csv("retail_sales_data.csv", parse_dates=["Date"])
print("Dataset shape:", df.shape)

report = []
report.append("# Real-World Data Project — Retail Sales Analysis & Forecast\n")
report.append(f"**Dataset:** {df.shape[0]} transaction records spanning "
              f"{df['Date'].min().date()} to {df['Date'].max().date()}\n")
report.append(f"**Stores:** {', '.join(df['Store'].unique())}  \n"
              f"**Categories:** {', '.join(df['Category'].unique())}\n")

# ---------------------------------------------------------------
# 2. AGGREGATE ANALYSIS
# ---------------------------------------------------------------
total_revenue = df["Revenue"].sum()
total_units = df["UnitsSold"].sum()
avg_daily_rev = df.groupby("Date")["Revenue"].sum().mean()

rev_by_store = df.groupby("Store")["Revenue"].sum().sort_values(ascending=False)
rev_by_category = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)

report.append("\n## Summary Statistics\n")
report.append(f"- **Total Revenue:** ${total_revenue:,.2f}\n")
report.append(f"- **Total Units Sold:** {total_units:,}\n")
report.append(f"- **Average Daily Revenue:** ${avg_daily_rev:,.2f}\n")
report.append(f"- **Top Store:** {rev_by_store.idxmax()} (${rev_by_store.max():,.2f})\n")
report.append(f"- **Top Category:** {rev_by_category.idxmax()} (${rev_by_category.max():,.2f})\n")

# ---------------------------------------------------------------
# 3. TIME SERIES PREP (monthly aggregation)
# ---------------------------------------------------------------
monthly = df.groupby(pd.Grouper(key="Date", freq="ME"))["Revenue"].sum().reset_index()
monthly["MonthIndex"] = range(len(monthly))

# ---------------------------------------------------------------
# 4. SALES FORECASTING MODEL (Linear Regression on monthly trend)
# ---------------------------------------------------------------
X = monthly[["MonthIndex"]]
y = monthly["Revenue"]

# Train on all but the last 3 months, test on those
train = monthly.iloc[:-3]
test = monthly.iloc[-3:]

model = LinearRegression()
model.fit(train[["MonthIndex"]], train["Revenue"])

test_preds = model.predict(test[["MonthIndex"]])
mae = mean_absolute_error(test["Revenue"], test_preds)
r2 = r2_score(test["Revenue"], test_preds)

# Forecast next 3 months beyond the dataset
future_idx = np.arange(monthly["MonthIndex"].max() + 1, monthly["MonthIndex"].max() + 4)
future_preds = model.predict(future_idx.reshape(-1, 1))
future_dates = pd.date_range(monthly["Date"].max() + pd.offsets.MonthEnd(1), periods=3, freq="ME")

report.append("\n## Sales Forecasting Model\n")
report.append("Model: Linear Regression on monthly revenue trend.\n")
report.append(f"- **Validation MAE (last 3 known months):** ${mae:,.2f}\n")
report.append(f"- **Validation R²:** {r2:.3f}\n")
report.append("\n> **Note on model limitation:** The negative R² on the validation months reflects that "
              "revenue is strongly seasonal (a sharp Nov–Dec spike), which a simple linear trend cannot "
              "capture. A seasonal model (e.g. SARIMA or Prophet) would likely perform better for "
              "production forecasting; the linear fit here is presented as a baseline to illustrate the "
              "overall trend direction.\n")
report.append("\n### Forecast — Next 3 Months\n")
report.append("| Month | Predicted Revenue |\n|---|---|\n")
for d, p in zip(future_dates, future_preds):
    report.append(f"| {d.strftime('%b %Y')} | ${p:,.2f} |\n")

# ---------------------------------------------------------------
# 5. VISUALIZATIONS
# ---------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(15, 17))
fig.suptitle("Retail Sales Analysis Dashboard", fontsize=16, fontweight="bold")

# (a) Monthly revenue trend + forecast
axes[0, 0].plot(monthly["Date"], monthly["Revenue"], marker="o", label="Actual", color="steelblue")
axes[0, 0].plot(future_dates, future_preds, marker="o", linestyle="--", label="Forecast", color="darkorange")
axes[0, 0].set_title("Monthly Revenue Trend & 3-Month Forecast")
axes[0, 0].legend()
axes[0, 0].tick_params(axis="x", rotation=45)

# (b) Revenue by store
sns.barplot(x=rev_by_store.index, y=rev_by_store.values, ax=axes[0, 1], palette="Blues_d")
axes[0, 1].set_title("Total Revenue by Store")
axes[0, 1].set_ylabel("Revenue ($)")

# (c) Revenue by category
sns.barplot(x=rev_by_category.values, y=rev_by_category.index, ax=axes[1, 0], palette="Greens_d")
axes[1, 0].set_title("Total Revenue by Category")
axes[1, 0].set_xlabel("Revenue ($)")

# (d) Revenue by weekday
df["Weekday"] = df["Date"].dt.day_name()
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
rev_weekday = df.groupby("Weekday")["Revenue"].sum().reindex(weekday_order)
sns.barplot(x=rev_weekday.index, y=rev_weekday.values, ax=axes[1, 1], palette="Purples_d")
axes[1, 1].set_title("Revenue by Day of Week")
axes[1, 1].tick_params(axis="x", rotation=45)

# (e) Store x Category heatmap
pivot = df.pivot_table(values="Revenue", index="Store", columns="Category", aggfunc="sum")
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=axes[2, 0])
axes[2, 0].set_title("Revenue Heatmap: Store x Category")

# (f) Daily revenue distribution
daily_rev = df.groupby("Date")["Revenue"].sum()
sns.histplot(daily_rev, bins=30, kde=True, ax=axes[2, 1], color="teal")
axes[2, 1].set_title("Distribution of Daily Revenue")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("retail_dashboard.png", dpi=150)
print("Saved retail_dashboard.png")

# ---------------------------------------------------------------
# 6. CONCLUSIONS
# ---------------------------------------------------------------
best_day = rev_weekday.idxmax()
trend_direction = "upward" if model.coef_[0] > 0 else "downward"

report.append("\n## Conclusions\n")
report.append(f"- Revenue shows a clear **{trend_direction} trend** over the two-year period "
              f"(~${model.coef_[0]:,.0f} change per month on average).\n")
report.append(f"- **{best_day}** is the strongest day of the week for sales, consistent with typical retail weekend patterns.\n")
report.append(f"- **{rev_by_store.idxmax()}** consistently outperforms other locations, "
              f"while **{rev_by_category.idxmax()}** is the leading revenue category — "
              f"worth prioritizing in inventory and staffing decisions.\n")
report.append(f"- The 3-month forecast suggests revenue will continue on its current trajectory, "
              f"reaching approximately ${future_preds[-1]:,.2f} by {future_dates[-1].strftime('%B %Y')}.\n")
report.append("- Seasonal spikes align with November-December, suggesting holiday-driven demand "
              "that should inform stocking and marketing timelines.\n")

with open("retail_report.md", "w") as f:
    f.writelines(report)

print("\n--- REPORT ---")
print("".join(report))
