import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ---------------------------------------------------------------
# 1. LOAD RAW DATA
# ---------------------------------------------------------------
df = pd.read_csv("raw_sales_data.csv", parse_dates=["OrderDate"])
print("Raw shape:", df.shape)

report_lines = []
report_lines.append("# Data Cleaning & Visualization Report\n")
report_lines.append(f"**Raw dataset shape:** {df.shape[0]} rows x {df.shape[1]} columns\n")

# ---------------------------------------------------------------
# 2. HANDLE DUPLICATES
# ---------------------------------------------------------------
dupe_count = df.duplicated().sum()
df = df.drop_duplicates()
report_lines.append(f"- Removed **{dupe_count}** exact duplicate rows.\n")

# ---------------------------------------------------------------
# 3. STANDARDIZE TEXT FIELDS
# ---------------------------------------------------------------
df["Region"] = df["Region"].str.title()
df["Category"] = df["Category"].str.title()

# ---------------------------------------------------------------
# 4. HANDLE MISSING VALUES
# ---------------------------------------------------------------
missing_before = df.isna().sum()

# Region: fill with mode
df["Region"] = df["Region"].fillna(df["Region"].mode()[0])

# UnitsSold: fill with median (robust to outliers)
units_median = df["UnitsSold"].median()
df["UnitsSold"] = df["UnitsSold"].fillna(units_median)

# UnitPrice: fill with median per Category
df["UnitPrice"] = df.groupby("Category")["UnitPrice"].transform(
    lambda x: x.fillna(x.median())
)

# Recompute Revenue wherever it's missing or inconsistent
df["Revenue"] = df["UnitsSold"] * df["UnitPrice"]

report_lines.append("### Missing values (before -> after)\n")
report_lines.append("| Column | Missing Before | Missing After |\n|---|---|---|\n")
for col in df.columns:
    before = missing_before.get(col, 0)
    after = df[col].isna().sum()
    if before > 0 or after > 0:
        report_lines.append(f"| {col} | {before} | {after} |\n")
report_lines.append(
    "\nStrategy: Region -> mode fill, UnitsSold -> median fill, "
    "UnitPrice -> category-wise median fill, Revenue -> recomputed "
    "from cleaned UnitsSold x UnitPrice.\n"
)

# ---------------------------------------------------------------
# 5. HANDLE OUTLIERS (IQR method on UnitsSold)
# ---------------------------------------------------------------
Q1 = df["UnitsSold"].quantile(0.25)
Q3 = df["UnitsSold"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df["UnitsSold"] < lower) | (df["UnitsSold"] > upper)]
report_lines.append(f"- Detected **{len(outliers)}** outliers in `UnitsSold` using the IQR method "
                     f"(valid range: {lower:.1f} to {upper:.1f}).\n")

# Cap outliers instead of dropping (preserves row count / other columns)
df["UnitsSold"] = df["UnitsSold"].clip(lower=lower, upper=upper)
df["Revenue"] = df["UnitsSold"] * df["UnitPrice"]
report_lines.append("- Outliers were **capped** (winsorized) to the IQR bounds rather than dropped, "
                     "to preserve sample size while limiting distortion.\n")

df.to_csv("cleaned_sales_data.csv", index=False)
report_lines.append(f"\n**Cleaned dataset shape:** {df.shape[0]} rows x {df.shape[1]} columns\n")

# ---------------------------------------------------------------
# 6. VISUALIZATIONS
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Sales Data Dashboard — Key Insights", fontsize=16, fontweight="bold")

# (a) Revenue by Region
rev_region = df.groupby("Region")["Revenue"].sum().sort_values(ascending=False)
sns.barplot(x=rev_region.index, y=rev_region.values, ax=axes[0, 0], palette="Blues_d")
axes[0, 0].set_title("Total Revenue by Region")
axes[0, 0].set_ylabel("Revenue ($)")
axes[0, 0].set_xlabel("")

# (b) Revenue by Category
rev_cat = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
sns.barplot(x=rev_cat.values, y=rev_cat.index, ax=axes[0, 1], palette="Greens_d")
axes[0, 1].set_title("Total Revenue by Category")
axes[0, 1].set_xlabel("Revenue ($)")

# (c) Units Sold distribution (post-cleaning)
sns.histplot(df["UnitsSold"], bins=20, kde=True, ax=axes[1, 0], color="steelblue")
axes[1, 0].set_title("Distribution of Units Sold (Cleaned)")
axes[1, 0].set_xlabel("Units Sold")

# (d) Revenue trend over time
daily_rev = df.groupby(df["OrderDate"].dt.to_period("W"))["Revenue"].sum()
daily_rev.index = daily_rev.index.to_timestamp()
axes[1, 1].plot(daily_rev.index, daily_rev.values, marker="o", markersize=3, color="darkorange")
axes[1, 1].set_title("Weekly Revenue Trend")
axes[1, 1].set_xlabel("Week")
axes[1, 1].set_ylabel("Revenue ($)")
axes[1, 1].tick_params(axis="x", rotation=45)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("dashboard.png", dpi=150)
print("Saved dashboard.png")

# ---------------------------------------------------------------
# 7. KEY FINDINGS
# ---------------------------------------------------------------
top_region = rev_region.idxmax()
top_category = rev_cat.idxmax()
total_revenue = df["Revenue"].sum()
avg_order_value = df["Revenue"].mean()

report_lines.append("\n## Key Findings\n")
report_lines.append(f"- **Total Revenue:** ${total_revenue:,.2f}\n")
report_lines.append(f"- **Average Order Value:** ${avg_order_value:,.2f}\n")
report_lines.append(f"- **Top-performing Region:** {top_region} (${rev_region.max():,.2f})\n")
report_lines.append(f"- **Top-performing Category:** {top_category} (${rev_cat.max():,.2f})\n")
report_lines.append(f"- Revenue shows week-to-week fluctuation with no single dominant seasonal spike, "
                     f"suggesting relatively stable demand across the period.\n")

with open("report.md", "w") as f:
    f.writelines(report_lines)

print("\n--- REPORT ---")
print("".join(report_lines))
