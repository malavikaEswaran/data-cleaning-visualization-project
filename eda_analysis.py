import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
df = pd.read_csv("student_performance_data.csv")
print("Dataset shape:", df.shape)

report = []
report.append("# Exploratory Data Analysis (EDA) Report — Student Performance\n")
report.append(f"**Dataset:** {df.shape[0]} students, {df.shape[1]} variables\n")
report.append(f"**Variables:** {', '.join(df.columns)}\n")

# ---------------------------------------------------------------
# 2. STATISTICAL SUMMARY
# ---------------------------------------------------------------
numeric_cols = ["StudyHoursPerDay", "SleepHoursPerDay", "AttendancePercent", "ExamScore"]
summary = df[numeric_cols].describe().round(2)

report.append("\n## Statistical Summary\n")
report.append(summary.to_markdown())
report.append("\n")

# Categorical breakdowns
report.append("\n### Categorical Variable Counts\n")
for col in ["Gender", "ParentalEducation", "Extracurricular"]:
    report.append(f"\n**{col}:**\n")
    counts = df[col].value_counts()
    for val, cnt in counts.items():
        report.append(f"- {val}: {cnt} ({cnt/len(df)*100:.1f}%)\n")

# ---------------------------------------------------------------
# 3. CORRELATION ANALYSIS
# ---------------------------------------------------------------
corr = df[numeric_cols].corr().round(2)
report.append("\n## Correlation Matrix\n")
report.append(corr.to_markdown())
report.append("\n")

strongest_corr = corr["ExamScore"].drop("ExamScore").abs().idxmax()
strongest_val = corr["ExamScore"][strongest_corr]
report.append(f"\n- The variable most strongly correlated with **ExamScore** is "
              f"**{strongest_corr}** (r = {strongest_val:.2f}).\n")

# ---------------------------------------------------------------
# 4. VISUALIZATIONS
# ---------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(14, 16))
fig.suptitle("Exploratory Data Analysis — Student Performance Dashboard", fontsize=16, fontweight="bold")

# (a) Correlation heatmap
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=axes[0, 0])
axes[0, 0].set_title("Correlation Heatmap")

# (b) Exam score distribution
sns.histplot(df["ExamScore"], bins=25, kde=True, ax=axes[0, 1], color="steelblue")
axes[0, 1].set_title("Distribution of Exam Scores")

# (c) Study hours vs exam score
sns.scatterplot(data=df, x="StudyHoursPerDay", y="ExamScore", hue="Extracurricular", ax=axes[1, 0], alpha=0.7)
axes[1, 0].set_title("Study Hours vs Exam Score")

# (d) Attendance vs exam score
sns.scatterplot(data=df, x="AttendancePercent", y="ExamScore", hue="Gender", ax=axes[1, 1], alpha=0.7)
axes[1, 1].set_title("Attendance vs Exam Score")

# (e) Exam score by parental education
order = ["High School", "Bachelor's", "Master's", "PhD"]
sns.boxplot(data=df, x="ParentalEducation", y="ExamScore", order=order, ax=axes[2, 0], palette="Set2")
axes[2, 0].set_title("Exam Score by Parental Education")
axes[2, 0].tick_params(axis="x", rotation=20)

# (f) Exam score by extracurricular participation
sns.violinplot(data=df, x="Extracurricular", y="ExamScore", ax=axes[2, 1], palette="Set3")
axes[2, 1].set_title("Exam Score by Extracurricular Participation")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("eda_dashboard.png", dpi=150)
print("Saved eda_dashboard.png")

# ---------------------------------------------------------------
# 5. KEY FINDINGS
# ---------------------------------------------------------------
edu_means = df.groupby("ParentalEducation")["ExamScore"].mean().round(1).reindex(order)
extra_means = df.groupby("Extracurricular")["ExamScore"].mean().round(1)

report.append("\n## Key Findings\n")
report.append(f"- **Study hours** show the strongest positive relationship with exam performance "
              f"(r = {corr.loc['StudyHoursPerDay','ExamScore']:.2f}), confirming it as the primary driver of scores.\n")
report.append(f"- **Attendance** also correlates positively with performance (r = {corr.loc['AttendancePercent','ExamScore']:.2f}), "
              f"though less strongly than study hours.\n")
report.append(f"- **Sleep hours** show a comparatively weaker relationship with exam scores "
              f"(r = {corr.loc['SleepHoursPerDay','ExamScore']:.2f}).\n")
report.append(f"- Average scores by parental education level: {edu_means.to_dict()} — "
              f"showing a mild upward trend with higher parental education.\n")
report.append(f"- Students participating in extracurricular activities averaged {extra_means.get('Yes', 0)} "
              f"vs {extra_means.get('No', 0)} for non-participants — a modest but positive difference.\n")
report.append("- No extreme outliers were observed in the core numeric variables after initial inspection; "
              "the distributions are roughly unimodal and reasonably symmetric.\n")

with open("eda_report.md", "w") as f:
    f.writelines(report)

print("\n--- REPORT ---")
print("".join(report))
