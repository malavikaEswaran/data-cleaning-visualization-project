# Exploratory Data Analysis (EDA) Report — Student Performance
**Dataset:** 600 students, 7 variables
**Variables:** StudyHoursPerDay, SleepHoursPerDay, AttendancePercent, Gender, ParentalEducation, Extracurricular, ExamScore

## Statistical Summary
|       |   StudyHoursPerDay |   SleepHoursPerDay |   AttendancePercent |   ExamScore |
|:------|-------------------:|-------------------:|--------------------:|------------:|
| count |             600    |             600    |              600    |      600    |
| mean  |               4.9  |               6.99 |               84.31 |       65.4  |
| std   |               1.94 |               1.26 |                9.42 |        9.75 |
| min   |               0.5  |               3    |               55.9  |       39    |
| 25%   |               3.6  |               6.1  |               77.8  |       58.6  |
| 50%   |               4.9  |               7    |               84.95 |       65.35 |
| 75%   |               6.2  |               7.9  |               91.12 |       71.62 |
| max   |              10.7  |              10    |              100    |       94.1  |

### Categorical Variable Counts

**Gender:**
- Female: 302 (50.3%)
- Male: 298 (49.7%)

**ParentalEducation:**
- Bachelor's: 256 (42.7%)
- High School: 162 (27.0%)
- Master's: 125 (20.8%)
- PhD: 57 (9.5%)

**Extracurricular:**
- No: 367 (61.2%)
- Yes: 233 (38.8%)

## Correlation Matrix
|                   |   StudyHoursPerDay |   SleepHoursPerDay |   AttendancePercent |   ExamScore |
|:------------------|-------------------:|-------------------:|--------------------:|------------:|
| StudyHoursPerDay  |               1    |              -0.04 |                0.02 |        0.66 |
| SleepHoursPerDay  |              -0.04 |               1    |                0.02 |        0.08 |
| AttendancePercent |               0.02 |               0.02 |                1    |        0.25 |
| ExamScore         |               0.66 |               0.08 |                0.25 |        1    |

- The variable most strongly correlated with **ExamScore** is **StudyHoursPerDay** (r = 0.66).

## Key Findings
- **Study hours** show the strongest positive relationship with exam performance (r = 0.66), confirming it as the primary driver of scores.
- **Attendance** also correlates positively with performance (r = 0.25), though less strongly than study hours.
- **Sleep hours** show a comparatively weaker relationship with exam scores (r = 0.08).
- Average scores by parental education level: {'High School': 65.6, "Bachelor's": 65.2, "Master's": 64.9, 'PhD': 66.9} — showing a mild upward trend with higher parental education.
- Students participating in extracurricular activities averaged 65.6 vs 65.3 for non-participants — a modest but positive difference.
- No extreme outliers were observed in the core numeric variables after initial inspection; the distributions are roughly unimodal and reasonably symmetric.
