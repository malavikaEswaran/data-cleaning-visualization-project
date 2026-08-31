# Data Cleaning & Visualization Report
**Raw dataset shape:** 520 rows x 7 columns
- Removed **5** exact duplicate rows.
### Missing values (before -> after)
| Column | Missing Before | Missing After |
|---|---|---|
| Region | 12 | 0 |
| UnitsSold | 15 | 0 |
| UnitPrice | 13 | 0 |
| Revenue | 10 | 0 |

Strategy: Region -> mode fill, UnitsSold -> median fill, UnitPrice -> category-wise median fill, Revenue -> recomputed from cleaned UnitsSold x UnitPrice.
- Detected **8** outliers in `UnitsSold` using the IQR method (valid range: -19.5 to 72.5).
- Outliers were **capped** (winsorized) to the IQR bounds rather than dropped, to preserve sample size while limiting distortion.

**Cleaned dataset shape:** 515 rows x 7 columns

## Key Findings
- **Total Revenue:** $3,431,851.08
- **Average Order Value:** $6,663.79
- **Top-performing Region:** North ($1,102,930.94)
- **Top-performing Category:** Home & Kitchen ($750,894.54)
- Revenue shows week-to-week fluctuation with no single dominant seasonal spike, suggesting relatively stable demand across the period.
