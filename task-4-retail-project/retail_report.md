# Real-World Data Project — Retail Sales Analysis & Forecast
**Dataset:** 10888 transaction records spanning 2024-01-01 to 2025-12-31
**Stores:** Store A, Store B, Store C  
**Categories:** Electronics, Clothing, Groceries, Home & Kitchen, Beauty

## Summary Statistics
- **Total Revenue:** $6,744,829.87
- **Total Units Sold:** 84,853
- **Average Daily Revenue:** $9,226.85
- **Top Store:** Store A ($2,697,620.30)
- **Top Category:** Electronics ($4,398,521.10)

## Sales Forecasting Model
Model: Linear Regression on monthly revenue trend.
- **Validation MAE (last 3 known months):** $40,595.18
- **Validation R²:** -0.575

> **Note on model limitation:** The negative R² on the validation months reflects that revenue is strongly seasonal (a sharp Nov–Dec spike), which a simple linear trend cannot capture. A seasonal model (e.g. SARIMA or Prophet) would likely perform better for production forecasting; the linear fit here is presented as a baseline to illustrate the overall trend direction.

### Forecast — Next 3 Months
| Month | Predicted Revenue |
|---|---|
| Jan 2026 | $298,668.59 |
| Feb 2026 | $299,673.36 |
| Mar 2026 | $300,678.13 |

## Conclusions
- Revenue shows a clear **upward trend** over the two-year period (~$1,005 change per month on average).
- **Sunday** is the strongest day of the week for sales, consistent with typical retail weekend patterns.
- **Store A** consistently outperforms other locations, while **Electronics** is the leading revenue category — worth prioritizing in inventory and staffing decisions.
- The 3-month forecast suggests revenue will continue on its current trajectory, reaching approximately $300,678.13 by March 2026.
- Seasonal spikes align with November-December, suggesting holiday-driven demand that should inform stocking and marketing timelines.
