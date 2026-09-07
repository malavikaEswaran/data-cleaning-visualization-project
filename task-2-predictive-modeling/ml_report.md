# Predictive Modeling Report — Customer Churn Prediction
**Dataset:** 800 customers, 6 features (target: `Churn`)
**Class balance:** {0: 539, 1: 261}

**Train/Test split:** 600 train / 200 test (75/25, stratified)
**Features used:** Age, TenureMonths, MonthlyCharge, SupportCalls, ContractType_enc

## Model Comparison
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.770 | 0.679 | 0.554 | 0.610 | 0.839 |
| Decision Tree | 0.730 | 0.612 | 0.462 | 0.526 | 0.777 |
| Random Forest | 0.750 | 0.636 | 0.538 | 0.583 | 0.820 |

**Best performing model (by F1-score): Logistic Regression**

## Key Findings
- **Logistic Regression** achieved the best overall balance of precision and recall (F1=0.610, AUC=0.839).
- The most influential feature in predicting churn (per Random Forest) is **TenureMonths**.
- Customers on Month-to-Month contracts and those with more support calls show a notably higher churn tendency, consistent with the feature importance ranking.
- ROC-AUC scores above 0.5 across all models confirm the features carry real predictive signal beyond random guessing.
