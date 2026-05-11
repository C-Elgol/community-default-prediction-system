import pandas as pd
import joblib

MODEL_PATH = "models/default_risk_model.pkl"

model = joblib.load(MODEL_PATH)

sample_member = pd.DataFrame([{
    "contribution_consistency": 0.6,
    "repayment_punctuality": 0.1,
    "default_history": 0.05,
    "attendance_score": 0.5,
    "savings_stability": 1.0,
    "loan_to_savings_ratio": 0.0,
    "average_overdue_days": 0.0,
    "current_unpaid_loans": 1.0
}])

pd_risk = model.predict_proba(sample_member)[0][1]

print("Probability of Default:", round(pd_risk * 100, 2), "%")

if pd_risk < 0.30:
    risk_level = "Low Risk"
elif pd_risk < 0.60:
    risk_level = "Medium Risk"
else:
    risk_level = "High Risk"

print("Risk Level:", risk_level)