import pandas as pd
import joblib

MODEL_PATH = "models/default_risk_model.pkl"

model = joblib.load(MODEL_PATH)

sample_member = pd.DataFrame([{
    "contribution_consistency": 65,
    "repayment_punctuality": 55,
    "default_history": 0,
    "attendance_score": 60,
    "savings_stability": 65,
    "loan_to_savings_ratio": 2.0,
    "average_overdue_days": 15,
    "current_unpaid_loans": 1
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