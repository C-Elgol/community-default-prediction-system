import pandas as pd
import numpy as np

np.random.seed(42)

n = 5000

# Positive behavior features
contribution_consistency = np.random.randint(30, 101, n)
repayment_punctuality = np.random.randint(20, 101, n)
attendance_score = np.random.randint(30, 101, n)
savings_stability = np.random.randint(30, 101, n)

# Risk features
default_history = np.random.binomial(1, 0.25, n)
current_unpaid_loans = np.random.randint(0, 6, n)
average_overdue_days = np.random.randint(0, 91, n)
loan_to_savings_ratio = np.round(np.random.uniform(0.5, 5.0, n), 2)

# Stronger weighting for medium-risk members
risk_score = (
    0.04 * average_overdue_days
    + 1.0 * default_history
    + 0.80 * current_unpaid_loans
    + 0.60 * loan_to_savings_ratio
    - 0.018 * contribution_consistency
    - 0.020 * repayment_punctuality
    - 0.015 * attendance_score
    - 0.018 * savings_stability
)

# Moderate noise
risk_score += np.random.normal(0, 1.0, n)

# Convert to probability
probability_default = 1 / (1 + np.exp(-risk_score))

# Create labels
defaulted = (probability_default > 0.50).astype(int)

df = pd.DataFrame({
    "contribution_consistency": contribution_consistency,
    "repayment_punctuality": repayment_punctuality,
    "default_history": default_history,
    "attendance_score": attendance_score,
    "savings_stability": savings_stability,
    "loan_to_savings_ratio": loan_to_savings_ratio,
    "average_overdue_days": average_overdue_days,
    "current_unpaid_loans": current_unpaid_loans,
    "defaulted": defaulted
})

print(df["defaulted"].value_counts())

df.to_csv(
    "data/community_loan_dataset_5000_v2.csv",
    index=False
)

print("Dataset saved successfully.")