import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score


DATA_PATH = "data/community_loan_dataset_5000_records.csv"
MODEL_PATH = "models/default_risk_model.pkl"


def load_data():
    df = pd.read_csv(DATA_PATH)

    # Remove ID column because it does not help prediction
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    return df


def train_model():
    df = load_data()

    X = df.drop(columns=["defaulted"])
    y = df["defaulted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, predictions))
    print("ROC AUC:", roc_auc_score(y_test, probabilities))
    print(classification_report(y_test, predictions))

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()