# Community Default Prediction System

An AI-powered prototype system for predicting loan default risk in community savings and cooperative financial systems using Logistic Regression and behavioral financial analytics.

---

## 📌 Project Overview

This project was developed as a Final Year Project to demonstrate how machine learning can assist community savings groups, cooperatives, and Njangi systems in analyzing borrower reliability and predicting the probability of loan default.

The system evaluates member financial behavior using indicators such as:

- Contribution consistency
- Repayment punctuality
- Default history
- Attendance score
- Savings stability
- Loan-to-savings ratio
- Overdue repayment behavior
- Current unpaid loans

The model then predicts:

- Probability of Default (PD)
- Risk Level Classification
- Loan Decision Support

---

## 🧠 Machine Learning Model

The project uses:

- Logistic Regression
- Scikit-learn
- Behavioral credit scoring
- Risk classification analysis

The model predicts whether a member is likely to default on a loan based on historical behavioral indicators.

---

## 📊 Features

- Credit risk analysis
- Probability of default prediction
- Risk classification (Low, Medium, High)
- Synthetic cooperative loan dataset
- Logistic Regression training pipeline
- Prediction engine
- Machine learning prototype workflow

---

## ⚙️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Joblib
- Matplotlib

---

## 📂 Project Structure

```text
community-default-prediction-system/
│
├── data/
│   └── community_loan_dataset_5000_records.csv
│
├── models/
│   └── default_risk_model.pkl
│
├── train_model.py
├── predict.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Setup Instructions

### 1. Clone repository

```bash
git clone https://github.com/your-username/community-default-prediction-system.git
cd community-default-prediction-system
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train model

```bash
python train_model.py
```

### 5. Run prediction

```bash
python predict.py
```

---

## 📈 Sample Output

```text
Probability of Default: 24.34%
Risk Level: Low Risk
```

---

## ⚠️ Important Note

This project uses a synthetic prototype dataset designed for academic demonstration and machine learning experimentation. Real-world deployment would require large-scale localized cooperative financial datasets for improved predictive accuracy.

---

## 🎓 Academic Purpose

This system was developed for educational and research purposes to demonstrate the integration of machine learning into community financial management systems.

---

## 👨‍💻 Author

AYEMELE ELGOL

Software Engineer | DevOps Engineer | AI & Financial Systems Enthusiast