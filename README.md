# 🔍 Real-Time Fraud Detection with Explainable AI

A machine learning project that detects fraudulent credit card transactions in real time, using **XGBoost** and **SHAP** for explainable AI. Built with **Streamlit** for an interactive user interface.

---

## 🚀 Features

- ⚡ Real-time fraud prediction on transaction data
- 🧠 Explainable predictions with SHAP waterfall plots
- 🎯 Tuned decision threshold for optimal F1-score
- 🧪 Interactive sliders for inputting transaction features
- 📊 Model trained on the **Kaggle Credit Card Fraud Detection** dataset
- 🧰 Virtual environment + reproducible setup

---

## 📁 Project Structure
fraud-detector/
├── app.py # Streamlit app
├── FraudDetection.ipynb # Notebook for model creation
├── model/
│ ├── xgb_model.pkl # Trained XGBoost model
│ └── threshold.pkl # Custom threshold (e.g. 0.9)
│ 
├── archive/
│ └── creditcard.csv # Dataset
├── requirements.txt # Dependencies
├── README.md # Project documentation
└── venv/ # Virtual environment (not committed)


---

## 🧠 Model Overview

- **Model**: XGBoost Classifier
- **Target**: Binary classification (Fraud: `1`, Non-Fraud: `0`)
- **Metrics**: Optimized for **F1-Score** using a custom threshold (`0.9`)
- **Explainability**: SHAP values highlight which features contributed to each decision

---

## ⚙️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/fraud-detector.git
cd fraud-detector
```

### 2. Create and Activate Virtual Environment

```
python -m venv venv
venv\Scripts\activate     # Windows
# source venv/bin/activate  # macOS/Linux
```

### 3. Install Dependencies

`pip install -r requirements.txt`


### 4. Run The App

`streamlit run app.py`

---

## 🧪 How It Works
1. Enter transaction features (V1–V28, Time, Amount)
2. App normalizes Time and Amount using the training scaler
3. Model predicts fraud probability
4. A custom threshold (e.g. 0.9) is applied to classify the transaction
5. SHAP waterfall plot explains which features influenced the prediction

---

## 📦 Requirements
Key packages:

* xgboost
* shap
* scikit-learn
* streamlit
* matplotlib
* numpy
* pandas

All in `requirements.txt`

---

## 📚 Dataset
* Source: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download)
* Rows: 284,807 transactions
* Imbalance: Only ~0.17% are frauds