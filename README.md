🌟 Business Outcome Predictor

A complete end-to-end machine learning pipeline for predicting business outcomes using real-world data (Telecom Customer Churn).

⸻

📌 Project Overview

This project builds a full ML workflow that predicts whether a telecom customer will churn (Yes/No).
It includes:

✔ Data loading
✔ Data cleaning
✔ Feature engineering
✔ EDA visualization
✔ Model training (Logistic Regression, Random Forest)
✔ Model evaluation
✔ ROC curve comparison
✔ Automatic outputs folder structure
✔ One-click pipeline using main.py

All scripts use a centralized paths.py for safe file management.

business_outcome_predictor/
│
├── data/
│   └── TelecomCustomerChurn.csv
│
├── outputs/
│   ├── figures/
│   ├── reports/
│   └── models/
│
├── src/
│   ├── paths.py
│   ├── data_loader.py
│   ├── cleaning.py
│   ├── features.py
│   ├── eda.py
│   ├── models.py
│   ├── evaluate.py
│   ├── visualize.py
│   └── main.py
│
├── .venv/
|--------------------------------------------------------------------------
├── requirements.txt
└── README.md

📊 Outputs Generated

📁 outputs/models/
	•	logreg.joblib
	•	rf.joblib

📁 outputs/figures/
	•	churn_distribution.png
	•	monthly_charges_hist.png
	•	tenure_vs_churn.png
	•	confusion_logreg.png
	•	confusion_rf.png
	•	roc_comparison.png

📁 outputs/reports/
	•	initial_explore.json
	•	model_comparison.json
   
❤️ Author

Sharon Karunya
Full-Stack AI Developer (in progress!)
Business ML Project – End-to-End
