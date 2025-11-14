Business Outcome Predictor
==========================

Run steps:

1. Put dataset CSV into `data/` (e.g. TelecomCustomerChurn.csv)
2. Create venv:
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt

3. Run modules in order (from project root):
   python src/data_loader.py --data data/TelecomCustomerChurn.csv
   python src/cleaning.py data/TelecomCustomerChurn.csv
   python src/eda.py
   python src/features.py
   python src/models.py
   python src/evaluate.py
   python src/visualize.py

Outputs: `outputs/figures/`, `outputs/reports/`, `outputs/models/`