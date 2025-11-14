# src/eda.py
import os
import matplotlib.pyplot as plt
import seaborn as sns
from paths import FIG_DIR
from data_loader import load_data
from cleaning import basic_cleaning

def churn_distribution(df):
    plt.figure(figsize=(6,4))
    sns.countplot(x="churn", data=df)
    plt.title("Churn distribution")
    out = os.path.join(FIG_DIR, "churn_distribution.png")
    plt.savefig(out); plt.close()
    return out

def monthly_charges_hist(df):
    if "monthly_charges" not in df.columns:
        return None
    plt.figure(figsize=(6,4))
    sns.histplot(df["monthly_charges"].dropna(), bins=30, kde=True)
    out = os.path.join(FIG_DIR, "monthly_charges_hist.png")
    plt.savefig(out); plt.close()
    return out

def tenure_vs_churn(df):
    if "tenure" not in df.columns:
        return None
    plt.figure(figsize=(7,5))
    sns.boxplot(x="churn", y="tenure", data=df)
    out = os.path.join(FIG_DIR, "tenure_vs_churn.png")
    plt.savefig(out); plt.close()
    return out

def run_all(path):
    df = load_data(path)
    df = basic_cleaning(df)
    files = []
    files.append(churn_distribution(df))
    files.append(monthly_charges_hist(df))
    files.append(tenure_vs_churn(df))
    files = [f for f in files if f]
    return files

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/TelecomCustomerChurn.csv")
    args = parser.parse_args()
    outs = run_all(args.data)
    print("Saved EDA plots:", outs)