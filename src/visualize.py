# src/visualize.py
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from paths import FIG_DIR
from data_loader import load_data
from cleaning import basic_cleaning
from features import make_features

def plot_roc(model_paths, X_test, y_test):
    plt.figure(figsize=(8,6))
    for name, path in model_paths.items():
        pipe = joblib.load(path)
        if hasattr(pipe, "predict_proba"):
            probs = pipe.predict_proba(X_test)[:,1]
            fpr, tpr, _ = roc_curve(y_test, probs)
            auc = roc_auc_score(y_test, probs)
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()
    out = os.path.join(FIG_DIR, "roc_comparison.png")
    plt.savefig(out); plt.close()
    return out

if __name__ == "__main__":
    df = load_data("TelecomCustomerChurn.csv")
    df = basic_cleaning(df)
    df = make_features(df)
    y = df["churn"]
    X = df.drop(columns=["churn","customerid"], errors="ignore")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model_paths = {"logreg": os.path.join("outputs","models","logreg.joblib"),
                   "rf": os.path.join("outputs","models","rf.joblib")}
    out = plot_roc(model_paths, X_test, y_test)
    print("Saved ROC plot:", out)