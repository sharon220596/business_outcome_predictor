
import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, roc_curve

from paths import DATA_DIR, FIG_DIR, REPORT_DIR, MODEL_DIR

from data_loader import load_data
from cleaning import basic_cleaning
from features import make_features


def compare_models(model_paths, X_test, y_test):
    results = {}

    for name, path in model_paths.items():
        pipe = joblib.load(path)

        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

        rep = classification_report(y_test, preds, output_dict=True)
        auc = roc_auc_score(y_test, probs) if probs is not None else None

        results[name] = {"report": rep, "roc_auc": auc}

        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion: {name}")
        plt.savefig(os.path.join(FIG_DIR, f"confusion_{name}.png"))
        plt.close()

    with open(os.path.join(REPORT_DIR, "model_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    
    dataset_path = os.path.join(DATA_DIR, "TelecomCustomerChurn.csv")

    # Load → Clean → Feature Engineering
    df = load_data(dataset_path)
    df = basic_cleaning(df)
    df = make_features(df)

    y = df["churn"]
    X = df.drop(columns=["churn", "customerid"], errors="ignore")

    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Load model files from correct directory
    model_paths = {
        "logreg": os.path.join(MODEL_DIR, "logreg.joblib"),
        "rf": os.path.join(MODEL_DIR, "rf.joblib"),
    }

    # Evaluate
    results = compare_models(model_paths, X_test, y_test)

    print("Saved results:", os.path.join(REPORT_DIR, "model_comparison.json"))