# src/models.py
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from packaging import version
import sklearn
from paths import MODEL_DIR
from data_loader import load_data
from cleaning import basic_cleaning
from features import make_features

def prep_data(path):
    df = load_data(path)
    df = basic_cleaning(df)
    df = make_features(df)
    if "customerid" in df.columns:
        df = df.drop(columns=["customerid"])
    if "churn" not in df.columns:
        raise ValueError("churn column is required")
    y = df["churn"]
    X = df.drop(columns=["churn"])
    num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()
    return X, y, num_cols, cat_cols

def build_preprocessor(num_cols, cat_cols):
    numeric_transformer = Pipeline([("scaler", StandardScaler())])
    # compatibility for OneHotEncoder argument
    skl_ver = version.parse(sklearn.__version__)
    ohe_kwargs = {"handle_unknown":"ignore"}
    if skl_ver >= version.parse("1.4"):
        ohe_kwargs["sparse_output"] = False
    else:
        ohe_kwargs["sparse"] = False
    categorical_transformer = Pipeline([("ohe", OneHotEncoder(**ohe_kwargs))])
    preproc = ColumnTransformer([("num", numeric_transformer, num_cols),
                                 ("cat", categorical_transformer, cat_cols)], remainder="drop")
    return preproc

def train_models(X_train, y_train, num_cols, cat_cols):
    preproc = build_preprocessor(num_cols, cat_cols)
    models = {
        "logreg": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    saved = {}
    for name, clf in models.items():
        pipe = Pipeline([("pre", preproc), ("clf", clf)])
        pipe.fit(X_train, y_train)
        path = os.path.join(MODEL_DIR, f"{name}.joblib")
        joblib.dump(pipe, path)
        saved[name] = path
    return saved

if __name__ == "__main__":
    p = "data/TelecomCustomerChurn.csv"
    X, y, num_cols, cat_cols = prep_data(p)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    trained = train_models(X_train, y_train, num_cols, cat_cols)
    for name, path in trained.items():
        pipe = joblib.load(path)
        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:,1] if hasattr(pipe, "predict_proba") else None
        print("Model:", name)
        print(classification_report(y_test, preds, digits=4))
        if probs is not None:
            print("ROC AUC:", roc_auc_score(y_test, probs))