import pandas as pd
import numpy as np

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "churn" in df.columns:
        df["churn"] = df["churn"].astype(str).str.strip().str.lower().map(
            {"yes":1,"y":1,"1":1,"true":1,"no":0,"n":0,"0":0,"false":0}
        ).fillna(0).astype(int)
    for c in ["total_charges","monthly_charges","tenure","age"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].fillna("Unknown").astype(str).str.strip()
    return df