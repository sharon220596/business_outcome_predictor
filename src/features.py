# src/features.py
import pandas as pd

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "total_charges" in df.columns and "tenure" in df.columns:
        df["avg_charge_per_month"] = df.apply(lambda r: r["total_charges"]/r["tenure"] if r["tenure"]>0 else r.get("monthly_charges",0), axis=1)
    else:
        df["avg_charge_per_month"] = df.get("monthly_charges", 0)
    if "tenure" in df.columns:
        df["tenure_group"] = pd.cut(df["tenure"], bins=[-1,6,12,24,48,120], labels=["0-6","7-12","13-24","25-48","49+"]).astype(str)
    else:
        df["tenure_group"] = "unknown"
    service_keywords = ["phone","internet","online_security","online_backup","device_protection","tech_support","streaming"]
    service_cols = [c for c in df.columns if any(k in c for k in service_keywords)]
    if service_cols:
        df["total_services"] = df[service_cols].apply(lambda row: sum(1 for v in row if str(v).lower() not in ["no","no service","no internet service","unknown","none"]), axis=1)
    else:
        df["total_services"] = df.apply(lambda r: sum(1 for v in r if str(v).lower()=="yes"), axis=1)
    if "internetservice" in df.columns:
        df["has_internet"] = df["internetservice"].apply(lambda x: 0 if str(x).lower().startswith("no") else 1)
    else:
        df["has_internet"] = 0
    return df