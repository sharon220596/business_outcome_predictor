# src/main.py
from data_loader import load_data, initial_explore
from cleaning import basic_cleaning
from features import make_features
from eda import run_all as run_eda
from models import prep_data, train_models
from evaluate import compare_models
from visualize import plot_roc
from paths import DATA_DIR, REPORT_DIR, FIG_DIR, MODEL_DIR

def run_full(data_filename="TelecomCustomerChurn.csv"):
    print("Loading and exploring...")
    df = load_data(data_filename)
    initial_explore(df)
    print("Cleaning...")
    df = basic_cleaning(df)
    print("Feature engineering...")
    df = make_features(df)
    print("Running EDA (plots)...")
    run_eda(data_filename)
    print("Preparing data for training...")
    X, y, num_cols, cat_cols = prep_data(data_filename)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Training models...")
    trained = train_models(X_train, y_train, num_cols, cat_cols)
    print("Evaluating...")
    results = compare_models(trained, X_test, y_test)
    print("Visualize ROC...")
    plot_roc(trained, X_test, y_test)
    print("Done. Check outputs/ for figures, models, reports.")

if __name__ == "__main__":
    run_full()