"""
preprocess.py
=============
Beginner-friendly data preprocessing and feature engineering script.

What this script does:
1. Loads the Telco Customer Churn dataset from data/raw/
2. Inspects dataset shape, columns, missing values, and churn class balance
3. Cleans messy values (e.g., spaces in TotalCharges)
4. Engineers 3 intuitive new features:
   - TenureGroup: Customer loyalty stage (0-12m, 13-24m, etc.)
   - TotalServices: Number of active telecom/internet services
   - MonthlyToTotalRatio: Spend velocity compared to historical spend
5. Splits data into Train (80%) and Test (20%) using stratification
6. Fits StandardScaler and OneHotEncoder on Train data ONLY (prevents data leakage)
7. Saves processed arrays and the preprocessor pipeline to data/processed/ and models/
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def find_raw_dataset(raw_dir="data/raw"):
    """
    Finds the churn CSV inside data/raw/.
    Handles various standard filenames so you don't run into naming issues.
    """
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir, exist_ok=True)

    # Check common names first
    candidates = [
        "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "telco_churn.csv",
        "Telco-Customer-Churn.csv",
        "churn.csv"
    ]
    for name in candidates:
        full_path = os.path.join(raw_dir, name)
        if os.path.isfile(full_path):
            return full_path

    # Check if any .csv file exists in data/raw
    csv_files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
    if csv_files:
        return os.path.join(raw_dir, csv_files[0])

    return None


def inspect_raw_data(df):
    """
    Task 1: Inspects the dataset and prints beginner-friendly summaries.
    """
    print("\n" + "=" * 60)
    print("STEP 1: DATA EXPLORATION & INSPECTION")
    print("=" * 60)
    print(f"Total Rows (Customers): {df.shape[0]}")
    print(f"Total Columns (Features): {df.shape[1]}")

    print("\n[Columns and Data Types]")
    print(df.dtypes)

    print("\n[Missing Values Check]")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No explicit NaN missing values found.")

    # TotalCharges is often stored as object/string because of blank spaces ' '
    blank_spaces = (df['TotalCharges'] == ' ').sum() if 'TotalCharges' in df.columns else 0
    print(f"Hidden blank spaces in 'TotalCharges': {blank_spaces} rows")

    print("\n[Churn Target Class Balance]")
    churn_counts = df['Churn'].value_counts()
    churn_percent = df['Churn'].value_counts(normalize=True) * 100
    for label in churn_counts.index:
        print(f"  {label}: {churn_counts[label]} customers ({churn_percent[label]:.2f}%)")
    print("=" * 60 + "\n")


def clean_and_engineer_features(df):
    """
    Task 2: Cleans data types and creates 3 new features.
    """
    print("Cleaning data and creating 3 new features...")
    data = df.copy()

    # WHY: customerID is a random identifier. Keeping it would cause models
    # to memorize random strings instead of learning real customer patterns.
    if 'customerID' in data.columns:
        data = data.drop(columns=['customerID'])

    # WHY: TotalCharges has 11 rows with empty spaces " " where tenure is 0.
    # We convert invalid values to NaN and fill with 0.0 because new customers
    # haven't paid their first monthly bill yet.
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce').fillna(0.0)

    # WHY: Binary target variable is easier for machine learning algorithms to learn
    # as 1 (Churned / Positive class) and 0 (Retained / Negative class).
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

    # -------------------------------------------------------------
    # FEATURE 1: TenureGroup (Categorical bucket)
    # WHY: Customer churn probability changes dramatically with loyalty stage.
    # Customers in month 1-12 churn far more frequently than customers after 2+ years.
    # -------------------------------------------------------------
    def get_tenure_group(tenure):
        if tenure <= 12:
            return "0-12m"
        elif tenure <= 24:
            return "13-24m"
        elif tenure <= 48:
            return "25-48m"
        else:
            return "49-72m"

    data['TenureGroup'] = data['tenure'].apply(get_tenure_group)

    # -------------------------------------------------------------
    # FEATURE 2: TotalServices (Count)
    # WHY: Customers subscribed to more services (e.g. streaming, backup, security)
    # have higher switching friction, making them much less likely to leave.
    # -------------------------------------------------------------
    service_cols = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    # Count how many of these services are set to 'Yes'
    data['TotalServices'] = data[service_cols].apply(
        lambda row: sum(row[col] == 'Yes' for col in service_cols), axis=1
    )

    # -------------------------------------------------------------
    # FEATURE 3: MonthlyToTotalRatio (Continuous numerical)
    # WHY: Compares the current monthly charge to overall spend. High ratios
    # signal new accounts or sudden price jumps that can trigger churn.
    # We add 1.0 to avoid division by zero.
    # -------------------------------------------------------------
    data['MonthlyToTotalRatio'] = data['MonthlyCharges'] / (data['TotalCharges'] + 1.0)

    print("Created features: 'TenureGroup', 'TotalServices', 'MonthlyToTotalRatio'.")
    return data


def build_and_apply_pipeline(df):
    """
    Splits data into train and test sets, then uses a scikit-learn
    ColumnTransformer to prevent data leakage.
    """
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    # WHY: 80/20 train/test split allows us to evaluate generalizability.
    # stratify=y preserves the exact 73:27 ratio in both train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Data split: {X_train.shape[0]} train rows, {X_test.shape[0]} test rows.")

    # Identify numeric and categorical columns
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'TotalServices', 'MonthlyToTotalRatio']
    categorical_features = [col for col in X.columns if col not in numeric_features]

    # WHY: StandardScaler puts numbers on the same scale (mean 0, variance 1)
    # so neural networks (MLP) and logistic regression don't give huge weights to TotalCharges.
    # OneHotEncoder converts categories into binary columns without imposing false ranking.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
        ]
    )

    # CRITICAL LEAKAGE PREVENTION:
    # We fit the transformers on X_train ONLY, then transform both X_train and X_test.
    print("Fitting preprocessor pipeline on training data only...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Extract clean column names after one-hot encoding
    encoded_cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = list(numeric_features) + list(encoded_cat_features)

    # Convert back to clean DataFrames for simple debugging and inspection
    X_train_df = pd.DataFrame(X_train_transformed, columns=all_feature_names)
    X_test_df = pd.DataFrame(X_test_transformed, columns=all_feature_names)

    # Save to data/processed and models/
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    X_train_df.to_csv("data/processed/X_train.csv", index=False)
    X_test_df.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    # Save the fitted preprocessor and feature names
    joblib.dump(preprocessor, "models/preprocessor.joblib")
    joblib.dump(all_feature_names, "models/feature_names.joblib")
    joblib.dump(categorical_features, "models/categorical_features.joblib")
    joblib.dump(numeric_features, "models/numeric_features.joblib")

    print("Successfully saved processed datasets to 'data/processed/' and preprocessor to 'models/'!")
    return X_train_df, X_test_df, y_train, y_test


def main():
    csv_path = find_raw_dataset()

    if not csv_path:
        print("\n" + "!" * 60)
        print("[ERROR] Telco Customer Churn dataset not found!")
        print("Please place the downloaded CSV file inside 'data/raw/'")
        print("Expected filename: WA_Fn-UseC_-Telco-Customer-Churn.csv")
        print("Path: data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
        print("!" * 60 + "\n")
        sys.exit(1)

    print(f"Loading raw dataset from: {csv_path}")
    raw_df = pd.read_csv(csv_path)

    # 1. Step 1: Inspect data
    inspect_raw_data(raw_df)

    # 2. Step 2: Clean and engineer features
    clean_df = clean_and_engineer_features(raw_df)

    # 3. Step 2 continued: Leak-free train/test split and pipeline scaling/encoding
    build_and_apply_pipeline(clean_df)


if __name__ == "__main__":
    main()
