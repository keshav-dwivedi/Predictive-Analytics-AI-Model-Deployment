"""
train.py
========
Trains 4 machine learning models on the preprocessed training dataset:
1. Logistic Regression (baseline linear model)
2. Random Forest (ensemble of decision trees)
3. Gradient Boosting (sequential boosting of decision trees)
4. MLPClassifier (multi-layer neural network for deep learning)

Why we use 5-Fold Cross-Validation:
Instead of relying on a single test score, 5-fold cross-validation splits
the training data into 5 equal parts, trains on 4, tests on 1, and repeats
5 times. This ensures the model's performance is consistent and not a fluke.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate


def load_training_data():
    """
    Loads preprocessed training data from data/processed/.
    Includes basic error handling if preprocessing hasn't been run yet.
    """
    x_path = "data/processed/X_train.csv"
    y_path = "data/processed/y_train.csv"

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print("\n[ERROR] Preprocessed training files not found!")
        print("Please run 'python preprocess.py' first to prepare the data.\n")
        sys.exit(1)

    X_train = pd.read_csv(x_path)
    # y_train is loaded as a 1D Series for scikit-learn
    y_train = pd.read_csv(y_path).squeeze("columns")

    print(f"Loaded training data successfully: {X_train.shape[0]} rows, {X_train.shape[1]} features.")
    return X_train, y_train


def get_models():
    """
    Initializes our 4 models with beginner-friendly explanations for parameters.
    """
    models = {
        # WHY Logistic Regression: Fast, highly interpretable baseline.
        # class_weight='balanced' gives higher penalty to churn mistakes to handle class imbalance.
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ),

        # WHY Random Forest: Strong ensemble model that handles non-linear patterns
        # and doesn't easily overfit due to bagging (bootstrap aggregating).
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        ),

        # WHY Gradient Boosting: Builds trees sequentially, each correcting errors of the previous one.
        # Often achieves top-tier accuracy on tabular data.
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        ),

        # WHY MLPClassifier (Neural Network): Fulfills the deep learning requirement.
        # Two hidden layers (64 and 32 neurons) with ReLU activation to learn complex feature interactions.
        "MLPClassifier (Neural Net)": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            random_state=42
        )
    }
    return models


def train_and_cross_validate(models, X_train, y_train):
    """
    Evaluates each model using 5-Fold Stratified Cross-Validation,
    then fits the model on the full training set and saves it.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # StratifiedKFold maintains the same 73:27 ratio of retain:churn across each fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    print("\n" + "=" * 65)
    print("STEP 3: 5-FOLD CROSS-VALIDATION ON TRAINING DATA")
    print("=" * 65)

    cv_summary = []

    for name, model in models.items():
        print(f"\nTraining and validating: {name}...")

        # Perform 5-Fold Cross-Validation
        scores = cross_validate(
            model, X_train, y_train,
            cv=cv,
            scoring=scoring_metrics,
            n_jobs=-1
        )

        acc = scores['test_accuracy'].mean()
        f1 = scores['test_f1'].mean()
        roc = scores['test_roc_auc'].mean()
        prec = scores['test_precision'].mean()
        rec = scores['test_recall'].mean()

        print(f"  -> Mean CV Accuracy : {acc:.4f}")
        print(f"  -> Mean CV F1-Score : {f1:.4f}")
        print(f"  -> Mean CV ROC-AUC  : {roc:.4f}")

        cv_summary.append({
            "Model": name,
            "CV Accuracy": round(acc, 4),
            "CV Precision": round(prec, 4),
            "CV Recall": round(rec, 4),
            "CV F1": round(f1, 4),
            "CV ROC-AUC": round(roc, 4)
        })

        # Fit model on the full training set for future testing and deployment
        model.fit(X_train, y_train)

        # Save individual model to models/
        file_slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        save_path = f"models/{file_slug}.joblib"
        joblib.dump(model, save_path)
        print(f"  -> Saved fitted model to {save_path}")

    # Save CV summary report
    cv_df = pd.DataFrame(cv_summary)
    cv_df.to_csv("reports/cv_metrics.csv", index=False)
    print("\n" + "=" * 65)
    print("Cross-Validation Summary:")
    print(cv_df.to_string(index=False))
    print("=" * 65)
    print("Saved CV metrics table to 'reports/cv_metrics.csv'.\n")


def main():
    X_train, y_train = load_training_data()
    models = get_models()
    train_and_cross_validate(models, X_train, y_train)


if __name__ == "__main__":
    main()
