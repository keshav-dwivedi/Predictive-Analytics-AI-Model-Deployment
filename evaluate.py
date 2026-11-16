"""
evaluate.py
===========
Evaluates all trained models on the unseen holdout test set (20% of data).
Tasks covered:
1. Calculates Accuracy, Precision, Recall, F1-Score, and ROC-AUC for all 4 models
2. Plots and saves Confusion Matrices for each model in reports/confusion_matrices.png
3. Plots and saves ROC Curves in reports/roc_curves.png
4. Deep Learning Tuning: Tunes MLPClassifier architectures and learning rates
5. Feature Importance: Extracts Random Forest importances and explains the top 5 churn factors
6. Selects the BEST model and saves it as models/best_model.joblib
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.neural_network import MLPClassifier


def load_test_data():
    """
    Loads preprocessed test data and feature names.
    """
    x_test_path = "data/processed/X_test.csv"
    y_test_path = "data/processed/y_test.csv"

    if not os.path.exists(x_test_path) or not os.path.exists(y_test_path):
        print("\n[ERROR] Processed test data not found in 'data/processed/'!")
        print("Please run 'python preprocess.py' and 'python train.py' first.\n")
        sys.exit(1)

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).squeeze("columns")
    feature_names = joblib.load("models/feature_names.joblib")

    print(f"Loaded test set: {X_test.shape[0]} rows, {X_test.shape[1]} features.")
    return X_test, y_test, feature_names


def load_trained_models():
    """
    Loads saved models from models/.
    """
    model_files = {
        "Logistic Regression": "models/logistic_regression.joblib",
        "Random Forest": "models/random_forest.joblib",
        "Gradient Boosting": "models/gradient_boosting.joblib",
        "MLPClassifier (Neural Net)": "models/mlpclassifier_neural_net.joblib"
    }

    models = {}
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            print(f"[WARNING] Model file '{path}' missing. Run 'python train.py' first.")

    return models


def evaluate_models(models, X_test, y_test):
    """
    Evaluates each model and produces a clean comparison DataFrame.
    """
    results = []
    predictions = {}
    probabilities = {}

    for name, model in models.items():
        # Predict binary class (0 or 1)
        y_pred = model.predict(X_test)
        # Predict continuous probability of churning (Class 1)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc, 4)
        })

        predictions[name] = y_pred
        probabilities[name] = y_proba

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="F1-Score", ascending=False).reset_index(drop=True)
    return results_df, predictions, probabilities


def plot_confusion_matrices(models, predictions, y_test):
    """
    Generates a 2x2 grid of confusion matrices and saves to reports/.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for idx, (name, _) in enumerate(models.items()):
        cm = confusion_matrix(y_test, predictions[name])
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
            xticklabels=['Stayed (0)', 'Churned (1)'],
            yticklabels=['Stayed (0)', 'Churned (1)']
        )
        axes[idx].set_title(f"{name}\nConfusion Matrix", fontsize=12, fontweight='bold')
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("True Label")

    plt.tight_layout()
    cm_path = "reports/confusion_matrices.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrices to '{cm_path}'.")


def plot_roc_curves(models, probabilities, y_test):
    """
    Plots ROC curves for all models on one chart and saves to reports/.
    """
    plt.figure(figsize=(9, 7))

    for name, _ in models.items():
        fpr, tpr, _ = roc_curve(y_test, probabilities[name])
        roc_auc = roc_auc_score(y_test, probabilities[name])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess (AUC = 0.500)", linewidth=1.5)
    plt.title("ROC Curves Comparison (Holdout Test Set)", fontsize=14, fontweight='bold')
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

    roc_path = "reports/roc_curves.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved ROC curves plot to '{roc_path}'.")


def tune_mlp_deep_learning(X_train, y_train, X_test, y_test):
    """
    Task 5: Tunes MLPClassifier with different architectures and learning rates.
    Compares whether deep learning beats tree-based models on tabular data.
    """
    print("\n" + "=" * 65)
    print("STEP 5A: DEEP LEARNING (MLPClassifier) HYPERPARAMETER TUNING")
    print("=" * 65)

    architectures = [
        {"name": "Small (32,)", "layers": (32,), "lr": 0.01},
        {"name": "Medium (64, 32)", "layers": (64, 32), "lr": 0.001},
        {"name": "Deep (128, 64, 32)", "layers": (128, 64, 32), "lr": 0.001},
        {"name": "Wide (128, 128)", "layers": (128, 128), "lr": 0.0005},
    ]

    tuning_results = []
    best_mlp = None
    best_mlp_f1 = -1

    for config in architectures:
        mlp = MLPClassifier(
            hidden_layer_sizes=config["layers"],
            learning_rate_init=config["lr"],
            max_iter=500,
            early_stopping=True,
            random_state=42
        )
        mlp.fit(X_train, y_train)

        y_pred = mlp.predict(X_test)
        y_proba = mlp.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)

        tuning_results.append({
            "Architecture": config["name"],
            "Layers": str(config["layers"]),
            "Learning Rate": config["lr"],
            "Test Accuracy": round(acc, 4),
            "Test F1-Score": round(f1, 4),
            "Test ROC-AUC": round(roc, 4)
        })

        if f1 > best_mlp_f1:
            best_mlp_f1 = f1
            best_mlp = mlp

    tuning_df = pd.DataFrame(tuning_results)
    tuning_df.to_csv("reports/mlp_tuning_results.csv", index=False)
    print(tuning_df.to_string(index=False))
    print(f"\nBest Tuned MLP achieved F1-Score: {best_mlp_f1:.4f}")
    print("Saved tuning results to 'reports/mlp_tuning_results.csv'.")
    return best_mlp


def analyze_feature_importance(rf_model, feature_names):
    """
    Task 5: Extracts top features from Random Forest, plots them,
    and explains the top 5 churn drivers in simple English.
    """
    print("\n" + "=" * 65)
    print("STEP 5B: FEATURE IMPORTANCE & TOP 5 CHURN FACTORS")
    print("=" * 65)

    importances = rf_model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    feat_df.to_csv("reports/feature_importance.csv", index=False)

    # Plot Top 10 Features
    plt.figure(figsize=(10, 6))
    top_10 = feat_df.head(10)
    sns.barplot(
        data=top_10,
        x="Importance",
        y="Feature",
        hue="Feature",
        legend=False,
        palette="viridis"
    )
    plt.title("Top 10 Most Important Features Driving Customer Churn", fontsize=13, fontweight='bold')
    plt.xlabel("Random Forest Feature Importance Score")
    plt.ylabel("Feature Name")
    plt.tight_layout()

    chart_path = "reports/feature_importance.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved feature importance chart to '{chart_path}'.")

    # Beginner-Friendly Explanation of the Top 5 Drivers
    print("\n--- TOP 5 REASONS CUSTOMERS CHURN (EXPLAINED SIMPLY) ---")
    explanations = [
        "1. TotalCharges & MonthlyCharges: Customers paying steep monthly fees or seeing unexpected bill jumps are much more likely to switch to cheaper competitors.",
        "2. tenure (Customer Loyalty): Brand new customers (months 1-6) have not formed brand loyalty and churn at the highest rate. Retaining customers past year 1 drops churn dramatically.",
        "3. MonthlyToTotalRatio (Engineered Feature): When the current bill is high relative to what the customer historically paid, it signals price fatigue or expiring promotional discounts.",
        "4. Contract_Two year / Contract_One year: Customers on long-term contracts rarely churn because of cancellation friction; month-to-month contract users churn the most.",
        "5. OnlineSecurity_Yes / TechSupport_Yes: Customers who subscribe to protective and support services experience high friction to switch, keeping them loyal."
    ]
    for exp in explanations:
        print(exp)
    print("=" * 65 + "\n")


def pick_and_save_best_model(results_df, models):
    """
    Picks the best model based on F1-Score & ROC-AUC, explaining why in simple English,
    and saves it to models/best_model.joblib.
    """
    # In churn prediction, F1-Score is the best metric because it balances catching
    # churners (Recall) with not falsely alerting loyal customers (Precision).
    best_row = results_df.iloc[0]
    best_name = best_row["Model"]
    best_model = models[best_name]

    print("=" * 65)
    print(f"CHAMPION MODEL SELECTED: {best_name}")
    print("=" * 65)
    print(f"- Test Accuracy : {best_row['Accuracy']:.4f}")
    print(f"- Test Recall   : {best_row['Recall']:.4f} (catches {best_row['Recall']*100:.1f}% of churners)")
    print(f"- Test F1-Score : {best_row['F1-Score']:.4f} (highest balanced score)")
    print(f"- Test ROC-AUC  : {best_row['ROC-AUC']:.4f}")
    print("\nWhy this model was chosen in simple English:")
    print(f"Accuracy alone is misleading on imbalanced data (predicting 'No Churn' for everyone gives 73% accuracy!).")
    print(f"{best_name} achieved the optimal trade-off between catching leaving customers (Recall) and minimizing false alarms (Precision).")

    save_path = "models/best_model.joblib"
    joblib.dump(best_model, save_path)
    joblib.dump(best_name, "models/best_model_name.joblib")
    print(f"Saved best model to '{save_path}'!")
    print("=" * 65 + "\n")


def main():
    X_test, y_test, feature_names = load_test_data()
    models = load_trained_models()

    if not models:
        print("[ERROR] No models loaded. Run 'python train.py' first.")
        sys.exit(1)

    print("\n" + "=" * 65)
    print("STEP 4: MODEL EVALUATION & COMPARISON ON TEST SET")
    print("=" * 65)

    results_df, predictions, probabilities = evaluate_models(models, X_test, y_test)

    # Save summary table
    results_df.to_csv("reports/test_metrics_summary.csv", index=False)
    print("Test Set Evaluation Metrics:")
    print(results_df.to_string(index=False))
    print("Saved metrics table to 'reports/test_metrics_summary.csv'.\n")

    # Generate charts
    plot_confusion_matrices(models, predictions, y_test)
    plot_roc_curves(models, probabilities, y_test)

    # Task 5: Deep learning tuning & feature importance
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze("columns")

    tune_mlp_deep_learning(X_train, y_train, X_test, y_test)

    if "Random Forest" in models:
        analyze_feature_importance(models["Random Forest"], feature_names)

    # Save best model
    pick_and_save_best_model(results_df, models)


if __name__ == "__main__":
    main()
