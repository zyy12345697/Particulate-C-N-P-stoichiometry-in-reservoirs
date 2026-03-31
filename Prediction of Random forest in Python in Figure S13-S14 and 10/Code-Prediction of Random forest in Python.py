# -*- coding: utf-8 -*-
"""
Random Forest regression workflow for tabular data.

This script includes:
1. Data loading and preprocessing
2. K-fold cross-validation
3. Final model training and hold-out evaluation
4. Scatter plots of observed vs. predicted values
5. Model saving

"""

import math
import warnings
from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Global settings
RANDOM_SEED = 100
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def load_dataset(
    file_path: str,
    feature_columns: slice = slice(1, 6),
    target_column: int = 0
) -> Tuple[pd.DataFrame, np.ndarray, pd.Series, pd.Index]:
    """
    Load dataset from an Excel file and perform preprocessing.

    Parameters
    ----------
    file_path : str
        Path to the Excel file.
    feature_columns : slice, optional
        Column range for features, by default slice(1, 6).
    target_column : int, optional
        Column index for target variable, by default 0.

    Returns
    -------
    df : pd.DataFrame
        Original dataframe.
    X_processed : np.ndarray
        Normalized and standardized feature matrix.
    y : pd.Series
        Target variable.
    numeric_cols : pd.Index
        Names of numeric feature columns used in modeling.
    """
    df = pd.read_excel(file_path)

    X = df.iloc[:, feature_columns]
    y = df.iloc[:, target_column]

    numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns
    X_numeric = X[numeric_cols].copy()

    # Min-max normalization
    X_normalized = X_numeric.apply(
        lambda col: (col - col.min()) / (col.max() - col.min())
        if col.max() != col.min() else 0
    )

    # Standardization
    scaler = StandardScaler()
    X_processed = scaler.fit_transform(X_normalized)

    return df, X_processed, y, numeric_cols


def plot_real_vs_predicted(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    dataset_type: str,
    output_dir: str
) -> None:
    """
    Plot observed versus predicted values and export plot data.

    Parameters
    ----------
    y_true : pd.Series
        True target values.
    y_pred : np.ndarray
        Predicted target values.
    model_name : str
        Model name.
    dataset_type : str
        Dataset label, e.g., 'Train Set' or 'Test Set'.
    output_dir : str
        Directory for saving exported plot data.
    """
    r2 = r2_score(y_true, y_pred)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, label="Predictions")
    plt.plot(
        [min(y_true), max(y_true)],
        [min(y_true), max(y_true)],
        "r--",
        label="y = x"
    )
    plt.title(f"{model_name} - {dataset_type} (R$^2$ = {r2:.4f})")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plot_data = pd.DataFrame({
        "True Values": y_true,
        "Predicted Values": y_pred
    })
    file_name = f"{model_name}_{dataset_type.lower().replace(' ', '_')}_plot_data.xlsx"
    plot_data.to_excel(output_path / file_name, index=False)


def evaluate_model(
    model: RandomForestRegressor,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str,
    output_dir: str
) -> None:
    """
    Evaluate model performance on training and test sets.

    Parameters
    ----------
    model : RandomForestRegressor
        Trained regression model.
    X_train : np.ndarray
        Training feature matrix.
    X_test : np.ndarray
        Test feature matrix.
    y_train : pd.Series
        Training target values.
    y_test : pd.Series
        Test target values.
    model_name : str
        Model name.
    output_dir : str
        Directory for saving exported plot data.
    """
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_rmse = math.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = math.sqrt(mean_squared_error(y_test, y_test_pred))

    print(f"R²  Train: {train_r2:.4f} | Test: {test_r2:.4f}")
    print(f"RMSE Train: {train_rmse:.4f} | Test: {test_rmse:.4f}")

    plot_real_vs_predicted(
        y_true=y_train,
        y_pred=y_train_pred,
        model_name=model_name,
        dataset_type="Train Set",
        output_dir=output_dir
    )
    plot_real_vs_predicted(
        y_true=y_test,
        y_pred=y_test_pred,
        model_name=model_name,
        dataset_type="Test Set",
        output_dir=output_dir
    )


def fit_with_random_forest(
    X: np.ndarray,
    y: pd.Series,
    n_splits: int = 5,
    test_size: float = 0.1,
    output_dir: str = "./results"
) -> RandomForestRegressor:
    """
    Train and evaluate a Random Forest regressor using K-fold CV
    and a final train-test split.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : pd.Series
        Target variable.
    n_splits : int, optional
        Number of folds for cross-validation, by default 5.
    test_size : float, optional
        Proportion of test set in final split, by default 0.1.
    output_dir : str, optional
        Directory for saving output files, by default './results'.

    Returns
    -------
    RandomForestRegressor
        Final trained model.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_r2_scores = []
    test_r2_scores = []

    for fold_idx, (train_index, test_index) in enumerate(kf.split(X), start=1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        rf = RandomForestRegressor(
            random_state=42,
            n_estimators=1000,
            max_depth=25,
            min_samples_leaf=1,
            min_samples_split=2,
            max_features="sqrt"
        )
        rf.fit(X_train, y_train)

        y_train_pred = rf.predict(X_train)
        y_test_pred = rf.predict(X_test)

        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        train_r2_scores.append(train_r2)
        test_r2_scores.append(test_r2)

        print(
            f"Fold {fold_idx}: "
            f"Train R² = {train_r2:.4f}, Test R² = {test_r2:.4f}"
        )

    print("\n===== Cross-validation Results =====")
    print(f"Mean Train R²: {np.mean(train_r2_scores):.4f}")
    print(f"Train R² by fold: {train_r2_scores}")
    print(f"Mean Test R²: {np.mean(test_r2_scores):.4f}")
    print(f"Test R² by fold: {test_r2_scores}")
    print("====================================")

    print("\n===== Final Train-Test Evaluation =====")
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42
    )

    final_rf = RandomForestRegressor(
        random_state=42,
        n_estimators=1000,
        max_depth=25,
        min_samples_leaf=2,
        min_samples_split=2,
        max_features="sqrt"
    )
    final_rf.fit(X_train_full, y_train_full)

    evaluate_model(
        model=final_rf,
        X_train=X_train_full,
        X_test=X_test_full,
        y_train=y_train_full,
        y_test=y_test_full,
        model_name="Random_Forest_Final",
        output_dir=output_dir
    )

    model_path = Path(output_dir) / "random_forest_model.pkl"
    joblib.dump(final_rf, model_path)
    print(f"Model saved to: {model_path}")

    return final_rf


def main() -> None:
    """
    Main execution function.
    """
    file_path = "your_path_here.xlsx"
    output_dir = "./results"

    _, X, y, numeric_cols = load_dataset(file_path)
    print("Selected numeric feature columns:")
    print(list(numeric_cols))

    fit_with_random_forest(
        X=X,
        y=y,
        n_splits=5,
        test_size=0.1,
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()