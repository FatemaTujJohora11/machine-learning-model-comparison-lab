"""Portfolio adaptation of the original Lab 2 console regression workflow.

Evaluates the five validation-selected models documented in the final report.
It does NOT rerun the original hyperparameter grid or guarantee historical
metrics. Car-price CSV is required locally and is not redistributed.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

CATEGORICAL = ["fuel type", "seller type", "transmission"]


def main():
    csv_path = Path("car_prices.csv")
    if not csv_path.exists():
        raise FileNotFoundError("Place an authorized car_prices.csv in this directory.")
    df = pd.read_csv(csv_path)
    target = "selling price"
    if target not in df.columns:
        raise ValueError(f"Expected target column: {target}")
    X, y = df.drop(columns=[target]), df[target].astype(float)
    numerical = [c for c in X.columns if c not in CATEGORICAL]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, shuffle=True)

    models = {
        "Linear Regression": LinearRegression(
            fit_intercept=True, copy_X=True, positive=False),
        "KNN": KNeighborsRegressor(
            n_neighbors=6, weights="uniform", p=1,
            algorithm="auto", leaf_size=25),
        "Decision Tree": DecisionTreeRegressor(
            criterion="squared_error", splitter="best", max_depth=10,
            min_samples_split=5, min_samples_leaf=2,
            random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=625, criterion="squared_error", max_depth=8,
            min_samples_split=5, min_samples_leaf=2,
            max_features=1.0, bootstrap=False, random_state=42),
        "SVR": SVR(kernel="rbf", C=4.5, tol=0.0003, max_iter=5500,
                   gamma=0.1, coef0=0, degree=3),
    }
    for name, estimator in models.items():
        def build_model():
            preprocessing = ColumnTransformer([
                ("num", StandardScaler(), numerical),
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ])
            pipeline = Pipeline([
                ("preprocess", preprocessing),
                ("model", estimator),
            ])
            if name == "SVR":
                return TransformedTargetRegressor(
                    regressor=pipeline, transformer=StandardScaler())
            return pipeline

        model = build_model()
        model.fit(X_train, y_train)
        val_rmse = np.sqrt(mean_squared_error(y_val, model.predict(X_val)))
        X_full = pd.concat([X_train, X_val])
        y_full = pd.concat([y_train, y_val])
        final = build_model()
        final.fit(X_full, y_full)
        predictions = final.predict(X_test)
        print(f"{name} | validation RMSE: {val_rmse:,.2f} | "
              f"test RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):,.2f} | "
              f"test MAE: {mean_absolute_error(y_test, predictions):,.2f} | "
              f"test R2: {r2_score(y_test, predictions):.4f}")
    print("Portfolio run, not an exact reproduction of historical console outputs.")


if __name__ == "__main__":
    main()
