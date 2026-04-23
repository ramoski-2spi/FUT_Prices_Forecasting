import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
from src.config import DEFAULT_MODEL_PATH, DEFAULT_RESIDUAL_STD_PATH


# ---------------------------------------------------------
# TRAIN MODEL (matches notebook exactly)
# ---------------------------------------------------------
def train_model(df, model_path=DEFAULT_MODEL_PATH):
    """
    Trains an XGBoost model using the same split logic as the notebook:
    70% train, 15% validation, 15% test (time-ordered).
    Target = log_price.
    """
    df = df.copy()

    # Target
    y = df["log_price"]

    # Features
    X = df.drop(["next_day_price", "log_price", "date", "player_id", "player_name"], axis=1)

    # --- MATCH NOTEBOOK SPLIT ---
    n = len(df)
    train_size = int(n * 0.7)
    valid_size = int(n * 0.85)

    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]

    X_valid = X.iloc[train_size:valid_size]
    y_valid = y.iloc[train_size:valid_size]

    X_test = X.iloc[valid_size:]
    y_test = y.iloc[valid_size:]

    # Model
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )

    # Evaluate on TEST SET (same as notebook)
    preds = model.predict(X_test)
    mae = np.mean(np.abs(np.expm1(y_test) - np.expm1(preds)))

    print(f"Test MAE: {mae:.2f} coins")

    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    return model, (X_train, y_train, X_valid, y_valid, X_test, y_test)


# ---------------------------------------------------------
# RESIDUAL STD FOR CONFIDENCE INTERVALS
# ---------------------------------------------------------
def compute_residual_std(model, X_test, y_test, save_path=DEFAULT_RESIDUAL_STD_PATH):
    """
    Computes residual standard deviation in log space using TEST SET ONLY.
    This matches the notebook and produces correct CIs.
    """

    preds = model.predict(X_test)
    residuals = y_test - preds
    residual_std = residuals.std()

    joblib.dump(residual_std, save_path)
    print(f"Residual std saved to {save_path}: {residual_std:.4f}")

    return residual_std


# ---------------------------------------------------------
# LOADERS
# ---------------------------------------------------------
def load_model(model_path=DEFAULT_MODEL_PATH):
    return joblib.load(model_path)


def load_residual_std(path=DEFAULT_RESIDUAL_STD_PATH):
    return joblib.load(path)