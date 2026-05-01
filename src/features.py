import pandas as pd
import numpy as np

def build_features(df):
    """
    Takes a raw price history DataFrame and adds useful features.
    Returns a new DataFrame ready for modeling.
    """

    df = df.copy()

    # Ensure proper types
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["log_price"] = np.log1p(df["price"])
    df["player_name"] = df["player_name"].astype(str)

    # Sort by player and date
    df = df.sort_values(["player_id", "date"])

    # Group object
    g = df.groupby("player_id")

    # Lag features
    df["lag_1"] = g["log_price"].shift(1)
    df["lag_3"] = g["log_price"].shift(3)
    df["lag_7"] = g["log_price"].shift(7)

    # Moving averages
    df["ma_3"] = g["log_price"].rolling(3).mean().reset_index(level=0, drop=True)
    df["ma_7"] = g["log_price"].rolling(7).mean().reset_index(level=0, drop=True)

    # Volatility
    df["pct_change"] = g["log_price"].pct_change()
    df["vol_3"] = g["pct_change"].rolling(3).std().reset_index(level=0, drop=True)
    df["vol_7"] = g["pct_change"].rolling(7).std().reset_index(level=0, drop=True)

    # Calendar features (same for all players)
    df["day_of_week"] = df["date"].dt.weekday
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Target: next day's price (per player)
    df["next_day_price"] = g["log_price"].shift(-1)

    df["date"] = df["date"].dt.date

    # Drop rows with missing values
    df = df.dropna()

    return df
