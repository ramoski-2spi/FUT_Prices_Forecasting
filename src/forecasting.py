import pandas as pd
from src.features import build_features
from src.modeling import load_model


def forecast_future(df_raw, days=30, model_path="data/models/xgb_model.pkl"):
    """
    Simple recursive forecasting:
    - take last known price
    - predict next day
    - append prediction
    - repeat
    """

    df = build_features(df_raw)
    model = load_model(model_path)

    future_rows = []

    last_df = df.copy()

    for _ in range(days):
        # Build features for the last row
        X = last_df.drop(["next_day_price", "date", "player_id", "player_name"], axis=1).iloc[-1:]
        pred = model.predict(X)[0]

        next_date = pd.to_datetime(last_df["date"].iloc[-1]) + pd.Timedelta(days=1)

        new_row = {
            "player_id": df_raw["player_id"].iloc[0],
            "player_name": df_raw["player_name"].iloc[0],
            "date": next_date.date(),
            "price": pred,
        }

        future_rows.append(new_row)

        # Append prediction to df to generate next features
        last_df = pd.concat([last_df, pd.DataFrame([new_row])], ignore_index=True)
        last_df = build_features(last_df)

    return pd.DataFrame(future_rows)

