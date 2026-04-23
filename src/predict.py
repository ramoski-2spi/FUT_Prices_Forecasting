from src.scraper import get_price_history
from src.features import build_features
from src.modeling import load_model

def predict_tomorrow(player_id, slug, model_path="data/models/xgb_model.pkl"):
    """
    Full pipeline:
    - scrape data
    - build features
    - load model
    - predict next day's price
    """

    df_raw = get_price_history(player_id, slug)

    if df_raw is None or df_raw.empty:
        print("Could not fetch price history.")
        return None

    df_feat = build_features(df_raw)

    if df_feat.empty:
        print("Not enough data to build features.")
        return None

    model = load_model(model_path)

    last_row = df_feat.drop(["next_day_price", "date", "player_id", "player_name"], axis=1).iloc[-1:]
    pred = model.predict(last_row)[0]

    return pred

