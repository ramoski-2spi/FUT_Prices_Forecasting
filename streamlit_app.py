import streamlit as st
import pandas as pd
import numpy as np
import joblib

from src.scraper import get_price_history
from src.features import build_features
from src.modeling import load_model, load_residual_std

st.set_page_config(layout="wide")
# -----------------------------
# Load model + residual std
# -----------------------------
@st.cache_resource
def load_all():
    model = load_model("data/models/xgb_model.pkl")
    residual_std = load_residual_std("data/models/residual_std.pkl")
    return model, residual_std

model, residual_std = load_all()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("FUT Console Next-Day Price Predictor")
st.write("Enter any FUT.GG player ID to predict their next-day price.")
st.write("""STEPS:
        \n1. Go to FUT.GG and search for a player you want.
        \n2. Copy the player ID after "26-" in the URL (e.g. https://www.fut.gg/players/158023-lionel-messi/26-158023/)
        \n3. Paste it into the "Player ID" input below (e.g. 158023).
        \n4. Enter the choosen player slug in the "Player Slug" below (e.g. lionel-messi).
        \n5. Press "Predict" to get the expected next-day price of the player.""")

player_id = st.text_input("Enter FUT.GG Player ID")
player_slug = st.text_input("Enter FU.GG Player Slug (optional)")

if st.button("Predict"):

    if not player_id or not player_slug:
        st.error("Please enter both Player ID and Player Slug.")
    else:
        st.write(f"Scraping FUT.GG data for **{player_slug}**...")

        df = get_price_history(player_id, player_slug)

        if df is None or df.empty:
            st.error("Could not fetch price history for this player.")
        else:
            st.success("Data scraped successfully!")

            # Show historical chart
            df_plot = df.copy()
            df_plot["date"] = pd.to_datetime(df_plot["date"])
            st.subheader("Historical Price Chart")
            st.line_chart(df_plot.set_index("date")["price"])

            # Build features
            df_feat = build_features(df)
            latest = df_feat.iloc[-1]

            feature_cols = ["lag_1", "lag_3", "lag_7",
                "ma_3", "ma_7","pct_change", "vol_3", 
                "vol_7", "day_of_week", "is_weekend"]

            X = latest[feature_cols].values.reshape(1, -1)

            # Predict log price
            log_pred = model.predict(X)[0]
            pred_price = np.expm1(log_pred)

            # Confidence interval
            lower_log = log_pred - 1.645 * residual_std
            upper_log = log_pred + 1.645 * residual_std

            lower = np.expm1(lower_log)
            upper = np.expm1(upper_log)

            # Display results
            st.subheader("Next-Day Price Prediction")
            st.metric("Predicted Price", f"{int(pred_price):,} coins")

            st.write(f"**90% Confidence Interval:** {int(lower):,} - {int(upper):,} coins")

st.markdown("""
    <div style='text-align: center; font-size: 14px; margin-top: 40px; opacity: 0.7;'>
        Data provided by <a href="https://www.fut.gg" target="_blank">FUT.gg</a>
    </div>
    """,
    unsafe_allow_html=True)
