"""
This script orchestrates the full daily ML pipeline:
1. Scrape FUTBIN player price history
2. Build processed feature dataset
3. Train XGBoost model
4. Compute residual_std for confidence intervals
5. Save all trained models into data/models/

GitHub Actions will run this script on demand.
"""

import pandas as pd
from pathlib import Path
import time

# Import pipeline functions
from src.scraper import PLAYERS, get_price_history
from src.features import build_features
from src.modeling import train_model,compute_residual_std
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, DEFAULT_MODEL_PATH, DEFAULT_RESIDUAL_STD_PATH

# ---------------------------------------------------------
# STEP 1 — SCRAPE FUTBIN DATA
# ---------------------------------------------------------
def run_scraper():
    """
    Scrapes FUT.GG for all players defined in the filter list.
    Saves raw CSV into data/raw/.
    """
    print("🔍 Running scraper...")

    # Get list of players to scrape
    players = PLAYERS

    # Scrape price history for each player
    all_data = []
    for player_id, slug in players.items():
        df = get_price_history(player_id, slug)       
        if df is not None:
            all_data.append(df)

        time.sleep(4)  # avoid rate limiting

    if not all_data:
        print("No data scraped.")
        return

    # Concatenate all scraped data
    full_df = pd.concat(all_data, ignore_index=True)

    # Save raw data
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DATA_DIR / "fut_prices_raw.csv"
    full_df.to_csv(raw_path, index=False)

    print(f"✅ Scraping complete. Saved raw data to {raw_path}")

# ---------------------------------------------------------
# STEP 2 — BUILD PROCESSED FEATURE DATASET
# ---------------------------------------------------------
def run_feature_builder():
    """
    Builds the processed dataset used for model training.
    Saves to data/processed/fut_prices_features.csv.
    """
    print("🛠️ Building feature dataset...")

    df = pd.read_csv(RAW_DATA_DIR / "fut_prices_raw.csv")
    processed_df = build_features(df)  # your feature builder

    # Save processed dataset
    PROCESSED_DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    processed_path = PROCESSED_DATA_DIR/"fut_prices_features.csv"
    processed_df.to_csv(processed_path, index=False)

    print(f"✅ Feature dataset saved to {processed_path}")

# ---------------------------------------------------------
# STEP 3 — TRAIN MODEL
# ---------------------------------------------------------
def run_training():
    """
    Trains the XGBoost model using modeling.py logic.
    Saves model to data/models/xgb_model.pkl.
    """
    print("🤖 Training model...")

    df = pd.read_csv(PROCESSED_DATA_DIR/"fut_prices_features.csv")
    model, splits = train_model(df)

    print(f"✅ Model saved to {DEFAULT_MODEL_PATH}")
    return model, splits

# ---------------------------------------------------------
# STEP 4 — COMPUTE RESIDUAL STD
# ---------------------------------------------------------
def run_residual_std(model, splits):
    """
    Computes residual standard deviation on the test set.
    Saves residual_std.pkl for Streamlit confidence intervals.
    """
    print("📏 Computing residual standard deviation...")

    _, _, _, _, X_test, y_test = splits #Only take X_test and y_test
    compute_residual_std(model, X_test, y_test)

    print(f"✅ Residual std saved to {DEFAULT_RESIDUAL_STD_PATH}")

# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------
def main():
    print("\n🚀 Starting full automation pipeline...\n")

    run_scraper()
    run_feature_builder()
    model, splits = run_training()
    run_residual_std(model, splits)

    print("\n🎉 Pipeline complete!\n")

if __name__ == "__main__":
    main()
