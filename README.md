# FUT Price Forecasting

![Pipeline Status](https://github.com/ramoski-2spi/FUT_Prices_Forecasting/actions/workflows/train_pipeline.yml/badge.svg)           

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?logo=streamlit&logoColor=white)](https://futpricesforecasting-jcbgfj8szw7uqdffatkzvb.streamlit.app/)

A modular machine learning pipeline and Streamlit application for forecasting FIFA Ultimate Team (FUT) player prices using historical market data from **FUT.gg**.  
This project demonstrates a real-world, production-style workflow: data ingestion, feature engineering, model training, evaluation, deployment, and CI automation.

---

## Business Problem

The FIFA Ultimate Team (FUT) transfer market behaves like a real-time digital economy where player prices fluctuate rapidly based on supply, demand, in‑game events, and market sentiment. Players, traders, and content creators often struggle to anticipate price movements, leading to:

- inefficient buying and selling decisions  
- missed investment opportunities  
- increased market risk  
- difficulty valuing players in real time  

### How This Project Solves It

This project transforms raw FUT market data into actionable insights through:

- **Automated data collection** from FUT.gg  
- **Feature-rich historical datasets** for analysis  
- **An XGBoost machine learning model** that predicts future player prices  
- **Uncertainty estimates** to quantify prediction confidence  
- **A Streamlit web app** that makes predictions accessible to non-technical users  
- **A reproducible GitHub Actions pipeline** that retrains the model in a clean environment  

This demonstrates how machine learning can support decision-making in fast-moving digital marketplaces.

---

## Features

- **Automated Data Scraping**  
  Scrapes historical FUT player prices from FUT.gg for a configurable list of players.

- **Feature Engineering**  
  Converts raw time-series data into supervised ML features (lags, rolling stats, trends).

- **XGBoost Price Prediction Model**  
  Trains a regression model and saves all artifacts to `data/models/`.

- **Uncertainty Estimates**  
  Computes residual standard deviation for confidence intervals in the UI.

- **Streamlit App**  
  Interactive interface for selecting players, viewing price history, and generating predictions.

- **GitHub Actions Pipeline**  
  Runs the entire pipeline in a clean cloud environment and uploads artifacts automatically.

---

## Project Structure
```
FUT-PRICE-FORECASTING/
├── notebooks/
│   ├── 01_EDA.ipynb                     # Exploratory Data Analysis (EDA)
│   ├── 02_features.ipynb                # Feature engineering + dataset construction
│   ├── 03_XGBoost_model.ipynb           # Model training, evaluation, residual analysis
│
├── src/
│   ├── scraper.py                       # FUT.gg scraping logic
│   ├── features.py                      # Feature engineering utilities
│   ├── modeling.py                      # Model training + residual std computation
│   ├── config.py                        # Paths and constants
│   └── predict.py                       # Prediction utilities for Streamlit
│
├── automation.py                        # Full pipeline orchestrator (scrape → features → train)
├── streamlit_app.py                     # Deployed Streamlit web UI
│
├── data/
│   ├── raw/                             # Raw scraped FUT price data
│   ├── processed/                       # Engineered feature datasets
│   └── models/                          # Trained model + residual_std artifacts
│
└── .github/
    └── workflows/
        └── train_pipeline.yml           # GitHub Actions CI pipeline

```
---
## Notebooks Overview
# 01_EDA (Exploratory Data Analysis)

This notebook explores the raw FUT 26 price history data scraped from FUTBIN.

The dataset includes a diverse set of players:
- cheap → expensive  
- gold → special → legendary 
- meta → non‑meta  

**Note:** The dataset currently includes only male players.  
Adding women’s players is a valuable improvement opportunity for future versions of the project.

---
# 02_Feature (Feature Engineering & Dataset Construction)
This notebook focuses on converting the scraped FUT price history into a structured, supervised learning dataset suitable for machine learning.

It covers:

- cleaning and validating raw scraped data

- handling missing values, duplicates, and irregular timestamps

- generating time‑series features (lags, rolling windows, trends)

The goal is to transform noisy, real‑world market data into a clean, consistent dataset that captures meaningful price dynamics.
This step is critical for enabling the XGBoost model to learn short‑term and long‑term market behavior.

---
# 03_XGBoost_Model (Model Training &Evaluation)
This notebook trains the machine learning model that powers the FUT price forecasting system.

It includes:

- splitting the processed dataset into train/validation/test sets

- training an XGBoost regression model on engineered features

- evaluating performance using error metrics and visual diagnostics

- analyzing residuals to understand model uncertainty

- computing the residual standard deviation, later used for confidence intervals in the Streamlit app

---
## How It Works

1. **Scraper**  
   Fetches price history for each player in `PLAYERS` (player ID + optional slug).

2. **Feature Builder**  
   Processes raw data into a modeling-ready dataset.

3. **Model Training**  
   Uses XGBoost to learn price patterns and trends.

4. **Residual Std Calculation**  
   Estimates prediction uncertainty for confidence intervals.

5. **Streamlit App**  
   Loads the model + residuals and provides interactive predictions.

6. **GitHub Actions**  
   Runs `automation.py` on demand and uploads:
   - `data/raw/`
   - `data/processed/`
   - `data/models/`

---

## How to Run the Project

### Try the Live App (No Installation Required)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?logo=streamlit&logoColor=white)](https://futpricesforecasting-jcbgfj8szw7uqdffatkzvb.streamlit.app/)

### Run the Automated Pipeline in GitHub Actions (No Cloning Required)
You can run the full ML pipeline directly from GitHub:

1. Go to the **Actions** tab  
2. Select **FUT Price Pipeline**  
3. Click **Run workflow**

This executes the entire pipeline in a clean cloud environment.

### Run the Pipeline Locally (Cloning Required)
```
bash

git clone https://github.com/ramoski-2spi/FUT_Prices_Forecasting.git

cd FUT_Prices_Forecasting

pip install -r requirements.txt

python automation.py
```
---

## Data Attribution

Price data is sourced from FUT.gg.