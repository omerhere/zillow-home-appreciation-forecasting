# Zillow 12-Month Home Appreciation Forecasting

A leakage-aware machine learning project that forecasts 12-month Zillow Home Value Index (ZHVI) appreciation across US regions.

## Problem

The goal is to predict the percentage change in a region's Zillow Home Value Index over the following 12 months.

[
\text{Appreciation}*{12M} =
\frac{\text{HomeValue}*{t+12} - \text{HomeValue}*{t}}
{\text{HomeValue}*{t}} \times 100
]

This is a regression problem.

## Why Appreciation Instead of Raw Home Value?

Predicting near-term raw home values using lagged prices can produce extremely high R² because home prices are highly persistent.

This project instead predicts 12-month appreciation, which is more difficult but more meaningful because it forecasts future market movement rather than reproducing a stable price level.

## Key Features

* Historical home-value lags: 1, 3, 6, and 12 months
* Recent growth rates: 3, 6, and 12 months
* Rolling mean and market-volatility features
* Seasonal month sine and cosine features
* Geographic features: State, Metro, County, RegionType, and SizeRank

## Leakage Prevention

* Chronological train, validation, and test split
* Gap years between datasets to reduce overlap in 12-month target windows
* Lag and rolling features calculated using only historical information
* Preprocessors fitted only on training data
* Future value and target columns excluded from model inputs

## Models Evaluated

| Model                      | MAE ↓ | RMSE ↓ |   R² ↑ |
| -------------------------- | ----: | -----: | -----: |
| Dummy Mean Baseline        | 3.717 |  4.855 | -0.306 |
| Naive Momentum Baseline    | 3.719 |  5.316 | -0.567 |
| SGDRegressor               | 4.021 |  5.152 | -0.471 |
| Random Forest, 400k sample | 3.005 |  4.091 |  0.073 |
| Tuned XGBoost              | 2.975 |  4.048 |  0.092 |

XGBoost was selected because it achieved the strongest validation performance across MAE, RMSE, and R².

## Streamlit Demonstration

A Streamlit application was built to explore historical forecasts.

The user can select:

* State
* Region
* Historical prediction date

The app retrieves the relevant engineered feature row, applies the saved preprocessor, generates a 12-month appreciation forecast, and compares it against the known historical outcome.

> This is a historical forecasting demonstration. The source data ends in March 2020, so the app does not provide live real-estate forecasts.

## Project Structure

```text
├── Zillow_Home_Appreciation_Forecasting.ipynb
├── app.py
├── requirements.txt
├── images/
└── README.md
```

## Tech Stack

Python, Pandas, Scikit-learn, XGBoost, Random Forest, Streamlit, Plotly, Joblib.

## Key Learning

High R² alone is not proof of a useful machine learning model. Target definition, baseline comparison, chronology, and leakage prevention are equally important.
