import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import plotly.express as px

st.set_page_config(
    page_title="Zillow Appreciation Forecaster",
    page_icon="🏠",
    layout="wide"
)

@st.cache_resource
def load_model_artifacts():
    preprocessor = joblib.load("artifacts/preprocessor.joblib")

    model = XGBRegressor()
    model.load_model("artifacts/xgb_model.json")

    return preprocessor, model


@st.cache_data
def load_data():
    df = pd.read_parquet("artifacts/demo_test_data.parquet")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


preprocessor, model = load_model_artifacts()
df = load_data()

numeric_features = [
    "HomeValue",
    "SizeRank",
    "year",
    "Month_Sin",
    "Month_Cos",
    "HomeValue_Lag_1M",
    "HomeValue_Lag_3M",
    "HomeValue_Lag_6M",
    "HomeValue_Lag_12M",
    "Growth_3M_pct",
    "Growth_6M_pct",
    "Growth_12M_pct",
    "RollingMean_3M",
    "RollingStd_6M"
]

categorical_features = [
    "RegionType",
    "State",
    "Metro",
    "CountyName"
]

feature_cols = numeric_features + categorical_features

st.title("Zillow 12-Month Home Appreciation Forecaster")
st.caption(
    "Historical forecasting demonstration using XGBoost and Zillow Home Value Index data."
)

state = st.sidebar.selectbox(
    "Select State",
    sorted(df["State"].dropna().unique())
)

state_df = df[df["State"] == state]

region_options = (
    state_df[["RegionID", "RegionName", "CountyName", "Metro"]]
    .drop_duplicates()
    .sort_values("RegionName")
)

region_options["Label"] = (
    region_options["RegionName"].astype(str)
    + " | "
    + region_options["CountyName"].astype(str)
    + " | "
    + region_options["Metro"].astype(str)
)

selected_label = st.sidebar.selectbox(
    "Select Region",
    region_options["Label"].tolist()
)

selected_region_id = region_options.loc[
    region_options["Label"] == selected_label,
    "RegionID"
].iloc[0]

region_df = (
    df[df["RegionID"] == selected_region_id]
    .sort_values("Date")
    .copy()
)

selected_date = st.sidebar.selectbox(
    "Prediction Date",
    region_df["Date"].dt.strftime("%Y-%m-%d").tolist()
)

selected_date = pd.to_datetime(selected_date)

selected_row = region_df[
    region_df["Date"] == selected_date
].iloc[[0]]

X_selected = selected_row[feature_cols]

X_selected_processed = preprocessor.transform(X_selected).astype("float32")

predicted_appreciation = float(model.predict(X_selected_processed)[0])

current_value = float(selected_row["HomeValue"].iloc[0])

predicted_future_value = current_value * (1 + predicted_appreciation / 100)

actual_appreciation = float(
    selected_row["Actual_Appreciation_12M"].iloc[0]
)

actual_future_value = current_value * (1 + actual_appreciation / 100)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Current ZHVI",
    f"${current_value:,.0f}"
)

col2.metric(
    "Predicted 12-Month Appreciation",
    f"{predicted_appreciation:.2f}%"
)

col3.metric(
    "Predicted Future ZHVI",
    f"${predicted_future_value:,.0f}"
)

absolute_error = abs(actual_appreciation - predicted_appreciation)

st.subheader("Historical Backtest: What Actually Happened")

col4, col5, col6 = st.columns(3)

col4.metric(
    "Actual 12-Month Appreciation",
    f"{actual_appreciation:.2f}%"
)

col5.metric(
    "Actual Future ZHVI",
    f"${actual_future_value:,.0f}"
)

col6.metric(
    "Forecast Error",
    f"{absolute_error:.2f} percentage points"
)

st.subheader("Historical Test Performance: Demo Market Subset")

st.caption(
    "Metrics below are calculated across all selected demo-market observations "
    "from the held-out historical test period, not only the currently selected region."
)

overall_mae = mean_absolute_error(
    df["Actual_Appreciation_12M"],
    df["Predicted_Appreciation_12M"]
)

overall_rmse = mean_squared_error(
    df["Actual_Appreciation_12M"],
    df["Predicted_Appreciation_12M"]
) ** 0.5

overall_r2 = r2_score(
    df["Actual_Appreciation_12M"],
    df["Predicted_Appreciation_12M"]
)

metric1, metric2 = st.columns(2)

metric1.metric(
    "Average Forecast Error",
    f"{overall_mae:.2f} percentage points"
)

metric2.metric(
    "Large-Error Sensitive Score",
    f"{overall_rmse:.2f} percentage points"
)

with st.expander("Technical metric details"):
    st.write(f"Subset R²: {overall_r2:.3f}")
    st.caption(
        "R² can change substantially across different geographic subsets and "
        "time periods. It is included for technical interpretation only."
    )

st.subheader("Home Value Trend")

fig = px.line(
    region_df,
    x="Date",
    y="HomeValue",
    title=f"ZHVI Trend: {selected_row['RegionName'].iloc[0]}"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Model Inputs Used")

input_display = selected_row[
    [
        "HomeValue",
        "Growth_3M_pct",
        "Growth_6M_pct",
        "Growth_12M_pct",
        "RollingStd_6M",
        "SizeRank",
        "State",
        "Metro",
        "CountyName"
    ]
].T

input_display.columns = ["Value"]

st.dataframe(input_display, use_container_width=True)

st.info(
    "This is a historical forecasting demonstration. The dataset ends in March 2020, "
    "so this interface does not provide live 2026 real-estate forecasts."
)
