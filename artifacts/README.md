# Deployment Artifacts

This folder contains the files required to run the Streamlit historical forecasting demo:

- `preprocessor.joblib` — fitted categorical preprocessing pipeline
- `xgb_model.json` — trained XGBoost regressor
- `demo_test_data.parquet` — lightweight historical demo dataset

Run the application with:

```bash
streamlit run app/app.py
