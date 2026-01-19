import numpy as np
import pandas as pd
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import shap
import streamlit as st
import plotly.express as px
import joblib

def generate_energy_data(days=30, freq='H'):
    timestamps = pd.date_range(end=datetime.now(), periods=days*24, freq=freq)
    base_demand = 1000 + 200*np.sin(2*np.pi*timestamps.hour/24) + np.random.normal(0,50,len(timestamps))
    renewable = 300 + 150*np.sin(2*np.pi*(timestamps.hour-6)/24) + np.random.normal(0,30,len(timestamps))
    df = pd.DataFrame({'timestamp': timestamps, 'demand': base_demand, 'renewable': renewable})
    return df

def preprocess(df):
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['demand'] = df['demand'].interpolate()
    df['renewable'] = df['renewable'].interpolate()
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x>=5 else 0)
    for lag in [1,24,168]:
        df[f'demand_lag_{lag}'] = df['demand'].shift(lag)
        df[f'renewable_lag_{lag}'] = df['renewable'].shift(lag)
    df = df.dropna()
    return df

def train_arima(series, order=(5,1,0)):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit

def train_xgboost(df, target='demand'):
    X = df.drop(columns=['timestamp', target])
    y = df[target]
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X, y)
    return model

def forecast_arima(model_fit, steps=24):
    return model_fit.forecast(steps=steps)

def forecast_xgboost(model, df):
    X = df.drop(columns=['timestamp','demand'])
    preds = model.predict(X)
    return preds

def evaluate(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def explain_model(model, df):
    X = df.drop(columns=['timestamp','demand'])
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X)
    return shap_values

def run_dashboard(df, forecast):
    st.title("AI Energy Forecast Dashboard")
    df['forecast'] = forecast
    st.subheader("Energy Data Preview")
    st.dataframe(df.tail(20))
    fig = px.line(df, x='timestamp', y=['demand','forecast'], title='Energy Demand Forecast')
    st.plotly_chart(fig)
    st.subheader("Forecast vs Actual MAE")
    mae = evaluate(df['demand'], df['forecast'])
    st.write(f"Mean Absolute Error: {mae:.2f} MW")

if __name__ == "__main__":
    df = generate_energy_data(days=60)
    df_processed = preprocess(df)
    arima_model = train_arima(df_processed['demand'])
    arima_forecast = forecast_arima(arima_model, steps=len(df_processed))
    xgb_model = train_xgboost(df_processed)
    xgb_forecast = forecast_xgboost(xgb_model, df_processed)
    joblib.dump(xgb_model, "xgb_energy_model.pkl")
    shap_values = explain_model(xgb_model, df_processed)
    run_dashboard(df_processed, xgb_forecast)
