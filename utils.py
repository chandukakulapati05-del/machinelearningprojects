import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from math import radians, cos, sin, asin, sqrt

def load_data(file_path):
    df = pd.read_csv(file_path)
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["quantity"] = pd.to_numeric(df["quantity"], errors='coerce').fillna(1)
    df["price"] = pd.to_numeric(df["price"], errors='coerce').fillna(0)
    df["freight_value"] = pd.to_numeric(df["freight_value"], errors='coerce').fillna(0)
    for col in ["seller_lat", "seller_lng", "customer_lat", "customer_lng"]:
        if col not in df.columns:
            df[col] = 0.0
    return df

def create_demand(df):
    demand_df = df.copy()
    demand_df["date"] = pd.to_datetime(demand_df["order_purchase_timestamp"]).dt.date
    demand_df["date"] = pd.to_datetime(demand_df["date"])
    daily_demand = demand_df.groupby(["product_id", "date"])["quantity"].sum().reset_index()
    daily_demand.rename(columns={"quantity": "daily_demand"}, inplace=True)
    all_dates = pd.date_range(daily_demand["date"].min(), daily_demand["date"].max(), freq='D')
    filled_data = []
    for product_id in daily_demand["product_id"].unique():
        product_data = daily_demand[daily_demand["product_id"] == product_id].copy()
        product_data = product_data.set_index("date").reindex(all_dates, fill_value=0)
        product_data["product_id"] = product_id
        product_data = product_data.reset_index().rename(columns={"index": "date"})
        filled_data.append(product_data)
    return pd.concat(filled_data, ignore_index=True)

def calculate_eoq(annual_demand, ordering_cost, holding_cost):
    if holding_cost <= 0 or annual_demand <= 0:
        return 0
    return np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)

def calculate_rop(avg_daily_demand, lead_time, demand_std, service_level=1.65):
    if lead_time <= 0:
        return avg_daily_demand
    safety_stock = service_level * demand_std * np.sqrt(lead_time)
    return (avg_daily_demand * lead_time) + safety_stock

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c

def train_lstm(demand_series, model_path, scaler_path):
    if len(demand_series) < 60:
        raise ValueError(f"Need at least 60 data points, got {len(demand_series)}")
    data = demand_series.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    window = 30
    X, y = [], []
    for i in range(window, len(scaled_data)):
        X.append(scaled_data[i-window:i])
        y.append(scaled_data[i])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(window, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')  # Use string 'mse'
    model.fit(X, y, epochs=20, batch_size=16, verbose=0)
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    return model, scaler

def forecast_lstm(model, scaler, demand_series, steps=14):
    window = 30
    last_window = demand_series[-window:].reshape(-1, 1)
    last_window_scaled = scaler.transform(last_window)
    predictions = []
    current_batch = last_window_scaled.reshape((1, window, 1))
    for _ in range(steps):
        pred = model.predict(current_batch, verbose=0)
        predictions.append(pred[0, 0])
        current_batch = np.append(current_batch[:, 1:, :], [[[pred[0, 0]]]], axis=1)
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    return predictions.flatten()