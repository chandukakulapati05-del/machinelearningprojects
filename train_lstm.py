import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model  # Add load_model
from tensorflow.keras.layers import LSTM, Dense
import joblib

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("data/SupplyChainDataset.csv")  # Forward slash

# Convert to datetime
df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])

# Aggregate daily demand
daily = (
    df.groupby(df["order_purchase_timestamp"].dt.date)["quantity"]
    .sum()
)

daily = daily.to_frame()
daily.index = pd.to_datetime(daily.index)
daily = daily.asfreq("D").fillna(0)

if len(daily) < 60:
    print("Not enough data to train LSTM")
    exit()

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(daily)

# Create sequences
def create_sequences(data, window=30):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)

# CRITICAL FIX: Reshape X to 3D (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], 1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X, y, epochs=30, batch_size=16)

# Save model and scaler
model.save("models/lstm_model.h5")
joblib.dump(scaler, "models/scaler.save")

print("Model trained and saved successfully.")