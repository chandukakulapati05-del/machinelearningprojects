import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import joblib
from utils import *
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Supply Chain Optimization", layout="wide")
st.title("📦 Supply Chain Optimization Dashboard")

# Create directories
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Check if data exists
data_path = "data/SupplyChainDataset.csv"
if not os.path.exists(data_path):
    st.error("⚠️ Data file not found! Please run `python generate_data.py` first to generate sample data.")
    st.code("python generate_data.py", language="bash")
    st.stop()

# Load Data
df = load_data(data_path)

# Sidebar
st.sidebar.header("Inventory Settings")
lead_time = st.sidebar.slider("Lead Time (days)", 1, 30, 7)
ordering_cost = st.sidebar.number_input("Ordering Cost", 100, 5000, 500)
holding_percent = st.sidebar.slider("Holding Cost %", 1, 50, 20)

# Create Demand
demand_df = create_demand(df)

# Get top products by total demand
product_demand = demand_df.groupby("product_id")["daily_demand"].sum().sort_values(ascending=False)
top_products = product_demand.head(20).index.tolist()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Available Products:** {len(top_products)}")

selected_product = st.sidebar.selectbox("Select Product", top_products)

product_data = demand_df[demand_df["product_id"] == selected_product].copy()
product_data = product_data.sort_values("date")

# Show product info
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Product Data Points:** {len(product_data)} days")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Demand", "📦 Inventory", "🚚 Logistics", "🤖 LSTM Forecast"]
)

# ---------------- Demand
with tab1:
    st.subheader("Daily Demand Analysis")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Demand", int(product_data["daily_demand"].sum()))
    col2.metric("Avg Daily Demand", round(product_data["daily_demand"].mean(), 2))
    col3.metric("Max Daily Demand", int(product_data["daily_demand"].max()))
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(product_data["date"], product_data["daily_demand"], color='#2E86AB', linewidth=1.5)
    ax.fill_between(product_data["date"], product_data["daily_demand"], alpha=0.3, color='#2E86AB')
    ax.set_title(f"Daily Demand - {selected_product}", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Quantity")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    with st.expander("View Raw Data"):
        st.dataframe(product_data.tail(10))

# ---------------- Inventory
with tab2:
    st.subheader("Inventory Optimization (EOQ & ROP)")
    
    annual_demand = product_data["daily_demand"].sum()
    avg_demand = product_data["daily_demand"].mean()
    demand_std = product_data["daily_demand"].std()

    avg_price = df[df["product_id"] == selected_product]["price"].mean()
    holding_cost = (holding_percent / 100) * avg_price

    eoq = calculate_eoq(annual_demand, ordering_cost, holding_cost)
    rop = calculate_rop(avg_demand, lead_time, demand_std)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Annual Demand", f"{int(annual_demand):,}")
    col2.metric("Avg Price", f"${avg_price:.2f}")
    col3.metric("EOQ", f"{int(eoq):,} units")
    col4.metric("Reorder Point", f"{int(rop):,} units")

# ---------------- Logistics
with tab3:
    st.subheader("Logistics & Shipping Analysis")
    
    product_orders = df[df["product_id"] == selected_product].copy()
    
    if len(product_orders) > 0:
        product_orders["distance_km"] = product_orders.apply(
            lambda x: haversine(
                x["seller_lat"], x["seller_lng"],
                x["customer_lat"], x["customer_lng"]
            ), axis=1
        )
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Orders", len(product_orders))
        col2.metric("Avg Distance (km)", f"{product_orders['distance_km'].mean():.1f}")
        col3.metric("Avg Freight Cost", f"${product_orders['freight_value'].mean():.2f}")
    else:
        st.warning("No order data available for this product.")

# ---------------- LSTM
with tab4:
    st.subheader("LSTM Demand Forecasting")
    
    demand_series = product_data["daily_demand"].values
    
    st.info(f"📊 Data points available: {len(demand_series)} (minimum required: 60)")
    
    if len(demand_series) >= 60:
        model_path = f"models/lstm_{selected_product}.h5"
        scaler_path = f"models/scaler_{selected_product}.save"
        
        # Check if model exists - if corrupted, delete and retrain
        model_exists = os.path.exists(model_path) and os.path.exists(scaler_path)
        
        if model_exists:
            try:
                # Try to load with compile=False (fixes the error)
                model = load_model(model_path, compile=False)
                model.compile(optimizer='adam', loss='mse')  # Recompile
                scaler = joblib.load(scaler_path)
                st.success("✅ Loaded existing model")
            except Exception as e:
                # If loading fails, delete and retrain
                st.warning(f"Model corrupted, retraining... ({str(e)[:50]})")
                if os.path.exists(model_path):
                    os.remove(model_path)
                if os.path.exists(scaler_path):
                    os.remove(scaler_path)
                model_exists = False
        
        if not model_exists:
            with st.spinner("Training LSTM model... This may take a moment."):
                try:
                    model, scaler = train_lstm(demand_series, model_path, scaler_path)
                    st.success("✅ Model trained successfully!")
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")
                    st.stop()
        
        # Generate forecast
        forecast_days = st.slider("Forecast Days", 7, 30, 14)
        
        with st.spinner("Generating forecast..."):
            forecast = forecast_lstm(model, scaler, demand_series, forecast_days)
        
        # Create future dates
        future_dates = pd.date_range(
            start=product_data["date"].max() + pd.Timedelta(days=1),
            periods=forecast_days
        )
        
        # Plot
        fig4, ax4 = plt.subplots(figsize=(14, 6))
        
        # Historical data (last 60 days)
        hist_days = min(60, len(demand_series))
        ax4.plot(product_data["date"].tail(hist_days), 
                demand_series[-hist_days:], 
                label="Historical Demand", color='#2E86AB', linewidth=2)
        
        # Forecast
        ax4.plot(future_dates, forecast, 
                label=f"{forecast_days}-Day Forecast", 
                color='#F18F01', linewidth=2, linestyle='--')
        
        ax4.set_title(f"Demand Forecast - {selected_product}", fontsize=14)
        ax4.set_xlabel("Date")
        ax4.set_ylabel("Daily Demand")
        ax4.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig4)
        
        # Forecast metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Forecast Avg", f"{np.mean(forecast):.1f}")
        col2.metric("Forecast Max", f"{np.max(forecast):.1f}")
        col3.metric("Forecast Min", f"{np.max([0, np.min(forecast)]):.1f}")
        
    else:
        st.warning("⚠️ Not enough data to train LSTM. Need at least 60 days of data.")
        st.info("Try selecting a different product with more historical data.")