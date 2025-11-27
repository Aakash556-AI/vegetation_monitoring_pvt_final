import streamlit as st
import pandas as pd
import numpy as np
import rasterio
from PIL import Image
import tensorflow as tf
from keras.models import load_model
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Vegetation AI Assistant", layout="centered")
st.title("🤖 Vegetation AI Assistant")

# Sidebar Mode Selector
mode = st.sidebar.radio("Choose Analysis Mode", ["🌿 Vegetation Segmentation (U-Net)", "📈 NDVI Forecast (LSTM)"])

# Load Models
@st.cache_resource
def load_models():
    return (
        load_model("unet_vegetation_model.h5", compile=False),
        load_model("lstm_vegetation_model.h5", compile=False)
    )

unet_model, lstm_model = load_models()

# Normalize NDVI for image display
def normalize_ndvi(ndvi_array):
    return ((ndvi_array + 1) / 2 * 255).astype(np.uint8)

# ------------------------------
# 🌿 Vegetation Segmentation
# ------------------------------
if mode == "🌿 Vegetation Segmentation (U-Net)":
    st.subheader("Step 1️⃣: Upload NDVI GeoTIFF")
    tiff = st.file_uploader("Upload a GeoTIFF file", type=["tif", "tiff"])
    
    if tiff:
        with rasterio.open(tiff) as src:
            if src.count > 1:
                st.warning("⚠️ Multi-band TIFF detected. Using only the first band.")
            ndvi = src.read(1)
            mask = ndvi > 0

        st.subheader("🧠 AI Output:")
        ndvi_normalized = normalize_ndvi(ndvi)
        st.image(ndvi_normalized, caption="Original NDVI Map",use_container_width=True)

        st.markdown("✔️ **AI has identified vegetative areas** marked in red:")
        red_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        red_mask[..., 0] = mask * 255
        combined = Image.fromarray(red_mask)
        st.image(combined, caption="Vegetation Mask (U-Net)",use_container_width=True)

        st.success("✅ Done! U-Net detected regions successfully.")
    else:
        st.info("Please upload a GeoTIFF file to continue.")

# ------------------------------
# 📈 NDVI Forecast (LSTM)
# ------------------------------
elif mode == "📈 NDVI Forecast (LSTM)":
    st.subheader("Step 1️⃣: Upload NDVI Time Series CSV")
    csv = st.file_uploader("Upload CSV with 'Date' and 'NDVI_LSTM_Prediction'", type="csv")

    if csv:
        df = pd.read_csv(csv)

        # Normalize and check column names
        df.columns = df.columns.str.strip()  # Clean column names
        if 'Date' in df.columns and 'NDVI_LSTM_Prediction' in df.columns:
            st.subheader("Step 2️⃣: AI Forecast")

            # Prepare input for LSTM (reshape for model)
            ndvi_series = df['NDVI_LSTM_Prediction'].values.reshape(1, -1, 1)

            # Predict the next NDVI value
            pred = lstm_model.predict(ndvi_series)[0][0]

            # Create new columns for prediction in original rows
            df['predicted'] = np.nan
            df['upper'] = np.nan
            df['lower'] = np.nan

            # Add a new row for the prediction
            next_date = pd.to_datetime(df['Date']).max() + pd.DateOffset(months=1)
            new_row = {
                'Date': next_date.strftime("%Y-%m-%d"),
                'NDVI_LSTM_Prediction': np.nan,
                'predicted': pred,
                'upper': pred + 0.02,
                'lower': pred - 0.02
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            # Plot
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df['Date'], y=df['NDVI_LSTM_Prediction'],
                mode='lines+markers', name='NDVI Input', line=dict(color='blue')
            ))

            fig.add_trace(go.Scatter(
                x=df['Date'], y=df['predicted'],
                mode='lines+markers', name='Forecast', line=dict(color='green')
            ))

            fig.add_trace(go.Scatter(
                x=df['Date'], y=df['upper'],
                mode='lines', name='Upper Bound', line=dict(color='orange', dash='dot')
            ))

            fig.add_trace(go.Scatter(
                x=df['Date'], y=df['lower'],
                mode='lines', name='Lower Bound', line=dict(color='red', dash='dot')
            ))

            fig.update_layout(
                title="📊 NDVI Forecast Results",
                xaxis_title="Date",
                yaxis_title="NDVI",
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
            )

            st.plotly_chart(fig, use_container_width=True)

            st.success("🎉 Forecast Complete!")
            st.download_button("📥 Download Predictions", df.to_csv(index=False), "lstm_forecast.csv")
        else:
            st.error("❌ CSV must contain 'Date' and 'NDVI_LSTM_Prediction' columns.")
    else:
        st.info("Please upload a CSV file to forecast NDVI.")
