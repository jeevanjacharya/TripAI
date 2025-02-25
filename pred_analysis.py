import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

def train_arima_model(data, column, order=(5,1,0)):
    """
    Train ARIMA model for time series forecasting.

    Parameters:
        data (pd.DataFrame): The dataset containing the time series.
        column (str): The column name to be forecasted.
        order (tuple): The ARIMA model order (p, d, q).

    Returns:
        model_fit: The trained ARIMA model.
    """
    # Ensure the column exists in data
    if column not in data.columns:
        st.error(f"Column '{column}' not found in dataset.")
        return None

    # Convert the column to a numeric format (if needed)
    data[column] = pd.to_numeric(data[column], errors='coerce')
    data = data.dropna()  # Remove NaN values
    
    # Fit ARIMA model
    model = ARIMA(data[column], order=order)
    model_fit = model.fit()

    return model_fit

def predict_future_values(model_fit, steps=10):
    """
    Generate future predictions using the trained ARIMA model.

    Parameters:
        model_fit: The trained ARIMA model.
        steps (int): Number of future steps to predict.

    Returns:
        forecast (np.array): Predicted future values.
    """
    forecast = model_fit.forecast(steps=steps)
    return forecast

def plot_forecast(data, column, forecast, steps):
    """
    Plot historical data along with forecasted values.

    Parameters:
        data (pd.DataFrame): The dataset containing the time series.
        column (str): The column name to be forecasted.
        forecast (np.array): Forecasted values.
        steps (int): Number of future steps.

    Returns:
        None
    """
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data[column], label="Actual Data", color='blue')
    
    # Creating future index
    future_index = np.arange(len(data), len(data) + steps)
    plt.plot(future_index, forecast, label="Predicted Data", color='red', linestyle="dashed")

    plt.xlabel("Time")
    plt.ylabel(column)
    plt.title("ARIMA Forecast")
    plt.legend()
    plt.grid()
    st.pyplot(plt)

def predictive_analysis():
    """
    Streamlit interface for performing predictive analysis on time series data.
    """
    st.title("🔮 Predictive Analysis using ARIMA")

    uploaded_file = st.file_uploader("Upload a CSV file with a time series column", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)

        st.write("### Data Preview")
        st.write(df.head())

        column = st.selectbox("Select a column for forecasting:", df.columns)
        steps = st.slider("Select number of future steps to predict", min_value=5, max_value=50, value=10, step=5)

        if st.button("Run ARIMA Prediction"):
            model_fit = train_arima_model(df, column)
            if model_fit:
                forecast = predict_future_values(model_fit, steps)
                st.write("### Forecasted Values")
                st.write(forecast)
                plot_forecast(df, column, forecast, steps)

if __name__ == "_main_":
    predictive_analysis()