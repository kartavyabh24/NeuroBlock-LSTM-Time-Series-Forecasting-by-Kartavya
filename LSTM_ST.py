import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import warnings
import streamlit as st

# ignore warnings
warnings.filterwarnings("ignore")

# Streamlit configuration
st.set_page_config(layout = "wide")
st.markdown("<style>.main {padding-top: 0px;}</style>", unsafe_allow_html = True)

# Add images
st.sidebar.image("Pic1.png", width = "stretch")
st.image("Pic2.png", width = "stretch")

# Add main title
st.markdown("<h1 style = 'text-align: center; margin-top: -20px; '>LSTM Forecasting Model</h1>", unsafe_allow_html = True)

# Sidebar inputs
st.sidebar.header("Model Parameters")
crypto_symbol = st.sidebar.text_input("Cryptocurrency Symbol", "BTC-USD")
prediction_ahead = st.sidebar.number_input("Prediction Days Ahead", min_value = 1, max_value = 30, value = 15, step = 1)

if st.sidebar.button("Predict"):
    # Step 1: Pull Crypto data for the past 1 year.

    btc_data = yf.download(
        crypto_symbol, period = '1y', interval = '1d', progress = False, threads = False
    )
    btc_data = btc_data[['Close']].dropna()

    if btc_data.empty:
        st.error(f"No data returned for symbol '{crypto_symbol}'.")
        st.stop()

    # Prepare data for LSTM
    scaler = MinMaxScaler(feature_range = (0,1))
    scaled_data = scaler.fit_transform(btc_data).astype(np.float32)

    # Correct split for training and testing datasets
    train_size = int(len(scaled_data) * 0.8)

    def create_dataset(data, time_step = 1):
        x, y = [], []
        for i in range(len(data) - time_step):
            x.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(x, dtype = np.float32), np.array(y, dtype = np.float32)

    # Use 80% of the total data for training and 20% for testing.
    time_step = 60
    if len(scaled_data) <= time_step or train_size <= time_step:
        st.error("Not enough data points to train the model with a 60-day window.")
        st.stop()

    x_train, y_train = create_dataset(scaled_data[:train_size], time_step)
    x_test, y_test = create_dataset(scaled_data[train_size - time_step:], time_step)

    # Reshape input to be [samples, time_steps, features]
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    # Build LSTM Model
    model = Sequential()
    model.add(LSTM(50, return_sequences = True, input_shape = (time_step, 1)))
    model.add(LSTM(50, return_sequences = False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    model.fit(x_train, y_train, batch_size = 16, epochs = 5, verbose = 0)

    # Make predictions
    train_predictions = model.predict(x_train, verbose = 0)
    test_predictions = model.predict(x_test, verbose = 0)

    # Inverse transform predictions and actual values
    train_predictions = scaler.inverse_transform(train_predictions)
    y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
    test_predictions = scaler.inverse_transform(test_predictions)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Forecasting for future days
    last_60_days = scaled_data[-time_step:]
    future_input = last_60_days.reshape(1, time_step, 1)
    future_forecast = []
    for _ in range(int(prediction_ahead)):
        next_pred = model.predict(future_input, verbose = 0)[0, 0]
        future_forecast.append(next_pred)
        next_input = np.append(future_input[0, 1:], [[next_pred]], axis = 0)
        future_input = next_input.reshape(1, time_step, 1)

    future_forecast = scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1)).ravel()

    # Latest close price and Last predicted price
    latest_close_price = float(btc_data['Close'].iloc[-1])
    last_predicted_price = float(future_forecast[-1])

    # Centered layout for metrics
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            f"""
            <div style = "display: flex; justify-content: space-around;">
                <div style = "background-color: #d5f5d5; color: black; padding: 10px; border-radius: 10px; text-align: center;">
                    <h3>Latest Close Price</h3>
                    <p style = "font-size: 20px;">${latest_close_price:,.2f}</p>
                </div>
                <div style = "background-color: #d5f5d5; color: black; padding: 10px; border-radius: 10px; text-align: center;">
                    <h3>Price After {prediction_ahead} Days</h3>
                    <p style = "font-size: 20px;">${last_predicted_price:,.2f}</p>
                </div>
            </div>
            """,
            unsafe_allow_html = True,
        )

    # Plot the results
    fig, ax = plt.subplots(figsize = (14, 5))
    ax.plot(btc_data.index, btc_data['Close'], label = 'Actual', color = 'blue')
    ax.axvline(x = btc_data.index[train_size], color = 'gray', linestyle = '--', label = 'Train/Test Split')

    # Train/Test and Predictions
    train_range = btc_data.index[time_step:train_size]
    test_range = btc_data.index[train_size:train_size + len(test_predictions)]
    ax.plot(train_range, train_predictions[:len(train_range)], label = 'Train Predictions', color = 'green')
    ax.plot(test_range, test_predictions[:len(test_range)], label = 'Test Predictions', color = 'orange')

    # Future Predictions
    future_index = pd.date_range(start = btc_data.index[-1], periods = int(prediction_ahead) + 1, freq = 'D')[1:]
    ax.plot(future_index, future_forecast, label = f'{prediction_ahead}-Day Forecast', color = 'red')

    ax.set_title(f'{crypto_symbol} LSTM Model Engine')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# Streamlit run LSTM_ST.py