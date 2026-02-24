import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import warnings
import streamlit as st
from llm_provider import explain

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

# LM Studio / IBM Watson Studio
st.sidebar.header("LLM Settings")

provider = st.sidebar.selectbox("LLM Provider", ["LM Studio", "IBM watsonx"])

# LM Studio model name must match what LM Studio shows in its server UI
lmstudio_model = st.sidebar.text_input("LM Studio model name", "llama-3-8b-instruct-finance-rag")

# IBM model_id examples depend on your watsonx account; you can change later
watsonx_model_id = st.sidebar.text_input("IBM watsonx model_id", "ibm/granite-4-h-small")

model_name = lmstudio_model if provider == "LM Studio" else watsonx_model_id

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

    # Create a prompt using your computed numbers (after we compute metrics)
    predicted_change_pct = ((last_predicted_price - latest_close_price) / latest_close_price) * 100.0

    prompt = f"""
    Educational use only (not financial advice).

    Asset: {crypto_symbol}
    Latest close price: ${latest_close_price:,.2f}
    Predicted price after {int(prediction_ahead)} day(s): ${last_predicted_price:,.2f}
    Predicted change (%): {predicted_change_pct:.2f}

    Task:
    1) Explain what this forecast suggests in simple terms.
    2) Give a risk label (Low/Medium/High) and why.
    3) Give 3 bullet points for "what to watch next".
    Rules:
    - Use only the numbers above.
    - Do not claim certainty or guaranteed returns.
    """

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

    # Adding the “Explain with AI” button (after our chart or after metrics)
    st.subheader("Explain with AI")

    if st.button("Explain with AI"):
        try:
            explanation = explain(provider, prompt, model_name)
            st.write(explanation)
        except Exception as e:
            st.error(f"LLM error: {e}")

    # Add a Chat box
    st.subheader("Chat (grounded)")

    user_q = st.text_input("Ask a question about this forecast (example: 'Explain risk', 'What does +% mean?')")

    if user_q:
        chat_prompt = f"""
    Educational use only.

    Context:
    Asset: {crypto_symbol}
    Latest close: ${latest_close_price:,.2f}
    Predicted after {int(prediction_ahead)} day(s): ${last_predicted_price:,.2f}
    Change (%): {predicted_change_pct:.2f}

    User question: {user_q}

    Rules:
    - Use only the context numbers.
    - If user asks for something not in context, say what is missing.
    """
        try:
            ans = explain(provider, chat_prompt, model_name)
            st.write(ans)
        except Exception as e:
            st.error(f"LLM error: {e}")

# Streamlit run LSTM_ST.py

# export WATSONX_APIKEY="D-F_5HKTRZ_m_a7NyYMIQmukwDDIf-QDg8wJ2OiNGWR_"
# export WATSONX_URL="https://us-south.ml.cloud.ibm.com"
# export WATSONX_PROJECT_ID="cc04fdd9-fb2c-46f0-9992-afe0f130b4cb"