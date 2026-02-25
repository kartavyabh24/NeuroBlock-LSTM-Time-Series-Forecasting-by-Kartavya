import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import warnings
import streamlit as st
from llm_provider import safe_explain, open_chat_answer, get_provider_options

# ignore warnings
warnings.filterwarnings("ignore")

# Streamlit configuration
st.set_page_config(layout = "wide")
st.markdown("<style>.main {padding-top: 0px;}</style>", unsafe_allow_html = True)

# Session state
if "results" not in st.session_state:
    st.session_state.results = None

if "ai_explanation" not in st.session_state:
    st.session_state.ai_explanation = None

if "chat_answer" not in st.session_state:
    st.session_state.chat_answer = None

# Add images
st.sidebar.image("Pic1.png", width = "stretch")
st.image("Pic2.png", width = "stretch")

# Add main title
st.markdown("<h1 style = 'text-align: center; margin-top: -20px; '>LSTM Forecasting Model</h1>", unsafe_allow_html = True)

# Sidebar inputs
st.sidebar.header("Model Parameters")
crypto_symbol = st.sidebar.text_input("Cryptocurrency Symbol", "BTC-USD")
prediction_ahead = st.sidebar.number_input("Prediction Days Ahead", min_value = 1, max_value = 30, value = 15, step = 1)

# LLM settings
st.sidebar.header("LLM Settings")

provider_options = get_provider_options()
provider = st.sidebar.selectbox("LLM Provider", provider_options)

if "LM Studio" not in provider_options:
    st.sidebar.caption("LM Studio disabled here (install openai to enable it).")

lmstudio_model = st.sidebar.text_input("LM Studio model name", "llama-3-8b-instruct-finance-rag")
watsonx_model_id = st.sidebar.text_input("IBM watsonx model_id", "ibm/granite-4-h-small")
model_name = lmstudio_model if provider == "LM Studio" else watsonx_model_id

# Local fallback for empty LLM responses
def local_explain_fallback(symbol_name, latest_price, predicted_price, change_pct):
    direction = "up" if change_pct >= 0 else "down"
    return (
        f"For {symbol_name}, the model projects price moving {direction} over the selected horizon. "
        f"Latest close is ${latest_price:,.2f} and projected price is ${predicted_price:,.2f}, "
        f"which is a change of {change_pct:.2f}%. "
        "This is a model-based estimate and can change quickly with market volatility. "
        "Educational use only (not financial advice)."
    )

def local_chat_fallback(user_question, symbol_name, latest_price, predicted_price, change_pct):
    return (
        f"I could not get a full model response right now. "
        f"From the current forecast for {symbol_name}: latest close ${latest_price:,.2f}, "
        f"predicted ${predicted_price:,.2f}, change {change_pct:.2f}%. "
        f"Question received: '{user_question}'. "
        "Please retry in a moment for a detailed AI explanation."
    )

if st.sidebar.button("Predict", key = "predict_btn"):
    # Step 1: Pull Crypto data for the past 1 year.
    btc_data = yf.download(
        crypto_symbol,
        period = "1y",
        interval = "1d",
        progress = False,
        threads = False,
    )
    btc_data = btc_data[["Close"]].dropna()

    if btc_data.empty:
        st.error(f"No data returned for symbol '{crypto_symbol}'.")
        st.stop()

    # Prepare data for LSTM
    scaler = MinMaxScaler(feature_range = (0, 1))
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
    x_test, _ = create_dataset(scaled_data[train_size - time_step:], time_step)

    if len(x_train) == 0 or len(x_test) == 0:
        st.error("Not enough sequences were created for train/test datasets.")
        st.stop()

    # Reshape input to be [samples, time_steps, features]
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    # Build LSTM Model
    model = Sequential()
    model.add(LSTM(50, return_sequences = True, input_shape = (time_step, 1)))
    model.add(LSTM(50, return_sequences = False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer = "adam", loss = "mean_squared_error")
    model.fit(x_train, y_train, batch_size = 16, epochs = 5, verbose = 0)

    # Make predictions
    train_predictions = model.predict(x_train, verbose = 0)
    test_predictions = model.predict(x_test, verbose = 0)

    # Inverse transform predictions and actual values
    train_predictions = scaler.inverse_transform(train_predictions).ravel()
    test_predictions = scaler.inverse_transform(test_predictions).ravel()

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
    latest_close_price = float(btc_data["Close"].iloc[-1])
    last_predicted_price = float(future_forecast[-1])
    predicted_change_pct = ((last_predicted_price - latest_close_price) / latest_close_price) * 100.0

    # Store results
    st.session_state.results = {
        "btc_data": btc_data,
        "train_size": train_size,
        "time_step": time_step,
        "train_predictions": train_predictions,
        "test_predictions": test_predictions,
        "future_forecast": future_forecast,
        "latest_close_price": latest_close_price,
        "last_predicted_price": last_predicted_price,
        "predicted_change_pct": predicted_change_pct,
        "symbol_used": crypto_symbol,
        "ahead_used": int(prediction_ahead),
    }

    # Reset AI outputs on new prediction
    st.session_state.ai_explanation = None
    st.session_state.chat_answer = None

if st.session_state.results is not None:
    data = st.session_state.results

    btc_data = data["btc_data"]
    train_size = data["train_size"]
    time_step = data["time_step"]
    train_predictions = data["train_predictions"]
    test_predictions = data["test_predictions"]
    future_forecast = data["future_forecast"]
    latest_close_price = data["latest_close_price"]
    last_predicted_price = data["last_predicted_price"]
    predicted_change_pct = data["predicted_change_pct"]
    symbol_used = data["symbol_used"]
    ahead_used = data["ahead_used"]

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
                    <h3>Price After {ahead_used} Days</h3>
                    <p style = "font-size: 20px;">${last_predicted_price:,.2f}</p>
                    <p style = "font-size: 14px;">{predicted_change_pct:.2f}%</p>
                </div>
            </div>
            """,
            unsafe_allow_html = True,
        )

    # Plot the results
    fig, ax = plt.subplots(figsize = (14, 5))
    ax.plot(btc_data.index, btc_data["Close"], label = "Actual", color = "blue")
    ax.axvline(x = btc_data.index[train_size], color = "gray", linestyle = "--", label = "Train/Test Split")

    # Train/Test and Predictions
    train_range = btc_data.index[time_step:train_size]
    test_range = btc_data.index[train_size:train_size + len(test_predictions)]
    ax.plot(train_range, train_predictions[:len(train_range)], label = "Train Predictions", color = "green")
    ax.plot(test_range, test_predictions[:len(test_range)], label = "Test Predictions", color = "orange")

    # Future Predictions
    future_index = pd.date_range(start = btc_data.index[-1], periods = ahead_used + 1, freq = "D")[1:]
    ax.plot(future_index, future_forecast, label = f"{ahead_used}-Day Forecast", color = "red")

    ax.set_title(f"{symbol_used} LSTM Model Predictions")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # Explain with AI
    st.subheader("Explain with AI")

    if st.button("Explain with AI", key = "explain_btn"):
        prompt = f"""
        Educational use only (not financial advice).

        Asset: {symbol_used}
        Latest close price: ${latest_close_price:,.2f}
        Predicted price after {ahead_used} day(s): ${last_predicted_price:,.2f}
        Predicted change (%): {predicted_change_pct:.2f}
        """

        try:
            with st.spinner("Generating explanation..."):
                st.session_state.ai_explanation = safe_explain(provider, prompt, model_name)
        except Exception as e:
            if provider == "IBM watsonx" and "empty generated_text" in str(e).lower():
                st.session_state.ai_explanation = local_explain_fallback(
                    symbol_used, latest_close_price, last_predicted_price, predicted_change_pct
                )
            else:
                st.session_state.ai_explanation = None
                st.error(f"LLM error: {e}")

    if st.session_state.ai_explanation:
        st.write(st.session_state.ai_explanation)

    # Chat (open)
    st.subheader("Chat (open)")

    user_q = st.text_input(
        "Ask anything (example: 'Explain risk in 5 words', 'What is inflation?', 'Summarize this forecast')",
        key = "chat_input"
    )

    if st.button("Ask", key = "ask_btn"):
        if not user_q.strip():
            st.warning("Please enter a question first.")
        else:
            chat_context = (
                f"Asset: {symbol_used}\n"
                f"Latest close: ${latest_close_price:,.2f}\n"
                f"Predicted after {ahead_used} day(s): ${last_predicted_price:,.2f}\n"
                f"Change (%): {predicted_change_pct:.2f}"
            )

            try:
                with st.spinner("Generating answer..."):
                    st.session_state.chat_answer = open_chat_answer(provider, user_q, model_name, chat_context)
            except Exception as e:
                if provider == "IBM watsonx" and "empty generated_text" in str(e).lower():
                    st.session_state.chat_answer = local_chat_fallback(
                        user_q, symbol_used, latest_close_price, last_predicted_price, predicted_change_pct
                    )
                else:
                    st.session_state.chat_answer = None
                    st.error(f"LLM error: {e}")
    if st.session_state.chat_answer:
        st.write(st.session_state.chat_answer)

# Streamlit run LSTM_ST.py
