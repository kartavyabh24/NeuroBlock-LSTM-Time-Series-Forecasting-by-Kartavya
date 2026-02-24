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

# MUST be first Streamlit command
st.set_page_config(layout="wide")

# ignore warnings
warnings.filterwarnings("ignore")

# -------------------------
# Session state
# -------------------------
if "results" not in st.session_state:
    st.session_state.results = None

if "ai_explanation" not in st.session_state:
    st.session_state.ai_explanation = None

if "chat_answer" not in st.session_state:
    st.session_state.chat_answer = None

# -------------------------
# Streamlit configuration
# -------------------------
st.markdown("<style>.main {padding-top: 0px;}</style>", unsafe_allow_html=True)

# Add images
st.sidebar.image("Pic1.png", width="stretch")
st.image("Pic2.png", width="stretch")

# Add main title
st.markdown(
    "<h1 style='text-align: center; margin-top: -20px;'>LSTM Forecasting Model</h1>",
    unsafe_allow_html=True
)

# -------------------------
# Sidebar inputs
# -------------------------
st.sidebar.header("Model Parameters")
crypto_symbol = st.sidebar.text_input("Cryptocurrency Symbol", "BTC-USD")
prediction_ahead = st.sidebar.number_input(
    "Prediction Days Ahead", min_value=1, max_value=30, value=15, step=1
)

# -------------------------
# LLM Settings
# -------------------------
st.sidebar.header("LLM Settings")

provider = st.sidebar.selectbox("LLM Provider", ["LM Studio", "IBM watsonx"])

lmstudio_model = st.sidebar.text_input(
    "LM Studio model name", "llama-3-8b-instruct-finance-rag"
)

watsonx_model_id = st.sidebar.text_input(
    "IBM watsonx model_id", "ibm/granite-4-h-small"
)

model_name = lmstudio_model if provider == "LM Studio" else watsonx_model_id


# ============================================================
# Predict Button (Model runs ONLY here)
# ============================================================
if st.sidebar.button("Predict", key="predict_btn"):

    # Step 1: Pull Crypto data for the past 1 year.
    btc_data = yf.download(
        crypto_symbol,
        period="1y",
        interval="1d",
        progress=False,
        threads=False
    )

    btc_data = btc_data[["Close"]].dropna()

    if btc_data.empty:
        st.error(f"No data returned for symbol '{crypto_symbol}'.")
        st.stop()

    # Prepare data for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(btc_data).astype(np.float32)

    train_size = int(len(scaled_data) * 0.8)
    time_step = 60

    def create_dataset(data, time_step=1):
        x, y = [], []
        for i in range(len(data) - time_step):
            x.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(x), np.array(y)

    if len(scaled_data) <= time_step or train_size <= time_step:
        st.error("Not enough data points to train the model with a 60-day window.")
        st.stop()

    x_train, y_train = create_dataset(scaled_data[:train_size], time_step)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

    # Build LSTM Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(x_train, y_train, batch_size=16, epochs=5, verbose=0)

    # Forecasting for future days
    last_60_days = scaled_data[-time_step:]
    future_input = last_60_days.reshape(1, time_step, 1)

    future_forecast = []

    for _ in range(int(prediction_ahead)):
        next_pred = model.predict(future_input, verbose=0)[0, 0]
        future_forecast.append(next_pred)
        next_input = np.append(future_input[0, 1:], [[next_pred]], axis=0)
        future_input = next_input.reshape(1, time_step, 1)

    future_forecast = scaler.inverse_transform(
        np.array(future_forecast).reshape(-1, 1)
    ).ravel()

    latest_close_price = float(btc_data["Close"].iloc[-1])
    last_predicted_price = float(future_forecast[-1])
    predicted_change_pct = (
        (last_predicted_price - latest_close_price)
        / latest_close_price
    ) * 100.0

    # Store results
    st.session_state.results = {
        "btc_data": btc_data,
        "future_forecast": future_forecast,
        "latest_close_price": latest_close_price,
        "last_predicted_price": last_predicted_price,
        "predicted_change_pct": predicted_change_pct
    }

    # Reset AI outputs on new prediction
    st.session_state.ai_explanation = None
    st.session_state.chat_answer = None


# ============================================================
# Display Section (NO retraining)
# ============================================================
if st.session_state.results is not None:

    data = st.session_state.results

    btc_data = data["btc_data"]
    future_forecast = data["future_forecast"]
    latest_close_price = data["latest_close_price"]
    last_predicted_price = data["last_predicted_price"]
    predicted_change_pct = data["predicted_change_pct"]

    # Centered layout for metrics
    col1, col2 = st.columns(2)

    col1.metric("Latest Close Price", f"${latest_close_price:,.2f}")
    col2.metric(
        f"Price After {prediction_ahead} Days",
        f"${last_predicted_price:,.2f}",
        f"{predicted_change_pct:.2f}%"
    )

    # Plot the results
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(btc_data.index, btc_data["Close"], label="Actual", color="blue")

    future_index = pd.date_range(
        start=btc_data.index[-1],
        periods=int(prediction_ahead) + 1,
        freq="D"
    )[1:]

    ax.plot(future_index, future_forecast, label="Forecast", color="red")

    ax.set_title(f"{crypto_symbol} LSTM Model Engine")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    # -------------------------------------------------------
    # Explain with AI
    # -------------------------------------------------------
    st.subheader("Explain with AI")

    if st.button("Explain with AI", key="explain_btn"):
        prompt = f"""
        Educational use only (not financial advice).

        Asset: {crypto_symbol}
        Latest close price: ${latest_close_price:,.2f}
        Predicted price after {int(prediction_ahead)} day(s): ${last_predicted_price:,.2f}
        Predicted change (%): {predicted_change_pct:.2f}
        """

        try:
            st.session_state.ai_explanation = explain(provider, prompt, model_name)
        except Exception as e:
            st.error(f"LLM error: {e}")

    if st.session_state.ai_explanation:
        st.write(st.session_state.ai_explanation)

    # -------------------------------------------------------
    # Chat (grounded)
    # -------------------------------------------------------
    st.subheader("Chat (grounded)")

    user_q = st.text_input(
        "Ask a question about this forecast (example: 'Explain risk', 'What does +% mean?')",
        key="chat_input"
    )

    if user_q:
        chat_prompt = f"""
        Educational use only.

        Asset: {crypto_symbol}
        Latest close: ${latest_close_price:,.2f}
        Predicted after {int(prediction_ahead)} day(s): ${last_predicted_price:,.2f}
        Change (%): {predicted_change_pct:.2f}

        User question: {user_q}
        """

        try:
            st.session_state.chat_answer = explain(provider, chat_prompt, model_name)
        except Exception as e:
            st.error(f"LLM error: {e}")

    if st.session_state.chat_answer:
        st.write(st.session_state.chat_answer)