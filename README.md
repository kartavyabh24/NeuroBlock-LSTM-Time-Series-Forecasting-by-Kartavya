# NeuroBlock – LSTM Cryptocurrency Forecasting Platform

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![License](https://img.shields.io/badge/License-MIT-green)

NeuroBlock is a deep learning powered cryptocurrency forecasting system that predicts future price trends using **LSTM (Long Short-Term Memory) neural networks**. The project combines **financial time series analysis, machine learning, and interactive visualization** to provide a user-friendly prediction dashboard.

The platform retrieves historical cryptocurrency data, trains an LSTM model to capture temporal patterns, and generates forecasts that can be explored through an **interactive Streamlit application**. The system also integrates an **AI explanation module** that can generate insights about predicted trends.

---

# Project Overview

Financial time series forecasting is widely used to analyze market behavior and potential future trends. This project demonstrates how deep learning models such as LSTM can capture sequential patterns in cryptocurrency price data.

The system processes historical market data, trains an LSTM neural network, and predicts future prices based on learned patterns. Users can visualize predictions, explore market trends, and generate AI explanations through an interactive web interface.

---

# Key Features

* Cryptocurrency price forecasting using LSTM neural networks
* Automated historical data collection using Yahoo Finance API
* Data preprocessing and time series window generation
* Interactive prediction dashboard built with Streamlit
* Visualization of historical vs predicted prices
* AI powered explanation module for prediction insights
* Modular architecture allowing extension to other financial assets

---

# Tech Stack

### Programming Language

Python

### Machine Learning / Deep Learning

TensorFlow
Keras
Scikit-learn

### Data Processing

Pandas
NumPy

### Data Source

Yahoo Finance API (yfinance)

### Visualization

Matplotlib
Altair

### Web Application

Streamlit

### AI / LLM Integration

OpenAI API / LLM Provider

---

# Skills Demonstrated

* Time Series Forecasting
* Deep Learning with LSTM Networks
* Financial Data Analysis
* Machine Learning Model Evaluation
* Interactive Data Visualization
* AI-powered Insight Generation
* End-to-End Machine Learning Pipeline Development

---

# Project Architecture

```
Data Source (Yahoo Finance API)
            │
            ▼
Data Preprocessing
(Pandas, NumPy)
            │
            ▼
Sequence Window Creation
(Time Series Segmentation)
            │
            ▼
LSTM Deep Learning Model
(TensorFlow / Keras)
            │
            ▼
Price Prediction Output
            │
            ▼
Streamlit Dashboard
(Visualization + Interaction)
            │
            ▼
AI Explanation Module
(LLM Integration)
```

---

# Project Workflow

```
Raw Cryptocurrency Data
        │
        ▼
Data Collection (Yahoo Finance API)
        │
        ▼
Data Cleaning and Preprocessing
        │
        ▼
Time Series Window Creation
        │
        ▼
LSTM Model Training
        │
        ▼
Price Prediction Generation
        │
        ▼
Visualization with Streamlit Dashboard
        │
        ▼
AI Explanation Module
```

---

# Project Structure

```
NeuroBlock-LSTM-Time-Series-Forecasting

├── LSTM_Project.ipynb        # Model training and experimentation
├── LSTM_ST.py                # Streamlit interactive dashboard
├── llm_provider.py           # AI explanation module
├── requirements.txt          # Python dependencies
├── Pic1.png                  # Visualization of predictions
├── Pic2.png                  # Model forecast output
└── .gitignore
```

---

# Model Performance

The LSTM model was evaluated using common regression metrics to measure prediction accuracy.

| Metric | Description                                             |
| ------ | ------------------------------------------------------- |
| MAE    | Mean Absolute Error between predicted and actual prices |
| RMSE   | Root Mean Squared Error measuring prediction deviation  |
| MAPE   | Mean Absolute Percentage Error for relative accuracy    |

Example model output

```
MAE: 1250.43
RMSE: 1874.65
MAPE: 2.91%
```

These metrics indicate how closely the predicted values align with actual market prices.

---

# Installation

Clone the repository

```
git clone https://github.com/kartavyabh24/NeuroBlock-LSTM-Time-Series-Forecasting.git
cd NeuroBlock-LSTM-Time-Series-Forecasting
```

Install dependencies

```
pip install -r requirements.txt
```

---

# Running the Application

Launch the Streamlit dashboard

```
streamlit run LSTM_ST.py
```

The application will open automatically in your browser where you can explore predictions interactively.

---

# Demo

### Historical vs Predicted Price Visualization

![Prediction Example](Pic1.png)

### Forecast Results

![Forecast Result](Pic2.png)

---

# Example Workflow

1. Load historical cryptocurrency price data
2. Clean and normalize the dataset
3. Generate time sequence windows for training
4. Train the LSTM neural network
5. Predict future cryptocurrency prices
6. Visualize predictions using the Streamlit dashboard
7. Generate AI explanations for predicted trends

---

# Use Cases

* Cryptocurrency trend forecasting
* Educational demonstration of deep learning models
* Financial data analysis experiments
* Machine learning research projects

---

# Future Improvements

* Multi asset forecasting (Ethereum, Gold, Stocks)
* Transformer based time series models
* Real time prediction pipeline
* Improved feature engineering
* Advanced explainable AI modules

---

# Educational Disclaimer

This project is created for **educational and research purposes only**. Predictions generated by this system should not be considered financial advice.

---

# Author

Kartavya Bhardwaj
M.S. Data Science
Saint Peter’s University

GitHub
[https://github.com/kartavyabh24](https://github.com/kartavyabh24)
