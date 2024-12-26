# This code performs time series analysis and stock price prediction for Apple Inc. using multiple models. It downloads historical stock data via yfinance, implements feature engineering with rolling statistics, and compares Linear Regression, Exponential Smoothing, and LSTM models. The analysis includes data visualization, model training, and performance evaluation using metrics like MAE and RMSE. The LSTM model incorporates sequence-based learning with dropout layers for robustness, while also handling data scaling and sequence preparation for deep learning.

# Import libraries
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import seaborn as sns

# Data Acquisition
stock_data = yf.download('AAPL', start='2022-01-01', end='2024-01-01')

# Exploratory Data Analysis
print("First 10 rows of data:")
print(stock_data.head(10))
print("\nDescriptive Statistics:")
print(stock_data.describe())

# Visualize closing price
plt.figure(figsize=(12, 6))
plt.plot(stock_data.index, stock_data['Close'], label="Closing Price")
plt.title("Stock Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()

# Calculate and plot returns
stock_data['Returns'] = stock_data['Close'].pct_change()
plt.figure(figsize=(10, 5))
stock_data['Returns'].dropna().hist(bins=50, alpha=0.75)
plt.title("Histogram of Stock Returns")
plt.xlabel("Returns")
plt.ylabel("Frequency")
plt.show()

# Feature Engineering
stock_data['Rolling_10'] = stock_data['Close'].rolling(window=10).mean()
stock_data['Rolling_30'] = stock_data['Close'].rolling(window=30).mean()
stock_data['Rolling_Std'] = stock_data['Close'].rolling(window=30).std()
stock_data['7_day_avg'] = stock_data['Close'].rolling(window=7, min_periods=1).mean()
stock_data = stock_data.dropna()

# Visualize rolling statistics
plt.figure(figsize=(12, 6))
plt.plot(stock_data.index, stock_data['Close'], label="Closing Price")
plt.plot(stock_data.index, stock_data['Rolling_10'], label="10-day Rolling Mean")
plt.plot(stock_data.index, stock_data['Rolling_30'], label="30-day Rolling Mean")
plt.fill_between(stock_data.index, 
                stock_data['Rolling_30'] - stock_data['Rolling_Std'], 
                stock_data['Rolling_30'] + stock_data['Rolling_Std'], 
                color='gray', alpha=0.2, label="30-day Std Dev")
plt.title("Rolling Statistics of Stock Prices")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()

# Linear Regression Model
train_data, test_data = train_test_split(stock_data, test_size=0.2, shuffle=False)
X_train = train_data[['Rolling_10', 'Rolling_30']]
y_train = train_data['Close']
X_test = test_data[['Rolling_10', 'Rolling_30']]
y_test = test_data['Close']

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# Exponential Smoothing Model
es_model = ExponentialSmoothing(train_data['Close'], trend="add", seasonal=None)
es_model_fit = es_model.fit()
es_predictions = es_model_fit.forecast(steps=len(test_data))

# Calculate Performance Metrics
lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_rmse = np.sqrt(lr_mse)

es_mae = mean_absolute_error(y_test, es_predictions)
es_mse = mean_squared_error(y_test, es_predictions)
es_rmse = np.sqrt(es_mse)

# LSTM Model Preparation
scaler = MinMaxScaler(feature_range=(0, 1))
stock_data_scaled = scaler.fit_transform(stock_data[['Close']])

sequence_length = 60
X, y = [], []
for i in range(sequence_length, len(stock_data_scaled)):
   X.append(stock_data_scaled[i-sequence_length:i, 0])
   y.append(stock_data_scaled[i, 0])
X, y = np.array(X), np.array(y)

# Split LSTM data
split = int(len(X) * 0.8)
X_train_lstm, X_test_lstm = X[:split], X[split:]
y_train_lstm, y_test_lstm = y[:split], y[split:]

X_train_lstm = np.reshape(X_train_lstm, (X_train_lstm.shape[0], X_train_lstm.shape[1], 1))
X_test_lstm = np.reshape(X_test_lstm, (X_test_lstm.shape[0], X_test_lstm.shape[1], 1))

# Build and train LSTM model
lstm_model = Sequential([
   LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
   Dropout(0.2),
   LSTM(50, return_sequences=False),
   Dropout(0.2),
   Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32, verbose=1)

# LSTM Predictions and Metrics
lstm_predictions_scaled = lstm_model.predict(X_test_lstm)
lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
y_test_rescaled = scaler.inverse_transform(y_test_lstm.reshape(-1, 1))

lstm_mae = mean_absolute_error(y_test_rescaled, lstm_predictions)
lstm_mse = mean_squared_error(y_test_rescaled, lstm_predictions)
lstm_rmse = np.sqrt(lstm_mse)

# Results Comparison
metrics = pd.DataFrame({
   'Model': ['Linear Regression', 'Exponential Smoothing', 'LSTM'],
   'MAE': [lr_mae, es_mae, lstm_mae],
   'MSE': [lr_mse, es_mse, lstm_mse],
   'RMSE': [lr_rmse, es_rmse, lstm_rmse]
})

print("\nModel Performance Metrics:")
print(metrics)

# Final Visualization
plt.figure(figsize=(14, 7))
plt.plot(test_data.index, y_test, label="Actual Prices", linewidth=2)
plt.plot(test_data.index, lr_predictions, label="Linear Regression", linestyle='--')
plt.plot(test_data.index, es_predictions, label="Exponential Smoothing", linestyle='-.')
plt.plot(stock_data.index[-len(y_test_lstm):], lstm_predictions, label="LSTM", linestyle=':')
plt.title("Model Predictions vs Actual Prices")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()
