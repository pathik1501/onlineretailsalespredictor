import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import GradientBoostingRegressor
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('online_retail_sales.csv', parse_dates=['InvoiceDate'], index_col='InvoiceDate')

# Exploratory Data Analysis
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='InvoiceDate', y='TotalPrice')
plt.title('Sales Over Time')
plt.show()

# Splitting the data into train and test sets
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# Function to calculate and print RMSE
def calculate_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

# Holt-Winters Exponential Smoothing (ETS)
ets_model = ExponentialSmoothing(train['TotalPrice'], seasonal='add', seasonal_periods=12).fit()
ets_forecast = ets_model.forecast(len(test))
ets_rmse = calculate_rmse(test['TotalPrice'], ets_forecast)
print(f'ETS RMSE: {ets_rmse}')

# SARIMA Model
sarima_model = SARIMAX(train['TotalPrice'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
sarima_forecast = sarima_model.forecast(len(test))
sarima_rmse = calculate_rmse(test['TotalPrice'], sarima_forecast)
print(f'SARIMA RMSE: {sarima_rmse}')

# Gradient Boosting Regression
gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
gbr_model.fit(np.arange(len(train)).reshape(-1, 1), train['TotalPrice'])
gbr_forecast = gbr_model.predict(np.arange(len(train), len(train) + len(test)).reshape(-1, 1))
gbr_rmse = calculate_rmse(test['TotalPrice'], gbr_forecast)
print(f'Gradient Boosting RMSE: {gbr_rmse}')

# Prophet Model
prophet_df = train.reset_index().rename(columns={'InvoiceDate': 'ds', 'TotalPrice': 'y'})
prophet_model = Prophet()
prophet_model.fit(prophet_df)
future = prophet_model.make_future_dataframe(periods=len(test))
prophet_forecast = prophet_model.predict(future)
prophet_rmse = calculate_rmse(test['TotalPrice'], prophet_forecast['yhat'].tail(len(test)))
print(f'Prophet RMSE: {prophet_rmse}')

# Neural Network Model
def create_nn_model(input_shape):
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(input_shape,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

nn_model = create_nn_model(1)
nn_model.fit(np.arange(len(train)).reshape(-1, 1), train['TotalPrice'], epochs=50, batch_size=32, verbose=0)
nn_forecast = nn_model.predict(np.arange(len(train), len(train) + len(test)).reshape(-1, 1)).flatten()
nn_rmse = calculate_rmse(test['TotalPrice'], nn_forecast)
print(f'Neural Network RMSE: {nn_rmse}')

# Plot the forecasts
plt.figure(figsize=(10, 6))
plt.plot(train.index, train['TotalPrice'], label='Train')
plt.plot(test.index, test['TotalPrice'], label='Test')
plt.plot(test.index, ets_forecast, label='ETS Forecast')
plt.plot(test.index, sarima_forecast, label='SARIMA Forecast')
plt.plot(test.index, gbr_forecast, label='Gradient Boosting Forecast')
plt.plot(test.index, prophet_forecast['yhat'].tail(len(test)), label='Prophet Forecast')
plt.plot(test.index, nn_forecast, label='Neural Network Forecast')
plt.legend()
plt.title('Sales Forecast')
plt.show()

# Combining models using Ensemble method (average)
ensemble_forecast = (ets_forecast + sarima_forecast + gbr_forecast + (prophet_forecast['yhat'].tail(len(test)).values) + nn_forecast) / 5
ensemble_rmse = calculate_rmse(test['TotalPrice'], ensemble_forecast)
print(f'Ensemble RMSE: {ensemble_rmse}')

# Plot the ensemble forecast
plt.figure(figsize=(10, 6))
plt.plot(train.index, train['TotalPrice'], label='Train')
plt.plot(test.index, test['TotalPrice'], label='Test')
plt.plot(test.index, ensemble_forecast, label='Ensemble Forecast')
plt.legend()
plt.title('Ensemble Sales Forecast')
plt.show()
