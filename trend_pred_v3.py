import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
import itertools

# Load the dataset
df = pd.read_csv('online_retail_sales.csv', parse_dates=['InvoiceDate'], index_col='InvoiceDate')

# Resampling data to weekly frequency
df = df.resample('W').sum()

# Feature Engineering
df['Month'] = df.index.month
df['Day'] = df.index.day
df['Weekday'] = df.index.weekday

# Split data into train and test sets
train_size = int(len(df) * 0.9)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# Function to calculate and print RMSE
def calculate_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

# Holt-Winters Exponential Smoothing (ETS)
ets_model = ExponentialSmoothing(train['TotalPrice'], seasonal='add', seasonal_periods=12).fit()
ets_forecast = ets_model.forecast(len(test))
ets_rmse = calculate_rmse(test['TotalPrice'], ets_forecast)
print(f'ETS RMSE: {ets_rmse}')

# SARIMA Model with Grid Search for hyperparameters
p = d = q = range(0, 2)
pdq = [(x[0], x[1], x[2]) for x in list(itertools.product(p, d, q))]
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
best_aic = float("inf")
best_params = None

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            temp_model = SARIMAX(train['TotalPrice'], order=param, seasonal_order=param_seasonal).fit(disp=False)
            if temp_model.aic < best_aic:
                best_aic = temp_model.aic
                best_params = (param, param_seasonal)
        except:
            continue

sarima_model = SARIMAX(train['TotalPrice'], order=best_params[0], seasonal_order=best_params[1]).fit(disp=False)
sarima_forecast = sarima_model.forecast(len(test))
sarima_rmse = calculate_rmse(test['TotalPrice'], sarima_forecast)
print(f'SARIMA RMSE: {sarima_rmse}')

# Gradient Boosting Regression with Hyperparameter Tuning
gbr = GradientBoostingRegressor()
params = {
    'n_estimators': [100, 500, 1000, 1500],
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3],
    'max_depth': [2, 3, 4, 5, 6, 7]
}
tscv = TimeSeriesSplit(n_splits=3)
gbr_grid = GridSearchCV(gbr, params, cv=tscv, scoring='neg_mean_squared_error')
gbr_grid.fit(train.drop(columns='TotalPrice'), train['TotalPrice'])
gbr_forecast = gbr_grid.predict(test.drop(columns='TotalPrice'))
gbr_rmse = calculate_rmse(test['TotalPrice'], gbr_forecast)
print(f'Gradient Boosting RMSE: {gbr_rmse}')

# LSTM Model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(input_shape, 1)))
    model.add(Dense(5))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train['TotalPrice'].values.reshape(-1, 1))
test_scaled = scaler.transform(test['TotalPrice'].values.reshape(-1, 1))

X_train, y_train = [], []
for i in range(1, len(train_scaled)):
    X_train.append(train_scaled[i-1:i])
    y_train.append(train_scaled[i])
X_train, y_train = np.array(X_train), np.array(y_train)

X_test, y_test = [], []
for i in range(1, len(test_scaled)):
    X_test.append(test_scaled[i-1:i])
    y_test.append(test_scaled[i])
X_test, y_test = np.array(X_test), np.array(y_test)

lstm_model = create_lstm_model(X_train.shape[1])
lstm_model.fit(X_train, y_train, epochs=50, batch_size=4, verbose=0)
lstm_forecast = lstm_model.predict(X_test).flatten()
lstm_forecast = scaler.inverse_transform(lstm_forecast.reshape(-1, 1)).flatten()
lstm_forecast = np.insert(lstm_forecast, 0, np.nan)  # Insert NaN to match the length
lstm_forecast = lstm_forecast[~np.isnan(lstm_forecast)]  # Remove NaNs

# Ensure forecast lengths are equal
min_length = min(len(sarima_forecast), len(gbr_forecast), len(lstm_forecast))
sarima_forecast = sarima_forecast[:min_length]
gbr_forecast = gbr_forecast[:min_length]
lstm_forecast = lstm_forecast[:min_length]
test_adjusted = test['TotalPrice'][:min_length]

# Ensemble Model
ensemble_forecast = (ets_forecast + sarima_forecast+gbr_forecast  + lstm_forecast) / 4
ensemble_rmse = calculate_rmse(test_adjusted, ensemble_forecast)
print(f'Ensemble RMSE: {ensemble_rmse}')

# Plot the forecasts
plt.figure(figsize=(10, 6))
plt.plot(train.index, train['TotalPrice'], label='Train')
plt.plot(test.index, test['TotalPrice'], label='Test')
plt.plot(test.index[:min_length], ets_forecast[:min_length], label='ETS Forecast')
plt.plot(test.index[:min_length], sarima_forecast, label='SARIMA Forecast')
plt.plot(test.index[:min_length], gbr_forecast, label='Gradient Boosting Forecast')
plt.plot(test.index[:min_length], lstm_forecast, label='LSTM Forecast')
plt.plot(test.index[:min_length], ensemble_forecast, label='Ensemble Forecast')
plt.legend()
plt.title('Sales Forecast')
plt.show()
