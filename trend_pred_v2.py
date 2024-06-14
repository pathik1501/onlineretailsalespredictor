import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import GradientBoostingRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
# Load the dataset
df = pd.read_csv('online_retail_sales.csv', parse_dates=['InvoiceDate'], index_col='InvoiceDate')

# Feature Engineering
df['Month'] = df.index.month
df['Day'] = df.index.day
df['Weekday'] = df.index.weekday

# Exploratory Data Analysis
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='InvoiceDate', y='TotalPrice')
plt.title('Sales Over Time')
plt.show()

# Splitting the data into train and test sets
train_size = int(len(df) * 0.8)
outlier_detector = IsolationForest(contamination=0.2)
train, test = df.iloc[:train_size], df.iloc[train_size:]
outliers = outlier_detector.fit_predict(train['TotalPrice'].values.reshape(-1, 1))
train = train[outliers == 1]
# test = test[outliers == 1]
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
gbr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('gbr', GradientBoostingRegressor())
])
gbr_params = {
    'gbr__n_estimators': [50, 100, 200],
    'gbr__learning_rate': [0.01, 0.1, 0.2]
}
gbr_grid = GridSearchCV(gbr_pipe, gbr_params, cv=3, scoring='neg_root_mean_squared_error')
gbr_grid.fit(np.arange(len(train)).reshape(-1, 1), train['TotalPrice'])
gbr_forecast = gbr_grid.predict(np.arange(len(train), len(train) + len(test)).reshape(-1, 1))
gbr_rmse = calculate_rmse(test['TotalPrice'], gbr_forecast)
print(f'Gradient Boosting RMSE: {gbr_rmse}')

# Neural Network Model
def create_nn_model(input_shape):
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(input_shape,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(np.arange(len(train)).reshape(-1, 1))
X_test_scaled = scaler.transform(np.arange(len(train), len(train) + len(test)).reshape(-1, 1))

nn_model = create_nn_model(1)
nn_model.fit(X_train_scaled, train['TotalPrice'], epochs=50, batch_size=32, verbose=0)
nn_forecast = nn_model.predict(X_test_scaled).flatten()
nn_rmse = calculate_rmse(test['TotalPrice'], nn_forecast)
print(f'Neural Network RMSE: {nn_rmse}')

# Ensemble Model
ensemble_forecast = (ets_forecast + sarima_forecast + gbr_forecast + nn_forecast) / 4
ensemble_rmse = calculate_rmse(test['TotalPrice'], ensemble_forecast)
print(f'Ensemble RMSE: {ensemble_rmse}')

# Plot the forecasts
plt.figure(figsize=(10, 6))
plt.plot(train.index, train['TotalPrice'], label='Train')
plt.plot(test.index, test['TotalPrice'], label='Test')
plt.plot(test.index, ets_forecast, label='ETS Forecast')
plt.plot(test.index, sarima_forecast, label='SARIMA Forecast')
plt.plot(test.index, gbr_forecast, label='Gradient Boosting Forecast')
plt.plot(test.index, nn_forecast, label='Neural Network Forecast')
plt.plot(test.index, ensemble_forecast, label='Ensemble Forecast')
plt.legend()
plt.title('Sales Forecast')
plt.show()
