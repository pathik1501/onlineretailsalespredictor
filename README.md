# onlineretailsalespredictor
*Overview*

This project aims to forecast weekly online retail sales using various time series forecasting and machine learning models. Accurate sales forecasting helps optimize inventory management and resource allocation, improving overall business efficiency.

*Features*

Forecasting Models: Implemented Holt-Winters Exponential Smoothing (ETS), SARIMA (Seasonal ARIMA), Gradient Boosting Regression, and LSTM (Long Short-Term Memory) neural network for sales forecasting.
Ensemble Forecasting: Combined forecasts from multiple models to improve prediction accuracy using weighted averaging.
Feature Engineering: Extracted features like month, day, and weekday to enhance model performance.


*Technologies Used*

Python Libraries: pandas, numpy, matplotlib, statsmodels, scikit-learn, TensorFlow/Keras
Machine Learning: Applied GridSearchCV for hyperparameter tuning and StandardScaler for data preprocessing.
Deep Learning: Utilized LSTM for capturing long-term dependencies in time series data.


*Dataset*

The 'online_retail_sales.csv' dataset was used for this project. It includes transaction data with a timestamp. The data was resampled to weekly frequency and feature engineering was performed to include month, day, and weekday columns.

*Models and Techniques*

Holt-Winters Exponential Smoothing (ETS):
Captures trend and seasonality in the data.
RMSE: 86,940.34
SARIMA (Seasonal ARIMA):
Handles complex seasonal patterns and trends.
RMSE: 116,830.29
Gradient Boosting Regression:
Models nonlinear relationships in the data.
RMSE: 104,957.25
LSTM (Long Short-Term Memory):
Captures long-term dependencies in time series data.
RMSE: Calculated within ensemble model.
Ensemble Model:
Combines top-performing models to improve accuracy.
RMSE: 15,469.47

*Results*

The Ensemble model achieved the lowest RMSE of 15,469.47, outperforming individual models.
The lower RMSE indicates better predictive performance, making the ensemble approach highly effective for this dataset.

*Future Improvements*

Explore additional models and techniques, such as Prophet or XGBoost.
Incorporate external factors like holidays and promotions to improve forecast accuracy.
Optimize model parameters further for each individual model.
