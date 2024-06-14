import pandas as pd

# Load the dataset using openpyxl engine
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'
df = pd.read_excel(url, engine='openpyxl')

# Data cleaning
df = df.dropna(subset=['CustomerID'])  # Drop rows with missing CustomerID

# Create a new column 'TotalPrice' which is Quantity * UnitPrice
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Extract 'InvoiceDate' and 'TotalPrice' for time series analysis
df = df[['InvoiceDate', 'TotalPrice']]

# Resample the data to daily sales
df = df.set_index('InvoiceDate')
daily_sales = df.resample('D').sum()

# Fill any missing dates with 0 sales
daily_sales = daily_sales.fillna(0)

# Save the prepared data
daily_sales.to_csv('online_retail_sales.csv')
