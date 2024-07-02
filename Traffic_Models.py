import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
# import matplotlib.pyplot as plt
import plotly.express as px

# Load the dataset
url = 'https://opendata.muenchen.de/dataset/5e73a82b-7cfb-40cc-9b30-45fe5a3fa24e/resource/40094bd6-f82d-4979-949b-26c8dc00b9a7/download/monatszahlen2405_verkehrsunfaelle_export_31_05_24_r.csv'
data = pd.read_csv(url)

# Rename columns to match the expected names
data.rename(columns={'MONATSZAHL': 'Category', 'AUSPRAEGUNG': 'Type', 'JAHR': 'Year', 'MONAT': 'Month', 'WERT':'Value'}, inplace=True)
#data

# Convert the 'month' column to string
data['Month'] = data['Month'].astype(str)

# Check the unique value of column month
print(data['Month'].unique())

data['Month'] = data['Month'].apply(lambda x: x[-2:] if x.isdigit() and len(x) >= 2 else np.nan).astype(float)

# Drop rows with invalid month values
data.dropna(subset=['Month'], inplace=True)
data['Month'] = data['Month'].astype(int)

# Recheck again the value of moth
print(data['Month'].unique())

# Add a day column with a default value of 1
data['Day'] = 1

# Create a Date column
data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']])

# Filter the data to include only records up to 2020
filtered_data = data[data['Date'].dt.year <= 2020]

# Display the first few rows of the filtered data
print(filtered_data.head())

# Filter for the specific category and type
filtered_data = filtered_data[(filtered_data['Category'] == 'Alkoholunfälle') &
                              (filtered_data['Type'] == 'insgesamt')]

# Prepare the data (assuming 'Date' is the index and target variable is 'Value')
X = filtered_data[['Year', 'Month']]
y = filtered_data['Value']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the LinearRegression model
model = LinearRegression()
model.fit(X_train, y_train)

# Train the Decision Tree model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)


# Train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_dt = dt_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# Compute error metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Compute error metrics for Decision Tree
mae_dt = mean_absolute_error(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = mean_squared_error(y_test, y_pred_dt, squared=False)

# Compute error metrics for Random Forest
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)

print(f'LinearRegression - MAE: {mae}, MSE: {mse}, RMSE: {rmse}')
print(f'Decision Tree - MAE: {mae_dt}, MSE: {mse_dt}, RMSE: {rmse_dt}')
print(f'Random Forest - MAE: {mae_rf}, MSE: {mse_rf}, RMSE: {rmse_rf}')

# Save the Linear Regression model
with open('linear_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save the Decision Tree model
with open('decision_tree_model.pkl', 'wb') as file:
    pickle.dump(dt_model, file)

# Save the Random Forest model
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)