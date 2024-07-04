import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

# Load the dataset
url = 'https://opendata.muenchen.de/dataset/5e73a82b-7cfb-40cc-9b30-45fe5a3fa24e/resource/40094bd6-f82d-4979-949b-26c8dc00b9a7/download/monatszahlen2405_verkehrsunfaelle_export_31_05_24_r.csv'
data = pd.read_csv(url)

# Rename columns to match the expected names
data.rename(columns={'MONATSZAHL': 'Category', 'AUSPRAEGUNG': 'Type', 'JAHR': 'Year', 'MONAT': 'Month', 'WERT':'Value'}, inplace=True)

# Convert the 'Month' column to string
data['Month'] = data['Month'].astype(str)
data['Month'] = data['Month'].apply(lambda x: x[-2:] if x.isdigit() and len(x) >= 2 else np.nan).astype(float)

# Drop rows with invalid month values
data.dropna(subset=['Month'], inplace=True)
data['Month'] = data['Month'].astype(int)

# Add a day column with a default value of 1
data['Day'] = 1

# Create a Date column
data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']])

def train_random_forest_model(filtered_data):
    # Prepare the data (assuming 'Date' is the index and target variable is 'Value')
    X = filtered_data[['Year', 'Month']]
    y = filtered_data['Value']

    # Check for NaN values in y
    if y.isnull().any():
        print("Warning: NaN values found in target variable 'Value'. Dropping rows with NaN values.")
        filtered_data = filtered_data.dropna(subset=['Value'])
        X = filtered_data[['Year', 'Month']]
        y = filtered_data['Value']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Save the Random Forest model
    with open('random_forest_model.pkl', 'wb') as file:
        pickle.dump(rf_model, file)

    return rf_model, X_test, y_test

def predict_and_evaluate(category, accident_type, year, month):
    # Filter for the specific category and type
    filtered_data = data[(data['Category'] == category) &
                         (data['Type'] == accident_type)]

    if filtered_data.empty:
        print(f"No data found for category '{category}' and type '{accident_type}'.")
        return

    # Handle NaN values in 'Value' before training
    rf_model, X_test, y_test = train_random_forest_model(filtered_data)

    # Read the Random Forest Model
    with open('random_forest_model.pkl', 'rb') as file:
        rf_model = pickle.load(file)

    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'Year': [year],
        'Month': [month]
    })

    # Make predictions
    predicted_values = rf_model.predict(input_data)

    # Print the predicted values
    print(f'Predicted number of {accident_type} accidents for {month}/{year}: {predicted_values[0]}')

    # Retrieve the actual value
    actual_values = data[(data['Year'] == year) & (data['Month'] == month) & (data['Category'] == category) & (data['Type'] == accident_type)]['Value'].values

    if len(actual_values) == 0 or np.isnan(actual_values[0]):
        print(f"No actual data found for {month}/{year} in category '{category}' and type '{accident_type}' or actual value is NaN.")
    else:
        actual_value = actual_values[0]
        print(f'Actual number of {accident_type} accidents for {month}/{year}: {actual_value}')

        # Compute error metrics if actual value is valid
        if not np.isnan(actual_value):
            cal_mae = mean_absolute_error([actual_value], predicted_values)
            cal_mse = mean_squared_error([actual_value], predicted_values)
            cal_rmse = mean_squared_error([actual_value], predicted_values, squared=False)

            # Print error metrics
            print(f'MAE: {cal_mae}, MSE: {cal_mse}, RMSE: {cal_rmse}')

# Example usage:
category = "Alkoholunfälle"
accident_type = "insgesamt"
year = 2021
month = 6

predict_and_evaluate(category, accident_type, year, month)