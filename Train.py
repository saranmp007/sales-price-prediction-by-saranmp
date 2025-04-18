import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("Sale_Report.csv")  # Replace "your_dataset.csv" with the path to your dataset

# Assuming 'Date' is a feature
# Convert 'Date' to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Sort the dataset by date
data = data.sort_values(by='Date')

# Split the dataset into features (X) and target variable (y)
X = data[['Date', 'Amount']]  # Assuming 'Amount' is the price
y = data['Amount']

# Extract numerical features from the datetime feature
X['Year'] = X['Date'].dt.year
X['Month'] = X['Date'].dt.month
X['Day'] = X['Date'].dt.day
X = X.drop(columns=['Date'])

# Handle missing values in the target variable
y = y.fillna(y.mean())

# Check for missing values in features
missing_values = X.isnull().sum().sum()
if missing_values > 0:
    print("Dataset contains missing values. Imputing...")
    
    # Handle missing values in features
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
predictions = model.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

# Plot the actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual Price')
plt.plot(y_test.index, predictions, label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Actual vs Predicted Price')
plt.legend()
plt.show()
import joblib

# Save the trained model to a file
joblib.dump(model, 'linear_regression_model.pkl')

# Load the trained model from the file
loaded_model = joblib.load('linear_regression_model.pkl')

# Make predictions using the loaded model
loaded_predictions = loaded_model.predict(X_test)

# Calculate MSE for the loaded model
loaded_mse = mean_squared_error(y_test, loaded_predictions)
print("Mean Squared Error (Loaded Model):", loaded_mse)
