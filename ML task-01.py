import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
data=pd.read_csv(r'C:\Users\juvva\OneDrive\Desktop\kc_house_data_NaN.csv')
data.head()
for column in data.columns:
    if data[column].dtype == 'object':
        # Fill missing values with the mode for categorical features
        data[column].fillna(data[column].mode()[0], inplace=True)
        if column in data.columns:
            data[column].fillna(data[column].mode()[0], inplace=True)
    else:
        # Fill missing values with the mean for numeric features
        data[column].fillna(data[column].mean(), inplace=True)
        if column in data.columns:
            data[column].fillna(data[column].mean(), inplace=True)

# Select features
features = ['sqft_living', 'sqft_lot', 'sqft_above', 'yr_built', 'sqft_living15']
X = data[features]
y = data['price']

# Split the training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred, alpha=0.5)
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted Sale Price')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.show()
residuals = y_val - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.title('Distribution of Residuals')
plt.show()
plt.xlabel('Predicted Sale Price')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()
plt.figure(figsize=(12, 8))
sns.pairplot(data[features + ['price']])
plt.show()
example = pd.DataFrame({
    'sqft_living': [2000],
    'sqft_lot': [3],
    'sqft_above': [2],
    'yr_built': [1],
    'sqft_living15': [7]
})
example_prediction = model.predict(example)
print(f'Example Prediction: ${example_prediction[0]:,.2f}')

# Prepare the test data and make predictions
X_test = data[features]
test_predictions = model.predict(X_test)

# Save predictions
submission = pd.DataFrame({'id': data['id'], 'price': test_predictions})
submission.to_csv('submission.csv', index=False)
