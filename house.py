import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the training dataset
train_data_path = "C:\\Users\\Akshaya Ganesh\\Downloads\\train.csv"
train_data = pd.read_csv(train_data_path)

# Select relevant features and target variable
# GrLivArea: Above grade (ground) living area square footage
# BedroomAbvGr: Number of bedrooms above ground
# FullBath: Number of full bathrooms
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
X = train_data[features]
y = train_data['SalePrice']  # SalePrice is the target variable (house prices)

# Handle missing values by filling them with the median value
# This step ensures that there are no missing values that could disrupt the model training
X = X.fillna(X.median())

# Split the data into training and testing sets
# test_size=0.2 means 20% of the data will be used for testing, and 80% for training
# random_state=42 ensures reproducibility of results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using various metrics
# MAE: Mean Absolute Error, gives an idea of the magnitude of the error
# MSE: Mean Squared Error, penalizes larger errors more
# R2: R-squared, indicates how well the model explains the variance in the data
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the evaluation results
print("Mean Absolute Error (MAE):",mae)
print("Mean Squared Error (MSE):" ,mse)
print("R-squared (R2):",r2)
import pandas as pd

# Load the test dataset
test_data_path = "C:\\Users\\Akshaya Ganesh\\Downloads\\test.csv"
test_data = pd.read_csv(test_data_path)

# Select relevant features (same as in training)
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
X_test = test_data[features]

# Handle missing values in the test data by filling them with the median value
X_test = X_test.fillna(X_test.median())

# Make predictions using the trained model
y_test_pred = model.predict(X_test)

# Load the sample submission file
submission_file_path = "C:\\Users\\Akshaya Ganesh\\Downloads\\sample_submission.csv"
submission = pd.read_csv(submission_file_path)

# Replace the 'SalePrice' column with the predictions
submission['SalePrice'] = y_test_pred

# Save the submission file
submission_file_output_path = "C:\\Users\\Akshaya Ganesh\\Downloads\\house-prices-submission.csv"
submission.to_csv(submission_file_output_path)




