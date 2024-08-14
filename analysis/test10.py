import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the data into a pandas DataFrame
data = pd.read_csv('reg1.csv')

# Define the independent variables and dependent variable
X = data[["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "valence", "tempo"]]
y = data['success']

# Standardize the independent variables
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the logistic regression model using statsmodels
model = sm.Logit(y_train, sm.add_constant(X_train))

# Fit the model to the training data
result = model.fit(method='lbfgs', maxiter=1000)

# Make predictions on the test data
y_pred = result.predict(sm.add_constant(X_test))
y_pred_binary = np.where(y_pred > 0.5, 1, 0)

# Calculate the accuracy of the model
accuracy = sum(y_pred_binary == y_test) / len(y_test)

# Print the summary of the model and the accuracy
print(result.summary())
print('Accuracy:', accuracy)
