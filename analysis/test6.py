import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load the data into a pandas DataFrame
data = pd.read_csv('reg1.csv')

# Create a PCA object with the desired number of components
pca = PCA(n_components=5)
# print('pca', pca)

# Fit the PCA model to the data and transform the data into the new space
pca_data = pca.fit_transform(data)
# print('pca_data', pca_data)


# Create a new DataFrame with the transformed data and the success/failure labels
labels = [1 if x in [1, 2, 3] else 0 for x in data['rank']]
pca_data_df = pd.DataFrame(data=pca_data, columns=[
                           'PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
pca_data_df['success'] = labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    pca_data_df[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']], pca_data_df['success'], test_size=0.3, random_state=42)

# Train a logistic regression model on the training data
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Print the accuracy of the model
print('Accuracy:', model.score(X_test, y_test))

# Calculate the p-values of each independent variable using the Wald test
p_values = model.coef_ / model.coef_.std() / X_train.std(axis=0)
p_values = 2 * (1 - np.linalg.norm.cdf(abs(p_values)))  # two-sided

# Print the p-values
for i, col in enumerate(X_train.columns):
    print(f'{col}: {p_values[0][i]}')
