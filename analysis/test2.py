# Importing necessary libraries
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Loading data
data = pd.read_csv('final-data.csv')
# print(data["danceability", "energy", "loudness", "speechiness",
#           "acousticness", "instrumentalness", "valence", "tempo"])

# Standardizing the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.iloc[:, 2:])

# Performing PCA
# You can choose the number of components you want to extract
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)

# Creating a new dataframe for the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Concatenating the principal components with the city and week information
final_df = pd.concat([data.iloc[:, :2], pca_df], axis=1)

# Printing the final dataframe
print(final_df.head())

# Importing necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Splitting the data into training and testing sets
print(scaled_data)
X_train, X_test, y_train, y_test = train_test_split(scaled_data, test_size=0.2, random_state=42)

# Creating a logistic regression model
logreg = LogisticRegression()

# Fitting the model on the training data
logreg.fit(X_train, y_train)

# Evaluating the model on the testing data
score_pca = logreg.score(X_test, y_test)

# Creating a new dataframe without PCA
no_pca_data = data.iloc[:, 2:]

# Standardizing the data
scaler = StandardScaler()
no_pca_scaled_data = scaler.fit_transform(no_pca_data)
print(no_pca_scaled_data)
# Splitting the data into training and testing sets
X_train_no_pca, X_test_no_pca, y_train_no_pca, y_test_no_pca = train_test_split(no_pca_scaled_data, data['label'], test_size=0.2, random_state=42)

# Creating a logistic regression model
logreg_no_pca = LogisticRegression()

# Fitting the model on the training data
logreg_no_pca.fit(X_train_no_pca, y_train_no_pca)

# Evaluating the model on the testing data
score_no_pca = logreg_no_pca.score(X_test_no_pca, y_test_no_pca)

# Printing the scores
print("Logistic Regression score with PCA: ", score_pca)
print("Logistic Regression score without PCA: ", score_no_pca)
