import pandas as pd
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# Load the data into a pandas DataFrame
# data = pd.read_csv('reg1.csv')
data = pd.read_csv('cluster2.csv')

# Standardizing the data
scaled_data = StandardScaler().fit_transform(data.iloc[:, 3:])

# Creating a new dataframe after Standardizing
standardized_df = pd.DataFrame(data=scaled_data, columns=[
    "danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "valence", "tempo"])

# Concatenating the dataframe with the city and week information
final_standardized_df = pd.concat([data.iloc[:, :3], standardized_df], axis=1)
# print(final_standardized_df.iloc[:, 2:])


# Create a PCA object with the desired number of components
pca = PCA(n_components=5)

# Fit the PCA model to the data and transform the data into the new space
pca_data = pca.fit_transform(final_standardized_df.iloc[:, 3:])

# Creating a new dataframe for the principal components
pca_df = pd.DataFrame(data=pca_data, columns=[
                      'PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
# print(pca_df)

# Concatenating the principal components with the city and week information
final_pca_df = pd.concat([data.iloc[:, :3], pca_df], axis=1)
# print(final_pca_df)

# Create a new DataFrame with the transformed data and the success/failure labels
labels = [1 if x in [1, 2, 3] else 0 for x in data['rank']]

raw_data_df = pd.DataFrame(data=final_standardized_df.iloc[:, 2:], columns=[
                           "danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "valence", "tempo"])
raw_data_df['success'] = labels
# print(final_pca_df.iloc[:, 2:])

# Define the logistic regression model using statsmodels

raw_model = sm.Logit(raw_data_df['success'], raw_data_df[[
    "danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "valence", "tempo"]])

# Fit the model to the data
raw_result = raw_model.fit()

# make predictions on test set
y_pred = raw_result.predict(raw_data_df[[
    "danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "valence", "tempo"]])

# convert predicted probabilities to binary predictions
y_pred_binary = [1 if i > 0.5 else 0 for i in y_pred]

# calculate accuracy
accuracy = sum(y_pred_binary == raw_data_df['success']) / len(raw_data_df['success'])

print('Accuracy:', accuracy)

# Print the summary of the model
# print(raw_result.summary())
