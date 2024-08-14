import pandas as pd
from sklearn.decomposition import PCA
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Load the data into a pandas DataFrame
data = pd.read_csv('cluster0.csv')

# Standardizing the data
scaled_data = StandardScaler().fit_transform(data.iloc[:, 3:])

# Creating a new dataframe after Standardizing
standardized_df = pd.DataFrame(data=scaled_data, columns=[
    "danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "valence", "tempo"])

# Concatenating the dataframe with the city, week and rank information
final_standardized_df = pd.concat([data.iloc[:, :3], standardized_df], axis=1)


# Create a PCA object with the desired number of components
pca = PCA(n_components=3)

# Fit the PCA model to the data and transform the data into the new space
pca_data = pca.fit_transform(final_standardized_df.iloc[:, 3:])

# Creating a new dataframe for the principal components
pca_df = pd.DataFrame(data=pca_data, columns=[
                      'PC1', 'PC2', 'PC3'])

# Concatenating the principal components with the city, week and rank information
final_pca_df = pd.concat([data.iloc[:, :3], pca_df], axis=1)

# Create a new DataFrame with the transformed PCA data and the success/failure labels
labels = [1 if x in [1, 2, 3] else 0 for x in data['rank']]
pca_data_df = pd.DataFrame(data=final_pca_df.iloc[:, 2:], columns=[
                           'PC1', 'PC2', 'PC3'])
pca_data_df['success'] = labels

# Create a new DataFrame with the transformed Non-PCA data and the success/failure labels
raw_data_df = pd.DataFrame(data=final_standardized_df.iloc[:, 2:], columns=[
                           "danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "valence", "tempo"])
raw_data_df['success'] = labels

# Split the data into train and test sets
pca_train, pca_test, pca_result_train, pca_result_test = train_test_split(
    pca_data_df, pca_data_df['success'], test_size=0.3, random_state=42)
raw_train, raw_test, raw_result_train, raw_result_test = train_test_split(
    raw_data_df, raw_data_df['success'], test_size=0.3, random_state=42)


# Define the logistic regression model using statsmodels
pca_model = smf.glm(formula='success ~ PC1 + PC2 + PC3',
                    data=pca_train, family=sm.families.Binomial())
raw_model = smf.glm(formula='success ~ danceability + energy + loudness + speechiness + acousticness + instrumentalness + valence + tempo',
                    data=raw_train, family=sm.families.Binomial())

# Fit the model to the data
pca_result = pca_model.fit()
raw_result = raw_model.fit()

# make predictions on test set
pca_y_pred = pca_result.predict(pca_test[[
    'PC1', 'PC2', 'PC3']])
raw_y_pred = raw_result.predict(raw_test[[
    "danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "valence", "tempo"]])

# convert predicted probabilities to binary predictions
pca_y_pred_binary = [1 if i > 0.5 else 0 for i in pca_y_pred]
raw_y_pred_binary = [1 if i > 0.5 else 0 for i in raw_y_pred]

# calculate accuracy
pca_accuracy = sum(pca_y_pred_binary ==
                   pca_result_test) / len(pca_result_test)
raw_accuracy = sum(raw_y_pred_binary ==
                   raw_result_test) / len(raw_result_test)


# Print the summary of the model
print(raw_result.summary())
print('Raw Data Accuracy:', raw_accuracy)
print(pca_result.summary())
print('PCA Data Accuracy:', pca_accuracy)
