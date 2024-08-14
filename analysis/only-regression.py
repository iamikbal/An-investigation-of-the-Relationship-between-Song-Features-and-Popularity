import pandas as pd
from sklearn.decomposition import PCA
import statsmodels.api as sm

# Load the data into a pandas DataFrame
data = pd.read_csv('reg1.csv')

# Create a new DataFrame with the transformed data and the success/failure labels
labels = [1 if x in [1, 2, 3] else 0 for x in data['rank']]
data_df = pd.DataFrame(data=data, columns=[
    "danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "valence", "tempo"
])
data_df['success'] = labels

# Define the logistic regression model using statsmodels
model = sm.Logit(data_df['success'], data_df[[
                 "danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "valence", "tempo"
                 ]])

# Fit the model to the data
result = model.fit()

# Print the summary of the model
print(result.summary())
